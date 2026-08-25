import os
import re
import json
import uuid
import threading
import secrets
import sqlite3
import dns.resolver
from datetime import datetime, timedelta, timezone
from functools import wraps
import bcrypt
import jwt
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pypdf import PdfReader
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

app = Flask(__name__)
CORS(app)  # harmless locally (same-origin doesn't need it), kept for flexibility


@app.errorhandler(Exception)
def handle_unexpected_error(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    app.logger.exception("Unhandled exception")
    return jsonify({"error": "An unexpected server error occurred. Check the terminal for details."}), 500


# ==============================================================================
# AUTH / ACCOUNTS — local SQLite instead of Firestore. No cloud project, no
# billing, no setup: the database file and JWT secret are created
# automatically on first run, right next to this script.
# ==============================================================================

DB_PATH = os.environ.get("LOCAL_DB_PATH", os.path.join(DATA_DIR, "olit_nexus.db"))
_db_write_lock = threading.Lock()


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            title TEXT DEFAULT 'New chat',
            messages TEXT DEFAULT '[]',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()  # cheap, local, no network — safe to run at import time unlike Firestore


def _get_or_create_local_secret():
    """A cloud deployment needs JWT_SECRET set manually so every instance
    shares the same value. A local single-machine app doesn't have that
    problem, so generate one automatically on first run and reuse it from
    a local file — no manual setup step needed."""
    secret_path = os.path.join(DATA_DIR, ".jwt_secret")
    if os.path.exists(secret_path):
        with open(secret_path, "r") as f:
            existing = f.read().strip()
            if existing:
                return existing
    new_secret = secrets.token_hex(32)
    with open(secret_path, "w") as f:
        f.write(new_secret)
    return new_secret


JWT_SECRET = os.environ.get("JWT_SECRET") or _get_or_create_local_secret()
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 30


# ==============================================================================
# EMAIL VALIDATION — same MX-record check as the cloud version. Requires
# internet access to actually resolve DNS; if you're fully offline, this
# will treat every address as unverifiable and reject signups. Loosen
# EMAIL_SYNTAX_RE-only validation yourself if you want offline signup.
# ==============================================================================

EMAIL_SYNTAX_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


def is_valid_email(email):
    if not email or not EMAIL_SYNTAX_RE.match(email):
        return False
    domain = email.rsplit("@", 1)[-1]
    try:
        if dns.resolver.resolve(domain, "MX", lifetime=3):
            return True
    except dns.resolver.NXDOMAIN:
        return False
    except Exception as e:
        app.logger.warning("MX lookup failed for %s: %s", domain, e)
    try:
        return bool(dns.resolver.resolve(domain, "A", lifetime=3))
    except Exception as e:
        app.logger.warning("A record lookup failed for %s: %s", domain, e)
        return False


def generate_token(email):
    payload = {
        "sub": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRY_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Authorization header required."}), 401
        token = auth_header[7:]
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Session expired, please log in again."}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid session."}), 401
        request.user_email = payload["sub"]
        return f(*args, **kwargs)
    return wrapper


# ==============================================================================
# CHAT / CODING MODEL — identical to the cloud version. Runs on YOUR CPU now,
# not a 2-vCPU Cloud Run instance, so speed depends entirely on your
# machine's hardware — could be faster or slower than the cloud version.
# ==============================================================================

MODEL_REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
MODEL_CACHE_DIR = os.path.join(DATA_DIR, "model_cache")

_llm = None
_llm_lock = threading.Lock()
_inference_lock = threading.Lock()


def get_llm():
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                from huggingface_hub import hf_hub_download
                from llama_cpp import Llama

                app.logger.info("Downloading/loading chat model (first run only, ~1GB)...")
                model_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=MODEL_FILE,
                    cache_dir=MODEL_CACHE_DIR,
                )
                _llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,
                    # Uses your machine's actual core count by default now —
                    # there's no fixed CPU quota to work around like on
                    # Cloud Run. Override with LLAMA_THREADS if needed.
                    n_threads=int(os.environ.get("LLAMA_THREADS", os.cpu_count() or 4)),
                    n_batch=256,
                    verbose=False,
                )
                app.logger.info("Chat model ready.")
    return _llm


CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

DEFAULT_SYSTEM_PROMPT = (
    "You are OLIT Nexus, an AI assistant developed by OLIT Technologies. "
    "If asked what you are, who made you, or similar questions about your "
    "identity, answer that you are OLIT Nexus, built by OLIT Technologies — "
    "do not mention any other model name or organization. "
    "You are a concise, helpful coding and chat assistant. Give direct, "
    "correct answers. For code, use fenced code blocks. If you are not "
    "confident about a fact, say so rather than guessing."
)


def run_chat_completion(user_message, system_prompt=DEFAULT_SYSTEM_PROMPT, max_tokens=900, temperature=0.4):
    llm = get_llm()
    prompt = CHATML_TEMPLATE.format(system=system_prompt, user=user_message)
    with _inference_lock:
        result = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=["<|im_end|>"])
    return _scrub_identity_leaks(result["choices"][0]["text"].strip())


# ==============================================================================
# IDENTITY — a system prompt alone is a suggestion, not a guarantee,
# especially for a small model; it can still slip and name the real
# underlying model when asked directly. Two backstops:
#   1. Detect an identity question and answer it directly, skipping the
#      model entirely — 100% reliable for the common phrasings.
#   2. Scrub the real model/org name out of EVERY reply as a safety net,
#      catching identity questions phrased in ways #1 doesn't match.
# ==============================================================================

IDENTITY_QUESTION_RE = re.compile(
    r"\b("
    r"what are you|who are you|what('?s| is) your name|"
    r"who (made|created|developed|built|trained) you|"
    r"(which|what) (model|llm|ai) are you|"
    r"are you (chatgpt|gpt-?\d|claude|gemini|qwen|llama|an? ai)"
    r")\b",
    re.IGNORECASE,
)

IDENTITY_RESPONSE = "I'm OLIT Nexus, an AI assistant developed by OLIT Technologies."


def is_identity_question(message: str) -> bool:
    return bool(IDENTITY_QUESTION_RE.search(message))


def _scrub_identity_leaks(reply: str) -> str:
    reply = re.sub(r"\bQwen(\s?2(\.5)?)?(-?Coder)?\b", "OLIT Nexus", reply, flags=re.IGNORECASE)
    reply = re.sub(r"\bAlibaba(\s?Cloud)?\b", "OLIT Technologies", reply, flags=re.IGNORECASE)
    return reply


# ==============================================================================
# WEB SEARCH — unchanged. Needs internet; skips gracefully if unreachable.
# ==============================================================================

SEARCH_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AssistantBot/1.0)"}


def web_search(query, max_results=3, per_page_chars=1200):
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=SEARCH_HEADERS,
            timeout=6,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        urls = []
        for a in soup.select("a.result__a")[:max_results]:
            href = a.get("href", "")
            if href.startswith("http"):
                urls.append(href)

        snippets = []
        for url in urls:
            try:
                page = requests.get(url, headers=SEARCH_HEADERS, timeout=6)
                page_soup = BeautifulSoup(page.text, "html.parser")
                for tag in page_soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                text = " ".join(p.get_text() for p in page_soup.find_all(["p", "h1", "h2", "h3"]))
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    snippets.append({"url": url, "text": text[:per_page_chars]})
            except requests.RequestException:
                continue
        return snippets
    except requests.RequestException as e:
        app.logger.warning(f"web_search failed: {e}")
        return []


WEB_SEARCH_TRIGGER_PATTERNS = re.compile(
    r"\b("
    r"latest|current(ly)?|today|right now|now\b|recent(ly)?|"
    r"this (week|month|year)|news|price|cost of|stock|weather|"
    r"score|result|release(d)?|version|update(d)?|"
    r"who is|who won|when is|when did|when was|what year"
    r")\b",
    re.IGNORECASE,
)


def should_use_web_search(message: str) -> bool:
    return bool(WEB_SEARCH_TRIGGER_PATTERNS.search(message))


def build_context_block(snippets):
    if not snippets:
        return ""
    parts = ["Here is recent information retrieved from the web. Use it if relevant, "
             "and mention when you're relying on it:\n"]
    for i, s in enumerate(snippets, 1):
        parts.append(f"[Source {i}: {s['url']}]\n{s['text']}\n")
    return "\n".join(parts)


# ==============================================================================
# DOCUMENT SUMMARIZATION — unchanged.
# ==============================================================================

_ocr_engine = None
_ocr_lock = threading.Lock()


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        with _ocr_lock:
            if _ocr_engine is None:
                from rapidocr_onnxruntime import RapidOCR
                _ocr_engine = RapidOCR()
    return _ocr_engine


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def extract_text_from_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img_np = np.array(img)
    engine = get_ocr_engine()
    result, _ = engine(img_np)
    if not result:
        return ""
    return "\n".join(line[1] for line in result)


def textrank_summarize(text, sentence_count=4):
    sentences = re.split(r"(?<=[.!?]) +", text.strip())
    clean_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(clean_sentences) <= sentence_count:
        return clean_sentences

    sentence_tokens = [
        set(re.findall(r"\b[a-zA-Z0-9]{2,}\b", s.lower())) for s in clean_sentences
    ]
    n = len(clean_sentences)
    similarity_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j or not sentence_tokens[i] or not sentence_tokens[j]:
                continue
            intersection = len(sentence_tokens[i].intersection(sentence_tokens[j]))
            union = len(sentence_tokens[i].union(sentence_tokens[j]))
            similarity_matrix[i][j] = intersection / float(union) if union > 0 else 0.0

    d = 0.85
    scores = [1.0] * n
    for _ in range(20):
        new_scores = [1.0 - d] * n
        for i in range(n):
            for j in range(n):
                if i != j and similarity_matrix[j][i] > 0:
                    weight_sum = sum(similarity_matrix[j])
                    if weight_sum > 0:
                        new_scores[i] += d * (scores[j] * (similarity_matrix[j][i] / weight_sum))
        scores = new_scores

    for index in range(n):
        scores[index] *= (1.0 + (1.0 / (index + 1)) * 0.2)

    ranked_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)[:sentence_count]
    sorted_indices = sorted(ranked_indices)
    return [clean_sentences[i] for i in sorted_indices]


# ==============================================================================
# FRONTEND — served directly by Flask from ./frontend, same origin as the
# API. No GitHub Pages, no separate domain, no CORS headaches.
# ==============================================================================

@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/login.html")
def serve_login():
    return send_from_directory(FRONTEND_DIR, "login.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


# ==============================================================================
# AUTH / PROFILE ROUTES
# ==============================================================================

def _public_user(row):
    return {"email": row["email"], "name": row["name"]}


@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    name = (data.get("name") or "").strip()

    if not is_valid_email(email):
        return jsonify({"error": "Please enter a valid, deliverable email address."}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400

    display_name = name or email.split("@")[0]
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    with _db_write_lock:
        conn = get_conn()
        try:
            if conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
                return jsonify({"error": "An account with this email already exists."}), 409
            conn.execute(
                "INSERT INTO users (email, name, password_hash) VALUES (?, ?, ?)",
                (email, display_name, password_hash),
            )
            conn.commit()
        finally:
            conn.close()

    token = generate_token(email)
    return jsonify({"success": True, "token": token, "user": {"email": email, "name": display_name}})


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    if not row or not bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8")):
        return jsonify({"error": "Invalid email or password."}), 401

    token = generate_token(email)
    return jsonify({"success": True, "token": token, "user": _public_user(row)})


@app.route("/api/auth/me", methods=["GET"])
@require_auth
def me():
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (request.user_email,)).fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "User not found."}), 404
    return jsonify({"success": True, "user": _public_user(row)})


@app.route("/api/profile", methods=["PUT"])
@require_auth
def update_profile():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    new_password = data.get("new_password") or ""

    if not name and not new_password:
        return jsonify({"error": "Nothing to update."}), 400
    if new_password and len(new_password) < 8:
        return jsonify({"error": "New password must be at least 8 characters."}), 400

    with _db_write_lock:
        conn = get_conn()
        if name:
            conn.execute("UPDATE users SET name = ? WHERE email = ?", (name, request.user_email))
        if new_password:
            new_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
            conn.execute("UPDATE users SET password_hash = ? WHERE email = ?", (new_hash, request.user_email))
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE email = ?", (request.user_email,)).fetchone()
        conn.close()

    return jsonify({"success": True, "user": _public_user(row)})


@app.route("/api/validate-email", methods=["POST"])
def validate_email_route():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "email is required."}), 400
    return jsonify({"success": True, "email": email, "valid": is_valid_email(email)})


# ==============================================================================
# CONVERSATIONS — same shape as the cloud version's Firestore documents,
# just stored as a SQLite row with messages as a JSON-encoded column.
# ==============================================================================

def _load_conversation(conversation_id, owner):
    conn = get_conn()
    row = conn.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
    conn.close()
    if not row or row["owner"] != owner:
        return None
    return {"id": row["id"], "title": row["title"], "messages": json.loads(row["messages"])}


def _create_conversation(owner):
    conv_id = str(uuid.uuid4())
    with _db_write_lock:
        conn = get_conn()
        conn.execute(
            "INSERT INTO conversations (id, owner, title, messages) VALUES (?, ?, ?, ?)",
            (conv_id, owner, "New chat", "[]"),
        )
        conn.commit()
        conn.close()
    return {"id": conv_id, "title": "New chat", "messages": []}


def _save_conversation(conversation_id, messages, title):
    with _db_write_lock:
        conn = get_conn()
        conn.execute(
            "UPDATE conversations SET messages = ?, title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (json.dumps(messages), title, conversation_id),
        )
        conn.commit()
        conn.close()


def _update_title_only(conversation_id, title):
    with _db_write_lock:
        conn = get_conn()
        conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))
        conn.commit()
        conn.close()


def _generate_title(user_message, reply):
    try:
        prompt = (
            f"User: {user_message}\nAssistant: {reply[:400]}\n\n"
            "Write a short title for this conversation: 3-6 words, plain text, "
            "no quotes, no trailing punctuation."
        )
        title = run_chat_completion(
            prompt,
            system_prompt=(
                "You write extremely short, plain chat titles that summarize what a "
                "conversation is about. Output ONLY the title text, nothing else."
            ),
            max_tokens=16,
        )
        return title.strip().strip('"').strip("'").split("\n")[0][:42] or "New chat"
    except Exception:
        app.logger.exception("_generate_title() failed")
        return "New chat"


@app.route("/api/conversations", methods=["GET"])
@require_auth
def list_conversations():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, title FROM conversations WHERE owner = ? ORDER BY updated_at DESC LIMIT 100",
        (request.user_email,),
    ).fetchall()
    conn.close()
    return jsonify({"success": True, "conversations": [{"id": r["id"], "title": r["title"]} for r in rows]})


@app.route("/api/conversations/<conversation_id>", methods=["GET"])
@require_auth
def get_conversation(conversation_id):
    conv = _load_conversation(conversation_id, request.user_email)
    if conv is None:
        return jsonify({"error": "Conversation not found."}), 404
    return jsonify({"success": True, **conv})


@app.route("/api/conversations/<conversation_id>", methods=["PATCH"])
@require_auth
def rename_conversation(conversation_id):
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()[:60]
    if not title:
        return jsonify({"error": "title is required."}), 400

    conv = _load_conversation(conversation_id, request.user_email)
    if conv is None:
        return jsonify({"error": "Conversation not found."}), 404

    _update_title_only(conversation_id, title)
    return jsonify({"success": True, "title": title})


@app.route("/api/conversations/<conversation_id>", methods=["DELETE"])
@require_auth
def delete_conversation(conversation_id):
    conv = _load_conversation(conversation_id, request.user_email)
    if conv is None:
        return jsonify({"error": "Conversation not found."}), 404

    with _db_write_lock:
        conn = get_conn()
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()
        conn.close()
    return jsonify({"success": True})


@app.route("/api/chat", methods=["POST"])
@require_auth
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    conversation_id = data.get("conversation_id")
    edit_index = data.get("edit_index")

    if not message:
        return jsonify({"error": "message is required."}), 400

    use_web = bool(data.get("web_search")) if "web_search" in data else should_use_web_search(message)

    # Model settings overrides from the client (Model Settings panel),
    # clamped to sane ranges so a bad value can't hang the model or blow
    # up memory — these are still local/personal-use bounds, not
    # security-hardened against a hostile client.
    temperature = data.get("temperature")
    try:
        temperature = max(0.0, min(1.5, float(temperature))) if temperature is not None else 0.4
    except (TypeError, ValueError):
        temperature = 0.4

    max_tokens = data.get("max_tokens")
    try:
        max_tokens = max(50, min(2000, int(max_tokens))) if max_tokens is not None else 900
    except (TypeError, ValueError):
        max_tokens = 900

    system_prompt = (data.get("system_prompt") or "").strip()[:2000] or DEFAULT_SYSTEM_PROMPT

    owner = request.user_email
    if conversation_id:
        conv = _load_conversation(conversation_id, owner)
        if conv is None:
            return jsonify({"error": "Conversation not found."}), 404
    else:
        conv = _create_conversation(owner)
        conversation_id = conv["id"]

    messages = conv["messages"]

    # Editing a previous message: drop everything from that point onward
    # (the old message and everything that followed it) and regenerate
    # from there, same as ChatGPT's "edit and resubmit" behavior.
    if isinstance(edit_index, int) and 0 <= edit_index < len(messages):
        messages = messages[:edit_index]

    is_first_exchange = len(messages) == 0

    try:
        context_block = ""
        sources = []

        if is_identity_question(message) and not use_web:
            # Deterministic — no model call, no chance of it saying the
            # wrong thing. Web-search questions still go through the model
            # even if they happen to match, since they need an actual answer.
            reply = IDENTITY_RESPONSE
        else:
            if use_web:
                snippets = web_search(message)
                sources = [s["url"] for s in snippets]
                context_block = build_context_block(snippets)

            user_prompt = f"{context_block}\n\nQuestion: {message}" if context_block else message
            reply = run_chat_completion(user_prompt, system_prompt=system_prompt, max_tokens=max_tokens, temperature=temperature)

        messages.append({"role": "user", "content": message})
        messages.append({"role": "bot", "content": reply, "sources": sources})

        placeholder_title = conv["title"]
        _save_conversation(conversation_id, messages, placeholder_title)

        if is_first_exchange:
            def _fill_in_title():
                try:
                    real_title = _generate_title(message, reply)
                    _update_title_only(conversation_id, real_title)
                except Exception:
                    app.logger.exception("background title generation failed")
            threading.Thread(target=_fill_in_title, daemon=True).start()

        return jsonify({
            "success": True,
            "reply": reply,
            "sources": sources,
            "conversation_id": conversation_id,
            "title": placeholder_title,
        })
    except Exception as e:
        app.logger.exception("chat() failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/summarize", methods=["POST"])
@require_auth
def summarize():
    try:
        conversation_id = request.form.get("conversation_id") or None
        raw_text = ""
        filename_display = None

        if "file" in request.files and request.files["file"].filename != "":
            uploaded_file = request.files["file"]
            filename_display = uploaded_file.filename
            filename = uploaded_file.filename.lower()

            if filename.endswith(".pdf"):
                raw_text = extract_text_from_pdf(uploaded_file)
            elif filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                raw_text = extract_text_from_image(uploaded_file)
            elif filename.endswith(".txt"):
                raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
            else:
                return jsonify({"error": "Unsupported file type."}), 400
        else:
            raw_text = request.form.get("text", "")

        if not raw_text.strip():
            return jsonify({"error": "No readable text could be extracted."}), 400

        total_words = len(raw_text.split())
        summary_points = textrank_summarize(raw_text, sentence_count=4)
        summary_words = sum(len(pt.split()) for pt in summary_points)
        reduction = round((1 - summary_words / total_words) * 100) if total_words > 0 else 0
        stats = {
            "total_words": total_words,
            "reduction": max(0, reduction),
            "read_time": max(1, round(total_words / 200)),
        }

        owner = request.user_email
        if conversation_id:
            conv = _load_conversation(conversation_id, owner)
            if conv is None:
                return jsonify({"error": "Conversation not found."}), 404
        else:
            conv = _create_conversation(owner)
            conversation_id = conv["id"]

        messages = conv["messages"]
        is_first_exchange = len(messages) == 0

        user_label = f"Attached file: {filename_display}" if filename_display else "Summarize this text"
        messages.append({"role": "user", "content": user_label})
        messages.append({"role": "bot", "type": "summary", "summary": summary_points, "stats": stats})

        title = (filename_display or "Document summary")[:42] if is_first_exchange else conv["title"]
        _save_conversation(conversation_id, messages, title)

        return jsonify({
            "success": True,
            "summary": summary_points,
            "stats": stats,
            "conversation_id": conversation_id,
            "title": title,
        })
    except Exception as e:
        app.logger.exception("summarize() failed")
        return jsonify({"error": str(e)}), 500


# Catch-all for frontend assets (logo1.png, etc.) placed alongside
# index.html/login.html — registered last so it never shadows an /api/*
# route (Flask/Werkzeug match literal path segments before this kind of
# wildcard regardless of registration order, but keeping it last is
# clearer to read).
@app.route("/<path:filename>")
def serve_frontend_asset(filename):
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=False)
