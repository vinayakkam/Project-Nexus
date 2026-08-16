import os
import re
import threading
import secrets
import dns.resolver
from datetime import datetime, timedelta, timezone
from functools import wraps
import bcrypt
import jwt
import requests
from google.cloud import firestore
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
from pypdf import PdfReader
from PIL import Image
import numpy as np
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ==============================================================================
# AUTH / ACCOUNTS
#
# Storage: Firestore (Native mode) — chosen over SQLite because Cloud Run's
# filesystem is ephemeral and this service can run multiple instances
# (--max-instances=2), which SQLite can't safely share. Firestore requires
# no server of its own and has a generous free tier for this kind of scale.
#
# Setup required before this works (one-time, in the GCP Console or gcloud):
#   1. Enable the Firestore API and create a Firestore database (Native
#      mode) in your project, if you haven't already.
#   2. Set a JWT_SECRET environment variable on the Cloud Run service —
#      a long random string, the same value across all instances. e.g.:
#      gcloud run services update assistant-backend --region=$_DEPLOY_REGION \
#        --set-env-vars=JWT_SECRET=<paste a long random string here>
#      Do NOT commit a real secret into this file or your repo.
# ==============================================================================

JWT_SECRET = os.environ.get("JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_DAYS = 30

# ==============================================================================
# EMAIL VALIDATION — no sending. Checks that an email address is
# syntactically valid AND that its domain has real mail servers configured
# (an MX record, or an A record as a fallback some domains use instead).
# This confirms the domain is capable of receiving mail — it does NOT
# confirm the specific mailbox exists, which would require actually
# probing the mail server (unreliable, often blocked, and easily mistaken
# for spam/abuse behavior) or sending a verification email.
# ==============================================================================

EMAIL_SYNTAX_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


def is_valid_email(email):
    if not email or not EMAIL_SYNTAX_RE.match(email):
        return False
    domain = email.rsplit("@", 1)[-1]
    try:
        if dns.resolver.resolve(domain, "MX", lifetime=5):
            return True
    except Exception:
        pass
    try:
        # Some small domains skip MX and rely on the A record instead.
        return bool(dns.resolver.resolve(domain, "A", lifetime=5))
    except Exception:
        return False


_db = None
_db_lock = threading.Lock()


def get_db():
    """Lazily create the Firestore CLIENT (the network connection) —
    avoids adding startup latency or a hard dependency on Firestore being
    reachable before the app is ready to serve /api/health. The firestore
    module itself is imported at top level since that part is cheap."""
    global _db
    if _db is None:
        with _db_lock:
            if _db is None:
                # Explicit database ID required whenever the Firestore
                # database wasn't created with the default ID "(default)" —
                # firestore.Client() with no argument only ever connects to
                # "(default)" and would fail against a named database.
                _db = firestore.Client(database="olit-database")
    return _db


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
        if not JWT_SECRET:
            return jsonify({"error": "Server is missing JWT_SECRET configuration."}), 500

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
# CHAT / CODING MODEL — self-hosted, open-weight, no third-party API calls.
#
# Uses Qwen2.5-Coder-1.5B-Instruct, a real pretrained model published by the
# Qwen team (not something trained from zero here — that isn't achievable
# at usable quality in a project like this). Quantized to GGUF (~1GB) and
# run via llama.cpp so it's fast enough on Cloud Run's CPU-only instances.
#
# Loaded lazily on first request, NOT at import time — loading a ~1GB model
# at import blocks gunicorn from becoming ready and is what caused the
# "Service Unavailable" issue on the previous backend.
# ==============================================================================

MODEL_REPO = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
MODEL_CACHE_DIR = "/tmp/model_cache"  # Cloud Run only allows writes under /tmp

_llm = None
_llm_lock = threading.Lock()
# Separate lock from _llm_lock (which only guards one-time model loading).
# gunicorn runs with --threads 4, so multiple requests can reach here
# concurrently — llama.cpp is not safe for concurrent calls against a
# single model context, so every actual inference call serializes through
# this lock. On a 2-vCPU instance this costs little anyway, since there
# isn't real spare CPU for two generations to usefully overlap.
_inference_lock = threading.Lock()


def get_llm():
    """Lazily download (first call only) and load the GGUF model."""
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:  # re-check inside the lock
                from huggingface_hub import hf_hub_download
                from llama_cpp import Llama

                app.logger.info("Downloading/loading chat model (first request only)...")
                model_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=MODEL_FILE,
                    cache_dir=MODEL_CACHE_DIR,
                )
                _llm = Llama(
                    model_path=model_path,
                    n_ctx=4096,
                    # Cloud Run allocates a fixed CPU quota (--cpu=2 in
                    # cloudbuild.yaml), but os.cpu_count() can report the
                    # HOST machine's full core count instead of that quota.
                    # Over-spawning threads relative to real available CPU
                    # causes heavy contention and made replies far slower
                    # than they should be. LLAMA_THREADS lets this be tuned
                    # without a code change if the Cloud Run --cpu value
                    # changes later.
                    n_threads=int(os.environ.get("LLAMA_THREADS", 2)),
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
    "You are a concise, helpful coding and chat assistant running on a small "
    "self-hosted model. Give direct, correct answers. For code, use fenced "
    "code blocks. If you are not confident about a fact, say so rather than "
    "guessing."
)


def run_chat_completion(user_message, system_prompt=DEFAULT_SYSTEM_PROMPT, max_tokens=900):
    llm = get_llm()
    prompt = CHATML_TEMPLATE.format(system=system_prompt, user=user_message)
    with _inference_lock:
        result = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.4,
            stop=["<|im_end|>"],
        )
    return result["choices"][0]["text"].strip()


# ==============================================================================
# WEB SEARCH — used as retrieval context for a single answer, not as
# "training data". Nothing is written to disk or fed back into the model's
# weights — that was the fragile, Cloud-Run-incompatible part of the old
# setup. This just fetches a few pages and hands their text to the model as
# context for THIS request only.
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


# Auto-detect whether a message likely needs current/live information,
# instead of relying on the user to flip a manual toggle. An explicit
# "web_search" field in the request still overrides this if present.
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
# DOCUMENT SUMMARIZATION — unchanged from the working, Cloud-Run-friendly
# pipeline: PDF/image/text extraction + TextRank. No model weights needed.
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
# API ENDPOINTS
# ==============================================================================

STATUS_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OLIT Nexus — API</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #08090b;
    --panel: #111216;
    --border: #1e1f24;
    --text: #f2f3f5;
    --text-dim: #8b8e97;
    --text-faint: #55575f;
    --accent: #3b6cf6;
    --accent-cyan: #22d3ee;
    --online: #10b981;
    --display-font: 'Space Grotesk', ui-sans-serif, sans-serif;
    --body-font: 'Inter', -apple-system, sans-serif;
    --mono-font: 'JetBrains Mono', ui-monospace, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    background-image:
      radial-gradient(circle at 15% 0%, rgba(59,108,246,0.06) 0%, transparent 45%),
      linear-gradient(rgba(255,255,255,0.012) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.012) 1px, transparent 1px);
    background-size: auto, 34px 34px, 34px 34px;
    color: var(--text);
    font-family: var(--body-font);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1.5rem;
  }
  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-left: 2px solid var(--accent-cyan);
    border-radius: 12px;
    padding: 2rem 2.25rem;
    max-width: 420px;
    width: 100%;
  }
  h1 {
    font-family: var(--display-font);
    font-size: 1.3rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
  }
  .status-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--mono-font);
    font-size: 0.78rem;
    color: var(--text-dim);
    margin-bottom: 1.5rem;
  }
  .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--online);
    box-shadow: 0 0 0 0 rgba(16,185,129,0.5);
    animation: ping 2.2s cubic-bezier(0.4,0,0.6,1) infinite;
  }
  @keyframes ping {
    0% { box-shadow: 0 0 0 0 rgba(16,185,129,0.45); }
    70% { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
    100% { box-shadow: 0 0 0 0 rgba(16,185,129,0); }
  }
  .endpoints { list-style: none; }
  .endpoints li {
    display: flex;
    gap: 0.6rem;
    padding: 0.5rem 0;
    border-top: 1px solid var(--border);
    font-family: var(--mono-font);
    font-size: 0.76rem;
  }
  .method { color: var(--accent-cyan); flex-shrink: 0; width: 40px; }
  .path { color: var(--text); }
  .desc { color: var(--text-faint); margin-left: auto; text-align: right; }
</style>
</head>
<body>
  <div class="card">
    <h1>OLIT Nexus</h1>
    <div class="status-row"><span class="dot"></span> assistant-backend · online</div>
    <ul class="endpoints">
      <li><span class="method">GET</span><span class="path">/api/health</span><span class="desc">status check</span></li>
      <li><span class="method">POST</span><span class="path">/api/chat</span><span class="desc">chat + coding</span></li>
      <li><span class="method">POST</span><span class="path">/api/summarize</span><span class="desc">doc summary</span></li>
      <li><span class="method">GET</span><span class="path">/api/conversations</span><span class="desc">list history</span></li>
      <li><span class="method">GET</span><span class="path">/api/conversations/:id</span><span class="desc">get chat</span></li>
      <li><span class="method">PATCH</span><span class="path">/api/conversations/:id</span><span class="desc">rename chat</span></li>
      <li><span class="method">DELETE</span><span class="path">/api/conversations/:id</span><span class="desc">delete chat</span></li>
      <li><span class="method">POST</span><span class="path">/api/auth/signup</span><span class="desc">create account</span></li>
      <li><span class="method">POST</span><span class="path">/api/auth/login</span><span class="desc">sign in</span></li>
      <li><span class="method">GET</span><span class="path">/api/auth/me</span><span class="desc">current user</span></li>
      <li><span class="method">PUT</span><span class="path">/api/profile</span><span class="desc">update profile</span></li>
      <li><span class="method">POST</span><span class="path">/api/validate-email</span><span class="desc">check email</span></li>
    </ul>
  </div>
</body>
</html>"""


@app.route("/")
def home():
    return STATUS_PAGE_HTML


@app.route("/api/health")
def health():
    # Deliberately plain JSON, not the styled page above — the frontend's
    # heartbeat check and any external monitoring hit this expecting
    # machine-readable data, not HTML. Doesn't touch the model, so it
    # responds instantly even before the first chat request triggers a
    # model download.
    return jsonify({"status": "ok"})


def _public_user(doc_data):
    """Strip the password hash before sending a user record to the client."""
    return {
        "email": doc_data.get("email"),
        "name": doc_data.get("name"),
    }


@app.route("/api/auth/signup", methods=["POST"])
def signup():
    if not JWT_SECRET:
        return jsonify({"error": "Server is missing JWT_SECRET configuration."}), 500

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    name = (data.get("name") or "").strip()

    if not is_valid_email(email):
        return jsonify({"error": "Please enter a valid, deliverable email address."}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters."}), 400

    db = get_db()
    user_ref = db.collection("users").document(email)
    if user_ref.get().exists:
        return jsonify({"error": "An account with this email already exists."}), 409

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    display_name = name or email.split("@")[0]
    user_ref.set({
        "email": email,
        "name": display_name,
        "password_hash": password_hash,
    })

    token = generate_token(email)
    return jsonify({"success": True, "token": token, "user": {"email": email, "name": display_name}})


@app.route("/api/auth/login", methods=["POST"])
def login():
    if not JWT_SECRET:
        return jsonify({"error": "Server is missing JWT_SECRET configuration."}), 500

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    db = get_db()
    doc = db.collection("users").document(email).get()
    if not doc.exists:
        return jsonify({"error": "Invalid email or password."}), 401

    user = doc.to_dict()
    if not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        return jsonify({"error": "Invalid email or password."}), 401

    token = generate_token(email)
    return jsonify({"success": True, "token": token, "user": _public_user(user)})


@app.route("/api/auth/me", methods=["GET"])
@require_auth
def me():
    db = get_db()
    doc = db.collection("users").document(request.user_email).get()
    if not doc.exists:
        return jsonify({"error": "User not found."}), 404
    return jsonify({"success": True, "user": _public_user(doc.to_dict())})


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

    db = get_db()
    user_ref = db.collection("users").document(request.user_email)
    updates = {}
    if name:
        updates["name"] = name
    if new_password:
        updates["password_hash"] = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    user_ref.update(updates)
    doc = user_ref.get()
    return jsonify({"success": True, "user": _public_user(doc.to_dict())})


@app.route("/api/validate-email", methods=["POST"])
def validate_email_route():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    if not email:
        return jsonify({"error": "email is required."}), 400
    return jsonify({"success": True, "email": email, "valid": is_valid_email(email)})


# ==============================================================================
# CONVERSATIONS — server-side, per-user chat history (Firestore), replacing
# the old localStorage-only history. Each conversation document holds its
# own messages array directly (fine at personal/small-team scale; a very
# heavy user would eventually want a subcollection instead, since Firestore
# caps a single document at 1MB).
# ==============================================================================

def _get_owned_conversation(conversation_id, owner):
    """Fetch a conversation and verify the requester owns it. Returns
    (conv_ref, conv_data) or (None, None) if missing/not owned."""
    db = get_db()
    conv_ref = db.collection("conversations").document(conversation_id)
    doc = conv_ref.get()
    if not doc.exists or doc.to_dict().get("owner") != owner:
        return None, None
    return conv_ref, doc.to_dict()


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
    db = get_db()
    # NOTE: this equality-filter + order-by-different-field query may
    # prompt Firestore to ask for a composite index the first time it
    # runs — if so, the error Cloud Run logs includes a direct link that
    # creates the index in one click. That's expected, one-time setup,
    # not a bug.
    query = (
        db.collection("conversations")
        .where("owner", "==", request.user_email)
        .order_by("updated_at", direction=firestore.Query.DESCENDING)
        .limit(100)
    )
    conversations = [{"id": doc.id, "title": doc.to_dict().get("title", "New chat")} for doc in query.stream()]
    return jsonify({"success": True, "conversations": conversations})


@app.route("/api/conversations/<conversation_id>", methods=["GET"])
@require_auth
def get_conversation(conversation_id):
    _, conv_data = _get_owned_conversation(conversation_id, request.user_email)
    if conv_data is None:
        return jsonify({"error": "Conversation not found."}), 404
    return jsonify({
        "success": True,
        "id": conversation_id,
        "title": conv_data.get("title", "New chat"),
        "messages": conv_data.get("messages", []),
    })


@app.route("/api/conversations/<conversation_id>", methods=["PATCH"])
@require_auth
def rename_conversation(conversation_id):
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "").strip()[:60]
    if not title:
        return jsonify({"error": "title is required."}), 400

    conv_ref, conv_data = _get_owned_conversation(conversation_id, request.user_email)
    if conv_data is None:
        return jsonify({"error": "Conversation not found."}), 404

    conv_ref.update({"title": title})
    return jsonify({"success": True, "title": title})


@app.route("/api/conversations/<conversation_id>", methods=["DELETE"])
@require_auth
def delete_conversation(conversation_id):
    conv_ref, conv_data = _get_owned_conversation(conversation_id, request.user_email)
    if conv_data is None:
        return jsonify({"error": "Conversation not found."}), 404

    conv_ref.delete()
    return jsonify({"success": True})


@app.route("/api/chat", methods=["POST"])
@require_auth
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    conversation_id = data.get("conversation_id")

    if not message:
        return jsonify({"error": "message is required."}), 400

    if "web_search" in data:
        use_web = bool(data.get("web_search"))
    else:
        use_web = should_use_web_search(message)

    db = get_db()
    owner = request.user_email

    if conversation_id:
        conv_ref, conv_data = _get_owned_conversation(conversation_id, owner)
        if conv_data is None:
            return jsonify({"error": "Conversation not found."}), 404
    else:
        conv_ref = db.collection("conversations").document()
        conv_data = {"owner": owner, "title": "New chat", "messages": []}
        conv_ref.set({**conv_data, "created_at": firestore.SERVER_TIMESTAMP})
        conversation_id = conv_ref.id

    messages = conv_data.get("messages", [])
    is_first_exchange = len(messages) == 0

    try:
        context_block = ""
        sources = []
        if use_web:
            snippets = web_search(message)
            sources = [s["url"] for s in snippets]
            context_block = build_context_block(snippets)

        user_prompt = f"{context_block}\n\nQuestion: {message}" if context_block else message
        reply = run_chat_completion(user_prompt)

        messages.append({"role": "user", "content": message})
        messages.append({"role": "bot", "content": reply, "sources": sources})

        # Title generation is a second model call — running it here would
        # make the user wait through it before seeing their actual reply.
        # Ship the reply now with a placeholder title, then fill in the
        # real one in the background; the frontend refreshes the sidebar
        # title on its next conversation-list fetch.
        placeholder_title = conv_data.get("title", "New chat")
        conv_ref.update({
            "messages": messages,
            "title": placeholder_title,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })

        if is_first_exchange:
            def _fill_in_title():
                try:
                    real_title = _generate_title(message, reply)
                    conv_ref.update({"title": real_title})
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

        db = get_db()
        owner = request.user_email

        if conversation_id:
            conv_ref, conv_data = _get_owned_conversation(conversation_id, owner)
            if conv_data is None:
                return jsonify({"error": "Conversation not found."}), 404
        else:
            conv_ref = db.collection("conversations").document()
            conv_data = {"owner": owner, "title": "New chat", "messages": []}
            conv_ref.set({**conv_data, "created_at": firestore.SERVER_TIMESTAMP})
            conversation_id = conv_ref.id

        messages = conv_data.get("messages", [])
        is_first_exchange = len(messages) == 0

        user_label = f"Attached file: {filename_display}" if filename_display else "Summarize this text"
        messages.append({"role": "user", "content": user_label})
        messages.append({"role": "bot", "type": "summary", "summary": summary_points, "stats": stats})

        title = (filename_display or "Document summary")[:42] if is_first_exchange else conv_data.get("title", "New chat")

        conv_ref.update({
            "messages": messages,
            "title": title,
            "updated_at": firestore.SERVER_TIMESTAMP,
        })

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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)