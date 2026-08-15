import os
import re
import threading
import requests
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


def run_chat_completion(user_message, system_prompt=DEFAULT_SYSTEM_PROMPT, max_tokens=9000):
    llm = get_llm()
    prompt = CHATML_TEMPLATE.format(system=system_prompt, user=user_message)
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
      <li><span class="method">POST</span><span class="path">/api/title</span><span class="desc">chat titles</span></li>
      <li><span class="method">POST</span><span class="path">/api/summarize</span><span class="desc">doc summary</span></li>
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


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"error": "message is required."}), 400

    # Explicit "web_search" in the request overrides the heuristic; otherwise
    # decide automatically based on the message content.
    if "web_search" in data:
        use_web = bool(data.get("web_search"))
    else:
        use_web = should_use_web_search(message)

    try:
        context_block = ""
        sources = []
        if use_web:
            snippets = web_search(message)
            sources = [s["url"] for s in snippets]
            context_block = build_context_block(snippets)

        user_prompt = f"{context_block}\n\nQuestion: {message}" if context_block else message
        reply = run_chat_completion(user_prompt)

        return jsonify({"success": True, "reply": reply, "sources": sources})
    except Exception as e:
        app.logger.exception("chat() failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/title", methods=["POST"])
def generate_title():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()
    reply = (data.get("reply") or "").strip()

    if not user_message:
        return jsonify({"error": "message is required."}), 400

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
        title = title.strip().strip('"').strip("'").split("\n")[0][:42]
        return jsonify({"success": True, "title": title or "New chat"})
    except Exception as e:
        app.logger.exception("generate_title() failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/summarize", methods=["POST"])
def summarize():
    try:
        raw_text = ""

        if "file" in request.files and request.files["file"].filename != "":
            uploaded_file = request.files["file"]
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

        return jsonify({
            "success": True,
            "summary": summary_points,
            "stats": {
                "total_words": total_words,
                "reduction": max(0, reduction),
                "read_time": max(1, round(total_words / 200)),
            },
        })
    except Exception as e:
        app.logger.exception("summarize() failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)