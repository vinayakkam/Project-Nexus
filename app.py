import os
import re
import sys
import math
import logging
import threading
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# --- Environment & Logging ---
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] Nexus Core: %(message)s')

app = Flask(__name__)
CORS(app)

DATASET_PATH = os.path.join(os.getcwd(), "dataset.txt")
TOKENIZER_PATH = os.path.join(os.getcwd(), "tokenizer.json")
MODEL_PATH = os.path.join(os.getcwd(), "model.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = None
model = None
learning_lock = threading.Lock()

# Worker thread pool for parallel web scraping
executor = ThreadPoolExecutor(max_workers=4)


# ==============================================================================
# 1. HIGH-PERFORMANCE TRANSFORMER ARCHITECTURE
# ==============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        return x * torch.rsqrt(var + self.eps) * self.weight


def apply_rotary_emb(x, seq_len):
    B, T, n_head, head_dim = x.shape
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float().to(x.device) / head_dim))
    t = torch.arange(T, device=x.device, dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    rotated_x = torch.cat((-x2, x1), dim=-1)

    cos = emb.cos().view(1, T, 1, head_dim)
    sin = emb.sin().view(1, T, 1, head_dim)
    return (x * cos) + (rotated_x * sin)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_head: int):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.head_dim = dim // n_head

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim)

        q = apply_rotary_emb(q, T)
        k = apply_rotary_emb(k, T)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if T > 1:
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_head: int, hidden_dim: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_head)
        self.ffn_norm = RMSNorm(dim)
        self.feed_forward = SwiGLU(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class NexusLLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, n_head: int = 8, n_layer: int = 4, hidden_dim: int = 768):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, n_head, hidden_dim) for _ in range(n_layer)])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.embeddings.weight = self.head.weight

    def forward(self, idx):
        x = self.embeddings(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens=90, temperature=0.6, top_k=40, eos_token_id=None):
        """Optimized generation with Top-K logit pruning."""
        for _ in range(max_new_tokens):
            logits = self(idx)[:, -1, :] / max(temperature, 1e-5)

            # Fast Top-K Filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break
        return idx


# ==============================================================================
# 2. ASYNCHRONOUS PARALLEL WEB SCRAPER
# ==============================================================================

def fetch_single_url(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=3) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html, 'html.parser')
            for el in soup(["script", "style", "nav", "footer", "header", "aside"]):
                el.decompose()
            text = ' '.join(p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3']))
            return re.sub(r'\s+', ' ', text).strip()
    except Exception:
        return ""


def parallel_learn_topic(topic_query):
    logging.info(f"Async Scraper: Fetching knowledge for '{topic_query}'")
    encoded = urllib.parse.quote_plus(topic_query)
    search_url = f"https://html.duckduckgo.com/html/?q={encoded}"
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        req = urllib.request.Request(search_url, headers=headers)
        with urllib.request.urlopen(req, timeout=3) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html, 'html.parser')

            urls = []
            for a in soup.find_all('a', class_='result__url', limit=3):
                href = a.get('href', '')
                if href and href.startswith('http'):
                    urls.append(href)

            if not urls:
                return False

            # Parallelize URL scraping
            futures = [executor.submit(fetch_single_url, u) for u in urls]
            combined_text = ""
            for future in as_completed(futures):
                res = future.result()
                if len(res) > 100:
                    combined_text += " " + res[:1000]

            if len(combined_text) > 100:
                entry = f"\nUser: Tell me about {topic_query}\nBot: {combined_text.strip()}\n<|endoftext|>\n"
                with open(DATASET_PATH, "a", encoding="utf-8") as f:
                    f.write(entry)
                logging.info(f"Async Scraper: Successfully added data for '{topic_query}'.")
                return True
    except Exception as e:
        logging.error(f"Async Scraper Error: {e}")
    return False


# ==============================================================================
# 3. BACKGROUND LEARNING PIPELINE
# ==============================================================================

def train_model_fast(epochs=30):
    if not os.path.exists(DATASET_PATH):
        seed_data = [
            "User: Print Hi\nBot: ```python\nprint(\"Hi\")\n```\n<|endoftext|>\n",
            "User: Who are you?\nBot: I am OLIT Nexus, a custom neural language model.\n<|endoftext|>\n"
        ]
        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            for item in seed_data * 30:
                f.write(item)

    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.BPEDecoder()

    trainer = trainers.BpeTrainer(
        special_tokens=["<|pad|>", "<|unk|>", "<|startoftext|>", "<|endoftext|>"],
        vocab_size=3000
    )
    tok.train(files=[DATASET_PATH], trainer=trainer)
    tok.save(TOKENIZER_PATH)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        text_data = f.read()

    tokens = tok.encode(text_data).ids
    data_tensor = torch.tensor(tokens, dtype=torch.long)
    vocab_size = tok.get_vocab_size()

    train_net = NexusLLM(vocab_size=vocab_size, dim=256, n_head=8, n_layer=4).to(device)
    if os.path.exists(MODEL_PATH):
        try:
            train_net.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
        except Exception:
            pass

    optimizer = torch.optim.AdamW(train_net.parameters(), lr=1e-3, weight_decay=0.01)
    batch_size = 16
    seq_len = 128

    if len(data_tensor) <= seq_len:
        return

    train_net.train()
    for _ in range(epochs):
        ix = torch.randint(len(data_tensor) - seq_len, (batch_size,))
        x = torch.stack([data_tensor[i:i + seq_len] for i in ix]).to(device)
        y = torch.stack([data_tensor[i + 1:i + seq_len + 1] for i in ix]).to(device)

        logits = train_net(x)
        loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_net.parameters(), 1.0)
        optimizer.step()

    torch.save(train_net.state_dict(), MODEL_PATH)


def load_active_model():
    global tokenizer, model
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        train_model_fast(epochs=50)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    model = NexusLLM(vocab_size=vocab_size, dim=256, n_head=8, n_layer=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()


def async_learning_worker(topic_query):
    with learning_lock:
        if parallel_learn_topic(topic_query):
            train_model_fast(epochs=30)
            load_active_model()


# Boot active model into memory
load_active_model()


# ==============================================================================
# 4. API ENDPOINTS
# ==============================================================================

@app.route('/')
def home():
    return jsonify({'service': 'nexus-llm', 'status': 'ready', 'device': device})


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'error': 'Message required.'}), 400

    if not model or not tokenizer:
        return jsonify({'error': 'Model loading failed.'}), 500

    try:
        # Trigger non-blocking async learning thread
        if any(kw in message.lower() for kw in ['learn', 'search', 'vande bharat', 'latest', 'what is', 'who is']):
            threading.Thread(target=async_learning_worker, args=(message,), daemon=True).start()

        prompt = f"User: {message}\nBot:"
        encoded_ids = tokenizer.encode(prompt).ids
        input_tensor = torch.tensor([encoded_ids], dtype=torch.long).to(device)

        eos_id = tokenizer.token_to_id("<|endoftext|>")
        output_ids = model.generate(
            input_tensor,
            max_new_tokens=80,
            temperature=0.6,
            top_k=40,
            eos_token_id=eos_id
        )

        full_text = tokenizer.decode(output_ids[0].cpu().numpy().tolist())
        reply = full_text[len(prompt):].split("<|endoftext|>")[0].split("User:")[0].strip()

        return jsonify({
            'success': True,
            'reply': reply if reply else "I am OLIT Nexus. How can I assist you today?",
            'sources': []
        })

    except Exception as e:
        app.logger.exception("Inference error")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)