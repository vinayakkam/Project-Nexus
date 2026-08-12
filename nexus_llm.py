import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        return x * torch.rsqrt(var + self.eps) * self.weight


def apply_rotary_emb(x, seq_len):
    """Rotary Position Embedding (RoPE) implementation."""
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.6, top_p=0.85, eos_token_id=None):
        """Generates tokens using Top-p (Nucleus) sampling."""
        for _ in range(max_new_tokens):
            logits = self(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                break
        return idx