import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class NotebookConfig:
    vocab_size: int
    block_size: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_kv_heads: int = 2
    n_layers: int = 4
    attention_type: str = "mla"  # mha | mqa | gqa | mla
    kv_latent_dim: int = 64
    moe_hidden_mult: int = 4
    moe_num_experts: int = 4
    dropout: float = 0.1
    mttp_steps: int = 0
    mttp_coeff: float = 0.25


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, Dh), Dh must be even.
    b, h, t, d = x.shape
    if d % 2 != 0:
        raise ValueError("RoPE requires even head dimension.")
    half = d // 2
    device = x.device
    dtype = x.dtype
    pos = torch.arange(t, device=device, dtype=dtype).unsqueeze(1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=dtype) / half))
    angles = pos * inv_freq.unsqueeze(0)  # (T, half)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)  # (1, 1, T, half)
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)

    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def _causal_mask(q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(q_len, kv_len, device=device, dtype=torch.bool), diagonal=1)


class MHAAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q)
        k = apply_rope(k)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(_causal_mask(t, t, x.device), float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.o_proj(out)


class MQAAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = apply_rope(self.k_proj(x).unsqueeze(1).expand(-1, self.n_heads, -1, -1))
        v = self.v_proj(x).unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        q = apply_rope(q)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(_causal_mask(t, t, x.device), float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.o_proj(out)


class GQAAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.group_size = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        k = apply_rope(k).repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        q = apply_rope(q)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(_causal_mask(t, t, x.device), float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.o_proj(out)


class MLAFromNotebook(nn.Module):
    """Notebook MLA idea with fixed implementation + RoPE."""

    def __init__(self, d_model: int, n_heads: int, kv_latent_dim: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_latent_dim = kv_latent_dim

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_dkv = nn.Linear(d_model, kv_latent_dim, bias=False)
        self.W_uk = nn.Linear(kv_latent_dim, d_model, bias=False)
        self.W_uv = nn.Linear(kv_latent_dim, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.ln = nn.LayerNorm(kv_latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.W_q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        latent = self.ln(self.W_dkv(x))
        k = self.W_uk(latent).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_uv(latent).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q)
        k = apply_rope(k)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = _causal_mask(t, t, x.device)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.W_o(out)


def build_attention_from_config(cfg: NotebookConfig) -> nn.Module:
    kind = cfg.attention_type.lower()
    if kind == "mha":
        return MHAAttention(d_model=cfg.d_model, n_heads=cfg.n_heads, dropout=cfg.dropout)
    if kind == "mqa":
        return MQAAttention(d_model=cfg.d_model, n_heads=cfg.n_heads, dropout=cfg.dropout)
    if kind == "gqa":
        return GQAAttention(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=max(1, cfg.n_kv_heads),
            dropout=cfg.dropout,
        )
    if kind == "mla":
        return MLAFromNotebook(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            kv_latent_dim=cfg.kv_latent_dim,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unsupported attention_type={cfg.attention_type}")


class SimpleExpert(nn.Module):
    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class SimpleMoEFromNotebook(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([SimpleExpert(d_model, d_hidden) for _ in range(num_experts)])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        top_expert = torch.argmax(gate_probs, dim=-1)
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = top_expert == i
            if mask.sum().item() == 0:
                continue
            selected = x[mask]
            out = self.experts[i](selected)
            output[mask] = out
        return output


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * (x * (x.pow(2).mean(-1, keepdim=True) + self.eps).rsqrt())


class NotebookTransformerBlock(nn.Module):
    def __init__(self, cfg: NotebookConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.attn = build_attention_from_config(cfg)
        self.moe = SimpleMoEFromNotebook(
            d_model=cfg.d_model,
            d_hidden=cfg.d_model * cfg.moe_hidden_mult,
            num_experts=cfg.moe_num_experts,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.moe(self.norm2(x))
        return x


class NotebookDeepSeekLM(nn.Module):
    def __init__(self, cfg: NotebookConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([NotebookTransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        b, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"Input sequence length {t} exceeds block size {self.cfg.block_size}")
        x = self.dropout(self.tok_emb(idx))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            mttp_loss_terms = []
            for offset in range(1, max(self.cfg.mttp_steps, 0) + 1):
                if t <= offset:
                    continue
                mttp_logits = logits[:, :-offset, :]
                mttp_targets = targets[:, offset:]
                mttp_term = F.cross_entropy(
                    mttp_logits.reshape(-1, mttp_logits.size(-1)),
                    mttp_targets.reshape(-1),
                )
                mttp_loss_terms.append(mttp_term)
            if mttp_loss_terms:
                mttp_loss = torch.stack(mttp_loss_terms).mean()
                loss = lm_loss + self.cfg.mttp_coeff * mttp_loss
            else:
                loss = lm_loss
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx


class NotebookAdam:
    """
    Param-based Adam optimizer following your notebook API style:
    pre_update_params -> update_params -> post_update_params
    """

    def __init__(self, lr: float = 3e-4, decay: float = 0.0, eps: float = 1e-8, beta1: float = 0.9, beta2: float = 0.999):
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay = decay
        self.lr = lr
        self.curr_lr = lr
        self.iterations = 0
        self.eps = eps
        self.state: Dict[int, Dict[str, torch.Tensor]] = {}

    def pre_update_params(self) -> None:
        if self.decay:
            self.curr_lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        else:
            self.curr_lr = self.lr

    def update_params(self, params: List[torch.nn.Parameter]) -> None:
        for p in params:
            if p.grad is None:
                continue
            pid = id(p)
            if pid not in self.state:
                self.state[pid] = {
                    "momentum": torch.zeros_like(p.data),
                    "cache": torch.zeros_like(p.data),
                }
            st = self.state[pid]
            grad = p.grad.data
            st["momentum"] = self.beta1 * st["momentum"] + (1.0 - self.beta1) * grad
            st["cache"] = self.beta2 * st["cache"] + (1.0 - self.beta2) * (grad * grad)

            t = self.iterations + 1
            m_hat = st["momentum"] / (1.0 - self.beta1**t)
            v_hat = st["cache"] / (1.0 - self.beta2**t)
            p.data = p.data - self.curr_lr * (m_hat / (torch.sqrt(v_hat) + self.eps))

    def post_update_params(self) -> None:
        self.iterations += 1

    def zero_grad(self, params: List[torch.nn.Parameter]) -> None:
        for p in params:
            if p.grad is not None:
                p.grad = None


DEFAULT_CORPUS = (
    "Notebook-based MLA+MoE+Adam experiment with RoPE. "
    "This default text is for sanity checks, not final model quality.\n"
) * 160


@dataclass
class TextDataset:
    train_data: torch.Tensor
    val_data: torch.Tensor
    stoi: Dict[str, int]
    itos: Dict[int, str]


def load_text(path: str) -> str:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return DEFAULT_CORPUS


def build_char_dataset(text: str, split_ratio: float = 0.9) -> TextDataset:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(split_ratio * len(encoded))
    return TextDataset(
        train_data=encoded[:split],
        val_data=encoded[split:],
        stoi=stoi,
        itos=itos,
    )


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError("Dataset is too small for selected block size.")
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in starts])
    return x.to(device), y.to(device)
