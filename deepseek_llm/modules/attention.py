import math
from typing import Optional

import torch
from torch import nn


def _causal_mask(q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    # True means "masked" to match masked_fill behavior.
    return torch.triu(torch.ones(q_len, kv_len, dtype=torch.bool, device=device), diagonal=1)


class MultiHeadAttention(nn.Module):
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

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = _causal_mask(t, t, x.device)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.o_proj(out)


class MultiQueryAttention(nn.Module):
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
        k = self.k_proj(x).unsqueeze(1)  # (B, 1, T, Hd)
        v = self.v_proj(x).unsqueeze(1)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = _causal_mask(t, t, x.device)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        v_expand = v.expand(-1, self.n_heads, -1, -1)
        out = torch.matmul(attn, v_expand).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.o_proj(out)


class GroupedQueryAttention(nn.Module):
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

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = _causal_mask(t, t, x.device)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.o_proj(out)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kv_latent_dim: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_latent_dim = kv_latent_dim
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_down = nn.Linear(d_model, kv_latent_dim, bias=False)
        self.k_up = nn.Linear(kv_latent_dim, d_model, bias=False)
        self.v_up = nn.Linear(kv_latent_dim, d_model, bias=False)
        self.kv_norm = nn.LayerNorm(kv_latent_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        latent = self.kv_norm(self.kv_down(x))
        k = self.k_up(latent).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(latent).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = _causal_mask(t, t, x.device)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.o_proj(out)


def build_attention(
    attention_type: str,
    d_model: int,
    n_heads: int,
    n_kv_heads: Optional[int],
    kv_latent_dim: int,
    dropout: float,
) -> nn.Module:
    attention_type = attention_type.lower()
    if attention_type == "mha":
        return MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
    if attention_type == "mqa":
        return MultiQueryAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
    if attention_type == "gqa":
        return GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads or max(1, n_heads // 2),
            dropout=dropout,
        )
    if attention_type == "mla":
        return MultiHeadLatentAttention(
            d_model=d_model,
            n_heads=n_heads,
            kv_latent_dim=kv_latent_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported attention type: {attention_type}")
