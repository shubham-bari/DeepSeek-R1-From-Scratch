from typing import Optional, Tuple

import torch
from torch import nn

from deepseek_llm.modules.attention_common import KVCache, concat_and_trim_cache, scaled_dot_product_attention


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kv_latent_dim: int, dropout: float, max_cache_len: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.kv_latent_dim = kv_latent_dim
        self.max_cache_len = max_cache_len
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_down = nn.Linear(d_model, kv_latent_dim, bias=False)
        self.k_up = nn.Linear(kv_latent_dim, d_model, bias=False)
        self.v_up = nn.Linear(kv_latent_dim, d_model, bias=False)
        self.kv_norm = nn.LayerNorm(kv_latent_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        latent_new = self.kv_norm(self.kv_down(x))

        latent_prev = None
        past_len = 0
        if kv_cache is not None:
            latent_prev = kv_cache.get("latent")
            if latent_prev is not None:
                past_len = int(latent_prev.size(1))

        latent = concat_and_trim_cache(latent_prev, latent_new, self.max_cache_len)
        past_len = min(past_len, latent.size(1) - t)

        k = self.k_up(latent).view(b, latent.size(1), self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_up(latent).view(b, latent.size(1), self.n_heads, self.head_dim).transpose(1, 2)
        out = scaled_dot_product_attention(q=q, k=k, v=v, dropout=self.dropout, past_len=past_len)
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        next_cache = {"latent": latent} if use_cache else None
        return self.o_proj(out), next_cache
