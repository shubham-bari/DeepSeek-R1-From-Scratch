from typing import Optional, Tuple

import torch
from torch import nn

from deepseek_llm.modules.attention_common import KVCache, concat_and_trim_cache, scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, max_cache_len: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_cache_len = max_cache_len
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
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
        k_new = self.k_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        k_prev = None
        v_prev = None
        if kv_cache is not None:
            k_prev = kv_cache.get("k")
            v_prev = kv_cache.get("v")
            if k_prev is not None:
                past_len = int(k_prev.size(-2))

        k = concat_and_trim_cache(k_prev, k_new, self.max_cache_len)
        v = concat_and_trim_cache(v_prev, v_new, self.max_cache_len)
        past_len = min(past_len, k.size(-2) - t)

        out = scaled_dot_product_attention(q=q, k=k, v=v, dropout=self.dropout, past_len=past_len)
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        next_cache = {"k": k, "v": v} if use_cache else None
        return self.o_proj(out), next_cache
