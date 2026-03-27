from typing import Optional
from torch import nn

from deepseek_llm.modules.attention_gqa import GroupedQueryAttention
from deepseek_llm.modules.attention_mha import MultiHeadAttention
from deepseek_llm.modules.attention_mla import MultiHeadLatentAttention
from deepseek_llm.modules.attention_mqa import MultiQueryAttention


def build_attention(
    attention_type: str,
    d_model: int,
    n_heads: int,
    n_kv_heads: Optional[int],
    kv_latent_dim: int,
    dropout: float,
    max_cache_len: int,
) -> nn.Module:
    attention_type = attention_type.lower()
    if attention_type == "mha":
        return MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, max_cache_len=max_cache_len)
    if attention_type == "mqa":
        return MultiQueryAttention(d_model=d_model, n_heads=n_heads, dropout=dropout, max_cache_len=max_cache_len)
    if attention_type == "gqa":
        return GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads or max(1, n_heads // 2),
            dropout=dropout,
            max_cache_len=max_cache_len,
        )
    if attention_type == "mla":
        return MultiHeadLatentAttention(
            d_model=d_model,
            n_heads=n_heads,
            kv_latent_dim=kv_latent_dim,
            dropout=dropout,
            max_cache_len=max_cache_len,
        )
    raise ValueError(f"Unsupported attention type: {attention_type}")
