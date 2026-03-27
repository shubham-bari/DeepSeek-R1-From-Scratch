import math
from typing import Dict, Optional, Tuple

import torch


KVCache = Dict[str, torch.Tensor]


def causal_mask_with_past(
    q_len: int,
    kv_len: int,
    past_len: int,
    device: torch.device,
) -> torch.Tensor:
    q_positions = torch.arange(past_len, past_len + q_len, device=device).unsqueeze(-1)
    kv_positions = torch.arange(kv_len, device=device).unsqueeze(0)
    return kv_positions > q_positions


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout: torch.nn.Dropout,
    past_len: int,
) -> torch.Tensor:
    q_len = q.size(-2)
    kv_len = k.size(-2)
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
    mask = causal_mask_with_past(q_len=q_len, kv_len=kv_len, past_len=past_len, device=q.device)
    scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = dropout(attn)
    return torch.matmul(attn, v)


def concat_and_trim_cache(
    prev: Optional[torch.Tensor],
    curr: torch.Tensor,
    max_cache_len: int,
) -> torch.Tensor:
    if prev is not None:
        out = torch.cat([prev, curr], dim=-2)
    else:
        out = curr
    if out.size(-2) > max_cache_len:
        out = out[..., -max_cache_len:, :]
    return out


def kv_cache_num_bytes(cache: Optional[KVCache]) -> int:
    if cache is None:
        return 0
    total = 0
    for tensor in cache.values():
        total += int(tensor.numel() * tensor.element_size())
    return total
