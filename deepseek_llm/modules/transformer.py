from typing import Dict, Optional, Tuple

import torch
from torch import nn

from deepseek_llm.config import DeepSeekConfig
from deepseek_llm.modules.attention import build_attention
from deepseek_llm.modules.moe import Top1MoE


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * norm)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        d_hidden = d_model * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: DeepSeekConfig, layer_idx: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)
        self.attn = build_attention(
            attention_type=cfg.attention_type,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            kv_latent_dim=cfg.kv_latent_dim,
            dropout=cfg.dropout,
            max_cache_len=cfg.block_size,
        )
        self.use_moe = cfg.use_moe and (layer_idx % max(cfg.moe_every, 1) == 0)
        if self.use_moe:
            self.ff = Top1MoE(
                d_model=cfg.d_model,
                d_hidden=cfg.d_model * cfg.mlp_hidden_mult,
                num_experts=cfg.moe_num_experts,
                dropout=cfg.dropout,
            )
        else:
            self.ff = FeedForward(
                d_model=cfg.d_model,
                hidden_mult=cfg.mlp_hidden_mult,
                dropout=cfg.dropout,
            )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        attn_out, next_cache = self.attn(self.norm1(x), kv_cache=kv_cache, use_cache=use_cache)
        x = x + attn_out
        ff_in = self.norm2(x)
        if self.use_moe:
            ff_out, aux_loss = self.ff(ff_in)
        else:
            ff_out = self.ff(ff_in)
            aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        x = x + ff_out
        return x, aux_loss, next_cache
