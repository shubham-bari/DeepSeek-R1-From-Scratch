from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from deepseek_llm.config import DeepSeekConfig
from deepseek_llm.modules.transformer import RMSNorm, TransformerBlock


class DeepSeekLM(nn.Module):
    def __init__(self, cfg: DeepSeekConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
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

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        moe_aux_coeff: float = 1e-2,
        past_kv: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, float], List[Optional[Dict[str, torch.Tensor]]]],
    ]:
        b, t = idx.shape
        if t > self.cfg.block_size:
            raise ValueError(f"Input sequence length {t} exceeds block size {self.cfg.block_size}")

        if past_kv is not None and len(past_kv) != self.cfg.n_layers:
            raise ValueError(f"past_kv should have {self.cfg.n_layers} entries, got {len(past_kv)}")
        past_len = 0
        if past_kv is not None and past_kv and past_kv[0]:
            first = past_kv[0]
            if first is not None:
                if "k" in first:
                    past_len = int(first["k"].size(-2))
                elif "latent" in first:
                    past_len = int(first["latent"].size(1))
        pos_ids = torch.arange(past_len, past_len + t, dtype=torch.long, device=idx.device)
        pos_shift = max(0, int(pos_ids[-1].item()) - self.cfg.block_size + 1)
        pos = (pos_ids - pos_shift).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        aux_loss_total = torch.zeros((), device=idx.device, dtype=x.dtype)
        next_kv: List[Optional[Dict[str, torch.Tensor]]] = []
        for i, block in enumerate(self.blocks):
            block_cache = past_kv[i] if past_kv is not None else None
            x, aux_loss, block_next_cache = block(x, kv_cache=block_cache, use_cache=use_cache)
            aux_loss_total = aux_loss_total + aux_loss
            if use_cache:
                next_kv.append(block_next_cache)

        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = lm_loss + moe_aux_coeff * aux_loss_total
        stats = {"moe_aux_loss": float(aux_loss_total.detach().item())}
        if use_cache:
            return logits, loss, stats, next_kv
        return logits, loss, stats

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        idx_cond = idx[:, -self.cfg.block_size :]
        logits, _, _, past_kv = self(idx_cond, use_cache=True)
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
            logits, _, _, past_kv = self(next_idx, use_cache=True, past_kv=past_kv)
        return idx
