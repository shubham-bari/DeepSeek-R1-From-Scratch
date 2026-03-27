from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Top1MoE(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, num_experts: int, dropout: float) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [ExpertFFN(d_model=d_model, d_hidden=d_hidden, dropout=dropout) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        top_expert = torch.argmax(probs, dim=-1)  # (B, T)
        out = torch.zeros_like(x)

        counts = torch.zeros(self.num_experts, device=x.device, dtype=torch.float32)
        for i, expert in enumerate(self.experts):
            mask = top_expert == i
            token_count = mask.sum()
            if token_count.item() == 0:
                continue
            counts[i] = token_count.float()
            out[mask] = expert(x[mask])

        # Simple load-balancing penalty to reduce routing collapse.
        frac = counts / counts.sum().clamp(min=1.0)
        target = torch.full_like(frac, 1.0 / self.num_experts)
        aux_loss = F.mse_loss(frac, target)
        return out, aux_loss
