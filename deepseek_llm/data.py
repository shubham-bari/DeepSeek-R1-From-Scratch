import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch


DEFAULT_CORPUS = (
    "DeepSeek style modular language model experiment. "
    "This tiny corpus is only for code validation and quick benchmarking. "
    "Use a much larger text file for meaningful training and evaluation.\n"
) * 128


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
    n = int(split_ratio * len(encoded))
    return TextDataset(train_data=encoded[:n], val_data=encoded[n:], stoi=stoi, itos=itos)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError("Dataset is too small for selected block_size.")
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
