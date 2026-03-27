import argparse
import json
import time
from typing import Dict, List

import torch

from notebook_components import (
    NotebookAdam,
    NotebookConfig,
    NotebookDeepSeekLM,
    build_char_dataset,
    get_batch,
    load_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark notebook-based MLA+MoE+Adam model (RoPE positional encoding)."
    )
    parser.add_argument("--text-path", type=str, default="")
    parser.add_argument("--steps", type=int, default=40, help="Micro-training steps per variant.")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--block-size", type=int, default=96)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def quick_eval(
    model: NotebookDeepSeekLM,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(10):
        xb, yb = get_batch(data, batch_size, block_size, device)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def estimate_mla_kv_cache_bytes(cfg: NotebookConfig, batch_size: int, seq_len: int, dtype_bytes: int = 4) -> int:
    return 2 * batch_size * seq_len * cfg.kv_latent_dim * dtype_bytes * cfg.n_layers


def run_variant(
    variant_name: str,
    kv_latent_dim: int,
    moe_num_experts: int,
    dataset,
    steps: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> Dict[str, float]:
    cfg = NotebookConfig(
        vocab_size=len(dataset.stoi),
        block_size=block_size,
        n_layers=3,
        d_model=128,
        n_heads=4,
        kv_latent_dim=kv_latent_dim,
        moe_num_experts=moe_num_experts,
        moe_hidden_mult=4,
        dropout=0.1,
    )
    model = NotebookDeepSeekLM(cfg).to(device)
    optimizer = NotebookAdam(lr=3e-4, decay=0.0)
    params = list(model.parameters())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    start = time.perf_counter()
    for _ in range(steps):
        xb, yb = get_batch(dataset.train_data, batch_size, block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(params)
        loss.backward()
        optimizer.pre_update_params()
        optimizer.update_params(params)
        optimizer.post_update_params()
    elapsed = time.perf_counter() - start

    val_loss = quick_eval(model, dataset.val_data, batch_size, block_size, device)
    ppl = float(torch.exp(torch.tensor(val_loss)).item())
    tokens = steps * batch_size * block_size
    toks_per_sec = tokens / max(elapsed, 1e-8)

    if device.type == "cuda":
        peak_mem = float(torch.cuda.max_memory_allocated(device=device) / (1024 * 1024))
    else:
        peak_mem = float(sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024))

    kv_cache_mb = estimate_mla_kv_cache_bytes(cfg, batch_size=batch_size, seq_len=block_size) / (1024 * 1024)
    return {
        "variant": variant_name,
        "kv_latent_dim": kv_latent_dim,
        "moe_num_experts": moe_num_experts,
        "val_loss": val_loss,
        "perplexity_proxy": ppl,
        "tokens_per_sec": float(toks_per_sec),
        "peak_mem_mb": peak_mem,
        "estimated_kv_cache_mb": float(kv_cache_mb),
    }


def print_table(rows: List[Dict[str, float]]) -> None:
    headers = ["variant", "kv_lat", "experts", "val_loss", "ppl_proxy", "tok/s", "peak_mem_mb", "est_kv_mb"]
    print("\n" + " | ".join(headers))
    print("-" * 110)
    for r in rows:
        print(
            f"{r['variant']:>15} | {r['kv_latent_dim']:>6d} | {r['moe_num_experts']:>7d} | "
            f"{r['val_loss']:.4f} | {r['perplexity_proxy']:.2f} | "
            f"{r['tokens_per_sec']:.1f} | {r['peak_mem_mb']:.2f} | {r['estimated_kv_cache_mb']:.4f}"
        )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    text = load_text(args.text_path)
    dataset = build_char_dataset(text)

    variants = [
        ("mla_lat32_exp4", 32, 4),
        ("mla_lat64_exp4", 64, 4),
        ("mla_lat64_exp8", 64, 8),
        ("mla_lat96_exp8", 96, 8),
    ]
    rows = []
    for name, kv_latent_dim, moe_num_experts in variants:
        print(f"Running benchmark for variant={name} ...")
        row = run_variant(
            variant_name=name,
            kv_latent_dim=kv_latent_dim,
            moe_num_experts=moe_num_experts,
            dataset=dataset,
            steps=args.steps,
            batch_size=args.batch_size,
            block_size=args.block_size,
            device=device,
        )
        rows.append(row)

    print_table(rows)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=True, indent=2)
    print(f"\nSaved benchmark output to: {args.output}")


if __name__ == "__main__":
    main()
