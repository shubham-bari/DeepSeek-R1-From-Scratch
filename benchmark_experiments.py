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
        description="Benchmark notebook-based attention/MTTP variants."
    )
    parser.add_argument("--text-path", type=str, default="")
    parser.add_argument("--steps", type=int, default=40, help="Micro-training steps per variant.")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--block-size", type=int, default=96)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default="benchmark_results.json")
    parser.add_argument("--seed", type=int, default=42)
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


@torch.no_grad()
def quick_eval_future_loss(
    model: NotebookDeepSeekLM,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    horizon: int = 2,
) -> float:
    model.eval()
    losses = []
    shift = max(horizon - 1, 0)
    for _ in range(10):
        xb, yb = get_batch(data, batch_size, block_size, device)
        logits, _ = model(xb, None)
        if shift <= 0:
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        elif logits.size(1) <= shift:
            continue
        else:
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-shift, :].reshape(-1, logits.size(-1)),
                yb[:, shift:].reshape(-1),
            )
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / max(len(losses), 1))


def estimate_kv_cache_bytes(cfg: NotebookConfig, batch_size: int, seq_len: int, dtype_bytes: int = 4) -> int:
    head_dim = cfg.d_model // cfg.n_heads
    if cfg.attention_type == "mha":
        kv_features = cfg.d_model
    elif cfg.attention_type == "mqa":
        kv_features = head_dim
    elif cfg.attention_type == "gqa":
        kv_features = cfg.n_kv_heads * head_dim
    else:
        kv_features = cfg.kv_latent_dim
    return 2 * batch_size * seq_len * kv_features * dtype_bytes * cfg.n_layers


def run_variant(
    variant_name: str,
    attention_type: str,
    kv_latent_dim: int,
    n_kv_heads: int,
    moe_num_experts: int,
    mttp_steps: int,
    mttp_coeff: float,
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
        n_kv_heads=n_kv_heads,
        attention_type=attention_type,
        kv_latent_dim=kv_latent_dim,
        moe_num_experts=moe_num_experts,
        moe_hidden_mult=4,
        mttp_steps=mttp_steps,
        mttp_coeff=mttp_coeff,
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
    future_loss_h2 = quick_eval_future_loss(model, dataset.val_data, batch_size, block_size, device, horizon=2)
    ppl = float(torch.exp(torch.tensor(val_loss)).item())
    tokens = steps * batch_size * block_size
    toks_per_sec = tokens / max(elapsed, 1e-8)

    if device.type == "cuda":
        peak_mem = float(torch.cuda.max_memory_allocated(device=device) / (1024 * 1024))
    else:
        peak_mem = float(sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024))

    kv_cache_mb = estimate_kv_cache_bytes(cfg, batch_size=batch_size, seq_len=block_size) / (1024 * 1024)
    return {
        "variant": variant_name,
        "attention_type": attention_type,
        "kv_latent_dim": kv_latent_dim,
        "n_kv_heads": n_kv_heads,
        "moe_num_experts": moe_num_experts,
        "mttp_steps": mttp_steps,
        "mttp_coeff": mttp_coeff,
        "val_loss": val_loss,
        "future_loss_h2": future_loss_h2,
        "perplexity_proxy": ppl,
        "tokens_per_sec": float(toks_per_sec),
        "peak_mem_mb": peak_mem,
        "estimated_kv_cache_mb": float(kv_cache_mb),
    }


def print_table(rows: List[Dict[str, float]]) -> None:
    headers = [
        "variant",
        "attn",
        "mttp",
        "kv_lat",
        "kv_heads",
        "experts",
        "val_loss",
        "future_h2",
        "ppl_proxy",
        "tok/s",
        "peak_mem_mb",
        "est_kv_mb",
    ]
    print("\n" + " | ".join(headers))
    print("-" * 160)
    for r in rows:
        print(
            f"{r['variant']:>15} | {r['attention_type']:>4} | {r['mttp_steps']:>4d} | "
            f"{r['kv_latent_dim']:>6d} | {r['n_kv_heads']:>8d} | {r['moe_num_experts']:>7d} | "
            f"{r['val_loss']:.4f} | {r['future_loss_h2']:.4f} | {r['perplexity_proxy']:.2f} | "
            f"{r['tokens_per_sec']:.1f} | {r['peak_mem_mb']:.2f} | {r['estimated_kv_cache_mb']:.4f}"
        )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    text = load_text(args.text_path)
    dataset = build_char_dataset(text)

    attention_variants = [
        ("mha_ref", "mha", 64, 2, 4, 0, 0.25),
        ("mqa_ref", "mqa", 64, 1, 4, 0, 0.25),
        ("gqa_ref", "gqa", 64, 2, 4, 0, 0.25),
        ("mla_ref", "mla", 64, 2, 4, 0, 0.25),
    ]
    mttp_variants = [
        ("mla_mttp0", "mla", 64, 2, 4, 0, 0.25),
        ("mla_mttp1", "mla", 64, 2, 4, 1, 0.05),
    ]

    attention_rows = []
    for name, attention_type, kv_latent_dim, n_kv_heads, moe_num_experts, mttp_steps, mttp_coeff in attention_variants:
        print(f"Running benchmark for variant={name} ...")
        row = run_variant(
            variant_name=name,
            attention_type=attention_type,
            kv_latent_dim=kv_latent_dim,
            n_kv_heads=n_kv_heads,
            moe_num_experts=moe_num_experts,
            mttp_steps=mttp_steps,
            mttp_coeff=mttp_coeff,
            dataset=dataset,
            steps=args.steps,
            batch_size=args.batch_size,
            block_size=args.block_size,
            device=device,
        )
        attention_rows.append(row)

    mttp_rows = []
    for name, attention_type, kv_latent_dim, n_kv_heads, moe_num_experts, mttp_steps, mttp_coeff in mttp_variants:
        print(f"Running benchmark for variant={name} ...")
        row = run_variant(
            variant_name=name,
            attention_type=attention_type,
            kv_latent_dim=kv_latent_dim,
            n_kv_heads=n_kv_heads,
            moe_num_experts=moe_num_experts,
            mttp_steps=mttp_steps,
            mttp_coeff=mttp_coeff,
            dataset=dataset,
            steps=args.steps,
            batch_size=args.batch_size,
            block_size=args.block_size,
            device=device,
        )
        mttp_rows.append(row)

    print("\n=== Attention benchmark (MLA advantage) ===")
    print_table(attention_rows)
    print("\n=== MTTP benchmark (MLA backbone) ===")
    print_table(mttp_rows)

    payload = {
        "metadata": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "device": str(device),
        },
        "attention_benchmark": attention_rows,
        "mttp_benchmark": mttp_rows,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    print(f"\nSaved benchmark output to: {args.output}")


if __name__ == "__main__":
    main()
