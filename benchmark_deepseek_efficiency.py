import argparse
import json
import time
from dataclasses import asdict
from typing import Dict, List

import torch
import torch.nn.functional as F

from deepseek_llm import DeepSeekConfig, DeepSeekLM
from deepseek_llm.data import build_char_dataset, get_batch, load_text
from deepseek_llm.modules.attention_common import kv_cache_num_bytes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark DeepSeekLM efficiency and KV-cache impact.")
    parser.add_argument("--text-path", type=str, default="data/wikitext2_small_all.txt")
    parser.add_argument("--steps", type=int, default=140, help="Training steps per variant.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=96)
    parser.add_argument("--decode-new-tokens", type=int, default=96)
    parser.add_argument("--decode-prompt-len", type=int, default=96)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="assets/results/deepseek_efficiency.json")
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
def evaluate_loss(
    model: DeepSeekLM,
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    eval_iters: int = 20,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        xb, yb = get_batch(data, batch_size, block_size, device)
        _, loss, _ = model(xb, yb)
        losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / len(losses))


@torch.no_grad()
def benchmark_decode_no_cache(model: DeepSeekLM, prompt: torch.Tensor, new_tokens: int) -> float:
    seq = prompt.clone()
    start = time.perf_counter()
    for _ in range(new_tokens):
        logits, _, _ = model(seq[:, -model.cfg.block_size :])
        next_idx = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        seq = torch.cat([seq, next_idx], dim=1)
    elapsed = time.perf_counter() - start
    return float(new_tokens / max(elapsed, 1e-8))


@torch.no_grad()
def benchmark_decode_with_cache(model: DeepSeekLM, prompt: torch.Tensor, new_tokens: int) -> Dict[str, float]:
    logits, _, _, past_kv = model(prompt[:, -model.cfg.block_size :], use_cache=True)
    cache_bytes = sum(kv_cache_num_bytes(layer_cache) for layer_cache in past_kv)
    start = time.perf_counter()
    next_idx = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    for _ in range(new_tokens):
        logits, _, _, past_kv = model(next_idx, past_kv=past_kv, use_cache=True)
        next_idx = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    elapsed = time.perf_counter() - start
    return {
        "decode_tokens_per_sec_cache": float(new_tokens / max(elapsed, 1e-8)),
        "runtime_kv_cache_mb": float(cache_bytes / (1024 * 1024)),
    }


def train_variant(
    cfg: DeepSeekConfig,
    dataset,
    steps: int,
    batch_size: int,
    block_size: int,
    device: torch.device,
    decode_prompt_len: int,
    decode_new_tokens: int,
) -> Dict[str, float]:
    model = DeepSeekLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    model.train()
    start = time.perf_counter()
    for _ in range(steps):
        xb, yb = get_batch(dataset.train_data, batch_size, block_size, device)
        _, loss, _ = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    elapsed = time.perf_counter() - start

    val_loss = evaluate_loss(model, dataset.val_data, batch_size, block_size, device)
    ppl = float(torch.exp(torch.tensor(val_loss)).item())
    train_tokens = steps * batch_size * block_size
    train_tokens_per_sec = float(train_tokens / max(elapsed, 1e-8))
    param_count = int(sum(p.numel() for p in model.parameters()))
    param_mb = float(param_count * 4 / (1024 * 1024))
    peak_mem_mb = (
        float(torch.cuda.max_memory_allocated(device=device) / (1024 * 1024))
        if device.type == "cuda"
        else param_mb
    )

    prompt_len = min(max(8, decode_prompt_len), len(dataset.val_data) - 2)
    prompt = dataset.val_data[:prompt_len].unsqueeze(0).to(device)
    decode_no_cache = benchmark_decode_no_cache(model, prompt, new_tokens=decode_new_tokens)
    decode_cache = benchmark_decode_with_cache(model, prompt, new_tokens=decode_new_tokens)

    return {
        "val_loss": val_loss,
        "perplexity_proxy": ppl,
        "train_tokens_per_sec": train_tokens_per_sec,
        "decode_tokens_per_sec_no_cache": float(decode_no_cache),
        "decode_tokens_per_sec_cache": decode_cache["decode_tokens_per_sec_cache"],
        "decode_cache_speedup": float(decode_cache["decode_tokens_per_sec_cache"] / max(decode_no_cache, 1e-8)),
        "runtime_kv_cache_mb": decode_cache["runtime_kv_cache_mb"],
        "param_count_m": float(param_count / 1_000_000),
        "peak_mem_mb": peak_mem_mb,
    }


def run_benchmark(args: argparse.Namespace) -> Dict:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    text = load_text(args.text_path)
    dataset = build_char_dataset(text)

    variants = [
        ("mha", 64, 2),
        ("mqa", 64, 1),
        ("gqa", 64, 2),
        ("mla", 64, 2),
        ("mla_lat32", 32, 2),
    ]
    rows: List[Dict[str, float]] = []
    for attention_type, kv_latent_dim, n_kv_heads in variants:
        print(f"Running efficiency benchmark for {attention_type}...")
        cfg = DeepSeekConfig(
            vocab_size=len(dataset.stoi),
            block_size=args.block_size,
            n_layers=3,
            n_heads=4,
            n_kv_heads=n_kv_heads,
            d_model=128,
            mlp_hidden_mult=4,
            dropout=0.1,
            attention_type=attention_type if attention_type != "mla_lat32" else "mla",
            kv_latent_dim=kv_latent_dim,
            moe_num_experts=4,
            moe_every=2,
            use_moe=True,
        )
        metrics = train_variant(
            cfg=cfg,
            dataset=dataset,
            steps=args.steps,
            batch_size=args.batch_size,
            block_size=args.block_size,
            device=device,
            decode_prompt_len=args.decode_prompt_len,
            decode_new_tokens=args.decode_new_tokens,
        )
        row = {
            "variant": attention_type,
            "attention_type": cfg.attention_type,
            "kv_latent_dim": cfg.kv_latent_dim,
            "n_kv_heads": cfg.n_kv_heads,
            **metrics,
        }
        rows.append(row)

    return {
        "metadata": {
            "text_path": args.text_path,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "device": str(device),
            "seed": args.seed,
            "config_template": asdict(
                DeepSeekConfig(
                    block_size=args.block_size,
                    n_layers=3,
                    n_heads=4,
                    d_model=128,
                    mlp_hidden_mult=4,
                    moe_num_experts=4,
                )
            ),
        },
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    out = run_benchmark(args)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=True, indent=2)
    print(f"Saved efficiency benchmark to: {args.output}")


if __name__ == "__main__":
    main()
