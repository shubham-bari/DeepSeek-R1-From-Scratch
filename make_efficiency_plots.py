import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate efficiency plots for DeepSeekLM benchmark.")
    parser.add_argument("--benchmark", type=str, default="assets/results/deepseek_efficiency.json")
    parser.add_argument("--out-dir", type=str, default="assets/plots_efficiency")
    return parser.parse_args()


def load_rows(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["rows"]


def save_plot(out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, name), dpi=180)
    plt.close()


def labels_and_arrays(rows: List[Dict]):
    labels = [r["variant"] for r in rows]
    val_loss = np.array([r["val_loss"] for r in rows], dtype=float)
    train_tps = np.array([r["train_tokens_per_sec"] for r in rows], dtype=float)
    decode_nc = np.array([r["decode_tokens_per_sec_no_cache"] for r in rows], dtype=float)
    decode_c = np.array([r["decode_tokens_per_sec_cache"] for r in rows], dtype=float)
    speedup = np.array([r["decode_cache_speedup"] for r in rows], dtype=float)
    kv_mb = np.array([r["runtime_kv_cache_mb"] for r in rows], dtype=float)
    params_m = np.array([r["param_count_m"] for r in rows], dtype=float)
    ppl = np.array([r["perplexity_proxy"] for r in rows], dtype=float)
    peak_mem = np.array([r["peak_mem_mb"] for r in rows], dtype=float)
    return labels, val_loss, train_tps, decode_nc, decode_c, speedup, kv_mb, params_m, ppl, peak_mem


def bar(labels: List[str], values: np.ndarray, title: str, ylabel: str, out_dir: str, name: str, color: str) -> None:
    x = np.arange(len(labels))
    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, values, color=color)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    save_plot(out_dir, name)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.benchmark)
    labels, val_loss, train_tps, decode_nc, decode_c, speedup, kv_mb, params_m, ppl, peak_mem = labels_and_arrays(rows)
    x = np.arange(len(labels))

    bar(
        labels,
        val_loss,
        "Efficiency Benchmark: Validation Loss",
        "val loss",
        args.out_dir,
        "01_eff_val_loss.png",
        "tab:red",
    )
    bar(
        labels,
        train_tps,
        "Efficiency Benchmark: Training Throughput",
        "tokens/sec",
        args.out_dir,
        "02_eff_train_tps.png",
        "tab:green",
    )
    bar(
        labels,
        speedup,
        "Decode Cache Speedup (cached / no-cache)",
        "speedup",
        args.out_dir,
        "03_eff_decode_cache_speedup.png",
        "tab:blue",
    )
    bar(
        labels,
        kv_mb,
        "Runtime KV Cache Footprint",
        "MB",
        args.out_dir,
        "04_eff_runtime_kv_cache_mb.png",
        "tab:brown",
    )
    bar(
        labels,
        peak_mem,
        "Peak Memory Footprint",
        "MB",
        args.out_dir,
        "05_eff_peak_memory_mb.png",
        "tab:orange",
    )
    bar(
        labels,
        params_m,
        "Parameter Count",
        "Millions",
        args.out_dir,
        "06_eff_param_count_m.png",
        "tab:purple",
    )

    plt.figure(figsize=(10, 5))
    w = 0.36
    plt.bar(x - w / 2, decode_nc, width=w, label="no cache", color="tab:gray")
    plt.bar(x + w / 2, decode_c, width=w, label="with cache", color="tab:cyan")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.title("Decode Throughput: Cached vs No-Cache")
    plt.ylabel("tokens/sec")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_plot(args.out_dir, "07_eff_decode_cached_vs_uncached.png")

    plt.figure(figsize=(8, 6))
    sizes = 90 + (kv_mb - kv_mb.min()) / max(kv_mb.max() - kv_mb.min(), 1e-8) * 220
    plt.scatter(decode_c, val_loss, s=sizes, c=train_tps, cmap="viridis", alpha=0.85)
    for lbl, xpt, ypt in zip(labels, decode_c, val_loss):
        plt.annotate(lbl, (xpt, ypt), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.title("Quality vs Cached Decode Speed (bubble size ~ KV MB)")
    plt.xlabel("decode tokens/sec (cache)")
    plt.ylabel("validation loss")
    plt.grid(alpha=0.25)
    cbar = plt.colorbar()
    cbar.set_label("training tokens/sec")
    save_plot(args.out_dir, "08_eff_quality_vs_cached_decode_scatter.png")

    # Composite normalized efficiency score (higher is better).
    decode_norm = decode_c / np.maximum(decode_c.max(), 1e-8)
    train_norm = train_tps / np.maximum(train_tps.max(), 1e-8)
    loss_norm = val_loss.min() / np.maximum(val_loss, 1e-8)
    kv_norm = kv_mb.min() / np.maximum(kv_mb, 1e-8)
    score = 0.35 * decode_norm + 0.25 * train_norm + 0.25 * loss_norm + 0.15 * kv_norm
    order = np.argsort(-score)
    plt.figure(figsize=(10, 5))
    plt.bar(np.array(labels)[order], score[order], color="tab:olive")
    plt.xticks(rotation=20, ha="right")
    plt.title("Composite Efficiency Score (higher is better)")
    plt.ylabel("score")
    plt.grid(axis="y", alpha=0.25)
    save_plot(args.out_dir, "09_eff_composite_score.png")

    print(f"Saved efficiency plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
