import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_plot(out_dir: str, name: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, name), dpi=180)
    plt.close()


def _variant_name(row: Dict) -> str:
    return row.get("variant", row.get("attention", "unknown"))


def plot_training(metrics: List[Dict], out_dir: str) -> None:
    steps = np.array([m["step"] for m in metrics])
    train_loss = np.array([m["train_loss"] for m in metrics], dtype=float)
    val_loss = np.array([m["val_loss"] for m in metrics], dtype=float)
    tps = np.array([m["tokens_per_sec"] for m in metrics], dtype=float)
    gap = val_loss - train_loss

    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_loss, marker="o", label="train")
    plt.plot(steps, val_loss, marker="o", label="val")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.25)
    save_plot(out_dir, "01_training_loss_curves.png")

    plt.figure(figsize=(8, 5))
    plt.plot(steps, np.log(train_loss + 1e-8), marker="o", label="log(train)")
    plt.plot(steps, np.log(val_loss + 1e-8), marker="o", label="log(val)")
    plt.title("Log Loss Curves")
    plt.xlabel("Step")
    plt.ylabel("log(loss)")
    plt.legend()
    plt.grid(alpha=0.25)
    save_plot(out_dir, "02_log_loss_curves.png")

    plt.figure(figsize=(8, 5))
    plt.plot(steps, tps, marker="o", color="tab:green")
    plt.title("Tokens / Second over Training")
    plt.xlabel("Step")
    plt.ylabel("Tokens/sec")
    plt.grid(alpha=0.25)
    save_plot(out_dir, "03_training_throughput.png")

    plt.figure(figsize=(8, 5))
    plt.plot(steps, gap, marker="o", color="tab:orange")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.title("Generalization Gap (val - train)")
    plt.xlabel("Step")
    plt.ylabel("Gap")
    plt.grid(alpha=0.25)
    save_plot(out_dir, "04_generalization_gap.png")

    # Per-interval improvements (positive is good).
    train_improve = np.zeros_like(train_loss)
    val_improve = np.zeros_like(val_loss)
    train_improve[1:] = train_loss[:-1] - train_loss[1:]
    val_improve[1:] = val_loss[:-1] - val_loss[1:]

    x = np.arange(len(steps))
    w = 0.38
    plt.figure(figsize=(9, 5))
    plt.bar(x - w / 2, train_improve, width=w, label="train improve")
    plt.bar(x + w / 2, val_improve, width=w, label="val improve")
    plt.xticks(x, steps)
    plt.title("Loss Improvement per Eval Interval")
    plt.xlabel("Step")
    plt.ylabel("Delta loss vs previous eval")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    save_plot(out_dir, "05_loss_improvement_bars.png")

    plt.figure(figsize=(8, 5))
    plt.scatter(tps, val_loss, c=steps, cmap="viridis", s=70)
    for s, xpt, ypt in zip(steps, tps, val_loss):
        plt.annotate(str(int(s)), (xpt, ypt), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.title("Training Throughput vs Validation Loss")
    plt.xlabel("Tokens/sec")
    plt.ylabel("Validation loss")
    plt.grid(alpha=0.25)
    cbar = plt.colorbar()
    cbar.set_label("Step")
    save_plot(out_dir, "06_train_speed_vs_val_loss_scatter.png")


def plot_benchmark(bench_rows: List[Dict], out_dir: str) -> None:
    labels = [_variant_name(r) for r in bench_rows]
    val_loss = np.array([r["val_loss"] for r in bench_rows], dtype=float)
    ppl = np.array([r["perplexity_proxy"] for r in bench_rows], dtype=float)
    tps = np.array([r["tokens_per_sec"] for r in bench_rows], dtype=float)
    mem = np.array([r["peak_mem_mb"] for r in bench_rows], dtype=float)
    kv = np.array([r["estimated_kv_cache_mb"] for r in bench_rows], dtype=float)
    x = np.arange(len(labels))

    def bar_plot(values: np.ndarray, title: str, ylabel: str, out_name: str, color: str = "tab:blue") -> None:
        plt.figure(figsize=(10, 5))
        bars = plt.bar(x, values, color=color)
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.title(title)
        plt.ylabel(ylabel)
        plt.grid(axis="y", alpha=0.25)
        for b, v in zip(bars, values):
            plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        save_plot(out_dir, out_name)

    bar_plot(val_loss, "Benchmark: Validation Loss by Variant", "val loss", "07_bench_val_loss_bar.png", "tab:red")
    bar_plot(ppl, "Benchmark: Perplexity Proxy by Variant", "ppl proxy", "08_bench_ppl_bar.png", "tab:purple")
    bar_plot(tps, "Benchmark: Throughput by Variant", "tokens/sec", "09_bench_tps_bar.png", "tab:green")
    bar_plot(mem, "Benchmark: Peak Memory by Variant", "MB", "10_bench_peak_mem_bar.png", "tab:orange")
    bar_plot(kv, "Benchmark: Estimated KV Cache by Variant", "MB", "11_bench_kv_cache_bar.png", "tab:brown")

    plt.figure(figsize=(8, 6))
    sizes = 80 + (mem - mem.min()) / max(mem.max() - mem.min(), 1e-8) * 240
    plt.scatter(tps, val_loss, s=sizes, c=kv, cmap="plasma", alpha=0.85)
    for lbl, xpt, ypt in zip(labels, tps, val_loss):
        plt.annotate(lbl, (xpt, ypt), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.title("Tradeoff: Speed vs Quality (bubble=size~mem, color~kv)")
    plt.xlabel("Tokens/sec")
    plt.ylabel("Validation loss")
    plt.grid(alpha=0.25)
    cbar = plt.colorbar()
    cbar.set_label("Estimated KV MB")
    save_plot(out_dir, "12_tradeoff_speed_quality_bubble.png")

    plt.figure(figsize=(8, 6))
    plt.scatter(kv, val_loss, s=130, c=tps, cmap="viridis")
    for lbl, xpt, ypt in zip(labels, kv, val_loss):
        plt.annotate(lbl, (xpt, ypt), fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.title("Tradeoff: KV Cache vs Validation Loss")
    plt.xlabel("Estimated KV cache (MB)")
    plt.ylabel("Validation loss")
    plt.grid(alpha=0.25)
    cbar = plt.colorbar()
    cbar.set_label("Tokens/sec")
    save_plot(out_dir, "13_tradeoff_kv_vs_quality_scatter.png")

    efficiency = tps / np.maximum(ppl, 1e-8)
    order = np.argsort(-efficiency)
    plt.figure(figsize=(10, 5))
    plt.bar(np.array(labels)[order], efficiency[order], color="tab:cyan")
    plt.xticks(rotation=20, ha="right")
    plt.title("Efficiency Score (tokens/sec / ppl_proxy)")
    plt.ylabel("Efficiency")
    plt.grid(axis="y", alpha=0.25)
    save_plot(out_dir, "14_efficiency_score_bar.png")

    plt.figure(figsize=(10, 5))
    speed_rank = np.argsort(np.argsort(-tps))
    quality_rank = np.argsort(np.argsort(val_loss))
    mem_rank = np.argsort(np.argsort(mem))
    kv_rank = np.argsort(np.argsort(kv))
    rank_sum = speed_rank + quality_rank + mem_rank + kv_rank
    order = np.argsort(rank_sum)
    plt.bar(np.array(labels)[order], rank_sum[order], color="tab:gray")
    plt.xticks(rotation=20, ha="right")
    plt.title("Composite Rank (lower is better)")
    plt.ylabel("Rank sum")
    plt.grid(axis="y", alpha=0.25)
    save_plot(out_dir, "15_composite_rank_bar.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate many plots from training and benchmark results.")
    parser.add_argument("--metrics", default="runs/notebook_full_run/metrics.json")
    parser.add_argument("--benchmark", default="runs/notebook_benchmark_full.json")
    parser.add_argument("--out-dir", default="runs/plots")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    metrics = load_json(args.metrics)
    benchmark = load_json(args.benchmark)

    plot_training(metrics, args.out_dir)
    plot_benchmark(benchmark, args.out_dir)

    print(f"Saved plots to: {args.out_dir}")
    print("Generated 15 plot files.")


if __name__ == "__main__":
    main()
