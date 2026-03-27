import argparse
import json
import os
from typing import Dict, List, Tuple

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


def split_benchmark_payload(benchmark_payload) -> Tuple[List[Dict], List[Dict]]:
    # Backward compatible with older benchmark format (flat list).
    if isinstance(benchmark_payload, list):
        return benchmark_payload, []
    attention = benchmark_payload.get("attention_benchmark", [])
    mttp = benchmark_payload.get("mttp_benchmark", [])
    return attention, mttp


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


def plot_attention_benchmark(bench_rows: List[Dict], out_dir: str) -> None:
    if not bench_rows:
        return
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

    bar_plot(val_loss, "Attention Benchmark: Validation Loss", "val loss", "07_attention_val_loss_bar.png", "tab:red")
    bar_plot(ppl, "Attention Benchmark: Perplexity Proxy", "ppl proxy", "08_attention_ppl_bar.png", "tab:purple")
    bar_plot(tps, "Attention Benchmark: Throughput", "tokens/sec", "09_attention_tps_bar.png", "tab:green")
    bar_plot(mem, "Attention Benchmark: Peak Memory", "MB", "10_attention_peak_mem_bar.png", "tab:orange")
    bar_plot(kv, "Attention Benchmark: Estimated KV Cache", "MB", "11_attention_kv_cache_bar.png", "tab:brown")

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
    save_plot(out_dir, "12_attention_tradeoff_speed_quality_bubble.png")

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
    save_plot(out_dir, "13_attention_tradeoff_kv_vs_quality_scatter.png")

    efficiency = tps / np.maximum(ppl, 1e-8)
    order = np.argsort(-efficiency)
    plt.figure(figsize=(10, 5))
    plt.bar(np.array(labels)[order], efficiency[order], color="tab:cyan")
    plt.xticks(rotation=20, ha="right")
    plt.title("Efficiency Score (tokens/sec / ppl_proxy)")
    plt.ylabel("Efficiency")
    plt.grid(axis="y", alpha=0.25)
    save_plot(out_dir, "14_attention_efficiency_score_bar.png")

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
    save_plot(out_dir, "15_attention_composite_rank_bar.png")


def plot_mttp_advantage(mttp_rows: List[Dict], out_dir: str) -> None:
    if not mttp_rows:
        return

    baseline_candidates = [r for r in mttp_rows if int(r.get("mttp_steps", 0)) == 0]
    if baseline_candidates:
        baseline = min(baseline_candidates, key=lambda r: float(r["val_loss"]))
    else:
        baseline = min(mttp_rows, key=lambda r: int(r.get("mttp_steps", 0)))
    experimental = min(mttp_rows, key=lambda r: float(r["val_loss"]))

    labels = [_variant_name(baseline), _variant_name(experimental)]
    val_loss = np.array([baseline["val_loss"], experimental["val_loss"]], dtype=float)
    ppl = np.array([baseline["perplexity_proxy"], experimental["perplexity_proxy"]], dtype=float)
    tps = np.array([baseline["tokens_per_sec"], experimental["tokens_per_sec"]], dtype=float)
    mem = np.array([baseline["peak_mem_mb"], experimental["peak_mem_mb"]], dtype=float)
    kv = np.array([baseline["estimated_kv_cache_mb"], experimental["estimated_kv_cache_mb"]], dtype=float)
    future_h2 = np.array(
        [
            float(baseline.get("future_loss_h2", baseline["val_loss"])),
            float(experimental.get("future_loss_h2", experimental["val_loss"])),
        ],
        dtype=float,
    )
    x = np.arange(2)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(x, val_loss, color=["tab:gray", "tab:blue"])
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.title("MTTP Comparison: Validation Loss")
    plt.ylabel("val loss")
    for b, v in zip(bars, val_loss):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.grid(axis="y", alpha=0.25)
    save_plot(out_dir, "16_mttp_val_loss_comparison.png")

    plt.figure(figsize=(7, 5))
    bars = plt.bar(x, tps, color=["tab:gray", "tab:green"])
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.title("MTTP Comparison: Throughput")
    plt.ylabel("tokens/sec")
    for b, v in zip(bars, tps):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    plt.grid(axis="y", alpha=0.25)
    save_plot(out_dir, "17_mttp_throughput_comparison.png")

    plt.figure(figsize=(7, 5))
    bars = plt.bar(x, future_h2, color=["tab:gray", "tab:blue"])
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.title("MTTP Comparison: 2-Step Future Token Loss")
    plt.ylabel("future token loss (horizon=2)")
    for b, v in zip(bars, future_h2):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.grid(axis="y", alpha=0.25)
    save_plot(out_dir, "18_mttp_future_loss_h2_comparison.png")

    plt.figure(figsize=(8, 5))
    metrics = np.array(
        [
            (val_loss[0] - val_loss[1]) / max(val_loss[0], 1e-8) * 100.0,
            (ppl[0] - ppl[1]) / max(ppl[0], 1e-8) * 100.0,
            (future_h2[0] - future_h2[1]) / max(future_h2[0], 1e-8) * 100.0,
            (tps[1] - tps[0]) / max(tps[0], 1e-8) * 100.0,
            (mem[0] - mem[1]) / max(mem[0], 1e-8) * 100.0,
            (kv[0] - kv[1]) / max(kv[0], 1e-8) * 100.0,
        ],
        dtype=float,
    )
    names = ["val loss", "ppl", "future-h2 loss", "throughput", "peak mem", "kv cache"]
    colors = ["tab:green" if m >= 0 else "tab:red" for m in metrics]
    bars = plt.bar(names, metrics, color=colors)
    plt.axhline(0.0, linestyle="--", linewidth=1, color="black")
    plt.title("MTTP Gains vs Baseline (positive is better)")
    plt.ylabel("% change")
    for b, v in zip(bars, metrics):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:+.2f}%",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
        )
    plt.grid(axis="y", alpha=0.25)
    save_plot(out_dir, "19_mttp_gain_vs_baseline.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate many plots from training and benchmark results.")
    parser.add_argument("--metrics", default="runs/notebook_full_run/metrics.json")
    parser.add_argument("--benchmark", default="runs/notebook_benchmark_full.json")
    parser.add_argument("--out-dir", default="runs/plots")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    metrics = load_json(args.metrics)
    benchmark_payload = load_json(args.benchmark)
    attention_benchmark, mttp_benchmark = split_benchmark_payload(benchmark_payload)

    plot_training(metrics, args.out_dir)
    plot_attention_benchmark(attention_benchmark, args.out_dir)
    plot_mttp_advantage(mttp_benchmark, args.out_dir)

    print(f"Saved plots to: {args.out_dir}")
    print("Generated plot files for training, attention benchmark, and MTTP comparison.")


if __name__ == "__main__":
    main()
