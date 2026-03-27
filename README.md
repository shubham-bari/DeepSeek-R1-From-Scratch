# DeepSeek-Style LLM From Scratch (Notebook Components)

Build a modular language model from your own notebook implementations:
- `MLA` (Multi-Head Latent Attention)
- `MoE` (Mixture of Experts, top-1 routing)
- custom `Adam` optimizer API (`pre_update_params` / `update_params` / `post_update_params`)
- `RoPE` rotary positional encoding

No DeepSeek package import is used in the training/benchmark pipeline.

## What This Project Does

- converts notebook ideas into clean, reusable Python modules
- trains a compact character-level language model end-to-end
- benchmarks multiple MLA+MoE configurations for speed, memory, and quality
- auto-generates a large plot suite for experiment analysis

Core files:
- `notebook_components.py` - notebook-based MLA, MoE, Adam, RoPE model stack
- `train_deepseek.py` - training script
- `benchmark_experiments.py` - controlled benchmark experiments
- `make_plots.py` - generates 15 analysis plots from JSON results

## Real Dataset Used

- **Tiny Shakespeare** (`data/tiny_shakespeare.txt`)
- Source: Karpathy char-rnn public dataset
- Size used: ~1.11M characters, 65 unique chars

This is small enough to run quickly on a laptop, but still realistic enough for reproducible LLM training/benchmark comparisons.

## Reproducible Commands

Setup:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch numpy matplotlib
```

Download dataset:

```bash
mkdir -p data
curl -L "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" -o data/tiny_shakespeare.txt
```

Train:

```bash
.venv/bin/python train_deepseek.py \
  --text-path data/tiny_shakespeare.txt \
  --steps 250 \
  --eval-interval 50 \
  --eval-iters 30 \
  --batch-size 32 \
  --block-size 128 \
  --d-model 192 \
  --n-layers 4 \
  --n-heads 6 \
  --kv-latent-dim 64 \
  --moe-num-experts 4 \
  --out-dir runs/tinyshakespeare_run
```

Benchmark:

```bash
.venv/bin/python benchmark_experiments.py \
  --text-path data/tiny_shakespeare.txt \
  --steps 70 \
  --batch-size 24 \
  --block-size 128 \
  --output runs/tinyshakespeare_benchmark.json
```

Generate plots:

```bash
MPLCONFIGDIR="$PWD/.mplconfig" MPLBACKEND=Agg \
.venv/bin/python make_plots.py \
  --metrics runs/tinyshakespeare_run/metrics.json \
  --benchmark runs/tinyshakespeare_benchmark.json \
  --out-dir runs/plots_tinyshakespeare
```

## Results (Tiny Shakespeare)

### Training progression

From `runs/tinyshakespeare_run/metrics.json`:
- Step 1: train `3.7967`, val `3.7981`
- Step 50: train `2.6320`, val `2.6459`
- Step 100: train `2.5421`, val `2.5532`
- Step 150: train `2.5164`, val `2.5365`
- Step 200: train `2.4958`, val `2.5064`
- Step 250: train `2.4245`, val `2.4374`

Throughput rises to ~`19.8k tokens/sec` during this run.

### Benchmark comparison

From `runs/tinyshakespeare_benchmark.json`:

| Variant | Val Loss | PPL Proxy | Tokens/sec | Peak Mem (MB) | Est KV Cache (MB) |
|---|---:|---:|---:|---:|---:|
| `mla_lat32_exp4` | 2.7112 | 15.05 | **56319.8** | 6.59 | **2.25** |
| `mla_lat64_exp4` | **2.7128** | **15.07** | 55919.6 | 6.73 | 4.50 |
| `mla_lat64_exp8` | 2.7212 | 15.20 | 48841.0 | 12.76 | 4.50 |
| `mla_lat96_exp8` | 2.7153 | 15.11 | 47257.8 | 12.90 | 6.75 |

Practical pick in this run:
- **Best speed/memory efficiency**: `mla_lat32_exp4`
- **Best quality among tested variants (very close overall)**: `mla_lat32_exp4`

## Why These Results Happen

- Lower `kv_latent_dim` reduces latent KV representation size, so memory and cache pressure drop, improving throughput.
- More experts (`exp8`) increase parameter count and routing overhead, which raises memory and lowers speed on this scale.
- Quality differences are modest because all variants were trained for short benchmark schedules; longer training usually separates them more clearly.
- RoPE helps position-aware attention without adding learned positional embedding parameters.

## Plot Outputs

`make_plots.py` generates 15 plots per run (loss curves, throughput trends, memory bars, tradeoff scatters, efficiency ranking, composite ranking).  
Current generated set: `runs/plots_tinyshakespeare/`.

## Roadmap

- Add longer-run benchmark presets (e.g., 1k+ steps)
- Add token-level datasets (WikiText-2, OpenWebText subset)
- Add validation text generation metrics and qualitative samples table
- Add optional FlashAttention and mixed precision modes
