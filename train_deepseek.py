import argparse
import json
import os
import time
from dataclasses import asdict
from typing import Tuple

import torch

from notebook_components import (
    NotebookAdam,
    NotebookConfig,
    NotebookDeepSeekLM,
    build_char_dataset,
    get_batch,
    load_text,
)


@torch.no_grad()
def estimate_loss(
    model: NotebookDeepSeekLM,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(data, batch_size, block_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = float(sum(losses) / len(losses))
    model.train()
    return out["train"], out["val"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train notebook-based MLA+MoE LM (with RoPE, custom Adam)."
    )
    parser.add_argument("--text-path", type=str, default="", help="Path to training text file.")
    parser.add_argument("--out-dir", type=str, default="runs/default", help="Output directory.")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--eval-interval", type=int, default=30)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--attention-type", type=str, default="mla", choices=["mha", "mqa", "gqa", "mla"])
    parser.add_argument("--kv-latent-dim", type=int, default=64)
    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--moe-hidden-mult", type=int, default=4)
    parser.add_argument("--mttp-steps", type=int, default=0, help="Number of future-token heads for MTTP.")
    parser.add_argument("--mttp-coeff", type=float, default=0.25, help="Loss weight for MTTP auxiliary objective.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--adam-decay", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
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


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)

    text = load_text(args.text_path)
    dataset = build_char_dataset(text)
    cfg = NotebookConfig(
        vocab_size=len(dataset.stoi),
        block_size=args.block_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        n_layers=args.n_layers,
        attention_type=args.attention_type,
        kv_latent_dim=args.kv_latent_dim,
        moe_num_experts=args.moe_num_experts,
        moe_hidden_mult=args.moe_hidden_mult,
        mttp_steps=args.mttp_steps,
        mttp_coeff=args.mttp_coeff,
        dropout=args.dropout,
    )
    model = NotebookDeepSeekLM(cfg).to(device)
    optimizer = NotebookAdam(lr=args.lr, decay=args.adam_decay)
    params = list(model.parameters())

    history = []
    start_time = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        xb, yb = get_batch(dataset.train_data, args.batch_size, args.block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(params)
        loss.backward()
        # Explicit L2 regularization to mirror weight decay behavior.
        if args.weight_decay > 0:
            for p in params:
                if p.grad is not None:
                    p.grad.data = p.grad.data + args.weight_decay * p.data
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.pre_update_params()
        optimizer.update_params(params)
        optimizer.post_update_params()

        if step % args.eval_interval == 0 or step == 1 or step == args.steps:
            train_loss, val_loss = estimate_loss(
                model=model,
                train_data=dataset.train_data,
                val_data=dataset.val_data,
                batch_size=args.batch_size,
                block_size=args.block_size,
                eval_iters=args.eval_iters,
                device=device,
            )
            elapsed = time.time() - start_time
            tokens_seen = step * args.batch_size * args.block_size
            tokens_per_sec = tokens_seen / max(elapsed, 1e-8)
            row = {
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "tokens_per_sec": tokens_per_sec,
            }
            history.append(row)
            print(f"step={step:4d} train={train_loss:.4f} val={val_loss:.4f} tok/s={tokens_per_sec:.1f}")

    ckpt_path = os.path.join(args.out_dir, "model.pt")
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, ckpt_path)
    with open(os.path.join(args.out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(dataset.stoi, f, ensure_ascii=True, indent=2)
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=True, indent=2)

    # Quick qualitative sample.
    model.eval()
    prompt = "DeepSeek "
    prompt_idx = torch.tensor([[dataset.stoi.get(ch, 0) for ch in prompt]], device=device)
    out_idx = model.generate(prompt_idx, max_new_tokens=80)
    decoded = "".join(dataset.itos[int(i)] for i in out_idx[0].tolist())
    with open(os.path.join(args.out_dir, "sample.txt"), "w", encoding="utf-8") as f:
        f.write(decoded)
    print(f"\nSaved checkpoint to: {ckpt_path}")
    print(f"Sample:\n{decoded}")


if __name__ == "__main__":
    main()
