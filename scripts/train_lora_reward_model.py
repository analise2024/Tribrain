#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from tribrain.trainers.lora_reward_model import LoRATrainConfig, train_text_reward_model_lora


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-jsonl", type=Path, required=True)
    p.add_argument("--base-model", default="distilbert-base-uncased")
    p.add_argument("--output-dir", default="out_lora_reward_model")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=384)
    args = p.parse_args()

    cfg = LoRATrainConfig(
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_length=args.max_length,
    )
    out = train_text_reward_model_lora(args.pairs_jsonl, cfg)
    print(str(out))


if __name__ == "__main__":
    main()
