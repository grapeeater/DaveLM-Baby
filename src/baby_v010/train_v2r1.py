from __future__ import annotations

import argparse
import json
import random

import torch

from . import train_v2 as base
from .data_v2 import make_item


PROTOCOL = "BABY_V010_FOUNDATION_V2R1"


def structured_batch(banks, rng: random.Random, batch: int, device: torch.device, stage: str):
    """Build a batch whose early capability loss excludes deterministic suffixes."""
    items = []
    for _ in range(batch):
        if stage == "primitive_induction":
            difficulty = "primitive"
            kind = "induction" if rng.random() < 0.80 else "keyed"
        elif stage == "short_retrieval":
            difficulty = "short"
            kind = "induction" if rng.random() < 0.35 else "keyed"
        else:
            difficulty = "full"
            kind = "induction" if rng.random() < 0.25 else "keyed"
        low_prior = stage == "full_foundation" and rng.random() < 0.20
        items.append(make_item(rng, banks, difficulty=difficulty, kind=kind, low_prior=low_prior))

    sequences = [[*item["input"], *item["target"]] for item in items]
    max_len = max(len(seq) for seq in sequences)
    x = torch.zeros((batch, max_len - 1), dtype=torch.long, device=device)
    y = torch.zeros((batch, max_len - 1), dtype=torch.long, device=device)
    mask = torch.zeros((batch, max_len - 1), dtype=torch.bool, device=device)
    answer_only = stage in {"primitive_induction", "short_retrieval"}
    for row, item in enumerate(items):
        seq = sequences[row]
        x[row, : len(seq) - 1] = torch.tensor(seq[:-1], dtype=torch.long, device=device)
        y[row, : len(seq) - 1] = torch.tensor(seq[1:], dtype=torch.long, device=device)
        start = len(item["input"]) - 1
        train_len = len(item["target_span"]) if answer_only else len(item["target"])
        mask[row, start : start + train_len] = True
    return x, y, mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=__import__("pathlib").Path, required=True)
    parser.add_argument("--updates", type=int, default=6000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    base.structured_batch = structured_batch
    base.PROTOCOL = PROTOCOL
    base.run(args.seed, args.out, args.updates, args.eval_interval, base.device_from_arg(args.device), args.batch_size)


if __name__ == "__main__":
    main()
