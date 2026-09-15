from __future__ import annotations

"""Continue the frozen v2R4 objective from a verified fresh-run handoff.

This utility does not define a new curriculum or alter thresholds. It exists so
the clean U6000 checkpoint can be explored after the owner handoff without
pretending that the continuation is an independent fresh-initialization
replication. The parent checkpoint and stochastic handoff are recorded.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import BabyVNextConfig
from .data import LANG_TRAIN, build_banks, read_u16
from .data_v2 import make_item
from .evaluate import evaluate_panels, language_ce
from .train_v2r4 import (
    HIGH_LR,
    LANGUAGE_END,
    LOW_LR,
    capability_optimizer,
    device_from_arg,
    language_batch,
    set_seed,
    structured_batch,
)

ROOT = Path(__file__).resolve().parents[2]
DEV_STREAM = Path(r"C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\LANGUAGE_DEV_STREAM.u16")
PANEL_PATH = ROOT / "data" / "generated" / "foundation_v2" / "panels.json"
CONFIG_PATH = ROOT / "configs" / "foundation_v1.json"
PROTOCOL = "BABY_V010_FOUNDATION_V2R4"
PRIMITIVE_END = 8000
SHORT_END = 10000


def stage_for(update: int) -> str:
    if update < PRIMITIVE_END:
        return "primitive_identity"
    if update < SHORT_END:
        return "short_retrieval"
    return "full_foundation"


def advance_language_rng(rng: random.Random, stream_size: int, context: int, batch: int) -> None:
    """Recreate the Python data-sampling position reached by U6000.

    Torch's dropout state is intentionally treated as a new continuation
    stochastic state and is recorded as such; no claim of bitwise replay is
    made.
    """
    upper = stream_size - context - 1
    for _ in range(LANGUAGE_END * 4):
        for _ in range(batch):
            rng.randrange(0, upper)


def save_checkpoint(path: Path, model, optimizer, config, update: int, seed: int, parent: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "protocol": PROTOCOL,
            "lineage": "Baby v0.10",
            "update": update,
            "seed": seed,
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "parent_checkpoint": parent,
            "protected_material_opened": False,
            "continuation_stochastic_state": "new_after_owner_handoff",
        },
        tmp,
    )
    os.replace(tmp, path)


def run(parent: Path, out: Path, seed: int, updates: int, eval_interval: int, device: torch.device, batch_size: int) -> None:
    if not parent.exists():
        raise FileNotFoundError(parent)
    if updates <= LANGUAGE_END:
        raise ValueError("continuation endpoint must exceed U6000")
    set_seed(seed + LANGUAGE_END)
    rng = random.Random(seed)
    config = BabyVNextConfig.load(CONFIG_PATH)
    config.validate()
    from .model import BabyVNextLM

    model = BabyVNextLM(config).to(device)
    checkpoint = torch.load(parent, map_location=device, weights_only=False)
    if checkpoint.get("protocol") != PROTOCOL or checkpoint.get("update") != LANGUAGE_END:
        raise RuntimeError("parent is not the verified v2R4 U6000 checkpoint")
    if checkpoint.get("parent_checkpoint") is not None or checkpoint.get("protected_material_opened"):
        raise RuntimeError("parent provenance is not a clean fresh handoff")
    model.load_state_dict(checkpoint["model_state_dict"])
    train_stream = torch.tensor(read_u16(LANG_TRAIN), dtype=torch.long)
    dev_stream = torch.tensor(read_u16(DEV_STREAM), dtype=torch.long)
    banks = build_banks(train_stream.tolist())
    panels = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    advance_language_rng(rng, train_stream.numel(), config.context_length, batch_size)
    optimizer, low_count, high_count = capability_optimizer(model)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError(f"output directory is not empty: {out}")
    (out / "RUN_CONFIG.json").write_text(json.dumps({
        "protocol": PROTOCOL,
        "lineage": "Baby v0.10",
        "seed": seed,
        "start_update": LANGUAGE_END,
        "updates": updates,
        "eval_interval": eval_interval,
        "device": str(device),
        "parent_checkpoint": str(parent),
        "parent_checkpoint_update": LANGUAGE_END,
        "protected_material_opened": False,
        "continuation_stochastic_state": "new_after_owner_handoff",
        "lower_scope_params": low_count,
        "upper_scope_params": high_count,
        "lower_scope_lr": LOW_LR,
        "upper_scope_lr": HIGH_LR,
    }, indent=2) + "\n", encoding="utf-8")
    baseline = language_ce(model, dev_stream, list(range(0, min(64 * 256, dev_stream.numel() - 257), 256)), device, limit=32)
    with (out / "metrics.jsonl").open("w", encoding="utf-8") as metrics:
        for update in range(LANGUAGE_END, updates + 1):
            stage = stage_for(update)
            if update == LANGUAGE_END or update % eval_interval == 0 or update == updates:
                began = time.perf_counter()
                probe_limit = None if update == updates and updates >= 16000 else 16
                report = evaluate_panels(model, panels, device, limit=probe_limit)
                report.update({
                    "update": update,
                    "stage": stage,
                    "language_dev_ce": language_ce(model, dev_stream, list(range(0, min(64 * 256, dev_stream.numel() - 257), 256)), device, limit=32),
                    "language_baseline_ce": baseline,
                    "elapsed_eval_s": time.perf_counter() - began,
                    "continuation": True,
                })
                metrics.write(json.dumps(report) + "\n")
                metrics.flush()
                print(json.dumps({"update": update, "stage": stage, "language_dev_ce": report["language_dev_ce"], "novel": report["summaries"].get("novel"), "heldout_surface": report["summaries"].get("heldout_surface"), "device": str(device)}), flush=True)
                save_checkpoint(out / f"checkpoint_{update:05d}.pt", model, optimizer, config, update, seed, str(parent))
            if update == updates:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if rng.random() < 0.20:
                x, y = language_batch(train_stream, rng, batch_size, config.context_length, device)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                task = "language_retention"
            else:
                x, y, mask = structured_batch(banks, rng, batch_size, device, stage)
                logits = model(x)
                loss = F.cross_entropy(logits[mask], y[mask])
                task = "structured"
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            if update % 25 == 0:
                print(json.dumps({"update": update + 1, "stage": stage, "task": task, "loss": float(loss.detach().cpu()), "grad_norm": float(grad_norm.detach().cpu()), "device": str(device)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=8000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    run(args.parent, args.out, args.seed, args.updates, args.eval_interval, device_from_arg(args.device), args.batch_size)


if __name__ == "__main__":
    main()
