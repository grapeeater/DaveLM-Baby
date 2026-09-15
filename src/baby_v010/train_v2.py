from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import BabyVNextConfig
from .data import LANG_TRAIN, read_u16
from .data_v2 import make_item
from .evaluate import evaluate_panels, language_ce

ROOT = Path(__file__).resolve().parents[2]
DEV_STREAM = Path(r"C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\LANGUAGE_DEV_STREAM.u16")
PANEL_PATH = ROOT / "data" / "generated" / "foundation_v2" / "panels.json"
CONFIG_PATH = ROOT / "configs" / "foundation_v1.json"
PROTOCOL = "BABY_V010_FOUNDATION_V2"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def language_batch(stream: torch.Tensor, rng: random.Random, batch: int, ctx: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    starts = [rng.randrange(0, stream.numel() - ctx - 1) for _ in range(batch)]
    offsets = torch.arange(ctx, dtype=torch.long)
    x = torch.stack([stream[start + offsets] for start in starts]).to(device)
    y = torch.stack([stream[start + offsets + 1] for start in starts]).to(device)
    return x, y


def stage_for(update: int) -> str:
    if update < 500:
        return "language_warmup"
    if update < 1800:
        return "primitive_induction"
    if update < 3500:
        return "short_retrieval"
    return "full_foundation"


def structured_batch(banks, rng: random.Random, batch: int, device: torch.device, stage: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    for row, item in enumerate(items):
        seq = sequences[row]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long, device=device)
        target_ids = torch.tensor(seq[1:], dtype=torch.long, device=device)
        x[row, : len(input_ids)] = input_ids
        y[row, : len(target_ids)] = target_ids
        start = len(item["input"]) - 1
        mask[row, start : start + len(item["target"])] = True
    return x, y, mask


def save_checkpoint(path: Path, model, optimizer, config, update: int, seed: int) -> None:
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
            "parent_checkpoint": None,
            "protected_material_opened": False,
        },
        tmp,
    )
    os.replace(tmp, path)


def run(seed: int, out: Path, updates: int, eval_interval: int, device: torch.device, batch_size: int) -> None:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"frozen panels missing: {PANEL_PATH}")
    set_seed(seed)
    rng = random.Random(seed)
    config = BabyVNextConfig.load(CONFIG_PATH)
    config.validate()
    from .model import BabyVNextLM

    model = BabyVNextLM(config).to(device)
    # Fresh-start optimization is deliberately lower than v1R2: the scaffold
    # should form before full-length retrieval is introduced.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05)
    train_stream = torch.tensor(read_u16(LANG_TRAIN), dtype=torch.long)
    dev_stream = torch.tensor(read_u16(DEV_STREAM), dtype=torch.long)
    banks = __import__("src.baby_v010.data", fromlist=["build_banks"]).build_banks(train_stream.tolist())
    panels = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise RuntimeError(f"output directory is not empty: {out}")
    (out / "RUN_CONFIG.json").write_text(json.dumps({"seed": seed, "updates": updates, "eval_interval": eval_interval, "device": str(device), "protocol": PROTOCOL, "parent_checkpoint": None, "protected_material_opened": False}, indent=2) + "\n", encoding="utf-8")
    baseline = language_ce(model, dev_stream, list(range(0, min(64 * 256, dev_stream.numel() - 257), 256)), device, limit=32)
    metrics_path = out / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as metrics:
        for update in range(updates + 1):
            stage = stage_for(update)
            if update == 0 or update % eval_interval == 0 or update == updates:
                began = time.perf_counter()
                probe_limit = None if update == updates and updates >= 6000 else 16
                report = evaluate_panels(model, panels, device, limit=probe_limit)
                report["update"] = update
                report["stage"] = stage
                report["language_dev_ce"] = language_ce(model, dev_stream, list(range(0, min(64 * 256, dev_stream.numel() - 257), 256)), device, limit=32)
                report["language_baseline_ce"] = baseline
                report["elapsed_eval_s"] = time.perf_counter() - began
                metrics.write(json.dumps(report) + "\n")
                metrics.flush()
                print(json.dumps({"update": update, "stage": stage, "language_dev_ce": report["language_dev_ce"], "novel": report["summaries"].get("novel"), "heldout_surface": report["summaries"].get("heldout_surface"), "device": str(device)}), flush=True)
                save_checkpoint(out / f"checkpoint_{update:04d}.pt", model, optimizer, config, update, seed)
            if update == updates:
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            structured = stage != "language_warmup" and (rng.random() < 0.80)
            if structured:
                x, y, mask = structured_batch(banks, rng, batch_size, device, stage)
                logits = model(x)
                loss = F.cross_entropy(logits[mask], y[mask])
                task = "structured"
            else:
                x, y = language_batch(train_stream, rng, batch_size, config.context_length, device)
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                task = "language"
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            if update % 25 == 0:
                print(json.dumps({"update": update + 1, "stage": stage, "task": task, "loss": float(loss.detach().cpu()), "grad_norm": float(grad_norm.detach().cpu()), "device": str(device)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=6000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    run(args.seed, args.out, args.updates, args.eval_interval, device_from_arg(args.device), args.batch_size)


if __name__ == "__main__":
    main()
