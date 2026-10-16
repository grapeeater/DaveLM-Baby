"""Checkpoints: atomic writes, hash sidecars, resume discovery.

A checkpoint directory ``ckpt_<step:07d>`` contains ``checkpoint.pt`` (model, optimizer, step, config, RNG states,
tokenizer fingerprint, running statistics) and ``CHECKPOINT.json`` (metadata + SHA-256 of the .pt). The .pt is
written to a temporary name and renamed into place, so an interrupted save never leaves a half-written file that
looks valid. All checkpoints of a run are kept: they are evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def checkpoint_dir(run_dir: Path, step: int) -> Path:
    return run_dir / f"ckpt_{step:07d}"


def save_checkpoint(run_dir: Path, step: int, model, optimizer, cfg_dict: dict, extra: dict) -> Path:
    d = checkpoint_dir(run_dir, step)
    d.mkdir(parents=True, exist_ok=True)
    state = {
        "format": "rnc-checkpoint-1",
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": cfg_dict,
        "rng": {
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "extra": extra,
        "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = d / "checkpoint.pt.partial"
    final = d / "checkpoint.pt"
    torch.save(state, tmp)
    os.replace(tmp, final)
    meta = {
        "step": step,
        "path": str(final),
        "bytes": final.stat().st_size,
        "sha256": _sha256(final),
        "saved_utc": state["saved_utc"],
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "extra": {k: v for k, v in extra.items() if isinstance(v, (int, float, str, bool, type(None)))},
    }
    (d / "CHECKPOINT.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d


def latest_checkpoint(run_dir: Path) -> Path | None:
    if not run_dir.exists():
        return None
    cands = []
    for p in run_dir.iterdir():
        if p.is_dir() and p.name.startswith("ckpt_") and (p / "checkpoint.pt").exists() and (p / "CHECKPOINT.json").exists():
            cands.append((int(p.name.split("_")[1]), p))
    return max(cands)[1] if cands else None


def load_checkpoint(path: Path, map_location="cpu", verify: bool = True) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "checkpoint.pt"
    if verify:
        meta = json.loads((path.parent / "CHECKPOINT.json").read_text(encoding="utf-8"))
        actual = _sha256(path)
        if actual != meta["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch for {path}: {actual} != {meta['sha256']}")
    return torch.load(path, map_location=map_location, weights_only=False)
