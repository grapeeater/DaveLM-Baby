"""Verify frozen S2 identity and a tiny parent batch. No optimizer steps."""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from src.baby_v010 import selection_s2 as s


def main() -> None:
    s.verify()
    if s.digest(s.PARENT) != s.PARENT_SHA:
        raise RuntimeError("parent mismatch")
    dev = json.loads((s.OUT / "DIAGNOSTIC.json").read_text(encoding="utf-8"))
    by_body: dict[str, list] = defaultdict(list)
    for row in dev:
        by_body[row["body_id"]].append(row)
    if len(by_body) != 144:
        raise RuntimeError("expected 144 diagnostic bodies")
    for group in by_body.values():
        k = group[0]["pair_count"]
        if len(group) != k:
            raise RuntimeError("diagnostic group is not all-K")
        for left, right in zip(group, group[1:]):
            diffs = [i for i, (a, b) in enumerate(zip(left["input"], right["input"])) if a != b]
            if diffs != [left["query_position"]]:
                raise RuntimeError("diagnostic twins changed more than the query")
            if left["target"][0] == right["target"][0]:
                raise RuntimeError("counterfactual gold heads are not distinct")
    schedule = json.loads((s.OUT / "SCHEDULE_120001.json").read_text(encoding="utf-8"))
    for batch in schedule:
        if batch["task"] == "language":
            continue
        items = batch["items"]
        if len(items) != 16:
            raise RuntimeError("batch size")
        keyed = [row for row in items if row["kind"] == "keyed"]
        if len(keyed) != 12:
            raise RuntimeError("keyed count")
        groups: dict[str, list] = defaultdict(list)
        for row in keyed:
            groups[row["body_id"]].append(row)
        k = batch["k"]
        if any(len(group) != k for group in groups.values()):
            raise RuntimeError("schedule is not all-K packed")
        if any(row["pair_count"] != k for row in keyed):
            raise RuntimeError("mixed K in a supposedly homogeneous batch")
    model, _ = s.model_load()
    model.eval()
    items = next(spec["items"] for spec in schedule if spec["task"] == "structured")
    x, y, mask, first = s.pack(items, torch.device("cuda"))
    began = time.monotonic()
    with torch.no_grad():
        z = model(x)
        ce = torch.nn.functional.cross_entropy(z[mask], y[mask])
        extra = s.keyed_contrast_loss(z, items, first)
        assert torch.isfinite(ce) and torch.isfinite(extra)
        n = len(items[0]["input"]) - 1
        solo = model(x[:1, : n + 1])[0, -1]
        maxdiff = (solo - z[0, n]).abs().max().item()
        if maxdiff >= 1e-3:
            raise RuntimeError("padding moved logits")
    report = {
        "status": "PASS",
        "optimizer_updates": 0,
        "parent_sha256": s.digest(s.PARENT),
        "manifest_sha256": s.digest(s.OUT / "MANIFEST.json"),
        "contrast_margin": s.CONTRAST_MARGIN,
        "counterfactual_bodies": len(by_body),
        "loss_finite": True,
        "padding_max_logit_difference": maxdiff,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "smoke_seconds": time.monotonic() - began,
        "protected_material_opened": False,
        "s1_data_mutated": False,
    }
    s.write(s.OUT / "PREFLIGHT.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
