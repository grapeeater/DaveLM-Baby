"""Run the read-only query-locality causal probe on hashed checkpoints.

Usage:
    python -B scripts/probe_query_locality.py --out runs/query_locality

No optimizer is constructed and no checkpoint is written. Frozen S1/S2 artifacts
are read, never rewritten. Protected material is not opened.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.baby_v010.query_locality import (  # noqa: E402
    ARMS,
    bucket,
    build_arm,
    candidate_rank,
    gap_of,
)
from src.baby_v010.selection_s1 import PARENT, PARENT_SHA, digest  # noqa: E402

DIAGNOSTIC = ROOT / "runs/selection_s2/DIAGNOSTIC.json"
S2_MANIFEST = ROOT / "runs/selection_s2/MANIFEST.json"
ARM_SEED = 130001

CHECKPOINTS = {
    "parent_u16000": PARENT,
    "s1_treatment_u16400": ROOT / "runs/selection_s1/treatment_110001/checkpoint_16400.pt",
    "s2_treatment_u16400": ROOT / "runs/selection_s2/treatment_120001/checkpoint_16400.pt",
    "s2_control_u16400": ROOT / "runs/selection_s2/control_120001/checkpoint_16400.pt",
    # Local-only, undocumented v2R5/v2R6 terminals. Diagnostic contrast only:
    # these are NOT candidates to replace the authoritative parent here.
    "v2r5_s107001_u16000": ROOT / "runs/structured_v2r5_seed107001_from6000/checkpoint_16000.pt",
    "v2r5_s107002_u16000": ROOT / "runs/structured_v2r5_seed107002_from6000/checkpoint_16000.pt",
    "v2r6_s108001_u16000": ROOT / "runs/structured_v2r6_seed108001_from6000/checkpoint_16000.pt",
    "v2r6_s108002_u16000": ROOT / "runs/structured_v2r6_seed108002_from6000/checkpoint_16000.pt",
    "t1_treatment_u16400": ROOT / "runs/selection_t1/treatment_130001/checkpoint_16400.pt",
    "t1_control_u16400": ROOT / "runs/selection_t1/control_130001/checkpoint_16400.pt",
}


def load(path: Path):
    import torch

    from src.baby_v010.config import BabyVNextConfig
    from src.baby_v010.model import BabyVNextLM

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("protected_material_opened"):
        raise RuntimeError("refusing a checkpoint that recorded protected access")
    config = BabyVNextConfig.from_dict(blob["config"])
    model = BabyVNextLM(config).cuda()
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    return model, int(blob["update"])


def score(model, requests: list[dict], batch: int = 16) -> list[dict]:
    import torch

    out: list[dict] = []
    with torch.no_grad():
        for start in range(0, len(requests), batch):
            chunk = requests[start : start + batch]
            width = max(len(row["input"]) for row in chunk)
            x = torch.zeros((len(chunk), width), dtype=torch.long, device="cuda")
            for i, row in enumerate(chunk):
                x[i, : len(row["input"])] = torch.tensor(row["input"], device="cuda")
            logits = model(x)
            for i, row in enumerate(chunk):
                z = logits[i, len(row["input"]) - 1]
                heads = row["candidate_heads"]
                values = z[heads].tolist()
                full = z.tolist()
                gold_token = heads[row["gold_index"]]
                out.append(
                    {
                        **{k: v for k, v in row.items() if k != "input"},
                        "candidate_logits": values,
                        "candidate_argmax": int(
                            max(range(len(values)), key=lambda j: values[j])
                        ),
                        "candidate_rank": candidate_rank(values, row["gold_index"]),
                        "vocab_rank": 1 + sum(1 for v in full if v > full[gold_token]),
                        "margin_vs_best_other": values[row["gold_index"]]
                        - max(
                            v
                            for j, v in enumerate(values)
                            if j != row["gold_index"]
                        ),
                    }
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/query_locality")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="score only these checkpoint labels (default: all)",
    )
    args = parser.parse_args()
    selected = (
        CHECKPOINTS
        if not args.only
        else {k: v for k, v in CHECKPOINTS.items() if k in set(args.only)}
    )

    from src.baby_v010.data import LANG_TRAIN, build_banks, read_u16
    from src.baby_v010.data_v2 import _reserved_surface_tokens

    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent checkpoint SHA mismatch")
    manifest = json.loads(S2_MANIFEST.read_text(encoding="utf-8"))
    if digest(DIAGNOSTIC) != manifest["files"]["runs/selection_s2/DIAGNOSTIC.json"]:
        raise RuntimeError("S2 diagnostic hash mismatch")

    banks = build_banks(read_u16(LANG_TRAIN))
    reserved = _reserved_surface_tokens(banks)
    key_pool = [t for t in banks.key if t not in reserved]
    span_pool = list(banks.span)

    items = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    requests: list[dict] = []
    for serial, item in enumerate(items):
        base_gap = gap_of(item)
        for arm in ARMS:
            built = build_arm(
                item,
                arm,
                key_pool=key_pool,
                span_pool=span_pool,
                seed=ARM_SEED + serial,
            )
            if built is None:
                continue
            requests.append(
                {
                    "input": built["input"],
                    "gold_index": built["gold_index"],
                    "arm": arm,
                    "body_id": item["body_id"],
                    "query_index": int(item["query_index"]),
                    "K": int(item["pair_count"]),
                    "variant": item["variant"],
                    "value_length": int(item["value_length"]),
                    "candidate_heads": [int(h) for h in item["candidate_heads"]],
                    "base_gap": base_gap,
                    "base_bucket": bucket(base_gap),
                    "arm_gap": len(built["input"]) - 1 - int(item["query_position"]),
                }
            )

    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "protocol": "V010_QUERY_LOCALITY_PROBE",
        "trained": False,
        "gates_changed": False,
        "frozen_panels_mutated": False,
        "protected_material_opened": False,
        "parent_sha256": PARENT_SHA,
        "diagnostic_rows": len(items),
        "arm_seed": ARM_SEED,
        "checkpoints": {},
    }

    for name, path in selected.items():
        if not path.exists():
            report["checkpoints"][name] = {"status": "missing", "path": str(path)}
            continue
        model, update = load(path)
        rows = score(model, requests)
        del model
        import torch

        torch.cuda.empty_cache()

        summary: dict = {}
        cells = defaultdict(lambda: {"hit": 0, "n": 0, "chance": 0.0, "rank": []})
        for row in rows:
            for key in (
                ("arm", row["arm"]),
                ("arm_x_bucket", f"{row['arm']}|{row['base_bucket']}"),
                ("arm_x_K", f"{row['arm']}|K{row['K']}"),
                ("arm_x_variant", f"{row['arm']}|{row['variant']}"),
            ):
                cell = cells[key]
                cell["hit"] += int(row["candidate_argmax"] == row["gold_index"])
                cell["n"] += 1
                cell["chance"] += 1.0 / row["K"]
                cell["rank"].append(row["candidate_rank"])
        for (group, label), cell in cells.items():
            summary.setdefault(group, {})[label] = {
                "hit": cell["hit"],
                "n": cell["n"],
                "accuracy": cell["hit"] / cell["n"],
                "chance": cell["chance"] / cell["n"],
                "excess": cell["hit"] / cell["n"] - cell["chance"] / cell["n"],
                "median_candidate_rank": statistics.median(cell["rank"]),
            }
        report["checkpoints"][name] = {
            "status": "scored",
            "path": str(path.relative_to(ROOT)),
            "sha256": digest(path),
            "update": update,
            "summary": summary,
        }
        (args.out / f"ROWS_{name}.json").write_text(
            json.dumps(rows) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "checkpoint": name,
                    "arm": {
                        k: round(v["accuracy"], 4)
                        for k, v in sorted(summary["arm"].items())
                    },
                },
            ),
            flush=True,
        )

    (args.out / "QUERY_LOCALITY.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("wrote", args.out / "QUERY_LOCALITY.json")


if __name__ == "__main__":
    main()
