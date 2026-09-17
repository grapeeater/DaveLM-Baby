"""Run frozen D1 query-presence localization on the hashed parent.

Usage:
    python -B scripts/probe_query_presence.py --out runs/query_presence_d1

No optimizer is constructed. Frozen S2 artifacts are read, never rewritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.baby_v010.query_presence import adjudicate, assert_query_only_twins  # noqa: E402
from src.baby_v010.selection_s1 import PARENT, PARENT_SHA, digest  # noqa: E402

DIAGNOSTIC = ROOT / "runs/selection_s2/DIAGNOSTIC.json"
S2_MANIFEST = ROOT / "runs/selection_s2/MANIFEST.json"
PROTOCOL = ROOT / "design/V010_QUERY_PRESENCE_D1.md"


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/query_presence_d1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    from src.baby_v010.query_presence_trace import (
        load_parent,
        patch_directed_pairs,
        sdpa_reference_max_abs,
        summarize,
        trace_items,
        twin_identity_rows,
        unembed_rank,
    )

    parent_sha = digest(PARENT)
    manifest = json.loads(S2_MANIFEST.read_text(encoding="utf-8"))
    diagnostic_sha = digest(DIAGNOSTIC)
    items = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    twins_ok = True
    try:
        assert_query_only_twins(items)
    except RuntimeError:
        twins_ok = False
        raise

    model, blob = load_parent(PARENT, device=args.device)
    smoke = items[:8]
    sdpa_max = sdpa_reference_max_abs(model, [row["input"] for row in smoke], args.device)
    rank = unembed_rank(model)

    preflight = {
        "parent_sha_match": parent_sha == PARENT_SHA,
        "parent_sha256": parent_sha,
        "diagnostic_sha_match": diagnostic_sha
        == manifest["files"]["runs/selection_s2/DIAGNOSTIC.json"],
        "diagnostic_sha256": diagnostic_sha,
        "protected_material_opened": bool(blob.get("protected_material_opened")),
        "twins_query_only": twins_ok,
        "sdpa_reference_max_abs": sdpa_max,
        "unembed": rank,
        "protocol_sha256": digest(PROTOCOL),
        "update": int(blob["update"]),
    }
    print(json.dumps({"preflight": {k: v for k, v in preflight.items() if k != "unembed"}}), flush=True)
    if not preflight["parent_sha_match"] or not preflight["diagnostic_sha_match"]:
        raise RuntimeError("refusing to score the wrong parent or diagnostic")

    traces = trace_items(model, items, args.device, batch=args.batch)
    print(json.dumps({"traced": len(traces)}), flush=True)
    identity = twin_identity_rows(traces)
    patches = patch_directed_pairs(
        model,
        traces,
        args.device,
        progress=lambda msg: print(json.dumps({"progress": msg}), flush=True),
    )
    summary = summarize(traces, identity, patches)
    summary["preflight"] = preflight
    report = {
        "protocol": "V010_QUERY_PRESENCE_D1",
        "trained": False,
        "gates_changed": False,
        "frozen_panels_mutated": False,
        "protected_material_opened": False,
        "parent_sha256": PARENT_SHA,
        "attention_backend": "reference",
        "preflight": preflight,
        "positive_control": summary["positive_control"],
        "long_gap": summary["long_gap"],
        "identity": summary["identity"],
        "attention": summary["attention"],
        "n_items": summary["n_items"],
        "n_identity_pairs": summary["n_identity_pairs"],
        "n_patch_pairs": summary["n_patch_pairs"],
    }
    decision = adjudicate(report)
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / "QUERY_PRESENCE.json", report)
    write(args.out / "ADJUDICATION.json", decision)
    write(args.out / "ROWS_identity.json", identity)
    write(args.out / "ROWS_patch.json", patches)
    attn_rows = []
    for row in traces:
        item = row["item"]
        attn_rows.append(
            {
                "body_id": item["body_id"],
                "query_index": int(item["query_index"]),
                "gap": row["gap"],
                "loc": row["loc"],
                "any_query_tracking_head": row["any_query_tracking_head"],
                "any_prev_tracking_head": row["any_prev_tracking_head"],
                "embed_cos_query": row["embed_cos_query"],
                "embed_cos_last": row["embed_cos_last"],
                "layers": [
                    {
                        "max_query": layer["max_query"],
                        "max_prev": layer["max_prev"],
                        "any_tracks_query": layer["any_tracks_query"],
                        "any_tracks_prev": layer["any_tracks_prev"],
                        "uniform": layer["uniform"],
                    }
                    for layer in row["attn_mass"]
                ],
            }
        )
    write(args.out / "ROWS_attention.json", attn_rows)
    print(json.dumps({"adjudication": decision["verdict"], "valid": decision["valid"]}), flush=True)


if __name__ == "__main__":
    main()
