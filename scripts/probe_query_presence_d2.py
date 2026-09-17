"""Run frozen D2 query-presence on a P1 checkpoint.

Usage:
    python -B scripts/probe_query_presence_d2.py --arm treatment
    python -B scripts/probe_query_presence_d2.py --arm control
    python -B scripts/probe_query_presence_d2.py --adjudicate

No optimizer. D1/D1b/P1 receipts are not rewritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.baby_v010.query_presence import assert_query_only_twins  # noqa: E402
from src.baby_v010.query_presence_d2 import (  # noqa: E402
    ARMS,
    D1B_DIAGNOSTIC,
    D1B_DIAGNOSTIC_SHA,
    D1B_MANIFEST,
    OUT,
    PROTOCOL,
    adjudicate_d2,
    assert_priors,
    write_manifest,
)
from src.baby_v010.selection_s1 import PARENT_SHA, digest, write  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS))
    parser.add_argument("--adjudicate", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    if args.manifest:
        assert_priors()
        payload = write_manifest()
        print(json.dumps({"protocol_sha256": payload["protocol_sha256"]}), flush=True)
        return

    if args.adjudicate:
        treat = json.loads((OUT / "treatment" / "QUERY_PRESENCE.json").read_text(encoding="utf-8"))
        control = json.loads((OUT / "control" / "QUERY_PRESENCE.json").read_text(encoding="utf-8"))
        decision = adjudicate_d2(treat, control)
        write(OUT / "ADJUDICATION.json", decision)
        print(json.dumps({"headline": decision["headline"], "treatment": decision["treatment"]["verdict"], "control": decision["control"]["verdict"]}), flush=True)
        return

    if not args.arm:
        raise SystemExit("--arm or --adjudicate required")

    from src.baby_v010.query_presence_trace import (
        load_parent,
        patch_directed_pairs,
        sdpa_reference_max_abs,
        summarize,
        trace_items,
        twin_identity_rows,
        unembed_rank,
    )
    from src.baby_v010.query_presence_d2 import arm_decision

    assert_priors()
    spec = ARMS[args.arm]
    dest = OUT / args.arm
    if (dest / "QUERY_PRESENCE.json").exists():
        raise RuntimeError("refusing to overwrite " + str(dest))

    items = json.loads(D1B_DIAGNOSTIC.read_text(encoding="utf-8"))
    twins_ok = True
    try:
        assert_query_only_twins(items)
    except RuntimeError:
        twins_ok = False
        raise

    d1b_manifest = json.loads(D1B_MANIFEST.read_text(encoding="utf-8"))
    diagnostic_sha = digest(D1B_DIAGNOSTIC)
    ckpt_sha = digest(spec["ckpt"])
    model, blob = load_parent(spec["ckpt"], device=args.device)
    smoke = items[:8]
    sdpa_max = sdpa_reference_max_abs(model, [row["input"] for row in smoke], args.device)
    rank = unembed_rank(model)

    preflight = {
        "arm": args.arm,
        "checkpoint_sha_match": ckpt_sha == spec["sha"],
        "checkpoint_sha256": ckpt_sha,
        "parent_sha256": PARENT_SHA,
        "diagnostic_sha_match": diagnostic_sha == D1B_DIAGNOSTIC_SHA
        and diagnostic_sha == d1b_manifest["files"]["runs/query_presence_d1b/DIAGNOSTIC.json"],
        "diagnostic_sha256": diagnostic_sha,
        "protected_material_opened": bool(blob.get("protected_material_opened")),
        "twins_query_only": twins_ok,
        "sdpa_reference_max_abs": sdpa_max,
        "unembed": rank,
        "protocol_sha256": digest(PROTOCOL),
        "update": int(blob["update"]),
        "update_16800": int(blob["update"]) == 16800,
        "d1_reopened": False,
        "d1b_reopened": False,
    }
    print(json.dumps({"preflight": {k: v for k, v in preflight.items() if k != "unembed"}}), flush=True)
    if not preflight["checkpoint_sha_match"] or not preflight["diagnostic_sha_match"]:
        raise RuntimeError("refusing to score the wrong checkpoint or diagnostic")
    if not preflight["update_16800"]:
        raise RuntimeError("refusing a non-terminal P1 checkpoint")

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
    report = {
        "protocol": "V010_QUERY_PRESENCE_D2",
        "arm": args.arm,
        "trained": False,
        "gates_changed": False,
        "frozen_panels_mutated": False,
        "protected_material_opened": False,
        "parent_sha256": PARENT_SHA,
        "checkpoint_sha256": ckpt_sha,
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
    decision = arm_decision(report)
    dest.mkdir(parents=True, exist_ok=True)
    write(dest / "QUERY_PRESENCE.json", report)
    write(dest / "ADJUDICATION.json", decision)
    write(dest / "ROWS_identity.json", identity)
    write(dest / "ROWS_patch.json", patches)
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
            }
        )
    write(dest / "ROWS_attention.json", attn_rows)
    print(json.dumps({"arm": args.arm, "adjudication": decision["verdict"], "valid": decision["valid"]}), flush=True)


if __name__ == "__main__":
    main()
