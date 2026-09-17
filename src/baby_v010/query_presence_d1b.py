"""D1b diagnostic generation: fresh all-K twins, denied against prior eval sets.

Read-only parent. No optimizer. Does not rewrite D1, S1, or S2 artifacts.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .selection_s1 import PANEL, PARENT, PARENT_SHA, digest, write
from .selection_s2 import (
    S1_DIAGNOSTIC,
    denied_from_panels,
    keyed_group,
    load_s1_denials,
)
from .v2r4_provenance import FROZEN_PANELS_SHA256, require_frozen_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/query_presence_d1b"
PROTOCOL = ROOT / "design/V010_QUERY_PRESENCE_D1B.md"
S2_DIAGNOSTIC = ROOT / "runs/selection_s2/DIAGNOSTIC.json"
D1_SUMMARY = ROOT / "runs/query_presence_d1/QUERY_PRESENCE.json"
D1_ADJUDICATION = ROOT / "runs/query_presence_d1/ADJUDICATION.json"
DIAGNOSTIC_SEED = 140200
BODIES_PER_K = 48


def posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def load_s2_denials(denied_inputs: set[tuple], denied_spans: set[tuple]) -> None:
    from .isolation_transforms import parse_records

    if not S2_DIAGNOSTIC.exists():
        raise RuntimeError("S2 diagnostic missing; refusing to generate overlapping D1b data")
    for row in json.loads(S2_DIAGNOSTIC.read_text(encoding="utf-8")):
        denied_inputs.add(tuple(row["input"]))
        denied_spans.add(tuple(row["target_span"]))
        if row.get("kind") == "keyed":
            denied_spans.update(tuple(value) for _, value in parse_records(row))


def assert_d1_stayed_invalid() -> dict:
    if not D1_ADJUDICATION.exists():
        raise RuntimeError("D1 adjudication missing; D1b is a successor, not a replacement")
    adj = json.loads(D1_ADJUDICATION.read_text(encoding="utf-8"))
    if adj.get("protocol") != "V010_QUERY_PRESENCE_D1":
        raise RuntimeError("D1 adjudication protocol mismatch")
    if adj.get("verdict") != "INVALID":
        raise RuntimeError("D1b refuses to generate unless D1 stayed INVALID")
    return adj


def generate() -> None:
    require_frozen_file(PANEL, FROZEN_PANELS_SHA256, "panels")
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    if not PROTOCOL.exists():
        raise RuntimeError("D1b protocol missing")
    if not D1_SUMMARY.exists():
        raise RuntimeError("D1 summary missing; D1b is a successor, not a replacement")
    if (OUT / "MANIFEST.json").exists():
        raise RuntimeError("D1b already frozen")
    if not S1_DIAGNOSTIC.exists():
        raise RuntimeError("S1 diagnostic missing")
    d1_adj = assert_d1_stayed_invalid()

    from .data import LANG_TRAIN, build_banks, read_u16

    banks = build_banks(read_u16(LANG_TRAIN))
    frozen = json.loads(PANEL.read_text(encoding="utf-8"))
    denied_inputs, denied_spans = denied_from_panels(frozen)
    load_s1_denials(denied_inputs, denied_spans)
    load_s2_denials(denied_inputs, denied_spans)

    rng = random.Random(DIAGNOSTIC_SEED)
    dev = []
    for k in (2, 3, 4):
        for count in range(BODIES_PER_K):
            body_id = f"D1B_K{k}_{count}"
            group, pairs = keyed_group(rng, banks, denied_inputs, denied_spans, k, body_id)
            for row in group:
                row["protocol"] = "V010_QUERY_PRESENCE_D1B"
            dev.extend(group)
            denied_spans.update(tuple(value) for _, value in pairs)
            denied_inputs.update(tuple(row["input"]) for row in group)

    write(OUT / "DIAGNOSTIC.json", dev)
    files = [
        PROTOCOL,
        Path(__file__),
        PANEL,
        OUT / "DIAGNOSTIC.json",
        S1_DIAGNOSTIC,
        S2_DIAGNOSTIC,
        D1_ADJUDICATION,
        ROOT / "src/baby_v010/query_presence.py",
        ROOT / "src/baby_v010/query_presence_trace.py",
        ROOT / "scripts/probe_query_presence.py",
        ROOT / "src/baby_v010/data_v2.py",
        ROOT / "src/baby_v010/selection_s1.py",
        ROOT / "src/baby_v010/selection_s2.py",
    ]
    ks = {2: 0, 3: 0, 4: 0}
    for row in dev:
        ks[int(row["pair_count"])] += 1
    write(
        OUT / "MANIFEST.json",
        {
            "protocol": "V010_QUERY_PRESENCE_D1B",
            "parent_sha256": PARENT_SHA,
            "diagnostic_seed": DIAGNOSTIC_SEED,
            "protocol_sha256": digest(PROTOCOL),
            "files": {posix(path): digest(path) for path in files},
            "diagnostic_rows": len(dev),
            "diagnostic_bodies": BODIES_PER_K * 3,
            "keyed_row_count": {str(k): ks[k] for k in (2, 3, 4)},
            "denied_prior_eval_sets": [
                "panels",
                "selection_s1",
                "selection_s2",
                "d1_scored_inputs=selection_s2",
            ],
            "d1_verdict": d1_adj["verdict"],
            "d1_reopened": False,
            "protected_material_opened": False,
            "trained": False,
        },
    )
    print(
        json.dumps(
            {
                "bodies": BODIES_PER_K * 3,
                "rows": len(dev),
                "protocol_sha256": digest(PROTOCOL),
                "diagnostic_sha256": digest(OUT / "DIAGNOSTIC.json"),
            }
        ),
        flush=True,
    )


def verify() -> dict:
    m = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    for path, expected in m["files"].items():
        if digest(ROOT / path) != expected:
            raise RuntimeError("hash mismatch " + path)
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    if digest(PROTOCOL) != m["protocol_sha256"]:
        raise RuntimeError("protocol hash mismatch")
    if m.get("d1_reopened"):
        raise RuntimeError("D1 was reopened")
    return m


if __name__ == "__main__":
    import sys

    cmd = sys.argv[-1] if len(sys.argv) > 1 else "generate"
    if cmd == "verify":
        print(json.dumps({"ok": True, "protocol_sha256": verify()["protocol_sha256"]}), flush=True)
    else:
        generate()
