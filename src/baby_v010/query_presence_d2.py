"""D2: read-only query-presence on P1 treatment vs control checkpoints.

Does not rewrite D1, D1b, or P1 receipts. No optimizer.
"""
from __future__ import annotations

import json
from pathlib import Path

from .query_presence import adjudicate_d1b
from .selection_s1 import PARENT, PARENT_SHA, digest, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/query_presence_d2"
PROTOCOL = ROOT / "design/V010_QUERY_PRESENCE_D2.md"
D1B_DIAGNOSTIC = ROOT / "runs/query_presence_d1b/DIAGNOSTIC.json"
D1B_MANIFEST = ROOT / "runs/query_presence_d1b/MANIFEST.json"
D1B_ADJUDICATION = ROOT / "runs/query_presence_d1b/ADJUDICATION.json"
P1_TREAT_RECEIPT = ROOT / "runs/selection_p1/treatment_150001/RECEIPT_0800.json"
P1_CTRL_RECEIPT = ROOT / "runs/selection_p1/control_150001/RECEIPT_0800.json"
TREAT_CKPT = ROOT / "runs/selection_p1/treatment_150001/checkpoint_16800.pt"
CTRL_CKPT = ROOT / "runs/selection_p1/control_150001/checkpoint_16800.pt"
TREAT_SHA = "5c8296ef759543dd608c18b8048ccce4bd4fcf8de7a6183b2d84e5f53d4fbafa"
CTRL_SHA = "3f005e19da8db071a763eb66ef0cb6f7e59ca63cf55078eeec9fad2338cd2fc1"
D1B_DIAGNOSTIC_SHA = "b7260e6761bbedecacc6f26801b54141d6539b9ecfe76ff72c34c2e32ce76c57"

ARMS = {
    "treatment": {"ckpt": TREAT_CKPT, "sha": TREAT_SHA, "receipt": P1_TREAT_RECEIPT},
    "control": {"ckpt": CTRL_CKPT, "sha": CTRL_SHA, "receipt": P1_CTRL_RECEIPT},
}


def posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def assert_priors() -> None:
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("authoritative parent mismatch")
    d1b = json.loads(D1B_ADJUDICATION.read_text(encoding="utf-8"))
    if d1b.get("protocol") != "V010_QUERY_PRESENCE_D1B" or d1b.get("verdict") != "B":
        raise RuntimeError("D2 requires D1b to have stayed B")
    if digest(D1B_DIAGNOSTIC) != D1B_DIAGNOSTIC_SHA:
        raise RuntimeError("D1b diagnostic hash mismatch")
    for arm, spec in ARMS.items():
        receipt = json.loads(spec["receipt"].read_text(encoding="utf-8"))
        expected = receipt["artifacts"]["checkpoint_16800.pt"]
        if expected != spec["sha"]:
            raise RuntimeError(f"{arm} receipt SHA drifted from protocol")
        if digest(spec["ckpt"]) != spec["sha"]:
            raise RuntimeError(f"{arm} checkpoint SHA mismatch")


def arm_decision(report: dict) -> dict:
    """D1b bars, with checkpoint SHA standing in for parent SHA."""
    patched = dict(report)
    preflight = dict(report.get("preflight", {}))
    preflight["parent_sha_match"] = bool(preflight.get("checkpoint_sha_match"))
    patched["preflight"] = preflight
    decision = adjudicate_d1b(patched)
    decision["protocol"] = "V010_QUERY_PRESENCE_D2"
    decision["checkpoint_sha256"] = preflight.get("checkpoint_sha256")
    decision["arm"] = preflight.get("arm")
    decision["authoritative_parent_unchanged"] = True
    decision["d1_reopened"] = False
    decision["d1b_reopened"] = False
    return decision


def headline(treat: dict, control: dict) -> str:
    if not treat["valid"] or not control["valid"]:
        return "INVALID"
    t, c = treat["verdict"], control["verdict"]
    if t == "A":
        return "RESIDUAL_WRITE"
    if t == "COMPOSITION" and c in {"B", "COMPOSITION"}:
        return "WEIGHTS_ONLY"
    if t == "B" and c == "B":
        return "STILL_B"
    return "MIXED"


def adjudicate_d2(treatment_report: dict, control_report: dict) -> dict:
    treat = arm_decision(treatment_report)
    control = arm_decision(control_report)
    long_t = treatment_report["long_gap"]
    long_c = control_report["long_gap"]
    return {
        "protocol": "V010_QUERY_PRESENCE_D2",
        "headline": headline(treat, control),
        "treatment": treat,
        "control": control,
        "long_gap": {
            "treatment_residual_cosine": long_t["median_final_residual_cosine"],
            "control_residual_cosine": long_c["median_final_residual_cosine"],
            "treatment_query_track": long_t["fraction_rows_any_query_tracking_head"],
            "control_query_track": long_c["fraction_rows_any_query_tracking_head"],
            "treatment_flip_final": long_t["flip_toward_donor_final"],
            "control_flip_final": long_c["flip_toward_donor_final"],
        },
        "hypothesis": {
            "H_WEIGHTS": headline(treat, control) == "WEIGHTS_ONLY",
            "H_RESIDUAL": headline(treat, control) == "RESIDUAL_WRITE",
            "H_STILL_B": headline(treat, control) == "STILL_B",
        },
        "trained": False,
        "gates_changed": False,
        "d1_reopened": False,
        "d1b_reopened": False,
        "p1_reopened": False,
        "protected_material_opened": False,
        "authoritative_parent_unchanged": True,
    }


def write_manifest() -> dict:
    files = [
        PROTOCOL,
        Path(__file__),
        D1B_DIAGNOSTIC,
        D1B_ADJUDICATION,
        P1_TREAT_RECEIPT,
        P1_CTRL_RECEIPT,
        ROOT / "src/baby_v010/query_presence.py",
        ROOT / "src/baby_v010/query_presence_trace.py",
        ROOT / "scripts/probe_query_presence_d2.py",
    ]
    payload = {
        "protocol": "V010_QUERY_PRESENCE_D2",
        "parent_sha256": PARENT_SHA,
        "protocol_sha256": digest(PROTOCOL),
        "diagnostic_sha256": D1B_DIAGNOSTIC_SHA,
        "treatment_checkpoint_sha256": TREAT_SHA,
        "control_checkpoint_sha256": CTRL_SHA,
        "files": {posix(path): digest(path) for path in files},
        "d1_reopened": False,
        "d1b_reopened": False,
        "trained": False,
        "protected_material_opened": False,
    }
    write(OUT / "MANIFEST.json", payload)
    return payload
