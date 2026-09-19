"""Adjudicate T1 against the frozen rules in design/V010_SELECTION_REPAIR_T1_GAP.md.

Thresholds are read from this file only; they are the ones committed in
8892545 before data generation and before any optimizer update.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "runs/selection_t1"
PROBE = ROOT / "runs/query_locality_t1/QUERY_LOCALITY.json"
BOOTSTRAP_SEED = 130300

# Frozen decision thresholds.
SUCCESS_EXCESS = 0.10
SUCCESS_VS_CONTROL = 0.07
PARTIAL_EXCESS = 0.05
PARTIAL_VS_CONTROL = 0.04
NULL_EXCESS = 0.02
FALSIFIED_SHORT_RISE = 0.10
FUTILITY_GAIN = 0.02
REGRESSION_DROP = 0.05

RETENTION = {
    "primitive_induction": "first_top1",
    "primitive_keyed": "first_top1",
    "short_keyed": "free_exact",
}
NEGATIVE_CONTROLS = {
    "broken_context": ("free_exact", 0.05),
    "broken_order": ("first_top1", 0.20 * 0.4479),
}


def load(arm: str, step: int) -> dict:
    return json.loads((OUT / f"{arm}_130001" / f"eval_{step:04d}.json").read_text(encoding="utf-8"))


def long_rows(report: dict, diagnostic: dict) -> dict:
    """Per-body hit/chance on the primary (gap >= 13) stratum."""
    bodies = defaultdict(list)
    for row in report["rows"]:
        item = diagnostic.get((row["body_id"], row["query_index"]))
        if item is None:
            continue
        if len(item["input"]) - 1 - int(item["query_position"]) < 13:
            continue
        bodies[row["body_id"]].append(
            (int(bool(row["inventory_correct"])), 1.0 / row["K"])
        )
    return bodies


def bootstrap(treatment: dict, control: dict, draws: int = 10000) -> tuple[float, float]:
    keys = sorted(set(treatment) & set(control))
    rng = random.Random(BOOTSTRAP_SEED)
    deltas = []
    for _ in range(draws):
        picked = [keys[rng.randrange(len(keys))] for _ in keys]
        t_hit = t_n = c_hit = c_n = 0
        for key in picked:
            for hit, _ch in treatment[key]:
                t_hit += hit
                t_n += 1
            for hit, _ch in control[key]:
                c_hit += hit
                c_n += 1
        deltas.append(t_hit / max(1, t_n) - c_hit / max(1, c_n))
    deltas.sort()
    return deltas[int(0.025 * draws)], deltas[int(0.975 * draws)]


def main() -> None:
    diagnostic = {
        (r["body_id"], r["query_index"]): r
        for r in json.loads(
            (ROOT / "runs/selection_s2/DIAGNOSTIC.json").read_text(encoding="utf-8")
        )
    }
    base = load("treatment", 0)
    treat = load("treatment", 400)
    ctrl = load("control", 400)

    base_excess = base["gap_strata"]["long"]["excess"]
    t_excess = treat["gap_strata"]["long"]["excess"]
    c_excess = ctrl["gap_strata"]["long"]["excess"]
    t_short = treat["gap_strata"]["a_gap_0_1"]["excess"]
    base_short = base["gap_strata"]["a_gap_0_1"]["excess"]

    low, high = bootstrap(long_rows(treat, diagnostic), long_rows(ctrl, diagnostic))
    vs_control = treat["gap_strata"]["long"]["accuracy"] - ctrl["gap_strata"]["long"]["accuracy"]

    regressions = []
    for panel, metric in RETENTION.items():
        before = base["frozen"]["summaries"][panel][metric]
        after = treat["frozen"]["summaries"][panel][metric]
        if before - after > REGRESSION_DROP:
            regressions.append(
                {"panel": panel, "metric": metric, "parent": before, "treatment": after}
            )
    ce_delta = treat["language_dev_ce"] - base["language_dev_ce"]
    if ce_delta > REGRESSION_DROP:
        regressions.append({"panel": "language_dev_ce", "parent": base["language_dev_ce"], "treatment": treat["language_dev_ce"]})
    rest_drop = base["summary"]["rest_lock"] - treat["summary"]["rest_lock"]
    if rest_drop > REGRESSION_DROP:
        regressions.append({"panel": "rest_lock", "parent": base["summary"]["rest_lock"], "treatment": treat["summary"]["rest_lock"]})

    controls_ok = {}
    for panel, (metric, bar) in NEGATIVE_CONTROLS.items():
        value = treat["frozen"]["summaries"][panel][metric]
        controls_ok[f"{panel}.{metric}"] = {"value": value, "bar": bar, "pass": value <= bar}
    probe = json.loads(PROBE.read_text(encoding="utf-8"))["checkpoints"]
    t_arms = probe["t1_treatment_u16400"]["summary"]["arm"]
    controls_ok["append_unused_key_excess"] = {
        "value": t_arms["append_unused_key"]["excess"],
        "bar": 0.05,
        "pass": t_arms["append_unused_key"]["excess"] <= 0.05,
    }
    controls_ok["append_other_key_not_collapsed"] = {
        "value": t_arms["append_other_key"]["accuracy"],
        "bar": 0.40,
        "pass": t_arms["append_other_key"]["accuracy"] >= 0.40,
    }

    if t_excess >= SUCCESS_EXCESS and vs_control >= SUCCESS_VS_CONTROL and low > 0:
        primary = "SUCCESS"
    elif t_excess >= PARTIAL_EXCESS and vs_control >= PARTIAL_VS_CONTROL and low > 0:
        primary = "MECHANISM_SUPPORTED_DOSE_INSUFFICIENT"
    elif t_excess < NULL_EXCESS and (t_short - base_short) >= FALSIFIED_SHORT_RISE:
        primary = "FALSIFIED"
    elif t_excess < NULL_EXCESS:
        primary = "NULL"
    else:
        primary = "INDETERMINATE"

    verdict = "REGRESSION" if regressions else primary

    report = {
        "protocol": "V010_SELECTION_REPAIR_T1_GAP",
        "protocol_freeze_commit": "8892545",
        "parent_sha256": base.get("parent_sha256", "94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827"),
        "futility_fired_at_400": (t_excess - base_excess) < FUTILITY_GAIN,
        "primary_endpoint": {
            "stratum": "frozen diagnostic rows with query->generation gap >= 13",
            "n": treat["gap_strata"]["long"]["n"],
            "parent_accuracy": base["gap_strata"]["long"]["accuracy"],
            "parent_excess": base_excess,
            "treatment_accuracy": treat["gap_strata"]["long"]["accuracy"],
            "treatment_excess": t_excess,
            "control_accuracy": ctrl["gap_strata"]["long"]["accuracy"],
            "control_excess": c_excess,
            "treatment_minus_control": vs_control,
            "bootstrap_ci95": [low, high],
        },
        "short_range_stratum_gap_0_1": {
            "parent": base["gap_strata"]["a_gap_0_1"]["accuracy"],
            "treatment": treat["gap_strata"]["a_gap_0_1"]["accuracy"],
            "control": ctrl["gap_strata"]["a_gap_0_1"]["accuracy"],
            "parent_excess": base_short,
            "treatment_excess": t_short,
            "control_excess": ctrl["gap_strata"]["a_gap_0_1"]["excess"],
        },
        "transport_range_probe": {
            label: {
                arm: probe[label]["summary"]["arm"][arm]["accuracy"]
                for arm in (
                    "as_is",
                    "append_query",
                    "append_query_then_filler_1",
                    "append_query_then_filler_2",
                    "append_query_then_filler_4",
                    "append_query_then_filler_8",
                    "append_other_key",
                    "append_unused_key",
                )
            }
            for label in ("t1_treatment_u16400", "t1_control_u16400")
        },
        "regressions": regressions,
        "negative_controls": controls_ok,
        "primary_verdict": primary,
        "verdict": verdict,
        "protected_material_opened": False,
        "gates_changed": False,
        "authoritative_parent_unchanged": True,
    }
    (OUT / "ADJUDICATION_130001_400.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
