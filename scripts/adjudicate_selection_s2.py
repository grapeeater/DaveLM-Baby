"""Apply S2's preregistered criteria without changing the experiment."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baby_v010 import selection_s2 as s


def check_finite(obj) -> None:
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise AssertionError("nonfinite metric")
    elif isinstance(obj, dict):
        for value in obj.values():
            check_finite(value)
    elif isinstance(obj, list):
        for value in obj:
            check_finite(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=120001)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    s.verify()
    base = json.loads((s.OUT / f"treatment_{args.seed}/eval_0000.json").read_text(encoding="utf-8"))
    control = json.loads((s.OUT / f"control_{args.seed}/eval_{args.step:04d}.json").read_text(encoding="utf-8"))
    end = json.loads((s.OUT / f"treatment_{args.seed}/eval_{args.step:04d}.json").read_text(encoding="utf-8"))
    for blob in (base, control, end):
        check_finite(blob)
    parent, ctrl, treat = (blob["summary"] for blob in (base, control, end))
    regression = {
        "language_ce": end["language_dev_ce"] - base["language_dev_ce"] > 0.10,
        "rest_lock": parent["rest_lock"] - treat["rest_lock"] > 0.05,
    }
    for name, summary in base["frozen"]["summaries"].items():
        if name in ("broken_context", "broken_order", "context_lift_proxy"):
            continue
        for metric in ("first_top1", "free_exact"):
            regression[f"{name}_{metric}"] = summary[metric] - end["frozen"]["summaries"][name][metric] > 0.05
    for name in ("broken_context", "broken_order"):
        regression[f"{name}_exact"] = end["frozen"]["summaries"][name]["free_exact"] > 0.05
    for name, summary in end["isolation"]["summaries"].items():
        if name.startswith("value_absent"):
            regression[f"{name}_exact"] = summary["free_exact"] > 0.05
    differences = [x - y for x, y in zip(treat["body_accuracies"], ctrl["body_accuracies"])]
    rng = random.Random(s.BOOTSTRAP_SEED)
    boot = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(10000))
    query = "query_swap_same_surface_novel"
    criteria = {
        "body_accuracy_ge_070": treat["body_accuracy"] >= 0.70,
        "gain_parent_ge_020": treat["body_accuracy"] - parent["body_accuracy"] >= 0.20,
        "gain_control_ge_015": treat["body_accuracy"] - ctrl["body_accuracy"] >= 0.15,
        "all_K_above_chance_plus_015": all(treat["per_K"][str(k)] >= 1 / k + 0.15 for k in (2, 3, 4)),
        "query_swap_gain_ge_015": end["isolation"]["summaries"][query]["first_top1"] - base["isolation"]["summaries"][query]["first_top1"] >= 0.15,
        "same_surface_exact_gain_ge_015": end["frozen"]["summaries"]["same_surface_novel"]["free_exact"] - base["frozen"]["summaries"]["same_surface_novel"]["free_exact"] >= 0.15,
        "query_logit_effect_gain_ge_050": treat["query_logit_effect"] - parent["query_logit_effect"] >= 0.50,
        "same_first_token_rate_le_050": treat["same_first_token_rate"] <= 0.50,
        "inventory_rank1_gain_ge_015": treat["queried_inventory_rank1"] - parent["queried_inventory_rank1"] >= 0.15,
        "bootstrap_control_gain_lower_positive": boot[249] > 0,
    }
    futility = args.step == 200 and s.futility_at_200(base, end)
    hard_stop = end["language_dev_ce"] - base["language_dev_ce"] > 0.20
    for arm in ("control", "treatment"):
        for path in (s.OUT / f"{arm}_{args.seed}").glob("RECEIPT_*.json"):
            hard_stop = hard_stop or json.loads(path.read_text(encoding="utf-8"))["reason"].startswith("hard_stop")
    mechanism_partial = (
        treat["query_logit_effect"] - parent["query_logit_effect"] >= 0.25
        and parent["same_first_token_rate"] - treat["same_first_token_rate"] >= 0.15
    )
    accuracy_partial = treat["body_accuracy"] - parent["body_accuracy"] >= 0.10 and treat["body_accuracy"] - ctrl["body_accuracy"] >= 0.05
    if hard_stop:
        status = "HARD_STOP"
    elif any(regression.values()):
        status = "REGRESSION"
    elif all(criteria.values()):
        status = "SUCCESS"
    elif accuracy_partial or mechanism_partial:
        status = "PARTIAL"
    else:
        status = "FAILURE"
    result = {
        "seed": args.seed,
        "step": args.step,
        "status": status,
        "futility": futility,
        "success_criteria": criteria,
        "regression_flags": regression,
        "baseline": parent,
        "control": ctrl,
        "treatment": treat,
        "language_ce": {name: blob["language_dev_ce"] for name, blob in (("baseline", base), ("control", control), ("treatment", end))},
        "paired_body_bootstrap_95": [boot[249], boot[9749]],
        "original_frozen_summaries": {name: blob["frozen"]["summaries"] for name, blob in (("baseline", base), ("control", control), ("treatment", end))},
        "isolation_summaries": {name: blob["isolation"]["summaries"] for name, blob in (("baseline", base), ("control", control), ("treatment", end))},
        "manifest_sha256": s.digest(s.OUT / "MANIFEST.json"),
        "protected_material_opened": False,
        "gates_weakened": False,
    }
    dest = s.OUT / f"ADJUDICATION_{args.seed}_{args.step}.json"
    if dest.exists():
        raise RuntimeError("refuse overwrite")
    s.write(dest, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("baseline", "control", "treatment", "original_frozen_summaries", "isolation_summaries")}), flush=True)


if __name__ == "__main__":
    main()
