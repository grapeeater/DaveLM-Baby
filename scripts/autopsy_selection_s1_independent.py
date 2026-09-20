"""Independent S1 autopsy from frozen machine-readable artifacts.

Read-only. Does not train, mutate S1 receipts, or open TEST/FINAL/SACRED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baby_v010.isolation_classify import classify_payload_origin
from src.baby_v010.selection_s1 import PARENT, PARENT_SHA, OUT, digest

SUMS = ROOT / "research" / "V010_SELECTION_REPAIR_S1_SHA256SUMS.txt"
ISOLATION_PANELS = ROOT / "runs" / "v2r4_isolation_panels" / "ISOLATION_PANELS.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_sha256sums(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest_hex, _, rel = line.partition("  ")
        out[rel.replace("\\", "/")] = digest_hex
    return out


def inventory_rank(candidate_logits: list[float], query_index: int) -> int:
    queried = candidate_logits[query_index]
    return 1 + sum(1 for value in candidate_logits if value > queried)


def greedy_inventory_index(candidate_logits: list[float]) -> int:
    best = max(candidate_logits)
    return next(i for i, value in enumerate(candidate_logits) if value == best)


def body_collapse(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["body_id"]].append(row)
    same_first = 0
    all_correct = 0
    n = len(groups)
    per_k_same = Counter()
    per_k_n = Counter()
    winner_index_constant = 0
    for body_id, group in groups.items():
        k = group[0]["K"]
        per_k_n[k] += 1
        firsts = [row["emitted"][0] if row["emitted"] else None for row in group]
        if len(set(firsts)) == 1:
            same_first += 1
            per_k_same[k] += 1
        if all(row["target_rank"] == 1 for row in group):
            all_correct += 1
        winners = [greedy_inventory_index(row["candidate_logits"]) for row in group]
        if len(set(winners)) == 1:
            winner_index_constant += 1
    return {
        "n_bodies": n,
        "same_emitted_first_token": same_first,
        "same_emitted_first_token_rate": same_first / n,
        "all_queries_correct": all_correct,
        "all_queries_correct_rate": all_correct / n,
        "constant_inventory_argmax": winner_index_constant,
        "constant_inventory_argmax_rate": winner_index_constant / n,
        "same_first_by_K": {str(k): {"same": per_k_same[k], "n": per_k_n[k]} for k in (2, 3, 4)},
    }


def rank_hist(rows: list[dict]) -> dict:
    vocab = Counter(row["target_rank"] for row in rows)
    inv = Counter(inventory_rank(row["candidate_logits"], row["query_index"]) for row in rows)
    inv_correct = sum(row["inventory_correct"] for row in rows)
    queried_wins = sum(inventory_rank(row["candidate_logits"], row["query_index"]) == 1 for row in rows)
    runner_up = sum(inventory_rank(row["candidate_logits"], row["query_index"]) == 2 for row in rows)
    worse = sum(inventory_rank(row["candidate_logits"], row["query_index"]) > 2 for row in rows)
    margins = []
    for row in rows:
        logits = row["candidate_logits"]
        q = logits[row["query_index"]]
        others = [v for i, v in enumerate(logits) if i != row["query_index"]]
        margins.append(q - max(others))
    return {
        "n": len(rows),
        "vocab_rank_hist": {str(k): vocab[k] for k in sorted(vocab)},
        "inventory_rank_hist": {str(k): inv[k] for k in sorted(inv)},
        "inventory_argmax_rate": inv_correct / len(rows),
        "queried_inventory_rank1": queried_wins / len(rows),
        "queried_inventory_rank2": runner_up / len(rows),
        "queried_inventory_rank_gt2": worse / len(rows),
        "vocab_rank1": sum(row["target_rank"] == 1 for row in rows) / len(rows),
        "mean_inventory_margin": statistics.mean(margins),
        "median_inventory_margin": statistics.median(margins),
        "mean_vocab_margin": statistics.mean(row["first_margin"] for row in rows),
        "mean_queried_vocab_rank": statistics.mean(row["target_rank"] for row in rows),
        "median_queried_vocab_rank": statistics.median(row["target_rank"] for row in rows),
        "rest_lock": statistics.mean(row["rest_lock"] for row in rows),
        "value_exact": statistics.mean(row["value_exact"] for row in rows),
        "free_exact": statistics.mean(row["free_exact"] for row in rows),
    }


def per_k_hist(rows: list[dict]) -> dict:
    out = {}
    for k in (2, 3, 4):
        subset = [row for row in rows if row["K"] == k]
        out[str(k)] = rank_hist(subset)
        out[str(k)]["chance"] = 1 / k
    return out


def classify_query_swap(items: list[dict], rows: list[dict]) -> dict:
    if len(items) != len(rows):
        raise RuntimeError(("query-swap length mismatch", len(items), len(rows)))
    origins = Counter()
    miss_rank1 = 0
    miss_runner = 0
    miss_n = 0
    rest_lock_new = 0
    by_k = defaultdict(Counter)
    n_by_k = Counter()
    for item, row in zip(items, rows):
        origin = classify_payload_origin(item, row)["payload_origin"]
        origins[origin] += 1
        k = int(item["pair_count"])
        n_by_k[k] += 1
        by_k[k][origin] += 1
        rest = all(rank == 1 for rank in row["all_ranks"][1 : len(item["target_span"])])
        rest_lock_new += int(rest)
        if origin != "queried_new":
            miss_n += 1
            if row["target_rank"] == 1:
                miss_rank1 += 1
            elif row["target_rank"] == 2:
                miss_runner += 1
    return {
        "n": len(rows),
        "origins": dict(origins),
        "follow": origins.get("queried_new", 0),
        "stuck_old": origins.get("original", 0),
        "other_competitor": origins.get("other_competitor", 0),
        "off": origins.get("off_inventory", 0) + origins.get("original_absent_copy", 0),
        "new_gold_rest_lock": rest_lock_new / len(rows),
        "miss_n": miss_n,
        "miss_queried_vocab_rank1": miss_rank1,
        "miss_queried_vocab_rank2": miss_runner,
        "by_K": {
            str(k): {
                "n": n_by_k[k],
                "follow": by_k[k].get("queried_new", 0),
                "stuck_old": by_k[k].get("original", 0),
                "other": by_k[k].get("other_competitor", 0),
                "off": by_k[k].get("off_inventory", 0) + by_k[k].get("original_absent_copy", 0),
                "follow_rate": by_k[k].get("queried_new", 0) / n_by_k[k],
                "chance": 1 / k,
            }
            for k in (2, 3, 4)
            if n_by_k[k]
        },
    }


def bootstrap_ci(treatment_acc: list[float], control_acc: list[float], seed: int = 110300, n: int = 10000):
    differences = [x - y for x, y in zip(treatment_acc, control_acc)]
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(n))
    return [boot[249], boot[9749]], statistics.mean(differences)


def schedule_stats(schedule: list[dict]) -> dict:
    keyed_k = Counter()
    keyed_rows = 0
    induction_rows = 0
    language = 0
    bodies_with_two_queries = 0
    structured = 0
    query_index_pairs = Counter()
    for spec in schedule:
        if spec["task"] == "language":
            language += 1
            continue
        structured += 1
        items = spec["items"]
        keyed = [row for row in items if row["kind"] == "keyed"]
        induction_rows += sum(row["kind"] == "induction" for row in items)
        keyed_rows += len(keyed)
        bodies: dict[tuple, list] = defaultdict(list)
        for row in keyed:
            keyed_k[row["pair_count"]] += 1
            bodies[tuple(row["input"][:40])].append(row["query_index"])
        for queries in bodies.values():
            if len(queries) == 2 and len(set(queries)) == 2:
                bodies_with_two_queries += 1
            query_index_pairs[tuple(sorted(queries))] += 1
    return {
        "language_updates": language,
        "structured_updates": structured,
        "keyed_rows": keyed_rows,
        "induction_rows": induction_rows,
        "keyed_pair_count": dict(keyed_k),
        "approx_two_query_bodies": bodies_with_two_queries,
    }


def frozen_delta(base: dict, other: dict) -> dict:
    out = {}
    for name, summary in base["summaries"].items():
        other_s = other["summaries"][name]
        out[name] = {
            "first_top1": [summary["first_top1"], other_s["first_top1"], other_s["first_top1"] - summary["first_top1"]],
            "free_exact": [summary["free_exact"], other_s["free_exact"], other_s["free_exact"] - summary["free_exact"]],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT / "INDEPENDENT_S1_AUTOPSY.json")
    args = parser.parse_args()

    expected = parse_sha256sums(SUMS)
    hash_report = {}
    hash_ok = True
    for rel, want in expected.items():
        path = ROOT / rel
        got = digest(path) if path.exists() else None
        match = got == want
        hash_ok &= bool(match)
        hash_report[rel] = {"expected": want, "got": got, "match": match}

    parent_sha = digest(PARENT)
    base = load(OUT / "treatment_110001" / "eval_0000.json")
    control0 = load(OUT / "control_110001" / "eval_0000.json")
    control = load(OUT / "control_110001" / "eval_0400.json")
    treatment = load(OUT / "treatment_110001" / "eval_0400.json")
    control200 = load(OUT / "control_110001" / "eval_0200.json")
    treatment200 = load(OUT / "treatment_110001" / "eval_0200.json")
    adj = load(OUT / "ADJUDICATION_110001_400.json")
    isolation_items = load(ISOLATION_PANELS)
    schedule = load(OUT / "SCHEDULE_110001.json")
    diagnostic = load(OUT / "DIAGNOSTIC.json")

    # Independent bootstrap
    ci, point = bootstrap_ci(treatment["summary"]["body_accuracies"], control["summary"]["body_accuracies"])

    qs_name = "query_swap_same_surface_novel"
    qs = {
        "parent": classify_query_swap(isolation_items[qs_name], base["isolation"]["rows"][qs_name]),
        "control": classify_query_swap(isolation_items[qs_name], control["isolation"]["rows"][qs_name]),
        "treatment": classify_query_swap(isolation_items[qs_name], treatment["isolation"]["rows"][qs_name]),
    }
    qs_short = "query_swap_short_keyed"
    qs_short_report = {
        "parent": classify_query_swap(isolation_items[qs_short], base["isolation"]["rows"][qs_short]),
        "control": classify_query_swap(isolation_items[qs_short], control["isolation"]["rows"][qs_short]),
        "treatment": classify_query_swap(isolation_items[qs_short], treatment["isolation"]["rows"][qs_short]),
    }

    report = {
        "status": "INDEPENDENT_S1_AUTOPSY",
        "protected_material_opened": False,
        "parent_sha256": parent_sha,
        "parent_sha_match": parent_sha == PARENT_SHA,
        "hash_ok": hash_ok,
        "hash_report": {k: v for k, v in hash_report.items() if not v["match"]},
        "hash_checked": len(hash_report),
        "baseline_eval_files_byte_identical_summary": base["summary"] == control0["summary"],
        "language_dev_ce": {
            "parent": base["language_dev_ce"],
            "control": control["language_dev_ce"],
            "treatment": treatment["language_dev_ce"],
            "treatment_delta": treatment["language_dev_ce"] - base["language_dev_ce"],
        },
        "body_macro": {
            "parent": base["summary"]["body_accuracy"],
            "control_200": control200["summary"]["body_accuracy"],
            "control_400": control["summary"]["body_accuracy"],
            "treatment_200": treatment200["summary"]["body_accuracy"],
            "treatment_400": treatment["summary"]["body_accuracy"],
            "control_gain": control["summary"]["body_accuracy"] - base["summary"]["body_accuracy"],
            "treatment_gain": treatment["summary"]["body_accuracy"] - base["summary"]["body_accuracy"],
            "treatment_minus_control": treatment["summary"]["body_accuracy"] - control["summary"]["body_accuracy"],
        },
        "query_logit_effect": {
            "parent": base["summary"]["query_logit_effect"],
            "control": control["summary"]["query_logit_effect"],
            "treatment": treatment["summary"]["query_logit_effect"],
        },
        "mean_margin": {
            "parent": base["summary"]["mean_margin"],
            "control": control["summary"]["mean_margin"],
            "treatment": treatment["summary"]["mean_margin"],
        },
        "per_K": {
            "parent": base["summary"]["per_K"],
            "control": control["summary"]["per_K"],
            "treatment": treatment["summary"]["per_K"],
        },
        "bootstrap_treatment_minus_control_95": ci,
        "bootstrap_point": point,
        "adjudication_status": adj["status"],
        "adjudication_futility": adj["futility"],
        "adjudication_success_all_false": not any(adj["success_criteria"].values()),
        "diagnostic_parent": {
            "collapse": body_collapse(base["rows"]),
            "ranks": rank_hist(base["rows"]),
            "per_K_ranks": per_k_hist(base["rows"]),
        },
        "diagnostic_control": {
            "collapse": body_collapse(control["rows"]),
            "ranks": rank_hist(control["rows"]),
            "per_K_ranks": per_k_hist(control["rows"]),
        },
        "diagnostic_treatment": {
            "collapse": body_collapse(treatment["rows"]),
            "ranks": rank_hist(treatment["rows"]),
            "per_K_ranks": per_k_hist(treatment["rows"]),
        },
        "query_swap_novel": qs,
        "query_swap_short": qs_short_report,
        "frozen_control_delta": frozen_delta(base["frozen"], control["frozen"]),
        "frozen_treatment_delta": frozen_delta(base["frozen"], treatment["frozen"]),
        "isolation_first_top1": {
            name: {
                "parent": base["isolation"]["summaries"][name]["first_top1"],
                "control": control["isolation"]["summaries"][name]["first_top1"],
                "treatment": treatment["isolation"]["summaries"][name]["first_top1"],
            }
            for name in base["isolation"]["summaries"]
        },
        "value_absent_exact": {
            name: {
                "parent": base["isolation"]["summaries"][name]["free_exact"],
                "control": control["isolation"]["summaries"][name]["free_exact"],
                "treatment": treatment["isolation"]["summaries"][name]["free_exact"],
            }
            for name in base["isolation"]["summaries"]
            if name.startswith("value_absent")
        },
        "schedule_110001": schedule_stats(schedule),
        "diagnostic_bodies": len({row["body_id"] for row in diagnostic}),
        "diagnostic_rows": len(diagnostic),
    }

    # Did inventory rank-2 move to rank-1?
    parent_r = report["diagnostic_parent"]["ranks"]
    treat_r = report["diagnostic_treatment"]["ranks"]
    control_r = report["diagnostic_control"]["ranks"]
    report["rank2_to_rank1"] = {
        "parent_inventory_rank1": parent_r["queried_inventory_rank1"],
        "control_inventory_rank1": control_r["queried_inventory_rank1"],
        "treatment_inventory_rank1": treat_r["queried_inventory_rank1"],
        "parent_inventory_rank2": parent_r["queried_inventory_rank2"],
        "control_inventory_rank2": control_r["queried_inventory_rank2"],
        "treatment_inventory_rank2": treat_r["queried_inventory_rank2"],
        "treatment_rank1_gain_vs_parent": treat_r["queried_inventory_rank1"] - parent_r["queried_inventory_rank1"],
        "treatment_rank1_gain_vs_control": treat_r["queried_inventory_rank1"] - control_r["queried_inventory_rank1"],
        "treatment_margin_gain_vs_parent": treat_r["mean_inventory_margin"] - parent_r["mean_inventory_margin"],
        "first_token_ce_bought_stronger_signal": (
            treat_r["queried_inventory_rank1"] - parent_r["queried_inventory_rank1"] >= 0.05
            or treat_r["mean_inventory_margin"] - parent_r["mean_inventory_margin"] >= 0.25
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "hash_ok": hash_ok,
        "parent_sha_match": report["parent_sha_match"],
        "body_macro": report["body_macro"],
        "bootstrap": report["bootstrap_treatment_minus_control_95"],
        "query_logit_effect": report["query_logit_effect"],
        "mean_margin": report["mean_margin"],
        "rank2_to_rank1": report["rank2_to_rank1"],
        "collapse_parent": report["diagnostic_parent"]["collapse"],
        "collapse_treatment": report["diagnostic_treatment"]["collapse"],
        "query_swap_novel": {k: {kk: vv for kk, vv in v.items() if kk != "by_K"} for k, v in qs.items()},
        "induction": report["frozen_treatment_delta"]["primitive_induction"],
        "adjudication": adj["status"],
        "futility": adj["futility"],
        "wrote": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
