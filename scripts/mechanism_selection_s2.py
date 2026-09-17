"""Read-only S2 mechanism splits from frozen eval JSON. No training."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baby_v010.isolation_classify import classify_payload_origin
from src.baby_v010.selection_s2 import OUT, digest

ISOLATION_PANELS = ROOT / "runs/v2r4_isolation_panels/ISOLATION_PANELS.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def classify_panel(items, rows):
    origins = Counter()
    by_k = defaultdict(Counter)
    n_k = Counter()
    rest = 0
    miss_rank1 = 0
    miss = 0
    for item, row in zip(items, rows):
        origin = classify_payload_origin(item, row)["payload_origin"]
        origins[origin] += 1
        k = int(item["pair_count"])
        n_k[k] += 1
        by_k[k][origin] += 1
        rest += int(all(rank == 1 for rank in row["all_ranks"][1 : len(item["target_span"])]))
        if origin != "queried_new":
            miss += 1
            miss_rank1 += int(row["target_rank"] == 1)
    return {
        "n": len(rows),
        "follow": origins.get("queried_new", 0),
        "stuck_old": origins.get("original", 0),
        "other": origins.get("other_competitor", 0),
        "off": origins.get("off_inventory", 0) + origins.get("original_absent_copy", 0),
        "new_gold_rest_lock": rest / max(1, len(rows)),
        "miss": miss,
        "miss_rank1": miss_rank1,
        "by_K": {
            str(k): {
                "n": n_k[k],
                "follow": by_k[k].get("queried_new", 0),
                "follow_rate": by_k[k].get("queried_new", 0) / n_k[k],
                "chance": 1 / k,
            }
            for k in (2, 3, 4)
            if n_k[k]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=120001)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    isolation_items = load(ISOLATION_PANELS)
    base = load(OUT / f"treatment_{args.seed}/eval_0000.json")
    control = load(OUT / f"control_{args.seed}/eval_{args.step:04d}.json")
    treatment = load(OUT / f"treatment_{args.seed}/eval_{args.step:04d}.json")
    name = "query_swap_same_surface_novel"
    report = {
        "step": args.step,
        "seed": args.seed,
        "parent_eval_sha256": digest(OUT / f"treatment_{args.seed}/eval_0000.json"),
        "summaries": {
            "parent": base["summary"],
            "control": control["summary"],
            "treatment": treatment["summary"],
        },
        "language_dev_ce": {
            "parent": base["language_dev_ce"],
            "control": control["language_dev_ce"],
            "treatment": treatment["language_dev_ce"],
        },
        "query_swap_novel": {
            "parent": classify_panel(isolation_items[name], base["isolation"]["rows"][name]),
            "control": classify_panel(isolation_items[name], control["isolation"]["rows"][name]),
            "treatment": classify_panel(isolation_items[name], treatment["isolation"]["rows"][name]),
        },
        "induction": {
            "parent": base["frozen"]["summaries"]["primitive_induction"],
            "control": control["frozen"]["summaries"]["primitive_induction"],
            "treatment": treatment["frozen"]["summaries"]["primitive_induction"],
        },
        "value_absent_exact": {
            key: {
                "parent": base["isolation"]["summaries"][key]["free_exact"],
                "control": control["isolation"]["summaries"][key]["free_exact"],
                "treatment": treatment["isolation"]["summaries"][key]["free_exact"],
            }
            for key in base["isolation"]["summaries"]
            if key.startswith("value_absent")
        },
        "protected_material_opened": False,
    }
    dest = OUT / f"MECHANISM_{args.seed}_{args.step}.json"
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "wrote": str(dest),
        "query_logit_effect": {k: v["query_logit_effect"] for k, v in report["summaries"].items()},
        "same_first_token_rate": {k: v["same_first_token_rate"] for k, v in report["summaries"].items()},
        "queried_inventory_rank1": {k: v["queried_inventory_rank1"] for k, v in report["summaries"].items()},
        "body_accuracy": {k: v["body_accuracy"] for k, v in report["summaries"].items()},
        "query_swap_follow": {k: v["follow"] for k, v in report["query_swap_novel"].items()},
        "induction_first_top1": {k: v["first_top1"] for k, v in report["induction"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
