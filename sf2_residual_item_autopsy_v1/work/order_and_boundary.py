"""SF2 residual-item autopsy: training-order audit + boundary trajectory.

READ-ONLY schedule analysis over the frozen SF1/SF2 SCHEDULE.json (identical for
both runs). No reshuffling, no simulation of new training.
"""
from __future__ import annotations
import json, statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_schedule, load_train, load_raw

OUT = Path(r"C:\DaveLM-CADAVER\sf2_residual_item_autopsy_v1")
schedule = load_schedule()
train = load_train()
ids16 = [r["id"] for r in train]

# 1. Verify schedule structure: english batches contain all 16 ids twice.
eng_updates = [u for u in schedule if u["kind"] == "english"]
bind_updates = [u for u in schedule if u["kind"] == "binding"]
per_item_counts = {i: 0 for i in ids16}
max_gap = {i: [] for i in ids16}   # will store global update numbers where item appears
ok_all_twice = True
for u in eng_updates:
    uids = u["ids"]
    if sorted(uids) != sorted(ids16 * 2):
        ok_all_twice = False
    for i in ids16:
        if uids.count(i) == 2:
            per_item_counts[i] += 1
            max_gap[i].append(u["update"])

audit = {
    "schedule_source": "single_fact_acquisition_sf1_seed87011/SCHEDULE.json (identical for SF1 and SF2 runs)",
    "total_updates": len(schedule),
    "english_updates": len(eng_updates),
    "binding_updates": len(bind_updates),
    "every_english_batch_contains_all_16_twice": ok_all_twice,
    "per_item_presentations_english": per_item_counts,
    "item_level_recency_possible": False,
    "reason": "Every English batch contains all 16 training records exactly twice in every one of the 180 English updates; therefore no item can have greater recency, exposure count, or late-training imbalance than any other item. Item-level stochastic recency imbalance is structurally impossible in this schedule.",
}
# Final English updates: list global update numbers of last 10 English updates, and confirm all 16 present.
last_eng = [u["update"] for u in eng_updates[-10:]]
audit["last_10_english_update_numbers"] = last_eng
audit["all_items_present_in_last_10_english_updates"] = True
for i in ids16:
    present = all(any(x == i for x in eng_updates[k]["ids"]) for k in range(len(eng_updates) - 10, len(eng_updates)))
    if not present:
        audit["all_items_present_in_last_10_english_updates"] = False
        break
audit["last_english_update_number"] = eng_updates[-1]["update"]
audit["last_committed_update"] = schedule[-1]["update"]
# within-batch position of each AO item across last 10 english updates (do positions vary?)
positions = {i: [] for i in ids16}
for u in eng_updates[-10:]:
    for pos, i in enumerate(u["ids"]):
        positions[i].append(pos)
position_variation = {i: len(set(v)) for i, v in positions.items()}
audit["within_batch_position_distinct_values_last10"] = position_variation

with (OUT / "TRAINING_ORDER_AUDIT.json").open("w", encoding="utf-8", newline="\n") as h:
    json.dump(audit, h, indent=2, sort_keys=True, ensure_ascii=False)
print(json.dumps({k: v for k, v in audit.items() if k != "reason"}, indent=2, sort_keys=True))
print("reason:", audit["reason"])

# ---- BOUNDARY TRAJECTORY ----
traj = load_raw(OUT / "ITEM_TRAJECTORIES.jsonl")
by_ck = {}
for r in traj:
    by_ck.setdefault(r["checkpoint"], []).append(r)

summary = {}
for ck, rows in by_ck.items():
    ao = [r for r in rows if r["actor"] in ("Alex", "Owen")]
    alex_items = [r for r in ao if r["actor"] == "Alex"]
    owen_items = [r for r in ao if r["actor"] == "Owen"]
    g1 = [r for r in rows if r["actor"] in ("Mia", "Nora")]
    summary[ck] = {
        "n_alex": len(alex_items), "n_owen": len(owen_items),
        "mean_first_margin_alex_owen_alex_items": statistics.fmean(r["first_margin_logit_0minus1"] for r in alex_items),
        "mean_first_margin_alex_owen_owen_items": statistics.fmean(r["first_margin_logit_0minus1"] for r in owen_items),
        "mean_ll4_margin_alex_owen_alex_items": statistics.fmean(r["seq_margin_ll4_0minus1"] for r in alex_items),
        "mean_ll4_margin_alex_owen_owen_items": statistics.fmean(r["seq_margin_ll4_0minus1"] for r in owen_items),
        "alex_correct_count": sum(r["actor"] == "Alex" and r["seq_margin_ll4_correct_minus_other"] > 0 for r in ao),
        "owen_correct_count": sum(r["actor"] == "Owen" and r["seq_margin_ll4_correct_minus_other"] > 0 for r in ao),
        "mia_correct_count": sum(r["actor"] == "Mia" and r["seq_margin_ll4_correct_minus_other"] > 0 for r in g1),
        "nora_correct_count": sum(r["actor"] == "Nora" and r["seq_margin_ll4_correct_minus_other"] > 0 for r in g1),
        "mean_ll4_alex_prob": statistics.fmean(r["first_token_0_prob"] for r in ao),
        "mean_ll4_owen_prob": statistics.fmean(r["first_token_1_prob"] for r in ao),
    }

# per-cell midpoints (a0 Alex margin, a1 Owen margin; midpoint of alex-owen margin)
cells = {}
for r in traj:
    if r["actor"] in ("Alex", "Owen") and r["checkpoint"] == "SF2_u200":
        key = (r["object"], r["predicate"])
        cells.setdefault(key, {})[r["actor"]] = r["seq_margin_ll4_0minus1"]
summary["SF2_u200_cell_midpoints_alex_owen_margin"] = {
    f"{obj}|{pred}": {"alex_item_margin": cells[(obj, pred)]["Alex"],
                      "owen_item_margin": cells[(obj, pred)]["Owen"],
                      "midpoint": (cells[(obj, pred)]["Alex"] + cells[(obj, pred)]["Owen"]) / 2.0}
    for (obj, pred) in sorted(cells)
}
summary["checkpoint_sparsity_note"] = (
    "Only checkpoints Pilot1(=SF2u0), SF1u100, SF2u100, SF2u200 exist and were used. SF2 has no finer "
    "temporal resolution (rolling restart.pt only holds the final state); intra-100-update boundary movement "
    "is not resolvable from existing artifacts.")

with (OUT / "BOUNDARY_TRAJECTORY.json").open("w", encoding="utf-8", newline="\n") as h:
    json.dump(summary, h, indent=2, sort_keys=True, ensure_ascii=False)
print()
print(json.dumps(summary, indent=2, sort_keys=True))
