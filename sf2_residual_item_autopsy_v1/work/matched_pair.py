"""SF2 residual-item autopsy: matched-pair (counterfactual) analysis."""
from __future__ import annotations
import json, statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_raw

OUT = Path(r"C:\DaveLM-CADAVER\sf2_residual_item_autopsy_v1")
traj = load_raw(OUT / "ITEM_TRAJECTORIES.jsonl")
by = {(r["id"], r["checkpoint"]): r for r in traj}

failed = "TRAIN:g0:carried:wooden boat:a0"
matched = "TRAIN:g0:carried:wooden boat:a1"
alex_ok = ["TRAIN:g0:found:small drum:a0", "TRAIN:g0:found:wooden boat:a0",
           "TRAIN:g0:carried:small drum:a0"]
owen_all = ["TRAIN:g0:found:small drum:a1", "TRAIN:g0:found:wooden boat:a1",
            "TRAIN:g0:carried:small drum:a1", "TRAIN:g0:carried:wooden boat:a1"]

checks = ["Pilot1_parent", "SF1_u100", "SF2_u100", "SF2_u200"]

def fields(r):
    return {"first_margin_alex_owen": r["first_margin_logit_0minus1"],
            "ll4_margin_alex_owen": r["seq_margin_ll4_0minus1"],
            "alex_prob": r["first_token_0_prob"], "owen_prob": r["first_token_1_prob"],
            "ll4_alex": r["cand0_ll4"], "ll4_owen": r["cand1_ll4"],
            "exact": r["exact"]}


out = {"failed_item": failed, "matched_owen_item": matched,
       "other_alex_correct_items": alex_ok, "all_owen_correct_items": owen_all,
       "per_checkpoint": {}}
for ck in checks:
    f, m = fields(by[(failed, ck)]), fields(by[(matched, ck)])
    # contextual separation within the failed pair's cell = owen_item - alex_item margin
    sep_failed_cell = m["ll4_margin_alex_owen"] - f["ll4_margin_alex_owen"]
    other_alex = [fields(by[(i, ck)]) for i in alex_ok]
    other_owen = [fields(by[(i, ck)]) for i in owen_all if i != matched]
    out["per_checkpoint"][ck] = {
        "failed_alex_item": f,
        "matched_owen_item": m,
        "cell_subject_conditioning_owen_minus_alex_ll4": sep_failed_cell,
        "mean_other_alex_items_ll4_margin": statistics.fmean(x["ll4_margin_alex_owen"] for x in other_alex),
        "min_other_alex_ll4_margin": min(x["ll4_margin_alex_owen"] for x in other_alex),
        "other_alex_margins": [x["ll4_margin_alex_owen"] for x in other_alex],
        "mean_other_owen_items_ll4_margin": statistics.fmean(x["ll4_margin_alex_owen"] for x in other_owen),
        "other_owen_margins": [x["ll4_margin_alex_owen"] for x in other_owen],
        "alex_vs_owen_evidence": {
            "failed_alex_owen_prob_ratio": f["owen_prob"] / f["alex_prob"] if f["alex_prob"] else None,
            "matched_owen_owen_prob": m["owen_prob"],
        },
    }
out["structure"] = {
    "shared_g0_pair_structure": "all 8 Alex/Owen items share pair g0; objects {small drum, wooden boat} x predicates {found, carried}; a0=Alex subject, a1=Owen subject; correct answer copies the prompt subject",
    "failed_cell_unique_features": "predicate=carried, object=wooden boat (also the matched Owen item cell); NOT a unique object/cue combination within g0",
    "token_geometry": "object 'wooden boat' and predicate 'carried' also appear in the passing cell TRAIN:g0:carried:wooden boat is the only cell where BOTH the Owen item margin is among the largest and the Alex item crossed zero",
}
with (OUT / "MATCHED_PAIR_ANALYSIS.json").open("w", encoding="utf-8", newline="\n") as h:
    json.dump(out, h, indent=2, sort_keys=True, ensure_ascii=False)
print(json.dumps(out, indent=2, sort_keys=True)[:6000])
