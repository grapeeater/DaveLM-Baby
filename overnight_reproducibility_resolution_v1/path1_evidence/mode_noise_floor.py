"""Endpoint-level mode-noise floor: per-item u100/u200 margins across the existing
M1 (historical) and M2 (replay A/B/SF3) checkpoints + SF4 common pairs. READ-ONLY."""
from __future__ import annotations
import json, statistics
from pathlib import Path

base = Path(r"C:\DaveLM-CADAVER")
pairs = [
    ("hist_u100", base / "sf2_kl_parent_retention_run_v2" / "run" / "update100_acquisition_RAW.jsonl"),
    ("replayA_u100", base / "sf2_update100_replay_mismatch_autopsy_v1" / "replay_A" / "update100_acquisition_RAW.jsonl"),
    ("replayB_u100", base / "sf2_update100_replay_mismatch_autopsy_v1" / "replay_B" / "update100_acquisition_RAW.jsonl"),
    ("sf4commonA_u100", base / "sf4_common_state_lr_anneal_v1" / "common_a" / "update100_acquisition_RAW.jsonl"),
    ("sf4commonB_u100", base / "sf4_common_state_lr_anneal_v1" / "common_b" / "update100_acquisition_RAW.jsonl"),
    ("hist_u200", base / "sf2_kl_parent_retention_run_v2" / "run" / "update200_acquisition_RAW.jsonl"),
    ("replayA_u200", base / "sf2_update100_replay_mismatch_autopsy_v1" / "replay_A" / "update200_acquisition_RAW.jsonl"),
    ("replayB_u200", base / "sf2_update100_replay_mismatch_autopsy_v1" / "replay_B" / "update200_acquisition_RAW.jsonl"),
]
data = {}
for name, path in pairs:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    data[name] = {r["id"]: r["margin"] for r in rows}

def compare(a_name, b_name, label):
    da, db = data[a_name], data[b_name]
    ids = sorted(da)
    diffs = {i: da[i] - db[i] for i in ids}
    return {
        "label": label,
        "max_abs_item_margin_diff": max(abs(v) for v in diffs.values()),
        "mean_abs_item_margin_diff": statistics.fmean(abs(v) for v in diffs.values()),
        "item_count_differing_in_correct_sign": sum(1 for i in ids if (da[i] > 0) != (db[i] > 0)),
    }

out = {
    "u100": [
        compare("hist_u100", "replayA_u100", "M1(hist) vs M2(replayA) u100"),
        compare("hist_u100", "sf4commonA_u100", "M1(hist) vs M2(common_a) u100"),
        compare("sf4commonA_u100", "sf4commonB_u100", "common_a vs common_b u100 (both fresh; mixed modes)"),
        compare("replayA_u100", "replayB_u100", "M2 vs M2 u100 (control: same mode)"),
    ],
    "u200": [
        compare("hist_u200", "replayA_u200", "M1(hist) vs M2(replayA) u200"),
        compare("hist_u200", "replayB_u200", "M1(hist) vs M2(replayB) u200"),
        compare("replayA_u200", "replayB_u200", "M2 vs M2 u200 (control: same mode)"),
    ],
}
print(json.dumps(out, indent=2))
(base / "overnight_reproducibility_resolution_v1" / "path1_evidence" / "mode_noise_floor.json").write_text(
    json.dumps(out, indent=2) + "\n", encoding="utf-8")
