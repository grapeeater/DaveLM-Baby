import json
from pathlib import Path

base = Path(r"C:\DaveLM-CADAVER")
items = [
    ("SF4b", base / "sf4_common_state_lr_anneal_v1" / "common_b" / "TRAIN_METRICS.jsonl"),
    ("SF4a", base / "sf4_common_state_lr_anneal_v1" / "common_a" / "TRAIN_METRICS.jsonl"),
    ("hist", base / "sf2_kl_parent_retention_run_v2" / "run" / "TRAIN_METRICS.jsonl"),
    ("replayA", base / "sf2_update100_replay_mismatch_autopsy_v1" / "replay_A" / "TRAIN_METRICS.jsonl"),
]
for name, path in items:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8")]
    u2 = [r for r in rows if r["update"] == 2][0]
    u12 = [r for r in rows if r["update"] == 12][0]
    print(name, "u2kl=%.15f" % u2["kl"], "u12kl=%.15f" % u12["kl"])
