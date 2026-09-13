import json
from pathlib import Path

base = Path(r"C:\DaveLM-CADAVER")
items = [
    ("SF4a", base / "sf4_common_state_lr_anneal_v1" / "common_a"),
    ("SF4b", base / "sf4_common_state_lr_anneal_v1" / "common_b"),
]
for name, d in items:
    u2 = json.loads((d / "metric_002.json").read_text(encoding="utf-8"))
    u12 = json.loads((d / "metric_012.json").read_text(encoding="utf-8"))
    print(name, "u2kl=%.15f" % u2.get("kl"), "u12kl=%.15f" % u12.get("kl"))
