import torch
from pathlib import Path

base = Path(r"C:\DaveLM-CADAVER")
ck = {
    "hist": base / "sf2_kl_parent_retention_run_v2" / "run" / "checkpoint_100.pt",
    "A": base / "sf2_update100_replay_mismatch_autopsy_v1" / "replay_A" / "checkpoint_100.pt",
    "B": base / "sf2_update100_replay_mismatch_autopsy_v1" / "replay_B" / "checkpoint_100.pt",
    "SF3": base / "sf3_prospective_treatment_v1" / "run" / "checkpoint_100.pt",
    "SF4a": base / "sf4_common_state_lr_anneal_v1" / "common_a" / "checkpoint_100.pt",
    "SF4b": base / "sf4_common_state_lr_anneal_v1" / "common_b" / "checkpoint_100.pt",
    "p1a": base / "overnight_reproducibility_resolution_v1" / "path1_evidence" / "probe_1a" / "checkpoint_100.pt",
    "p1b": base / "overnight_reproducibility_resolution_v1" / "path1_evidence" / "probe_1b" / "checkpoint_100.pt",
}
st = {k: torch.load(v, map_location="cpu", weights_only=False)["model_state_dict"] for k, v in ck.items()}
names = list(st)
print("pairwise max_abs (== means bitwise equal)")
hdr = "      " + "".join(f"{n:>7}" for n in names)
print(hdr)
for a in names:
    row = f"{a:>6} "
    for b in names:
        if a == b:
            row += "    == "
            continue
        mx = 0.0
        for k in st[a]:
            if not torch.equal(st[a][k], st[b][k]):
                mx = max(mx, float((st[a][k] - st[b][k]).abs().max()))
        row += f"{mx:7.2e}"
    print(row)
