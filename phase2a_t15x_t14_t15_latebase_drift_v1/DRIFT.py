"""Read-only T14 vs T15 late-base drift vs Phase1G.

Does not train. Does not load T3 TEST. Does not parent checkpoints.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path  # re used for block index

import torch

ROOT = Path(r"C:\DaveLM-CADAVER")
OUT = ROOT / "phase2a_t15x_t14_t15_latebase_drift_v1"
PARENT = ROOT / "baby_vnext_phase1g_language_v1_run_seed610001" / "checkpoints" / "best.pt"

CKPTS = {
    "t14_730001_u750": ROOT / "phase2a_t14_factclause_pointer_lr3p75e5_750_v1_run_seed730001" / "checkpoints" / "checkpoint_0750.pt",
    "t14_730002_u750": ROOT / "phase2a_t14_factclause_pointer_lr3p75e5_750_v1_run_seed730002" / "checkpoints" / "checkpoint_0750.pt",
    "t14_730003_u750": ROOT / "phase2a_t14_factclause_pointer_lr3p75e5_750_v1_run_seed730003" / "checkpoints" / "checkpoint_0750.pt",
    "t15_740001_u750": ROOT / "phase2a_t15_latebase_blocks4to7_lr3p75e5_750_v1_run_seed740001" / "checkpoints" / "checkpoint_0750.pt",
    "t15_740002_u750": ROOT / "phase2a_t15_latebase_blocks4to7_lr3p75e5_750_v1_run_seed740002" / "checkpoints" / "checkpoint_0750.pt",
    "t15_740003_u750": ROOT / "phase2a_t15_latebase_blocks4to7_lr3p75e5_750_v1_run_seed740003" / "checkpoints" / "checkpoint_0750.pt",
}


def coarse_key(name: str) -> str:
    if name.startswith("binding") or name.startswith("localizer") or "binding" in name.split(".")[0]:
        return "binding"
    if not name.startswith("base_model."):
        return "other"
    if name.startswith("base_model.blocks."):
        i = int(name.split(".")[2])
        if ".attention." in name:
            return f"block{i}.attn"
        if ".feed_forward." in name:
            return f"block{i}.mlp"
        if ".norm" in name:
            return f"block{i}.norm"
        return f"block{i}.other"
    if "token_embedding" in name:
        return "embed.token"
    if "position_embedding" in name:
        return "embed.pos"
    if "final_norm" in name:
        return "final_norm"
    if "language_head" in name:
        return "language_head"
    return "base_other"


def load_sd(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["model_state_dict"]


def pack(src: dict) -> dict:
    out = {}
    for k, v in src.items():
        out[k] = {"rms": math.sqrt(v["sq"] / v["n"]) if v["n"] else 0.0, "n": v["n"]}
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["rms"]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    parent = load_sd(PARENT)
    drifts = {}
    deltas = {}  # label -> {group: (sum_dot_self, tensor? too big)} — store per-group flattened delta stats only
    group_dots = {}  # label -> group -> {'sq': , 'n':} already in drifts
    # For cosine between two ckpts' deltas, accumulate dot(d1,d2) while second is loaded.
    # First pass: compute drift vs parent and keep block4-7 / 8-11 / embed/head group vectors on disk is too big.
    # Instead accumulate per-group: we cannot keep all deltas. Compute pairwise in a second nested loop
    # loading two at a time is 2*246MB — OK. But 6*6 is a lot of IO. Do T15 vs T14_730002 and T15 vs T14_730001 only.

    for label, path in CKPTS.items():
        sd = load_sd(path)
        coarse = defaultdict(lambda: {"sq": 0.0, "n": 0})
        max_abs_811 = 0.0
        for n, p in parent.items():
            if n not in sd:
                continue
            d = (sd[n].float() - p.float()).reshape(-1)
            key = coarse_key(n)
            coarse[key]["sq"] += float(torch.dot(d, d))
            coarse[key]["n"] += int(d.numel())
            m = re.search(r"base_model\.blocks\.(\d+)\.", n)
            if m and int(m.group(1)) >= 8:
                max_abs_811 = max(max_abs_811, float(d.abs().max()))
        packed = pack(coarse)
        blocks47 = ["block4.attn", "block4.mlp", "block4.norm",
                    "block5.attn", "block5.mlp", "block5.norm",
                    "block6.attn", "block6.mlp", "block6.norm",
                    "block7.attn", "block7.mlp", "block7.norm"]
        blocks811 = ["block8.attn", "block8.mlp", "block8.norm",
                     "block9.attn", "block9.mlp", "block9.norm",
                     "block10.attn", "block10.mlp", "block10.norm",
                     "block11.attn", "block11.mlp", "block11.norm"]
        def mean_rms(keys):
            sq = sum(packed[k]["rms"] ** 2 * packed[k]["n"] for k in keys if k in packed)
            n = sum(packed[k]["n"] for k in keys if k in packed)
            return math.sqrt(sq / n) if n else 0.0
        drifts[label] = {
            "coarse": packed,
            "block4to7_rms": mean_rms(blocks47),
            "block8to11_rms": mean_rms(blocks811),
            "block4_rms": mean_rms(["block4.attn", "block4.mlp", "block4.norm"]),
            "embed_token_rms": packed.get("embed.token", {}).get("rms", 0.0),
            "embed_pos_rms": packed.get("embed.pos", {}).get("rms", 0.0),
            "language_head_rms": packed.get("language_head", {}).get("rms", 0.0),
            "binding_rms": packed.get("binding", {}).get("rms", 0.0),
            "max_abs_blocks8to11": max_abs_811,
            "path": str(path),
        }
        print(f"{label} 4-7={drifts[label]['block4to7_rms']:.6g} 8-11={drifts[label]['block8to11_rms']:.6g} "
              f"b4={drifts[label]['block4_rms']:.6g} head={drifts[label]['language_head_rms']:.6g} "
              f"max811={max_abs_811:.3g}", flush=True)
        del sd

    # Pairwise cosine of (ckpt-parent) on blocks 4-7 and on embed+head
    pairs = [
        ("t15_740001_u750", "t14_730002_u750"),
        ("t15_740002_u750", "t14_730002_u750"),
        ("t15_740003_u750", "t14_730002_u750"),
        ("t15_740001_u750", "t14_730001_u750"),
        ("t15_740003_u750", "t14_730001_u750"),
        ("t15_740001_u750", "t15_740003_u750"),
        ("t14_730001_u750", "t14_730002_u750"),
    ]
    cosines = []
    for a, b in pairs:
        sda = load_sd(CKPTS[a])
        sdb = load_sd(CKPTS[b])
        buckets = {
            "block4": (0.0, 0.0, 0.0),
            "block4to7": (0.0, 0.0, 0.0),
            "block8to11": (0.0, 0.0, 0.0),
            "embed_head": (0.0, 0.0, 0.0),
        }
        # store as mutable
        acc = {k: [0.0, 0.0, 0.0] for k in buckets}

        def add(key, da, db):
            acc[key][0] += float(torch.dot(da, db))
            acc[key][1] += float(torch.dot(da, da))
            acc[key][2] += float(torch.dot(db, db))

        for n, p in parent.items():
            if n not in sda or n not in sdb:
                continue
            da = (sda[n].float() - p.float()).reshape(-1)
            db = (sdb[n].float() - p.float()).reshape(-1)
            m = re.search(r"base_model\.blocks\.(\d+)\.", n)
            if m:
                i = int(m.group(1))
                if i == 4:
                    add("block4", da, db)
                if 4 <= i <= 7:
                    add("block4to7", da, db)
                if 8 <= i <= 11:
                    add("block8to11", da, db)
            elif any(s in n for s in ("token_embedding", "position_embedding", "language_head", "final_norm")):
                add("embed_head", da, db)

        def cosine(trip):
            num, a2, b2 = trip
            den = math.sqrt(a2) * math.sqrt(b2)
            return num / den if den else 0.0

        row = {"a": a, "b": b, **{k: cosine(acc[k]) for k in acc}}
        cosines.append(row)
        print(f"COS {a} vs {b} b4={row['block4']:.4f} 4-7={row['block4to7']:.4f} 8-11={row['block8to11']:.4f} emb={row['embed_head']:.4f}", flush=True)
        del sda, sdb

    payload = {
        "test_loaded": False,
        "parent": str(PARENT),
        "drifts": drifts,
        "delta_cosines": cosines,
    }
    (OUT / "DRIFT.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("WROTE", OUT / "DRIFT.json")


if __name__ == "__main__":
    main()
