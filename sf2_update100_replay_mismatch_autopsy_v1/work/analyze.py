"""SF2 update-100 replay mismatch autopsy — core comparisons (READ-ONLY)."""
from __future__ import annotations
import json, math, sys, hashlib
from pathlib import Path
import torch

ROOT = Path(r"C:\DaveLM-CADAVER")
AUT = ROOT / "sf2_update100_replay_mismatch_autopsy_v1"

HIST_CK = ROOT / "sf2_kl_parent_retention_run_v2" / "run" / "checkpoint_100.pt"
SF3_CK = ROOT / "sf3_prospective_treatment_v1" / "run" / "checkpoint_100.pt"
A_CK = AUT / "replay_A" / "checkpoint_100.pt"
B_CK = AUT / "replay_B" / "checkpoint_100.pt"

HIST_MET = ROOT / "sf2_kl_parent_retention_run_v2" / "run" / "TRAIN_METRICS.jsonl"
SF3_MET = ROOT / "sf3_prospective_treatment_v1" / "run" / "TRAIN_METRICS.jsonl"
A_MET = AUT / "replay_A" / "TRAIN_METRICS.jsonl"
B_MET = AUT / "replay_B" / "TRAIN_METRICS.jsonl"

HIST_RESTART = ROOT / "sf2_kl_parent_retention_run_v2" / "run" / "restart.pt"
SF3_RESTART = ROOT / "sf3_prospective_treatment_v1" / "run" / "restart.pt"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def component(name: str) -> str:
    if name.startswith("base_model.token_embedding"): return "token_embedding"
    if name.startswith("base_model.position_embedding"): return "position_embedding"
    if name.startswith("base_model.blocks."):
        idx = int(name.split(".")[2])
        return f"blocks{idx}"
    if name.startswith("base_model.final_norm"): return "final_norm"
    if name.startswith("base_model.language_head"): return "language_head"
    if name.startswith(("localizer.", "wq.", "wk.", "wv.", "wo.")): return "t13_special"
    return "other"


def load_state(p):
    return torch.load(p, map_location="cpu", weights_only=False)["model_state_dict"]


def cmp_pair(a_path, b_path, label):
    sa, sb = load_state(a_path), load_state(b_path)
    assert set(sa) == set(sb)
    per = {}
    n_mismatch = 0
    max_abs = 0.0
    mean_abs = 0.0
    rms = 0.0
    tot_el = 0
    sq = 0.0
    for k in sa:
        d = (sa[k].float() - sb[k].float()).abs()
        if d.numel():
            m = float(d.max())
            max_abs = max(max_abs, m)
            mean_abs += float(d.sum())
            sq += float((d ** 2).sum())
            tot_el += d.numel()
            comp = component(k)
            per.setdefault(comp, {"n": 0, "mismatch": 0, "max_abs": 0.0})
            per[comp]["n"] += 1
            if not torch.equal(sa[k], sb[k]):
                per[comp]["mismatch"] += 1
                n_mismatch += 1
            per[comp]["max_abs"] = max(per[comp]["max_abs"], m)
    return {
        "label": label, "a": str(a_path), "b": str(b_path),
        "a_sha256": sha(a_path), "b_sha256": sha(b_path),
        "tensor_count": len(sa), "mismatched_tensors": n_mismatch,
        "bitwise_equal": n_mismatch == 0,
        "max_abs": max_abs, "mean_abs": mean_abs / tot_el if tot_el else None,
        "rms": math.sqrt(sq / tot_el) if tot_el else None,
        "per_component": {k: {"n_tensors": v["n"], "mismatched": v["mismatch"], "max_abs": v["max_abs"]}
                          for k, v in sorted(per.items())},
    }


def cmp_metrics(a_path, b_path, label):
    ra = [json.loads(l) for l in a_path.read_text(encoding="utf-8").splitlines()]
    rb = [json.loads(l) for l in b_path.read_text(encoding="utf-8").splitlines()]
    n = min(len(ra), len(rb))
    fields = ["loss", "grad_norm", "ce", "kl"]
    first = None
    diff_count = 0
    max_diffs = {}
    for i in range(n):
        x, y = ra[i], rb[i]
        for f in fields:
            a = x.get(f); b = y.get(f)
            if a is None and b is None:
                continue
            if a is None or b is None or a != b:
                d = abs(a - b) if (a is not None and b is not None) else None
                max_diffs[f] = max(max_diffs.get(f, 0.0), d if d is not None else 0.0)
                diff_count += 1
                if first is None:
                    first = {"update": x["update"], "kind": x["kind"], "field": f,
                             "a": a, "b": b, "delta": d}
    return {"label": label, "a_records": len(ra), "b_records": len(rb), "compared": n,
            "diff_record_count": diff_count,
            "first_divergence": first, "max_abs_delta_per_field": max_diffs}


def inventory(ck, restart, label):
    c = torch.load(ck, map_location="cpu", weights_only=False)
    r = torch.load(restart, map_location="cpu", weights_only=False)
    return {
        "label": label,
        "checkpoint_top_keys": sorted(c.keys()),
        "checkpoint_has_model_state": "model_state_dict" in c,
        "checkpoint_has_optimizer": "optimizer" in c,
        "checkpoint_has_rng": "rng" in c,
        "checkpoint_update_field": c.get("update"),
        "restart_top_keys": sorted(r.keys()),
        "restart_completed": r.get("completed"),
        "restart_has_model": "model" in r, "restart_has_optimizer": "optimizer" in r,
        "restart_has_rng": "rng" in r,
    }


out = {
    "checkpoint_pairs": [
        cmp_pair(HIST_CK, A_CK, "historical_sf2_u100_vs_replayA_u100"),
        cmp_pair(HIST_CK, B_CK, "historical_sf2_u100_vs_replayB_u100"),
        cmp_pair(HIST_CK, SF3_CK, "historical_sf2_u100_vs_sf3replay_u100"),
        cmp_pair(A_CK, B_CK, "replayA_vs_replayB"),
        cmp_pair(A_CK, SF3_CK, "replayA_vs_sf3replay"),
    ],
    "metric_pairs": [
        cmp_metrics(HIST_MET, A_MET, "historical_vs_replayA"),
        cmp_metrics(A_MET, B_MET, "replayA_vs_replayB"),
        cmp_metrics(HIST_MET, SF3_MET, "historical_vs_sf3replay"),
    ],
    "state_inventory": [
        inventory(HIST_CK, HIST_RESTART, "historical_sf2"),
        inventory(SF3_CK, SF3_RESTART, "sf3_replay"),
    ],
    "checkpoint_sha256s": {
        "historical_sf2_u100": sha(HIST_CK),
        "sf3_replay_u100": sha(SF3_CK),
        "replayA_u100": sha(A_CK),
        "replayB_u100": sha(B_CK),
    },
}
(AUT / "ANALYSIS_RAW.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                                      encoding="utf-8", newline="\n")
print(json.dumps({"checkpoint_sha256s": out["checkpoint_sha256s"]}, indent=2))
for p in out["checkpoint_pairs"]:
    print(p["label"], "bitwise_equal=", p["bitwise_equal"], "mismatched=", p["mismatched_tensors"],
          "max_abs=", p["max_abs"], "mean_abs=", p["mean_abs"], "rms=", p["rms"])
    print("   per_component:", json.dumps(p["per_component"]))
for m in out["metric_pairs"]:
    print(m["label"], "diff_records=", m["diff_record_count"], "first=", json.dumps(m["first_divergence"]),
          "max_deltas=", json.dumps(m["max_abs_delta_per_field"]))
for s in out["state_inventory"]:
    print(s["label"], json.dumps(s))
