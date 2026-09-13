"""Diagnostic (NOT part of the frozen SF5 study): characterize within-process
repeatability of snapshot->restore->continuation. Runs prefix of 12 updates,
snapshot, then three identical 3-update continuations (C1, C2, C3), recording
per-update loss/ce/kl and final model digests, to determine whether divergence
between identical in-process continuations is systematic or a rare event.
Read-only with respect to authoritative artifacts; writes only to this package's
diagnostics/ directory.
"""
from __future__ import annotations
import json, os, sys, hashlib
from pathlib import Path

ROOT = Path(r"C:\DaveLM-CADAVER\sf5_matched_prefix_anneal_v1")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sources"))
import torch
import SF5_ENGINE as S
import SF2_ENGINE as E
from PINNED_MASKING import set_scope

BUNDLE = Path(r"C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011")
SEED = 87999
N_PREFIX = 12
N_ARM = 3


def main():
    p = E.read(BUNDLE / "PROTOCOL.json")
    E.verify(BUNDLE)
    E.rt.configure_runtime(SEED)
    dev = torch.device("cuda")
    rows = E.read(BUNDLE / "TRAIN.json")
    rows_idx = {r["id"]: r for r in rows}
    schedule = E.read(BUNDLE / "SCHEDULE.json")
    binding_pool = E.read(BUNDLE / "data/binding_rehearsal.json")["quartets"]
    tok = E.Tokenizer.from_file(p["tokenizer"])
    kl_pool, _ = E.load_kl_pool()
    m = E.rt.load_model(Path(p["parent"]), dev, BUNDLE)
    teacher = E.build_teacher(Path(p["parent"]), dev)
    pin = E.rt.pinned_binding(BUNDLE)
    opt = S.make_optimizer(m)
    out = ROOT / "diagnostics"
    out.mkdir(exist_ok=True)
    S.kl_pool_entries = kl_pool["entries"]
    S.kl_pool_rows = kl_pool["rows"]
    eng = 0
    for u in schedule[:N_PREFIX]:
        loss, ce, kl, norm, eng = S.run_one_update(m, opt, teacher, dev, u, rows_idx, binding_pool, pin, eng, S.BASE_LR)
    snapshot = {"completed": N_PREFIX, "model": m.state_dict(), "optimizer": opt.state_dict(),
                "rng": E.rt.capture_rng(), "english_index": eng}
    trials = {}
    for name in ("C1", "C2", "C3"):
        m.load_state_dict(snapshot["model"])
        opt.load_state_dict(snapshot["optimizer"])
        E.rt.restore_rng(snapshot["rng"])
        set_scope(m, False)
        e2 = snapshot["english_index"]
        recs = []
        for u in schedule[N_PREFIX:N_PREFIX + N_ARM]:
            loss, ce, kl, norm, e2 = S.run_one_update(m, opt, teacher, dev, u, rows_idx, binding_pool, pin, e2, S.BASE_LR)
            recs.append({"update": u["update"], "loss": loss, "ce": ce, "kl": kl, "grad_norm": norm})
        trials[name] = {"model": S.digest(m.state_dict()), "optimizer": S.digest(opt.state_dict()),
                        "rng": S.digest(E.rt.capture_rng()), "english_index": e2, "metrics": recs}
    result = {
        "C1_model": trials["C1"]["model"], "C2_model": trials["C2"]["model"], "C3_model": trials["C3"]["model"],
        "C1_eq_C2": trials["C1"]["model"] == trials["C2"]["model"],
        "C1_eq_C3": trials["C1"]["model"] == trials["C3"]["model"],
        "metrics": {k: v["metrics"] for k, v in trials.items()},
    }
    (out / "WITHIN_PROCESS_DIAG.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"C1_eq_C2": result["C1_eq_C2"], "C1_eq_C3": result["C1_eq_C3"],
                      "C1": trials["C1"]["model"][:16], "C2": trials["C2"]["model"][:16],
                      "C3": trials["C3"]["model"][:16]}, indent=2))


if __name__ == "__main__":
    main()
