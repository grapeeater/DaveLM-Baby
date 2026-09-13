"""Diagnostic bisection: replicate withinproc trial exactly but only ONE update
per trial, 6 trials, to find which element (opt.load_state_dict / set_scope /
run_one_update sequence) triggers divergence in u13 CE.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(r"C:\DaveLM-CADAVER\sf5_matched_prefix_anneal_v1")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "sources"))
import torch
import SF5_ENGINE as S
import SF2_ENGINE as E

BUNDLE = Path(r"C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011")
SEED = 87999
N_PREFIX = 12


def main():
    p = E.read(BUNDLE / "PROTOCOL.json")
    E.verify(BUNDLE)
    E.rt.configure_runtime(SEED)
    dev = torch.device("cuda")
    rows = E.read(BUNDLE / "TRAIN.json")
    rows_idx = {r["id"]: r for r in rows}
    schedule = E.read(BUNDLE / "SCHEDULE.json")
    binding_pool = E.read(BUNDLE / "data/binding_rehearsal.json")["quartets"]
    kl_pool, _ = E.load_kl_pool()
    S.kl_pool_entries = kl_pool["entries"]; S.kl_pool_rows = kl_pool["rows"]
    m = E.rt.load_model(Path(p["parent"]), dev, BUNDLE)
    teacher = E.build_teacher(Path(p["parent"]), dev)
    pin = E.rt.pinned_binding(BUNDLE)
    opt = S.make_optimizer(m)
    eng = 0
    for u in schedule[:N_PREFIX]:
        loss, ce, kl, norm, eng = S.run_one_update(m, opt, teacher, dev, u, rows_idx, binding_pool, pin, eng, S.BASE_LR)
    snapshot = {"model": m.state_dict(), "optimizer": opt.state_dict(), "rng": E.rt.capture_rng(), "english_index": eng}
    u13 = schedule[N_PREFIX]
    from PINNED_MASKING import set_scope

    def trial(with_opt_load):
        m.load_state_dict(snapshot["model"])
        if with_opt_load:
            opt.load_state_dict(snapshot["optimizer"])
        E.rt.restore_rng(snapshot["rng"])
        set_scope(m, False)
        loss, ce, kl, norm, eng2 = S.run_one_update(m, opt, teacher, dev, u13, rows_idx, binding_pool, pin,
                                                    snapshot["english_index"], S.BASE_LR)
        return {"ce": ce, "kl": kl, "loss": loss}

    A = [trial(True) for _ in range(6)]   # exact withinproc replica (opt load)
    B = [trial(False) for _ in range(6)]  # no opt load
    out = {"A_with_opt_load": A, "B_without_opt_load": B}
    (ROOT / "diagnostics" / "BISECT.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
