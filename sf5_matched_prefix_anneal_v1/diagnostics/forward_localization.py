"""Diagnostic: localize the within-process non-reproducibility.
Q1: repeated eval-mode forward on identical weights in one process - reproducible?
Q2: after full restore (model/opt/RNG), is the first English update's CE forward
    reproducible?  Q3: is the KL forward reproducible?
Writes to diagnostics/ only.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

ROOT = Path(r"C:\DaveLM-CADAVER\sf5_matched_prefix_anneal_v1")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "sources"))
import torch
import torch.nn.functional as F
import SF5_ENGINE as S
import SF2_ENGINE as E
from PINNED_MASKING import set_scope, pad_batch

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
    u13 = schedule[N_PREFIX]  # update 13, english
    batch = [rows_idx[i] for i in u13["ids"]]
    x, y = pad_batch(batch, u13["pad"])

    def ce_forward():
        m.eval()
        with torch.no_grad():
            lg = m.base_model(x.to(dev))
        return float(F.cross_entropy(lg.reshape(-1, 1024), y.to(dev).reshape(-1), ignore_index=-100))

    def kl_forward():
        m.eval()
        sel = E.pick_kl_entries(kl_pool["entries"], snapshot["english_index"] + 1)
        with torch.no_grad():
            v = E.compute_kl(m, teacher, dev, kl_pool["rows"], sel)
        return float(v)

    # Q1: same eval forward repeated in-process, no restore in between
    q1 = [ce_forward() for _ in range(3)]
    # Q2/Q3: restore then forward, repeated
    q2 = []
    q3 = []
    for _ in range(3):
        m.load_state_dict(snapshot["model"])
        E.rt.restore_rng(snapshot["rng"])
        m.eval()
        q2.append(ce_forward())
        q3.append(kl_forward())
    out = {"q1_repeat_eval_ce_no_restore": q1,
           "q2_eval_ce_after_restore": q2,
           "q3_kl_after_restore": q3}
    (ROOT / "diagnostics" / "FORWARD_LOCALIZATION.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
