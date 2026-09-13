"""Diagnostic: which operation between restore and train-CE breaks reproducibility?
T1: restore -> trainCE
T2: restore -> KL(eval, grad enabled) -> trainCE
T3: restore -> KL(eval, no_grad) -> trainCE
Each repeated 4x with fresh restore.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(r"C:\DaveLM-CADAVER\sf5_matched_prefix_anneal_v1")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "sources"))
import torch
import torch.nn.functional as F
import SF5_ENGINE as S
import SF2_ENGINE as E
from PINNED_MASKING import pad_batch

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
    batch = [rows_idx[i] for i in u13["ids"]]
    x, y = pad_batch(batch, u13["pad"])
    sel = E.pick_kl_entries(kl_pool["entries"], eng + 1)

    def trainCE():
        m.train()
        lg = m.base_model(x.to(dev))
        return float(F.cross_entropy(lg.reshape(-1, 1024), y.to(dev).reshape(-1), ignore_index=-100))

    def kl_grad():
        m.eval()
        return float(E.compute_kl(m, teacher, dev, kl_pool["rows"], sel))

    def kl_nograd():
        m.eval()
        with torch.no_grad():
            return float(E.compute_kl(m, teacher, dev, kl_pool["rows"], sel))

    def fresh():
        m.load_state_dict(snapshot["model"])
        E.rt.restore_rng(snapshot["rng"])

    T1, T2, T3 = [], [], []
    for _ in range(4):
        fresh(); T1.append(trainCE())
        fresh(); T2.append((kl_grad(), trainCE()))
        fresh(); T3.append((kl_nograd(), trainCE()))
    out = {"T1_restore_trainCE": T1, "T2_restore_KLgrad_trainCE": T2, "T3_restore_KLnograd_trainCE": T3}
    (ROOT / "diagnostics" / "KL_PERTURBATION.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
