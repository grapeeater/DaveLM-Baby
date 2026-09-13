"""Diagnostic: isolate which op between restore and train-CE breaks reproducibility.
V0 restore->CE; V1 +zero_grad; V2 +set_scope; V3 +inactive clone; V4 all three.
Each variant repeated 4x with fresh restore.
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
from PINNED_MASKING import pad_batch, set_scope

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

    def trainCE():
        m.train()
        lg = m.base_model(x.to(dev))
        return float(F.cross_entropy(lg.reshape(-1, 1024), y.to(dev).reshape(-1), ignore_index=-100).detach())

    def fresh():
        m.load_state_dict(snapshot["model"])
        E.rt.restore_rng(snapshot["rng"])

    results = {}
    variants = {
        "V0_bare": lambda: None,
        "V1_zero_grad": lambda: opt.zero_grad(set_to_none=True),
        "V2_set_scope": lambda: set_scope(m, False),
        "V3_inactive_clone": lambda: [x.detach().clone() for n, x in m.named_parameters() if not x.requires_grad],
        "V4_all3": lambda: (opt.zero_grad(set_to_none=True), set_scope(m, False),
                            [x.detach().clone() for n, x in m.named_parameters() if not x.requires_grad]),
    }
    for vname, vfn in variants.items():
        vals = []
        for _ in range(4):
            fresh()
            vfn()
            vals.append(trainCE())
        results[vname] = vals
    (ROOT / "diagnostics" / "TRIGGER.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
