"""Diagnostic: is train-mode (dropout) forward reproducible after RNG restore?
If eval-mode is reproducible but train-mode is not, the dropout RNG stream is not
being restored by capture/restore_rng on this ROCm/torch stack.
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

    def train_ce():
        m.train()
        lg = m.base_model(x.to(dev))
        return float(F.cross_entropy(lg.reshape(-1, 1024), y.to(dev).reshape(-1), ignore_index=-100))

    vals_restored = []
    for _ in range(4):
        m.load_state_dict(snapshot["model"])
        E.rt.restore_rng(snapshot["rng"])
        vals_restored.append(train_ce())
    # consecutive train-mode forwards without restore (masks advance; expected to differ)
    vals_consec = []
    for _ in range(3):
        vals_consec.append(train_ce())
    out = {"train_ce_after_restore_x4": vals_restored,
           "train_ce_consecutive_no_restore_x3": vals_consec}
    (ROOT / "diagnostics" / "TRAINMODE_LOCALIZATION.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
