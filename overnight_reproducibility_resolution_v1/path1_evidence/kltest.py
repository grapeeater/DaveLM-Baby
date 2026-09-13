"""kltest: two-process micro-test. Runs update 1 identically, then computes the
update-2 KL forward THREE times on identical inputs, printing exact bits of KL and CE.
Determines (a) within-process determinism of the KL op, (b) cross-process mode
dependence of the fp64 KL reduction vs the fp32 CE.
"""
from __future__ import annotations
import json, os, struct, sys
from pathlib import Path

V2 = Path(r"C:\DaveLM-CADAVER\sf2_kl_parent_retention_run_v2")
sys.path.insert(0, str(V2)); sys.path.insert(0, str(V2 / "sources")); sys.path.insert(0, r"C:\DaveLM-v0.9")
import torch
import torch.nn.functional as F
import SF2_ENGINE as E
from PINNED_MASKING import set_scope, pad_batch

BUNDLE = Path(r"C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011")

def bits(x):
    return struct.pack('<f', float(x)).hex()

def main():
    p = E.read(BUNDLE / "PROTOCOL.json")
    E.verify(BUNDLE)
    E.rt.configure_runtime(p["seed"])
    import platform, tokenizers
    assert platform.python_version() == "3.12.14" and tokenizers.__version__ == "0.23.1"
    schedule = E.read(BUNDLE / "SCHEDULE.json")
    train = E.read(BUNDLE / "TRAIN.json")
    idx = {r["id"]: r for r in train}
    kl_pool, proto = E.load_kl_pool()
    KL_ROWS, KL_ENTRIES = kl_pool["rows"], kl_pool["entries"]
    device = torch.device("cuda")
    m = E.rt.load_model(Path(p["parent"]), device, BUNDLE)
    teacher = E.build_teacher(Path(p["parent"]), device)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-5, betas=(.9, .999), eps=1e-8, weight_decay=.05,
                            amsgrad=False, foreach=False, fused=False)
    # run exactly update 1 (english)
    u = schedule[0]
    assert u["kind"] == "english"
    set_scope(m, False)
    m.eval()
    sel = E.pick_kl_entries(KL_ENTRIES, 1)
    kl1 = E.compute_kl(m, teacher, device, KL_ROWS, sel)
    m.train()
    batch = [idx[i] for i in u["ids"]]
    x, y = pad_batch(batch, u["pad"])
    x = x.to(device); y = y.to(device)
    ce1 = F.cross_entropy(m.base_model(x).reshape(-1, 1024), y.reshape(-1), ignore_index=-100)
    loss = ce1 + kl1
    loss.backward()
    torch.nn.utils.clip_grad_norm_([q for q in m.parameters() if q.grad is not None], 2.0)
    opt.step()
    # now update-2 forward on identical weights, KL computed 3x + CE 1x
    u2 = schedule[1]
    assert u2["kind"] == "english"
    set_scope(m, False)
    m.eval()
    sel2 = E.pick_kl_entries(KL_ENTRIES, 2)
    kl_reps = []
    for _ in range(3):
        kl_reps.append(bits(E.compute_kl(m, teacher, device, KL_ROWS, sel2)))
    m.train()
    batch2 = [idx[i] for i in u2["ids"]]
    x2, y2 = pad_batch(batch2, u2["pad"])
    x2 = x2.to(device); y2 = y2.to(device)
    ce2 = bits(F.cross_entropy(m.base_model(x2).reshape(-1, 1024), y2.reshape(-1), ignore_index=-100))
    print(json.dumps({"kl1_bits": bits(kl1), "ce1_bits": bits(ce1),
                      "u2_kl_3reps_bits": kl_reps, "u2_ce_bits": ce2}))
if __name__ == "__main__":
    main()
