"""probe8g: like probe8 but records per-group gradient fingerprints at update 1.

Diagnostic only. Helps identify which gradient/op differs between trajectory modes.
"""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path

V2 = Path(r"C:\DaveLM-CADAVER\sf2_kl_parent_retention_run_v2")
sys.path.insert(0, str(V2))
sys.path.insert(0, str(V2 / "sources"))
sys.path.insert(0, r"C:\DaveLM-v0.9")

import torch
import torch.nn.functional as F

import SF2_ENGINE as E
from PINNED_MASKING import set_scope, pad_batch

BUNDLE = Path(r"C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011")
N_UPDATES = 8

def gfp(t):
    h = hashlib.sha256()
    h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def group(name):
    if name.startswith("base_model.token_embedding"): return "token_embd"
    if name.startswith("base_model.position_embedding"): return "pos_embd"
    if name.startswith("base_model.blocks."):
        return f"block{name.split('.')[2]}"
    if name.startswith("base_model.final_norm"): return "final_norm"
    if name.startswith("base_model.language_head"): return "head"
    return "t13"

def main():
    p = E.read(BUNDLE / "PROTOCOL.json")
    E.verify(BUNDLE)
    E.rt.configure_runtime(p["seed"])
    import platform, tokenizers
    assert platform.python_version() == "3.12.14" and tokenizers.__version__ == "0.23.1"
    schedule = E.read(BUNDLE / "SCHEDULE.json")
    train = E.read(BUNDLE / "TRAIN.json")
    idx = {r["id"]: r for r in train}
    pool = E.read(BUNDLE / "data/binding_rehearsal.json")["quartets"]
    kl_pool, proto = E.load_kl_pool()
    KL_ROWS, KL_ENTRIES = kl_pool["rows"], kl_pool["entries"]
    device = torch.device("cuda")
    m = E.rt.load_model(Path(p["parent"]), device, BUNDLE)
    teacher = E.build_teacher(Path(p["parent"]), device)
    E.assert_teacher_disjoint(m.base_model, teacher)
    pin = E.rt.pinned_binding(BUNDLE)
    opt = torch.optim.AdamW(m.parameters(), lr=5e-5, betas=(.9, .999), eps=1e-8, weight_decay=.05,
                            amsgrad=False, foreach=False, fused=False)
    english_index = 0
    metrics = []
    grad_fp_u1 = None
    for u in schedule[:N_UPDATES]:
        opt.zero_grad(set_to_none=True)
        set_scope(m, u["kind"] == "binding")
        if u["kind"] == "english":
            english_index += 1
            m.eval()
            sel = E.pick_kl_entries(KL_ENTRIES, english_index)
            kl = E.compute_kl(m, teacher, device, KL_ROWS, sel)
            m.train()
            batch = [idx[i] for i in u["ids"]]
            x, y = pad_batch(batch, u["pad"])
            x = x.to(device); y = y.to(device)
            ce = F.cross_entropy(m.base_model(x).reshape(-1, 1024), y.reshape(-1), ignore_index=-100)
            loss = ce + 1.0 * kl
        else:
            m.train()
            loss, _ = E.rt.binding_loss(m, E.rt.binding_docs_for_batch(pool, u["quartets"], u["documents"]), device, pin)
        loss.backward()
        if u["update"] == 1:
            g = {}
            for n, prm in m.named_parameters():
                if prm.grad is not None:
                    g.setdefault(group(n), []).append(prm.grad.detach())
            grad_fp_u1 = {k: gfp(torch.cat([x.reshape(-1) for x in v])) for k, v in g.items()}
        norm = torch.nn.utils.clip_grad_norm_([x for x in m.parameters() if x.grad is not None], 2.0)
        opt.step()
        rec = {"update": u["update"], "kind": u["kind"], "loss": float(loss.detach()), "grad_norm": float(norm)}
        if u["kind"] == "english":
            rec["ce"] = float(ce.detach()); rec["kl"] = float(kl.detach())
        metrics.append(rec)
    h = hashlib.sha256()
    for name, t in m.state_dict().items():
        if "mask" in name:
            continue
        h.update(name.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    out = {"updates": N_UPDATES,
           "env": {k: v for k, v in os.environ.items() if k in ("TORCH_BLAS_PREFER_HIPBLASLT", "HIPBLASLT_AUTOTUNE")},
           "weight_fingerprint_sha256": h.hexdigest(),
           "update1_group_grad_fingerprints": grad_fp_u1,
           "metrics": metrics}
    print(json.dumps(out))

if __name__ == "__main__":
    main()
