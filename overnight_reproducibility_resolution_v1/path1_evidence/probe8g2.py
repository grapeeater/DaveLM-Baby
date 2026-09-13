"""probe8g2: capture exact clip-grad-norm bits at updates 1-3 across modes."""
from __future__ import annotations
import hashlib, json, os, struct, sys
from pathlib import Path

V2 = Path(r"C:\DaveLM-CADAVER\sf2_kl_parent_retention_run_v2")
sys.path.insert(0, str(V2)); sys.path.insert(0, str(V2 / "sources")); sys.path.insert(0, r"C:\DaveLM-v0.9")
import torch
import torch.nn.functional as F
import SF2_ENGINE as E
from PINNED_MASKING import set_scope, pad_batch

BUNDLE = Path(r"C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011")
N_UPDATES = 6

def f32bits(x):
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
    normbits = {}
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
        if u["update"] <= 3:
            grads = [x.grad.detach() for x in m.parameters() if x.grad is not None]
            with torch.no_grad():
                total = torch.cat([g.reshape(-1) for g in grads]).norm().float()
            normbits[u["update"]] = {"torch_cat_norm_f32_bits": f32bits(total)}
        norm = torch.nn.utils.clip_grad_norm_([x for x in m.parameters() if x.grad is not None], 2.0)
        if u["update"] <= 3:
            normbits[u["update"]]["clip_norm_f32_bits"] = f32bits(norm)
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
    print(json.dumps({"updates": N_UPDATES, "weight_fingerprint_sha256": h.hexdigest(),
                      "norm_bits_updates_1_3": normbits, "metrics": metrics}))
if __name__ == "__main__":
    main()
