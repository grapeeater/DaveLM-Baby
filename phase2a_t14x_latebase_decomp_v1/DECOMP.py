"""Read-only late-base failure decomposition after T14.

Does not train. Does not load T3 TEST. Does not write T14 runners/checkpoints.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(r"C:\DaveLM-CADAVER")
OUT = ROOT / "phase2a_t14x_latebase_decomp_v1"
T14B = ROOT / "phase2a_t14_factclause_pointer_lr3p75e5_750_v1"
sys.path.insert(0, str(T14B))
sys.path.insert(0, str(ROOT / "baby_vnext_60m_design_v1"))

from T14_RUNNER import (  # noqa: E402
    ARCH, CTX, LAMBDA_PTR, LANG_DEV, LANG_STARTS, LANG_TRAIN, PARENT, PARENT_SHA,
    T3_DATA, TOK_PATH, assignment_reversals, complete_families, fact_clause_spans_bos,
    freeze_early, lang_ce, load_jsonl, load_model, pointer_and_margin, read_u16,
    read_u32, set_scope, subset_eval, write_json,
)

CKPTS = {
    "t10_690003_u500": ROOT / "phase2a_t10_factclause_pointer_1k_v1_run_seed690003" / "checkpoints" / "checkpoint_0500.pt",
    "t10_690003_u750": ROOT / "phase2a_t10_factclause_pointer_1k_v1_run_seed690003" / "checkpoints" / "checkpoint_0750.pt",
    "t12_710001_u500": ROOT / "phase2a_t12_factclause_pointer_lr2p5e5_750_v1_run_seed710001" / "checkpoints" / "checkpoint_0500.pt",
    "t12_710001_u750": ROOT / "phase2a_t12_factclause_pointer_lr2p5e5_750_v1_run_seed710001" / "checkpoints" / "checkpoint_0750.pt",
    "t13_720002_u750": ROOT / "phase2a_t13_factclause_pointer_lr2p5e5_1k_v1_run_seed720002" / "checkpoints" / "checkpoint_0750.pt",
    "t13_720003_u750": ROOT / "phase2a_t13_factclause_pointer_lr2p5e5_1k_v1_run_seed720003" / "checkpoints" / "checkpoint_0750.pt",
    "t13_720003_u1000": ROOT / "phase2a_t13_factclause_pointer_lr2p5e5_1k_v1_run_seed720003" / "checkpoints" / "checkpoint_1000.pt",
    "t14_730001_u500": ROOT / "phase2a_t14_factclause_pointer_lr3p75e5_750_v1_run_seed730001" / "checkpoints" / "checkpoint_0500.pt",
    "t14_730001_u750": ROOT / "phase2a_t14_factclause_pointer_lr3p75e5_750_v1_run_seed730001" / "checkpoints" / "checkpoint_0750.pt",
    "t14_730002_u500": ROOT / "phase2a_t14_factclause_pointer_lr3p75e5_750_v1_run_seed730002" / "checkpoints" / "checkpoint_0500.pt",
    "t14_730002_u750": ROOT / "phase2a_t14_factclause_pointer_lr3p75e5_750_v1_run_seed730002" / "checkpoints" / "checkpoint_0750.pt",
    "t14_730003_u750": ROOT / "phase2a_t14_factclause_pointer_lr3p75e5_750_v1_run_seed730003" / "checkpoints" / "checkpoint_0750.pt",
}


def module_key(name: str) -> str:
    if not name.startswith("base_model."):
        return "binding_frozen"
    if name.startswith("base_model.blocks."):
        parts = name.split(".")
        i = int(parts[2])
        kind = parts[3]
        if i < 4:
            return f"block{i}_frozen"
        if kind == "attention":
            if "qkv" in name:
                return f"block{i}.attn.qkv"
            if "projection" in name:
                return f"block{i}.attn.proj"
            return f"block{i}.attn.other"
        if kind == "feed_forward":
            return f"block{i}.mlp"
        if kind.startswith("norm"):
            return f"block{i}.norm"
        return f"block{i}.other"
    if "token_embedding" in name:
        return "embed.token"
    if "position_embedding" in name:
        return "embed.pos"
    if "final_norm" in name:
        return "final_norm"
    if "language_head" in name:
        return "language_head"
    return "base_other"


def coarse_key(name: str) -> str:
    k = module_key(name)
    if k.startswith("block") and "_frozen" not in k:
        i = k.split(".")[0].replace("block", "")
        if ".attn" in k:
            return f"block{i}.attn"
        if ".mlp" in k:
            return f"block{i}.mlp"
        if ".norm" in k:
            return f"block{i}.norm"
    return k


def load_sd(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["model_state_dict"]


def rms(t: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(t.float() ** 2)))


def drift_report(parent_sd: dict) -> dict:
    rows = {}
    for label, path in CKPTS.items():
        if not path.is_file():
            rows[label] = {"missing": True, "path": str(path)}
            continue
        sd = load_sd(path)
        by = defaultdict(lambda: {"sq": 0.0, "n": 0})
        coarse = defaultdict(lambda: {"sq": 0.0, "n": 0})
        for n, p in parent_sd.items():
            if n not in sd:
                continue
            d = (sd[n].float() - p.float()).reshape(-1)
            rec = by[module_key(n)]
            rec["sq"] += float(torch.dot(d, d))
            rec["n"] += d.numel()
            cr = coarse[coarse_key(n)]
            cr["sq"] += float(torch.dot(d, d))
            cr["n"] += d.numel()
        def pack(src):
            out = {}
            for k, v in src.items():
                out[k] = {"rms": math.sqrt(v["sq"] / v["n"]) if v["n"] else 0.0, "n": v["n"]}
            return dict(sorted(out.items(), key=lambda kv: -kv[1]["rms"]))
        rows[label] = {"modules": pack(by), "coarse": pack(coarse), "path": str(path)}
        print(f"DRIFT {label} top3={list(rows[label]['coarse'].items())[:3]}", flush=True)
        del sd
    return rows


@torch.no_grad()
def eval_fast(model, dev_rows, nd_ids, td_ids, stream_dev, stream_train, dev_starts, train_fit, device):
    was = model.training
    model.eval()
    per_row = []
    try:
        for r in dev_rows:
            ptr, mar, scores, sims = pointer_and_margin(model, r, device, False)
            ci = int(r["correct_index"])
            pred_ptr = int(torch.argmax(sims).item())
            pred_fc = int(max(range(len(scores)), key=lambda j: float(scores[j].detach())))
            per_row.append({
                "id": r["id"], "family_id": r["family_id"],
                "pointer_ok": pred_ptr == ci, "native_ok": pred_fc == ci,
                "exact_ok": False,
                "assignment": r["assignment"], "query_index": r["query_index"],
                "fact_order": r["fact_order"],
            })
    finally:
        model.train(was)
    dce = lang_ce(model, stream_dev, dev_starts, device, 16)
    tce = lang_ce(model, stream_train, train_fit, device, 16)
    nd_sub = [p for p in per_row if p["id"] in nd_ids]
    td_sub = [p for p in per_row if p["id"] in td_ids]
    return {
        "dev_ce": dce, "train_ce": tce, "gap": dce - tce,
        "pointer": sum(p["pointer_ok"] for p in per_row),
        "native": sum(p["native_ok"] for p in per_row),
        "rev": assignment_reversals(per_row)["pointer"],
        "fam": complete_families(per_row, "pointer_ok")["complete"],
        "nd": sum(p["pointer_ok"] for p in nd_sub),
        "td": sum(p["pointer_ok"] for p in td_sub),
        "n": len(per_row),
    }


def collect_grads(model, loss_fn, device):
    model.zero_grad(set_to_none=True)
    loss = loss_fn()
    loss.backward()
    grads = {}
    for n, p in model.named_parameters():
        if p.grad is None or not p.requires_grad:
            continue
        grads[n] = p.grad.detach().float().cpu().reshape(-1).clone()
    model.zero_grad(set_to_none=True)
    return float(loss.detach().cpu()), grads


def bucket_cos(a: dict, b: dict) -> dict:
    keys_a = defaultdict(list)
    keys_b = defaultdict(list)
    for n, g in a.items():
        keys_a[coarse_key(n)].append(g)
        keys_a[module_key(n)].append(g)
    for n, g in b.items():
        keys_b[coarse_key(n)].append(g)
        keys_b[module_key(n)].append(g)
    out = {}
    for k in set(keys_a) | set(keys_b):
        if k not in keys_a or k not in keys_b:
            out[k] = None
            continue
        va = torch.cat(keys_a[k])
        vb = torch.cat(keys_b[k])
        na = float(torch.linalg.vector_norm(va))
        nb = float(torch.linalg.vector_norm(vb))
        if na == 0.0 or nb == 0.0:
            out[k] = None
        else:
            out[k] = float(torch.dot(va, vb) / (na * nb))
    return dict(sorted(out.items(), key=lambda kv: -999 if kv[1] is None else kv[1]))


def revert_names(parent_sd, model_sd, predicate):
    patched = dict(model_sd)
    n = 0
    for k, v in parent_sd.items():
        if predicate(k):
            patched[k] = v
            n += 1
    return patched, n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    seal = json.loads((T3_DATA / "TEST_SEAL.json").read_text(encoding="utf-8"))
    if seal["status"] != "SEALED_UNOPENED":
        raise RuntimeError("TEST seal unexpected")
    model, parent_sd = load_model()
    skip_early = (OUT / "DRIFT.json").is_file() and (OUT / "GRADS.json").is_file()
    if not skip_early:
        write_json(OUT / "STATUS.json", {"status": "DRIFT", "test_loaded": False})
        drift = drift_report(parent_sd)
        write_json(OUT / "DRIFT.json", drift)
    else:
        print("SKIP drift/grads; files present", flush=True)

    train_rows = load_jsonl(T3_DATA / "qa_train.jsonl")
    dev_rows = load_jsonl(T3_DATA / "qa_dev.jsonl")
    panels = json.loads((T3_DATA / "panels.json").read_text(encoding="utf-8"))
    nd_ids = set(panels["name_disjoint_ids"])
    td_ids = set(panels["template_disjoint_ids"])
    schedules = json.loads((T14B / "data" / "qa_schedules.json").read_text(encoding="utf-8"))
    rehear = json.loads((T14B / "data" / "rehearsal_index.json").read_text(encoding="utf-8"))["starts_mod_320"]
    stream_train = read_u16(LANG_TRAIN)
    stream_dev = read_u16(LANG_DEV)
    starts_all = read_u32(LANG_STARTS, 1600)
    dev_starts = starts_all[:1280]
    train_fit = starts_all[1280:]

    set_scope(model)
    model.to(device)

    def qa_ptr_only(indices):
        loss = 0.0
        for j in indices[:8]:
            ptr, mar, _, _ = pointer_and_margin(model, train_rows[j], device, True)
            loss = loss + ptr / 8
        return loss

    def qa_mar_only(indices):
        loss = 0.0
        for j in indices[:8]:
            ptr, mar, _, _ = pointer_and_margin(model, train_rows[j], device, True)
            loss = loss + mar / 8
        return loss

    def qa_combo(indices):
        loss = 0.0
        for j in indices[:8]:
            ptr, mar, _, _ = pointer_and_margin(model, train_rows[j], device, True)
            loss = loss + (mar + LAMBDA_PTR * ptr) / 8
        return loss

    def lang_loss():
        mods = rehear[0]
        starts = train_fit[torch.tensor(mods[:16], dtype=torch.long)]
        off = torch.arange(CTX)
        x = stream_train[starts[:, None] + off[None, :]].to(device)
        y = stream_train[starts[:, None] + off[None, :] + 1].to(device)
        logits, _ = model(x)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

    if not skip_early:
        write_json(OUT / "STATUS.json", {"status": "GRADS", "test_loaded": False})
        grad_out = {}
        for tag, ck in (("730001_u750", CKPTS["t14_730001_u750"]),
                        ("730002_u750", CKPTS["t14_730002_u750"])):
            sd = load_sd(ck)
            model.load_state_dict(sd, strict=True)
            set_scope(model)
            model.to(device)
            qa_idx = schedules["730001"][10]
            lp, g_ptr = collect_grads(model, lambda: qa_ptr_only(qa_idx), device)
            lm, g_mar = collect_grads(model, lambda: qa_mar_only(qa_idx), device)
            lq, g_qa = collect_grads(model, lambda: qa_combo(qa_idx), device)
            ll, g_lang = collect_grads(model, lang_loss, device)
            rec = {
                "losses": {"pointer": lp, "margin": lm, "qa_combo": lq, "language": ll},
                "cos_ptr_vs_lang": bucket_cos(g_ptr, g_lang),
                "cos_mar_vs_lang": bucket_cos(g_mar, g_lang),
                "cos_qa_vs_lang": bucket_cos(g_qa, g_lang),
            }
            grad_out[tag] = rec
            print(f"GRADS {tag} losses={rec['losses']}", flush=True)
            write_json(OUT / "GRADS.json", grad_out)

    patches = [
        ("baseline", lambda n: False),
        ("revert_embed", lambda n: n.startswith("base_model.token_embedding") or n.startswith("base_model.position_embedding")),
        ("revert_language_head", lambda n: n.startswith("base_model.language_head")),
        ("revert_final_norm", lambda n: n.startswith("base_model.final_norm")),
        ("revert_attn_4_11", lambda n: n.startswith("base_model.blocks.") and int(n.split(".")[2]) >= 4 and ".attention." in n),
        ("revert_mlp_4_11", lambda n: n.startswith("base_model.blocks.") and int(n.split(".")[2]) >= 4 and ".feed_forward." in n),
        ("revert_blocks_4_7", lambda n: n.startswith("base_model.blocks.") and 4 <= int(n.split(".")[2]) <= 7),
        ("revert_blocks_8_11", lambda n: n.startswith("base_model.blocks.") and int(n.split(".")[2]) >= 8),
    ]
    for i in range(4, 12):
        patches.append((f"revert_block_{i}", lambda n, i=i: n.startswith(f"base_model.blocks.{i}.")))

    write_json(OUT / "STATUS.json", {"status": "REVERT", "test_loaded": False})
    revert_out = {}
    targets = (
        ("730001_u750", CKPTS["t14_730001_u750"], patches),
        ("730002_u750", CKPTS["t14_730002_u750"], [
            ("baseline", lambda n: False),
            ("revert_language_head", lambda n: n.startswith("base_model.language_head")),
            ("revert_attn_4_11", lambda n: n.startswith("base_model.blocks.") and int(n.split(".")[2]) >= 4 and ".attention." in n),
            ("revert_mlp_4_11", lambda n: n.startswith("base_model.blocks.") and int(n.split(".")[2]) >= 4 and ".feed_forward." in n),
            ("revert_blocks_8_11", lambda n: n.startswith("base_model.blocks.") and int(n.split(".")[2]) >= 8),
        ]),
    )
    for tag, ck, plist in targets:
        base_sd = load_sd(ck)
        revert_out[tag] = {}
        for name, pred in plist:
            patched, nrep = revert_names(parent_sd, base_sd, pred)
            model.load_state_dict(patched, strict=True)
            set_scope(model)
            model.to(device)
            rec = eval_fast(model, dev_rows, nd_ids, td_ids, stream_dev, stream_train,
                            dev_starts, train_fit, device)
            rec["n_tensors_reverted"] = nrep
            revert_out[tag][name] = rec
            print(f"REVERT {tag} {name} {rec}", flush=True)
            write_json(OUT / "REVERT.json", revert_out)

    write_json(OUT / "STATUS.json", {"status": "COMPLETE", "test_loaded": False})
    write_json(OUT / "SUMMARY_INPUTS.json", {
        "parent_sha256": PARENT_SHA,
        "test_loaded": False,
        "primary_near_miss": str(CKPTS["t14_730001_u750"]),
        "primary_rep_success": str(CKPTS["t14_730002_u750"]),
    })
    print("DECOMP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
