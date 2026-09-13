"""SF5: matched-prefix replicated study controller (single process per seed).

Derived from the frozen SF2/SF3/SF4 controllers. Scientific operations use the
pinned SF2 functions (CE + parent-KL objective, KL pool, scope, optimizer,
binding, evaluation, gates). Prefix (updates 1-100) is byte-identical to SF2
semantics. Arms (updates 101-200) restart from ONE shared in-process snapshot
and differ ONLY by the English learning-rate schedule. Matched-step-101 equality
and RNG invariance are enforced. No historical reproduction gate is used.
"""
from __future__ import annotations
import argparse, json, os, platform, hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
import tokenizers

ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sources"))
import SF2_ENGINE as E  # frozen SF2 engine (pinned functions)
from PINNED_MASKING import set_scope, pad_batch  # noqa: F401

LAMBDA_KL = 1.0
BASE_LR = 5e-5
SEEDS = [87011, 87012, 87013]
ARM_TRAJ_UPDATES = {125, 150, 175, 200}
PREFLIGHT_SEED = 87999
PREFLIGHT_PREFIX_UPDATES = 12
PREFLIGHT_ARM_UPDATES = 3
SF1_BUNDLE = Path(r"C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011")
SF2_RUN_DIR = Path(r"C:\DaveLM-CADAVER\sf2_kl_parent_retention_run_v2")

read = E.read
sha = E.sha


def equality(a, b):
    """Deep structural equality incl. tensor dtype/shape/values; RNG-state aware."""
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu())
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equality(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equality(x, y) for x, y in zip(a, b))
    return a == b


def digest(value):
    h = hashlib.sha256()

    def visit(v):
        h.update(type(v).__name__.encode() + b"\0")
        if torch.is_tensor(v):
            h.update(str(v.dtype).encode())
            h.update(str(tuple(v.shape)).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                visit(k)
                visit(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(str(len(v)).encode())
            for x in v:
                visit(x)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def verify_sf5_freeze():
    receipt = ROOT / "FREEZE_RECEIPT.json"
    assert sha(receipt) == (ROOT / "FREEZE_RECEIPT.sha256").read_text(encoding="utf-8").split()[0]
    r = read(receipt)
    assert r["status"] == "SF5_FROZEN_PRETRAIN_PASS", r["status"]
    manifest = ROOT / "SHA256SUMS.txt"
    assert sha(manifest) == r["manifest_sha256"]
    for line in manifest.read_text(encoding="utf-8").splitlines():
        h, n = line.split("  ", 1)
        assert sha(ROOT / n) == h, n
    return r


def verify(b):
    assert sha(ROOT / "SF2_ENGINE.py") == "150d1f33fab0cfe7027159badec2973f4995b2c76cad2165e5726b43297a8ab7"
    assert sha(ROOT / "SF2_PROTOCOL.json") == sha(SF2_RUN_DIR / "SF2_PROTOCOL.json")
    assert sha(ROOT / "KL_POOL.json") == "aa8f6810f80c62018d026bbbb36e9b5fb85785be694044a1fe9646a8823c5a06"
    assert sha(ROOT / "KL_POOL_MANIFEST.json") == "cc4aaa454cfe1532590752d44a2a31c4e78e6ab005add31afa07c6566906a012"
    assert sha(ROOT / "D3_SELECTION.json") == "3ae0f6a748f5545b5d0af2fd89e5beb5482e9e42713ba61567d1fea353db2226"
    p = E.verify(b)  # SF1 bundle + parent + tokenizer + external hashes
    return p


def provenance():
    return {
        "sf5_protocol": sha(ROOT / "SF5_PROTOCOL.json"),
        "manifest": sha(ROOT / "SHA256SUMS.txt"),
        "sf2_engine": sha(ROOT / "SF2_ENGINE.py"),
        "kl_pool": sha(ROOT / "KL_POOL.json"),
        "d3": sha(ROOT / "D3_SELECTION.json"),
        "schedule": sha(SF1_BUNDLE / "SCHEDULE.json"),
        "parent": sha(Path(r"C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt")),
    }


def english_lr(english_index: int, update_kind: str, arm: str) -> float:
    """Frozen SF3/SF4 anneal schedule for the treatment arm; constant for control."""
    if arm == "control":
        return BASE_LR
    if update_kind == "binding" or english_index <= 90:
        return BASE_LR
    j = english_index - 90
    assert 1 <= j <= 90
    return BASE_LR * (90 - j) / 89


def load_model_and_teacher(parent, device, b):
    m = E.rt.load_model(parent, device, b)
    teacher = E.build_teacher(parent, device)
    E.assert_teacher_disjoint(m.base_model, teacher)
    return m, teacher


def make_optimizer(m):
    return torch.optim.AdamW(m.parameters(), lr=BASE_LR, betas=(.9, .999), eps=1e-8, weight_decay=.05,
                             amsgrad=False, foreach=False, fused=False)


def run_one_update(m, opt, teacher, dev, u, rows_idx, binding_pool, pin, english_index, lr):
    """Execute one frozen-semantics update. Returns (loss, ce, kl, grad_norm, new_english_index).

    Mirrors the frozen SF2 engine loop exactly: for English updates the KL is
    computed (eval mode) with the incremented English ordinal; CE is computed in
    train mode; binding updates use the pinned binding objective.
    """
    for g in opt.param_groups:
        g["lr"] = lr
    opt.zero_grad(set_to_none=True)
    set_scope(m, u["kind"] == "binding")
    inactive = {n: (x.detach().clone(),
                    {k: v.clone() if torch.is_tensor(v) else v for k, v in opt.state.get(x, {}).items()})
                for n, x in m.named_parameters() if not x.requires_grad}
    ce = kl = None
    if u["kind"] == "english":
        english_index += 1
        m.eval()
        sel = E.pick_kl_entries(kl_pool_entries, english_index)
        kl = E.compute_kl(m, teacher, dev, kl_pool_rows, sel)
        m.train()
        batch = [rows_idx[i] for i in u["ids"]]
        x, y = pad_batch(batch, u["pad"])
        x = x.to(dev)
        y = y.to(dev)
        ce = F.cross_entropy(m.base_model(x).reshape(-1, 1024), y.reshape(-1), ignore_index=-100)
        loss = ce + LAMBDA_KL * kl
    else:
        m.train()
        loss, _ = E.rt.binding_loss(m, E.rt.binding_docs_for_batch(binding_pool, u["quartets"], u["documents"]), dev, pin)
    loss.backward()
    assert all(x.grad is None for x in m.parameters() if not x.requires_grad)
    norm = torch.nn.utils.clip_grad_norm_([x for x in m.parameters() if x.grad is not None], 2.0)
    opt.step()
    for n, x in m.named_parameters():
        if n in inactive:
            old, st = inactive[n]
            assert torch.equal(x, old) and equality(opt.state.get(x, {}), st)
    return float(loss.detach()), (float(ce.detach()) if ce is not None else None), \
           (float(kl.detach()) if kl is not None else None), float(norm), english_index


# module-level KL pool (set in preflight/run_seed entry points before any update)
kl_pool_entries = None
kl_pool_rows = None


def eval_at(m, b, out, u, rows, tok, dev, base_loss, arm_label, full):
    """RNG-invariant evaluation. Returns acquisition aggregate."""
    rng_before = E.rt.capture_rng()
    agg = E.panel(m, b, out, f"update{u}_acquisition", rows, tok, dev)
    if full:
        qpath = out / f"update{u}_checks.json"
        if not qpath.exists():
            E.rt.atomic_json(E.checks(m, b, dev), qpath)
        q = read(qpath)
        dpath = out / f"d3_update{u}_summary.json"
        if not dpath.exists():
            sel = read(ROOT / "D3_SELECTION.json")
            raw = E.measure_d3(m, sel, tok, dev)
            with (out / f"d3_update{u}_RAW.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
                for r in raw:
                    fh.write(json.dumps(r) + "\n")
            E.rt.atomic_json(E.d3_summary(raw), dpath)
        d = read(dpath)
        gates = {
            "binding": all(v["gate"] for v in q["binding"].values()),
            "language": q["language"]["loss"] <= base_loss + 0.25,
            "d3": d["mean_combined_name_probability"] <= 0.01,
            "acquisition_guard": not (u == 100 and agg["correct"] < 10 and agg["exact"] < 10),
        }
        E.rt.atomic_json(gates, out / f"update{u}_gates.json")
        if not all(gates.values()):
            raise RuntimeError(f"HARD_STOP_RETENTION_OR_ACQUISITION_GUARD arm={arm_label} u={u} gates={gates}")
    assert equality(rng_before, E.rt.capture_rng()), "evaluation consumed RNG"
    return agg


def write_metric(out, rec):
    with (out / "TRAIN_METRICS.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def preflight(args):
    """Deterministic preflight: prove snapshot->restore identity and arm equality at the fork."""
    b = Path(args.bundle)
    p = verify(b)
    verify_sf5_freeze()
    E.rt.configure_runtime(PREFLIGHT_SEED)
    assert os.environ.get("PYTHONHASHSEED") == str(PREFLIGHT_SEED)
    dev = torch.device("cuda")
    rows = read(b / "TRAIN.json")
    rows_idx = {r["id"]: r for r in rows}
    schedule = read(b / "SCHEDULE.json")
    binding_pool = read(b / "data/binding_rehearsal.json")["quartets"]
    tok = E.Tokenizer.from_file(p["tokenizer"])
    global kl_pool_entries, kl_pool_rows
    kl_pool, _ = E.load_kl_pool()
    kl_pool_entries = kl_pool["entries"]
    kl_pool_rows = kl_pool["rows"]
    out = ROOT / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "STATUS.json").exists():
        raise RuntimeError("preflight dir already used; refusing overwrite")
    m, teacher = load_model_and_teacher(Path(p["parent"]), dev, b)
    pin = E.rt.pinned_binding(b)
    opt = make_optimizer(m)
    # Prefix of 12 updates (one binding at u10).
    eng_idx = 0
    for u in schedule[:PREFLIGHT_PREFIX_UPDATES]:
        loss, ce, kl, norm, eng_idx = run_one_update(m, opt, teacher, dev, u, rows_idx, binding_pool, pin, eng_idx, BASE_LR)
    snapshot = {"completed": PREFLIGHT_PREFIX_UPDATES, "model": m.state_dict(), "optimizer": opt.state_dict(),
                "rng": E.rt.capture_rng(), "scope": "english", "english_index": eng_idx,
                "provenance": provenance()}
    digests = {}
    for trial in ("A", "B"):
        m.load_state_dict(snapshot["model"])
        opt.load_state_dict(snapshot["optimizer"])
        E.rt.restore_rng(snapshot["rng"])
        set_scope(m, False)
        eng_idx2 = snapshot["english_index"]
        for u in schedule[PREFLIGHT_PREFIX_UPDATES:PREFLIGHT_PREFIX_UPDATES + PREFLIGHT_ARM_UPDATES]:
            loss, ce, kl, norm, eng_idx2 = run_one_update(m, opt, teacher, dev, u, rows_idx, binding_pool, pin, eng_idx2, BASE_LR)
        digests[trial] = {"model": digest(m.state_dict()), "optimizer": digest(opt.state_dict()),
                          "rng": digest(E.rt.capture_rng()), "english_index": eng_idx2}
    exact = digests["A"] == digests["B"]
    result = {"status": "SF5_PREFLIGHT_FORK_IDENTITY_PASS" if exact else "SF5_PREFLIGHT_FORK_IDENTITY_FAIL",
              "exact": exact, "trialA": digests["A"], "trialB": digests["B"],
              "prefix_updates": PREFLIGHT_PREFIX_UPDATES, "arm_updates": PREFLIGHT_ARM_UPDATES,
              "seed": PREFLIGHT_SEED}
    E.rt.atomic_json(result, out / "PREFLIGHT_RESULT.json")
    E.rt.atomic_json({"status": "SF5_PREFLIGHT_COMPLETE"}, out / "STATUS.json")
    print(json.dumps({"status": result["status"], "exact": exact}))
    if not exact:
        raise RuntimeError("SF5_PREFLIGHT_FORK_IDENTITY_FAIL")


def arm_continuation(m, opt, teacher, dev, b, schedule, rows_idx, binding_pool, pin, snapshot,
                     arm, out, seed, base_loss, rows, tok):
    """Run updates 101-200 for an arm from the snapshot. Returns (endpoint dict, step101 state dict)."""
    out.mkdir(parents=True, exist_ok=True)
    restart = out / "restart.pt"
    assert not restart.exists(), "arm dir not fresh"
    m.load_state_dict(snapshot["model"])
    opt.load_state_dict(snapshot["optimizer"])
    E.rt.restore_rng(snapshot["rng"])
    set_scope(m, False)
    assert equality(m.state_dict(), snapshot["model"])
    assert equality(opt.state_dict(), snapshot["optimizer"])
    assert equality(E.rt.capture_rng(), snapshot["rng"])
    english_index = snapshot["english_index"]
    step101 = None
    for u in schedule[100:]:
        nxt = english_index + (1 if u["kind"] == "english" else 0)
        lr = english_lr(nxt, u["kind"], arm)
        loss, ce, kl, norm, english_index = run_one_update(
            m, opt, teacher, dev, u, rows_idx, binding_pool, pin, english_index, lr)
        number = u["update"]
        state_val = {"completed": number, "arm": arm, "model": m.state_dict(), "optimizer": opt.state_dict(),
                     "rng": E.rt.capture_rng(), "scope": "binding" if number % 10 == 0 else "english",
                     "english_index": english_index, "provenance": snapshot["provenance"]}
        E.rt.atomic_torch_save(state_val, restart)
        rec = {"update": number, "kind": u["kind"], "arm": arm, "lr": lr, "loss": loss, "grad_norm": norm}
        if ce is not None:
            rec["ce"] = ce
            rec["kl"] = kl
        write_metric(out, rec)
        if number == 101:
            step101 = state_val
            E.rt.atomic_torch_save(state_val, out / "matched_step101_state.pt")
        if number in ARM_TRAJ_UPDATES:
            full = number == 200
            eval_at(m, b, out, number, rows, tok, dev, base_loss, arm, full)
        if number == 200:
            break
    return step101


def endpoint_summary(arm_out, ce0):
    q = read(arm_out / "update200_checks.json")
    agg = read(arm_out / "update200_acquisition_RESULT.json")
    d = read(arm_out / "d3_update200_summary.json")
    acq_pass = agg["correct"] == 16 and agg["exact"] == 16 and agg["reversals"] == 8 and agg["families"] == 4
    return {
        "acquisition": {"correct": agg["correct"], "exact": agg["exact"], "reversals": agg["reversals"],
                        "families": agg["families"]},
        "acquisition_pass": acq_pass,
        "binding_pass": all(v["gate"] for v in q["binding"].values()),
        "language_pass": q["language"]["loss"] <= ce0 + 0.25,
        "d3_pass": d["mean_combined_name_probability"] <= 0.01,
        "language_ce": q["language"]["loss"],
        "d3_mass": d["mean_combined_name_probability"],
    }


def run_seed(args):
    b = Path(args.bundle)
    p = verify(b)
    verify_sf5_freeze()
    seed = args.seed
    E.rt.configure_runtime(seed)
    assert os.environ.get("PYTHONHASHSEED") == str(seed)
    assert platform.python_version() == "3.12.14"
    assert torch.__version__ == "2.12.0+rocm7.14.0"
    assert tokenizers.__version__ == "0.23.1"
    dev = torch.device("cuda")
    rows = read(b / "TRAIN.json")
    rows_idx = {r["id"]: r for r in rows}
    schedule = read(b / "SCHEDULE.json")
    binding_pool = read(b / "data/binding_rehearsal.json")["quartets"]
    tok = E.Tokenizer.from_file(p["tokenizer"])
    global kl_pool_entries, kl_pool_rows
    kl_pool, _ = E.load_kl_pool()
    kl_pool_entries = kl_pool["entries"]
    kl_pool_rows = kl_pool["rows"]
    seed_out = ROOT / f"run_seed{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    status_path = seed_out / "STATUS.json"
    if status_path.exists():
        raise RuntimeError(f"seed {seed} already completed; refusing overwrite")
    m, teacher = load_model_and_teacher(Path(p["parent"]), dev, b)
    pin = E.rt.pinned_binding(b)
    # --- update-0 baseline reproduction (M9) ---
    rng0 = E.rt.capture_rng()
    agg0 = E.panel(m, b, seed_out, "update0_acquisition", rows, tok, dev)
    checks0 = E.checks(m, b, dev)
    E.rt.atomic_json(checks0, seed_out / "update0_checks.json")
    assert agg0["correct"] == 9 and agg0["exact"] == 0, agg0
    assert all(v["gate"] for v in checks0["binding"].values())
    ce0 = checks0["language"]["loss"]
    assert abs(ce0 - 3.3907) / 3.3907 < 1e-4, ce0
    d3sel = read(ROOT / "D3_SELECTION.json")
    d3r0 = E.measure_d3(m, d3sel, tok, dev)
    E.rt.atomic_json(E.d3_summary(d3r0), seed_out / "d3_update0_summary.json")
    assert abs(E.d3_summary(d3r0)["mean_combined_name_probability"] - 0.000907) < 0.0001
    assert equality(rng0, E.rt.capture_rng()), "update0 eval consumed RNG"
    # --- prefix updates 1..100 ---
    opt = make_optimizer(m)
    eng_idx = 0
    for u in schedule[:100]:
        loss, ce, kl, norm, eng_idx = run_one_update(m, opt, teacher, dev, u, rows_idx, binding_pool, pin, eng_idx, BASE_LR)
    eval_at(m, b, seed_out, 100, rows, tok, dev, ce0, "prefix", full=True)
    # --- snapshot ---
    snapshot = {"completed": 100, "model": m.state_dict(), "optimizer": opt.state_dict(),
                "rng": E.rt.capture_rng(), "scope": "binding", "english_index": eng_idx,
                "provenance": provenance()}
    E.rt.atomic_torch_save(snapshot, seed_out / "snapshot100.pt")
    ctrl_out = seed_out / "control"
    trt_out = seed_out / "treatment"
    ctrl_out.mkdir(exist_ok=True)
    trt_out.mkdir(exist_ok=True)
    ctrl101 = arm_continuation(m, opt, teacher, dev, b, schedule, rows_idx, binding_pool, pin, snapshot,
                               "control", ctrl_out, seed, ce0, rows, tok)
    trt101 = arm_continuation(m, opt, teacher, dev, b, schedule, rows_idx, binding_pool, pin, snapshot,
                              "treatment", trt_out, seed, ce0, rows, tok)
    # matched-step-101 fork-integrity gate
    if ctrl101 is None or trt101 is None:
        raise RuntimeError("matched step-101 state missing")
    ck = [k for k in trt101 if k != "arm"]
    exact101 = all(equality(trt101[k], ctrl101[k]) for k in ck)
    E.rt.atomic_json({"matched_step101_exact": bool(exact101)}, seed_out / "MATCHED_STEP101.json")
    if not exact101:
        raise RuntimeError("HARD_STOP_MATCHED_STEP101_MISMATCH")
    ctrl_sum = endpoint_summary(ctrl_out, ce0)
    trt_sum = endpoint_summary(trt_out, ce0)
    for s in (ctrl_sum, trt_sum):
        s["endpoint_pass"] = s["acquisition_pass"] and s["binding_pass"] and s["language_pass"] and s["d3_pass"]
    E.rt.atomic_json({"seed": seed, "control": ctrl_sum, "treatment": trt_sum}, seed_out / "SEED_ENDPOINTS.json")
    E.rt.atomic_json({"status": "SF5_SEED_COMPLETE", "seed": seed}, status_path)
    print(json.dumps({"seed": seed, "control_endpoint_pass": ctrl_sum["endpoint_pass"],
                      "treatment_endpoint_pass": trt_sum["endpoint_pass"],
                      "matched_step101_exact": bool(exact101)}))


def finalize():
    results = {}
    for seed in SEEDS:
        ep_path = ROOT / f"run_seed{seed}" / "SEED_ENDPOINTS.json"
        results[str(seed)] = read(ep_path)
    ctrl_passes = [results[s]["control"]["endpoint_pass"] for s in results]
    trt_passes = [results[s]["treatment"]["endpoint_pass"] for s in results]
    support_pairs = sum(1 for c, t in zip(ctrl_passes, trt_passes) if (not c) and t)
    harm_pairs = sum(1 for c, t in zip(ctrl_passes, trt_passes) if c and (not t))
    if support_pairs >= 2 and harm_pairs == 0:
        decision = "SUPPORT"
    elif harm_pairs >= 1:
        decision = "HARM"
    else:
        decision = "WEAK_NO_SUPPORT"
    out = {"decision": decision, "per_seed": results, "support_pairs": support_pairs, "harm_pairs": harm_pairs}
    E.rt.atomic_json(out, ROOT / "FINAL_CLASSIFICATION.json")
    print(json.dumps({"decision": decision, "control_passes": ctrl_passes,
                      "treatment_passes": trt_passes}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["preflight", "seed", "finalize"], required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--bundle", type=Path, default=SF1_BUNDLE)
    args = ap.parse_args()
    if args.mode == "preflight":
        preflight(args)
    elif args.mode == "seed":
        assert args.seed in SEEDS, args.seed
        run_seed(args)
    elif args.mode == "finalize":
        finalize()


if __name__ == "__main__":
    main()
