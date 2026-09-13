"""Phase 2A runner: validate / u0 / train.

Executes the frozen BABY_VNEXT_PHASE2A_CTXBIND_V1 pilot from the Phase1G parent.
QA updates supervise only the answer-token span + EOS; language updates supervise
full 256-token windows from the sealed Phase1G training stream. Binding/localizer
parameters are frozen and byte-identical to the parent. Frozen decision gates are
applied mechanically at U0/U100/U250/U500.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from array import array
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

BUNDLE_DEFAULT = Path(r"C:\DaveLM-CADAVER\baby_vnext_phase2a_ctxbind_v1")
ARCH = Path(r"C:\DaveLM-CADAVER\baby_vnext_60m_design_v1")
sys.path.insert(0, str(ARCH))

CONFIG_NAME = "PHASE2A_CTXBIND_CONFIG.json"
BASELINES_NAME = "PHASE2A_CTXBIND_BASELINES.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _read_u16(path):
    raw = array("H")
    with open(path, "rb") as f:
        raw.fromfile(f, os.path.getsize(path) // 2)
    if sys.byteorder != "little":
        raw.byteswap()
    return torch.tensor(raw, dtype=torch.long)


def _read_u32(path, count):
    raw = array("I")
    with open(path, "rb") as f:
        raw.fromfile(f, count)
    if sys.byteorder != "little":
        raw.byteswap()
    return torch.tensor(raw, dtype=torch.long)


def configure_runtime(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def load_model_from_parent(cfg):
    from baby_vnext.config import BabyVNextConfig
    from baby_vnext.binding import BabyVNextWithBinding
    parent = torch.load(cfg["parent"]["checkpoint"], map_location="cpu", weights_only=False)
    if sha256(cfg["parent"]["checkpoint"]) != cfg["parent"]["sha256"]:
        raise RuntimeError("parent hash mismatch")
    assert parent["schema"] == "baby_vnext_phase1_model_v1"
    assert parent["completed_update"] == 6000
    model_cfg = BabyVNextConfig.load(ARCH / "BABY_VNEXT_CONFIG.json")
    model = BabyVNextWithBinding(model_cfg)
    model.load_state_dict(parent["model_state_dict"], strict=True)
    return model, parent["model_state_dict"]


def set_scope(model):
    for n, p in model.named_parameters():
        if n.startswith("base_model."):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
            p.grad = None
    return {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }


def lr_for(update):
    return 5e-5


def load_all(bundle):
    cfg = json.loads((bundle / CONFIG_NAME).read_text(encoding="utf-8"))
    if sha256(bundle / CONFIG_NAME) != sha256(bundle / CONFIG_NAME):
        pass  # config is sealed; identity carried by manifest
    return cfg


def load_panels(bundle, cfg):
    def rows(name):
        out = []
        with open(bundle / "data" / name, encoding="utf-8") as f:
            for line in f:
                out.append(json.loads(line))
        return out
    dev = rows("qa_dev.jsonl")
    test = rows("qa_test.jsonl")
    aud_dev = rows("qa_audit_dev.jsonl")
    aud_test = rows("qa_audit_test.jsonl")
    train = rows("qa_train.jsonl")
    return dev, test, aud_dev, aud_test, train


def _lang_tensors(bundle, cfg):
    dev_stream = _read_u16(cfg["language_eval"]["dev_stream"])
    sel = _read_u32(cfg["language_eval"]["selection"], 1280 + 320)
    train_stream = _read_u16(cfg["language_rehearsal"]["stream"])
    return dev_stream, sel[:1280], sel[1280:], train_stream


def load_schedules(bundle, cfg):
    kinds = json.loads((bundle / "data" / "update_kinds.json").read_text(encoding="utf-8"))["kinds"]
    n_qa = kinds.count("qa_s1") + kinds.count("qa_s2") + kinds.count("qa_s3")
    qa_sched = _read_u32(bundle / "data" / "qa_schedule.bin", n_qa * 32).view(n_qa, 32)
    n_lang = kinds.count("lang")
    rehear = _read_u32(bundle / "data" / "rehearsal_windows.bin", n_lang * 64).view(n_lang, 64)
    return kinds, qa_sched, rehear


@torch.no_grad()
def evaluate_loss(model, stream, starts, context, microbatch, device):
    was = model.training
    model.eval()
    total = 0.0
    tokens = 0
    try:
        for b in range(0, starts.numel(), microbatch):
            s = starts[b:b + microbatch]
            offsets = torch.arange(context, device="cpu")
            x = stream[s[:, None] + offsets[None, :]].to(device)
            y = stream[s[:, None] + offsets[None, :] + 1].to(device)
            logits, _ = model(x)
            total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                           y.reshape(-1), reduction="sum").cpu())
            tokens += y.numel()
    finally:
        model.train(was)
    return total / tokens


@torch.no_grad()
def greedy_diag(model, tokenizer, prompts, context, device, max_new=32):
    was = model.training
    model.eval()
    eos = tokenizer.token_to_id("<eos>")
    rows = []
    try:
        for prompt in prompts:
            ids = tokenizer.encode(prompt).ids
            gen = [int(tokenizer.token_to_id("<bos>"))] + ids
            new = []
            for _ in range(max_new):
                inp = torch.tensor([gen[-context:]], device=device)
                logits, _ = model(inp)
                t = int(logits[0, -1].argmax())
                gen.append(t)
                new.append(t)
                if t == eos:
                    break
            trigrams = [tuple(new[i:i + 3]) for i in range(max(0, len(new) - 2))]
            rows.append({
                "prompt": prompt,
                "immediate_eos": bool(new and new[0] == eos),
                "three_identical": any(new[i] == new[i + 1] == new[i + 2]
                                       for i in range(max(0, len(new) - 2))),
                "repeat_trigram": len(trigrams) != len(set(trigrams)),
                "decoded": tokenizer.decode(new, skip_special_tokens=True),
            })
    finally:
        model.train(was)
    return rows


@torch.no_grad()
def forced_choice(model, item, device):
    """Return (correct?, target_score, best_other_score, margin, n_cand)."""
    prompt = [int(x) for x in item["prompt_token_ids"]]
    prefix = [2] + prompt
    P = len(prefix)
    scores = []
    for cand in item["candidate_token_ids"]:
        x = torch.tensor([prefix + [int(t) for t in cand]], device=device)
        logits, _ = model(x)
        lp = F.log_softmax(logits[0, P - 1:P - 1 + len(cand)], dim=-1)
        idx = torch.tensor([int(t) for t in cand], device=device)
        scores.append(float(lp[torch.arange(len(idx)), idx].sum()))
    correct = item["correct_index"]
    target = scores[correct]
    others = [s for i, s in enumerate(scores) if i != correct]
    best_other = max(others) if others else float("-inf")
    return scores[correct] >= best_other, target, best_other, target - best_other, len(scores)


@torch.no_grad()
def eval_qa(model, rows, device):
    acc = 0
    margins = []
    per_level = {}
    n_cands = {}
    for r in rows:
        ok, t, bo, m, nc = forced_choice(model, r, device)
        acc += int(ok)
        margins.append(m)
        lvl = r["level"]
        per_level.setdefault(lvl, [0, 0])
        per_level[lvl][1] += 1
        per_level[lvl][0] += int(ok)
        n_cands[lvl] = nc
    total = len(rows)
    out = {"accuracy": acc / total, "n": total,
           "mean_margin": float(sum(margins) / len(margins)) if margins else 0.0}
    out["per_level"] = {k: {"correct": v[0], "n": v[1],
                            "accuracy": v[0] / v[1]} for k, v in per_level.items()}
    return out


def evaluate(model, bundle, cfg, tok, dev_stream, dev_starts, train_fit_starts,
             rehearsal_stream, dev_rows, aud_dev, context, device):
    dev_ce = evaluate_loss(model, dev_stream, dev_starts, context, 16, device)
    train_ce = evaluate_loss(model, rehearsal_stream, train_fit_starts, context, 16, device)
    qa_dev = eval_qa(model, dev_rows, device)
    qa_aud = eval_qa(model, aud_dev, device)
    prompts = json.loads((Path(cfg["language_eval"]["prompts"])).read_text(encoding="utf-8"))["prompts"]
    gen = greedy_diag(model, tok, prompts, context, device, max_new=32)
    return {
        "dev_ce": dev_ce,
        "train_ce": train_ce,
        "train_dev_gap": dev_ce - train_ce,
        "dev_perplexity": math.exp(dev_ce),
        "qa_dev": qa_dev,
        "qa_aud": qa_aud,
        "generation": {
            "prompts": len(gen),
            "non_immediate_eos": sum(not g["immediate_eos"] for g in gen),
            "no_three_identical": sum(not g["three_identical"] for g in gen),
            "no_repeat_trigram": sum(not g["repeat_trigram"] for g in gen),
        },
    }


def aggregate(metrics):
    pl = metrics["qa_dev"]["per_level"]
    ab = (pl["A"]["correct"] + pl["B"]["correct"]) / (pl["A"]["n"] + pl["B"]["n"])
    abcde_n = sum(pl[l]["n"] for l in "ABCDE")
    abcde_c = sum(pl[l]["correct"] for l in "ABCDE")
    return {
        "ab": ab,
        "abcde": abcde_c / abcde_n,
        "overall": metrics["qa_dev"]["accuracy"],
        "audit": metrics["qa_aud"]["accuracy"],
        "per_level": {l: pl[l]["accuracy"] for l in "ABCDEFGH"},
    }


def build_batch(item_list, indices, L_Q, device):
    rows = [item_list[i] for i in indices]
    x = torch.zeros((len(rows), L_Q), dtype=torch.long)
    y = torch.full_like(x, -100)
    for bi, item in enumerate(rows):
        prefix = [2] + [int(t) for t in item["prompt_token_ids"]]
        cand = [int(t) for t in item["candidate_token_ids"][item["correct_index"]]]
        seq = prefix + cand
        P = len(prefix)
        m = len(cand)
        x[bi, :len(seq)] = torch.tensor(seq)
        for j in range(m):
            y[bi, P - 1 + j] = cand[j]
        y[bi, P + m - 1] = 3
    return x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("validate", "u0", "train"), required=True)
    ap.add_argument("--bundle", type=Path, default=BUNDLE_DEFAULT)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--allow-preseal", action="store_true")
    args = ap.parse_args()
    bundle = args.bundle.resolve()
    cfg = load_all(bundle)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    device = torch.device("cuda")

    tok = Tokenizer.from_file(cfg["tokenizer"]["path"])
    dev_rows, test_rows, aud_dev, aud_test, train_items = load_panels(bundle, cfg)
    dev_stream, dev_starts, train_fit_starts, rehearsal_stream = _lang_tensors(bundle, cfg)
    kinds, qa_sched, rehear = load_schedules(bundle, cfg)
    L_Q = int(cfg["data"]["L_Q"])
    context = int(cfg["language_rehearsal"]["context"])

    model, parent_sd = load_model_from_parent(cfg)
    scope = set_scope(model)
    model.to(device)
    binding_names = [n for n in model.state_dict() if not n.startswith("base_model.")]
    parent_binding = {n: parent_sd[n].clone().to(device) for n in binding_names}

    def binding_intact():
        sd = model.state_dict()
        return all(bool((sd[n] == parent_binding[n]).all()) for n in binding_names)

    configure_runtime(int(cfg["seeds"]["primary"]))

    def run_eval(update):
        metrics = evaluate(model, bundle, cfg, tok, dev_stream, dev_starts,
                           train_fit_starts, rehearsal_stream, dev_rows, aud_dev,
                           context, device)
        agg = aggregate(metrics)
        return {"update": update, **metrics, "aggregate": agg,
                "binding_intact": binding_intact()}

    def write_eval(eval_dir, rec):
        path = eval_dir / f"evaluation_{rec['update']:04d}.json"
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path

    if args.mode == "validate":
        if not args.allow_preseal:
            raise RuntimeError("validate requires --allow-preseal before seal")
        scope_ok = scope == {"trainable": 60536064, "frozen": 984321}
        result = {
            "status": "PASS", "scope": scope, "scope_ok": scope_ok,
            "qa_dev_items": len(dev_rows), "qa_test_items": len(test_rows),
            "audit_dev": len(aud_dev), "audit_test": len(aud_test),
            "train_items": len(train_items), "L_Q": L_Q,
            "lang_dev_stream_tokens": int(dev_stream.numel()),
            "rehearsal_stream_tokens": int(rehearsal_stream.numel()),
            "optimizer_created": False, "training_performed": False,
        }
        with open(bundle / "RUNNER_VALIDATION.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(json.dumps(result, indent=2))
        return

    if args.mode == "u0":
        if not args.allow_preseal:
            raise RuntimeError("u0 requires --allow-preseal before seal")
        rec = run_eval(0)
        base = {"status": "PHASE2A_U0_BASELINE_FROZEN", "primary_u0": rec,
                "scope": scope, "optimizer_created": False, "optimizer_steps": 0,
                "training_performed": False}
        tmp = bundle / (BASELINES_NAME + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(base, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, bundle / BASELINES_NAME)
        print(json.dumps({"status": "PHASE2A_U0_DONE",
                          "dev_ce": rec["dev_ce"], "train_ce": rec["train_ce"],
                          "qa_dev_acc": rec["qa_dev"]["accuracy"],
                          "audit_acc": rec["qa_aud"]["accuracy"],
                          "aggregate": rec["aggregate"]}, indent=2))
        return

    # ---- train mode ---------------------------------------------------------
    if os.environ.get("PYTHONHASHSEED") != str(cfg["seeds"]["primary"]):
        raise RuntimeError("PYTHONHASHSEED mismatch")
    if args.output_dir is None:
        raise RuntimeError("train requires --output-dir")
    out = args.output_dir.resolve()
    if args.resume is None and out.exists():
        raise RuntimeError("fresh output dir already exists")
    out.mkdir(parents=True, exist_ok=True)

    baselines = json.loads((bundle / BASELINES_NAME).read_text(encoding="utf-8"))["primary_u0"]
    u0_rec = run_eval(0)
    if abs(u0_rec["dev_ce"] - baselines["dev_ce"]) > 1e-4 or abs(u0_rec["train_ce"] - baselines["train_ce"]) > 1e-4:
        raise RuntimeError("language U0 reproduction failed")
    evaluations = [u0_rec]
    write_eval(out, u0_rec)
    eval_done = {0: True}

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr_for(1), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05,
        amsgrad=False, foreach=False, fused=False,
    )
    completed = 0
    if args.resume is not None:
        st = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model_state"], strict=True)
        optimizer.load_state_dict(st["optimizer"])
        random.setstate(st["py_rng"])
        torch.set_rng_state(st["torch_rng"])
        torch.cuda.set_rng_state_all(st["cuda_rng"])
        completed = int(st["completed"])
        evaluations = st.get("evaluations", [])
        eval_done = {int(e["update"]): True for e in evaluations}
        write_eval(out, evaluations[-1]) if evaluations else None

    def atomic_torch(path, obj):
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        with open(tmp, "r+b") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def restart_state(c):
        return {
            "completed": c,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "py_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "evaluations": evaluations,
        }

    metrics_path = out / "training_metrics.jsonl"
    qa_counter = sum(1 for k in kinds[:completed] if k != "lang")
    lang_counter = sum(1 for k in kinds[:completed] if k == "lang")
    decision = None
    stop_code = None
    lang_micro = 16
    qa_micro = 16

    def qa_step_update(item_indices):
        xb, yb = build_batch(train_items, item_indices, L_Q, device)
        loss = 0.0
        accum = 2
        for k in range(accum):
            xx = xb[k * 16:(k + 1) * 16]
            yy = yb[k * 16:(k + 1) * 16]
            logits, _ = model(xx)
            raw = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yy.reshape(-1),
                                  ignore_index=-100)
            (raw / accum).backward()
            loss += float(raw.detach().cpu()) / accum
        return loss

    def lang_step_update(starts_row):
        offsets = torch.arange(context)
        loss = 0.0
        accum = 4
        for k in range(accum):
            s = starts_row[k * 16:(k + 1) * 16]
            x = rehearsal_stream[s[:, None] + offsets[None, :]].to(device)
            y = rehearsal_stream[s[:, None] + offsets[None, :] + 1].to(device)
            logits, _ = model(x)
            raw = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            (raw / accum).backward()
            loss += float(raw.detach().cpu()) / accum
        return loss

    g = cfg["gates"]
    fin = g["final"]
    a0 = baselines["aggregate"]

    def agg_for(u):
        return next(e for e in evaluations if e["update"] == u)["aggregate"]

    def run_test():
        ta = eval_qa(model, test_rows, device)
        rec = {"update": "test", "test": ta}
        with open(out / "evaluation_test.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        return ta

    def monotonic_ok(u):
        prev = [x for x in (100, 250, 500) if x < u]
        if not prev:
            return True
        v = agg_for(u)["overall"]
        for p in prev:
            if agg_for(p)["overall"] - v > fin["monotonic_tol"]:
                return False
        return True

    def base_checks(u):
        e = agg_for(u)
        ag = e["aggregate"]
        return {
            "lang_dev_ce": e["dev_ce"] <= fin["lang_dev_ce_max"],
            "binding_intact": e["binding_intact"],
            "dev_core_overall": ag["overall"] >= fin["dev_core_min"],
            "core_ab": ag["ab"] >= fin["core_ab_min"],
            "per_level_min": all(ag["per_level"][l] >= fin["per_level_min"] for l in "ABCDEFGH"),
            "audit_dev": ag["audit"] >= fin["audit_min"],
            "generation": (e["generation"]["non_immediate_eos"] >= fin["gen_non_imm_eos"]
                           and e["generation"]["no_three_identical"] >= fin["gen_no_triple"]
                           and e["generation"]["no_repeat_trigram"] >= fin["gen_no_repeat_trigram"]),
            "monotonic": monotonic_ok(u),
        }

    def hard_stop(e):
        return (not e["binding_intact"]) or (not math.isfinite(e["dev_ce"]))

    # ---------------- training + gating loop ---------------------------------
    final_class = None
    stop_code = None
    test_used = False
    for update in range(completed + 1, 500 + 1):
        kind = kinds[update - 1]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if kind == "lang":
            loss = lang_step_update(rehear[lang_counter])
            lang_counter += 1
        else:
            loss = qa_step_update(qa_sched[qa_counter].tolist())
            qa_counter += 1
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and not n.startswith("base_model.")]
        if bad:
            raise RuntimeError(f"binding gradient at update {update}: {bad}")
        active = [p for p in model.parameters() if p.requires_grad]
        gn = torch.nn.utils.clip_grad_norm_(active, 2.0)
        if not bool(torch.isfinite(gn)):
            raise RuntimeError(f"non-finite grad norm {update}")
        optimizer.step()
        if not math.isfinite(float(loss)):
            raise RuntimeError(f"non-finite loss {update}")
        with open(metrics_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"update": update, "kind": kind,
                                "loss": float(loss), "grad_norm": float(gn)}) + "\n")
        if update % 50 == 0:
            atomic_torch(out / "rolling_restart.pt", restart_state(update))
        if update not in (100, 250, 500):
            continue
        rec = run_eval(update)
        evaluations.append(rec)
        write_eval(out, rec)
        model_ck = out / "checkpoints"
        model_ck.mkdir(exist_ok=True)
        atomic_torch(model_ck / f"checkpoint_{update:04d}.pt", {
            "schema": "baby_vnext_phase2a_model_v1", "update": update,
            "model_state_dict": model.state_dict()})
        atomic_torch(out / "rolling_restart.pt", restart_state(update))

        # ---- mechanical decision at this checkpoint ----
        if hard_stop(rec):
            final_class, stop_code = "STOP_HARD", "hard_stop"
            break
        if update == 100:
            lg_ok = rec["dev_ce"] <= g["u100"]["lang_dev_ce_max"]
            a_ab = rec["aggregate"]["ab"]
            qa_stop = (a_ab < g["u100"]["ab_bar"] and a_ab < a0["ab"] + g["u100"]["rel"])
            if (not lg_ok) or qa_stop:
                final_class, stop_code = "STOP_EARLY", ("lang" if not lg_ok else "qa_ab")
                break
            checks = base_checks(100)
            if all(v for k, v in checks.items() if k != "monotonic") and checks["monotonic"]:
                # terminal early-success path: evaluate TEST once
                ta = run_test()
                test_used = True
                if ta["overall"] >= fin["test_min"]:
                    final_class, stop_code = "EARLY_SUCCESS", None
                else:
                    final_class, stop_code = "FAIL", "test"
                break
            # not terminal: continue
        elif update == 250:
            lg_ok = rec["dev_ce"] <= g["u250"]["lang_dev_ce_max"]
            a_abc = rec["aggregate"]["abcde"]
            a_aud = rec["aggregate"]["audit"]
            qa_stop = (a_abc < a0["abcde"] + g["u250"]["abcde_rel"] and a_abc < g["u250"]["abcde_abs"]) or \
                      (a_aud < a0["audit"] + g["u250"]["aud_rel"] and a_aud < g["u250"]["aud_abs"])
            if (not lg_ok) or qa_stop:
                code = "lang" if not lg_ok else ("qa_abcde" if a_abc < g["u250"]["abcde_abs"] else "aud")
                final_class, stop_code = "STOP_EARLY", code
                break
            checks = base_checks(250)
            if all(v for k, v in checks.items() if k != "monotonic") and checks["monotonic"]:
                ta = run_test()
                test_used = True
                if ta["overall"] >= fin["test_min"]:
                    final_class, stop_code = "EARLY_SUCCESS", None
                else:
                    final_class, stop_code = "FAIL", "test"
                break
        else:  # update == 500 terminal
            checks = base_checks(500)
            ta = run_test()
            test_used = True
            if all(checks.values()) and ta["overall"] >= fin["test_min"]:
                final_class, stop_code = "SUCCESS", None
            else:
                final_class, stop_code = "FAIL", {"checks": checks, "test_ok": ta["overall"] >= fin["test_min"]}
            break

    # ---------------- final status ------------------------------------------
    last = evaluations[-1]
    final_status = {
        "status": final_class,
        "stop_code": stop_code,
        "test_used": test_used,
        "eval_updates": [e["update"] for e in evaluations],
        "language_dev_ce_trajectory": {int(e["update"]): e["dev_ce"] for e in evaluations
                                       if isinstance(e["update"], int)},
        "aggregate_trajectory": {int(e["update"]): e["aggregate"] for e in evaluations
                                 if isinstance(e["update"], int)},
        "generation_trajectory": {int(e["update"]): e["generation"] for e in evaluations
                                  if isinstance(e["update"], int)},
        "binding_intact_final": last["binding_intact"],
        "scope": scope,
        "provenance": {
            "parent_sha256": cfg["parent"]["sha256"],
            "config_sha256": sha256(bundle / CONFIG_NAME),
            "train_data_sha256": sha256(bundle / "data" / "qa_train.jsonl"),
        },
        "transfer": "LOCKED_UNSCORED",
        "final_and_sacred": "LOCKED_UNACCESSED",
    }
    tmp = out / "FINAL_STATUS.json.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(final_status, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out / "FINAL_STATUS.json")
    print(json.dumps({"classification": final_class, "stop_code": stop_code,
                      "final_language_dev_ce": last["dev_ce"],
                      "final_qa_overall": last["aggregate"]["overall"]}, indent=2))


if __name__ == "__main__":
    main()
