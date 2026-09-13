"""Phase 2A Treatment 2 runner.

QA objective: length-normalized hardest-distractor pairwise margin hinge,
M = 1.0 nat, candidate score = mean per-token log-likelihood of canonical
' Name.' (trailing period, no EOS). No prompt CE, no continuation CE, no EOS
supervision, no KL, no localization term. Language rehearsal unchanged (9:1).
TEST is scored exactly once at the terminal committed checkpoint only.
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

BUNDLE_DEFAULT = Path(r"C:\DaveLM-CADAVER\baby_vnext_phase2a_t2_ctxbind_v1")
ARCH = Path(r"C:\DaveLM-CADAVER\baby_vnext_60m_design_v1")
sys.path.insert(0, str(ARCH))

CONFIG_NAME = "PHASE2A_T2_CONFIG.json"
BASELINES_NAME = "PHASE2A_T2_BASELINES.json"


def margin_hinge_loss(scores, correct_index, M):
    """Length-normalized hardest-distractor pairwise margin hinge."""
    correct = scores[correct_index]
    others = [s for i, s in enumerate(scores) if i != correct_index]
    best_other = torch.stack(others).max()
    return torch.clamp(M - (correct - best_other), min=0.0)


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
    if sha256(cfg["parent"]["checkpoint"]) != cfg["parent"]["sha256"]:
        raise RuntimeError("parent hash mismatch")
    payload = torch.load(cfg["parent"]["checkpoint"], map_location="cpu", weights_only=False)
    assert payload["schema"] == "baby_vnext_phase1_model_v1"
    assert payload["completed_update"] == 6000
    model_cfg = BabyVNextConfig.load(ARCH / "BABY_VNEXT_CONFIG.json")
    model = BabyVNextWithBinding(model_cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload["model_state_dict"]


def set_scope(model):
    for n, p in model.named_parameters():
        if n.startswith("base_model."):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
            p.grad = None
    return {"trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad)}


def load_rows(bundle, name):
    out = []
    with open(bundle / "data" / name, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def _lang_tensors(bundle, cfg):
    dev_stream = _read_u16(cfg["language_eval"]["dev_stream"])
    sel = _read_u32(cfg["language_eval"]["selection"], 1280 + 320)
    train_stream = _read_u16(cfg["language_rehearsal"]["stream"])
    return dev_stream, sel[:1280], sel[1280:], train_stream


def load_schedules(bundle, cfg):
    kinds = json.loads((bundle / "data" / "update_kinds.json").read_text(encoding="utf-8"))["kinds"]
    n_qa = sum(1 for k in kinds if k != "lang")
    qa_sched = _read_u32(bundle / "data" / "qa_schedule.bin", n_qa * 32).view(n_qa, 32)
    n_lang = sum(1 for k in kinds if k == "lang")
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
            rows.append({"prompt": prompt, "immediate_eos": bool(new and new[0] == eos),
                         "three_identical": any(new[i] == new[i + 1] == new[i + 2]
                                                for i in range(max(0, len(new) - 2))),
                         "repeat_trigram": len(trigrams) != len(set(trigrams)),
                         "decoded": tokenizer.decode(new, skip_special_tokens=True)})
    finally:
        model.train(was)
    return rows


@torch.no_grad()
def candidate_scores(model, item, device):
    prompt = [2] + [int(x) for x in item["prompt_token_ids"]]
    P = len(prompt)
    scores = []
    for cand in item["candidate_token_ids"]:
        c = [int(t) for t in cand]
        x = torch.tensor([prompt + c], device=device)
        logits, _ = model(x)
        lp = F.log_softmax(logits[0, P - 1:P - 1 + len(c)], dim=-1)
        idx = torch.tensor(c, device=device)
        scores.append(float(lp[torch.arange(len(idx)), idx].mean()))
    return scores


def qa_eval(model, rows, device):
    correct = 0
    margins = []
    conf = []
    per_level = {}
    for r in rows:
        s = candidate_scores(model, r, device)
        ci = r["correct_index"]
        others = [v for i, v in enumerate(s) if i != ci]
        best_other = max(others)
        ok = s[ci] >= best_other
        correct += int(ok)
        margins.append(s[ci] - best_other)
        conf.append(max(s))
        lvl = r["level"]
        per_level.setdefault(lvl, {"correct": 0, "n": 0, "margins": []})
        per_level[lvl]["n"] += 1
        per_level[lvl]["correct"] += int(ok)
        per_level[lvl]["margins"].append(s[ci] - best_other)
    n = len(rows)
    out_levels = {}
    for k, v in per_level.items():
        out_levels[k] = {"correct": v["correct"], "n": v["n"],
                         "accuracy": v["correct"] / v["n"],
                         "margin": sum(v["margins"]) / v["n"]}
    return {"accuracy": correct / n, "n": n,
            "margin": sum(margins) / n, "confidence": sum(conf) / n,
            "per_level": out_levels}


def aggregate(dev, aud, novel):
    pl = dev["per_level"]

    def wavg(levels):
        c = sum(pl[l]["correct"] for l in levels)
        n = sum(pl[l]["n"] for l in levels)
        m = sum(pl[l]["margin"] * pl[l]["n"] for l in levels) / n
        return c / n, m

    ab, ab_m = wavg("AB")
    abcde, abcde_m = wavg("ABCDE")
    return {"ab": ab, "ab_margin": ab_m, "abcde": abcde, "abcde_margin": abcde_m,
            "overall": dev["accuracy"],
            "audit": aud["accuracy"], "audit_margin": aud["margin"],
            "novel": novel["accuracy"], "novel_margin": novel["margin"],
            "per_level": {l: pl[l]["accuracy"] for l in "ABCDEFGH"}}


def reversal_rate(dev_rows, model, device):
    from collections import defaultdict
    fam = defaultdict(list)
    for r in dev_rows:
        if r["level"] in ("A", "B") and isinstance(r.get("style"), dict) and "assign" in r["style"]:
            fam[r["family_id"]].append(r)
    pairs = 0
    both = 0
    for items in fam.values():
        by = defaultdict(dict)
        for r in items:
            by[(r["style"].get("q"), r["style"].get("order"))][r["style"].get("assign")] = r
        for key, d in by.items():
            if 0 in d and 1 in d:
                pairs += 1
                r0, r1 = d[0], d[1]
                s0 = candidate_scores(model, r0, device)
                s1 = candidate_scores(model, r1, device)
                ok0 = s0.index(max(s0)) == r0["correct_index"]
                ok1 = s1.index(max(s1)) == r1["correct_index"]
                both += int(ok0 and ok1)
    return {"pairs": pairs, "rate": (both / pairs if pairs else 0.0)}


def run_eval(model, bundle, cfg, tok, dev_rows, aud_rows, nov_rows, dev_stream,
             dev_starts, train_fit, rehearse_stream, context, device):
    dev_ce = evaluate_loss(model, dev_stream, dev_starts, context, 16, device)
    train_ce = evaluate_loss(model, rehearse_stream, train_fit, context, 16, device)
    dev = qa_eval(model, dev_rows, device)
    aud = qa_eval(model, aud_rows, device)
    nov = qa_eval(model, nov_rows, device)
    prompts = json.loads(Path(cfg["language_eval"]["prompts"]).read_text(encoding="utf-8"))["prompts"]
    gen = greedy_diag(model, tok, prompts, context, device, 32)
    return {
        "dev_ce": dev_ce, "train_ce": train_ce, "dev_perplexity": math.exp(dev_ce),
        "qa_dev": dev, "qa_aud": aud, "qa_novel": nov,
        "aggregate": aggregate(dev, aud, nov),
        "reversal": reversal_rate(dev_rows, model, device),
        "generation": {"non_immediate_eos": sum(not g["immediate_eos"] for g in gen),
                       "no_three_identical": sum(not g["three_identical"] for g in gen),
                       "no_repeat_trigram": sum(not g["repeat_trigram"] for g in gen)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("validate", "u0", "train"), required=True)
    ap.add_argument("--bundle", type=Path, default=BUNDLE_DEFAULT)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--allow-preseal", action="store_true")
    args = ap.parse_args()
    bundle = args.bundle.resolve()
    cfg = json.loads((bundle / CONFIG_NAME).read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    device = torch.device("cuda")

    tok = Tokenizer.from_file(cfg["tokenizer"]["path"])
    dev_rows = load_rows(bundle, "qa_dev.jsonl")
    aud_rows = load_rows(bundle, "qa_audit_dev.jsonl")
    nov_rows = load_rows(bundle, "qa_novel_dev.jsonl")
    train_items = load_rows(bundle, "qa_train.jsonl")
    dev_stream, dev_starts, train_fit, rehearse_stream = _lang_tensors(bundle, cfg)
    kinds, qa_sched, rehear = load_schedules(bundle, cfg)
    L_Q = int(cfg["data"]["L_Q"])
    context = int(cfg["language_rehearsal"]["context"])
    M = float(cfg["objective"]["M"])

    model, parent_sd = load_model_from_parent(cfg)
    scope = set_scope(model)
    model.to(device)
    binding_names = [n for n in model.state_dict() if not n.startswith("base_model.")]
    parent_binding = {n: parent_sd[n].clone().to(device) for n in binding_names}

    def binding_intact():
        sd = model.state_dict()
        return all(bool((sd[n] == parent_binding[n]).all()) for n in binding_names)

    configure_runtime(int(cfg["seeds"]["primary"]))

    def run_eval_full(update):
        rec = run_eval(model, bundle, cfg, tok, dev_rows, aud_rows, nov_rows,
                       dev_stream, dev_starts, train_fit, rehearse_stream, context, device)
        rec["update"] = update
        rec["binding_intact"] = binding_intact()
        return rec

    def write_json(path, obj):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    if args.mode == "validate":
        if not args.allow_preseal:
            raise RuntimeError("validate requires --allow-preseal")
        result = {"status": "PASS", "scope": scope,
                  "scope_ok": scope == {"trainable": 60536064, "frozen": 984321},
                  "qa_dev_items": len(dev_rows), "qa_audit_dev": len(aud_rows),
                  "qa_novel_dev": len(nov_rows), "train_items": len(train_items),
                  "L_Q": L_Q, "M": M,
                  "test_files_present": all((bundle / "data" / n).is_file() for n in
                                            ("qa_test.jsonl", "qa_audit_test.jsonl", "qa_novel_test.jsonl")),
                  "test_contents_loaded": False,
                  "optimizer_created": False, "training_performed": False}
        write_json(bundle / "RUNNER_VALIDATION.json", result)
        print(json.dumps(result, indent=2))
        return

    if args.mode == "u0":
        if not args.allow_preseal:
            raise RuntimeError("u0 requires --allow-preseal")
        rec = run_eval_full(0)
        base = {"status": "PHASE2A_T2_U0_BASELINE_FROZEN", "primary_u0": rec,
                "scope": scope, "test_contents_loaded": False,
                "optimizer_created": False, "optimizer_steps": 0, "training_performed": False}
        write_json(bundle / BASELINES_NAME, base)
        print(json.dumps({"status": "PHASE2A_T2_U0_DONE",
                          "dev_ce": rec["dev_ce"], "qa_dev_acc": rec["qa_dev"]["accuracy"],
                          "ab": rec["aggregate"]["ab"], "abcde": rec["aggregate"]["abcde"],
                          "audit": rec["aggregate"]["audit"], "novel": rec["aggregate"]["novel"],
                          "reversal": rec["reversal"], "confidence": rec["qa_dev"]["confidence"],
                          "margin": rec["qa_dev"]["margin"]}, indent=2))
        return

    # ---------------- train ------------------------------------------------
    if os.environ.get("PYTHONHASHSEED") != str(cfg["seeds"]["primary"]):
        raise RuntimeError("PYTHONHASHSEED mismatch")
    out = args.output_dir.resolve()
    if args.resume is None and out.exists():
        raise RuntimeError("fresh output dir already exists")
    out.mkdir(parents=True, exist_ok=True)
    baselines = json.loads((bundle / BASELINES_NAME).read_text(encoding="utf-8"))["primary_u0"]

    u0 = run_eval_full(0)
    if abs(u0["dev_ce"] - baselines["dev_ce"]) > 1e-4 or abs(u0["train_ce"] - baselines["train_ce"]) > 1e-4:
        raise RuntimeError("language U0 reproduction failed")
    evaluations = [u0]
    write_json(out / "evaluation_0000.json", u0)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=5e-5, betas=(0.9, 0.999), eps=1e-8,
                                  weight_decay=0.05, amsgrad=False, foreach=False, fused=False)
    completed = 0
    if args.resume is not None:
        st = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model_state"], strict=True)
        optimizer.load_state_dict(st["optimizer"])
        random.setstate(st["py_rng"])
        torch.set_rng_state(st["torch_rng"])
        torch.cuda.set_rng_state_all(st["cuda_rng"])
        completed = int(st["completed"])
        evaluations = st.get("evaluations", evaluations)

    def atomic_torch(path, obj):
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        with open(tmp, "r+b") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def restart_state(c):
        return {"completed": c, "model_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                "py_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "evaluations": evaluations}

    def qa_item_loss(item):
        prompt = [2] + [int(x) for x in item["prompt_token_ids"]]
        P = len(prompt)
        scores = []
        for cand in item["candidate_token_ids"]:
            c = [int(t) for t in cand]
            x = torch.tensor([prompt + c], device=device)
            logits, _ = model(x)
            lp = F.log_softmax(logits[0, P - 1:P - 1 + len(c)], dim=-1)
            idx = torch.tensor(c, device=device)
            scores.append(lp[torch.arange(len(idx)), idx].mean())
        ci = item["correct_index"]
        return margin_hinge_loss(scores, ci, M)

    def qa_update_loss(indices):
        micro = 8
        total = 0.0
        n_micro = 0
        for b in range(0, len(indices), micro):
            group = indices[b:b + micro]
            loss = sum(qa_item_loss(train_items[i]) for i in group) / len(group)
            (loss / math.ceil(len(indices) / micro)).backward()
            total += float(loss.detach().cpu())
            n_micro += 1
        return total / n_micro

    def lang_update_loss(starts_row):
        offsets = torch.arange(context)
        accum = 4
        total = 0.0
        for k in range(accum):
            s = starts_row[k * 16:(k + 1) * 16]
            x = rehearse_stream[s[:, None] + offsets[None, :]].to(device)
            y = rehearse_stream[s[:, None] + offsets[None, :] + 1].to(device)
            logits, _ = model(x)
            raw = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            (raw / accum).backward()
            total += float(raw.detach().cpu()) / accum
        return total

    metrics_path = out / "training_metrics.jsonl"
    qa_counter = sum(1 for k in kinds[:completed] if k != "lang")
    lang_counter = sum(1 for k in kinds[:completed] if k == "lang")

    g = cfg["gates"]
    fin = g["final"]
    a0 = baselines["aggregate"]
    c0 = baselines["qa_dev"]["confidence"]
    m0 = baselines["qa_dev"]["margin"]

    def agg_for(u):
        return next(e for e in evaluations if e["update"] == u)["aggregate"]

    def conf_for(u):
        return next(e for e in evaluations if e["update"] == u)["qa_dev"]["confidence"]

    def marg_for(u):
        return next(e for e in evaluations if e["update"] == u)["qa_dev"]["margin"]

    def v1_mode(rec):
        dc = rec["qa_dev"]["confidence"] - c0
        dm = rec["qa_dev"]["margin"] - m0
        return bool(dc >= fin["v1_dc_min"] and dm <= fin["v1_dm_max"]), dc, dm

    def base_checks(u):
        e = next(x for x in evaluations if x["update"] == u)
        ag = e["aggregate"]
        v1flag, dc, dm = v1_mode(e)
        prior = [x for x in (100, 250) if x < u]
        mono = all(marg_for(p) - e["qa_dev"]["margin"] <= fin["monotonic_tol"] for p in prior)
        return {
            "lang_dev_ce": e["dev_ce"] <= fin["lang_dev_ce_max"],
            "binding_intact": e["binding_intact"],
            "ab_acc": ag["ab"] >= fin["ab_acc_min"],
            "ab_margin": ag["ab_margin"] >= fin["ab_margin_min"],
            "abcde_acc": ag["abcde"] >= fin["abcde_acc_min"],
            "per_level_min": all(ag["per_level"][l] >= fin["per_level_min"] for l in "ABCDEFGH"),
            "audit_acc": ag["audit"] >= fin["aud_acc_min"],
            "audit_margin": ag["audit_margin"] >= fin["aud_margin_min"],
            "novel_acc": ag["novel"] >= fin["novel_acc_min"],
            "generation": (e["generation"]["non_immediate_eos"] >= fin["gen_non_imm_eos"]
                           and e["generation"]["no_three_identical"] >= fin["gen_no_triple"]
                           and e["generation"]["no_repeat_trigram"] >= fin["gen_no_repeat_trigram"]),
            "v1_diagnostic": (not v1flag) and (dm >= fin["v1_dm_min"]),
            "reversal": e["reversal"]["rate"] >= fin["reversal_min"],
            "monotonic": mono,
        }

    def run_test_once():
        test_core = load_rows(bundle, "qa_test.jsonl")
        test_aud = load_rows(bundle, "qa_audit_test.jsonl")
        test_nov = load_rows(bundle, "qa_novel_test.jsonl")
        core = qa_eval(model, test_core, device)
        aud = qa_eval(model, test_aud, device)
        nov = qa_eval(model, test_nov, device)
        rec = {"update": "test", "core": core, "audit": aud, "novel": nov}
        write_json(out / "evaluation_test.json", rec)
        return rec

    final_class = None
    stop_code = None
    test_used = False
    for update in range(completed + 1, 501):
        kind = kinds[update - 1]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if kind == "lang":
            loss = lang_update_loss(rehear[lang_counter])
            lang_counter += 1
        else:
            loss = qa_update_loss(qa_sched[qa_counter].tolist())
            qa_counter += 1
        bad = [n for n, p in model.named_parameters() if p.grad is not None and not n.startswith("base_model.")]
        if bad:
            raise RuntimeError(f"binding gradient at {update}: {bad}")
        active = [p for p in model.parameters() if p.requires_grad]
        gn = torch.nn.utils.clip_grad_norm_(active, 2.0)
        if not bool(torch.isfinite(gn)):
            raise RuntimeError(f"non-finite grad {update}")
        optimizer.step()
        if not math.isfinite(float(loss)):
            raise RuntimeError(f"non-finite loss {update}")
        with open(metrics_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"update": update, "kind": kind, "loss": float(loss),
                                "grad_norm": float(gn)}) + "\n")
        if update % 50 == 0:
            atomic_torch(out / "rolling_restart.pt", restart_state(update))
        if update not in (100, 250, 500):
            continue
        rec = run_eval_full(update)
        evaluations.append(rec)
        write_json(out / f"evaluation_{update:04d}.json", rec)
        ck = out / "checkpoints"
        ck.mkdir(exist_ok=True)
        atomic_torch(ck / f"checkpoint_{update:04d}.pt",
                     {"schema": "baby_vnext_phase2a_t2_model_v1", "update": update,
                      "model_state_dict": model.state_dict()})
        atomic_torch(out / "rolling_restart.pt", restart_state(update))

        v1flag, dc, dm = v1_mode(rec)
        hard = (not rec["binding_intact"]) or (not math.isfinite(rec["dev_ce"]))
        if hard:
            final_class, stop_code = "STOP_HARD", "hard_stop"
            break
        if update == 100:
            if rec["dev_ce"] > g["u100"]["lang_dev_ce_max"]:
                final_class, stop_code = "STOP_EARLY", "lang"
                break
            if (rec["aggregate"]["ab"] < g["u100"]["ab_acc_min"]
                    and rec["aggregate"]["ab_margin"] < g["u100"]["ab_margin_min"]):
                final_class, stop_code = "STOP_EARLY", "ab_no_progress"
                break
            if v1flag:
                final_class, stop_code = "STOP_EARLY", "V1_FAILURE_MODE"
                break
        elif update == 250:
            if rec["dev_ce"] > g["u250"]["lang_dev_ce_max"]:
                final_class, stop_code = "STOP_EARLY", "lang"
                break
            if (rec["aggregate"]["abcde"] < g["u250"]["abcde_acc_min"]
                    or rec["aggregate"]["abcde_margin"] < g["u250"]["abcde_margin_min"]):
                final_class, stop_code = "STOP_EARLY", "abcde_no_progress"
                break
            if (rec["aggregate"]["audit"] < g["u250"]["aud_acc_min"]
                    or rec["aggregate"]["audit_margin"] <= g["u250"]["aud_margin_min"]):
                final_class, stop_code = "STOP_EARLY", "audit_no_progress"
                break
            if v1flag:
                final_class, stop_code = "STOP_EARLY", "V1_FAILURE_MODE"
                break
        # early-success / terminal adjudication
        checks = base_checks(update)
        if update == 500:
            ta = run_test_once()
            test_used = True
            checks["test_core"] = ta["core"]["accuracy"] >= fin["test_core_min"]
            checks["test_audit"] = ta["audit"]["accuracy"] >= fin["test_aud_min"]
            checks["test_novel"] = ta["novel"]["accuracy"] >= fin["test_novel_min"]
            final_class = "SUCCESS" if all(checks.values()) else "FAIL"
            stop_code = {"checks": checks} if final_class == "FAIL" else None
            break
        else:
            if all(checks.values()):
                ta = run_test_once()
                test_used = True
                ok = (ta["core"]["accuracy"] >= fin["test_core_min"]
                      and ta["audit"]["accuracy"] >= fin["test_aud_min"]
                      and ta["novel"]["accuracy"] >= fin["test_novel_min"])
                final_class = "EARLY_SUCCESS" if ok else "FAIL"
                stop_code = None if ok else "test"
                break

    last = evaluations[-1]
    status = {
        "status": final_class, "stop_code": stop_code, "test_used": test_used,
        "eval_updates": [e["update"] for e in evaluations],
        "dev_ce_trajectory": {int(e["update"]): e["dev_ce"] for e in evaluations},
        "qa_dev_trajectory": {int(e["update"]): {"accuracy": e["qa_dev"]["accuracy"],
                                                 "margin": e["qa_dev"]["margin"],
                                                 "confidence": e["qa_dev"]["confidence"],
                                                 "ab": e["aggregate"]["ab"],
                                                 "abcde": e["aggregate"]["abcde"],
                                                 "audit": e["aggregate"]["audit"],
                                                 "novel": e["aggregate"]["novel"]}
                              for e in evaluations},
        "reversal_trajectory": {int(e["update"]): e["reversal"] for e in evaluations},
        "generation_trajectory": {int(e["update"]): e["generation"] for e in evaluations},
        "binding_intact_final": last["binding_intact"],
        "scope": scope,
        "v1_diagnostic_final": {"dc": last["qa_dev"]["confidence"] - c0,
                                "dm": last["qa_dev"]["margin"] - m0},
        "provenance": {"parent_sha256": cfg["parent"]["sha256"],
                       "config_sha256": sha256(bundle / CONFIG_NAME),
                       "train_data_sha256": sha256(bundle / "data" / "qa_train.jsonl")},
        "transfer": "LOCKED_UNSCORED", "final_and_sacred": "LOCKED_UNACCESSED",
        "v1_qa_eval_test": "LOCKED_UNOPENED",
    }
    write_json(out / "FINAL_STATUS.json", status)
    print(json.dumps({"classification": final_class, "stop_code": stop_code,
                      "test_used": test_used, "last_update": last["update"],
                      "dev_ce": last["dev_ce"],
                      "qa_dev_acc": last["qa_dev"]["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
