"""Phase 2A T8: late-base + binding, loc CE + layout margin, NO T3 pointer.

Never loads T3 TEST.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from array import array
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

BUNDLE_DEFAULT = Path(r"C:\DaveLM-CADAVER\phase2a_t8_latebase_loc_layout_v1")
T3_DATA = Path(r"C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1\data")
ARCH = Path(r"C:\DaveLM-CADAVER\baby_vnext_60m_design_v1")
sys.path.insert(0, str(ARCH))
from baby_vnext.binding import BindingLayout

PARENT = Path(r"C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001\checkpoints\best.pt")
PARENT_SHA = "c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1"
TOK_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
TOK_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
LANG_TRAIN = Path(r"C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\LANGUAGE_TRAIN_STREAM.u16")
LANG_DEV = Path(r"C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\LANGUAGE_DEV_STREAM.u16")
LANG_STARTS = Path(r"C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\EVAL_WINDOW_STARTS.u32")
GEN_PROMPTS = Path(r"C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\GENERATION_PROMPTS.json")
M = 1.0
TAU = 1.0
LAMBDA_LOC = 1.0
CTX = 256
EOS = 3
SEEDS = (670001, 670002, 670003)
SCOPE_TRAINABLE = 41833985
SCOPE_FROZEN = 19686400
U0_CE = 1.2040123894810677


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    with open(tmp, "r+b") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_u16(path: Path) -> torch.Tensor:
    raw = array("H")
    with open(path, "rb") as f:
        raw.fromfile(f, os.path.getsize(path) // 2)
    if sys.byteorder != "little":
        raw.byteswap()
    return torch.tensor(raw, dtype=torch.long)


def read_u32(path: Path, count: int) -> torch.Tensor:
    raw = array("I")
    with open(path, "rb") as f:
        raw.fromfile(f, count)
    if sys.byteorder != "little":
        raw.byteswap()
    return torch.tensor(raw, dtype=torch.long)


def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def configure_runtime(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def freeze_early(name: str) -> bool:
    return any(name.startswith(f"base_model.blocks.{i}.") for i in range(4))


def load_model():
    from baby_vnext.config import BabyVNextConfig
    from baby_vnext.binding import BabyVNextWithBinding
    if sha256(PARENT) != PARENT_SHA or sha256(TOK_PATH) != TOK_SHA:
        raise RuntimeError("parent/tokenizer hash mismatch")
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    if payload.get("schema") != "baby_vnext_phase1_model_v1" or payload.get("completed_update") != 6000:
        raise RuntimeError("parent schema/update mismatch")
    model = BabyVNextWithBinding(BabyVNextConfig.load(ARCH / "BABY_VNEXT_CONFIG.json"))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload["model_state_dict"]


def set_scope(model):
    for n, p in model.named_parameters():
        p.requires_grad_(not freeze_early(n))
        if freeze_early(n):
            p.grad = None
    return {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }


def make_layout(row, device):
    keys = torch.tensor([[int(s[0]) for s in row["mention_span_bos"]]], device=device)
    vals = torch.tensor([[int(s[-1]) for s in row["mention_span_bos"]]], device=device)
    q = torch.tensor([int(row["decision_position_bos"])], device=device)
    valid = torch.ones(1, keys.shape[1], dtype=torch.bool, device=device)
    return BindingLayout(q, q, keys, vals, valid)


def qa_losses(model, row, device):
    prompt = [2] + [int(x) for x in row["prompt_token_ids"]]
    p_len = len(prompt)
    x = torch.tensor([prompt], device=device)
    hs = model.base_model.forward_hidden(x)
    q = F.normalize(hs[0, p_len - 1], dim=-1)
    sims = []
    for span in row["mention_span_bos"]:
        idx = torch.tensor(span, device=device, dtype=torch.long)
        k = F.normalize(hs[0, idx].mean(0), dim=-1)
        sims.append((q * k).sum() / TAU)
    sims_t = torch.stack(sims)
    ci = int(row["correct_index"])
    ptr = F.cross_entropy(sims_t.view(1, -1), torch.tensor([ci], device=device))
    layout = make_layout(row, device)
    scores = []
    loc_logits = None
    for cand in row["candidate_token_ids"]:
        c = [int(t) for t in cand]
        xx = torch.tensor([prompt + c], device=device)
        logits, info = model(xx, layout=layout)
        lp = F.log_softmax(logits[0, p_len - 1:p_len - 1 + len(c)], dim=-1)
        tok = torch.tensor(c, device=device)
        scores.append(lp[torch.arange(len(c), device=device), tok].mean())
        if loc_logits is None:
            loc_logits = info["localization_scores"][0].mean(-1)
    other = torch.stack([s for j, s in enumerate(scores) if j != ci]).max()
    mar = torch.clamp(torch.tensor(M, device=device) - (scores[ci] - other), min=0.0)
    loc = F.cross_entropy(loc_logits.view(1, -1), torch.tensor([ci], device=device))
    return ptr, loc, mar, scores, sims_t, loc_logits


@torch.no_grad()
def lang_ce(model, stream, starts, device, microbatch=8):
    was = model.training
    model.eval()
    total = 0.0
    tokens = 0
    off = torch.arange(CTX)
    try:
        for i in range(0, len(starts), microbatch):
            s = starts[i:i + microbatch]
            x = stream[s[:, None] + off[None, :]].to(device)
            y = stream[s[:, None] + off[None, :] + 1].to(device)
            logits, _ = model(x)
            total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                           y.reshape(-1), reduction="sum").cpu())
            tokens += y.numel()
    finally:
        model.train(was)
    return total / tokens


@torch.no_grad()
def greedy_diag(model, tokenizer, prompts, device, max_new=32):
    was = model.training
    model.eval()
    eos = tokenizer.token_to_id("<eos>")
    rows = []
    try:
        for prompt in prompts:
            ids = tokenizer.encode(prompt).ids
            gen = [2] + ids
            new = []
            for _ in range(max_new):
                inp = torch.tensor([gen[-CTX:]], device=device)
                logits, _ = model(inp)
                t = int(logits[0, -1].argmax())
                gen.append(t)
                new.append(t)
                if t == eos:
                    break
            trigrams = [tuple(new[i:i + 3]) for i in range(max(0, len(new) - 2))]
            rows.append({
                "immediate_eos": bool(new and new[0] == eos),
                "three_identical": any(new[i] == new[i + 1] == new[i + 2]
                                       for i in range(max(0, len(new) - 2))),
                "repeat_trigram": len(trigrams) != len(set(trigrams)),
            })
    finally:
        model.train(was)
    return {
        "non_immediate_eos": sum(not g["immediate_eos"] for g in rows),
        "no_three_identical": sum(not g["three_identical"] for g in rows),
        "no_repeat_trigram": sum(not g["repeat_trigram"] for g in rows),
        "n": len(rows),
    }


@torch.no_grad()
def exact_answer_eos(model, row, device):
    prompt = [2] + [int(x) for x in row["prompt_token_ids"]]
    target = [int(t) for t in row["candidate_token_ids"][int(row["correct_index"])]] + [EOS]
    layout = make_layout(row, device)
    gen = list(prompt)
    for _ in range(len(target) + 2):
        inp = torch.tensor([gen[-CTX:]], device=device)
        if len(gen) == len(prompt):
            logits, _ = model(inp, layout=layout)
        else:
            logits, _ = model(inp)
        t = int(logits[0, -1].argmax())
        gen.append(t)
        if t == EOS:
            break
    return gen[len(prompt):] == target


def eval_panel(model, rows, device):
    was = model.training
    model.eval()
    n = len(rows)
    ptr_ok = loc_ok = fc_ok = exact = 0
    margins = []
    per_row = []
    try:
        for r in rows:
            ptr, loc, mar, scores, sims, loc_logits = qa_losses(model, r, device)
            ci = int(r["correct_index"])
            pred_ptr = int(torch.argmax(sims).item())
            pred_loc = int(torch.argmax(loc_logits).item())
            pred_fc = int(max(range(len(scores)), key=lambda j: float(scores[j].detach())))
            p_ok = pred_ptr == ci
            l_ok = pred_loc == ci
            f_ok = pred_fc == ci
            ex = exact_answer_eos(model, r, device)
            ptr_ok += int(p_ok)
            loc_ok += int(l_ok)
            fc_ok += int(f_ok)
            exact += int(ex)
            others = [float(scores[j].detach()) for j in range(len(scores)) if j != ci]
            margins.append(float(scores[ci].detach()) - max(others))
            per_row.append({
                "id": r["id"], "family_id": r["family_id"],
                "pointer_ok": p_ok, "loc_ok": l_ok, "native_ok": f_ok, "exact_ok": ex,
                "correct_index": ci, "pred_pointer": pred_ptr, "pred_loc": pred_loc,
                "pred_native": pred_fc, "assignment": r["assignment"],
                "query_index": r["query_index"], "fact_order": r["fact_order"],
            })
    finally:
        model.train(was)
    return {
        "n": n,
        "pointer_retrieval": ptr_ok,
        "localization_retrieval": loc_ok,
        "native_forced_choice": fc_ok,
        "exact_answer_eos": exact,
        "mean_margin": (sum(margins) / n) if n else 0.0,
        "per_row": per_row,
    }


def assignment_reversals(per_row, field):
    groups = defaultdict(dict)
    for r in per_row:
        groups[(r["family_id"], r["query_index"], r["fact_order"])][r["assignment"]] = r
    pairs = both = 0
    for d in groups.values():
        if 0 in d and 1 in d:
            pairs += 1
            both += int(d[0][field] and d[1][field])
    return {"pairs": pairs, field: both}


def complete_families(per_row, field):
    fam = defaultdict(list)
    for r in per_row:
        fam[r["family_id"]].append(r)
    ok = sum(1 for items in fam.values() if len(items) == 8 and all(x[field] for x in items))
    return {"families": len(fam), "complete": ok}


def classify(rec, u0_ce):
    if rec.get("hard_stop") or not math.isfinite(rec["dev_ce"]) or not rec.get("early_intact", True):
        return "T8_HARD_STOP"
    lang_cap = min(1.30, u0_ce + 0.25)
    lang_ok = rec["dev_ce"] <= lang_cap and rec["gap"] <= 0.5
    loc = rec["dev"]["localization_retrieval"]
    rev_l = rec["dev_reversals_loc"]["loc_ok"]
    fam = rec["dev_complete_loc"]["complete"]
    routing = loc >= 96 and rev_l >= 48 and fam >= 8
    nat = rec["dev"]["native_forced_choice"] >= 96 and rec["dev_reversals_nat"]["native_ok"] >= 48
    if not lang_ok:
        return "T8_LANGUAGE_REGRESSION"
    if not routing:
        return "T8_FAIL_NO_ROUTING"
    if not nat:
        return "T8_ROUTING_SUCCESS_OUTPUT_FAIL"
    return "T8_BINDING_SUCCESS"


def make_qa_schedule(seed, n_train, n_qa=450, batch=32):
    idx = list(range(n_train))
    rng = random.Random(seed + 101)
    rng.shuffle(idx)
    order = []
    cursor = 0
    for _ in range(n_qa):
        order.append([idx[(cursor + j) % n_train] for j in range(batch)])
        cursor += batch
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("train",), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--bundle", type=Path, default=BUNDLE_DEFAULT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.seed not in SEEDS:
        raise RuntimeError("unauthorized seed")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    # Never pass T3 qa_test.jsonl to load_jsonl.
    if not (T3_DATA / "qa_test.jsonl").exists():
        raise RuntimeError("expected sealed T3 TEST file to exist unread")
    device = torch.device("cuda")
    configure_runtime(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    tok = Tokenizer.from_file(str(TOK_PATH))
    train_rows = load_jsonl(T3_DATA / "qa_train.jsonl")
    dev_rows = load_jsonl(T3_DATA / "qa_dev.jsonl")
    acq_rows = json.loads((T3_DATA / "acq16.json").read_text(encoding="utf-8"))
    seal = json.loads((T3_DATA / "TEST_SEAL.json").read_text(encoding="utf-8"))
    if seal["status"] != "SEALED_UNOPENED":
        raise RuntimeError("T3 TEST seal unexpected")
    kinds = json.loads((T3_DATA / "update_kinds.json").read_text(encoding="utf-8"))["kinds"]
    rehear = json.loads((T3_DATA / "rehearsal_index.json").read_text(encoding="utf-8"))["starts_mod_320"]
    bundle = args.bundle.resolve()
    sched_path = bundle / "data" / "qa_schedules.json"
    if sched_path.is_file():
        qa_sched = json.loads(sched_path.read_text(encoding="utf-8"))[str(args.seed)]
    else:
        schedules = {str(s): make_qa_schedule(s, len(train_rows)) for s in SEEDS}
        write_json(sched_path, schedules)
        qa_sched = schedules[str(args.seed)]
    stream_train = read_u16(LANG_TRAIN)
    stream_dev = read_u16(LANG_DEV)
    starts_all = read_u32(LANG_STARTS, 1600)
    dev_starts = starts_all[:1280]
    train_fit = starts_all[1280:]
    gen_prompts = json.loads(GEN_PROMPTS.read_text(encoding="utf-8"))["prompts"]
    model, parent_sd = load_model()
    scope = set_scope(model)
    if scope != {"trainable": SCOPE_TRAINABLE, "frozen": SCOPE_FROZEN}:
        raise RuntimeError(f"scope mismatch {scope}")
    model.to(device)
    early_names = [n for n in model.state_dict() if freeze_early(n)]
    parent_early = {n: parent_sd[n].clone().to(device) for n in early_names}

    def early_intact():
        sd = model.state_dict()
        return all(bool((sd[n] == parent_early[n]).all()) for n in early_names)

    def run_full(update):
        model.eval()
        dev = eval_panel(model, dev_rows, device)
        acq = eval_panel(model, acq_rows, device)
        dce = lang_ce(model, stream_dev, dev_starts, device, 16)
        tce = lang_ce(model, stream_train, train_fit, device, 16)
        rec = {
            "update": update, "dev_ce": dce, "train_ce": tce, "gap": dce - tce,
            "early_intact": early_intact(), "t3_test_loaded": False,
            "dev": {k: v for k, v in dev.items() if k != "per_row"},
            "dev_reversals_ptr": assignment_reversals(dev["per_row"], "pointer_ok"),
            "dev_reversals_loc": assignment_reversals(dev["per_row"], "loc_ok"),
            "dev_reversals_nat": assignment_reversals(dev["per_row"], "native_ok"),
            "dev_complete_ptr": complete_families(dev["per_row"], "pointer_ok"),
            "dev_complete_loc": complete_families(dev["per_row"], "loc_ok"),
            "dev_complete_nat": complete_families(dev["per_row"], "native_ok"),
            "acq16": {k: v for k, v in acq.items() if k != "per_row"},
            "generation": greedy_diag(model, tok, gen_prompts, device),
            "scope": scope, "item_results": dev["per_row"],
        }
        return rec

    slim = lambda r: {k: v for k, v in r.items() if k != "item_results"}
    out = args.out.resolve()
    resume_from = 0
    if args.resume:
        restart = torch.load(out / "rolling_restart.pt", map_location="cpu", weights_only=False)
        resume_from = int(restart["completed"])
        model.load_state_dict(restart["model_state_dict"], strict=True)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05,
            foreach=False, fused=False,
        )
        optimizer.load_state_dict(restart["optimizer"])
        for state in optimizer.state.values():
            for k, v in list(state.items()):
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        u0_ce = float(json.loads((out / "evaluation_0000.json").read_text(encoding="utf-8"))["dev_ce"])
        write_json(out / "STATUS.json", {
            "status": "TRAINING_RUNNING", "seed": args.seed, "update": resume_from, "resumed": True,
            "parent_sha256": PARENT_SHA,
        })
    else:
        if out.exists() and any(out.iterdir()):
            raise RuntimeError("fresh output dir already exists")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "STATUS.json", {
            "status": "TRAINING_RUNNING", "seed": args.seed, "update": 0, "parent_sha256": PARENT_SHA,
        })
        u0 = run_full(0)
        u0_ce = u0["dev_ce"]
        if abs(u0_ce - U0_CE) > 1e-3:
            raise RuntimeError(f"language U0 failed {u0_ce}")
        write_json(out / "evaluation_0000.json", slim(u0))
        write_json(out / "item_results_0000.json", {"dev": u0["item_results"]})
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05,
            foreach=False, fused=False,
        )

    metrics_path = out / "training_metrics.jsonl"
    qa_i = sum(1 for k in kinds[:resume_from] if k != "lang")
    lang_i = sum(1 for k in kinds[:resume_from] if k == "lang")
    final_class = None
    stop_code = None

    def qa_update(indices):
        micro = 8
        total = 0.0
        n_micro = 0
        for b in range(0, len(indices), micro):
            group = indices[b:b + micro]
            loss = 0.0
            for j in group:
                ptr, loc, mar, _, _, _ = qa_losses(model, train_rows[j], device)
                loss = loss + (mar + LAMBDA_LOC * loc) / len(group)
            (loss / math.ceil(len(indices) / micro)).backward()
            total += float(loss.detach().cpu())
            n_micro += 1
        return total / n_micro

    def lang_update(mods):
        starts = train_fit[torch.tensor(mods, dtype=torch.long)]
        off = torch.arange(CTX)
        accum = 4
        total = 0.0
        for k in range(accum):
            s = starts[k * 16:(k + 1) * 16]
            x = stream_train[s[:, None] + off[None, :]].to(device)
            y = stream_train[s[:, None] + off[None, :] + 1].to(device)
            logits, _ = model(x)
            raw = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            (raw / accum).backward()
            total += float(raw.detach().cpu()) / accum
        return total

    for update in range(resume_from + 1, 501):
        kind = kinds[update - 1]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if kind == "lang":
            loss = lang_update(rehear[lang_i])
            lang_i += 1
        else:
            loss = qa_update(qa_sched[qa_i])
            qa_i += 1
        bad = [n for n, p in model.named_parameters() if p.grad is not None and freeze_early(n)]
        if bad:
            raise RuntimeError(f"frozen-block gradient at {update}: {bad}")
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0)
        if not bool(torch.isfinite(gn)) or not math.isfinite(float(loss)):
            raise RuntimeError(f"nonfinite at {update}")
        optimizer.step()
        with open(metrics_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"update": update, "kind": kind, "loss": float(loss),
                                "grad_norm": float(gn)}) + "\n")
        if update % 50 == 0:
            atomic_torch(out / "rolling_restart.pt", {
                "completed": update,
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            })
            write_json(out / "STATUS.json", {
                "status": "TRAINING_RUNNING", "seed": args.seed, "update": update,
            })
        if update not in (100, 300, 500):
            continue
        rec = run_full(update)
        rec["hard_stop"] = (not rec["early_intact"]) or (not math.isfinite(rec["dev_ce"])) or rec["dev_ce"] > 1.50
        rec["classification"] = classify(rec, u0_ce)
        write_json(out / f"evaluation_{update:04d}.json", slim(rec))
        write_json(out / f"item_results_{update:04d}.json", {"dev": rec["item_results"]})
        ckdir = out / "checkpoints"
        ckdir.mkdir(exist_ok=True)
        atomic_torch(ckdir / f"checkpoint_{update:04d}.pt", {
            "schema": "baby_vnext_phase2a_t8_latebase_loc_layout_model_v1",
            "update": update, "seed": args.seed,
            "model_state_dict": model.state_dict(),
        })
        if rec["hard_stop"]:
            final_class, stop_code = "T8_HARD_STOP", "integrity_or_catastrophe"
            break
        if rec["dev_ce"] > 1.30:
            final_class, stop_code = "T8_LANGUAGE_REGRESSION", "language_ce_gt_1_30"
            break
        if update == 500:
            final_class = rec["classification"]

    last = json.loads(sorted(out.glob("evaluation_*.json"))[-1].read_text(encoding="utf-8"))
    status = {
        "seed": args.seed, "update": last["update"],
        "classification": final_class or last.get("classification"),
        "stop_code": stop_code, "test_opened": False, "t3_test_loaded": False,
        "dev_pointer": last["dev"]["pointer_retrieval"],
        "dev_loc": last["dev"]["localization_retrieval"],
        "dev_native": last["dev"]["native_forced_choice"],
        "dev_exact": last["dev"]["exact_answer_eos"],
        "dev_ce": last["dev_ce"], "early_intact": last["early_intact"],
        "parent_sha256": PARENT_SHA,
        "locks": {"T3_TEST": "SEALED_UNOPENED", "T2_EVAL_TEST": "LOCKED_UNOPENED",
                  "FINAL": "LOCKED", "sacred": "LOCKED"},
    }
    write_json(out / "FINAL_STATUS.json", status)
    write_json(out / "STATUS.json", {"status": "COMPLETE", **status})
    ledger_path = bundle / "RUN_LEDGER.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
        if not isinstance(ledger.get("runs"), dict):
            ledger["runs"] = {}
        ledger["runs"][str(args.seed)] = status
        ledger["status"] = "TRAINING_IN_PROGRESS_OR_COMPLETE"
        ledger["test_opened"] = False
        ledger["failed_returncode"] = None
        write_json(ledger_path, ledger)
    except Exception as exc:
        print(json.dumps({"ledger_patch_error": str(exc), "seed": args.seed}), flush=True)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
