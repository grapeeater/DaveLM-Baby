"""Phase 2A T31: T24 recipe plus an explicit ordered native source-source-memory prefix.

The existing query-to-fact-clause pointer is made available at every answer
prediction position through a zero-initialized 640-parameter channel gate.
All inherited losses/data/scopes remain fixed; no fork or suffix-margin loss.
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
import time
from array import array
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

BUNDLE_DEFAULT = Path(r"C:\DaveLM-CADAVER\phase2a_t32_causal_ordered_source_memory_v1")
PHASE_A_SCOPE = {"trainable": 40850304, "frozen": 20670721}
PHASE_B_SCOPE = {"trainable": 21163904, "frozen": 40357121}
REVERT_AFTER = 750
TOTAL_UPDATES = 1000
T3_DATA = Path(r"C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1\data")
ARCH = Path(r"C:\DaveLM-CADAVER\baby_vnext_60m_design_v1")
sys.path.insert(0, str(ARCH))

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
LAMBDA_PTR = 1.0
LAMBDA_ANS = 1.0
LAMBDA_UNL = 1.0
LAMBDA_FORK = 0.0
FORK_MARGIN = 1.0
AUTHORIZED_SEEDS = (890001, 890002, 890003)
CTX = 256
EOS = 3
PERIOD = 18  # tokenizer token '.'


def identity_fork_table(identity_names: list, sequences: dict) -> dict:
    """First unique-commit token among identities that share a nonempty prefix.

    Fire only when another identity shares a nonempty token prefix. Fork index k
    is the first position where gold[:k] is shared by >=2 identities and
    gold[:k+1] is unique to gold. Competitor tokens are the differing next
    tokens of those prefix-compatible identities. Do not invent a fork for
    unique-first-token names.
    """
    seq = {n: list(sequences[n]) for n in identity_names}
    out = {}
    for gold, gids in seq.items():
        nonempty = False
        for other, oids in seq.items():
            if other == gold:
                continue
            shared = 0
            for a, b in zip(gids, oids):
                if a != b:
                    break
                shared += 1
            if shared > 0:
                nonempty = True
                break
        if not nonempty:
            out[gold] = {
                "has_fork": False, "fork_index": None, "gold_token": None,
                "competitor_tokens": [], "competitor_names": [],
            }
            continue
        found = None
        for k in range(len(gids)):
            pref = tuple(gids[:k])
            remain = [o for o, oids in seq.items() if tuple(oids[:k]) == pref]
            if len(remain) < 2:
                continue
            remain1 = [o for o, oids in seq.items() if tuple(oids[:k + 1]) == tuple(gids[:k + 1])]
            if len(remain1) == 1 and remain1[0] == gold:
                comps = []
                for other in remain:
                    if other == gold:
                        continue
                    oids = seq[other]
                    if k < len(oids) and oids[k] != gids[k]:
                        comps.append((other, int(oids[k])))
                found = {
                    "has_fork": True,
                    "fork_index": k,
                    "gold_token": int(gids[k]),
                    "competitor_tokens": sorted({t for _, t in comps}),
                    "competitor_names": sorted({n for n, _ in comps}),
                }
                break
        out[gold] = found or {
            "has_fork": False, "fork_index": None, "gold_token": None,
            "competitor_tokens": [], "competitor_names": [],
        }
    return out


def failmode(generated: list, target: list) -> str:
    if generated == target:
        return "exact"
    if not generated or generated[0] == EOS:
        return "immediate_eos"
    if generated[0] == target[0]:
        return "first_token_correct_then_diverge"
    if EOS in generated:
        return "other_then_eos"
    return "unterminated_other"


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
    last = None
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last = e
            time.sleep(0.25)
    raise last


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
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def configure_runtime(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def load_model():
    from baby_vnext.config import BabyVNextConfig
    from T32_MEMORY import T32MemoryModel
    if sha256(PARENT) != PARENT_SHA:
        raise RuntimeError("parent hash mismatch")
    if sha256(TOK_PATH) != TOK_SHA:
        raise RuntimeError("tokenizer hash mismatch")
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    if payload.get("schema") != "baby_vnext_phase1_model_v1" or payload.get("completed_update") != 6000:
        raise RuntimeError("parent schema/update mismatch")
    model = T32MemoryModel(BabyVNextConfig.load(ARCH / "BABY_VNEXT_CONFIG.json"))
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
    if unexpected or missing != ["memory_gate"]:
        raise RuntimeError(f"T32 parent schema mismatch missing={missing} unexpected={unexpected}")
    with torch.no_grad():
        model.memory_gate.zero_()
    return model, payload["model_state_dict"]


def freeze_early(name: str, phase: str) -> bool:
    if any(name.startswith(f"base_model.blocks.{i}.") for i in range(4)):
        return True
    if phase == "B" and any(name.startswith(f"base_model.blocks.{i}.") for i in range(8, 12)):
        return True
    return False


def set_scope(model, phase: str):
    for n, p in model.named_parameters():
        train = ((n.startswith("base_model.") and not freeze_early(n, phase))
                 or n == "memory_gate")
        p.requires_grad_(train)
        if not train:
            p.grad = None
    return {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "phase": phase,
    }


def revert_blocks_8_11(model, parent_sd) -> None:
    with torch.no_grad():
        for n, p in model.named_parameters():
            if any(n.startswith(f"base_model.blocks.{i}.") for i in range(8, 12)):
                p.copy_(parent_sd[n].to(device=p.device, dtype=p.dtype))


def make_optimizer(model):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3.75e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05,
        foreach=False, fused=False,
    )


def fact_clause_spans_bos(row):
    """Name mention through next period (assignment-sensitive fact clause)."""
    prompt = [2] + [int(x) for x in row["prompt_token_ids"]]
    spans = []
    for mention in row["mention_span_bos"]:
        start = int(mention[0])
        end = int(mention[-1])
        for i in range(end, len(prompt)):
            if prompt[i] == PERIOD:
                end = i
                break
        else:
            raise RuntimeError(f"no period after mention in {row.get('id')}")
        if end >= int(row["decision_position_bos"]) or end < start:
            raise RuntimeError(f"invalid fact span in {row.get('id')}")
        spans.append(list(range(start, end + 1)))
    return spans


def _memory_specs(row, seq_len, device):
    """Build the frozen TRAIN/protocol source-span metadata for one row."""
    fact = fact_clause_spans_bos(row)
    mentions = [[int(x) for x in s] for s in row["mention_span_bos"]]
    q = torch.tensor([int(row["decision_position_bos"])], device=device)
    mask = torch.zeros((1, seq_len), dtype=torch.bool, device=device)
    answer_start = len([2] + [int(x) for x in row["prompt_token_ids"]]) - 1
    if answer_start < seq_len:
        mask[:, answer_start:] = True
    return [fact], [mentions], q, mask


def _forward_row(model, ids, row, device, answer_start):
    x = torch.tensor([ids], device=device)
    fact, mentions, q, mask = _memory_specs(row, len(ids), device)
    mask.zero_()
    if answer_start < len(ids):
        mask[:, answer_start:] = True
    hidden, stats = model.forward_hidden_with_memory(x, fact, mentions, q, mask)
    return model.base_model.language_head(hidden), stats


def pointer_and_margin(model, row, device, train_mode: bool):
    prompt = [2] + [int(x) for x in row["prompt_token_ids"]]
    p_len = len(prompt)
    spans = fact_clause_spans_bos(row)
    x = torch.tensor([prompt], device=device)
    fact, mentions, q, mask = _memory_specs(row, len(prompt), device)
    mask.zero_()
    hidden, _ = model.forward_hidden_with_memory(x, fact, mentions, q, mask)
    sims, weights, retrieved = model.pointer_route(hidden, spans, p_len - 1)
    ci = int(row["correct_index"])
    ptr = F.cross_entropy(sims.view(1, -1), torch.tensor([ci], device=device))
    scores = []
    for cand in row["candidate_token_ids"]:
        c = [int(t) for t in cand]
        xx = torch.tensor([prompt + c], device=device)
        logits, _ = _forward_row(model, prompt + c, row, device, p_len - 1)
        lp = F.log_softmax(logits[0, p_len - 1:p_len - 1 + len(c)], dim=-1)
        idx = torch.tensor(c, device=device)
        scores.append(lp[torch.arange(len(c), device=device), idx].mean())
    other = torch.stack([s for j, s in enumerate(scores) if j != ci]).max()
    mar = torch.clamp(torch.tensor(M, device=device) - (scores[ci] - other), min=0.0)
    return ptr, mar, scores, sims


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
def greedy_answer(model, row, device):
    prompt = [2] + [int(x) for x in row["prompt_token_ids"]]
    p_len = len(prompt)
    spans = fact_clause_spans_bos(row)
    target = [int(t) for t in row["candidate_token_ids"][int(row["correct_index"])]] + [EOS]
    gen = list(prompt)
    out_ids = []
    for _ in range(len(target) + 2):
        inp = torch.tensor([gen], device=device)
        logits, _ = _forward_row(model, gen, row, device, p_len - 1)
        t = int(logits[0, -1].argmax())
        gen.append(t)
        out_ids.append(t)
        if t == EOS:
            break
    return out_ids, target


def eval_panel(model, rows, device, eval_forks: dict | None = None):
    was = model.training
    model.eval()
    n = len(rows)
    ptr_ok = 0
    fc_ok = 0
    exact = 0
    margins = []
    per_row = []
    try:
        for r in rows:
            ptr, mar, scores, sims = pointer_and_margin(model, r, device, False)
            ci = int(r["correct_index"])
            pred_ptr = int(torch.argmax(sims).item())
            pred_fc = int(max(range(len(scores)), key=lambda j: float(scores[j].detach())))
            gen, target = greedy_answer(model, r, device)
            ex = gen == target
            p_ok = pred_ptr == ci
            f_ok = pred_fc == ci
            ptr_ok += int(p_ok)
            fc_ok += int(f_ok)
            exact += int(ex)
            others = [float(scores[j]) for j in range(len(scores)) if j != ci]
            margins.append(float(scores[ci]) - max(others))
            name = r["correct_name"]
            fork = (eval_forks or {}).get(name) or {"has_fork": False, "fork_index": None}
            k = fork.get("fork_index")
            fork_ok = None
            if fork.get("has_fork") and k is not None:
                fork_ok = int(k) < len(gen) and gen[int(k)] == target[int(k)]
            mode = failmode(gen, target)
            per_row.append({
                "id": r["id"],
                "family_id": r["family_id"],
                "pointer_ok": p_ok,
                "native_ok": f_ok,
                "exact_ok": ex,
                "correct_index": ci,
                "pred_pointer": pred_ptr,
                "pred_native": pred_fc,
                "assignment": r["assignment"],
                "query_index": r["query_index"],
                "fact_order": r["fact_order"],
                "first_mentioned": r["first_mentioned"],
                "correct_name": name,
                "template": r["template"],
                "answer_len": len(r["candidate_token_ids"][ci]),
                "has_shared_prefix": bool(fork.get("has_fork")),
                "fork_index": k,
                "fork_token_ok": fork_ok,
                "first_token_ok": bool(gen) and gen[0] == target[0],
                "failmode": mode,
                "generated_len": len(gen),
            })
    finally:
        model.train(was)
    shared = [p for p in per_row if p["has_shared_prefix"]]
    unique = [p for p in per_row if not p["has_shared_prefix"]]
    fork_rows = [p for p in per_row if p["fork_token_ok"] is not None]
    diverge = sum(p["failmode"] == "first_token_correct_then_diverge" for p in per_row)
    shared_div = sum(p["failmode"] == "first_token_correct_then_diverge" for p in shared)
    fork_correct = [p for p in fork_rows if p["fork_token_ok"]]
    post_fork = sum(p["exact_ok"] for p in fork_correct)
    return {
        "n": n,
        "pointer_retrieval": ptr_ok,
        "native_forced_choice": fc_ok,
        "exact_answer_eos": exact,
        "mean_margin": (sum(margins) / n) if n else 0.0,
        "per_row": per_row,
        "fork_metrics": {
            "first_disambiguating_token_n": len(fork_rows),
            "first_disambiguating_token_correct": sum(bool(p["fork_token_ok"]) for p in fork_rows),
            "first_disambiguating_token_accuracy": (
                sum(bool(p["fork_token_ok"]) for p in fork_rows) / len(fork_rows) if fork_rows else None
            ),
            "shared_prefix_n": len(shared),
            "shared_prefix_exact": sum(p["exact_ok"] for p in shared),
            "shared_prefix_exact_rate": (sum(p["exact_ok"] for p in shared) / len(shared) if shared else None),
            "unique_prefix_n": len(unique),
            "unique_prefix_exact": sum(p["exact_ok"] for p in unique),
            "unique_prefix_exact_rate": (sum(p["exact_ok"] for p in unique) / len(unique) if unique else None),
            "first_token_correct_then_diverge": diverge,
            "first_token_correct_then_diverge_shared": shared_div,
            "first_token_correct_then_diverge_rate": diverge / n if n else None,
            "post_fork_free_completion_n": len(fork_correct),
            "post_fork_free_completion": post_fork,
            "post_fork_free_completion_rate": (post_fork / len(fork_correct) if fork_correct else None),
        },
    }


def assignment_reversals(per_row: list) -> dict:
    groups = defaultdict(dict)
    for r in per_row:
        key = (r["family_id"], r["query_index"], r["fact_order"])
        groups[key][r["assignment"]] = r
    pairs = 0
    both_ptr = 0
    both_nat = 0
    for d in groups.values():
        if 0 in d and 1 in d:
            pairs += 1
            both_ptr += int(d[0]["pointer_ok"] and d[1]["pointer_ok"])
            both_nat += int(d[0]["native_ok"] and d[1]["native_ok"])
    return {"pairs": pairs, "pointer": both_ptr, "native": both_nat}


def complete_families(per_row: list, field: str) -> dict:
    fam = defaultdict(list)
    for r in per_row:
        fam[r["family_id"]].append(r)
    n_fam = len(fam)
    ok = sum(1 for items in fam.values() if items and all(x[field] for x in items) and len(items) == 8)
    return {"families": n_fam, "complete": ok}


def shortcut_audits(rows: list, per_row: list) -> dict:
    by_id = {r["id"]: r for r in rows}
    n = len(per_row)
    always0 = sum(1 for p in per_row if p["pred_pointer"] == 0)
    recency = 0
    first = 0
    longer = 0
    for p in per_row:
        src = by_id[p["id"]]
        last_i = src["candidates"].index(src["last_mentioned"])
        first_i = src["candidates"].index(src["first_mentioned"])
        recency += int(p["pred_pointer"] == last_i)
        first += int(p["pred_pointer"] == first_i)
        lens = [len(c) for c in src["candidate_token_ids"]]
        longer_i = 0 if lens[0] >= lens[1] else 1
        longer += int(p["pred_pointer"] == longer_i)
    return {
        "always_index0": always0 / n,
        "last_mentioned": recency / n,
        "first_mentioned": first / n,
        "longer_name": longer / n,
    }


def subset_eval(panel: dict, id_set: set) -> dict:
    sub = [p for p in panel["per_row"] if p["id"] in id_set]
    n = len(sub)
    return {
        "n": n,
        "pointer_retrieval": sum(p["pointer_ok"] for p in sub),
        "native_forced_choice": sum(p["native_ok"] for p in sub),
        "exact_answer_eos": sum(p["exact_ok"] for p in sub),
        "reversals": assignment_reversals(sub),
        "complete_families": complete_families(sub, "pointer_ok"),
    }


def futility_decision(rec: dict, update: int) -> tuple[bool, str]:
    """Conservative T31-specific futility gate frozen before training.
    At U750, only a clearly dead native trajectory with an active copy route
    can stop early; this condition is below every known historical U1000
    exact trajectory and therefore has no known historical false kill.
    """
    if update < 750:
        return False, "CONTINUE"
    fm = rec["dev"].get("fork_metrics", {})
    exact = int(rec["dev"].get("exact_answer_eos", 0))
    shared = int(fm.get("shared_prefix_exact", 0))
    diverge = int(fm.get("first_token_correct_then_diverge", 0))
    active = float(rec.get("memory_gate_l2", 0.0)) > 0.01
    stop = active and exact <= 20 and shared == 0 and diverge >= 55
    return stop, ("STOP" if stop else "CONTINUE")

def classify(rec: dict, u0_ce: float) -> str:
    if rec.get("hard_stop"):
        return "T32_HARD_STOP"
    lang_cap = min(1.30, u0_ce + 0.25)
    lang_ok = rec["dev_ce"] <= lang_cap and rec["gap"] <= 0.5 and math.isfinite(rec["dev_ce"])
    if not rec["binding_intact"] or not rec.get("early_intact", True) or not math.isfinite(rec["dev_ce"]):
        return "T32_HARD_STOP"
    ptr = rec["dev"]["pointer_retrieval"]
    rev_p = rec["dev_reversals"]["pointer"]
    fam = rec["dev_complete_ptr"]["complete"]
    nd = rec["name_disjoint"]["pointer_retrieval"]
    td = rec["template_disjoint"]["pointer_retrieval"]
    rep = ptr >= 96 and rev_p >= 48 and fam >= 8 and nd >= 48 and td >= 48
    sc = rec["shortcuts"]
    if rep and max(sc.values()) >= (ptr / rec["dev"]["n"]) - 0.02:
        return "T32_SHORTCUT_FAILURE"
    nat = (rec["dev"]["native_forced_choice"] >= 96
           and rec["dev_reversals"]["native"] >= 48
           and rec["dev"]["exact_answer_eos"] >= 80)
    acq = rec["acq16_pass"]
    if not lang_ok:
        return "T32_LANGUAGE_REGRESSION"
    if not rep:
        return "T32_FAIL_NO_REPRESENTATION"
    if not nat or not acq:
        return "T32_REPRESENTATION_SUCCESS_OUTPUT_FAIL"
    return "T32_FULL_SUCCESS"


def acq16_pass(panel: dict) -> bool:
    if panel["n"] != 16:
        return False
    if panel["native_forced_choice"] < 16 or panel["exact_answer_eos"] < 16:
        return False
    if panel["reversals"]["native"] < 8:
        return False
    fam = complete_families(panel["per_row"], "native_ok")
    # ACQ16 has 4 families x 4 items (fact_order=0 only), so complete-8 does not apply.
    fams = {p["family_id"] for p in panel["per_row"]}
    per = defaultdict(list)
    for p in panel["per_row"]:
        per[p["family_id"]].append(p)
    all_fam = all(len(v) == 4 and all(x["native_ok"] for x in v) for v in per.values())
    return len(fams) == 4 and all_fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("preflight", "train", "smoke", "score-test"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--bundle", type=Path, default=BUNDLE_DEFAULT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true",
                    help="continue from rolling_restart.pt in an existing run dir")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    if args.seed not in AUTHORIZED_SEEDS:
        raise RuntimeError("unauthorized seed")
    bundle = args.bundle.resolve()
    device = torch.device("cuda")
    configure_runtime(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    tok = Tokenizer.from_file(str(TOK_PATH))
    train_rows = load_jsonl(T3_DATA / "qa_train.jsonl")
    dev_rows = load_jsonl(T3_DATA / "qa_dev.jsonl")
    acq_rows = json.loads((T3_DATA / "acq16.json").read_text(encoding="utf-8"))
    panels = json.loads((T3_DATA / "panels.json").read_text(encoding="utf-8"))
    kinds = json.loads((bundle / "data" / "update_kinds.json").read_text(encoding="utf-8"))["kinds"]
    schedules = json.loads((bundle / "data" / "qa_schedules.json").read_text(encoding="utf-8"))
    qa_sched = schedules[str(args.seed)]
    rehear = json.loads((bundle / "data" / "rehearsal_index.json").read_text(encoding="utf-8"))["starts_mod_320"]
    if len(kinds) != 1000 or kinds.count("lang") != 100 or kinds.count("qa") != 900:
        raise RuntimeError(f"kinds freeze mismatch len={len(kinds)}")
    if len(qa_sched) != 900 or any(len(b) != 32 for b in qa_sched):
        raise RuntimeError(f"qa schedule freeze mismatch len={len(qa_sched)}")
    if len(rehear) != 100 or any(len(x) != 64 for x in rehear):
        raise RuntimeError("rehearsal freeze mismatch")
    inv_payload = json.loads((bundle / "data" / "name_inventory.json").read_text(encoding="utf-8"))
    if inv_payload.get("never_loaded") != ["qa_test.jsonl"] or inv_payload.get("n_names") != 24:
        raise RuntimeError("name inventory freeze mismatch")
    name_inventory = {k: list(v) for k, v in inv_payload["names"].items()}
    audits = json.loads((T3_DATA / "audits.json").read_text(encoding="utf-8"))
    train_names = list(audits["train_names"])
    dev_names = list(audits["dev_names"])
    if set(train_names) & set(dev_names):
        raise RuntimeError("train/dev name overlap")
    train_forks = identity_fork_table(train_names, name_inventory)
    eval_forks_dev = identity_fork_table(dev_names, name_inventory)
    nd_ids = set(panels["name_disjoint_ids"])
    td_ids = set(panels["template_disjoint_ids"])
    seal = json.loads((T3_DATA / "TEST_SEAL.json").read_text(encoding="utf-8"))
    if seal["status"] != "SEALED_UNOPENED" and args.mode != "score-test":
        raise RuntimeError("TEST seal status unexpected")

    stream_train = read_u16(LANG_TRAIN)
    stream_dev = read_u16(LANG_DEV)
    starts_all = read_u32(LANG_STARTS, 1600)
    dev_starts = starts_all[:1280]
    train_fit = starts_all[1280:]
    gen_prompts = json.loads(GEN_PROMPTS.read_text(encoding="utf-8"))["prompts"]

    model, parent_sd = load_model()
    phase = "A"
    scope = set_scope(model, phase)
    if scope["trainable"] != PHASE_A_SCOPE["trainable"] or scope["frozen"] != PHASE_A_SCOPE["frozen"]:
        raise RuntimeError(f"phase A scope mismatch {scope}")
    model.to(device)
    binding_names = [n for n in model.state_dict() if not n.startswith("base_model.") and n != "memory_gate"]
    parent_binding = {n: parent_sd[n].clone().to(device) for n in binding_names}
    frozen_a_names = [n for n in model.state_dict() if freeze_early(n, "A")]
    frozen_b_extra = [n for n in model.state_dict() if freeze_early(n, "B") and n not in set(frozen_a_names)]
    parent_early = {n: parent_sd[n].clone().to(device) for n in frozen_a_names + frozen_b_extra}

    def binding_intact():
        sd = model.state_dict()
        return all(bool((sd[n] == parent_binding[n]).all()) for n in binding_names)

    def early_intact():
        sd = model.state_dict()
        names = frozen_a_names if phase == "A" else (frozen_a_names + frozen_b_extra)
        return all(bool((sd[n] == parent_early[n]).all()) for n in names)

    optimizer = None

    def apply_phase_b(reason: str):
        nonlocal phase, scope, optimizer
        revert_blocks_8_11(model, parent_sd)
        phase = "B"
        scope = set_scope(model, "B")
        if scope["trainable"] != PHASE_B_SCOPE["trainable"] or scope["frozen"] != PHASE_B_SCOPE["frozen"]:
            raise RuntimeError(f"phase B scope mismatch {scope}")
        optimizer = make_optimizer(model)
        print(json.dumps({"phase_b": True, "reason": reason, "scope": scope}), flush=True)

    def run_full(update: int) -> dict:
        model.eval()
        dev = eval_panel(model, dev_rows, device, eval_forks_dev)
        acq = eval_panel(model, acq_rows, device, train_forks)
        acq["reversals"] = assignment_reversals(acq["per_row"])
        dce = lang_ce(model, stream_dev, dev_starts, device, 16)
        tce = lang_ce(model, stream_train, train_fit, device, 16)
        rec = {
            "update": update,
            "dev_ce": dce,
            "train_ce": tce,
            "gap": dce - tce,
            "binding_intact": binding_intact(),
            "early_intact": early_intact(),
            "dev": {k: v for k, v in dev.items() if k != "per_row"},
            "dev_reversals": assignment_reversals(dev["per_row"]),
            "dev_complete_ptr": complete_families(dev["per_row"], "pointer_ok"),
            "dev_complete_nat": complete_families(dev["per_row"], "native_ok"),
            "name_disjoint": subset_eval(dev, nd_ids),
            "template_disjoint": subset_eval(dev, td_ids),
            "acq16": {k: v for k, v in acq.items() if k != "per_row"},
            "acq16_reversals": acq["reversals"],
            "acq16_pass": acq16_pass({**acq, "reversals": acq["reversals"]}),
            "shortcuts": shortcut_audits(dev_rows, dev["per_row"]),
            "generation": greedy_diag(model, tok, gen_prompts, device),
            "scope": scope,
            "memory_gate_l2": float(torch.linalg.vector_norm(model.memory_gate.detach()).cpu()),
            "memory_gate_mean_abs": float(model.memory_gate.detach().abs().mean().cpu()),
            "test_loaded": False,
        }
        rec["dev"]["per_row_count"] = len(dev["per_row"])
        rec["item_results"] = dev["per_row"]
        rec["acq_item_results"] = acq["per_row"]
        return rec

    out = args.out.resolve()
    if args.mode == "preflight":
        out.mkdir(parents=True, exist_ok=True)
        rec = run_full(0)
        rec["classification"] = "PREFLIGHT"
        write_json(out / "PREFLIGHT.json", {k: v for k, v in rec.items()
                                            if k not in ("item_results", "acq_item_results")})
        write_json(out / "evaluation_0000.json", {k: v for k, v in rec.items()
                                                  if k not in ("item_results", "acq_item_results")})
        print(json.dumps({
            "status": "PREFLIGHT_OK",
            "dev_ce": rec["dev_ce"],
            "pointer": rec["dev"]["pointer_retrieval"],
            "native": rec["dev"]["native_forced_choice"],
            "binding_intact": rec["binding_intact"],
        }, indent=2))
        return

    if args.mode == "score-test":
        raise RuntimeError("score-test is only valid after a terminal DEV commit that passes representation and native gates; refusing implicit open")

    slim = lambda r: {k: v for k, v in r.items() if k not in ("item_results", "acq_item_results")}
    resume_from = 0
    if args.mode == "smoke":
        out.mkdir(parents=True, exist_ok=True)
        optimizer = make_optimizer(model)
        u0_ce = 1.2040123894810677
    elif args.resume:
        restart_path = out / "rolling_restart.pt"
        u0_path = out / "evaluation_0000.json"
        if not restart_path.is_file() or not u0_path.is_file():
            raise RuntimeError("resume requires rolling_restart.pt and evaluation_0000.json")
        u0_ce = float(json.loads(u0_path.read_text(encoding="utf-8"))["dev_ce"])
        restart = torch.load(restart_path, map_location="cpu", weights_only=False)
        resume_from = int(restart["completed"])
        model.load_state_dict(restart["model_state_dict"], strict=True)
        phase = restart.get("phase", "A" if resume_from < REVERT_AFTER else "B")
        if resume_from >= REVERT_AFTER and phase != "B":
            apply_phase_b("resume_unarmed_phase_b")
        else:
            scope = set_scope(model, phase)
            optimizer = make_optimizer(model)
            if "optimizer" in restart:
                optimizer.load_state_dict(restart["optimizer"])
                for state in optimizer.state.values():
                    for k, v in list(state.items()):
                        if torch.is_tensor(v):
                            state[k] = v.to(device)
        write_json(out / "STATUS.json", {
            "status": "TRAINING_RUNNING", "seed": args.seed, "update": resume_from,
            "resumed": True, "phase": phase, "parent_sha256": PARENT_SHA,
        })
    else:
        if out.exists() and any(out.iterdir()):
            raise RuntimeError("fresh output dir already exists")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "STATUS.json", {
            "status": "TRAINING_RUNNING",
            "seed": args.seed,
            "update": 0,
            "parent_sha256": PARENT_SHA,
        })
        u0 = run_full(0)
        u0_ce = u0["dev_ce"]
        if abs(u0_ce - 1.2040123894810677) > 1e-3:
            raise RuntimeError(f"language U0 reproduction failed: {u0_ce}")
        write_json(out / "evaluation_0000.json", slim(u0))
        write_json(out / "item_results_0000.json", {
            "dev": u0["item_results"], "acq16": u0["acq_item_results"]
        })
        optimizer = make_optimizer(model)
    metrics_path = out / "training_metrics.jsonl"
    qa_i = sum(1 for k in kinds[:resume_from] if k != "lang")
    lang_i = sum(1 for k in kinds[:resume_from] if k == "lang")
    final_class = None
    stop_code = None

    def answer_span_ce_and_unl(row):
        prompt = [2] + [int(x) for x in row["prompt_token_ids"]]
        p_len = len(prompt)
        ci = int(row["correct_index"])
        target = [int(t) for t in row["candidate_token_ids"][ci]] + [EOS]
        gold_name = row["correct_name"]
        rivals = [toks for name, toks in name_inventory.items() if name != gold_name]
        ids = prompt + target
        if len(ids) > CTX + 1:
            raise RuntimeError(f"T32 sequence exceeds frozen context for {row.get('id')}")
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor([ids[1:]], device=device)
        logits, _ = _forward_row(model, ids[:-1], row, device, p_len - 1)
        n = len(target)
        ans = F.cross_entropy(logits[0, -n:, :], y[0, -n:])
        probs = torch.softmax(logits[0, -n:, :], dim=-1)
        unl_terms = []
        for pos in range(n):
            gold_t = target[pos]
            rival_toks = {r[pos] for r in rivals if pos < len(r) and r[pos] != gold_t}
            for t in rival_toks:
                p = probs[pos, t].clamp(max=1.0 - 1e-6)
                unl_terms.append(-torch.log1p(-p))
        unl = torch.stack(unl_terms).mean() if unl_terms else logits.new_zeros(())
        return ans, unl


    def qa_update(indices):
        micro = 8
        total = 0.0
        n_micro = 0
        for b in range(0, len(indices), micro):
            group = indices[b:b + micro]
            loss = 0.0
            for j in group:
                ptr, mar, _, _ = pointer_and_margin(model, train_rows[j], device, True)
                ans, unl = answer_span_ce_and_unl(train_rows[j])
                loss = loss + (mar + LAMBDA_PTR * ptr + LAMBDA_ANS * ans + LAMBDA_UNL * unl) / len(group)
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

    if args.mode == "smoke":
        model.train()
        n_fork = sum(1 for j in qa_sched[0] if train_forks.get(train_rows[j]["correct_name"], {}).get("has_fork"))
        t0 = time.perf_counter()
        losses = []
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = qa_update(qa_sched[0])
            gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0)
            if not bool(torch.isfinite(gn)) or not math.isfinite(float(loss)):
                raise RuntimeError("smoke nonfinite")
            optimizer.step()
            losses.append(float(loss))
        dt = time.perf_counter() - t0
        payload = {
            "status": "SMOKE_OK",
            "seconds_for_two_qa_updates": dt,
            "seconds_per_qa_update": dt / 2.0,
            "losses": losses,
            "fork_rows_in_first_batch": n_fork,
            "batch_size": 32,
            "device": str(device),
            "cuda": bool(torch.cuda.is_available()),
            "t3_test_loaded": False,
            "extra_forward_policy": "persistent source-memory prefix reuses one base hidden; no generated-prefix rollout",
            "memory_parameters": int(model.memory_gate.numel()),
        }
        write_json(out / "SMOKE.json", payload)
        print(json.dumps(payload, indent=2))
        return

    for update in range(resume_from + 1, TOTAL_UPDATES + 1):
        kind = kinds[update - 1]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if kind == "lang":
            loss = lang_update(rehear[lang_i])
            lang_i += 1
        else:
            loss = qa_update(qa_sched[qa_i])
            qa_i += 1
        bad = [n for n, p in model.named_parameters()
               if p.grad is not None and (freeze_early(n, phase) or (not n.startswith("base_model.") and n != "memory_gate"))]
        if bad:
            raise RuntimeError(f"binding gradient at {update}: {bad}")
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 2.0)
        if not bool(torch.isfinite(gn)) or not math.isfinite(float(loss)):
            raise RuntimeError(f"nonfinite at {update}")
        optimizer.step()
        with open(metrics_path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"update": update, "kind": kind, "loss": float(loss),
                                "grad_norm": float(gn)}) + "\n")
        if update % 50 == 0:
            ckdir = out / "checkpoints"
            ckdir.mkdir(exist_ok=True)
            atomic_torch(out / "rolling_restart.pt", {
                "completed": update,
                "phase": phase,
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            })
            write_json(out / "STATUS.json", {
                "status": "TRAINING_RUNNING", "seed": args.seed, "update": update, "phase": phase
            })
        if update not in (100, 300, 500, 750, 1000):
            continue
        rec = run_full(update)
        rec["hard_stop"] = ((not rec["binding_intact"]) or (not rec["early_intact"])
                            or (not math.isfinite(rec["dev_ce"])))
        rec["mechanical_health"] = "PASS" if not rec["hard_stop"] else "FAIL"
        rec["scientific_efficacy"] = ("PROMISING" if rec["dev"]["exact_answer_eos"] > 20
                                      or rec["dev"]["fork_metrics"]["shared_prefix_exact"] > 0
                                      else "WEAK")
        fut_stop, fut_status = futility_decision(rec, update)
        rec["futility_status"] = fut_status
        rec["classification"] = classify(rec, u0_ce)
        write_json(out / f"evaluation_{update:04d}.json", slim(rec))
        write_json(out / f"item_results_{update:04d}.json", {
            "dev": rec["item_results"], "acq16": rec["acq_item_results"]
        })
        ckdir = out / "checkpoints"
        ckdir.mkdir(exist_ok=True)
        atomic_torch(ckdir / f"checkpoint_{update:04d}.pt", {
            "schema": "baby_vnext_phase2a_t32_causal_ordered_source_memory_lr3p75e5_1k_model_v1",
            "update": update,
            "seed": args.seed,
            "phase": phase,
            "model_state_dict": model.state_dict(),
        })
        if update == REVERT_AFTER and phase == "A":
            apply_phase_b("after_u750_eval")
            atomic_torch(out / "rolling_restart.pt", {
                "completed": update,
                "phase": phase,
                "model_state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            })
            write_json(out / "STATUS.json", {
                "status": "TRAINING_RUNNING", "seed": args.seed, "update": update, "phase": phase
            })
        if rec["hard_stop"] or rec["dev_ce"] > 1.50:
            final_class, stop_code = "T32_HARD_STOP", "integrity_or_catastrophe"
            break
        if fut_stop:
            final_class, stop_code = "T32_FUTILITY_STOP", "native_trajectory_futility_u750"
            break
        if rec["dev_ce"] > 1.30:
            final_class, stop_code = "T32_LANGUAGE_REGRESSION", "language_ce_gt_1_30"
            break
        if update == TOTAL_UPDATES:
            final_class = rec["classification"]
            stop_code = None

    last_eval = sorted(out.glob("evaluation_*.json"))[-1]
    last = json.loads(last_eval.read_text(encoding="utf-8"))
    status = {
        "seed": args.seed,
        "update": last["update"],
        "classification": final_class or last.get("classification"),
        "stop_code": stop_code,
        "test_opened": False,
        "dev_pointer": last["dev"]["pointer_retrieval"],
        "dev_native": last["dev"]["native_forced_choice"],
        "dev_exact": last["dev"]["exact_answer_eos"],
        "dev_ce": last["dev_ce"],
        "dev_fork_metrics": last.get("dev", {}).get("fork_metrics"),
        "binding_intact": last["binding_intact"],
        "early_intact": last["early_intact"],
        "parent_sha256": PARENT_SHA,
        "locks": {
            "T3_TEST": "SEALED_UNOPENED",
            "T2_EVAL_TEST": "LOCKED_UNOPENED",
            "FINAL": "LOCKED",
            "sacred": "LOCKED",
        },
        "t3_test_loaded": False,
    }
    write_json(out / "FINAL_STATUS.json", status)
    write_json(out / "STATUS.json", {"status": "COMPLETE", **status})
    ledger_path = bundle / "RUN_LEDGER.json"
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not isinstance(ledger.get("runs"), dict):
            ledger["runs"] = {}
        ledger["runs"][str(args.seed)] = status
        ledger["status"] = "TRAINING_IN_PROGRESS_OR_COMPLETE"
        ledger["test_opened"] = False
        ledger["failed_returncode"] = None
        write_json(ledger_path, ledger)
    except Exception as exc:
        print(json.dumps({"ledger_patch_error": str(exc), "seed": args.seed}), flush=True)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()














