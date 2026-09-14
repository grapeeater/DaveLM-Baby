"""DEV-only post-T28 single-token oracle prefix rescue. Forward-pass only."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(r"C:\DaveLM-CADAVER")
ARCH = ROOT / "baby_vnext_60m_design_v1"
sys.path.insert(0, str(ARCH))
OUT = Path(__file__).resolve().parent
DEV_PATH = ROOT / "phase2a_t3_rebuilt_study_v1" / "data" / "qa_dev.jsonl"
INV_PATH = ROOT / "phase2a_t28_fast_v2" / "data" / "name_inventory.json"
TOK_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
STREAM = ROOT / "baby_vnext_phase1g_language_v1" / "data" / "LANGUAGE_TRAIN_STREAM.u16"
TOK_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
PARENT_SHA = "c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1"
DEV_SHA = "309f55dec7057b39b2972034e1f62753fc4edae989679c5f0d4fed85d2a25f99"
STREAM_SHA = "5f36b283cebe00d88445383166257fbb9831b759657a28c745f3d888f9626a41"
CKPT_SHA = {
    850001: "d6f5be589c3905f76265431fa377c74245e972d9b28e8cee2c085d8f219e2684",
    850002: "f01066dd90aa3ff65b6affc7caad942486d6bfcc740733b24a440615ecf4f172",
    850003: "95e34fd52bc1b836a726816a2d8ae259e392620733146c2941aa53e00deaf597",
}
EXPECTED_POP = {850001: 39, 850002: 40, 850003: 45}
SEEDS = (850001, 850002, 850003)
EOS, BOS, CTX, VOCAB = 3, 2, 256, 1024
MAX_NEW = 16
BLOCK_LAYERS = (8, 9, 10, 11, 12)  # after blocks 7-11
FORBIDDEN = ("qa_test.jsonl",)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]


def language_counts() -> Counter:
    raw = array("H")
    with STREAM.open("rb") as f:
        raw.fromfile(f, STREAM.stat().st_size // 2)
    return Counter(raw)


def load_model(seed: int, device: torch.device):
    from baby_vnext.config import BabyVNextConfig
    from baby_vnext.binding import BabyVNextWithBinding
    ckpt = ROOT / f"phase2a_t28_fast_v2_run_seed{seed}" / "rolling_restart.pt"
    got = sha(ckpt)
    if got != CKPT_SHA[seed]:
        raise RuntimeError(f"checkpoint hash mismatch {seed}: {got}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = BabyVNextWithBinding(BabyVNextConfig.load(ARCH / "BABY_VNEXT_CONFIG.json"))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def greedy_all(model, rows: list[dict], device: torch.device) -> dict[str, tuple[list[int], list[int]]]:
    states = {}
    for r in rows:
        prompt = [BOS] + list(map(int, r["prompt_token_ids"]))
        target = list(map(int, r["candidate_token_ids"][int(r["correct_index"])])) + [EOS]
        states[r["id"]] = {
            "run": prompt,
            "gen": [],
            "target": target,
            "limit": max(MAX_NEW, len(target) + 4),
            "done": False,
        }
    while any(not s["done"] for s in states.values()):
        groups = defaultdict(list)
        for rid, s in states.items():
            if not s["done"]:
                groups[len(s["run"][-CTX:])].append((rid, s))
        for _, group in groups.items():
            batch = torch.tensor([s["run"][-CTX:] for _, s in group], device=device)
            out = model(batch)
            out = out[0] if isinstance(out, tuple) else out
            nxt = out[:, -1].argmax(dim=-1).tolist()
            for (rid, s), token in zip(group, nxt):
                token = int(token)
                s["run"].append(token)
                s["gen"].append(token)
                if token == EOS or len(s["gen"]) >= s["limit"]:
                    s["done"] = True
    return {rid: (s["gen"], s["target"]) for rid, s in states.items()}


@torch.no_grad()
def greedy_from(model, prefixes: list[list[int]], device: torch.device, limit: int = MAX_NEW) -> list[list[int]]:
    states = []
    for p in prefixes:
        run = list(p)
        done = (not run) or run[-1] == EOS
        states.append({"run": run, "gen": [], "done": done})
    while any(not s["done"] for s in states):
        groups = defaultdict(list)
        for i, s in enumerate(states):
            if not s["done"]:
                groups[len(s["run"][-CTX:])].append((i, s))
        for _, group in groups.items():
            batch = torch.tensor([s["run"][-CTX:] for _, s in group], device=device)
            out = model(batch)
            out = out[0] if isinstance(out, tuple) else out
            nxt = out[:, -1].argmax(dim=-1).tolist()
            for (i, s), token in zip(group, nxt):
                token = int(token)
                s["run"].append(token)
                s["gen"].append(token)
                if token == EOS or len(s["gen"]) >= limit:
                    s["done"] = True
    return [s["gen"] for s in states]


@torch.no_grad()
def last_logits(model, ids: list[int], device: torch.device) -> torch.Tensor:
    out = model(torch.tensor([ids[-CTX:]], device=device))
    out = out[0] if isinstance(out, tuple) else out
    return out[0, -1].detach()


@torch.no_grad()
def layer_logits_selected(base, ids: list[int], device: torch.device, layers: tuple[int, ...]) -> dict[int, torch.Tensor]:
    x = torch.tensor([ids[-CTX:]], device=device)
    t = x.shape[1]
    pos = torch.arange(t, device=device)
    h = base.embedding_dropout(base.token_embedding(x) + base.position_embedding(pos))
    outs = [h]
    for b in base.blocks:
        h = b(h)
        outs.append(h)
    picked = {}
    for li in layers:
        z = base.language_head(base.final_norm(outs[li])[0, -1])
        picked[li] = z.detach()
    return picked


def first_div(gen: list[int], target: list[int]) -> int:
    n = max(len(gen), len(target))
    for i in range(n):
        g = gen[i] if i < len(gen) else None
        t = target[i] if i < len(target) else None
        if g != t:
            return i
    return n


def token_class(piece: str, tid: int) -> str:
    if tid == EOS:
        return "eos"
    if tid in (0, 1, 2):
        return "special"
    if not any(ch.isalnum() for ch in piece):
        return "punct"
    if piece.startswith(" "):
        return "leading_space"
    return "continuation"


def logit_stats(logits: torch.Tensor, token: int | None) -> dict | None:
    if token is None:
        return None
    x = logits.float()
    logp = torch.log_softmax(x, dim=-1)
    p = torch.softmax(x, dim=-1)
    top = int(x.argmax().item())
    rank = int((x > x[token]).sum().item() + 1)
    return {
        "token": int(token),
        "p": float(p[token].item()),
        "logp": float(logp[token].item()),
        "rank": rank,
        "top_token": top,
        "top_p": float(p[top].item()),
        "margin_vs_top": float((x[token] - x[top]).item()),
        "logit": float(x[token].item()),
    }


def rate(xs: list[bool]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def fmt_rate(n_yes: int, n: int) -> str:
    if n == 0:
        return "n/a"
    return f"{n_yes}/{n} ({100.0 * n_yes / n:.1f}%)"


def assign_d(target: list[int], div_pos: int, gold: int, inventory: dict[str, list[int]]) -> dict:
    prefix = target[:div_pos]
    rivals = []
    for name in sorted(inventory):
        ids = inventory[name]
        if len(ids) <= div_pos:
            continue
        if ids[:div_pos] != prefix:
            continue
        nxt = int(ids[div_pos])
        if nxt != gold:
            rivals.append({"name": name, "token": nxt})
    if not rivals:
        return {"feasible": False, "token": None, "rival_name": None, "rivals": []}
    chosen = rivals[0]
    return {
        "feasible": True,
        "token": chosen["token"],
        "rival_name": chosen["name"],
        "rivals": rivals,
    }


def assign_b(
    gold: int,
    wrong: int,
    pieces: list[str],
    classes: list[str],
    lengths: list[int],
    freq: Counter,
) -> dict:
    g_class = classes[gold]
    g_len = lengths[gold]
    g_f = int(freq[gold])
    excluded = {gold, wrong}

    def score(tid: int, want_len: bool, want_class: bool) -> tuple:
        return (
            0 if (not want_class or classes[tid] == g_class) else 1,
            abs(lengths[tid] - g_len) if not want_len else (0 if lengths[tid] == g_len else 10**9),
            abs(int(freq[tid]) - g_f),
            abs(tid - gold),
            tid,
        )

    def pick(want_len: bool, want_class: bool) -> int | None:
        cands = []
        for tid in range(VOCAB):
            if tid in excluded:
                continue
            if want_class and classes[tid] != g_class:
                continue
            if want_len and lengths[tid] != g_len:
                continue
            cands.append(tid)
        if not cands:
            return None
        return min(cands, key=lambda t: score(t, want_len, want_class))

    token = pick(True, True)
    tier = "class_and_length_freq"
    if token is None:
        token = pick(False, True)
        tier = "class_closest_length_freq"
    if token is None:
        token = pick(False, False)
        tier = "freq_only"
    if token is None:
        raise RuntimeError("unable to assign matched control B")
    return {
        "token": int(token),
        "piece": pieces[token],
        "class": classes[token],
        "utf8_len": lengths[token],
        "train_freq": int(freq[token]),
        "gold_class": g_class,
        "gold_utf8_len": g_len,
        "gold_train_freq": g_f,
        "match_tier": tier,
    }


def family_of(name: str) -> str:
    return name if name in {"Sal", "Skye", "Omar", "Opal", "Wes"} else "Other"


def observed_surface(gen_text: str, name: str) -> str:
    t = gen_text.strip()
    if name == "Sal" and t.startswith("Salt"):
        return "Sal_to_Salt"
    if name == "Skye" and (t == "Sky." or t.startswith("Sky.") or t == "Sky"):
        return "Skye_to_Sky"
    if name == "Wes" and t.startswith("Walt"):
        return "Wes_to_Walt"
    if name == "Omar":
        return "Omar"
    if name == "Opal":
        return "Opal"
    return "other_surface"


def outcome_metrics(answer: list[int], target: list[int], div_pos: int) -> dict:
    exact = answer == target
    next_i = div_pos + 1
    if next_i < len(target):
        next_tok = answer[next_i] if next_i < len(answer) else None
        next_ok = next_tok == target[next_i]
    else:
        next_tok = None
        next_ok = exact
    rem_ok = answer[div_pos + 1 :] == target[div_pos + 1 :] if div_pos + 1 <= len(target) else exact
    second = None
    n = max(len(answer), len(target))
    for i in range(n):
        a = answer[i] if i < len(answer) else None
        t = target[i] if i < len(target) else None
        if a != t:
            second = i
            break
    free_matched = 0
    if second is None:
        free_matched = max(0, len(target) - div_pos - 1)
    else:
        free_matched = max(0, second - div_pos - 1)
    eos = bool(answer) and answer[-1] == EOS
    period_eos = len(answer) >= 2 and answer[-2] == 18 and answer[-1] == EOS
    return {
        "answer_ids": answer,
        "exact_plus_eos": bool(exact),
        "next_token_recovery": bool(next_ok),
        "remaining_suffix_exact": bool(rem_ok),
        "eos": eos,
        "period_eos": period_eos,
        "second_divergence_pos": second,
        "free_tokens_matched_after_repair": int(free_matched),
        "next_token": next_tok,
    }


def classify(results: dict) -> str:
    gold = results["conditions"]["A_GOLD_REPAIR"]
    b = results["conditions"]["B_WRONG_MATCHED"]
    d = results["conditions"]["D_PREFIX_COMPATIBLE"]
    seeds = [gold["by_seed"][str(s)]["exact_plus_eos"] for s in SEEDS]
    if any(gold["by_seed"][str(s)]["n"] < 20 for s in SEEDS) or (max(seeds) - min(seeds) >= 0.30):
        return "INCONCLUSIVE"
    shared = gold["by_shared_prefix"]["True"]["exact_plus_eos"]
    unique = gold["by_shared_prefix"]["False"]["exact_plus_eos"]
    ns = gold["by_shared_prefix"]["True"]["n"]
    nu = gold["by_shared_prefix"]["False"]["n"]
    if ns >= 20 and nu >= 20 and shared is not None and unique is not None and abs(shared - unique) >= 0.30:
        return "MIXED_BY_COLLISION_FAMILY"
    g = gold["pooled"]["exact_plus_eos"]
    b_ex = b["pooled"]["exact_plus_eos"] or 0.0
    d_ex = d["pooled_feasible"]["exact_plus_eos"] or 0.0
    delta = g - max(b_ex, d_ex)
    mn = min(seeds)
    nxt = gold["pooled"]["next_token_recovery"]
    rem = gold["pooled"]["remaining_suffix_exact"]
    if g >= 0.70 and mn >= 0.60 and delta >= 0.40:
        return "EARLY_TOKEN_ERROR_CAUSALLY_DOMINATES"
    if g >= 0.40 and mn >= 0.25 and delta >= 0.20:
        return "EARLY_TOKEN_ERROR_MAJOR_CONTRIBUTOR"
    seed_partial = 0
    for s in SEEDS:
        row = gold["by_seed"][str(s)]
        if row["next_token_recovery"] >= 0.40 or row["remaining_suffix_exact"] >= 0.20:
            seed_partial += 1
    if g < 0.40 and ((nxt or 0) >= 0.50 or (rem or 0) >= 0.30) and seed_partial >= 2:
        return "EARLY_TOKEN_REPAIR_PARTIAL_RECOVERY"
    if g < 0.15 and (nxt or 0) < 0.20:
        return "EARLY_TOKEN_REPAIR_NO_MEANINGFUL_EFFECT"
    return "INCONCLUSIVE"


def cond_summary(rows: list[dict], key: str, feasible_only: bool = False) -> dict:
    use = []
    for r in rows:
        block = r["conditions"][key]
        if feasible_only and not block.get("feasible", True):
            continue
        if block.get("skipped"):
            continue
        use.append(block)
    def by(pred):
        sub = [u for u, r in zip(use, rows) if pred(r) and (not feasible_only or r["conditions"][key].get("feasible", True))]
        # simpler: filter rows first
        return sub
    # rebuild from rows
    filtered = []
    for r in rows:
        block = r["conditions"][key]
        if feasible_only and not block.get("feasible", True):
            continue
        if block.get("skipped"):
            continue
        filtered.append((r, block))

    def pack(pairs):
        if not pairs:
            return {"n": 0, "exact_plus_eos": None, "next_token_recovery": None, "remaining_suffix_exact": None, "eos": None, "period_eos": None}
        b = [p[1] for p in pairs]
        return {
            "n": len(b),
            "exact_plus_eos": rate([x["exact_plus_eos"] for x in b]),
            "next_token_recovery": rate([x["next_token_recovery"] for x in b]),
            "remaining_suffix_exact": rate([x["remaining_suffix_exact"] for x in b]),
            "eos": rate([x["eos"] for x in b]),
            "period_eos": rate([x["period_eos"] for x in b]),
            "mean_free_tokens_matched": sum(x["free_tokens_matched_after_repair"] for x in b) / len(b),
        }

    out = {
        "pooled": pack(filtered),
        "by_seed": {str(s): pack([(r, b) for r, b in filtered if r["seed"] == s]) for s in SEEDS},
        "by_shared_prefix": {
            "True": pack([(r, b) for r, b in filtered if r["shared_first_token_dev"]]),
            "False": pack([(r, b) for r, b in filtered if not r["shared_first_token_dev"]]),
        },
        "by_family": {},
    }
    for fam in ("Sal", "Skye", "Omar", "Opal", "Wes", "Other"):
        out["by_family"][fam] = pack([(r, b) for r, b in filtered if r["family"] == fam])
    return out


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    for bad in FORBIDDEN:
        if bad in str(DEV_PATH) or bad in str(INV_PATH):
            raise RuntimeError("refusing forbidden path")
    if sha(TOK_PATH) != TOK_SHA:
        raise RuntimeError("tokenizer hash mismatch")
    if sha(DEV_PATH) != DEV_SHA:
        raise RuntimeError("qa_dev hash mismatch")
    if sha(STREAM) != STREAM_SHA:
        raise RuntimeError("language stream hash mismatch")
    tok = Tokenizer.from_file(str(TOK_PATH))
    pieces = [tok.decode([i]) for i in range(VOCAB)]
    classes = [token_class(pieces[i], i) for i in range(VOCAB)]
    lengths = [len(pieces[i].encode("utf-8")) for i in range(VOCAB)]
    freq = language_counts()
    dev = load_jsonl(DEV_PATH)
    inv_raw = json.loads(INV_PATH.read_text(encoding="utf-8"))
    inventory = {k: list(map(int, v)) for k, v in inv_raw["names"].items()}
    all_dev_names = sorted({n for r in dev for n in r["candidates"]})
    name_ids = {n: tok.encode(" " + n).ids for n in all_dev_names}

    def shared_first(name: str) -> bool:
        ids = name_ids[name]
        return any(name_ids[x][0] == ids[0] for x in all_dev_names if x != name)

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    greedy_store: dict[int, dict[str, tuple[list[int], list[int]]]] = {}
    population: list[dict] = []
    for seed in SEEDS:
        model = load_model(seed, device)
        generated = greedy_all(model, dev, device)
        greedy_store[seed] = generated
        for r in dev:
            gen, target = generated[r["id"]]
            if not gen or gen[0] != target[0] or gen == target:
                continue
            div = first_div(gen, target)
            gold = int(target[div]) if div < len(target) else None
            wrong = int(gen[div]) if div < len(gen) else None
            if gold is None:
                continue
            name = r["correct_name"]
            d_ctrl = assign_d(target, div, gold, inventory)
            b_ctrl = assign_b(gold, wrong if wrong is not None else -1, pieces, classes, lengths, freq)
            population.append({
                "seed": seed,
                "id": r["id"],
                "name": name,
                "family": family_of(name),
                "shared_first_token_dev": shared_first(name),
                "target_ids": target,
                "generated_ids": gen,
                "generated_text": tok.decode(gen),
                "target_text": tok.decode(target),
                "first_divergence_pos": div,
                "gold_token": gold,
                "gold_piece": pieces[gold],
                "wrong_token": wrong,
                "wrong_piece": pieces[wrong] if wrong is not None else None,
                "shared_prefix_ids": target[:div],
                "remaining_suffix_ids": target[div:],
                "tokenization_pieces": [pieces[t] for t in target],
                "observed_surface": observed_surface(tok.decode(gen), name),
                "prompt": [BOS] + list(map(int, r["prompt_token_ids"])),
                "control_B": b_ctrl,
                "control_D": {
                    "feasible": d_ctrl["feasible"],
                    "token": d_ctrl["token"],
                    "piece": pieces[d_ctrl["token"]] if d_ctrl["token"] is not None else None,
                    "rival_name": d_ctrl["rival_name"],
                    "n_rivals": len(d_ctrl["rivals"]),
                },
            })
        del model
        torch.cuda.empty_cache()

    by_seed_n = {s: sum(1 for x in population if x["seed"] == s) for s in SEEDS}
    if by_seed_n != EXPECTED_POP:
        raise RuntimeError(f"population mismatch: {by_seed_n} != {EXPECTED_POP}")

    pop_public = []
    for x in population:
        pop_public.append({k: v for k, v in x.items() if k != "prompt"})
    write(OUT / "POPULATION_FREEZE.json", {
        "frozen_before_rescue": True,
        "n": len(population),
        "by_seed": by_seed_n,
        "controls_assigned_before_rescue": True,
        "rows": pop_public,
    })

    # PART 2-6: rescue + probs + internal (no patching)
    rows_out: list[dict] = []
    internal_agg = defaultdict(lambda: {"n": 0, "top1": 0, "rank_sum": 0.0, "margin_sum": 0.0})
    for seed in SEEDS:
        model = load_model(seed, device)
        base = model.base_model
        seed_rows = [x for x in population if x["seed"] == seed]
        prefixes_a, prefixes_b, prefixes_d = [], [], []
        d_index = []
        for x in seed_rows:
            prompt = x["prompt"]
            shared = x["shared_prefix_ids"]
            prefixes_a.append(prompt + shared + [x["gold_token"]])
            prefixes_b.append(prompt + shared + [x["control_B"]["token"]])
            if x["control_D"]["feasible"]:
                prefixes_d.append(prompt + shared + [x["control_D"]["token"]])
                d_index.append(True)
            else:
                d_index.append(False)
        cont_a = greedy_from(model, prefixes_a, device)
        cont_b = greedy_from(model, prefixes_b, device)
        cont_d = greedy_from(model, prefixes_d, device) if prefixes_d else []
        d_iter = iter(cont_d)
        for x, ca, cb, has_d in zip(seed_rows, cont_a, cont_b, d_index):
            target = x["target_ids"]
            div = x["first_divergence_pos"]
            gold = x["gold_token"]
            wrong = x["wrong_token"]
            ans_a = x["shared_prefix_ids"] + [gold] + ca
            ans_b = x["shared_prefix_ids"] + [x["control_B"]["token"]] + cb
            ans_c = x["generated_ids"]
            conds = {
                "A_GOLD_REPAIR": {"feasible": True, **outcome_metrics(ans_a, target, div), "intervention_token": gold, "intervention_piece": x["gold_piece"]},
                "B_WRONG_MATCHED": {"feasible": True, **outcome_metrics(ans_b, target, div), "intervention_token": x["control_B"]["token"], "intervention_piece": x["control_B"]["piece"]},
                "C_NO_INTERVENTION": {"feasible": True, **outcome_metrics(ans_c, target, div), "intervention_token": wrong, "intervention_piece": x["wrong_piece"]},
            }
            if has_d:
                cd = next(d_iter)
                ans_d = x["shared_prefix_ids"] + [x["control_D"]["token"]] + cd
                conds["D_PREFIX_COMPATIBLE"] = {
                    "feasible": True,
                    **outcome_metrics(ans_d, target, div),
                    "intervention_token": x["control_D"]["token"],
                    "intervention_piece": x["control_D"]["piece"],
                    "equals_natural_wrong": x["control_D"]["token"] == wrong,
                }
            else:
                conds["D_PREFIX_COMPATIBLE"] = {"feasible": False, "skipped": True}

            prompt = x["prompt"]
            at_div = last_logits(model, prompt + x["shared_prefix_ids"], device)
            after_gold = last_logits(model, prompt + x["shared_prefix_ids"] + [gold], device)
            next_gold = target[div + 1] if div + 1 < len(target) else None
            seq = {
                "at_divergence": {
                    "gold": logit_stats(at_div, gold),
                    "wrong": logit_stats(at_div, wrong) if wrong is not None else None,
                    "B": logit_stats(at_div, x["control_B"]["token"]),
                    "D": logit_stats(at_div, x["control_D"]["token"]) if x["control_D"]["feasible"] else None,
                },
                "after_gold_repair_next": logit_stats(after_gold, next_gold) if next_gold is not None else None,
                "gold_suffix_trajectory": [],
            }
            running = prompt + x["shared_prefix_ids"] + [gold]
            for step_i in range(div + 1, len(target)):
                lg = last_logits(model, running, device)
                want = target[step_i]
                seq["gold_suffix_trajectory"].append(logit_stats(lg, want))
                running.append(want)

            internal = {"after_gold": {}, "after_wrong": {}}
            if wrong is not None and next_gold is not None:
                g_layers = layer_logits_selected(base, prompt + x["shared_prefix_ids"] + [gold], device, BLOCK_LAYERS)
                w_layers = layer_logits_selected(base, prompt + x["shared_prefix_ids"] + [wrong], device, BLOCK_LAYERS)
                for li in BLOCK_LAYERS:
                    gs = logit_stats(g_layers[li], next_gold)
                    ws = logit_stats(w_layers[li], next_gold)
                    internal["after_gold"][str(li)] = {"rank": gs["rank"], "top1": gs["rank"] == 1, "margin_vs_top": gs["margin_vs_top"], "p": gs["p"]}
                    internal["after_wrong"][str(li)] = {"rank": ws["rank"], "top1": ws["rank"] == 1, "margin_vs_top": ws["margin_vs_top"], "p": ws["p"]}
                    for tag, st in (("gold", gs), ("wrong", ws)):
                        key = f"layer{li}:{tag}"
                        internal_agg[key]["n"] += 1
                        internal_agg[key]["top1"] += int(st["rank"] == 1)
                        internal_agg[key]["rank_sum"] += st["rank"]
                        internal_agg[key]["margin_sum"] += st["margin_vs_top"]

            rows_out.append({
                "seed": seed,
                "id": x["id"],
                "name": x["name"],
                "family": x["family"],
                "shared_first_token_dev": x["shared_first_token_dev"],
                "observed_surface": x["observed_surface"],
                "first_divergence_pos": div,
                "gold_token": gold,
                "gold_piece": x["gold_piece"],
                "wrong_token": wrong,
                "wrong_piece": x["wrong_piece"],
                "target_text": x["target_text"],
                "generated_text": x["generated_text"],
                "control_B": x["control_B"],
                "control_D": x["control_D"],
                "conditions": {
                    k: {kk: vv for kk, vv in v.items() if kk != "answer_ids"} | {"answer_text": tok.decode(v["answer_ids"]) if v.get("answer_ids") else None}
                    if not v.get("skipped") else v
                    for k, v in conds.items()
                },
                "sequence_probability": seq,
                "internal_state_compact": internal,
            })
            # restore answer_ids stripped: decode already done; keep exact flags
        del model
        torch.cuda.empty_cache()

    # strip leftover answer_ids if present
    for r in rows_out:
        for k, v in r["conditions"].items():
            v.pop("answer_ids", None)

    cond_keys = {
        "A_GOLD_REPAIR": False,
        "B_WRONG_MATCHED": False,
        "C_NO_INTERVENTION": False,
    }
    summaries = {k: cond_summary(rows_out, k, False) for k in cond_keys}
    summaries["D_PREFIX_COMPATIBLE"] = cond_summary(rows_out, "D_PREFIX_COMPATIBLE", feasible_only=True)
    summaries["D_PREFIX_COMPATIBLE"]["pooled_feasible"] = summaries["D_PREFIX_COMPATIBLE"]["pooled"]
    summaries["D_PREFIX_COMPATIBLE"]["n_infeasible"] = sum(1 for r in rows_out if not r["conditions"]["D_PREFIX_COMPATIBLE"].get("feasible"))

    classification = classify({"conditions": summaries})

    def seq_pool(field_path):
        vals = []
        for r in rows_out:
            cur = r["sequence_probability"]
            for part in field_path:
                cur = None if cur is None else cur.get(part) if isinstance(cur, dict) else None
            if cur and "p" in cur:
                vals.append(cur)
        if not vals:
            return None
        return {
            "n": len(vals),
            "mean_p": mean([v["p"] for v in vals]),
            "mean_logp": mean([v["logp"] for v in vals]),
            "mean_rank": mean([v["rank"] for v in vals]),
            "top1_rate": rate([v["rank"] == 1 for v in vals]),
            "mean_margin_vs_top": mean([v["margin_vs_top"] for v in vals]),
        }

    seq_summary = {
        "at_divergence_gold": seq_pool(["at_divergence", "gold"]),
        "at_divergence_wrong": seq_pool(["at_divergence", "wrong"]),
        "after_gold_repair_next": seq_pool(["after_gold_repair_next"]),
    }
    traj_p = []
    traj_rank = []
    traj_top1 = []
    for r in rows_out:
        for step in r["sequence_probability"]["gold_suffix_trajectory"]:
            traj_p.append(step["p"])
            traj_rank.append(step["rank"])
            traj_top1.append(step["rank"] == 1)
    seq_summary["gold_suffix_trajectory_pooled"] = {
        "n_steps": len(traj_p),
        "mean_p": mean(traj_p),
        "mean_rank": mean(traj_rank),
        "top1_rate": rate(traj_top1),
    }

    internal_summary = {}
    for k, v in sorted(internal_agg.items()):
        n = v["n"]
        internal_summary[k] = {
            "n": n,
            "top1_rate": v["top1"] / n if n else None,
            "mean_rank": v["rank_sum"] / n if n else None,
            "mean_margin_vs_top": v["margin_sum"] / n if n else None,
        }

    earned = None
    if classification in ("EARLY_TOKEN_ERROR_CAUSALLY_DOMINATES", "EARLY_TOKEN_ERROR_MAJOR_CONTRIBUTOR"):
        earned = (
            "Earned hypothesis only: a single first-wrong-token repair followed by free greedy "
            "recovers exact+EOS often enough, and more than matched/prefix-compatible wrong controls, "
            "that the first divergent token is a major (or causally dominant) contributor to these "
            "first-token-correct-then-diverge failures. This does not authorize activation patching, "
            "tokenizer mutation, T29, or TEST access."
        )

    results = {
        "study": "POST_T28_CAUSAL_PREFIX_RESCUE_V1",
        "status": "COMPLETE",
        "classification": classification,
        "population_n": len(rows_out),
        "population_by_seed": by_seed_n,
        "test_loaded": False,
        "activation_patching": "NOT_RUN",
        "tokenizer_lineage": "PAUSED",
        "t29_authorized": False,
        "conditions": summaries,
        "sequence_probability": seq_summary,
        "internal_state_blocks_7_to_11": internal_summary,
        "earned_hypothesis": earned,
        "rows": rows_out,
    }
    write(OUT / "PREFIX_RESCUE_RESULTS.json", results)
    write(OUT / "STATUS.json", {
        "status": "COMPLETE",
        "classification": classification,
        "n": len(rows_out),
        "test_loaded": False,
        "activation_patching": "NOT_RUN",
    })

    g = summaries["A_GOLD_REPAIR"]
    b = summaries["B_WRONG_MATCHED"]
    c = summaries["C_NO_INTERVENTION"]
    d = summaries["D_PREFIX_COMPATIBLE"]

    def line(label, pack):
        if pack["n"] == 0:
            return f"{label}: n=0"
        n = pack["n"]
        ex = pack["exact_plus_eos"]
        nxt = pack["next_token_recovery"]
        rem = pack["remaining_suffix_exact"]
        return (
            f"{label}: n={n}; exact+EOS {ex:.3f}; next-token {nxt:.3f}; "
            f"remaining-suffix {rem:.3f}; period+EOS {pack['period_eos']:.3f}"
        )

    report = []
    report.append("# Post-T28 causal prefix-rescue study")
    report.append("")
    report.append(f"Status: **COMPLETE — `{classification}`**.")
    report.append("")
    report.append("Tokenizer engineering remains paused. T29 was not launched. No optimizer was created.")
    report.append("Checkpoints and v0_7 were not modified. TEST / FINAL / sacred were not loaded.")
    report.append("Activation patching was not run.")
    report.append("")
    report.append("## Primary question")
    report.append("")
    report.append("When Baby makes the first wrong answer token after an otherwise-correct answer prefix,")
    report.append("if we repair only that token and return to free greedy AR generation, does she recover")
    report.append("the correct answer?")
    report.append("")
    report.append("## Population (frozen before rescue)")
    report.append("")
    report.append(
        f"DEV first-token-correct-then-diverge across T28 seeds 850001–850003: "
        f"{by_seed_n[850001]}/{by_seed_n[850002]}/{by_seed_n[850003]} (pooled {len(rows_out)}). "
        "All rows were retained. Sal/Skye were not used as a filter."
    )
    report.append("")
    report.append("Control tokens B and D were assigned from frozen population fields, inventory geometry,")
    report.append("and class/length/frequency matching before any rescue metric was computed.")
    report.append("")
    report.append("## Primary intervention (A: gold repair)")
    report.append("")
    report.append(line("Pooled", g["pooled"]))
    for s in SEEDS:
        report.append(line(f"Seed {s}", g["by_seed"][str(s)]))
    report.append(line("Shared-first DEV names", g["by_shared_prefix"]["True"]))
    report.append(line("Unique-first DEV names", g["by_shared_prefix"]["False"]))
    for fam in ("Sal", "Skye", "Omar", "Opal", "Wes", "Other"):
        report.append(line(f"Family {fam}", g["by_family"][fam]))
    report.append("")
    report.append("## Controls")
    report.append("")
    report.append(line("B wrong-matched", b["pooled"]))
    for s in SEEDS:
        report.append(line(f"B seed {s}", b["by_seed"][str(s)]))
    report.append(line("C no intervention", c["pooled"]))
    report.append(
        line("D prefix-compatible (feasible rows only)", d["pooled"])
        + f"; infeasible rows={d['n_infeasible']}"
    )
    for s in SEEDS:
        report.append(line(f"D seed {s}", d["by_seed"][str(s)]))
    report.append("")
    report.append("## Sequence probability (descriptive)")
    report.append("")
    for k, v in seq_summary.items():
        report.append(f"- `{k}`: {v}")
    report.append("")
    report.append("These are final-layer readout statistics. They do not by themselves prove that Baby")
    report.append("uses any intermediate state as a causal code.")
    report.append("")
    report.append("## Internal state after wrong vs gold (blocks 7–11, frozen head, no probe)")
    report.append("")
    for k, v in internal_summary.items():
        report.append(f"- `{k}`: top1={v['top1_rate']:.3f} mean_rank={v['mean_rank']:.1f} n={v['n']}")
    report.append("")
    report.append("A negative or weak readout is not evidence that information is absent.")
    report.append("")
    report.append("## Activation patching")
    report.append("")
    report.append("Not run. If gold rescue is large and replicated, the earned hypothesis is recorded below")
    report.append("and the study stops. No next treatment was designed.")
    report.append("")
    report.append("## Classification")
    report.append("")
    report.append(f"`{classification}`")
    report.append("")
    if earned:
        report.append("## Earned hypothesis")
        report.append("")
        report.append(earned)
        report.append("")
    report.append("## Conservative interpretation")
    report.append("")
    report.append("This is a single-token oracle intervention on already-identified diverge cases.")
    report.append("It does not show that Baby would have sampled the gold token, does not repair the")
    report.append("tokenizer, and does not authorize retokenization, T29, or TEST.")
    report.append("")
    OUT.joinpath("PREFIX_RESCUE_FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    gp = g["pooled"]
    op = []
    op.append("STUDY: POST_T28_CAUSAL_PREFIX_RESCUE_V1")
    op.append("PRIMARY QUESTION: first-wrong-token gold repair then free greedy — does exact answer recover?")
    op.append(f"POPULATION: DEV first-token-correct-then-diverge {by_seed_n[850001]}/{by_seed_n[850002]}/{by_seed_n[850003]} pooled {len(rows_out)}")
    op.append("CHERRY PICK SAL/SKYE: NO")
    op.append(f"GOLD RESCUE EXACT+EOS: {fmt_rate(int(round(gp['exact_plus_eos']*gp['n'])), gp['n'])} pooled; "
              + "; ".join(fmt_rate(int(round(g['by_seed'][str(s)]['exact_plus_eos']*g['by_seed'][str(s)]['n'])), g['by_seed'][str(s)]['n']) for s in SEEDS))
    op.append(f"NEXT-TOKEN RECOVERY: {gp['next_token_recovery']:.3f} pooled")
    op.append(f"REMAINING-SUFFIX EXACT: {gp['remaining_suffix_exact']:.3f} pooled")
    op.append(f"EOS/PERIOD+EOS: {gp['eos']:.3f} / {gp['period_eos']:.3f}")
    op.append(f"CONTROL B WRONG-MATCHED EXACT+EOS: {b['pooled']['exact_plus_eos']:.3f} n={b['pooled']['n']}")
    op.append(f"CONTROL C NO INTERVENTION EXACT+EOS: {c['pooled']['exact_plus_eos']:.3f} (0 expected)")
    op.append(f"CONTROL D PREFIX-COMPATIBLE EXACT+EOS: {d['pooled']['exact_plus_eos']} n_feasible={d['pooled']['n']} infeasible={d['n_infeasible']}")
    op.append(f"SHARED-FIRST GOLD EXACT+EOS: {g['by_shared_prefix']['True']['exact_plus_eos']}")
    op.append(f"UNIQUE-FIRST GOLD EXACT+EOS: {g['by_shared_prefix']['False']['exact_plus_eos']}")
    for fam in ("Sal", "Skye", "Omar", "Opal", "Wes"):
        pack = g["by_family"][fam]
        op.append(f"FAMILY {fam} GOLD EXACT+EOS: {pack['exact_plus_eos']} n={pack['n']}")
    op.append(f"AT-DIVERGENCE GOLD: {seq_summary['at_divergence_gold']}")
    op.append(f"AT-DIVERGENCE WRONG: {seq_summary['at_divergence_wrong']}")
    op.append(f"AFTER GOLD REPAIR NEXT: {seq_summary['after_gold_repair_next']}")
    op.append("INTERNAL STATE: compact block 7-11 frozen-head readout after gold vs wrong; no probe")
    op.append("ACTIVATION PATCHING: NOT RUN")
    op.append(f"CLASSIFICATION: {classification}")
    op.append(f"EARNED HYPOTHESIS: {earned or 'none'}")
    op.append("TOKENIZER LINEAGE: PAUSED")
    op.append("T29: NOT AUTHORIZED / NOT LAUNCHED")
    op.append("TEST / FINAL / SACRED: SEALED")
    op.append("TRAINING / OPTIMIZER / CHECKPOINT / TOKENIZER WRITES: NONE")
    op.append("GIT COMMIT: pending")
    op.append("PUSH STATUS: pending")
    OUT.joinpath("OPERATOR_SUMMARY.md").write_text("\n".join(op) + "\n", encoding="utf-8")

    write(OUT / "PROVENANCE.json", {
        "status": "COMPLETE",
        "read_only": True,
        "date": "2026-09-13",
        "forward_pass_only": True,
        "optimizer_created": False,
        "training_updates": 0,
        "activation_patching": "NOT_RUN",
        "inputs": {
            "qa_dev.jsonl": sha(DEV_PATH),
            "tokenizer": sha(TOK_PATH),
            "phase1g_language_train_stream": sha(STREAM),
            "name_inventory": sha(INV_PATH),
            "t28_seed850001": CKPT_SHA[850001],
            "t28_seed850002": CKPT_SHA[850002],
            "t28_seed850003": CKPT_SHA[850003],
            "phase1g_parent": PARENT_SHA,
        },
        "protected_data": {
            "T3_TEST": "SEALED_UNOPENED",
            "T2_EVAL_TEST": "LOCKED_UNOPENED",
            "FINAL": "LOCKED",
            "sacred": "LOCKED",
            "loaded": False,
        },
        "classification": classification,
    })


if __name__ == "__main__":
    main()
