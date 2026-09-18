from __future__ import annotations

"""Stack2 experimental line from E12: escape the D3 ceiling without sacrificing chat.

U16000 stays authoritative. TEST/FINAL/SACRED stay sealed. Never loads P11 Baby
weights. Never golds query_position. Does not rewrite the E12 STOP ledger.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import LANG_TRAIN, read_u16
from .data_language_bridge import (
    banks_from_language,
    build_e13_panels,
    load_tokenizer,
    make_dialogue_item,
    make_english_item,
    make_mixed_item,
    make_open_item,
    make_saystop_item,
    make_size_item,
    make_story_item,
    make_story_mixed_item,
)
from .evaluate import language_ce
from .selection_language_bridge import (
    BATCH,
    CANARY_EVAL,
    CANARY_UPDATES,
    E12_SURVIVOR,
    E12_SURVIVOR_SHA,
    EXPECTED_D3_SHA,
    PROTOCOL,
    attach_d3_for_retention,
    cheap_d3_retention,
    eval_panels,
    load_experimental_baby,
    pack_bridge_batch,
    run_decode_canary,
    run_usable_chat,
    slim_panels,
    structured_retention_batch,
    uninstall_c2_d3,
)
from .selection_p11_u16000_runtime import resolve_device
from .selection_rapid_treat import POLICY_A1, measure_stage1
from .selection_s1 import PARENT, PARENT_SHA, digest, write
from .train_v2r4 import DEV_STREAM, capability_optimizer, language_batch, set_seed

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "actual_baby" / "stack2"
LEDGER = OUT / "LEDGER.md"
CAMPAIGN = OUT / "CAMPAIGN.json"
D3_PATH = ROOT / "src" / "baby_v010" / "selection_rapid_treat_d.py"
HIST_E13 = ROOT / "runs" / "actual_baby" / "e13_mix_311301" / "checkpoint_00050.pt"
HIST_E13_SHA = "4fa5060de08076458416d660429bdfed9fec989e292a36472b3278ecbdae2838"

DROP_BAR = 0.05
E12_COLOR = 0.96875
E12_SIZE_STOP = 0.90625
E12_STORY = 0.9375
E12_USABLE = 0.857
E12_USABLE_4 = 0.906
E12_STOP = 1.0
E12_REUSE = 1.0
E12_INDUCTION = 0.421875
E12_D3 = 202
COLOR_COLLAPSE = 0.70
SIZE_STOP_COLLAPSE = 0.60
USABLE_COLLAPSE = 0.70
INDUCTION_FLOOR = 0.297
MIX_COMBINE_GATE = 0.50
MIX_MIXED_GATE = 0.60
MIX_SIGNAL = 0.50

RECIPES = {
    "s2a": {
        "seed": 322001,
        "mix": "protect40_combine",
        "language_p": 0.15,
        "structured_p": 0.40,
        "updates": CANARY_UPDATES,
        "eval_every": CANARY_EVAL,
        "note": "E12 dual-objective mix: 40% structured + ~40% E12 color/size/dialogue/open; remaining mix/combine. D3 199 allowed if English holds.",
    },
    "s2b": {
        "seed": 322011,
        "mix": "e13_replica",
        "language_p": 0.20,
        "structured_p": 0.25,
        "updates": CANARY_UPDATES,
        "eval_every": CANARY_EVAL,
        "note": "Exact E13 English mix that taught fact-combine 0.531 at D3 199; new English-first retention.",
    },
    "s2c": {
        "seed": 322021,
        "mix": "combine_heavy",
        "language_p": 0.15,
        "structured_p": 0.40,
        "updates": CANARY_UPDATES,
        "eval_every": CANARY_EVAL,
        "note": "Heavier fact-combine dose if s2a/s2b keep English but mix stays cold.",
    },
}


def ledger_init() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if LEDGER.exists():
        return
    LEDGER.write_text(
        "# Actual Baby stack2 ledger\n\n"
        "Experimental line from E12 (`e12_stoponly_311211/checkpoint_00050.pt`).\n"
        "U16000 remains AUTHORITATIVE. C2/D3 stay frozen fallbacks (a1|b1|c2|d3).\n"
        "TEST/FINAL/SACRED stay sealed. Do not promote stack2 or E12 over U16000.\n\n"
        "E12 STOP is a closed-vocab chat milestone, not a failure. This stack attacks\n"
        "parked mix/combine/sentence/who/place/copy/happened/5-turn without the old\n"
        "D3<200 auto-kill. Primary retention is E12 English (usable-chat, period-stop,\n"
        "color, size-stop). D3 long-gap is measured on every survivor and logged; it is\n"
        "not a sole kill switch.\n\n",
        encoding="utf-8",
    )


def ledger_append(block: dict) -> None:
    ledger_init()
    lines = [
        f"## {block.get('id', 'unknown')}",
        "",
        f"- Change: {block.get('change', '')}",
        f"- verdict: **{block.get('verdict')}**",
        f"- lesson: {block.get('lesson', '')}",
    ]
    for key in ("native", "usable", "d3", "induction", "checkpoint_sha256", "recipe", "update"):
        if key in block:
            lines.append(f"- {key}: {block[key]}")
    lines.append("")
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def campaign_update(patch: dict) -> dict:
    ledger_init()
    current = {}
    if CAMPAIGN.exists():
        current = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    current.update(patch)
    current.setdefault("campaign", "actual_baby_stack2")
    current.setdefault("test_closed", True)
    current.setdefault("u16000_replaced", False)
    current.setdefault("authoritative_baby", {"path": str(PARENT), "sha256": PARENT_SHA})
    current.setdefault(
        "parent_experimental",
        {"path": str(E12_SURVIVOR), "sha256": E12_SURVIVOR_SHA, "authoritative": False},
    )
    write(CAMPAIGN, current)
    return current


def _top1(native: dict, name: str) -> float:
    return float((native.get(name) or {}).get("first_top1") or 0.0)


def _exact(native: dict, name: str) -> float:
    return float((native.get(name) or {}).get("free_exact") or 0.0)


def mix_scores(native: dict) -> dict:
    return {
        "mixed_2e": _top1(native, "mixed_2e_heldout"),
        "fact_combine": _top1(native, "fact_combine_heldout"),
        "story_combine": _top1(native, "story_combine_heldout"),
        "story_mixed": _top1(native, "story_mixed_heldout"),
        "color": _top1(native, "qa_2fact_heldout"),
        "size_stop": _exact(native, "size_stop_heldout"),
        "story_color": _top1(native, "story_color_heldout"),
        "dialogue": _top1(native, "dialogue_2fact_heldout"),
    }


def english_native_holds(native: dict) -> tuple[bool, str, bool]:
    """Return (holds, lesson, collapsed). Drop ~0.05 vs E12; collapse is a hard kill."""
    color = _top1(native, "qa_2fact_heldout")
    stop = _exact(native, "size_stop_heldout")
    story = _top1(native, "story_color_heldout")
    collapsed = color < COLOR_COLLAPSE or stop < SIZE_STOP_COLLAPSE
    drop_fail = (
        color + 1e-12 < E12_COLOR - DROP_BAR
        or stop + 1e-12 < E12_SIZE_STOP - DROP_BAR
        or (story and story + 1e-12 < E12_STORY - DROP_BAR)
    )
    lesson = f"color={color:.3f} size_stop={stop:.3f} story={story:.3f}"
    if collapsed:
        return False, f"E12 collapse {lesson}", True
    if drop_fail:
        return False, f"E12 drop {lesson}", False
    return True, f"E12 English hold {lesson}", False


def usable_holds(usable: dict | None) -> tuple[bool, str, bool]:
    if usable is None:
        return True, "usable-chat not measured", False
    auto = usable.get("autoregressive") or {}
    auto4 = usable.get("autoregressive_4turn") or auto
    turn = float(auto.get("usable_turn") or 0.0)
    turn4 = float(auto4.get("usable_turn") or 0.0)
    stop = float(auto4.get("period_stop") or auto.get("period_stop") or 0.0)
    reuse = float(auto4.get("fact_reuse") or auto.get("fact_reuse") or 0.0)
    ramble = float(auto4.get("rambling") or auto.get("rambling") or 0.0)
    collapsed = turn4 < USABLE_COLLAPSE or stop < 0.80 or ramble > 0.30
    drop_fail = (
        turn4 + 1e-12 < E12_USABLE_4 - DROP_BAR
        or stop + 1e-12 < E12_STOP - DROP_BAR
        or reuse + 1e-12 < E12_REUSE - DROP_BAR
        or turn + 1e-12 < E12_USABLE - DROP_BAR
    )
    lesson = f"usable4={turn4:.3f} usable={turn:.3f} stop={stop:.3f} reuse={reuse}"
    if collapsed:
        return False, f"usable-chat collapse {lesson}", True
    if drop_fail:
        return False, f"usable-chat drop {lesson}", False
    return True, f"usable-chat hold {lesson}", False


def mix_gate(native: dict) -> tuple[str, str]:
    scores = mix_scores(native)
    combine_ok = scores["fact_combine"] >= MIX_COMBINE_GATE or scores["story_combine"] >= MIX_COMBINE_GATE
    mixed_ok = scores["mixed_2e"] >= MIX_MIXED_GATE
    if mixed_ok and combine_ok:
        return (
            "GRAD",
            f"mix GRAD mixed={scores['mixed_2e']:.3f} fact_combine={scores['fact_combine']:.3f} "
            f"story_combine={scores['story_combine']:.3f} mixed_story={scores['story_mixed']:.3f}",
        )
    if max(scores["mixed_2e"], scores["fact_combine"], scores["story_combine"], scores["story_mixed"]) >= MIX_SIGNAL:
        return (
            "ADVANCE",
            f"mix ADVANCE mixed={scores['mixed_2e']:.3f} fact_combine={scores['fact_combine']:.3f} "
            f"story_combine={scores['story_combine']:.3f} mixed_story={scores['story_mixed']:.3f}",
        )
    return (
        "FAIL",
        f"mix cold mixed={scores['mixed_2e']:.3f} fact_combine={scores['fact_combine']:.3f} "
        f"story_combine={scores['story_combine']:.3f} mixed_story={scores['story_mixed']:.3f}",
    )


def stack2_adjudicate(
    native: dict,
    *,
    usable: dict | None = None,
    d3_free: int | None = None,
    induction: float | None = None,
) -> tuple[str, str, bool]:
    """English-first. D3 is logged; sole D3 184-199 is not a kill. Returns (verdict, lesson, major)."""
    eng_ok, eng_lesson, eng_collapse = english_native_holds(native)
    use_ok, use_lesson, use_collapse = usable_holds(usable)
    mix_verdict, mix_lesson = mix_gate(native)
    d3_note = f" D3 {d3_free}/215" if d3_free is not None else ""
    ind_note = f" induction {induction:.3f}" if induction is not None else ""
    english_dead = (not eng_ok) or (usable is not None and not use_ok)
    collapsed = eng_collapse or use_collapse
    if collapsed:
        return "KILL", f"{eng_lesson}; {use_lesson}; {mix_lesson}{d3_note}{ind_note}", False
    if english_dead:
        return "KILL", f"{eng_lesson}; {use_lesson}; {mix_lesson}{d3_note}{ind_note}", False
    if induction is not None and induction + 1e-12 < INDUCTION_FLOOR and english_dead:
        return "KILL", f"induction tanks with English {eng_lesson}{ind_note}; {mix_lesson}", False
    major = mix_verdict == "GRAD" and eng_ok and use_ok
    if mix_verdict == "GRAD":
        return "SURVIVE", f"{mix_lesson}; {eng_lesson}; {use_lesson}{d3_note}{ind_note}", major
    if mix_verdict == "ADVANCE":
        combine_clear = _top1(native, "fact_combine_heldout") >= MIX_COMBINE_GATE or _top1(native, "story_combine_heldout") >= MIX_COMBINE_GATE
        return "SURVIVE", f"{mix_lesson}; {eng_lesson}; {use_lesson}{d3_note}{ind_note}", combine_clear and eng_ok and use_ok
    return "HOLD", f"{mix_lesson}; {eng_lesson}; {use_lesson}{d3_note}{ind_note}", False


def sample_s2_item(rng: random.Random, tokenizer, mix: str) -> dict:
    roll = rng.random()
    if mix == "e13_replica":
        if roll < 0.12:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.24:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.32:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.40:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.72:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", period=True)
        if roll < 0.84:
            return make_story_mixed_item(rng, tokenizer, surface="train", combine=False, period=True)
        if roll < 0.96:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", combine=True, period=True)
        return make_story_mixed_item(rng, tokenizer, surface="train", combine=True, period=True)
    if mix == "combine_heavy":
        if roll < 0.08:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.16:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.22:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.28:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.32:
            return make_open_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.58:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", period=True)
        if roll < 0.68:
            return make_story_mixed_item(rng, tokenizer, surface="train", combine=False, period=True)
        if roll < 0.90:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", combine=True, period=True)
        return make_story_mixed_item(rng, tokenizer, surface="train", combine=True, period=True)
    if mix != "protect40_combine":
        raise ValueError(f"unknown mix {mix}")
    if roll < 0.10:
        return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
    if roll < 0.20:
        return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
    if roll < 0.28:
        return make_story_item(rng, tokenizer, surface="train", ask="color")
    if roll < 0.36:
        return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
    if roll < 0.40:
        if rng.random() < 0.5:
            return make_open_item(rng, tokenizer, n_facts=2, surface="train", about=True)
        return make_saystop_item(rng, tokenizer, n_facts=2, surface="train")
    if roll < 0.66:
        return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", period=True)
    if roll < 0.76:
        return make_story_mixed_item(rng, tokenizer, surface="train", combine=False, period=True)
    if roll < 0.94:
        return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", combine=True, period=True)
    return make_story_mixed_item(rng, tokenizer, surface="train", combine=True, period=True)


def save_stack2_checkpoint(path: Path, model, optimizer, config, update: int, seed: int, phase: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "protocol": PROTOCOL,
            "lineage": "Baby v0.10 stack2 experimental (E12 parent; U16000 not replaced)",
            "update": update,
            "seed": seed,
            "phase": phase,
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "parent_checkpoint": str(PARENT),
            "parent_checkpoint_sha256": PARENT_SHA,
            "stack2_parent": str(E12_SURVIVOR),
            "stack2_parent_sha256": E12_SURVIVOR_SHA,
            "protected_material_opened": False,
            "authoritative": False,
        },
        tmp,
    )
    os.replace(tmp, path)
    return digest(path)


def require_identities() -> None:
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("U16000 hash mismatch")
    if digest(E12_SURVIVOR) != E12_SURVIVOR_SHA:
        raise RuntimeError("E12 survivor hash mismatch")
    if digest(D3_PATH) != EXPECTED_D3_SHA:
        raise RuntimeError(f"D3 sha mismatch: {digest(D3_PATH)}")


def slim_usable(usable: dict) -> dict:
    return {k: v for k, v in usable.items() if k not in {"chats_auto", "chats_teacher"}}


def run_d3_log(model, device, tag: str, *, english_ok: bool) -> dict:
    overwrite, hooks = attach_d3_for_retention(model, device)
    try:
        cheap = cheap_d3_retention(model, overwrite, device)
        print(json.dumps({"phase": "retention_slice", "tag": tag, "slice": cheap}, default=str), flush=True)
        stage1 = measure_stage1(model, overwrite, device, POLICY_A1)
        long_gap = {k: stage1["long_gap"].get(k) for k in ("n", "free_exact", "first_correct", "free_accuracy", "first_accuracy")}
        induction = float(stage1["primitive_induction"]["first_top1"])
        d3_free = int(long_gap["free_exact"])
        if (not english_ok) and d3_free < 180:
            verdict = "KILL"
            lesson = f"D3 {d3_free}/215 and English dead"
        else:
            verdict = "LOG"
            lesson = f"D3 long-gap {d3_free}/215 induction {induction:.3f} (not an auto-kill)"
        report = {
            "id": f"{tag}_d3",
            "verdict": verdict,
            "lesson": lesson,
            "long_gap": long_gap,
            "primitive_induction": stage1["primitive_induction"],
            "primitive_keyed": stage1["primitive_keyed"],
            "slice": cheap,
            "auto_kill": False,
        }
        write(OUT / f"{tag}_d3.json", {k: v for k, v in report.items()})
        ledger_append(
            {
                "id": report["id"],
                "change": "C2+D3 long-gap logged (stack2: not an auto-kill)",
                "verdict": verdict,
                "lesson": lesson,
                "d3": long_gap,
                "induction": induction,
            }
        )
        print(json.dumps({"phase": "d3_log", "tag": tag, "verdict": verdict, "lesson": lesson}, default=str), flush=True)
        return report
    finally:
        uninstall_c2_d3(*hooks)
        overwrite.gen_index = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_recipe(device, recipe_id: str, *, resume: Path | None = None) -> dict:
    recipe = RECIPES[recipe_id]
    set_seed(int(recipe["seed"]))
    rng = random.Random(int(recipe["seed"]))
    tokenizer = load_tokenizer()
    banks = banks_from_language()
    train_stream = torch.tensor(read_u16(LANG_TRAIN), dtype=torch.long)
    dev_stream = torch.tensor(read_u16(DEV_STREAM), dtype=torch.long)
    start = resume or E12_SURVIVOR
    if start == E12_SURVIVOR and digest(start) != E12_SURVIVOR_SHA:
        raise RuntimeError("E12 survivor hash mismatch")
    model, config, _ckpt = load_experimental_baby(start, device)
    print(json.dumps({"phase": "resume", "checkpoint": str(start), "sha256": digest(start), "recipe": recipe_id}), flush=True)
    optimizer, _low, _high = capability_optimizer(model)
    out_dir = OUT / f"{recipe_id}_{recipe['mix']}_{recipe['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = build_e13_panels(tokenizer, n=32)
    language_p = float(recipe["language_p"])
    structured_p = float(recipe["structured_p"])
    updates = int(recipe["updates"])
    eval_every = int(recipe["eval_every"])
    mix = str(recipe["mix"])
    baseline_ce = language_ce(model, dev_stream, list(range(0, min(64 * 256, dev_stream.numel() - 257), 256)), device, limit=16)
    history = []
    best = None
    for update in range(1, updates + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        draw = rng.random()
        if draw < language_p:
            x, y = language_batch(train_stream, rng, BATCH, 256, device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            task = "language"
        elif draw < language_p + structured_p:
            x, y, mask = structured_retention_batch(banks, rng, BATCH, device)
            logits = model(x)
            loss = F.cross_entropy(logits[mask], y[mask])
            task = "structured"
        else:
            items = [sample_s2_item(rng, tokenizer, mix) for _ in range(BATCH)]
            x, y, mask = pack_bridge_batch(items, device)
            logits = model(x)
            loss = F.cross_entropy(logits[mask], y[mask])
            task = "bridge"
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if update % 25 == 0 or update == 1:
            print(
                json.dumps(
                    {
                        "update": update,
                        "recipe": recipe_id,
                        "task": task,
                        "loss": float(loss.detach().cpu()),
                        "grad_norm": float(grad_norm.detach().cpu()),
                    }
                ),
                flush=True,
            )
        if update % eval_every == 0 or update == updates:
            model.eval()
            native = eval_panels(model, panels, device, overwrite=None, arms=("native",))["native"]
            ce = language_ce(model, dev_stream, list(range(0, min(64 * 256, dev_stream.numel() - 257), 256)), device, limit=16)
            ckpt_path = out_dir / f"checkpoint_{update:05d}.pt"
            sha = save_stack2_checkpoint(ckpt_path, model, optimizer, config, update, int(recipe["seed"]), recipe_id)
            mix_verdict, mix_lesson = mix_gate(native)
            eng_ok, eng_lesson, collapsed = english_native_holds(native)
            row = {
                "update": update,
                "native": slim_panels(native),
                "language_dev_ce": ce,
                "language_baseline_ce": baseline_ce,
                "checkpoint": str(ckpt_path),
                "checkpoint_sha256": sha,
                "mix_verdict": mix_verdict,
                "english_ok": eng_ok,
                "collapsed": collapsed,
                "lesson": f"{mix_lesson}; {eng_lesson}",
            }
            history.append(row)
            write(out_dir / f"eval_{update:05d}.json", row)
            print(json.dumps({"phase": f"{recipe_id}_eval", "update": update, **{k: row[k] for k in ("mix_verdict", "english_ok", "collapsed", "lesson")}}, default=str), flush=True)
            best = row
            if collapsed:
                print(json.dumps({"phase": "kill_english", "recipe": recipe_id, "update": update}), flush=True)
                break
    return {"history": history, "best": best, "model": model, "config": config, "out_dir": out_dir, "recipe": recipe}


def evaluate_checkpoint(device, path: Path, tag: str, *, with_usable: bool = True, with_d3: bool = False) -> dict:
    tokenizer = load_tokenizer()
    model, _config, ckpt = load_experimental_baby(path, device)
    sha = digest(path)
    native = eval_panels(model, build_e13_panels(tokenizer, n=32), device, overwrite=None, arms=("native",))["native"]
    decode = run_decode_canary(model, tokenizer, device)
    usable = run_usable_chat(model, tokenizer, device) if with_usable else None
    eng_ok, _eng_lesson, _collapsed = english_native_holds(native)
    d3 = run_d3_log(model, device, tag, english_ok=eng_ok) if with_d3 else None
    d3_free = None if d3 is None else int(d3["long_gap"]["free_exact"])
    induction = None if d3 is None else float(d3["primitive_induction"]["first_top1"])
    verdict, lesson, major = stack2_adjudicate(
        native,
        usable=slim_usable(usable) if usable else None,
        d3_free=d3_free,
        induction=induction,
    )
    report = {
        "id": tag,
        "resume": str(path),
        "checkpoint_sha256": sha,
        "authoritative": False,
        "protected_material_opened": bool(ckpt.get("protected_material_opened")),
        "native": slim_panels(native),
        "decode": {k: v for k, v in decode.items() if k not in {"rows", "stories"}},
        "usable": slim_usable(usable) if usable else None,
        "d3": None if d3 is None else {k: v for k, v in d3.items() if k != "slice"},
        "verdict": verdict,
        "major": major,
        "lesson": lesson,
    }
    write(OUT / f"{tag}.json", report)
    ledger_append(
        {
            "id": tag,
            "change": f"Stack2 eval {tag} (U16000 not replaced)",
            "verdict": verdict,
            "lesson": lesson,
            "native": mix_scores(native),
            "usable": None
            if usable is None
            else {
                "turn": usable["autoregressive"]["usable_turn"],
                "turn4": usable["autoregressive_4turn"]["usable_turn"],
                "stop": usable["autoregressive_4turn"]["period_stop"],
                "reuse": usable["autoregressive_4turn"]["fact_reuse"],
            },
            "checkpoint_sha256": sha,
        }
    )
    print(json.dumps({"phase": tag, "verdict": verdict, "major": major, "lesson": lesson, "native": mix_scores(native)}, default=str), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def run_recipe(device, recipe_id: str) -> dict:
    trained = train_recipe(device, recipe_id, resume=E12_SURVIVOR)
    best = trained["best"]
    if best is None:
        report = {"id": recipe_id, "verdict": "FAIL", "lesson": "no eval row", "major": False}
        ledger_append({"id": recipe_id, "change": RECIPES[recipe_id]["note"], "verdict": "FAIL", "lesson": report["lesson"], "recipe": recipe_id})
        return report
    tokenizer = load_tokenizer()
    model = trained["model"]
    native = best["native"]
    eng_ok, eng_lesson, collapsed = english_native_holds(native)
    usable = None
    if not collapsed:
        usable = slim_usable(run_usable_chat(model, tokenizer, device))
    d3 = None
    if eng_ok and (usable is None or usable_holds(usable)[0]):
        d3 = run_d3_log(model, device, recipe_id, english_ok=True)
    d3_free = None if d3 is None else int(d3["long_gap"]["free_exact"])
    induction = None if d3 is None else float(d3["primitive_induction"]["first_top1"])
    verdict, lesson, major = stack2_adjudicate(native, usable=usable, d3_free=d3_free, induction=induction)
    report = {
        "id": recipe_id,
        "recipe": RECIPES[recipe_id],
        "best": {k: v for k, v in best.items()},
        "usable": usable,
        "d3": None if d3 is None else {k: v for k, v in d3.items() if k != "slice"},
        "verdict": verdict,
        "major": major,
        "lesson": lesson,
        "english_ok": eng_ok,
        "collapsed": collapsed,
    }
    write(OUT / f"{recipe_id}_final.json", report)
    ledger_append(
        {
            "id": recipe_id,
            "change": RECIPES[recipe_id]["note"],
            "verdict": verdict,
            "lesson": lesson,
            "native": mix_scores(native),
            "checkpoint_sha256": best.get("checkpoint_sha256"),
            "recipe": recipe_id,
            "update": best.get("update"),
        }
    )
    campaign_update(
        {
            "status": f"{recipe_id} {verdict}",
            "last": {k: report[k] for k in ("id", "verdict", "major", "lesson") if k in report},
            recipe_id: {
                "path": best.get("checkpoint"),
                "sha256": best.get("checkpoint_sha256"),
                "verdict": verdict,
                "major": major,
                "mix": mix_scores(native),
                "d3": d3_free,
                "induction": induction,
            },
        }
    )
    print(json.dumps({"phase": f"{recipe_id}_done", "verdict": verdict, "major": major, "lesson": lesson}, default=str), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def run_loop(device) -> dict:
    require_identities()
    ledger_init()
    campaign_update(
        {
            "status": "running first mix/combine canary from E12",
            "retention": "english-first; D3 logged not auto-killed",
            "operators": "a1|b1|c2|d3 frozen U16000 fallbacks",
        }
    )
    zero = evaluate_checkpoint(device, E12_SURVIVOR, "e12_mix_zeroshot", with_usable=True, with_d3=False)
    hist = None
    if HIST_E13.exists() and digest(HIST_E13) == HIST_E13_SHA:
        hist = evaluate_checkpoint(device, HIST_E13, "hist_e13_new_bars", with_usable=True, with_d3=False)
    results = {"zeroshot": {k: v for k, v in zero.items() if k != "usable_full"}, "hist_e13": None if hist is None else {k: v for k, v in hist.items()}}
    order = ["s2a", "s2b", "s2c"]
    for recipe_id in order:
        row = run_recipe(device, recipe_id)
        results[recipe_id] = {k: v for k, v in row.items() if k != "best"}
        if row.get("major"):
            campaign_update({"status": f"MAJOR {recipe_id}: {row['lesson']}", "survivor": recipe_id})
            results["stop"] = f"major capability on {recipe_id}"
            write(OUT / "LOOP.json", results)
            return results
    campaign_update({"status": "stack2 first-pass done; no major combine+chat yet", "results": {k: results.get(k, {}).get("verdict") if isinstance(results.get(k), dict) else results.get(k) for k in ["zeroshot", "hist_e13", "s2a", "s2b", "s2c"]}})
    write(OUT / "LOOP.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "probe", "train", "usable", "d3"), default="run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recipe", default="s2a", choices=tuple(RECIPES))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tag", default="eval")
    args = parser.parse_args()
    device = resolve_device(args.device)
    require_identities()
    ledger_init()
    if args.mode == "probe":
        path = args.resume or E12_SURVIVOR
        evaluate_checkpoint(device, path, args.tag, with_usable=True, with_d3=False)
        return
    if args.mode == "train":
        run_recipe(device, args.recipe)
        return
    if args.mode == "usable":
        if args.resume is None:
            raise SystemExit("--resume is required for usable")
        evaluate_checkpoint(device, args.resume, args.tag, with_usable=True, with_d3=False)
        return
    if args.mode == "d3":
        if args.resume is None:
            raise SystemExit("--resume is required for d3")
        model, _config, _ckpt = load_experimental_baby(args.resume, device)
        run_d3_log(model, device, args.tag, english_ok=True)
        return
    run_loop(device)


if __name__ == "__main__":
    main()
