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
S2A_SURVIVOR = OUT / "s2a_protect40_combine_322001" / "checkpoint_00200.pt"
S2A_SURVIVOR_SHA = "0642a2f2a4044d9936cc3f930fa096ef55fc69c5e7cfe6f7c539a3c47b5fd959"
S2I25_SURVIVOR = OUT / "s2i_protect40_combine_323051" / "checkpoint_00025.pt"
S2I25_SURVIVOR_SHA = "664e8ead0f922f1ee937f4e1ef12c8f1f85d196fa03f7b9a6ec48bd905229bb3"
S2I50_SURVIVOR = OUT / "s2i_protect40_combine_323051" / "checkpoint_00050.pt"
S2I50_SURVIVOR_SHA = "c2137471f1d53adce3fd272794380b140f6cd12606ed706f6bffc0d95e1e1659"
S2M_SURVIVOR = OUT / "s2m_protect40_combine_323091" / "checkpoint_00025.pt"
S2M_SURVIVOR_SHA = "bd10f0ea8239fe09bd5bec933611ba92b27c20402b24474ad287331153240ff9"
S2A_D3_SLICE = 0.825
S2A_D3_FULL = 169
S2A_MIXED = 0.875
S2A_COMBINE = 0.65625
S2A_USABLE = 0.952
S2A_USABLE4 = 0.969
S2A_STOP = 1.0
S2A_REUSE = 1.0
RECOVER_UPDATES = 50
RECOVER_EVAL = 25
LOCK_UPDATES = 20
LOCK_EVAL = 10
D3_SLICE_SIGNAL = 0.05
D3_SLICE_HOLDPLUS = 0.05
D3_SLICE_ADVANCE = 0.075
D3_SLICE_KILL = 0.125
RECOVER_ORDER = ("s2g", "s2d", "s2h", "s2j", "s2i", "s2f", "s2k", "s2e")
# Mix-starved 5–10% remainder arms (s2g/s2h/s2j) lift D3 slice but kill combine.
# s2i pulse: u25 mix still holds at D3 slice 0.875; u50 D3 slice 0.950, combine 0.562.
# Next: lock mix from those pulses; independently steal structured not mix.
RECOVER_MIXHOLD_ORDER = ("s2o", "s2q", "s2n", "s2p", "s2l", "s2m")

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
    "s2d": {
        "seed": 323001,
        "mix": "protect40_combine",
        "language_p": 0.55,
        "structured_p": 0.35,
        "updates": RECOVER_UPDATES,
        "eval_every": RECOVER_EVAL,
        "parent": "s2a",
        "cheap_d3": True,
        "note": "From s2a: language-heavy restore of D3 greedy span (first-token already 215/215).",
    },
    "s2e": {
        "seed": 323011,
        "mix": "protect40_combine",
        "language_p": 0.20,
        "structured_p": 0.70,
        "updates": RECOVER_UPDATES,
        "eval_every": RECOVER_EVAL,
        "parent": "s2a",
        "cheap_d3": True,
        "note": "From s2a: structured keyed-span CE restore. Mix hold 10%.",
    },
    "s2f": {
        "seed": 323021,
        "mix": "protect40_combine",
        "language_p": 0.40,
        "structured_p": 0.50,
        "updates": RECOVER_UPDATES,
        "eval_every": RECOVER_EVAL,
        "parent": "s2a",
        "cheap_d3": True,
        "note": "From s2a: language+structured span restore, 10% mix hold.",
    },
    "s2g": {
        "seed": 323031,
        "mix": "protect40_combine",
        "language_p": 0.70,
        "structured_p": 0.20,
        "updates": RECOVER_UPDATES,
        "eval_every": RECOVER_EVAL,
        "parent": "s2a",
        "cheap_d3": True,
        "note": "From s2a: cut mix 45%→10% and structured 40%→20%; language restores greedy span. Prior lesson: skip extra structured.",
    },
    "s2h": {
        "seed": 323041,
        "mix": "protect40_combine",
        "language_p": 0.45,
        "structured_p": 0.45,
        "updates": RECOVER_UPDATES,
        "eval_every": RECOVER_EVAL,
        "parent": "s2a",
        "cheap_d3": True,
        "remainder_span": True,
        "note": "From s2a: remainder-masked keyed/induction CE (skip first answer token) + 10% mix hold.",
    },
    "s2i": {
        "seed": 323051,
        "mix": "protect40_combine",
        "language_p": 0.90,
        "structured_p": 0.10,
        "updates": RECOVER_UPDATES,
        "eval_every": RECOVER_EVAL,
        "parent": "s2a",
        "cheap_d3": True,
        "note": "From s2a: mix-off diagnostic. If D3 lifts and mix dies, mix dose is the span tax.",
    },
    "s2j": {
        "seed": 323061,
        "mix": "protect40_combine",
        "language_p": 0.80,
        "structured_p": 0.15,
        "updates": RECOVER_UPDATES,
        "eval_every": RECOVER_EVAL,
        "parent": "s2a",
        "cheap_d3": True,
        "note": "From s2a: 5% mix hold, language-heavy span restore.",
    },
    "s2k": {
        "seed": 323071,
        "mix": "protect40_combine",
        "language_p": 0.70,
        "structured_p": 0.20,
        "updates": RECOVER_UPDATES,
        "eval_every": RECOVER_EVAL,
        "parent": "s2a",
        "cheap_d3": True,
        "lr_scale": 0.5,
        "note": "s2g mix at half LR.",
    },
    "s2l": {
        "seed": 323081,
        "mix": "protect40_combine",
        "language_p": 0.45,
        "structured_p": 0.10,
        "updates": RECOVER_EVAL,
        "eval_every": RECOVER_EVAL,
        "parent": "s2a",
        "cheap_d3": True,
        "note": "From s2a: keep 45% mix; steal structured→language for span restore.",
    },
    "s2m": {
        "seed": 323091,
        "mix": "protect40_combine",
        "language_p": 0.30,
        "structured_p": 0.25,
        "updates": RECOVER_EVAL,
        "eval_every": RECOVER_EVAL,
        "parent": "s2a",
        "cheap_d3": True,
        "remainder_span": True,
        "note": "From s2a: keep 45% mix; remainder-span structured + modest language.",
    },
    "s2n": {
        "seed": 323101,
        "mix": "protect40_combine",
        "language_p": 0.15,
        "structured_p": 0.40,
        "updates": LOCK_UPDATES,
        "eval_every": LOCK_EVAL,
        "parent": "s2i25",
        "cheap_d3": True,
        "note": "Lock s2i@25 (mix still at s2a, D3 slice 0.875) with the original s2a diet.",
    },
    "s2o": {
        "seed": 323111,
        "mix": "combine_heavy",
        "language_p": 0.0,
        "structured_p": 0.0,
        "updates": LOCK_UPDATES,
        "eval_every": LOCK_EVAL,
        "parent": "s2i50",
        "cheap_d3": True,
        "lock_from_drop": True,
        "note": "Pure combine-heavy lock from s2i@50 (D3 slice 0.950, combine 0.562). Isolates whether mix restores combine without erasing the span.",
    },
    "s2p": {
        "seed": 323121,
        "mix": "protect40_combine",
        "language_p": 0.05,
        "structured_p": 0.15,
        "updates": LOCK_UPDATES,
        "eval_every": LOCK_EVAL,
        "parent": "s2i50",
        "cheap_d3": True,
        "lock_from_drop": True,
        "remainder_span": True,
        "mix_remainder_span": True,
        "lr_scale": 0.5,
        "note": "Gentle half-LR lock from s2i@50; remainder-span on structured and mix answers.",
    },
    "s2q": {
        "seed": 323131,
        "mix": "combine_heavy",
        "language_p": 0.10,
        "structured_p": 0.20,
        "updates": LOCK_UPDATES,
        "eval_every": LOCK_EVAL,
        "parent": "s2i50",
        "cheap_d3": True,
        "lock_from_drop": True,
        "note": "From s2i@50: 70% combine-heavy mix + 10% language + 20% structured.",
    },
    "s3a": {
        "seed": 324001,
        "mix": "compose_bind",
        "language_p": 0.20,
        "structured_p": 0.20,
        "updates": 25,
        "eval_every": 25,
        "parent": "s2m",
        "cheap_d3": True,
        "remainder_span": True,
        "compose": True,
        "note": "From s2m: who-bind + 3-entity mix/combine. Keep 20/20/60 language/structured/mix.",
    },
    "s3b": {
        "seed": 324011,
        "mix": "compose_sent",
        "language_p": 0.20,
        "structured_p": 0.20,
        "updates": 25,
        "eval_every": 25,
        "parent": "s2m",
        "cheap_d3": True,
        "remainder_span": True,
        "compose": True,
        "note": "From s2m: unprompted short-sentence answers (held-out templates). No 'answer in a sentence' prefix.",
    },
    "s3c": {
        "seed": 324021,
        "mix": "compose_both",
        "language_p": 0.15,
        "structured_p": 0.20,
        "updates": 25,
        "eval_every": 25,
        "parent": "s2m",
        "cheap_d3": True,
        "remainder_span": True,
        "compose": True,
        "note": "From s2m: who-bind + unprompted sentences together.",
    },
    "s3d": {
        "seed": 324031,
        "mix": "compose_sent_instr",
        "language_p": 0.20,
        "structured_p": 0.20,
        "updates": 25,
        "eval_every": 25,
        "parent": "s2m",
        "cheap_d3": True,
        "remainder_span": True,
        "compose": True,
        "note": "Scaffold diagnostic: sentence-prefix instructions. Kill if only the prefix operator moves.",
    },
    "s3e": {
        "seed": 324041,
        "mix": "compose_phrase",
        "language_p": 0.20,
        "structured_p": 0.20,
        "updates": 25,
        "eval_every": 25,
        "parent": "s2m",
        "cheap_d3": True,
        "remainder_span": True,
        "compose": True,
        "note": "From s2m: existing phrase family (query asks for a sentence) vs unprompted s3b.",
    },
    "s3i": {
        "seed": 324051,
        "mix": "compose_bind_protect",
        "language_p": 0.25,
        "structured_p": 0.20,
        "updates": 25,
        "eval_every": 25,
        "parent": "s2m",
        "cheap_d3": True,
        "remainder_span": True,
        "compose": True,
        "note": "s3a who-bind worked (0→0.44) but combine dropped. Same who skill, more 2e combine protection.",
    },
    "s3k": {
        "seed": 324071,
        "mix": "protect40_combine",
        "language_p": 0.30,
        "structured_p": 0.25,
        "updates": 25,
        "eval_every": 25,
        "parent": "s2m",
        "cheap_d3": True,
        "remainder_span": True,
        "compose": True,
        "lock_from_drop": True,
        "note": "Lock s3a who-pulse onto s2m mix diet. Test whether who-bind sticks while combine recovers.",
    },
    "s3n": {
        "seed": 324081,
        "mix": "compose_lock",
        "language_p": 0.25,
        "structured_p": 0.25,
        "updates": 25,
        "eval_every": 25,
        "parent": "s2m",
        "cheap_d3": True,
        "remainder_span": True,
        "compose": True,
        "lock_from_drop": True,
        "lr_scale": 0.5,
        "note": "From s3a: half-LR lock with 18% who rehearsal so who-bind does not vanish while combine returns.",
    },
    "s3p": {
        "seed": 324091,
        "mix": "compose_lock",
        "language_p": 0.25,
        "structured_p": 0.25,
        "updates": 25,
        "eval_every": 25,
        "parent": "s2m",
        "cheap_d3": True,
        "remainder_span": True,
        "compose": True,
        "lr_scale": 0.5,
        "note": "Extend s3n +25. who=0.25 is a real signal, not a milestone; keep mix-held lock.",
    },
    "s3r": {
        "seed": 324101,
        "mix": "compose_lock_story",
        "language_p": 0.25,
        "structured_p": 0.25,
        "updates": 25,
        "eval_every": 25,
        "parent": "s2m",
        "cheap_d3": True,
        "remainder_span": True,
        "compose": True,
        "lock_from_drop": True,
        "lr_scale": 0.5,
        "note": "From s3a: half-LR restore of 2e combine + story_combine with who rehearsal. Balanced who panel n=32.",
    },
    "s3v": {
        "seed": 324111,
        "mix": "compose_lock_who",
        "language_p": 0.30,
        "structured_p": 0.25,
        "updates": 25,
        "eval_every": 25,
        "parent": "s2m",
        "cheap_d3": True,
        "remainder_span": True,
        "compose": True,
        "lock_from_drop": True,
        "lr_scale": 0.5,
        "note": "From s3r (D3 183, combine 0.719): more who rehearsal under s2m language/structured diet.",
    },
    "s3s": {
        "seed": 324121,
        "mix": "compose_lock",
        "language_p": 0.35,
        "structured_p": 0.30,
        "updates": 25,
        "eval_every": 25,
        "parent": "s2m",
        "cheap_d3": True,
        "remainder_span": True,
        "compose": True,
        "lock_from_drop": True,
        "lr_scale": 0.5,
        "note": "From s3n (who3=0.50, D3 159, first=215): steal mix→language/structured remainder like s2m D3 restore, keep who rehearsal.",
    },
    "s3w": {
        "seed": 324131,
        "mix": "compose_lock_story",
        "language_p": 0.30,
        "structured_p": 0.25,
        "updates": 25,
        "eval_every": 25,
        "parent": "s2m",
        "cheap_d3": True,
        "remainder_span": True,
        "compose": True,
        "lock_from_drop": True,
        "lr_scale": 0.5,
        "note": "From s3s (who2=0.50, combine 0.656): s3r-style combine/D3 lock while rehearsing who.",
    },
}


def parent_checkpoint(recipe: dict) -> Path:
    parent = recipe.get("parent")
    if parent == "s2a":
        return S2A_SURVIVOR
    if parent == "s2i25":
        return S2I25_SURVIVOR
    if parent == "s2i50":
        return S2I50_SURVIVOR
    if parent == "s2m":
        return S2M_SURVIVOR
    return E12_SURVIVOR


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
    if mix.startswith("compose_"):
        from .selection_stack2_s3 import sample_s3_item

        return sample_s3_item(rng, tokenizer, mix)
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


def require_identities(*, require_s2a: bool = False, require_s2m: bool = False) -> None:
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("U16000 hash mismatch")
    if digest(E12_SURVIVOR) != E12_SURVIVOR_SHA:
        raise RuntimeError("E12 survivor hash mismatch")
    if digest(D3_PATH) != EXPECTED_D3_SHA:
        raise RuntimeError(f"D3 sha mismatch: {digest(D3_PATH)}")
    if require_s2a:
        if not S2A_SURVIVOR.exists():
            raise RuntimeError(f"missing s2a parent {S2A_SURVIVOR}")
        if digest(S2A_SURVIVOR) != S2A_SURVIVOR_SHA:
            raise RuntimeError("s2a survivor hash mismatch")
    if require_s2m:
        if not S2M_SURVIVOR.exists():
            raise RuntimeError(f"missing s2m parent {S2M_SURVIVOR}")
        if digest(S2M_SURVIVOR) != S2M_SURVIVOR_SHA:
            raise RuntimeError("s2m survivor hash mismatch")


def cheap_d3_slice(model, device) -> dict:
    overwrite, hooks = attach_d3_for_retention(model, device)
    try:
        return cheap_d3_retention(model, overwrite, device)
    finally:
        uninstall_c2_d3(*hooks)
        overwrite.gen_index = None


def mix_holds_s2a(native: dict) -> tuple[bool, str]:
    scores = mix_scores(native)
    mixed_ok = scores["mixed_2e"] + 1e-12 >= S2A_MIXED - DROP_BAR
    combine_ok = scores["fact_combine"] + 1e-12 >= S2A_COMBINE - DROP_BAR
    lesson = f"mixed={scores['mixed_2e']:.3f} fact_combine={scores['fact_combine']:.3f}"
    if mixed_ok and combine_ok:
        return True, f"s2a mix hold {lesson}"
    return False, f"s2a mix drop {lesson}"


def usable_holds_s2a(usable: dict | None) -> tuple[bool, str]:
    ok, lesson, collapsed = usable_holds(usable)
    if usable is None:
        return False, "usable-chat missing"
    if collapsed or not ok:
        return False, lesson
    auto = usable.get("autoregressive") or {}
    auto4 = usable.get("autoregressive_4turn") or auto
    turn = float(auto.get("usable_turn") or 0.0)
    turn4 = float(auto4.get("usable_turn") or 0.0)
    stop = float(auto4.get("period_stop") or auto.get("period_stop") or 0.0)
    reuse = float(auto4.get("fact_reuse") or auto.get("fact_reuse") or 0.0)
    hold = (
        turn4 + 1e-12 >= S2A_USABLE4 - DROP_BAR
        and turn + 1e-12 >= S2A_USABLE - DROP_BAR
        and stop + 1e-12 >= S2A_STOP - DROP_BAR
        and reuse + 1e-12 >= S2A_REUSE - DROP_BAR
    )
    note = f"usable4={turn4:.3f} usable={turn:.3f} stop={stop:.3f} reuse={reuse}"
    if hold:
        return True, f"s2a usable hold {note}"
    return False, f"s2a usable drop {note}"


def remainder_span_mask(mask: torch.Tensor) -> torch.Tensor:
    """Drop the first target token from CE so training matches D3 first-step overwrite."""
    if mask.ndim != 2:
        raise ValueError("remainder_span_mask expects [batch, time]")
    out = mask.clone()
    counts = out.sum(dim=1)
    has_remainder = counts > 1
    if not bool(has_remainder.any()):
        return out
    first = out.to(dtype=torch.int64).argmax(dim=1)
    rows = torch.nonzero(has_remainder, as_tuple=False).squeeze(1)
    out[rows, first[rows]] = False
    return out


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
    start = resume or parent_checkpoint(recipe)
    expected = {
        E12_SURVIVOR: E12_SURVIVOR_SHA,
        S2A_SURVIVOR: S2A_SURVIVOR_SHA,
        S2I25_SURVIVOR: S2I25_SURVIVOR_SHA,
        S2I50_SURVIVOR: S2I50_SURVIVOR_SHA,
        S2M_SURVIVOR: S2M_SURVIVOR_SHA,
    }.get(start)
    if expected is not None and digest(start) != expected:
        raise RuntimeError(f"parent hash mismatch for {start}")
    model, config, _ckpt = load_experimental_baby(start, device)
    print(json.dumps({"phase": "resume", "checkpoint": str(start), "sha256": digest(start), "recipe": recipe_id}), flush=True)
    optimizer, _low, _high = capability_optimizer(model)
    lr_scale = float(recipe.get("lr_scale", 1.0))
    if lr_scale != 1.0:
        for group in optimizer.param_groups:
            group["lr"] = float(group["lr"]) * lr_scale
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
            if recipe.get("remainder_span"):
                rem = remainder_span_mask(mask)
                if bool(rem.any()):
                    mask = rem
            logits = model(x)
            loss = F.cross_entropy(logits[mask], y[mask])
            task = "structured_remainder" if recipe.get("remainder_span") else "structured"
        else:
            items = [sample_s2_item(rng, tokenizer, mix) for _ in range(BATCH)]
            x, y, mask = pack_bridge_batch(items, device)
            if recipe.get("mix_remainder_span"):
                rem = remainder_span_mask(mask)
                if bool(rem.any()):
                    mask = rem
            logits = model(x)
            loss = F.cross_entropy(logits[mask], y[mask])
            task = "bridge_remainder" if recipe.get("mix_remainder_span") else "bridge"
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
            if recipe.get("cheap_d3"):
                slice_row = cheap_d3_slice(model, device)
                row["d3_slice"] = {k: slice_row.get(k) for k in ("n", "first_top1", "free_exact", "tf_exact")}
                row["lesson"] = f"{row['lesson']}; D3 slice free_exact={slice_row.get('free_exact')}"
            if recipe.get("compose"):
                from .data_language_bridge import build_s3_panels
                from .selection_language_bridge import run_sentence_decode
                from .selection_stack2_s3 import LIGHT_SENTENCE, slim_sentence

                compose_native = eval_panels(model, build_s3_panels(tokenizer, n=32), device, overwrite=None, arms=("native",))["native"]
                row["compose"] = slim_panels(compose_native)
                sentence = run_sentence_decode(model, tokenizer, device, include=LIGHT_SENTENCE)
                row["sentence"] = slim_sentence(sentence)
                extra_sent = (row["sentence"].get("operators") or {}).get("bare") or {}
                row["lesson"] = (
                    f"{row['lesson']}; who2={_top1(compose_native, 'who_bind_2e_heldout'):.3f} "
                    f"sent={_exact(compose_native, 'compose_sent_heldout'):.3f} "
                    f"bare={float(extra_sent.get('sentence_ok') or 0.0):.3f}"
                )
            history.append(row)
            write(out_dir / f"eval_{update:05d}.json", row)
            extra = {k: row[k] for k in ("mix_verdict", "english_ok", "collapsed", "lesson")}
            if "d3_slice" in row:
                extra["d3_slice"] = row["d3_slice"]
            if "compose" in row:
                extra["compose"] = {
                    name: {k: row["compose"][name].get(k) for k in ("first_top1", "free_exact")}
                    for name in row["compose"]
                }
            print(json.dumps({"phase": f"{recipe_id}_eval", "update": update, **extra}, default=str), flush=True)
            best = row
            slice_exact = float((row.get("d3_slice") or {}).get("free_exact") or 0.0)
            if recipe.get("parent") == "s2m":
                from .selection_stack2_s3 import S2M_D3_SLICE, mix_holds_s2m

                mix_ok_hold, _mix_hold = mix_holds_s2m(native)
                d3_floor = S2M_D3_SLICE - D3_SLICE_KILL
            else:
                mix_ok_hold, _mix_hold = mix_holds_s2a(native)
                d3_floor = S2A_D3_SLICE - D3_SLICE_KILL
            mix_kill = (not mix_ok_hold) and (not recipe.get("lock_from_drop"))
            kill_canary = collapsed or (
                bool(recipe.get("cheap_d3")) and (mix_kill or slice_exact + 1e-12 < d3_floor)
            )
            if kill_canary:
                print(json.dumps({"phase": "kill_canary", "recipe": recipe_id, "update": update, "lesson": row["lesson"]}, default=str), flush=True)
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


def adjudicate_recover(row: dict, *, baseline_slice: float = S2A_D3_SLICE) -> tuple[str, str]:
    native = row.get("native") or {}
    eng_ok, eng_lesson, collapsed = english_native_holds(native)
    mix_ok, mix_lesson = mix_holds_s2a(native)
    slice_exact = float((row.get("d3_slice") or {}).get("free_exact") or 0.0)
    first = float((row.get("d3_slice") or {}).get("first_top1") or 0.0)
    delta = slice_exact - float(baseline_slice)
    d3_note = f"D3 slice {slice_exact:.3f} (parent {float(baseline_slice):.3f}) first={first:.3f}"
    if collapsed or not eng_ok:
        return "KILL", f"{eng_lesson}; {mix_lesson}; {d3_note}"
    if not mix_ok:
        return "KILL", f"{mix_lesson}; {eng_lesson}; {d3_note}"
    if delta <= -D3_SLICE_KILL:
        return "KILL", f"{d3_note} delta={delta:+.3f}; {mix_lesson}; {eng_lesson}"
    if delta >= D3_SLICE_ADVANCE or slice_exact >= 0.90:
        return "ADVANCE", f"{d3_note} delta={delta:+.3f}; {mix_lesson}; {eng_lesson}"
    if delta >= D3_SLICE_HOLDPLUS:
        return "HOLD+", f"{d3_note} delta={delta:+.3f}; {mix_lesson}; {eng_lesson}"
    return "HOLD", f"{d3_note} delta={delta:+.3f}; {mix_lesson}; {eng_lesson}"


def _slice_exact(row: dict | None) -> float:
    if not row:
        return -1.0
    return float((row.get("d3_slice") or {}).get("free_exact") or 0.0)


def _drop_cuda(model=None) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def slim_recover_row(row: dict) -> dict:
    return {k: v for k, v in row.items() if k != "native"} | {"native": mix_scores(row["native"])}


def measure_s2a_slice(device) -> dict:
    model, _config, _ckpt = load_experimental_baby(S2A_SURVIVOR, device)
    try:
        return cheap_d3_slice(model, device)
    finally:
        _drop_cuda(model)


def _score_history(recipe_id: str, history: list, *, baseline_slice: float) -> dict:
    best_row = None
    best_key = (-1.0, -1)
    rank = {"KILL": -1, "HOLD": 0, "HOLD+": 1, "ADVANCE": 2}
    scored = []
    for row in history:
        verdict, lesson = adjudicate_recover(row, baseline_slice=baseline_slice)
        slice_exact = _slice_exact(row)
        ledger_append(
            {
                "id": f"{recipe_id}_u{int(row['update']):05d}",
                "change": RECIPES[recipe_id]["note"],
                "verdict": verdict,
                "lesson": lesson,
                "native": mix_scores(row["native"]),
                "d3": row.get("d3_slice"),
                "checkpoint_sha256": row.get("checkpoint_sha256"),
                "recipe": recipe_id,
                "update": row.get("update"),
            }
        )
        print(
            json.dumps(
                {"phase": "recover_canary", "recipe": recipe_id, "update": row["update"], "verdict": verdict, "lesson": lesson},
                default=str,
            ),
            flush=True,
        )
        scored.append({**slim_recover_row(row), "recover_verdict": verdict, "recover_lesson": lesson})
        if verdict == "KILL":
            continue
        key = (slice_exact, rank.get(verdict, 0))
        if key > best_key:
            best_key = key
            best_row = {**row, "recover_verdict": verdict, "recover_lesson": lesson}
    return {
        "history": scored,
        "best": None
        if best_row is None
        else {
            "update": best_row["update"],
            "checkpoint": best_row["checkpoint"],
            "checkpoint_sha256": best_row["checkpoint_sha256"],
            "d3_slice": best_row.get("d3_slice"),
            "verdict": best_row.get("recover_verdict"),
            "lesson": best_row.get("recover_lesson"),
            "native": mix_scores(best_row["native"]),
        },
        "best_row": best_row,
    }


def run_canary(device, recipe_id: str, *, resume: Path | None = None, baseline_slice: float) -> dict:
    trained = train_recipe(device, recipe_id, resume=resume)
    scored = _score_history(recipe_id, trained.get("history") or [], baseline_slice=baseline_slice)
    scored["out_dir"] = str(trained.get("out_dir"))
    _drop_cuda(trained.pop("model", None))
    return scored


def register_extend(base_id: str) -> str:
    ext_id = f"{base_id}x"
    n = 1
    while ext_id in RECIPES:
        n += 1
        ext_id = f"{base_id}x{n}"
    src = RECIPES[base_id]
    RECIPES[ext_id] = {
        **src,
        "seed": int(src["seed"]) + 17 * n,
        "updates": RECOVER_UPDATES,
        "eval_every": RECOVER_EVAL,
        "cheap_d3": True,
        "note": f"Extend {base_id} +{RECOVER_UPDATES} from its canary checkpoint",
    }
    return ext_id


def recover_milestone(d3_free, native, usable) -> bool:
    mix_ok, _ = mix_holds_s2a(native)
    eng_ok, _, _ = english_native_holds(native)
    use_ok, _ = usable_holds_s2a(usable)
    return bool(
        d3_free is not None
        and d3_free >= 190
        and mix_ok
        and eng_ok
        and use_ok
        and d3_free - S2A_D3_FULL >= 15
    )


def verify_candidate(device, recipe_id: str, best_row: dict, results: dict) -> dict:
    ckpt = Path(best_row["checkpoint"])
    print(json.dumps({"phase": "recover_verify", "recipe": recipe_id, "checkpoint": str(ckpt)}, default=str), flush=True)
    report = evaluate_checkpoint(device, ckpt, f"{recipe_id}_verify", with_usable=True, with_d3=True)
    d3_free = None if report.get("d3") is None else int(report["d3"]["long_gap"]["free_exact"])
    mix_ok, mix_lesson = mix_holds_s2a(report["native"])
    eng_ok, eng_lesson, _ = english_native_holds(report["native"])
    use_ok, use_lesson = usable_holds_s2a(report.get("usable"))
    milestone = recover_milestone(d3_free, report["native"], report.get("usable"))
    payload = {
        "recipe": recipe_id,
        "checkpoint": str(ckpt),
        "sha256": best_row.get("checkpoint_sha256"),
        "d3_full": d3_free,
        "mix_ok": mix_ok,
        "eng_ok": eng_ok,
        "use_ok": use_ok,
        "milestone": milestone,
        "lesson": f"{mix_lesson}; {eng_lesson}; {use_lesson}; D3 {d3_free}/215",
        "native": mix_scores(report["native"]),
        "usable": None
        if report.get("usable") is None
        else {
            "turn": report["usable"]["autoregressive"]["usable_turn"],
            "turn4": report["usable"]["autoregressive_4turn"]["usable_turn"],
            "stop": report["usable"]["autoregressive_4turn"]["period_stop"],
            "reuse": report["usable"]["autoregressive_4turn"]["fact_reuse"],
        },
        "d3": None if report.get("d3") is None else report["d3"].get("long_gap"),
        "induction": None if report.get("d3") is None else report["d3"].get("primitive_induction", {}).get("first_top1"),
    }
    results.setdefault("verifies", []).append(payload)
    results["last_verify"] = payload
    results["milestone"] = bool(results.get("milestone")) or milestone
    if milestone:
        results["winner"] = payload
    campaign_update(
        {
            "status": "D3 recover MILESTONE" if milestone else f"D3 recover verify {recipe_id} d3={d3_free}",
            "recover_winner": payload if milestone else results.get("winner"),
            "last_verify": {k: payload[k] for k in ("recipe", "checkpoint", "sha256", "d3_full", "milestone", "lesson") if k in payload},
        }
    )
    write(OUT / "RECOVER.json", {k: v for k, v in results.items() if k != "promising_row"})
    return payload


def run_d3_recover_loop(device) -> dict:
    require_identities(require_s2a=True)
    ledger_init()
    parent = measure_s2a_slice(device)
    baseline_slice = float(parent.get("free_exact") or S2A_D3_SLICE)
    campaign_update(
        {
            "status": "D3 span-recover canaries from s2a",
            "s2a_parent": {"path": str(S2A_SURVIVOR), "sha256": S2A_SURVIVOR_SHA},
            "hypothesis": "s2a D3 hole is greedy span. Mix-off pulse (s2i) restores span: u25 keeps mix, u50 reaches slice 0.95 then kills combine. Do not cut mix to 5–10%. Lock mix from s2i pulses; independently steal structured not mix remainder.",
            "parent_slice": {k: parent.get(k) for k in ("n", "first_top1", "free_exact", "tf_exact")},
            "baseline_slice": baseline_slice,
        }
    )
    print(json.dumps({"phase": "s2a_parent_slice", "slice": parent, "baseline_slice": baseline_slice}, default=str), flush=True)
    prior_arms = {}
    recover_path = OUT / "RECOVER.json"
    if recover_path.exists():
        try:
            prior_arms = json.loads(recover_path.read_text(encoding="utf-8")).get("arms") or {}
        except json.JSONDecodeError:
            prior_arms = {}
    results = {
        "parent_slice": {k: parent.get(k) for k in ("n", "first_top1", "free_exact", "tf_exact")},
        "baseline_slice": baseline_slice,
        "arms": dict(prior_arms),
        "verifies": [],
        "milestone": False,
    }
    promising = None

    def consider(recipe_id: str, scored: dict):
        nonlocal promising
        results["arms"][recipe_id] = {k: v for k, v in scored.items() if k != "best_row"}
        write(OUT / "RECOVER.json", {k: v for k, v in results.items() if k != "promising_row"})
        best_row = scored.get("best_row")
        if best_row is None:
            return None
        if promising is None or _slice_exact(best_row) > _slice_exact(promising[1]):
            if best_row.get("recover_verdict") in {"HOLD+", "ADVANCE"}:
                promising = (recipe_id, best_row)
        return best_row

    for recipe_id in RECOVER_MIXHOLD_ORDER:
        print(json.dumps({"phase": "recover_arm", "recipe": recipe_id, "note": RECIPES[recipe_id]["note"]}, default=str), flush=True)
        scored = run_canary(device, recipe_id, baseline_slice=baseline_slice)
        best_row = consider(recipe_id, scored)
        if best_row is None:
            continue
        verdict = best_row.get("recover_verdict")
        if verdict == "HOLD+":
            ext_id = register_extend(recipe_id)
            print(json.dumps({"phase": "recover_extend", "from": recipe_id, "to": ext_id}, default=str), flush=True)
            ext = run_canary(device, ext_id, resume=Path(best_row["checkpoint"]), baseline_slice=baseline_slice)
            ext_row = consider(ext_id, ext)
            if ext_row is not None:
                best_row, recipe_id, verdict = ext_row, ext_id, ext_row.get("recover_verdict")
        if verdict == "ADVANCE":
            verified = verify_candidate(device, recipe_id, best_row, results)
            if verified.get("milestone"):
                print(json.dumps({"phase": "recover_done", "milestone": True, "winner": results.get("winner")}, default=str), flush=True)
                return results
            d3_free = verified.get("d3_full")
            if (
                d3_free is not None
                and d3_free >= 180
                and verified.get("mix_ok")
                and verified.get("use_ok")
            ):
                ext_id = register_extend(recipe_id)
                print(json.dumps({"phase": "recover_extend_signal", "from": recipe_id, "to": ext_id, "d3_full": d3_free}, default=str), flush=True)
                ext = run_canary(device, ext_id, resume=Path(best_row["checkpoint"]), baseline_slice=baseline_slice)
                ext_row = consider(ext_id, ext)
                if ext_row is not None and ext_row.get("recover_verdict") in {"HOLD+", "ADVANCE"}:
                    verified = verify_candidate(device, ext_id, ext_row, results)
                    if verified.get("milestone"):
                        print(json.dumps({"phase": "recover_done", "milestone": True, "winner": results.get("winner")}, default=str), flush=True)
                        return results

    if promising is not None and not results.get("milestone"):
        recipe_id, best_row = promising
        if _slice_exact(best_row) >= baseline_slice + D3_SLICE_HOLDPLUS:
            verified = verify_candidate(device, recipe_id, best_row, results)
            if verified.get("milestone"):
                print(json.dumps({"phase": "recover_done", "milestone": True, "winner": results.get("winner")}, default=str), flush=True)
                return results
    campaign_update(
        {
            "status": "D3 recover canaries finished without milestone",
            "recover": {k: v.get("best") for k, v in results.get("arms", {}).items()},
            "last_verify": results.get("last_verify"),
        }
    )
    write(OUT / "RECOVER.json", {k: v for k, v in results.items() if k != "promising_row"})
    print(json.dumps({"phase": "recover_done", "milestone": False, "last_verify": results.get("last_verify")}, default=str), flush=True)
    return results


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
    parser.add_argument("--mode", choices=("run", "probe", "train", "usable", "d3", "recover", "compose"), default="run")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recipe", default="s2a", choices=tuple(RECIPES))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--tag", default="eval")
    args = parser.parse_args()
    device = resolve_device(args.device)
    require_identities(
        require_s2a=args.mode in {"recover", "compose"} or RECIPES.get(args.recipe, {}).get("parent") in {"s2a", "s2m"},
        require_s2m=args.mode == "compose" or RECIPES.get(args.recipe, {}).get("parent") == "s2m",
    )
    ledger_init()
    if args.mode == "probe":
        path = args.resume or E12_SURVIVOR
        evaluate_checkpoint(device, path, args.tag, with_usable=True, with_d3=False)
        return
    if args.mode == "train":
        if args.resume is not None:
            train_recipe(device, args.recipe, resume=args.resume)
        else:
            run_recipe(device, args.recipe)
        return
    if args.mode == "recover":
        run_d3_recover_loop(device)
        return
    if args.mode == "compose":
        from .selection_stack2_s3 import run_compose_loop

        run_compose_loop(device)
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
