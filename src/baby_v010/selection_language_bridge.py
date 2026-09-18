from __future__ import annotations

"""Language-bridge campaign: less-rigid bind/retrieval toward usable English.

Eval-time C2/D3 stay selectable fallbacks on the original (key, value, SEP)
panel. Training writes an experimental checkpoint and does NOT replace U16000.
TEST/FINAL/SACRED stay closed. Never loads P11 model_state_dict as Baby.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import BOS, EOS, LANG_TRAIN, read_u16
from .data_language_bridge import (
    banks_from_language,
    build_e2_panels,
    build_e3_panels,
    build_e4_panels,
    build_e6_panels,
    build_e7_panels,
    build_e8_panels,
    build_e9_panels,
    build_e10_panels,
    build_e11_panels,
    build_e12_panels,
    build_e13_panels,
    build_e14_panels,
    build_e15_panels,
    build_e16_panels,
    build_e17_panels,
    build_e18_panels,
    build_e19_panels,
    build_e20_panels,
    build_probe_panels,
    build_sentence_decode_pack,
    build_usable_chat_pack,
    load_tokenizer,
    make_english_item,
    encode_ids,
    period_token_id,
    sample_train_item,
    make_size_item,
    make_story_item,
    score_sentence_answer,
    score_usable_turn,
)
from .data_v2 import make_item
from .evaluate import language_ce, score_items, summarize
from .residual_overwrite import attach_overwrite
from .selection_p11 import GATE_BIAS
from .selection_p11_u16000_runtime import TREATMENT_CKPT, TREATMENT_SHA, resolve_device
from .selection_rapid_treat import (
    INDUCTION_DROP_BAR,
    OFF_INDUCTION_TOP1,
    POLICY_A1,
    _long_gap_items,
    measure_stage1,
    score_items_routed,
)
from .selection_rapid_treat_c import LOCATOR_TOKEN, LocatorOverwrite, LocatorRuntime, load_locator
from .selection_rapid_treat_d import (
    MASK_MATCHED,
    WRITE_HARD,
    WRITE_QUERY,
    DownstreamSession,
    tiling_parse,
)
from .selection_s1 import PARENT, PARENT_SHA, digest, write
from .train_v2r4 import DEV_STREAM, capability_optimizer, language_batch, set_seed

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "actual_baby"
LEDGER = OUT / "LEDGER.md"
E12_SURVIVOR = OUT / "e12_stoponly_311211" / "checkpoint_00050.pt"
E12_SURVIVOR_SHA = "6b000ffc4974244710c233d316b3bbbadcc45a4fa2f8f3cb5f4b561af72150d1"
USABLE_TURN_GATE = 0.70
USABLE_REUSE_GATE = 0.60
USABLE_STOP_GATE = 0.80
USABLE_RAMBLE_MAX = 0.30
SENTENCE_LIGHT_GATE = 0.40
PROTOCOL = "BABY_V010_LANGUAGE_BRIDGE"
EXPECTED_D3_SHA = "8d9e3e209f3449e4bdb5ad6bfeb3592dc1afe57bb10958bad3d7e1b94c40c9fb"
D3_PATH = ROOT / "src" / "baby_v010" / "selection_rapid_treat_d.py"
PROBE_N = 32
E1_HOLDOUT_GATE = 0.70
E1_CLOZE_GATE = 0.70
E2_QA4_GATE = 0.60
E2_INSTR_GATE = 0.60
E2_PRONOUN_GATE = 0.50
E2_DIALOGUE_GATE = 0.55
E3_TURN_GATE = 0.60
E4_SIZE_GATE = 0.60
E4_COLOR_KEEP = 0.70
E4_MIXED_GATE = 0.50
E4_DECODE_GATE = 0.70
E6_PLACE_GATE = 0.60
E7_FOLLOW_GATE = 0.55
E8_STORY_COLOR_GATE = 0.80
E8_STORY_WHO_GATE = 0.50
E8_STORY_EVENT_GATE = 0.50
E9_STORY_3E_GATE = 0.70
E9_STORY_PRONOUN_GATE = 0.50
E10_COMBINE_GATE = 0.50
E10_MIXED_STORY_GATE = 0.60
E11_LONG3_GATE = 0.55
E11_LONG4_GATE = 0.50
E12_SIZE_STOP_GATE = 0.60
E13_MIXED_GATE = 0.60
E13_COMBINE_GATE = 0.50
E13_MIXED_STORY_GATE = 0.50
E13_FACT_COMBINE_GATE = 0.50
E14_PHRASE_GATE = 0.45
E15_OPEN_GATE = 0.55
E16_ABOUT_GATE = 0.55
E17_COPY_GATE = 0.50
E17_NEW_GATE = 0.50
E17_FORMAT_SENT_GATE = 0.40
E18_CHAT_GATE = 0.55
E18_YESNO_GATE = 0.50
E18_LONG_GATE = 0.50
E19_HAPPENED_GATE = 0.50
E20_LOOP_GATE = 0.50
D3_LONG_MIN = 200
CANARY_UPDATES = 200
CANARY_EVAL = 50
LONG_UPDATES = 800
BATCH = 16
LANGUAGE_P = 0.20
STRUCTURED_P = 0.25
RETENTION_SLICE = 40


def ledger_append(block: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        LEDGER.write_text(
            "# Actual Baby campaign ledger\n\n"
            "U16000 remains authoritative. C2/D3 stay fallbacks. TEST closed. Not promoted.\n\n",
            encoding="utf-8",
        )
    lines = [
        f"## {block.get('id', 'unknown')}",
        "",
        f"- Change: {block.get('change', '')}",
        f"- verdict: **{block.get('verdict')}**",
        f"- lesson: {block.get('lesson', '')}",
    ]
    for key in ("native", "d3", "d3_tiling", "retention", "checkpoint_sha256", "update"):
        if key in block:
            lines.append(f"- {key}: {block[key]}")
    lines.append("")
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def install_c2_d3(model, overwrite) -> tuple:
    overwrite.set_eval_write(WRITE_HARD, locator=LOCATOR_TOKEN)
    handle, _ = attach_overwrite(model, overwrite)
    locator_rt = LocatorRuntime(model, overwrite).install()
    session = DownstreamSession(model, overwrite).install()
    session.configure(mask_mode=MASK_MATCHED, write_src=WRITE_QUERY, later_site=None)
    return handle, locator_rt, session


def uninstall_c2_d3(handle, locator_rt, session) -> None:
    session.remove()
    locator_rt.remove()
    handle.remove()


def load_overwrite_only(device) -> LocatorOverwrite:
    """P11 overwrite sidecar only. Never loads P11 Baby weights."""
    if digest(TREATMENT_CKPT) != TREATMENT_SHA:
        raise RuntimeError("P11 overwrite checkpoint hash mismatch")
    ckpt = torch.load(TREATMENT_CKPT, map_location="cpu", weights_only=False)
    if ckpt.get("parent_checkpoint_sha256") != PARENT_SHA:
        raise RuntimeError("overwrite checkpoint parent mismatch")
    if ckpt.get("protected_material_opened"):
        raise RuntimeError("P11 checkpoint opened protected material")
    overwrite = LocatorOverwrite(640, gate_bias=GATE_BIAS, gen_only=True).to(device)
    overwrite.load_state_dict(ckpt["overwrite_state_dict"])
    overwrite.eval()
    return overwrite


def tiling_rate(items: list[dict]) -> float:
    if not items:
        return 0.0
    hits = 0
    for item in items:
        gen = len(item["input"]) - 1
        hits += int(tiling_parse(item["input"], gen) is not None)
    return hits / len(items)


def score_arm(model, items: list[dict], device, overwrite, arm: str) -> dict:
    if not items:
        return {"n": 0}
    if arm == "native":
        rows = []
        for start in range(0, len(items), 16):
            rows.extend(score_items(model, items[start : start + 16], device, overwrite=overwrite, first_answer_only=False))
        summary = summarize(rows)
        summary["tiling_parse_rate"] = tiling_rate(items)
        return summary
    if arm == "d3":
        rows = []
        for start in range(0, len(items), 16):
            rows.extend(
                score_items_routed(
                    model,
                    items[start : start + 16],
                    device,
                    overwrite=overwrite,
                    first_answer_only=True,
                    policy=POLICY_A1,
                )
            )
        summary = summarize(rows)
        summary["tiling_parse_rate"] = tiling_rate(items)
        return summary
    yes, no, yes_i, no_i = [], [], [], []
    for i, item in enumerate(items):
        gen = len(item["input"]) - 1
        if tiling_parse(item["input"], gen) is not None:
            yes.append(item)
            yes_i.append(i)
        else:
            no.append(item)
            no_i.append(i)
    result: list[dict | None] = [None] * len(items)
    if yes:
        scored = score_items_routed(model, yes, device, overwrite=overwrite, first_answer_only=True, policy=POLICY_A1)
        for i, row in zip(yes_i, scored):
            result[i] = row
    if no:
        scored = score_items(model, no, device, overwrite=overwrite, first_answer_only=False)
        for i, row in zip(no_i, scored):
            result[i] = row
    summary = summarize(result)  # type: ignore[arg-type]
    summary["tiling_parse_rate"] = tiling_rate(items)
    summary["n_tiling_armed"] = len(yes)
    return summary


def slim_panels(report: dict) -> dict:
    return {name: {k: v for k, v in summary.items() if k != "rows"} for name, summary in report.items()}


@torch.no_grad()
def eval_panels(model, panels: dict[str, list[dict]], device, overwrite, arms: tuple[str, ...]) -> dict:
    model.eval()
    out: dict[str, dict] = {}
    for arm in arms:
        out[arm] = {}
        for name, items in panels.items():
            out[arm][name] = score_arm(model, items, device, overwrite, arm)
            print(
                json.dumps(
                    {
                        "phase": "probe_family",
                        "arm": arm,
                        "family": name,
                        "first_top1": out[arm][name].get("first_top1"),
                        "free_exact": out[arm][name].get("free_exact"),
                        "tiling": out[arm][name].get("tiling_parse_rate"),
                    }
                ),
                flush=True,
            )
    return out


def save_bridge_checkpoint(path: Path, model, optimizer, config, update: int, seed: int, phase: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "protocol": PROTOCOL,
            "lineage": "Baby v0.10 language-bridge experimental",
            "update": update,
            "seed": seed,
            "phase": phase,
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "parent_checkpoint": str(PARENT),
            "parent_checkpoint_sha256": PARENT_SHA,
            "protected_material_opened": False,
            "authoritative": False,
        },
        tmp,
    )
    os.replace(tmp, path)
    return digest(path)


def pack_bridge_batch(items: list[dict], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences = [[*item["input"], *item["target"]] for item in items]
    max_len = max(len(seq) for seq in sequences)
    x = torch.zeros((len(items), max_len - 1), dtype=torch.long, device=device)
    y = torch.zeros((len(items), max_len - 1), dtype=torch.long, device=device)
    mask = torch.zeros((len(items), max_len - 1), dtype=torch.bool, device=device)
    for row, item in enumerate(items):
        seq = sequences[row]
        x[row, : len(seq) - 1] = torch.tensor(seq[:-1], dtype=torch.long, device=device)
        y[row, : len(seq) - 1] = torch.tensor(seq[1:], dtype=torch.long, device=device)
        start = len(item["input"]) - 1
        train_len = len(item["target_span"])
        mask[row, start : start + train_len] = True
    return x, y, mask


def structured_retention_batch(banks, rng: random.Random, n: int, device):
    items = []
    for _ in range(n):
        kind = "induction" if rng.random() < 0.30 else "keyed"
        difficulty = rng.choice(("primitive", "short", "full"))
        items.append(make_item(rng, banks, difficulty=difficulty, kind=kind))
    return pack_bridge_batch(items, device)


def e1_gate(native: dict) -> tuple[str, str]:
    hold = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    cloze = float(native.get("cloze_2fact", {}).get("first_top1") or 0.0)
    if hold >= E1_HOLDOUT_GATE and cloze >= E1_CLOZE_GATE:
        return "GRAD", f"heldout QA {hold:.3f} and cloze2 {cloze:.3f} clear the E1 bind gate"
    if hold >= 0.45 or cloze >= 0.45:
        return "ADVANCE", f"signal but below gate: heldout {hold:.3f} cloze2 {cloze:.3f}"
    return "FAIL", f"no English bind yet: heldout {hold:.3f} cloze2 {cloze:.3f}"


def e3_gate(native: dict) -> tuple[str, str]:
    t2 = float(native.get("multiturn_2fact_heldout", {}).get("first_top1") or 0.0)
    t3 = float(native.get("multiturn_3fact_heldout", {}).get("first_top1") or 0.0)
    dialogue = float(native.get("dialogue_2fact_heldout", {}).get("first_top1") or 0.0)
    if t2 >= E3_TURN_GATE and t3 >= 0.50 and dialogue >= E2_DIALOGUE_GATE:
        return "GRAD", f"E3 turn2={t2:.3f} turn3={t3:.3f} dialogue={dialogue:.3f}"
    if max(t2, t3) >= 0.40:
        return "ADVANCE", f"E3 partial turn2={t2:.3f} turn3={t3:.3f} dialogue={dialogue:.3f}"
    return "FAIL", f"E3 cold turn2={t2:.3f} turn3={t3:.3f} dialogue={dialogue:.3f}"


def e2_gate(native: dict) -> tuple[str, str]:
    qa4 = float(native.get("qa_4fact_heldout", {}).get("first_top1") or 0.0)
    instr = float(native.get("instr_2fact_heldout", {}).get("first_top1") or 0.0)
    pronoun = float(native.get("pronoun_2fact_heldout", {}).get("first_top1") or 0.0)
    dialogue = float(native.get("dialogue_2fact_heldout", {}).get("first_top1") or 0.0)
    if qa4 >= E2_QA4_GATE and instr >= E2_INSTR_GATE and dialogue >= E2_DIALOGUE_GATE and pronoun >= E2_PRONOUN_GATE:
        return "GRAD", f"E2 qa4={qa4:.3f} instr={instr:.3f} pronoun={pronoun:.3f} dialogue={dialogue:.3f}"
    if max(qa4, instr, dialogue, pronoun) >= 0.40:
        return "ADVANCE", f"E2 partial qa4={qa4:.3f} instr={instr:.3f} pronoun={pronoun:.3f} dialogue={dialogue:.3f}"
    return "FAIL", f"E2 cold qa4={qa4:.3f} instr={instr:.3f} pronoun={pronoun:.3f} dialogue={dialogue:.3f}"


def e4_gate(native: dict) -> tuple[str, str]:
    size = float(native.get("size_2fact_heldout", {}).get("first_top1") or 0.0)
    color = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    mixed = float(native.get("mixed_2e_heldout", {}).get("first_top1") or 0.0)
    if size >= E4_SIZE_GATE and color >= E4_COLOR_KEEP and mixed >= E4_MIXED_GATE:
        return "GRAD", f"E4 size={size:.3f} color={color:.3f} mixed={mixed:.3f}"
    if size >= 0.40 or mixed >= 0.40:
        return "ADVANCE", f"E4 partial size={size:.3f} color={color:.3f} mixed={mixed:.3f}"
    return "FAIL", f"E4 cold size={size:.3f} color={color:.3f} mixed={mixed:.3f}"


def e6_gate(native: dict) -> tuple[str, str]:
    place = float(native.get("place_2fact_heldout", {}).get("first_top1") or 0.0)
    color = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    size = float(native.get("size_2fact_heldout", {}).get("first_top1") or 0.0)
    if place >= E6_PLACE_GATE and color >= E4_COLOR_KEEP and size >= E4_SIZE_GATE:
        return "GRAD", f"E6 place={place:.3f} color={color:.3f} size={size:.3f}"
    if place >= 0.40:
        return "ADVANCE", f"E6 partial place={place:.3f} color={color:.3f} size={size:.3f}"
    return "FAIL", f"E6 cold place={place:.3f} color={color:.3f} size={size:.3f}"


def e7_gate(native: dict) -> tuple[str, str]:
    follow = float(native.get("attr_followup_heldout", {}).get("first_top1") or 0.0)
    color = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    size = float(native.get("size_2fact_heldout", {}).get("first_top1") or 0.0)
    instr = float(native.get("instr_2fact_heldout", {}).get("first_top1") or 0.0)
    if follow >= E7_FOLLOW_GATE and color >= E4_COLOR_KEEP and size >= E4_SIZE_GATE and instr >= E2_INSTR_GATE:
        return "GRAD", f"E7 follow={follow:.3f} color={color:.3f} size={size:.3f} instr={instr:.3f}"
    if follow >= 0.40:
        return "ADVANCE", f"E7 partial follow={follow:.3f} color={color:.3f} size={size:.3f} instr={instr:.3f}"
    return "FAIL", f"E7 cold follow={follow:.3f} color={color:.3f} size={size:.3f} instr={instr:.3f}"


def e8_gate(native: dict) -> tuple[str, str]:
    story_c = float(native.get("story_color_heldout", {}).get("first_top1") or 0.0)
    story_who = float(native.get("story_who_heldout", {}).get("first_top1") or 0.0)
    story_ev = float(native.get("story_event_heldout", {}).get("first_top1") or 0.0)
    color = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    if (
        story_c >= E8_STORY_COLOR_GATE
        and story_who >= E8_STORY_WHO_GATE
        and story_ev >= E8_STORY_EVENT_GATE
        and color >= E4_COLOR_KEEP
    ):
        return "GRAD", f"E8 story_c={story_c:.3f} who={story_who:.3f} event={story_ev:.3f} color={color:.3f}"
    if max(story_c, story_who, story_ev) >= 0.40:
        return "ADVANCE", f"E8 partial story_c={story_c:.3f} who={story_who:.3f} event={story_ev:.3f} color={color:.3f}"
    return "FAIL", f"E8 cold story_c={story_c:.3f} who={story_who:.3f} event={story_ev:.3f} color={color:.3f}"


def e9_gate(native: dict) -> tuple[str, str]:
    story2 = float(native.get("story_color_heldout", {}).get("first_top1") or 0.0)
    story3 = float(native.get("story_color_3e_heldout", {}).get("first_top1") or 0.0)
    pronoun = float(native.get("story_pronoun_heldout", {}).get("first_top1") or 0.0)
    color = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    if story3 >= E9_STORY_3E_GATE and pronoun >= E9_STORY_PRONOUN_GATE and color >= E4_COLOR_KEEP and story2 >= E8_STORY_COLOR_GATE:
        return "GRAD", f"E9 3e={story3:.3f} pronoun={pronoun:.3f} story2={story2:.3f} color={color:.3f}"
    if max(story3, pronoun) >= 0.40:
        return "ADVANCE", f"E9 partial 3e={story3:.3f} pronoun={pronoun:.3f} story2={story2:.3f} color={color:.3f}"
    return "FAIL", f"E9 cold 3e={story3:.3f} pronoun={pronoun:.3f} story2={story2:.3f} color={color:.3f}"


def e10_gate(native: dict) -> tuple[str, str]:
    combine = float(native.get("story_combine_heldout", {}).get("first_top1") or 0.0)
    mixed_story = float(native.get("story_mixed_heldout", {}).get("first_top1") or 0.0)
    color = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    story2 = float(native.get("story_color_heldout", {}).get("first_top1") or 0.0)
    if combine >= E10_COMBINE_GATE and mixed_story >= E10_MIXED_STORY_GATE and color >= E4_COLOR_KEEP and story2 >= E8_STORY_COLOR_GATE:
        return "GRAD", f"E10 combine={combine:.3f} mixed_story={mixed_story:.3f} story2={story2:.3f} color={color:.3f}"
    if max(combine, mixed_story) >= 0.40:
        return "ADVANCE", f"E10 partial combine={combine:.3f} mixed_story={mixed_story:.3f} story2={story2:.3f} color={color:.3f}"
    return "FAIL", f"E10 cold combine={combine:.3f} mixed_story={mixed_story:.3f} story2={story2:.3f} color={color:.3f}"


def e11_gate(native: dict) -> tuple[str, str]:
    long3 = float(native.get("longturn_3_heldout", {}).get("first_top1") or 0.0)
    long4 = float(native.get("longturn_4_heldout", {}).get("first_top1") or 0.0)
    color = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    dialogue = float(native.get("dialogue_2fact_heldout", {}).get("first_top1") or 0.0)
    if long3 >= E11_LONG3_GATE and long4 >= E11_LONG4_GATE and color >= E4_COLOR_KEEP and dialogue >= E2_DIALOGUE_GATE:
        return "GRAD", f"E11 long3={long3:.3f} long4={long4:.3f} color={color:.3f} dialogue={dialogue:.3f}"
    if max(long3, long4) >= 0.40:
        return "ADVANCE", f"E11 partial long3={long3:.3f} long4={long4:.3f} color={color:.3f} dialogue={dialogue:.3f}"
    return "FAIL", f"E11 cold long3={long3:.3f} long4={long4:.3f} color={color:.3f} dialogue={dialogue:.3f}"


def e12_gate(native: dict) -> tuple[str, str]:
    stop = float(native.get("size_stop_heldout", {}).get("free_exact") or 0.0)
    size = float(native.get("size_2fact_heldout", {}).get("first_top1") or 0.0)
    color = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    story = float(native.get("story_color_heldout", {}).get("first_top1") or 0.0)
    if stop >= E12_SIZE_STOP_GATE and size >= E4_SIZE_GATE and color >= E4_COLOR_KEEP:
        extra = f" story={story:.3f}" if story else ""
        return "GRAD", f"E12 size_stop={stop:.3f} size={size:.3f} color={color:.3f}{extra}"
    if stop >= 0.40:
        return "ADVANCE", f"E12 partial size_stop={stop:.3f} size={size:.3f} color={color:.3f}"
    return "FAIL", f"E12 cold size_stop={stop:.3f} size={size:.3f} color={color:.3f}"


def e13_gate(native: dict) -> tuple[str, str]:
    mixed = float(native.get("mixed_2e_heldout", {}).get("first_top1") or 0.0)
    combine = float(native.get("story_combine_heldout", {}).get("first_top1") or 0.0)
    mixed_story = float(native.get("story_mixed_heldout", {}).get("first_top1") or 0.0)
    fact_combine = float(native.get("fact_combine_heldout", {}).get("first_top1") or 0.0)
    color = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    stop = float(native.get("size_stop_heldout", {}).get("free_exact") or 0.0)
    combine_ok = combine >= E13_COMBINE_GATE or fact_combine >= E13_FACT_COMBINE_GATE
    stop_ok = ("size_stop_heldout" not in native) or stop >= E12_SIZE_STOP_GATE
    if mixed >= E13_MIXED_GATE and combine_ok and mixed_story >= E13_MIXED_STORY_GATE and color >= E4_COLOR_KEEP and stop_ok:
        return (
            "GRAD",
            f"E13 mixed={mixed:.3f} combine={combine:.3f} fact_combine={fact_combine:.3f} mixed_story={mixed_story:.3f}",
        )
    if max(mixed, combine, mixed_story, fact_combine) >= 0.50:
        return (
            "ADVANCE",
            f"E13 partial mixed={mixed:.3f} combine={combine:.3f} fact_combine={fact_combine:.3f} mixed_story={mixed_story:.3f}",
        )
    return (
        "FAIL",
        f"E13 cold mixed={mixed:.3f} combine={combine:.3f} fact_combine={fact_combine:.3f} mixed_story={mixed_story:.3f}",
    )


def e14_gate(native: dict) -> tuple[str, str]:
    phrase = float(native.get("phrase_2fact_heldout", {}).get("free_exact") or 0.0)
    color = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    stop = float(native.get("size_stop_heldout", {}).get("free_exact") or 0.0)
    stop_ok = ("size_stop_heldout" not in native) or stop >= E12_SIZE_STOP_GATE
    if phrase >= E14_PHRASE_GATE and color >= E4_COLOR_KEEP and stop_ok:
        return "GRAD", f"E14 phrase={phrase:.3f} color={color:.3f} size_stop={stop:.3f}"
    if phrase >= 0.20:
        return "ADVANCE", f"E14 partial phrase={phrase:.3f} color={color:.3f} size_stop={stop:.3f}"
    return "FAIL", f"E14 cold phrase={phrase:.3f} color={color:.3f} size_stop={stop:.3f}"


def e15_gate(native: dict) -> tuple[str, str]:
    open_qa = float(native.get("open_2fact_heldout", {}).get("first_top1") or 0.0)
    color = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    stop = float(native.get("size_stop_heldout", {}).get("free_exact") or 0.0)
    dialogue = float(native.get("dialogue_2fact_heldout", {}).get("first_top1") or 0.0)
    stop_ok = ("size_stop_heldout" not in native) or stop >= E12_SIZE_STOP_GATE
    if open_qa >= E15_OPEN_GATE and color >= E4_COLOR_KEEP and stop_ok and dialogue >= E2_DIALOGUE_GATE:
        return "GRAD", f"E15 open={open_qa:.3f} color={color:.3f} dialogue={dialogue:.3f}"
    if open_qa >= 0.30:
        return "ADVANCE", f"E15 partial open={open_qa:.3f} color={color:.3f} dialogue={dialogue:.3f}"
    return "FAIL", f"E15 cold open={open_qa:.3f} color={color:.3f} dialogue={dialogue:.3f}"


def e16_gate(native: dict) -> tuple[str, str]:
    about = float(native.get("about_2fact_heldout", {}).get("first_top1") or 0.0)
    open_qa = float(native.get("open_2fact_heldout", {}).get("first_top1") or 0.0)
    color = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    stop = float(native.get("size_stop_heldout", {}).get("free_exact") or 0.0)
    stop_ok = ("size_stop_heldout" not in native) or stop >= E12_SIZE_STOP_GATE
    if about >= E16_ABOUT_GATE and open_qa >= E15_OPEN_GATE and color >= E4_COLOR_KEEP and stop_ok:
        return "GRAD", f"E16 about={about:.3f} open={open_qa:.3f} color={color:.3f}"
    if about >= 0.30:
        return "ADVANCE", f"E16 partial about={about:.3f} open={open_qa:.3f} color={color:.3f}"
    return "FAIL", f"E16 cold about={about:.3f} open={open_qa:.3f} color={color:.3f}"


def e17_gate(native: dict) -> tuple[str, str]:
    copy = float(native.get("copy_2fact_heldout", {}).get("first_top1") or 0.0)
    saystop = float(native.get("saystop_2fact_heldout", {}).get("first_top1") or 0.0)
    field = float(native.get("field_2e_heldout", {}).get("first_top1") or 0.0)
    nostory = float(native.get("nostory_heldout", {}).get("first_top1") or 0.0)
    fmt_word = float(native.get("format_word_heldout", {}).get("first_top1") or 0.0)
    fmt_sent = float(native.get("format_sent_heldout", {}).get("first_top1") or 0.0)
    color = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    stop = float(native.get("size_stop_heldout", {}).get("free_exact") or 0.0)
    stop_ok = ("size_stop_heldout" not in native) or stop >= E12_SIZE_STOP_GATE
    keep_ok = color >= E4_COLOR_KEEP and stop_ok
    hard = copy >= E17_COPY_GATE or field >= E17_NEW_GATE or fmt_sent >= E17_FORMAT_SENT_GATE
    easy = saystop >= E17_NEW_GATE or nostory >= E17_NEW_GATE or fmt_word >= E2_INSTR_GATE
    if keep_ok and hard and easy:
        return (
            "GRAD",
            f"E17 copy={copy:.3f} saystop={saystop:.3f} field={field:.3f} nostory={nostory:.3f} sent={fmt_sent:.3f}",
        )
    if max(copy, saystop, field, nostory, fmt_sent, fmt_word) >= 0.30:
        return (
            "ADVANCE",
            f"E17 partial copy={copy:.3f} saystop={saystop:.3f} field={field:.3f} nostory={nostory:.3f} sent={fmt_sent:.3f}",
        )
    return (
        "FAIL",
        f"E17 cold copy={copy:.3f} saystop={saystop:.3f} field={field:.3f} nostory={nostory:.3f} sent={fmt_sent:.3f}",
    )


def e18_gate(native: dict) -> tuple[str, str]:
    chat = float(native.get("chat_open_heldout", {}).get("first_top1") or 0.0)
    role = float(native.get("chat_role_heldout", {}).get("first_top1") or 0.0)
    reuse = float(native.get("chat_reuse_heldout", {}).get("first_top1") or 0.0)
    long3 = float(native.get("chat_long3_heldout", {}).get("first_top1") or 0.0)
    color = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    stop = float(native.get("size_stop_heldout", {}).get("free_exact") or 0.0)
    stop_ok = ("size_stop_heldout" not in native) or stop >= E12_SIZE_STOP_GATE
    if (
        chat >= E18_CHAT_GATE
        and max(role, long3) >= E18_LONG_GATE
        and reuse >= E18_LONG_GATE
        and color >= E4_COLOR_KEEP
        and stop_ok
    ):
        return "GRAD", f"E18 chat={chat:.3f} role={role:.3f} reuse={reuse:.3f} long3={long3:.3f}"
    if max(chat, role, reuse, long3) >= 0.30:
        return "ADVANCE", f"E18 partial chat={chat:.3f} role={role:.3f} reuse={reuse:.3f} long3={long3:.3f}"
    return "FAIL", f"E18 cold chat={chat:.3f} role={role:.3f} reuse={reuse:.3f} long3={long3:.3f}"


def e19_gate(native: dict) -> tuple[str, str]:
    happened = float(native.get("happened_heldout", {}).get("first_top1") or 0.0)
    yesno = float(native.get("yesno_2fact_heldout", {}).get("first_top1") or 0.0)
    color = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    stop = float(native.get("size_stop_heldout", {}).get("free_exact") or 0.0)
    stop_ok = ("size_stop_heldout" not in native) or stop >= E12_SIZE_STOP_GATE
    if happened >= E19_HAPPENED_GATE and yesno >= E18_YESNO_GATE and color >= E4_COLOR_KEEP and stop_ok:
        return "GRAD", f"E19 happened={happened:.3f} yesno={yesno:.3f} color={color:.3f}"
    if max(happened, yesno) >= 0.30:
        return "ADVANCE", f"E19 partial happened={happened:.3f} yesno={yesno:.3f} color={color:.3f}"
    return "FAIL", f"E19 cold happened={happened:.3f} yesno={yesno:.3f} color={color:.3f}"


def e20_gate(native: dict) -> tuple[str, str]:
    loop_c = float(native.get("chat_loop_color_heldout", {}).get("first_top1") or 0.0)
    loop_f = float(native.get("chat_loop_fact_heldout", {}).get("first_top1") or 0.0)
    chat = float(native.get("chat_open_heldout", {}).get("first_top1") or 0.0)
    color = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0)
    stop = float(native.get("size_stop_heldout", {}).get("free_exact") or 0.0)
    stop_ok = ("size_stop_heldout" not in native) or stop >= E12_SIZE_STOP_GATE
    if loop_c >= E20_LOOP_GATE and chat >= E18_CHAT_GATE and color >= E4_COLOR_KEEP and stop_ok:
        extra = f" loop_fact={loop_f:.3f}" if loop_f else ""
        return "GRAD", f"E20 loop_color={loop_c:.3f}{extra} color={color:.3f}"
    if max(loop_c, loop_f, chat) >= 0.30:
        return "ADVANCE", f"E20 partial loop_color={loop_c:.3f} loop_fact={loop_f:.3f} color={color:.3f}"
    return "FAIL", f"E20 cold loop_color={loop_c:.3f} loop_fact={loop_f:.3f} color={color:.3f}"


def first_word_match(decoded: str, gold: str) -> bool:
    text = decoded.strip().lower()
    want = gold.strip().lower()
    if not text:
        return False
    if text == want or text.startswith(want):
        return True
    word = text.split()[0].strip(".,!?;:")
    return word == want


@torch.no_grad()
def greedy_decode_until_stop(model, input_ids: list[int], device, tokenizer, max_new: int = 12) -> tuple[list[int], bool]:
    stop = {period_token_id(tokenizer), int(EOS)}
    generated = list(input_ids)
    if not generated or generated[0] != BOS:
        generated = [BOS, *generated]
    emitted: list[int] = []
    stopped = False
    model.eval()
    for _ in range(max_new):
        tokens = torch.tensor([generated[-256:]], dtype=torch.long, device=device)
        token = int(model(tokens)[0, -1].argmax().item())
        emitted.append(token)
        generated.append(token)
        if token in stop:
            stopped = True
            break
    return emitted, stopped


def run_decode_canary(model, tokenizer, device, *, n: int = 8, seed: int = 310511) -> dict:
    rng = random.Random(seed)
    rows = []
    makers = (
        lambda: make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout"),
        lambda: make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="heldout"),
        lambda: make_story_item(rng, tokenizer, surface="heldout", ask="color"),
        lambda: make_story_item(rng, tokenizer, surface="heldout", ask="who"),
    )
    for i in range(n):
        item = makers[i % len(makers)]()
        emitted, stopped = greedy_decode_until_stop(model, item["input"], device, tokenizer)
        decoded = tokenizer.decode(emitted, skip_special_tokens=True)
        gold = str(item["value_text"])
        match = first_word_match(decoded, gold)
        rows.append(
            {
                "prompt": item["prompt_text"],
                "gold": gold,
                "decoded": decoded,
                "stopped": stopped,
                "first_word_ok": match,
                "n_tokens": len(emitted),
                "attr": item.get("attr"),
            }
        )
    stories = []
    for prompt in ("Once upon a time there was a cat.", "The little hen sat down."):
        prompt_ids = [BOS, *encode_ids(tokenizer, prompt)]
        emitted, stopped = greedy_decode_until_stop(model, prompt_ids, device, tokenizer, max_new=16)
        stories.append(
            {
                "prompt": prompt,
                "decoded": tokenizer.decode(emitted, skip_special_tokens=True),
                "stopped": stopped,
            }
        )
    n_ok = sum(int(row["first_word_ok"]) for row in rows)
    n_stop = sum(int(row["stopped"]) for row in rows)
    n_both = sum(int(row["first_word_ok"] and row["stopped"]) for row in rows)
    return {
        "n": n,
        "first_word_match": n_ok / n,
        "stopped_at_period_or_eos": n_stop / n,
        "usable": n_both / n,
        "rows": rows,
        "stories": stories,
    }


def _rate(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return sum(int(flag) for flag in flags) / len(flags)


def _gold_answer(gold: str) -> str:
    text = gold if gold.startswith(" ") else " " + gold
    return text if text.endswith(".") else text + "."


def play_usable_chat(model, tokenizer, device, script: dict, *, teacher_force: bool) -> dict:
    facts = str(script["facts"])
    transcript = facts
    turn_rows = []
    ramble_gap = False
    for index, turn in enumerate(script["turns"]):
        prompt = f"{transcript}\nHuman: {turn['human']}\nBaby:"
        prompt_ids = [BOS, *encode_ids(tokenizer, prompt)]
        emitted, stopped = greedy_decode_until_stop(model, prompt_ids, device, tokenizer, max_new=16)
        decoded = tokenizer.decode(emitted, skip_special_tokens=True)
        scored = score_usable_turn(
            decoded=decoded,
            stopped=stopped,
            gold=str(turn["gold"]),
            distractors=tuple(turn["distractors"]),
            skills=list(turn["skills"]),
            n_tokens=len(emitted),
        )
        turn_rows.append(
            {
                "i": index,
                "human": turn["human"],
                "gold": turn["gold"],
                "entity": turn["entity"],
                "skills": turn["skills"],
                "prompt": prompt,
                "decoded": decoded,
                "stopped": stopped,
                "n_tokens": len(emitted),
                **scored,
            }
        )
        if scored["rambling"] or scored["generic_continuation"]:
            ramble_gap = True
        baby = _gold_answer(str(turn["gold"])) if teacher_force else (decoded if decoded else _gold_answer(str(turn["gold"])))
        transcript = prompt + baby
    reuse_flags = [bool(row["fact_reuse"]) for row in turn_rows if row["fact_reuse"] is not None]
    return {
        "id": script["id"],
        "n_turns": int(script["n_turns"]),
        "teacher_force": teacher_force,
        "turns": turn_rows,
        "usable_turn": _rate([row["usable"] for row in turn_rows]),
        "fact_hit": _rate([row["fact_hit"] for row in turn_rows]),
        "on_topic": _rate([row["on_topic"] for row in turn_rows]),
        "period_stop": _rate([row["period_stop"] for row in turn_rows]),
        "rambling": _rate([row["rambling"] for row in turn_rows]),
        "generic_continuation": _rate([row["generic_continuation"] for row in turn_rows]),
        "fact_reuse": _rate(reuse_flags) if reuse_flags else None,
        "chat_usable": all(row["usable"] for row in turn_rows),
        "ramble_gap": ramble_gap,
    }


def _slice_summary(rows: list[dict]) -> dict:
    reuse = [float(row["fact_reuse"]) for row in rows if row.get("fact_reuse") is not None]
    n_turns = max(1, sum(int(row["n_turns"]) for row in rows))
    return {
        "n_chats": len(rows),
        "n_turns": sum(int(row["n_turns"]) for row in rows),
        "usable_turn": sum(row["usable_turn"] * row["n_turns"] for row in rows) / n_turns,
        "fact_hit": sum(row["fact_hit"] * row["n_turns"] for row in rows) / n_turns,
        "on_topic": sum(row["on_topic"] * row["n_turns"] for row in rows) / n_turns,
        "period_stop": sum(row["period_stop"] * row["n_turns"] for row in rows) / n_turns,
        "rambling": sum(row["rambling"] * row["n_turns"] for row in rows) / n_turns,
        "generic_continuation": sum(row["generic_continuation"] * row["n_turns"] for row in rows) / n_turns,
        "fact_reuse": (sum(reuse) / len(reuse)) if reuse else None,
        "chat_usable": _rate([bool(row["chat_usable"]) for row in rows]),
        "ramble_gap_chats": _rate([bool(row["ramble_gap"]) for row in rows]),
    }


def usable_chat_verdict(auto_all: dict, auto_4: dict, auto_5: dict) -> tuple[str, str, bool]:
    """STRONG / MIXED / WEAK. MIXED = 4-turn strong, 5-turn weak (do not train 5-turn)."""
    four_strong = (
        float(auto_4["usable_turn"]) >= USABLE_TURN_GATE
        and float(auto_4.get("fact_reuse") or 0.0) >= USABLE_REUSE_GATE
        and float(auto_4["period_stop"]) >= USABLE_STOP_GATE
        and float(auto_4["rambling"]) <= USABLE_RAMBLE_MAX
    )
    five_strong = (
        float(auto_5["usable_turn"]) >= USABLE_TURN_GATE
        and float(auto_5.get("fact_reuse") or 0.0) >= USABLE_REUSE_GATE
        and float(auto_5["period_stop"]) >= USABLE_STOP_GATE
        and float(auto_5["rambling"]) <= USABLE_RAMBLE_MAX
    )
    ramble_is_gap = float(auto_4["rambling"]) > USABLE_RAMBLE_MAX or float(auto_4["generic_continuation"]) > 0.25
    wrong_fact = float(auto_4["fact_hit"]) < 0.60 and float(auto_4["rambling"]) <= USABLE_RAMBLE_MAX
    if four_strong and five_strong:
        return "STRONG", f"usable-chat 4+5 usable={auto_all['usable_turn']:.3f} reuse={auto_all.get('fact_reuse')} stop={auto_all['period_stop']:.3f}", False
    if four_strong and not five_strong:
        return (
            "MIXED",
            f"4-turn usable={auto_4['usable_turn']:.3f} reuse={auto_4.get('fact_reuse')} stop={auto_4['period_stop']:.3f}; 5-turn usable={auto_5['usable_turn']:.3f} parked",
            False,
        )
    gap = "rambling/generic continuation" if ramble_is_gap else ("wrong-fact" if wrong_fact else "mixed skill drop")
    return "WEAK", f"4-turn usable={auto_4['usable_turn']:.3f} stop={auto_4['period_stop']:.3f} ramble={auto_4['rambling']:.3f} gap={gap}", ramble_is_gap


def run_usable_chat(model, tokenizer, device) -> dict:
    pack = build_usable_chat_pack()
    auto_rows = [play_usable_chat(model, tokenizer, device, script, teacher_force=False) for script in pack]
    gold_rows = [play_usable_chat(model, tokenizer, device, script, teacher_force=True) for script in pack]
    auto_4 = [row for row in auto_rows if row["n_turns"] == 4]
    auto_5 = [row for row in auto_rows if row["n_turns"] == 5]
    summary_auto = _slice_summary(auto_rows)
    summary_4 = _slice_summary(auto_4)
    summary_5 = _slice_summary(auto_5)
    summary_gold = _slice_summary(gold_rows)
    verdict, lesson, ramble_gap = usable_chat_verdict(summary_auto, summary_4, summary_5)
    return {
        "id": "usable_chat",
        "n": len(pack),
        "autoregressive": summary_auto,
        "autoregressive_4turn": summary_4,
        "autoregressive_5turn": summary_5,
        "teacher_force": summary_gold,
        "chats_auto": auto_rows,
        "chats_teacher": gold_rows,
        "verdict": verdict,
        "lesson": lesson,
        "ramble_gap": ramble_gap,
    }


SENTENCE_OPERATORS = (
    ("bare", "{facts} {query}", ""),
    ("sent_prefix", "Please answer in a sentence. {facts} {query}", ""),
    ("short_prefix", "Give a short sentence. {facts} {query}", ""),
    ("baby", "{facts}\nHuman: {query}\nBaby:", ""),
    ("force_the", "{facts} {query} The", " The"),
    ("force_entity_is", "{facts} {query} {entity} is", None),
    ("force_the_entity_is", "{facts} {query} The {entity} is", None),
)
LIGHT_SENTENCE_OPS = frozenset({"bare", "sent_prefix", "short_prefix", "baby", "force_the"})


def run_sentence_decode(model, tokenizer, device) -> dict:
    pack = build_sentence_decode_pack()
    by_op: dict[str, list[dict]] = {name: [] for name, _template, _lead in SENTENCE_OPERATORS}
    for item in pack:
        for name, template, lead in SENTENCE_OPERATORS:
            prompt = template.format(facts=item["facts"], query=item["query"], entity=item["entity"])
            if name == "force_entity_is":
                lead_text = f" {item['entity']} is"
            elif name == "force_the_entity_is":
                lead_text = f" The {item['entity']} is"
            else:
                lead_text = lead
            prompt_ids = [BOS, *encode_ids(tokenizer, prompt)]
            emitted, stopped = greedy_decode_until_stop(model, prompt_ids, device, tokenizer, max_new=16)
            decoded = tokenizer.decode(emitted, skip_special_tokens=True)
            full = (lead_text or "") + decoded
            scored = score_sentence_answer(full_text=full, entity=item["entity"], color=item["color"], stopped=stopped)
            by_op[name].append(
                {
                    "id": item["id"],
                    "prompt": prompt,
                    "decoded": decoded,
                    "full": full,
                    "gold": item["color"],
                    "entity": item["entity"],
                    "stopped": stopped,
                    "n_tokens": len(emitted),
                    **scored,
                }
            )
    operators = {}
    for name, rows in by_op.items():
        operators[name] = {
            "sentence_ok": _rate([bool(row["sentence_ok"]) for row in rows]),
            "one_word_color": _rate([bool(row["one_word_color"]) for row in rows]),
            "has_color": _rate([bool(row["has_color"]) for row in rows]),
            "has_entity": _rate([bool(row["has_entity"]) for row in rows]),
            "period": _rate([bool(row["period"]) for row in rows]),
            "stopped": _rate([bool(row["stopped"]) for row in rows]),
            "light": name in LIGHT_SENTENCE_OPS,
            "rows": rows,
        }
    light_best = max((name for name in operators if operators[name]["light"]), key=lambda name: operators[name]["sentence_ok"])
    light_rate = float(operators[light_best]["sentence_ok"])
    keep = light_rate >= SENTENCE_LIGHT_GATE
    heavy_best = max((name for name in operators if not operators[name]["light"]), key=lambda name: operators[name]["sentence_ok"])
    if keep:
        lesson = f"keep operator {light_best} sentence_ok={light_rate:.3f}"
        verdict = "KEEP"
    else:
        lesson = (
            f"park sentence decode; best light {light_best}={light_rate:.3f}; "
            f"heavy {heavy_best}={operators[heavy_best]['sentence_ok']:.3f}"
        )
        verdict = "PARK"
    return {
        "id": "sentence_decode",
        "verdict": verdict,
        "lesson": lesson,
        "keep_operator": light_best if keep else None,
        "operators": operators,
    }


def load_e12_survivor(device, resume: Path | None = None):
    path = resume or E12_SURVIVOR
    sha = digest(path)
    if sha != E12_SURVIVOR_SHA:
        raise RuntimeError(f"E12 survivor hash mismatch: {sha}")
    model, config, ckpt = load_experimental_baby(path, device)
    return path, sha, model, config, ckpt


def run_canary(device, resume: Path | None = None) -> dict:
    tokenizer = load_tokenizer()
    path, sha, model, _config, ckpt = load_e12_survivor(device, resume)
    t0 = time.time()
    usable = run_usable_chat(model, tokenizer, device)
    sentence = run_sentence_decode(model, tokenizer, device)
    report = {
        "id": "usable_chat_canary",
        "resume": str(path),
        "checkpoint_sha256": sha,
        "authoritative": False,
        "protected_material_opened": bool(ckpt.get("protected_material_opened")),
        "elapsed_s": time.time() - t0,
        "usable_chat": {k: v for k, v in usable.items() if k not in {"chats_auto", "chats_teacher"}},
        "usable_chat_chats_auto": usable["chats_auto"],
        "usable_chat_chats_teacher": usable["chats_teacher"],
        "sentence_decode": {k: v for k, v in sentence.items() if k != "operators"},
        "sentence_operators": {
            name: {kk: vv for kk, vv in row.items() if kk != "rows"}
            for name, row in sentence["operators"].items()
        },
        "sentence_rows": {name: row["rows"] for name, row in sentence["operators"].items()},
        "verdict": usable["verdict"],
        "lesson": f"{usable['lesson']}; sentence {sentence['verdict']}: {sentence['lesson']}",
        "ramble_gap": usable["ramble_gap"],
        "sentence_keep_operator": sentence.get("keep_operator"),
    }
    write(OUT / "usable_chat.json", report)
    ledger_append(
        {
            "id": "usable_chat",
            "change": "Eval-only held-out 4-5 turn usable-chat pack + no-train sentence decode on E12 (U16000 not replaced)",
            "verdict": usable["verdict"],
            "lesson": report["lesson"],
            "checkpoint_sha256": sha,
        }
    )
    print(
        json.dumps(
            {
                "phase": "usable_chat",
                "verdict": usable["verdict"],
                "lesson": report["lesson"],
                "auto": usable["autoregressive"],
                "auto4": usable["autoregressive_4turn"],
                "auto5": usable["autoregressive_5turn"],
                "sentence": report["sentence_decode"],
            },
            default=str,
        ),
        flush=True,
    )
    return report


def retention_ok(stage1: dict) -> tuple[bool, str]:
    long_on = int(stage1["long_gap"]["free_exact"])
    n_long = int(stage1["long_gap"]["n"])
    ind = float(stage1["primitive_induction"]["first_top1"])
    if long_on < D3_LONG_MIN:
        return False, f"D3 long-gap {long_on}/{n_long} below {D3_LONG_MIN}"
    if ind + 1e-12 < (OFF_INDUCTION_TOP1 - INDUCTION_DROP_BAR):
        return False, f"induction {ind:.3f} dropped vs 0.297"
    return True, f"D3 long-gap {long_on}/{n_long} induction {ind:.3f}"


def cheap_d3_retention(model, overwrite, device) -> dict:
    items = _long_gap_items()[:RETENTION_SLICE]
    summary = score_arm(model, items, device, overwrite, "d3")
    summary["slice"] = RETENTION_SLICE
    return summary


def run_zeroshot(device, *, include_e2: bool = True) -> dict:
    if digest(D3_PATH) != EXPECTED_D3_SHA:
        raise RuntimeError(f"D3 sha mismatch: {digest(D3_PATH)}")
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("U16000 hash mismatch")
    tokenizer = load_tokenizer()
    banks = banks_from_language()
    panels = build_probe_panels(banks, tokenizer, n=PROBE_N)
    if include_e2:
        panels.update(build_e2_panels(tokenizer, n=PROBE_N))
    model, overwrite, _ckpt = load_locator(device)
    _handle, _rt, _session = install_c2_d3(model, overwrite)
    t0 = time.time()
    scored = eval_panels(model, panels, device, overwrite, ("native", "d3", "d3_tiling"))
    report = {
        "id": "e1_zeroshot",
        "protocol": PROTOCOL,
        "u16000_sha256": PARENT_SHA,
        "d3_sha256": EXPECTED_D3_SHA,
        "protected_material_opened": False,
        "authoritative_replaced": False,
        "n": PROBE_N,
        "elapsed_s": time.time() - t0,
        "native": slim_panels(scored["native"]),
        "d3": slim_panels(scored["d3"]),
        "d3_tiling": slim_panels(scored["d3_tiling"]),
    }
    verdict, lesson = e1_gate(report["native"])
    report["verdict"] = verdict
    report["lesson"] = lesson
    report["e2"] = dict(zip(("verdict", "lesson"), e2_gate(report["native"])))
    write(OUT / "e1_zeroshot.json", report)
    ledger_append(
        {
            "id": "e1_zeroshot",
            "change": "U16000 native vs C2+D3 vs tiling-gated D3 on less-rigid bind probes",
            "verdict": verdict,
            "lesson": lesson,
            "native": {
                name: {"first_top1": row.get("first_top1"), "free_exact": row.get("free_exact"), "tiling": row.get("tiling_parse_rate")}
                for name, row in report["native"].items()
            },
            "d3": {
                name: {"first_top1": row.get("first_top1"), "free_exact": row.get("free_exact")}
                for name, row in report["d3"].items()
            },
        }
    )
    print(json.dumps({"phase": "zeroshot", "verdict": verdict, "lesson": lesson, "native": report["native"]}, default=str), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def load_experimental_baby(path: Path, device):
    from .config import BabyVNextConfig
    from .model import BabyVNextLM

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("protected_material_opened"):
        raise RuntimeError("checkpoint opened protected material")
    if ckpt.get("protocol") not in {PROTOCOL, "BABY_V010_FOUNDATION_V2R4"}:
        raise RuntimeError(f"unexpected protocol {ckpt.get('protocol')}")
    if "overwrite_state_dict" in ckpt and ckpt.get("protocol") != PROTOCOL:
        raise RuntimeError("refusing P11 overwrite checkpoint as Baby")
    config = BabyVNextConfig.from_dict(ckpt["config"])
    model = BabyVNextLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config, ckpt


def train_phase(
    device,
    *,
    phase: str,
    updates: int,
    eval_every: int,
    seed: int,
    out_dir: Path,
    eval_panels_map: dict[str, list[dict]],
    start_model=None,
    start_checkpoint: Path | None = None,
    language_p: float = LANGUAGE_P,
    structured_p: float = STRUCTURED_P,
    early_stop: bool = True,
) -> dict:
    set_seed(seed)
    rng = random.Random(seed)
    tokenizer = load_tokenizer()
    banks = banks_from_language()
    train_stream = torch.tensor(read_u16(LANG_TRAIN), dtype=torch.long)
    dev_stream = torch.tensor(read_u16(DEV_STREAM), dtype=torch.long)
    if start_checkpoint is not None:
        model, config, _ckpt = load_experimental_baby(start_checkpoint, device)
        print(json.dumps({"phase": "resume", "checkpoint": str(start_checkpoint), "sha256": digest(start_checkpoint)}), flush=True)
    elif start_model is None:
        from .config import BabyVNextConfig
        from .model import BabyVNextLM

        if digest(PARENT) != PARENT_SHA:
            raise RuntimeError("U16000 hash mismatch")
        ckpt = torch.load(PARENT, map_location="cpu", weights_only=False)
        if int(ckpt["update"]) != 16000 or ckpt.get("protected_material_opened"):
            raise RuntimeError("parent is not clean U16000")
        if "overwrite_state_dict" in ckpt:
            raise RuntimeError("refusing a checkpoint that carries overwrite_state_dict as Baby")
        config = BabyVNextConfig.from_dict(ckpt["config"])
        model = BabyVNextLM(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = start_model
        from .config import BabyVNextConfig

        config = BabyVNextConfig.load(ROOT / "configs" / "foundation_v1.json")
    optimizer, _low, _high = capability_optimizer(model)
    out_dir.mkdir(parents=True, exist_ok=True)
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
            items = [sample_train_item(rng, banks, tokenizer, phase=phase) for _ in range(BATCH)]
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
                        "phase": phase,
                        "task": task,
                        "loss": float(loss.detach().cpu()),
                        "grad_norm": float(grad_norm.detach().cpu()),
                    }
                ),
                flush=True,
            )
        if update % eval_every == 0 or update == updates:
            model.eval()
            native = eval_panels(model, eval_panels_map, device, overwrite=None, arms=("native",))["native"]
            ce = language_ce(model, dev_stream, list(range(0, min(64 * 256, dev_stream.numel() - 257), 256)), device, limit=16)
            ckpt_path = out_dir / f"checkpoint_{update:05d}.pt"
            sha = save_bridge_checkpoint(ckpt_path, model, optimizer, config, update, seed, phase)
            row = {
                "update": update,
                "native": slim_panels(native),
                "language_dev_ce": ce,
                "language_baseline_ce": baseline_ce,
                "checkpoint": str(ckpt_path),
                "checkpoint_sha256": sha,
            }
            if phase == "e1":
                verdict, lesson = e1_gate(row["native"])
            elif phase == "e3":
                verdict, lesson = e3_gate(row["native"])
            elif phase == "e5":
                verdict, lesson = e3_gate(row["native"])
            elif phase == "e4":
                verdict, lesson = e4_gate(row["native"])
            elif phase == "e6":
                verdict, lesson = e6_gate(row["native"])
            elif phase == "e7":
                verdict, lesson = e7_gate(row["native"])
            elif phase == "e8":
                verdict, lesson = e8_gate(row["native"])
            elif phase == "e9":
                verdict, lesson = e9_gate(row["native"])
            elif phase == "e10":
                verdict, lesson = e10_gate(row["native"])
            elif phase == "e11":
                verdict, lesson = e11_gate(row["native"])
            elif phase == "e12":
                verdict, lesson = e12_gate(row["native"])
            elif phase.startswith("e13"):
                verdict, lesson = e13_gate(row["native"])
            elif phase.startswith("e14"):
                verdict, lesson = e14_gate(row["native"])
            elif phase.startswith("e15"):
                verdict, lesson = e15_gate(row["native"])
            elif phase.startswith("e16"):
                verdict, lesson = e16_gate(row["native"])
            elif phase.startswith("e17"):
                verdict, lesson = e17_gate(row["native"])
            elif phase.startswith("e18"):
                verdict, lesson = e18_gate(row["native"])
            elif phase.startswith("e19"):
                verdict, lesson = e19_gate(row["native"])
            elif phase.startswith("e20"):
                verdict, lesson = e20_gate(row["native"])
            else:
                verdict, lesson = e2_gate(row["native"])
            row["verdict"] = verdict
            row["lesson"] = lesson
            history.append(row)
            write(out_dir / f"eval_{update:05d}.json", row)
            print(json.dumps({"phase": f"{phase}_eval", "update": update, "verdict": verdict, "lesson": lesson, "ce": ce}, default=str), flush=True)
            score = float(native.get("qa_2fact_heldout", {}).get("first_top1") or 0.0) + 0.5 * float(
                native.get("cloze_2fact", native.get("qa_4fact_heldout", {})).get("first_top1") or 0.0
            )
            prev = -1.0
            if best is not None:
                prev = float((best.get("native") or {}).get("qa_2fact_heldout", {}).get("first_top1") or 0.0) + 0.5 * float(
                    (best.get("native") or {}).get("cloze_2fact", (best.get("native") or {}).get("qa_4fact_heldout", {})).get("first_top1") or 0.0
                )
            if best is None or score >= prev:
                best = row
            if early_stop and verdict == "GRAD":
                break
    return {"history": history, "best": best, "model": model, "config": config, "optimizer": optimizer}


def attach_d3_for_retention(model, device):
    overwrite = load_overwrite_only(device)
    hooks = install_c2_d3(model, overwrite)
    return overwrite, hooks


def run_retention(model, device, tag: str) -> dict:
    overwrite, hooks = attach_d3_for_retention(model, device)
    try:
        cheap = cheap_d3_retention(model, overwrite, device)
        print(json.dumps({"phase": "retention_slice", "tag": tag, "slice": cheap}, default=str), flush=True)
        if float(cheap.get("free_exact") or 0) < 0.80:
            report = {
                "id": f"{tag}_retention",
                "verdict": "KILL",
                "lesson": f"cheap D3 slice free_exact {cheap.get('free_exact')}",
                "slice": cheap,
            }
            write(OUT / f"{tag}_retention.json", report)
            ledger_append(report)
            return report
        stage1 = measure_stage1(model, overwrite, device, POLICY_A1)
        ok, lesson = retention_ok(stage1)
        report = {
            "id": f"{tag}_retention",
            "verdict": "PASS" if ok else "KILL",
            "lesson": lesson,
            "long_gap": {k: stage1["long_gap"].get(k) for k in ("n", "free_exact", "first_correct", "free_accuracy", "first_accuracy")},
            "primitive_induction": stage1["primitive_induction"],
            "primitive_keyed": stage1["primitive_keyed"],
            "slice": cheap,
        }
        write(OUT / f"{tag}_retention.json", {k: v for k, v in report.items()})
        ledger_append(
            {
                "id": report["id"],
                "change": "C2+D3 retention on original long-gap after language-bridge train",
                "verdict": report["verdict"],
                "lesson": lesson,
                "retention": report["long_gap"],
            }
        )
        print(json.dumps({"phase": "retention", **{k: v for k, v in report.items() if k != "slice"}}, default=str), flush=True)
        return report
    finally:
        uninstall_c2_d3(*hooks)
        overwrite.gen_index = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_campaign(device_name: str = "cuda") -> dict:
    device = resolve_device(device_name)
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"phase": "start", "device": str(device), "u16000": PARENT_SHA, "d3": EXPECTED_D3_SHA}), flush=True)
    zero = run_zeroshot(device, include_e2=True)
    tokenizer = load_tokenizer()
    banks = banks_from_language()
    e1_panels = {k: v for k, v in build_probe_panels(banks, tokenizer, n=PROBE_N).items() if k in {"cloze_1fact", "cloze_2fact", "qa_2fact_train", "qa_2fact_heldout", "syntax_wrap", "aperiodic"}}
    e2_panels = build_e2_panels(tokenizer, n=PROBE_N)
    campaign = {"zeroshot": {k: v for k, v in zero.items() if k != "model"}, "e1": None, "e2": None}

    need_e1_train = zero["verdict"] != "GRAD"
    model = None
    if need_e1_train:
        e1_dir = OUT / "e1_bridge_310201"
        trained = train_phase(
            device,
            phase="e1",
            updates=CANARY_UPDATES,
            eval_every=CANARY_EVAL,
            seed=310201,
            out_dir=e1_dir,
            eval_panels_map=e1_panels,
        )
        best = trained["best"]
        if best is None or best.get("verdict") == "FAIL":
            print(json.dumps({"phase": "e1_extend", "reason": "canary weak; continue to 800"}), flush=True)
            trained = train_phase(
                device,
                phase="e1",
                updates=LONG_UPDATES,
                eval_every=100,
                seed=310201,
                out_dir=OUT / "e1_bridge_310201_long",
                eval_panels_map=e1_panels,
                start_model=trained["model"],
            )
            best = trained["best"]
        model = trained["model"]
        ret = run_retention(model, device, "e1")
        campaign["e1"] = {"best": {k: v for k, v in (best or {}).items() if k != "model"}, "retention": ret}
        if (best or {}).get("verdict") != "GRAD":
            ledger_append(
                {
                    "id": "e1_stop",
                    "change": "E1 language-bridge did not hit held-out QA/cloze gates",
                    "verdict": (best or {}).get("verdict", "FAIL"),
                    "lesson": (best or {}).get("lesson", "no best row"),
                }
            )
            campaign["stop"] = "E1 did not graduate; still continuing a cheaper E2 mix only if ADVANCE"
            if (best or {}).get("verdict") != "ADVANCE":
                write(OUT / "CAMPAIGN.json", {k: v for k, v in campaign.items() if k != "model"})
                return campaign
    else:
        campaign["e1"] = {"best": "zeroshot_already_grad", "retention": "u16000_d3_unchanged"}

    e2_zero_verdict = zero.get("e2", {}).get("verdict")
    if e2_zero_verdict == "GRAD" and not need_e1_train:
        campaign["e2"] = {"best": "zeroshot_already_grad"}
        write(OUT / "CAMPAIGN.json", campaign)
        return campaign

    e2_dir = OUT / "e2_bridge_310301"
    trained = train_phase(
        device,
        phase="e2",
        updates=CANARY_UPDATES,
        eval_every=CANARY_EVAL,
        seed=310301,
        out_dir=e2_dir,
        eval_panels_map=e2_panels,
        start_model=model,
    )
    best = trained["best"]
    if best is None or best.get("verdict") in {"FAIL", "ADVANCE"}:
        trained = train_phase(
            device,
            phase="e2",
            updates=LONG_UPDATES,
            eval_every=100,
            seed=310301,
            out_dir=OUT / "e2_bridge_310301_long",
            eval_panels_map=e2_panels,
            start_model=trained["model"],
        )
        best = trained["best"]
    ret = run_retention(trained["model"], device, "e2")
    campaign["e2"] = {"best": {k: v for k, v in (best or {}).items() if k != "model"}, "retention": ret}
    write(OUT / "CAMPAIGN.json", {k: v for k, v in campaign.items() if k != "model"})
    ledger_append(
        {
            "id": "campaign_e2",
            "change": "E2 instruction/pronoun/dialogue/4-fact after E1",
            "verdict": (best or {}).get("verdict"),
            "lesson": (best or {}).get("lesson"),
            "checkpoint_sha256": (best or {}).get("checkpoint_sha256"),
        }
    )
    return campaign


def run_probe(device, resume: Path, *, phase: str = "e4") -> dict:
    tokenizer = load_tokenizer()
    model, _config, ckpt = load_experimental_baby(resume, device)
    sha = digest(resume)
    if phase == "e3":
        panels = build_e3_panels(tokenizer, n=PROBE_N)
        gate = e3_gate
    elif phase == "e2":
        panels = build_e2_panels(tokenizer, n=PROBE_N)
        gate = e2_gate
    elif phase == "e6":
        panels = build_e6_panels(tokenizer, n=PROBE_N)
        gate = e6_gate
    elif phase == "e7":
        panels = build_e7_panels(tokenizer, n=PROBE_N)
        gate = e7_gate
    elif phase == "e8":
        panels = build_e8_panels(tokenizer, n=PROBE_N)
        gate = e8_gate
    elif phase == "e9":
        panels = build_e9_panels(tokenizer, n=PROBE_N)
        gate = e9_gate
    elif phase == "e10":
        panels = build_e10_panels(tokenizer, n=PROBE_N)
        gate = e10_gate
    elif phase == "e11":
        panels = build_e11_panels(tokenizer, n=PROBE_N)
        gate = e11_gate
    elif phase == "e12":
        panels = build_e12_panels(tokenizer, n=PROBE_N)
        gate = e12_gate
    elif phase.startswith("e13"):
        panels = build_e13_panels(tokenizer, n=PROBE_N)
        gate = e13_gate
    elif phase.startswith("e14"):
        panels = build_e14_panels(tokenizer, n=PROBE_N)
        gate = e14_gate
    elif phase.startswith("e15"):
        panels = build_e15_panels(tokenizer, n=PROBE_N)
        gate = e15_gate
    elif phase.startswith("e16"):
        panels = build_e16_panels(tokenizer, n=PROBE_N)
        gate = e16_gate
    elif phase.startswith("e17"):
        panels = build_e17_panels(tokenizer, n=PROBE_N)
        gate = e17_gate
    elif phase.startswith("e18"):
        panels = build_e18_panels(tokenizer, n=PROBE_N)
        gate = e18_gate
    elif phase.startswith("e19"):
        panels = build_e19_panels(tokenizer, n=PROBE_N)
        gate = e19_gate
    elif phase.startswith("e20"):
        panels = build_e20_panels(tokenizer, n=PROBE_N)
        gate = e20_gate
    else:
        panels = build_e4_panels(tokenizer, n=PROBE_N)
        gate = e4_gate
    native = eval_panels(model, panels, device, overwrite=None, arms=("native",))["native"]
    decode = run_decode_canary(model, tokenizer, device)
    verdict, lesson = gate(native)
    decode_ok = float(decode["usable"]) >= E4_DECODE_GATE
    report = {
        "id": f"{phase}_probe",
        "resume": str(resume),
        "checkpoint_sha256": sha,
        "authoritative": False,
        "protected_material_opened": bool(ckpt.get("protected_material_opened")),
        "native": slim_panels(native),
        "decode": {k: v for k, v in decode.items() if k not in {"rows", "stories"}},
        "decode_rows": decode["rows"],
        "decode_stories": decode.get("stories", []),
        "verdict": verdict,
        "lesson": lesson,
        "decode_usable_ok": decode_ok,
    }
    write(OUT / f"{phase}_probe.json", report)
    ledger_append(
        {
            "id": report["id"],
            "change": f"Zero-shot {phase} probe on experimental checkpoint (U16000 not replaced)",
            "verdict": verdict,
            "lesson": f"{lesson}; decode usable={decode['usable']:.3f} stop={decode['stopped_at_period_or_eos']:.3f}",
            "checkpoint_sha256": sha,
        }
    )
    print(json.dumps({"phase": f"{phase}_probe", "verdict": verdict, "lesson": lesson, "decode": report["decode"], "native": report["native"]}, default=str), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("zeroshot", "campaign", "train", "retention", "probe", "canary"), default="campaign")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--phase", default="e1")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--updates", type=int, default=CANARY_UPDATES)
    parser.add_argument("--eval-every", type=int, default=CANARY_EVAL)
    parser.add_argument("--seed", type=int, default=310211)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--structured-p", type=float, default=STRUCTURED_P)
    parser.add_argument("--language-p", type=float, default=LANGUAGE_P)
    parser.add_argument("--no-early-stop", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    if args.mode == "canary":
        run_canary(device, args.resume)
        return
    if args.mode == "probe":
        if args.resume is None:
            raise SystemExit("--resume is required for probe")
        run_probe(device, args.resume, phase=args.phase)
        return
    if args.mode == "zeroshot":
        run_zeroshot(device, include_e2=True)
        return
    if args.mode == "retention":
        if args.resume is None:
            raise SystemExit("--resume is required for retention")
        model, _config, _ckpt = load_experimental_baby(args.resume, device)
        run_retention(model, device, args.phase)
        return
    if args.mode == "train":
        tokenizer = load_tokenizer()
        banks = banks_from_language()
        if args.phase == "e1":
            panels = {
                k: v
                for k, v in build_probe_panels(banks, tokenizer, n=PROBE_N).items()
                if k in {"cloze_1fact", "cloze_2fact", "qa_2fact_train", "qa_2fact_heldout", "qa_3fact_heldout", "syntax_wrap", "aperiodic"}
            }
        elif args.phase == "e3":
            panels = build_e3_panels(tokenizer, n=PROBE_N)
        elif args.phase == "e5":
            panels = build_e3_panels(tokenizer, n=PROBE_N)
        elif args.phase == "e4":
            panels = build_e4_panels(tokenizer, n=PROBE_N)
        elif args.phase == "e6":
            panels = build_e6_panels(tokenizer, n=PROBE_N)
        elif args.phase == "e7":
            panels = build_e7_panels(tokenizer, n=PROBE_N)
        elif args.phase == "e8":
            panels = build_e8_panels(tokenizer, n=PROBE_N)
        elif args.phase == "e9":
            panels = build_e9_panels(tokenizer, n=PROBE_N)
        elif args.phase == "e10":
            panels = build_e10_panels(tokenizer, n=PROBE_N)
        elif args.phase == "e11":
            panels = build_e11_panels(tokenizer, n=PROBE_N)
        elif args.phase == "e12":
            panels = build_e12_panels(tokenizer, n=PROBE_N)
        elif args.phase.startswith("e13"):
            panels = build_e13_panels(tokenizer, n=PROBE_N)
        elif args.phase.startswith("e14"):
            panels = build_e14_panels(tokenizer, n=PROBE_N)
        elif args.phase.startswith("e15"):
            panels = build_e15_panels(tokenizer, n=PROBE_N)
        elif args.phase.startswith("e16"):
            panels = build_e16_panels(tokenizer, n=PROBE_N)
        elif args.phase.startswith("e17"):
            panels = build_e17_panels(tokenizer, n=PROBE_N)
        elif args.phase.startswith("e18"):
            panels = build_e18_panels(tokenizer, n=PROBE_N)
        elif args.phase.startswith("e19"):
            panels = build_e19_panels(tokenizer, n=PROBE_N)
        elif args.phase.startswith("e20"):
            panels = build_e20_panels(tokenizer, n=PROBE_N)
        else:
            panels = build_e2_panels(tokenizer, n=PROBE_N)
        out_dir = args.out or (OUT / f"{args.phase}_bridge_{args.seed}")
        trained = train_phase(
            device,
            phase=args.phase,
            updates=args.updates,
            eval_every=args.eval_every,
            seed=args.seed,
            out_dir=out_dir,
            eval_panels_map=panels,
            start_checkpoint=args.resume,
            language_p=args.language_p,
            structured_p=args.structured_p,
            early_stop=not args.no_early_stop,
        )
        best = trained["best"]
        print(json.dumps({"phase": "train_done", "best": {k: v for k, v in (best or {}).items() if k != "native"}, "native": None if best is None else best.get("native")}, default=str), flush=True)
        if best and best.get("verdict") == "GRAD":
            run_retention(trained["model"], device, args.phase)
        return
    run_campaign(args.device)


if __name__ == "__main__":
    main()
