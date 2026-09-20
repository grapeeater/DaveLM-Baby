from __future__ import annotations

"""Stack2 s3: multi-entity composition + unprompted short sentences from s2m.

U16000 stays authoritative. TEST/FINAL/SACRED stay sealed. s2m is not promoted.
"""

import json
from pathlib import Path

from .data_language_bridge import (
    build_s3_panels,
    make_compose_sentence_item,
    make_dialogue_item,
    make_english_item,
    make_mixed_item,
    make_phrase_item,
    make_size_item,
    make_story_item,
    make_story_mixed_item,
    make_who_bind_item,
)
from .selection_language_bridge import (
    eval_panels,
    load_experimental_baby,
    run_sentence_decode,
    run_usable_chat,
    slim_panels,
)
from .selection_s1 import digest, write
from .selection_stack2 import (
    CAMPAIGN,
    DROP_BAR,
    OUT,
    S2A_COMBINE,
    campaign_update,
    cheap_d3_slice,
    english_native_holds,
    ledger_append,
    mix_scores,
    require_identities,
    run_d3_log,
    slim_usable,
    usable_holds_s2a,
    _drop_cuda,
    _exact,
    _top1,
)

S2M_SURVIVOR = OUT / "s2m_protect40_combine_323091" / "checkpoint_00025.pt"
S2M_SURVIVOR_SHA = "bd10f0ea8239fe09bd5bec933611ba92b27c20402b24474ad287331153240ff9"
S2M_D3_FULL = 197
S2M_D3_SLICE = 0.925
S2M_MIXED = 0.875
S2M_COMBINE = 0.71875
S2M_STORY_COMBINE = 0.5625
S2M_USABLE = 0.9285714285714286
S2M_USABLE4 = 0.96875
COMPOSE_UPDATES = 25
COMPOSE_EVAL = 25
COMPOSE_N = 32
D3_SLICE_KILL = 0.125
WHO_SIGNAL = 0.25
WHO_GRAD = 0.70
MIXED3_GRAD = 0.65
SENT_BARE_GRAD = 0.375
SENT_EXACT_GRAD = 0.50
SCAFFOLD_GAP = 0.25
COMPOSE_ORDER = ("s3a", "s3b", "s3c", "s3d", "s3e")
RECIPES_MIX = {
    "s3a": "compose_bind",
    "s3b": "compose_sent",
    "s3c": "compose_both",
    "s3d": "compose_sent_instr",
    "s3e": "compose_phrase",
}
LIGHT_SENTENCE = ("bare", "sent_prefix", "baby")


def mix_holds_s2m(native: dict) -> tuple[bool, str]:
    scores = mix_scores(native)
    mixed_ok = scores["mixed_2e"] + 1e-12 >= S2M_MIXED - DROP_BAR
    # s2m's extra combine (0.719 vs s2a's 0.656) may float; do not kill a real
    # who-bind gain that still holds the s2a combine survivor.
    combine_ok = scores["fact_combine"] + 1e-12 >= S2A_COMBINE
    lesson = f"mixed={scores['mixed_2e']:.3f} fact_combine={scores['fact_combine']:.3f} story_combine={scores['story_combine']:.3f}"
    if mixed_ok and combine_ok:
        return True, f"s2m mix hold {lesson}"
    return False, f"s2m mix drop {lesson}"


def usable_holds_s2m(usable: dict | None) -> tuple[bool, str]:
    ok, lesson = usable_holds_s2a(usable)
    if usable is None:
        return False, "usable-chat missing"
    auto = usable.get("autoregressive") or {}
    auto4 = usable.get("autoregressive_4turn") or auto
    turn = float(auto.get("usable_turn") or 0.0)
    turn4 = float(auto4.get("usable_turn") or 0.0)
    hold = turn + 1e-12 >= S2M_USABLE - DROP_BAR and turn4 + 1e-12 >= S2M_USABLE4 - DROP_BAR
    note = f"usable4={turn4:.3f} usable={turn:.3f}"
    if ok and hold:
        return True, f"s2m usable hold {note}"
    if not ok:
        return False, lesson
    return False, f"s2m usable drop {note}"


def sample_s3_item(rng, tokenizer, mix: str) -> dict:
    roll = rng.random()
    if mix == "compose_bind":
        if roll < 0.30:
            return make_who_bind_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.45:
            return make_who_bind_item(rng, tokenizer, n_entities=3, surface="train")
        if roll < 0.60:
            return make_mixed_item(rng, tokenizer, n_entities=3, surface="train", period=True)
        if roll < 0.72:
            return make_mixed_item(rng, tokenizer, n_entities=3, surface="train", combine=True, period=True)
        if roll < 0.84:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", period=True)
        if roll < 0.94:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", combine=True, period=True)
        return make_story_mixed_item(rng, tokenizer, surface="train", combine=True, period=True)
    if mix == "compose_bind_protect":
        # s3a taught who-bind (0→0.44) but stole 2e combine. Keep who, restore combine dose.
        if roll < 0.18:
            return make_who_bind_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.26:
            return make_who_bind_item(rng, tokenizer, n_entities=3, surface="train")
        if roll < 0.48:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", period=True)
        if roll < 0.70:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", combine=True, period=True)
        if roll < 0.80:
            return make_story_mixed_item(rng, tokenizer, surface="train", combine=True, period=True)
        if roll < 0.90:
            return make_mixed_item(rng, tokenizer, n_entities=3, surface="train", period=True)
        return make_mixed_item(rng, tokenizer, n_entities=3, surface="train", combine=True, period=True)
    if mix == "compose_lock":
        # Rehearse who-bind while restoring the s2m mix/combine diet.
        if roll < 0.12:
            return make_who_bind_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.18:
            return make_who_bind_item(rng, tokenizer, n_entities=3, surface="train")
        if roll < 0.28:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.38:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.46:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.54:
            return make_dialogue_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.78:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", period=True)
        if roll < 0.86:
            return make_story_mixed_item(rng, tokenizer, surface="train", combine=False, period=True)
        if roll < 0.96:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", combine=True, period=True)
        return make_story_mixed_item(rng, tokenizer, surface="train", combine=True, period=True)
    if mix == "compose_lock_story":
        # s3a who is real; story_combine was the ugly drop. Keep who, restore combine+story.
        if roll < 0.14:
            return make_who_bind_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.20:
            return make_who_bind_item(rng, tokenizer, n_entities=3, surface="train")
        if roll < 0.30:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.38:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.46:
            return make_story_item(rng, tokenizer, surface="train", ask="color")
        if roll < 0.66:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", period=True)
        if roll < 0.82:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", combine=True, period=True)
        if roll < 0.92:
            return make_story_mixed_item(rng, tokenizer, surface="train", combine=True, period=True)
        return make_story_mixed_item(rng, tokenizer, surface="train", combine=False, period=True)
    if mix == "compose_lock_who":
        if roll < 0.20:
            return make_who_bind_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.28:
            return make_who_bind_item(rng, tokenizer, n_entities=3, surface="train")
        if roll < 0.38:
            return make_english_item(rng, tokenizer, n_facts=2, family="qa", surface="train")
        if roll < 0.46:
            return make_size_item(rng, tokenizer, n_facts=2, family="qa", surface="train", period=True)
        if roll < 0.66:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", period=True)
        if roll < 0.82:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", combine=True, period=True)
        if roll < 0.92:
            return make_story_mixed_item(rng, tokenizer, surface="train", combine=True, period=True)
        return make_story_item(rng, tokenizer, surface="train", ask="color")
    if mix == "compose_sent":
        if roll < 0.40:
            return make_compose_sentence_item(rng, tokenizer, surface="train", combine=False)
        if roll < 0.58:
            return make_compose_sentence_item(rng, tokenizer, surface="train", combine=True)
        if roll < 0.70:
            return make_who_bind_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.84:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", period=True)
        if roll < 0.94:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", combine=True, period=True)
        return make_story_mixed_item(rng, tokenizer, surface="train", combine=False, period=True)
    if mix == "compose_both":
        if roll < 0.22:
            return make_who_bind_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.32:
            return make_who_bind_item(rng, tokenizer, n_entities=3, surface="train")
        if roll < 0.48:
            return make_compose_sentence_item(rng, tokenizer, surface="train", combine=False)
        if roll < 0.58:
            return make_compose_sentence_item(rng, tokenizer, surface="train", combine=True)
        if roll < 0.68:
            return make_mixed_item(rng, tokenizer, n_entities=3, surface="train", period=True)
        if roll < 0.80:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", period=True)
        if roll < 0.90:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", combine=True, period=True)
        return make_story_mixed_item(rng, tokenizer, surface="train", combine=True, period=True)
    if mix == "compose_sent_instr":
        if roll < 0.50:
            return make_compose_sentence_item(rng, tokenizer, surface="train", combine=False, instruct=True)
        if roll < 0.65:
            return make_compose_sentence_item(rng, tokenizer, surface="train", combine=True, instruct=True)
        if roll < 0.80:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", period=True)
        if roll < 0.92:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", combine=True, period=True)
        return make_who_bind_item(rng, tokenizer, n_entities=2, surface="train")
    if mix == "compose_phrase":
        if roll < 0.40:
            return make_phrase_item(rng, tokenizer, n_facts=2, surface="train")
        if roll < 0.55:
            return make_compose_sentence_item(rng, tokenizer, surface="train", combine=False)
        if roll < 0.70:
            return make_who_bind_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.85:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", period=True)
        if roll < 0.95:
            return make_mixed_item(rng, tokenizer, n_entities=2, surface="train", combine=True, period=True)
        return make_story_mixed_item(rng, tokenizer, surface="train", combine=False, period=True)
    raise ValueError(f"unknown compose mix {mix}")


def slim_sentence(report: dict) -> dict:
    ops = report.get("operators") or {}
    return {
        "verdict": report.get("verdict"),
        "lesson": report.get("lesson"),
        "operators": {
            name: {k: v for k, v in row.items() if k != "rows"}
            for name, row in ops.items()
        },
    }


def compose_scores(native: dict, sentence: dict | None = None) -> dict:
    sent_ops = (sentence or {}).get("operators") or {}
    bare = sent_ops.get("bare") or {}
    prefix = sent_ops.get("sent_prefix") or {}
    return {
        "who_2e": _top1(native, "who_bind_2e_heldout"),
        "who_3e": _top1(native, "who_bind_3e_heldout"),
        "mixed_3e": _top1(native, "mixed_3e_heldout"),
        "combine_3e": _top1(native, "fact_combine_3e_heldout"),
        "sent_exact": _exact(native, "compose_sent_heldout"),
        "combine_sent_exact": _exact(native, "compose_combine_sent_heldout"),
        "instr_exact": _exact(native, "compose_sent_instr_heldout"),
        "phrase_exact": _exact(native, "phrase_2fact_heldout"),
        "bare_sentence": float(bare.get("sentence_ok") or 0.0),
        "bare_one_word": float(bare.get("one_word_color") or 0.0),
        "prefix_sentence": float(prefix.get("sentence_ok") or 0.0),
    }


def adjudicate_compose(row: dict, *, parent: dict) -> tuple[str, str]:
    native = row.get("native") or {}
    compose = row.get("compose") or {}
    sentence = row.get("sentence")
    eng_ok, eng_lesson, collapsed = english_native_holds(native)
    mix_ok, mix_lesson = mix_holds_s2m(native)
    scores = compose_scores(compose, sentence)
    parent_scores = parent
    slice_exact = float((row.get("d3_slice") or {}).get("free_exact") or 0.0)
    d3_note = f"D3 slice {slice_exact:.3f}"
    who_delta = scores["who_2e"] - float(parent_scores.get("who_2e") or 0.0)
    who3_delta = scores["who_3e"] - float(parent_scores.get("who_3e") or 0.0)
    combine3_delta = scores["combine_3e"] - float(parent_scores.get("combine_3e") or 0.0)
    sent_delta = scores["bare_sentence"] - float(parent_scores.get("bare_sentence") or 0.0)
    exact_delta = scores["sent_exact"] - float(parent_scores.get("sent_exact") or 0.0)
    scaffold = scores["prefix_sentence"] - scores["bare_sentence"]
    lesson = (
        f"{mix_lesson}; {eng_lesson}; who2={scores['who_2e']:.3f} who3={scores['who_3e']:.3f} "
        f"combine3={scores['combine_3e']:.3f} sent_exact={scores['sent_exact']:.3f} "
        f"bare={scores['bare_sentence']:.3f} prefix={scores['prefix_sentence']:.3f}; {d3_note}"
    )
    if collapsed or not eng_ok:
        return "KILL", lesson
    if not mix_ok:
        return "KILL", lesson
    if slice_exact + 1e-12 < S2M_D3_SLICE - D3_SLICE_KILL:
        return "KILL", lesson
    sent_real = scores["bare_sentence"] >= SENT_BARE_GRAD or scores["sent_exact"] >= SENT_EXACT_GRAD
    sent_fake = sent_real and scores["bare_sentence"] < 0.25 and scaffold >= SCAFFOLD_GAP
    # mixed_3e is already solved on s2m (0.9375). Do not treat it as a composition win.
    comp_real = (
        scores["who_2e"] >= WHO_GRAD
        or who_delta >= WHO_SIGNAL
        or scores["who_3e"] >= 0.50
        or (combine3_delta >= 0.20 and scores["combine_3e"] >= 0.45)
    )
    if sent_fake and not comp_real:
        return "KILL", f"scaffold-only sentence {lesson}"
    if sent_real and not sent_fake and comp_real:
        return "ADVANCE", f"compose+sentence {lesson} whoΔ={who_delta:+.3f} sentΔ={sent_delta:+.3f}"
    if sent_real and not sent_fake:
        return "HOLD+", f"sentence signal {lesson} sentΔ={sent_delta:+.3f} exactΔ={exact_delta:+.3f}"
    if comp_real:
        return "HOLD+", f"composition signal {lesson} whoΔ={who_delta:+.3f}"
    if who_delta >= 0.12 or who3_delta >= 0.12 or combine3_delta >= 0.12 or sent_delta >= 0.12 or exact_delta >= 0.12:
        return "HOLD", lesson
    return "HOLD", lesson


def compose_milestone(d3_free, native, usable, compose, sentence) -> tuple[bool, str]:
    mix_ok, mix_lesson = mix_holds_s2m(native)
    eng_ok, eng_lesson, _ = english_native_holds(native)
    use_ok, use_lesson = usable_holds_s2m(usable)
    scores = compose_scores(compose, sentence)
    scaffold = scores["prefix_sentence"] - scores["bare_sentence"]
    sent_real = (scores["bare_sentence"] >= SENT_BARE_GRAD or scores["sent_exact"] >= SENT_EXACT_GRAD) and not (
        scores["bare_sentence"] < 0.25 and scaffold >= SCAFFOLD_GAP
    )
    # who_3e is the harder multi-entity bind; s2m is 0.0. Do not require who_2e=0.70
    # if 3-entity held-out bind is already clearly real.
    comp_real = (
        scores["who_2e"] >= WHO_GRAD
        or scores["who_3e"] >= 0.50
        or (scores["who_2e"] >= 0.40 and scores["who_3e"] >= 0.40 and scores["mixed_3e"] >= MIXED3_GRAD)
    )
    d3_ok = d3_free is None or d3_free >= 180
    if mix_ok and eng_ok and use_ok and d3_ok and comp_real and sent_real:
        return True, "compose+sentence"
    if mix_ok and eng_ok and use_ok and d3_ok and comp_real and not sent_real:
        return True, "composition-first"
    return False, "no"


def probe_compose(device, path: Path, tag: str, *, with_sentence: bool = True, with_d3: bool = False, with_usable: bool = False) -> dict:
    from .data_language_bridge import build_e13_panels, load_tokenizer

    tokenizer = load_tokenizer()
    model, _config, _ckpt = load_experimental_baby(path, device)
    native = eval_panels(model, build_e13_panels(tokenizer, n=32), device, overwrite=None, arms=("native",))["native"]
    compose = eval_panels(model, build_s3_panels(tokenizer, n=COMPOSE_N), device, overwrite=None, arms=("native",))["native"]
    sentence = run_sentence_decode(model, tokenizer, device, include=LIGHT_SENTENCE) if with_sentence else None
    usable = run_usable_chat(model, tokenizer, device) if with_usable else None
    d3 = None
    if with_d3:
        eng_ok, _, _ = english_native_holds(native)
        d3 = run_d3_log(model, device, tag, english_ok=eng_ok)
        d3_free = int(d3["long_gap"]["free_exact"])
    else:
        slice_row = cheap_d3_slice(model, device)
        d3_free = None
        d3 = {"slice": slice_row}
    scores = compose_scores(compose, sentence)
    report = {
        "id": tag,
        "checkpoint": str(path),
        "checkpoint_sha256": digest(path),
        "authoritative": False,
        "native": mix_scores(native),
        "compose": {k: {"first_top1": (compose.get(k) or {}).get("first_top1"), "free_exact": (compose.get(k) or {}).get("free_exact")} for k in compose},
        "compose_scores": scores,
        "sentence": None if sentence is None else slim_sentence(sentence),
        "usable": None if usable is None else slim_usable(usable),
        "d3_free": d3_free,
        "d3_slice": None if d3 is None else (d3.get("slice") or d3.get("long_gap")),
        "native_full": slim_panels(native),
        "compose_full": slim_panels(compose),
    }
    write(OUT / f"{tag}.json", {k: v for k, v in report.items() if k not in {"native_full", "compose_full"}} | {"native_full": report["native_full"], "compose_full": report["compose_full"]})
    print(json.dumps({"phase": f"{tag}_compose", "mix": report["native"], "compose": scores, "d3_free": d3_free}, default=str), flush=True)
    _drop_cuda(model)
    return report


def score_trained(recipe_id: str, trained: dict, parent_scores: dict) -> dict:
    history = trained.get("history") or []
    scored = []
    best_row = None
    rank = {"KILL": -1, "HOLD": 0, "HOLD+": 1, "ADVANCE": 2}
    best_key = (-1, -1.0, -1.0)
    for row in history:
        verdict, lesson = adjudicate_compose(row, parent=parent_scores)
        scores = compose_scores(row.get("compose") or {}, row.get("sentence"))
        ledger_append(
            {
                "id": f"{recipe_id}_u{int(row['update']):05d}",
                "change": trained["recipe"]["note"],
                "verdict": verdict,
                "lesson": lesson,
                "native": mix_scores(row["native"]),
                "d3": row.get("d3_slice"),
                "checkpoint_sha256": row.get("checkpoint_sha256"),
                "recipe": recipe_id,
                "update": row.get("update"),
            }
        )
        print(json.dumps({"phase": "compose_canary", "recipe": recipe_id, "update": row["update"], "verdict": verdict, "lesson": lesson, "compose": scores}, default=str), flush=True)
        packed = {
            "update": row["update"],
            "checkpoint": row["checkpoint"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "verdict": verdict,
            "lesson": lesson,
            "native": mix_scores(row["native"]),
            "compose": scores,
            "d3_slice": row.get("d3_slice"),
            "sentence": None if row.get("sentence") is None else slim_sentence(row["sentence"]),
        }
        scored.append(packed)
        if verdict == "KILL":
            continue
        key = (rank.get(verdict, 0), scores["who_2e"], scores["bare_sentence"] + scores["sent_exact"])
        if key > best_key:
            best_key = key
            best_row = {**row, "recover_verdict": verdict, "recover_lesson": lesson, "compose_scores": scores}
    return {"history": scored, "best": None if best_row is None else packed_best(best_row), "best_row": best_row, "out_dir": str(trained.get("out_dir"))}


def packed_best(row: dict) -> dict:
    return {
        "update": row["update"],
        "checkpoint": row["checkpoint"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "verdict": row.get("recover_verdict"),
        "lesson": row.get("recover_lesson"),
        "native": mix_scores(row["native"]),
        "compose": row.get("compose_scores") or compose_scores(row.get("compose") or {}, row.get("sentence")),
        "d3_slice": row.get("d3_slice"),
    }


def verify_compose(device, recipe_id: str, best_row: dict, parent_scores: dict) -> dict:
    ckpt = Path(best_row["checkpoint"])
    print(json.dumps({"phase": "compose_verify", "recipe": recipe_id, "checkpoint": str(ckpt)}, default=str), flush=True)
    report = probe_compose(device, ckpt, f"{recipe_id}_verify", with_sentence=True, with_d3=True, with_usable=True)
    native = report["native_full"]
    mix_ok, mix_lesson = mix_holds_s2m(native)
    eng_ok, eng_lesson, _ = english_native_holds(native)
    use_ok, use_lesson = usable_holds_s2m(report.get("usable"))
    d3_free = report.get("d3_free")
    kind_ok, kind = compose_milestone(d3_free, native, report.get("usable"), report["compose_full"], report.get("sentence"))
    payload = {
        "recipe": recipe_id,
        "checkpoint": str(ckpt),
        "sha256": best_row.get("checkpoint_sha256"),
        "d3_full": d3_free,
        "mix_ok": mix_ok,
        "eng_ok": eng_ok,
        "use_ok": use_ok,
        "milestone": kind_ok,
        "milestone_kind": kind if kind_ok else None,
        "lesson": f"{mix_lesson}; {eng_lesson}; {use_lesson}; D3 {d3_free}/215; {report.get('compose_scores')}",
        "native": mix_scores(native),
        "compose": report.get("compose_scores"),
        "sentence": report.get("sentence"),
        "usable": None
        if report.get("usable") is None
        else {
            "turn": report["usable"]["autoregressive"]["usable_turn"],
            "turn4": report["usable"]["autoregressive_4turn"]["usable_turn"],
            "stop": report["usable"]["autoregressive_4turn"]["period_stop"],
            "reuse": report["usable"]["autoregressive_4turn"]["fact_reuse"],
        },
    }
    campaign_update(
        {
            "status": f"s3 {kind} MILESTONE" if kind_ok else f"s3 verify {recipe_id}",
            "last_verify": {k: payload[k] for k in ("recipe", "checkpoint", "sha256", "d3_full", "milestone", "milestone_kind", "lesson") if k in payload},
        }
    )
    write(OUT / "S3.json", {"last_verify": payload})
    print(json.dumps({"phase": "compose_verify_done", "milestone": kind_ok, "kind": kind, "lesson": payload["lesson"]}, default=str), flush=True)
    return payload


def _rank_key(best: dict | None) -> tuple:
    if not best:
        return (-1, 0.0, 0.0)
    rank = {"KILL": -1, "HOLD": 0, "HOLD+": 1, "ADVANCE": 2}
    compose = best.get("compose") or {}
    return (
        rank.get(best.get("verdict"), 0),
        float(compose.get("who_2e") or 0.0),
        float(compose.get("bare_sentence") or 0.0) + float(compose.get("sent_exact") or 0.0),
    )


def _install_followup(recipe_id: str, mix: str, seed: int, note: str) -> None:
    from .selection_stack2 import RECIPES

    RECIPES[recipe_id] = {
        "seed": seed,
        "mix": mix,
        "language_p": 0.20,
        "structured_p": 0.20,
        "updates": COMPOSE_UPDATES,
        "eval_every": COMPOSE_EVAL,
        "parent": "s2m",
        "cheap_d3": True,
        "remainder_span": True,
        "compose": True,
        "note": note,
    }


def _run_arm(device, recipe_id: str, parent_scores: dict, results: dict, *, resume: Path | None = None) -> dict:
    from .selection_stack2 import RECIPES, train_recipe

    print(json.dumps({"phase": "compose_arm", "recipe": recipe_id, "note": RECIPES[recipe_id]["note"], "resume": None if resume is None else str(resume)}, default=str), flush=True)
    trained = train_recipe(device, recipe_id, resume=resume)
    scored = score_trained(recipe_id, trained, parent_scores)
    _drop_cuda(trained.pop("model", None))
    results["arms"][recipe_id] = {k: v for k, v in scored.items() if k != "best_row"}
    write(OUT / "S3.json", {k: v for k, v in results.items() if k != "promising_row"})
    return scored


def _verify_best(device, recipe_id: str, best_row: dict, parent_scores: dict, results: dict) -> dict:
    verified = verify_compose(device, recipe_id, best_row, parent_scores)
    results.setdefault("verifies", []).append(verified)
    results["last_verify"] = verified
    if verified.get("milestone"):
        results["milestone"] = True
        results["winner"] = verified
    write(OUT / "S3.json", {k: v for k, v in results.items() if k != "promising_row"})
    return verified


def run_compose_loop(device) -> dict:
    require_identities(require_s2a=True, require_s2m=True)
    campaign_update(
        {
            "status": "s3 composition+sentence canaries from s2m",
            "s2m_parent": {"path": str(S2M_SURVIVOR), "sha256": S2M_SURVIVOR_SHA},
            "hypothesis": "s2m one-word 2e bind is not multi-property composition or generated sentences. Attack entity-from-property (who-bind) and unprompted short sentences with held-out templates. Do not count sentence-prefix scaffolds as success.",
            "d3_role": "retention, not the objective",
        }
    )
    parent = probe_compose(device, S2M_SURVIVOR, "s2m_compose_zeroshot", with_sentence=True, with_d3=False, with_usable=False)
    parent_scores = parent["compose_scores"]
    results = {"parent": {k: parent[k] for k in ("compose_scores", "native", "sentence", "d3_slice") if k in parent}, "arms": {}, "verifies": [], "milestone": False}
    write(OUT / "S3.json", results)

    for recipe_id in COMPOSE_ORDER:
        _run_arm(device, recipe_id, parent_scores, results)

    ranked = sorted(
        ((rid, (results["arms"].get(rid) or {}).get("best")) for rid in results["arms"]),
        key=lambda item: _rank_key(item[1]),
        reverse=True,
    )
    promising = [(rid, best) for rid, best in ranked if best and best.get("verdict") in {"HOLD+", "ADVANCE"}]
    if not promising:
        hold = next(((rid, best) for rid, best in ranked if best and best.get("verdict") == "HOLD"), None)
        if hold is None:
            campaign_update({"status": "s3 canaries finished without milestone", "s3": {k: v.get("best") for k, v in results.get("arms", {}).items()}})
            write(OUT / "S3.json", {k: v for k, v in results.items() if k != "promising_row"})
            print(json.dumps({"phase": "compose_done", "milestone": False}, default=str), flush=True)
            return results
        parent_id, parent_best = hold
        ext_id = f"{parent_id}x"
        _install_followup(
            ext_id,
            str(RECIPES_MIX[parent_id]),
            324051,
            f"Extend {parent_id} +25 from its u25 HOLD; 25 was mixed, not a stop.",
        )
        scored = _run_arm(device, ext_id, parent_scores, results, resume=Path(parent_best["checkpoint"]))
        best_row = scored.get("best_row")
        if best_row is None or best_row.get("recover_verdict") not in {"HOLD+", "ADVANCE"}:
            campaign_update({"status": "s3 canaries finished without milestone", "s3": {k: v.get("best") for k, v in results.get("arms", {}).items()}})
            write(OUT / "S3.json", {k: v for k, v in results.items() if k != "promising_row"})
            print(json.dumps({"phase": "compose_done", "milestone": False}, default=str), flush=True)
            return results
        promising = [(ext_id, packed_best(best_row))]
        results["arms"][ext_id]["best_row_checkpoint"] = best_row["checkpoint"]

    recipe_id, best = promising[0]
    best_row = {"checkpoint": best["checkpoint"], "checkpoint_sha256": best.get("checkpoint_sha256")}
    verified = _verify_best(device, recipe_id, best_row, parent_scores, results)
    if verified.get("milestone_kind") == "compose+sentence":
        print(json.dumps({"phase": "compose_done", "milestone": True, "kind": "compose+sentence", "winner": recipe_id}, default=str), flush=True)
        return results
    if not verified.get("mix_ok"):
        campaign_update({"status": "s3 verify lost s2m mix; no follow-up from a dropped parent", "s3": {k: v.get("best") for k, v in results.get("arms", {}).items()}})
        write(OUT / "S3.json", {k: v for k, v in results.items() if k != "promising_row"})
        print(json.dumps({"phase": "compose_done", "milestone": False, "reason": "verify mix drop"}, default=str), flush=True)
        return results

    scores = verified.get("compose") or {}
    sent_real = scores.get("bare_sentence", 0) >= SENT_BARE_GRAD or scores.get("sent_exact", 0) >= SENT_EXACT_GRAD
    scaffold = float(scores.get("prefix_sentence") or 0.0) - float(scores.get("bare_sentence") or 0.0)
    if sent_real and scores.get("bare_sentence", 0) < 0.25 and scaffold >= SCAFFOLD_GAP:
        sent_real = False
    comp_real = scores.get("who_2e", 0) >= WHO_GRAD or (
        scores.get("who_2e", 0) >= 0.55 and scores.get("mixed_3e", 0) >= MIXED3_GRAD
    )
    if comp_real and not sent_real:
        follow_mix, follow_id, follow_seed = "compose_sent", "s3f", 324061
    elif sent_real and not comp_real:
        follow_mix, follow_id, follow_seed = "compose_bind", "s3g", 324071
    else:
        follow_mix, follow_id, follow_seed = "compose_both", "s3h", 324081
    _install_followup(
        follow_id,
        follow_mix,
        follow_seed,
        f"From {recipe_id}: train the missing half ({follow_mix}) instead of stopping at one signal.",
    )
    scored = _run_arm(device, follow_id, parent_scores, results, resume=Path(verified["checkpoint"]))
    follow_best = scored.get("best_row")
    if follow_best is not None and follow_best.get("recover_verdict") in {"HOLD+", "ADVANCE"}:
        follow_verified = _verify_best(device, follow_id, follow_best, parent_scores, results)
        if follow_verified.get("milestone"):
            print(json.dumps({"phase": "compose_done", "milestone": True, "kind": follow_verified.get("milestone_kind"), "winner": follow_id}, default=str), flush=True)
            return results
    if verified.get("milestone"):
        print(json.dumps({"phase": "compose_done", "milestone": True, "kind": verified.get("milestone_kind"), "winner": recipe_id}, default=str), flush=True)
        return results
    campaign_update({"status": "s3 canaries finished without milestone", "s3": {k: v.get("best") for k, v in results.get("arms", {}).items()}})
    write(OUT / "S3.json", {k: v for k, v in results.items() if k != "promising_row"})
    print(json.dumps({"phase": "compose_done", "milestone": False}, default=str), flush=True)
    return results
