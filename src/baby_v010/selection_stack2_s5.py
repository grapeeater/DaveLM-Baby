from __future__ import annotations

"""Stack2 s5: compositional WHO sentences (A) then richer relations (B) from s4m.

U16000 stays authoritative. TEST/FINAL/SACRED stay sealed. s4m is experimental only.
Nothing here is promoted. Pulse → lock; do not leave a high-dose pulse on.
"""

import json
from pathlib import Path

from .data import BOS
from .data_language_bridge import (
    build_beside_decode_pack,
    build_has_decode_pack,
    build_s3_panels,
    build_who_sentence_decode_pack,
    encode_ids,
    make_beside_item,
    make_compose_sentence_item,
    make_has_object_item,
    make_who_bind_item,
    make_who_pair_items,
    make_who_sentence_item,
    score_bind_sentence,
    score_has_sentence,
)
from .selection_language_bridge import (
    SENTENCE_OPERATORS,
    _rate,
    eval_panels,
    greedy_decode_until_stop,
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
    campaign_update,
    cheap_d3_slice,
    ledger_append,
    mix_scores,
    require_identities,
    run_d3_log,
    slim_usable,
    _drop_cuda,
    _exact,
    _top1,
)
from .selection_stack2_s3 import LIGHT_SENTENCE, SCAFFOLD_GAP, slim_sentence
from .selection_stack2_s4 import sentence_scores, who_holds_s3s

S4M_SURVIVOR = OUT / "s4m_compose_lock_325231" / "checkpoint_00025.pt"
S4M_SURVIVOR_SHA = "4c0f142768aa57e2421a94a5214392e5c2d71e3805a2c0f6d5c9f0bc984c95e5"
S4M_D3_SLICE = 0.850
S4M_D3_FULL = 185
S4M_MIXED = 0.8125
S4M_COMBINE = 0.6875
S4M_COLOR = 0.75
S4M_USABLE = 0.857
S4M_USABLE4 = 0.906
S4M_BARE = 0.375
WHO2_KEEP = 0.35
WHO3_KEEP = 0.25
DIRECT_BARE_KEEP = 0.25
WHO_SENT_SIGNAL = 0.25
WHO_SENT_GRAD = 0.75
REL_GRAD = 0.75
REL_DIR_GRAD = 0.75
A_KEEP_AFTER_B = 0.50
COLOR_KEEP = 0.70
SIZE_KEEP = 0.90
COMBINE_KEEP = 0.60
MIXED_KEEP = 0.75
D3_SLICE_KILL = 0.125
S5_UPDATES = 25
MAX_ARMS_A = 6
MAX_ARMS_B = 6
CHEAP_OPS = ("bare", "sent_prefix")
_WHO_PAIR_Q: list[dict] = []


def mix_holds_s4m(native: dict) -> tuple[bool, str]:
    scores = mix_scores(native)
    mixed_ok = scores["mixed_2e"] + 1e-12 >= MIXED_KEEP
    combine_ok = scores["fact_combine"] + 1e-12 >= COMBINE_KEEP
    lesson = f"mixed={scores['mixed_2e']:.3f} fact_combine={scores['fact_combine']:.3f} story_combine={scores['story_combine']:.3f}"
    if mixed_ok and combine_ok:
        return True, f"s4m mix hold {lesson}"
    return False, f"s4m mix drop {lesson}"


def english_holds_s4m(native: dict) -> tuple[bool, str, bool]:
    scores = mix_scores(native)
    collapsed = scores["color"] < COLOR_KEEP or scores["size_stop"] < 0.60
    ok = scores["color"] + 1e-12 >= COLOR_KEEP and scores["size_stop"] + 1e-12 >= SIZE_KEEP
    lesson = f"color={scores['color']:.3f} size_stop={scores['size_stop']:.3f} story={scores['story_color']:.3f}"
    if collapsed:
        return False, f"s4m collapse {lesson}", True
    if ok:
        return True, f"s4m english hold {lesson}", False
    return False, f"s4m english drop {lesson}", False


def usable_holds_s4m(usable: dict | None) -> tuple[bool, str]:
    if usable is None:
        return False, "usable-chat missing"
    auto = usable.get("autoregressive") or {}
    auto4 = usable.get("autoregressive_4turn") or auto
    turn = float(auto.get("usable_turn") or 0.0)
    turn4 = float(auto4.get("usable_turn") or 0.0)
    stop = float(auto4.get("period_stop") or auto.get("period_stop") or 0.0)
    reuse = float(auto4.get("fact_reuse") or auto.get("fact_reuse") or 0.0)
    hold = (
        turn + 1e-12 >= S4M_USABLE - DROP_BAR
        and turn4 + 1e-12 >= S4M_USABLE4 - DROP_BAR
        and stop + 1e-12 >= 0.95
        and reuse + 1e-12 >= 0.95
    )
    note = f"usable4={turn4:.3f} usable={turn:.3f} stop={stop:.3f} reuse={reuse}"
    if hold:
        return True, f"s4m usable hold {note}"
    return False, f"s4m usable drop {note}"


def sample_s5_item(rng, tokenizer, kind: str) -> dict:
    if kind == "who_pulse":
        roll = rng.random()
        if roll < 0.40:
            return make_who_sentence_item(rng, tokenizer, n_entities=1, surface="train")
        if roll < 0.80:
            return make_who_sentence_item(rng, tokenizer, n_entities=2, surface="train")
        return make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train")
    if kind == "who_lock":
        roll = rng.random()
        if roll < 0.25:
            return make_who_sentence_item(rng, tokenizer, n_entities=1, surface="train")
        if roll < 0.55:
            return make_who_sentence_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.80:
            return make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train")
        return make_compose_sentence_item(rng, tokenizer, n_entities=2, surface="train")
    if kind == "who_focus":
        roll = rng.random()
        if roll < 0.25:
            return make_who_sentence_item(rng, tokenizer, n_entities=1, surface="train")
        if roll < 0.65:
            return make_who_sentence_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.85:
            return make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train")
        return make_compose_sentence_item(rng, tokenizer, n_entities=2, surface="train")
    if kind == "who_2e":
        if rng.random() < 0.85:
            return make_who_sentence_item(rng, tokenizer, n_entities=2, surface="train")
        return make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train")
    if kind == "who_entity":
        n_entities = 3 if rng.random() < 0.5 else 2
        if rng.random() < 0.85:
            return make_who_sentence_item(rng, tokenizer, n_entities=n_entities, surface="train")
        return make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train")
    if kind == "who_select":
        roll = rng.random()
        if roll < 0.55:
            return make_who_bind_item(
                rng, tokenizer, n_entities=2, surface="train", asked_attr_only=True, anti_recency=True
            )
        if roll < 0.80:
            return make_who_bind_item(rng, tokenizer, n_entities=2, surface="train", asked_attr_only=True)
        if roll < 0.92:
            return make_who_bind_item(rng, tokenizer, n_entities=2, surface="train")
        return make_who_bind_item(rng, tokenizer, n_entities=3, surface="train", anti_recency=True)
    if kind == "who_select_uniform":
        roll = rng.random()
        if roll < 0.70:
            return make_who_bind_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.90:
            return make_who_bind_item(rng, tokenizer, n_entities=2, surface="train", asked_attr_only=True)
        return make_who_bind_item(rng, tokenizer, n_entities=3, surface="train")
    if kind == "who_select_sent":
        roll = rng.random()
        if roll < 0.35:
            return make_who_sentence_item(rng, tokenizer, n_entities=2, surface="train", anti_recency=True)
        if roll < 0.55:
            return make_who_sentence_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.70:
            return make_who_sentence_item(rng, tokenizer, n_entities=1, surface="train")
        if roll < 0.88:
            return make_who_bind_item(
                rng, tokenizer, n_entities=2, surface="train", asked_attr_only=True, anti_recency=True
            )
        return make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train")
    if kind == "who_pair":
        global _WHO_PAIR_Q
        if rng.random() < 0.12:
            return make_who_sentence_item(rng, tokenizer, n_entities=1, surface="train")
        as_sentence = rng.random() < 0.75
        if len(_WHO_PAIR_Q) < 2:
            _WHO_PAIR_Q.extend(make_who_pair_items(rng, tokenizer, surface="train", as_sentence=as_sentence))
        return _WHO_PAIR_Q.pop(0)
    if kind == "who_a_pulse":
        roll = rng.random()
        if roll < 0.40:
            return make_who_sentence_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.55:
            return make_who_sentence_item(rng, tokenizer, n_entities=1, surface="train")
        if roll < 0.85:
            return make_who_bind_item(rng, tokenizer, n_entities=2, surface="train", asked_attr_only=True)
        return make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train")
    if kind == "rel_pulse":
        roll = rng.random()
        if roll < 0.35:
            return make_has_object_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.70:
            return make_beside_item(rng, tokenizer, surface="train")
        if roll < 0.85:
            return make_who_sentence_item(rng, tokenizer, n_entities=2, surface="train")
        return make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train")
    if kind == "rel_balance":
        roll = rng.random()
        if roll < 0.22:
            return make_has_object_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.44:
            return make_beside_item(rng, tokenizer, surface="train")
        if roll < 0.72:
            n_entities = 1 if rng.random() < 0.4 else 2
            return make_who_sentence_item(rng, tokenizer, n_entities=n_entities, surface="train")
        return make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train")
    if kind == "rel_lock":
        roll = rng.random()
        if roll < 0.25:
            return make_has_object_item(rng, tokenizer, n_entities=2, surface="train")
        if roll < 0.50:
            return make_beside_item(rng, tokenizer, surface="train")
        if roll < 0.70:
            return make_who_sentence_item(rng, tokenizer, n_entities=2, surface="train")
        return make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train")
    if kind == "rel_forward":
        roll = rng.random()
        if roll < 0.28:
            return make_has_object_item(rng, tokenizer, n_entities=2, surface="train", direction="what")
        if roll < 0.46:
            return make_has_object_item(rng, tokenizer, n_entities=2, surface="train", direction="who")
        if roll < 0.68:
            return make_beside_item(rng, tokenizer, surface="train", direction="where")
        if roll < 0.80:
            return make_beside_item(rng, tokenizer, surface="train", direction="who")
        if roll < 0.92:
            return make_who_sentence_item(rng, tokenizer, n_entities=2, surface="train")
        return make_compose_sentence_item(rng, tokenizer, n_entities=1, surface="train")
    raise ValueError(f"unknown s5 sent_kind {kind}")


def entity_select_margin(logits, mask, items, tokenizer):
    """Hinge: gold entity first-token must beat scene distractors at answer start."""
    import torch

    from .data_language_bridge import spaced_first_id

    parts = []
    for i, item in enumerate(items):
        gold = str(item.get("entity") or "")
        distractors = [str(name) for name in (item.get("distractors") or []) if name and name != gold]
        if not gold or not distractors:
            continue
        row_mask = mask[i]
        if not bool(row_mask.any()):
            continue
        pos = int(row_mask.nonzero(as_tuple=False)[0, 0])
        gold_id = spaced_first_id(tokenizer, gold)
        d_ids = [spaced_first_id(tokenizer, name) for name in distractors]
        row_logits = logits[i, pos]
        best_d = row_logits[d_ids].max()
        parts.append(torch.relu(1.0 - row_logits[gold_id] + best_d))
    if not parts:
        return None
    return torch.stack(parts).mean()


def run_bind_decode(
    model,
    tokenizer,
    device,
    pack: list[dict],
    *,
    predicates: tuple[str, ...],
    include: tuple[str, ...] | None = None,
    scorer=None,
    max_new: int = 16,
) -> dict:
    wanted = tuple(name for name, _t, _l in SENTENCE_OPERATORS if include is None or name in include)
    by_op: dict[str, list[dict]] = {name: [] for name in wanted}
    score_fn = scorer or (
        lambda full, entity, value, stopped: score_bind_sentence(
            full_text=full, entity=entity, value=value, stopped=stopped, predicates=predicates
        )
    )
    for item in pack:
        entity = item["entity"]
        value = item.get("value") or item.get("color") or ""
        for name, template, lead in SENTENCE_OPERATORS:
            if name not in by_op:
                continue
            prompt = template.format(facts=item["facts"], query=item["query"], entity=entity)
            if name == "force_entity_is":
                lead_text = f" {entity} is"
            elif name == "force_the_entity_is":
                lead_text = f" The {entity} is"
            else:
                lead_text = lead
            prompt_ids = [BOS, *encode_ids(tokenizer, prompt)]
            emitted, stopped = greedy_decode_until_stop(model, prompt_ids, device, tokenizer, max_new=max_new)
            decoded = tokenizer.decode(emitted, skip_special_tokens=True)
            full = (lead_text or "") + decoded
            scored = score_fn(full, entity, value, stopped)
            by_op[name].append(
                {
                    "id": item["id"],
                    "kind": item.get("kind"),
                    "prompt": prompt,
                    "decoded": decoded,
                    "full": full,
                    "gold": value,
                    "entity": entity,
                    "stopped": stopped,
                    "n_tokens": len(emitted),
                    **scored,
                }
            )
    operators = {}
    for name, rows in by_op.items():
        operators[name] = {
            "sentence_ok": _rate([bool(row["sentence_ok"]) for row in rows]),
            "first_entity": _rate([bool(row.get("first_entity")) for row in rows]),
            "has_entity": _rate([bool(row.get("has_entity")) for row in rows]),
            "has_value": _rate([bool(row.get("has_value")) for row in rows]),
            "predicate_ok": _rate([bool(row.get("predicate_ok")) for row in rows]),
            "period": _rate([bool(row.get("period")) for row in rows]),
            "stopped": _rate([bool(row.get("stopped")) for row in rows]),
            "one_word_entity": _rate([bool(row.get("one_word_entity")) for row in rows]),
            "one_word_color": _rate([bool(row.get("one_word_color")) for row in rows]),
            "light": name in {"bare", "sent_prefix", "short_prefix", "baby", "force_the"},
            "rows": rows,
        }
    return {"id": "bind_decode", "operators": operators}


def summarize_bind(report: dict) -> dict:
    ops = report.get("operators") or {}
    bare = ops.get("bare") or {}
    prefix = ops.get("sent_prefix") or {}
    rows = bare.get("rows") or []
    by_kind: dict[str, list[dict]] = {}
    for row in rows:
        by_kind.setdefault(str(row.get("kind") or "all"), []).append(row)
    return {
        "bare": float(bare.get("sentence_ok") or 0.0),
        "prefix": float(prefix.get("sentence_ok") or 0.0),
        "first_entity": float(bare.get("first_entity") or 0.0),
        "has_entity": float(bare.get("has_entity") or 0.0),
        "has_value": float(bare.get("has_value") or 0.0),
        "predicate": float(bare.get("predicate_ok") or 0.0),
        "one_word_entity": float(bare.get("one_word_entity") or 0.0),
        "by_kind": {kind: _rate([bool(row["sentence_ok"]) for row in kind_rows]) for kind, kind_rows in by_kind.items()},
    }


def slim_bind(report: dict) -> dict:
    return slim_sentence(report)


def bind_examples(report: dict, *, n: int = 8) -> list[dict]:
    rows = ((report.get("operators") or {}).get("bare") or {}).get("rows") or []
    return [
        {
            "id": row.get("id"),
            "kind": row.get("kind"),
            "decoded": row.get("decoded"),
            "full": row.get("full"),
            "sentence_ok": row.get("sentence_ok"),
            "first_entity": row.get("first_entity"),
        }
        for row in rows[:n]
    ]


def run_who_sentence_decode(model, tokenizer, device, *, include: tuple[str, ...] | None = None) -> dict:
    return run_bind_decode(
        model,
        tokenizer,
        device,
        build_who_sentence_decode_pack(),
        predicates=("is", "looks"),
        include=include or LIGHT_SENTENCE,
    )


def run_has_decode(model, tokenizer, device, *, include: tuple[str, ...] | None = None) -> dict:
    return run_bind_decode(
        model,
        tokenizer,
        device,
        build_has_decode_pack(),
        predicates=("has",),
        include=include or LIGHT_SENTENCE,
        scorer=lambda full, entity, value, stopped: score_has_sentence(
            full_text=full, entity=entity, value=value, stopped=stopped
        ),
        max_new=20,
    )


def run_beside_decode(model, tokenizer, device, *, include: tuple[str, ...] | None = None) -> dict:
    return run_bind_decode(
        model,
        tokenizer,
        device,
        build_beside_decode_pack(),
        predicates=("beside",),
        include=include or LIGHT_SENTENCE,
        max_new=20,
    )


def attach_s5_eval(row: dict, model, tokenizer, device, recipe: dict) -> None:
    runtime = None
    assist = None
    if recipe.get("router"):
        from .selection_stack2_r2 import RelAssistRuntime, WhoPropRuntime

        runtime = WhoPropRuntime(model, tokenizer).install()
        runtime.enabled = True
        assist = RelAssistRuntime(model, tokenizer).install()
        assist.enabled = True
    try:
        _attach_s5_eval_body(row, model, tokenizer, device, recipe)
    finally:
        if assist is not None:
            assist.uninstall()
        if runtime is not None:
            runtime.uninstall()


def _attach_s5_eval_body(row: dict, model, tokenizer, device, recipe: dict) -> None:
    who = run_who_sentence_decode(model, tokenizer, device, include=CHEAP_OPS)
    row["who_sent"] = slim_bind(who)
    row["who_sent_scores"] = summarize_bind(who)
    lesson = (
        f"{row.get('lesson', '')}; who_sent_bare={row['who_sent_scores']['bare']:.3f} "
        f"first={row['who_sent_scores']['first_entity']:.3f}"
    )
    if recipe.get("relate"):
        has = run_has_decode(model, tokenizer, device, include=CHEAP_OPS)
        beside = run_beside_decode(model, tokenizer, device, include=CHEAP_OPS)
        row["has_scores"] = summarize_bind(has)
        row["beside_scores"] = summarize_bind(beside)
        row["has"] = slim_bind(has)
        row["beside"] = slim_bind(beside)
        lesson = (
            f"{lesson}; has_bare={row['has_scores']['bare']:.3f} "
            f"beside_bare={row['beside_scores']['bare']:.3f}"
        )
    row["lesson"] = lesson


def _scaffold_only(bare: float, prefix: float) -> bool:
    return bare < WHO_SENT_SIGNAL and (prefix - bare) >= SCAFFOLD_GAP


def retention_ok(native: dict, compose: dict, sentence: dict | None, d3_slice: dict | None) -> tuple[bool, str, bool]:
    eng_ok, eng_lesson, collapsed = english_holds_s4m(native)
    mix_ok, mix_lesson = mix_holds_s4m(native)
    scores = sentence_scores(compose, sentence)
    who_ok, who_lesson = who_holds_s3s(scores)
    slice_exact = float((d3_slice or {}).get("free_exact") or 0.0)
    d3_ok = slice_exact + 1e-12 >= S4M_D3_SLICE - D3_SLICE_KILL
    direct_ok = float(scores.get("bare_sentence") or 0.0) + 1e-12 >= DIRECT_BARE_KEEP
    lesson = f"{mix_lesson}; {eng_lesson}; {who_lesson}; direct_bare={scores.get('bare_sentence', 0):.3f} D3 slice {slice_exact:.3f}"
    if collapsed:
        return False, lesson, True
    if not (eng_ok and mix_ok and who_ok and d3_ok and direct_ok):
        return False, lesson, False
    return True, lesson, False


def adjudicate_a(row: dict, *, parent: dict) -> tuple[str, str]:
    native = row.get("native") or {}
    compose = row.get("compose") or {}
    ok, ret_lesson, collapsed = retention_ok(native, compose, row.get("sentence"), row.get("d3_slice"))
    who = row.get("who_sent_scores") or {}
    bare = float(who.get("bare") or 0.0)
    prefix = float(who.get("prefix") or 0.0)
    first = float(who.get("first_entity") or 0.0)
    kinds = who.get("by_kind") or {}
    parent_bare = float(parent.get("who_sent_bare") or 0.0)
    lesson = f"{ret_lesson}; who_sent={bare:.3f} prefix={prefix:.3f} first={first:.3f} kinds={kinds}"
    if collapsed:
        return "KILL", lesson
    if not ok and float(mix_scores(native)["color"]) < COLOR_KEEP:
        return "KILL", lesson
    if not ok and float(mix_scores(native)["fact_combine"]) < COMBINE_KEEP - 0.05:
        return "KILL", lesson
    if _scaffold_only(bare, prefix):
        return "HOLD", f"scaffold-only WHO sentence {lesson}"
    if bare + 1e-12 >= WHO_SENT_GRAD and not _scaffold_only(bare, prefix):
        return "ADVANCE", f"WHO sentence {lesson} Δ={bare - parent_bare:+.3f}"
    if bare + 1e-12 >= WHO_SENT_SIGNAL or first + 1e-12 >= 0.50:
        return "HOLD+", f"WHO sentence signal {lesson} Δ={bare - parent_bare:+.3f}"
    if bare > parent_bare + 0.06:
        return "HOLD", lesson
    return "HOLD", lesson


def adjudicate_b(row: dict, *, parent: dict) -> tuple[str, str]:
    native = row.get("native") or {}
    compose = row.get("compose") or {}
    ok, ret_lesson, collapsed = retention_ok(native, compose, row.get("sentence"), row.get("d3_slice"))
    who = row.get("who_sent_scores") or {}
    has = row.get("has_scores") or {}
    beside = row.get("beside_scores") or {}
    has_bare = float(has.get("bare") or 0.0)
    beside_bare = float(beside.get("bare") or 0.0)
    who_bare = float(who.get("bare") or 0.0)
    lesson = (
        f"{ret_lesson}; A_who={who_bare:.3f} has={has_bare:.3f} {has.get('by_kind')} "
        f"beside={beside_bare:.3f} {beside.get('by_kind')}"
    )
    if collapsed:
        return "KILL", lesson
    if who_bare + 1e-12 < A_KEEP_AFTER_B and has_bare + beside_bare > 0:
        return "KILL", f"A died while learning relations {lesson}"
    if has_bare + 1e-12 >= REL_GRAD and beside_bare + 1e-12 >= REL_GRAD:
        return "ADVANCE", lesson
    if has_bare + 1e-12 >= WHO_SENT_SIGNAL or beside_bare + 1e-12 >= WHO_SENT_SIGNAL:
        return "HOLD+", lesson
    return "HOLD", lesson


def milestone_a(native, usable, compose, sentence, who_scores, *, d3_free=None) -> tuple[bool, str]:
    eng_ok, _, collapsed = english_holds_s4m(native)
    mix_ok, _ = mix_holds_s4m(native)
    use_ok, _ = usable_holds_s4m(usable)
    sent = sentence_scores(compose, sentence)
    who_ok, _ = who_holds_s3s(sent)
    bare = float(who_scores.get("bare") or 0.0)
    prefix = float(who_scores.get("prefix") or 0.0)
    kinds = who_scores.get("by_kind") or {}
    color_k = float(kinds.get("who_color") or 0.0)
    size_k = float(kinds.get("who_size") or 0.0)
    d3_ok = d3_free is None or d3_free >= 170
    if collapsed or not (eng_ok and mix_ok and use_ok and who_ok and d3_ok):
        return False, "retention"
    if _scaffold_only(bare, prefix):
        return False, "scaffold"
    if sent["bare_sentence"] + 1e-12 < DIRECT_BARE_KEEP:
        return False, "direct-sentence-dead"
    if bare + 1e-12 < WHO_SENT_GRAD:
        return False, "who-sent-weak"
    if color_k + 1e-12 < 0.50 or size_k + 1e-12 < 0.50:
        return False, "one-family-only"
    return True, "who-sentence"


def milestone_b(native, usable, compose, sentence, who_scores, has_scores, beside_scores, *, d3_free=None) -> tuple[bool, str]:
    a_ok, a_why = milestone_a(native, usable, compose, sentence, who_scores, d3_free=d3_free)
    if not a_ok and a_why != "who-sent-weak":
        # A may dip numerically but must stay alive.
        if float(who_scores.get("bare") or 0.0) + 1e-12 < A_KEEP_AFTER_B:
            return False, f"A-dead:{a_why}"
    elif not a_ok and float(who_scores.get("bare") or 0.0) + 1e-12 < A_KEEP_AFTER_B:
        return False, "A-dead"
    has_bare = float(has_scores.get("bare") or 0.0)
    beside_bare = float(beside_scores.get("bare") or 0.0)
    has_kinds = has_scores.get("by_kind") or {}
    beside_kinds = beside_scores.get("by_kind") or {}
    if has_bare + 1e-12 < REL_GRAD:
        return False, "has-weak"
    if beside_bare + 1e-12 < REL_GRAD:
        return False, "beside-weak"
    if float(has_kinds.get("has_who") or 0.0) + 1e-12 < REL_DIR_GRAD:
        return False, "has-inverse-weak"
    if float(has_kinds.get("has_what") or 0.0) + 1e-12 < REL_DIR_GRAD:
        return False, "has-forward-weak"
    if float(beside_kinds.get("beside_where") or 0.0) + 1e-12 < REL_DIR_GRAD:
        return False, "beside-forward-weak"
    if float(beside_kinds.get("beside_who") or 0.0) + 1e-12 < REL_DIR_GRAD:
        return False, "beside-inverse-weak"
    if _scaffold_only(has_bare, float(has_scores.get("prefix") or 0.0)) or _scaffold_only(
        beside_bare, float(beside_scores.get("prefix") or 0.0)
    ):
        return False, "scaffold"
    return True, "relations"


def probe_s5(
    device,
    path: Path,
    tag: str,
    *,
    with_d3: bool = False,
    with_usable: bool = False,
    with_relate: bool = False,
    include: tuple[str, ...] | None = None,
    router: bool = False,
) -> dict:
    from .data_language_bridge import build_e13_panels, load_tokenizer

    ops = include or LIGHT_SENTENCE
    tokenizer = load_tokenizer()
    model, _config, _ckpt = load_experimental_baby(path, device)
    runtime = None
    assist = None
    if router:
        from .selection_stack2_r2 import RelAssistRuntime, WhoPropRuntime

        runtime = WhoPropRuntime(model, tokenizer).install()
        runtime.enabled = True
        assist = RelAssistRuntime(model, tokenizer).install()
        assist.enabled = True
    native = eval_panels(model, build_e13_panels(tokenizer, n=32), device, overwrite=None, arms=("native",))["native"]
    compose = eval_panels(model, build_s3_panels(tokenizer, n=32), device, overwrite=None, arms=("native",))["native"]
    sentence = run_sentence_decode(model, tokenizer, device, include=LIGHT_SENTENCE)
    who = run_who_sentence_decode(model, tokenizer, device, include=ops)
    has = run_has_decode(model, tokenizer, device, include=ops) if with_relate else None
    beside = run_beside_decode(model, tokenizer, device, include=ops) if with_relate else None
    usable = run_usable_chat(model, tokenizer, device) if with_usable else None
    d3 = None
    d3_free = None
    if with_d3:
        eng_ok, _, _ = english_holds_s4m(native)
        d3 = run_d3_log(model, device, tag, english_ok=eng_ok)
        d3_free = int(d3["long_gap"]["free_exact"])
    else:
        d3 = {"slice": cheap_d3_slice(model, device)}
    report = {
        "id": tag,
        "checkpoint": str(path),
        "checkpoint_sha256": digest(path),
        "authoritative": False,
        "native": mix_scores(native),
        "compose": {
            k: {"first_top1": (compose.get(k) or {}).get("first_top1"), "free_exact": (compose.get(k) or {}).get("free_exact")}
            for k in compose
        },
        "sentence_scores": sentence_scores(compose, sentence),
        "who_sent_scores": summarize_bind(who),
        "has_scores": None if has is None else summarize_bind(has),
        "beside_scores": None if beside is None else summarize_bind(beside),
        "sentence": slim_sentence(sentence),
        "who_sent": slim_bind(who),
        "has": None if has is None else slim_bind(has),
        "beside": None if beside is None else slim_bind(beside),
        "who_examples": bind_examples(who, n=16),
        "has_examples": [] if has is None else bind_examples(has, n=12),
        "beside_examples": [] if beside is None else bind_examples(beside, n=12),
        "bare_examples": [
            {"id": row.get("id"), "decoded": row.get("decoded"), "full": row.get("full"), "sentence_ok": row.get("sentence_ok")}
            for row in ((sentence.get("operators") or {}).get("bare") or {}).get("rows") or []
        ][:8],
        "usable": None if usable is None else slim_usable(usable),
        "d3_free": d3_free,
        "d3_slice": None if d3 is None else (d3.get("slice") or d3.get("long_gap")),
        "native_full": slim_panels(native),
        "compose_full": slim_panels(compose),
        "router": bool(router),
    }
    if assist is not None:
        assist.uninstall()
    if runtime is not None:
        runtime.uninstall()
    write(OUT / f"{tag}.json", {k: v for k, v in report.items() if k not in {"native_full", "compose_full"}})
    print(
        json.dumps(
            {
                "phase": f"{tag}_s5",
                "mix": report["native"],
                "sentence": report["sentence_scores"],
                "who_sent": report["who_sent_scores"],
                "has": report["has_scores"],
                "beside": report["beside_scores"],
                "d3_free": d3_free,
            },
            default=str,
        ),
        flush=True,
    )
    _drop_cuda(model)
    return report


def packed_best(row: dict) -> dict:
    return {
        "update": row["update"],
        "checkpoint": row["checkpoint"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "verdict": row.get("recover_verdict"),
        "lesson": row.get("recover_lesson"),
        "native": mix_scores(row["native"]),
        "sentence": row.get("sentence_scores") or sentence_scores(row.get("compose") or {}, row.get("sentence")),
        "who_sent": row.get("who_sent_scores"),
        "has": row.get("has_scores"),
        "beside": row.get("beside_scores"),
        "d3_slice": row.get("d3_slice"),
    }


def score_trained(recipe_id: str, trained: dict, parent_scores: dict, *, phase: str) -> dict:
    history = trained.get("history") or []
    scored = []
    best_row = None
    rank = {"KILL": -1, "HOLD": 0, "HOLD+": 1, "ADVANCE": 2}
    best_key = (-1, -1.0, -1.0)
    last_packed = None
    judge = adjudicate_b if phase == "B" else adjudicate_a
    for row in history:
        if "sentence_scores" not in row:
            row["sentence_scores"] = sentence_scores(row.get("compose") or {}, row.get("sentence"))
        verdict, lesson = judge(row, parent=parent_scores)
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
        print(
            json.dumps(
                {
                    "phase": f"s5_{phase}_canary",
                    "recipe": recipe_id,
                    "update": row["update"],
                    "verdict": verdict,
                    "lesson": lesson,
                    "who_sent": row.get("who_sent_scores"),
                    "has": row.get("has_scores"),
                    "beside": row.get("beside_scores"),
                },
                default=str,
            ),
            flush=True,
        )
        packed = {
            "update": row["update"],
            "checkpoint": row["checkpoint"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "verdict": verdict,
            "lesson": lesson,
            "native": mix_scores(row["native"]),
            "sentence": row.get("sentence_scores"),
            "who_sent": row.get("who_sent_scores"),
            "has": row.get("has_scores"),
            "beside": row.get("beside_scores"),
            "d3_slice": row.get("d3_slice"),
        }
        last_packed = packed
        scored.append(packed)
        if verdict == "KILL":
            continue
        target = float((row.get("has_scores") or {}).get("bare") or 0.0) + float((row.get("beside_scores") or {}).get("bare") or 0.0)
        if phase == "A":
            target = float((row.get("who_sent_scores") or {}).get("bare") or 0.0)
        key = (rank.get(verdict, 0), target, float((row.get("who_sent_scores") or {}).get("first_entity") or 0.0))
        if key > best_key:
            best_key = key
            best_row = {**row, "recover_verdict": verdict, "recover_lesson": lesson}
    return {
        "history": scored,
        "best": None if best_row is None else packed_best(best_row),
        "best_row": best_row,
        "last": last_packed,
        "out_dir": str(trained.get("out_dir")),
    }


def _install(
    recipe_id: str,
    *,
    sentence_p: float,
    sent_kind: str,
    seed: int,
    note: str,
    relate: bool = False,
    language_p: float | None = None,
    structured_p: float | None = None,
) -> None:
    from .selection_stack2 import RECIPES

    if language_p is None or structured_p is None:
        if sentence_p >= 0.20:
            language_p, structured_p = 0.20, 0.20
        else:
            language_p, structured_p = 0.32, 0.28
    RECIPES[recipe_id] = {
        "seed": seed,
        "mix": "compose_lock",
        "language_p": language_p,
        "structured_p": structured_p,
        "sentence_p": sentence_p,
        "sent_kind": sent_kind,
        "sent_sampler": "s5",
        "who_sent": True,
        "relate": relate,
        "updates": S5_UPDATES,
        "eval_every": S5_UPDATES,
        "parent": "s4m",
        "cheap_d3": True,
        "remainder_span": True,
        "compose": True,
        "lock_from_drop": True,
        "lr_scale": 0.5,
        "note": note,
    }


def _run_arm(device, recipe_id: str, parent_scores: dict, results: dict, *, resume: Path | None, phase: str) -> dict:
    from .selection_stack2 import RECIPES, train_recipe

    print(
        json.dumps(
            {
                "phase": f"s5_{phase}_arm",
                "recipe": recipe_id,
                "note": RECIPES[recipe_id]["note"],
                "resume": None if resume is None else str(resume),
            },
            default=str,
        ),
        flush=True,
    )
    trained = train_recipe(device, recipe_id, resume=resume)
    scored = score_trained(recipe_id, trained, parent_scores, phase=phase)
    _drop_cuda(trained.pop("model", None))
    results.setdefault("arms", {})[recipe_id] = {k: v for k, v in scored.items() if k != "best_row"}
    write(OUT / "S5.json", {k: v for k, v in results.items() if k != "promising_row"})
    return scored


def _verdict(best: dict | None) -> str:
    return str((best or {}).get("verdict") or "KILL")


def verify_a(device, recipe_id: str, ckpt: Path, sha: str | None) -> dict:
    print(json.dumps({"phase": "s5_A_verify", "recipe": recipe_id, "checkpoint": str(ckpt)}, default=str), flush=True)
    report = probe_s5(device, ckpt, f"{recipe_id}_verify_a", with_d3=True, with_usable=True, with_relate=False)
    ok, kind = milestone_a(
        report["native_full"],
        report.get("usable"),
        report["compose_full"],
        report.get("sentence"),
        report["who_sent_scores"],
        d3_free=report.get("d3_free"),
    )
    payload = {
        "recipe": recipe_id,
        "checkpoint": str(ckpt),
        "sha256": sha or report.get("checkpoint_sha256"),
        "milestone": ok,
        "milestone_kind": kind if ok else None,
        "why": None if ok else kind,
        "who_sent": report["who_sent_scores"],
        "sentence": report["sentence_scores"],
        "native": report["native"],
        "who_examples": report["who_examples"],
        "bare_examples": report["bare_examples"],
        "d3_free": report.get("d3_free"),
        "usable": None
        if report.get("usable") is None
        else {
            "turn": report["usable"]["autoregressive"]["usable_turn"],
            "turn4": report["usable"]["autoregressive_4turn"]["usable_turn"],
            "stop": report["usable"]["autoregressive_4turn"]["period_stop"],
            "reuse": report["usable"]["autoregressive_4turn"]["fact_reuse"],
        },
        "authoritative": False,
        "promoted": False,
    }
    write(OUT / "s5_A.json", payload)
    campaign_update({"status": "s5 A WHO-sentence MILESTONE (not promoted)" if ok else f"s5 A verify {recipe_id}", "s5_A": payload})
    print(json.dumps({"phase": "s5_A_verify_done", "milestone": ok, "kind": kind, "who_sent": report["who_sent_scores"]}, default=str), flush=True)
    return payload


def verify_b(device, recipe_id: str, ckpt: Path, sha: str | None) -> dict:
    print(json.dumps({"phase": "s5_B_verify", "recipe": recipe_id, "checkpoint": str(ckpt)}, default=str), flush=True)
    report = probe_s5(device, ckpt, f"{recipe_id}_verify_b", with_d3=True, with_usable=True, with_relate=True)
    ok, kind = milestone_b(
        report["native_full"],
        report.get("usable"),
        report["compose_full"],
        report.get("sentence"),
        report["who_sent_scores"],
        report["has_scores"] or {},
        report["beside_scores"] or {},
        d3_free=report.get("d3_free"),
    )
    payload = {
        "recipe": recipe_id,
        "checkpoint": str(ckpt),
        "sha256": sha or report.get("checkpoint_sha256"),
        "milestone": ok,
        "milestone_kind": kind if ok else None,
        "why": None if ok else kind,
        "who_sent": report["who_sent_scores"],
        "has": report["has_scores"],
        "beside": report["beside_scores"],
        "sentence": report["sentence_scores"],
        "native": report["native"],
        "who_examples": report["who_examples"],
        "has_examples": report["has_examples"],
        "beside_examples": report["beside_examples"],
        "bare_examples": report["bare_examples"],
        "d3_free": report.get("d3_free"),
        "usable": None
        if report.get("usable") is None
        else {
            "turn": report["usable"]["autoregressive"]["usable_turn"],
            "turn4": report["usable"]["autoregressive_4turn"]["usable_turn"],
            "stop": report["usable"]["autoregressive_4turn"]["period_stop"],
            "reuse": report["usable"]["autoregressive_4turn"]["fact_reuse"],
        },
        "authoritative": False,
        "promoted": False,
    }
    write(OUT / "s5_B.json", payload)
    campaign_update({"status": "s5 A+B MILESTONE (not promoted)" if ok else f"s5 B verify {recipe_id}", "s5_B": payload})
    print(
        json.dumps(
            {"phase": "s5_B_verify_done", "milestone": ok, "kind": kind, "has": report["has_scores"], "beside": report["beside_scores"]},
            default=str,
        ),
        flush=True,
    )
    return payload


def _phase_loop(
    device,
    *,
    phase: str,
    parent_ckpt: Path,
    parent_scores: dict,
    results: dict,
    first_id: str,
    first_kind: str,
    first_p: float,
    first_note: str,
    seed0: int,
) -> dict | None:
    seed = seed0
    next_id = first_id
    resume: Path | None = None if parent_ckpt == S4M_SURVIVOR and phase == "A" else parent_ckpt
    if phase == "A" and first_id == "s5a":
        resume = None
    pulsed = 0
    locked = 0
    for step in range(MAX_ARMS_A if phase == "A" else MAX_ARMS_B):
        if next_id not in results.get("arms", {}):
            if next_id == first_id:
                _install(next_id, sentence_p=first_p, sent_kind=first_kind, seed=seed, note=first_note, relate=phase == "B")
            scored = _run_arm(device, next_id, parent_scores, results, resume=resume, phase=phase)
        else:
            scored = {"best": (results["arms"][next_id] or {}).get("best"), "last": (results["arms"][next_id] or {}).get("last")}
        best = scored.get("best")
        last = scored.get("last") or best
        verdict = _verdict(best) if best is not None else "KILL"
        last_who = (last or {}).get("who_sent") or {}
        last_has = (last or {}).get("has") or {}
        last_beside = (last or {}).get("beside") or {}
        who_bare = float(last_who.get("bare") or 0.0)
        first_ent = float(last_who.get("first_entity") or 0.0)
        has_bare = float(last_has.get("bare") or 0.0)
        beside_bare = float(last_beside.get("bare") or 0.0)
        primary = who_bare if phase == "A" else min(has_bare, beside_bare) if has_bare and beside_bare else has_bare + beside_bare
        ckpt = Path((best or last)["checkpoint"]) if (best or last) else None
        sha = (best or last or {}).get("checkpoint_sha256")

        if best is not None and verdict in {"HOLD+", "ADVANCE"} and (
            (phase == "A" and who_bare + 1e-12 >= WHO_SENT_GRAD)
            or (phase == "B" and has_bare + 1e-12 >= REL_GRAD and beside_bare + 1e-12 >= REL_GRAD)
        ):
            verified = verify_a(device, next_id, ckpt, sha) if phase == "A" else verify_b(device, next_id, ckpt, sha)
            results.setdefault("verifies", []).append(verified)
            write(OUT / "S5.json", {k: v for k, v in results.items() if k != "promising_row"})
            if verified.get("milestone"):
                return verified
            # Strong canary, shy verify: lock, do not keep pulsing.
            if ckpt is not None and locked < 2:
                seed += 10
                follow = f"{next_id}l"
                _install(
                    follow,
                    sentence_p=0.12,
                    sent_kind="who_lock" if phase == "A" else "rel_lock",
                    seed=seed,
                    note=f"Verify of {next_id} was shy. Lock old+new together.",
                    relate=phase == "B",
                )
                next_id, resume, locked = follow, ckpt, locked + 1
                continue

        if verdict == "KILL" and best is None:
            if pulsed < 2:
                seed += 10
                follow = f"{first_id}{pulsed + 2}"
                _install(
                    follow,
                    sentence_p=0.15,
                    sent_kind=first_kind,
                    seed=seed,
                    note=f"{next_id} killed retention. Lower pulse 15%, steal language/structured only.",
                    relate=phase == "B",
                    language_p=0.24,
                    structured_p=0.24,
                )
                next_id, resume, pulsed = follow, parent_ckpt if phase == "B" else None, pulsed + 1
                continue
            return None

        last_pred = float(last_who.get("predicate") or 0.0)
        last_prefix = float(last_who.get("prefix") or 0.0)
        syntax_spark = last_pred + 1e-12 >= 0.35 or last_prefix + 1e-12 >= 0.20
        if syntax_spark and phase == "A" and who_bare + 1e-12 < WHO_SENT_GRAD and ckpt is not None and locked < 3:
            seed += 10
            follow = f"{next_id}l{locked}"
            _install(
                follow,
                sentence_p=0.12,
                sent_kind="who_lock",
                seed=seed,
                note=f"Syntax spark on {next_id} (pred/prefix) without flying colors. STOP pulse. Lock.",
                relate=False,
            )
            from .selection_stack2 import RECIPES as _R

            _R[follow]["mix"] = "compose_lock_who"
            next_id, resume, locked = follow, ckpt, locked + 1
            continue
        if primary + 1e-12 < WHO_SENT_SIGNAL and not syntax_spark and (phase == "A" and first_ent + 1e-12 < 0.35 or phase == "B"):
            if pulsed < 2:
                seed += 10
                follow = f"{next_id}x"
                heavier = "who_pulse" if phase == "A" else "rel_pulse"
                _install(
                    follow,
                    sentence_p=0.25,
                    sent_kind=heavier,
                    seed=seed,
                    note=f"{next_id} stayed at zero. One more mix-protected 25% pulse, then lock.",
                    relate=phase == "B",
                )
                next_id, resume, pulsed = follow, ckpt or (parent_ckpt if phase == "B" else None), pulsed + 1
                continue
            if locked < 2 and ckpt is not None:
                seed += 10
                follow = f"{next_id}l"
                _install(
                    follow,
                    sentence_p=0.12,
                    sent_kind="who_lock" if phase == "A" else "rel_lock",
                    seed=seed,
                    note=f"No flying-colors signal. Lock rehearsal from {next_id}.",
                    relate=phase == "B",
                )
                next_id, resume, locked = follow, ckpt, locked + 1
                continue
            return None

        if WHO_SENT_SIGNAL <= primary + 1e-12 < (WHO_SENT_GRAD if phase == "A" else REL_GRAD) or (
            phase == "A" and first_ent + 1e-12 >= 0.50 and who_bare + 1e-12 < WHO_SENT_GRAD
        ):
            if locked < 3 and ckpt is not None:
                seed += 10
                follow = f"{next_id}l{locked}"
                _install(
                    follow,
                    sentence_p=0.12,
                    sent_kind="who_lock" if phase == "A" else "rel_lock",
                    seed=seed,
                    note=f"Capability appeared on {next_id}. STOP pulse. Lock old+new.",
                    relate=phase == "B",
                )
                next_id, resume, locked = follow, ckpt, locked + 1
                continue
            if ckpt is not None:
                verified = verify_a(device, next_id, ckpt, sha) if phase == "A" else verify_b(device, next_id, ckpt, sha)
                results.setdefault("verifies", []).append(verified)
                if verified.get("milestone"):
                    return verified
            return None

        if ckpt is not None:
            verified = verify_a(device, next_id, ckpt, sha) if phase == "A" else verify_b(device, next_id, ckpt, sha)
            results.setdefault("verifies", []).append(verified)
            if verified.get("milestone"):
                return verified
        return None
    return None


def run_s5_loop(device) -> dict:
    require_identities(require_s2a=True, require_s2m=True, require_s3s=True, require_s4m=True)
    campaign_update(
        {
            "status": "s5 A+B from s4m: WHO sentences then richer relations. Not promoted.",
            "s4m_parent": {"path": str(S4M_SURVIVOR), "sha256": S4M_SURVIVOR_SHA},
            "hypothesis": (
                "s4m has inverse WHO (one word) and direct-fact sentences separately. "
                "A strong brief WHO-sentence pulse, then lock, should combine them. "
                "B then adds has-object and beside using existing tokens."
            ),
            "authoritative": False,
            "promoted": False,
        }
    )
    parent = probe_s5(device, S4M_SURVIVOR, "s4m_s5_zeroshot", with_relate=True)
    parent_scores = {
        "who_sent_bare": float(parent["who_sent_scores"]["bare"]),
        "bare_sentence": float(parent["sentence_scores"]["bare_sentence"]),
        "who_2e": float(parent["sentence_scores"]["who_2e"]),
    }
    results = {
        "parent": {
            k: parent[k]
            for k in ("sentence_scores", "who_sent_scores", "has_scores", "beside_scores", "native", "d3_slice")
            if k in parent
        },
        "arms": {},
        "verifies": [],
        "milestone_a": False,
        "milestone_b": False,
        "authoritative": False,
        "promoted": False,
    }
    write(OUT / "S5.json", results)

    a = _phase_loop(
        device,
        phase="A",
        parent_ckpt=S4M_SURVIVOR,
        parent_scores=parent_scores,
        results=results,
        first_id="s5a",
        first_kind="who_pulse",
        first_p=0.25,
        first_note="Mix-protected 25% WHO-sentence pulse from s4m. Steal language/structured, keep compose_lock.",
        seed0=326001,
    )
    if a is None or not a.get("milestone"):
        campaign_update({"status": "s5 A did not pass strongly", "s5": {k: results.get(k) for k in ("arms", "verifies")}})
        write(OUT / "S5.json", results)
        print(json.dumps({"phase": "s5_done", "milestone_a": False, "milestone_b": False}, default=str), flush=True)
        return results
    results["milestone_a"] = True
    results["winner_a"] = a
    write(OUT / "S5.json", results)
    print(json.dumps({"phase": "s5_A_passed", "checkpoint": a["checkpoint"], "sha256": a["sha256"]}, default=str), flush=True)

    b_parent = {
        "who_sent_bare": float((a.get("who_sent") or {}).get("bare") or 0.0),
        "bare_sentence": float((a.get("sentence") or {}).get("bare_sentence") or 0.0),
        "who_2e": float((a.get("sentence") or {}).get("who_2e") or 0.0),
    }
    b = _phase_loop(
        device,
        phase="B",
        parent_ckpt=Path(a["checkpoint"]),
        parent_scores=b_parent,
        results=results,
        first_id="s5b",
        first_kind="rel_pulse",
        first_p=0.25,
        first_note="From A survivor: mix-protected 25% has-object + beside pulse. Keep WHO sentences alive.",
        seed0=326101,
    )
    if b is None or not b.get("milestone"):
        campaign_update({"status": "s5 A passed; B did not pass strongly", "s5_A": a, "s5": {k: results.get(k) for k in ("arms", "verifies")}})
        write(OUT / "S5.json", results)
        print(json.dumps({"phase": "s5_done", "milestone_a": True, "milestone_b": False, "A": a.get("checkpoint")}, default=str), flush=True)
        return results
    results["milestone_b"] = True
    results["winner_b"] = b
    write(OUT / "S5.json", results)
    campaign_update({"status": "s5 A+B passed. Not promoted.", "s5_A": a, "s5_B": b})
    print(json.dumps({"phase": "s5_done", "milestone_a": True, "milestone_b": True, "B": b.get("checkpoint")}, default=str), flush=True)
    return results
