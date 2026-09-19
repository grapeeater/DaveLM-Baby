from __future__ import annotations

import random

import torch

from src.baby_v010.data_language_bridge import load_tokenizer, build_s3_panels
from src.baby_v010.selection_rapid_treat_d import production_d_uses_query_position, tiling_parse
from src.baby_v010.selection_stack2 import (
    RECIPES,
    S2A_D3_SLICE,
    S2A_SURVIVOR,
    S2I25_SURVIVOR,
    S2I50_SURVIVOR,
    S2M_SURVIVOR,
    S3S_SURVIVOR,
    adjudicate_recover,
    english_native_holds,
    mix_gate,
    mix_holds_s2a,
    parent_checkpoint,
    remainder_span_mask,
    sample_s2_item,
    stack2_adjudicate,
    usable_holds,
    usable_holds_s2a,
)


def test_d3_operator_still_gold_free() -> None:
    assert production_d_uses_query_position() is False


def test_english_drop_and_collapse() -> None:
    hold = {
        "qa_2fact_heldout": {"first_top1": 0.94},
        "size_stop_heldout": {"free_exact": 0.88},
        "story_color_heldout": {"first_top1": 0.91},
    }
    assert english_native_holds(hold)[0] is True
    drop = dict(hold)
    drop["qa_2fact_heldout"] = {"first_top1": 0.80}
    ok, lesson, collapsed = english_native_holds(drop)
    assert ok is False and collapsed is False
    assert "drop" in lesson
    dead = dict(hold)
    dead["qa_2fact_heldout"] = {"first_top1": 0.40}
    dead["size_stop_heldout"] = {"free_exact": 0.20}
    ok, lesson, collapsed = english_native_holds(dead)
    assert ok is False and collapsed is True


def test_d3_199_does_not_kill_if_english_holds() -> None:
    native = {
        "qa_2fact_heldout": {"first_top1": 0.94},
        "size_stop_heldout": {"free_exact": 0.91},
        "story_color_heldout": {"first_top1": 0.91},
        "mixed_2e_heldout": {"first_top1": 0.53},
        "fact_combine_heldout": {"first_top1": 0.53},
        "story_combine_heldout": {"first_top1": 0.25},
        "story_mixed_heldout": {"first_top1": 0.41},
        "dialogue_2fact_heldout": {"first_top1": 0.94},
    }
    usable = {
        "autoregressive": {"usable_turn": 0.84, "period_stop": 1.0, "fact_reuse": 1.0, "rambling": 0.0},
        "autoregressive_4turn": {"usable_turn": 0.88, "period_stop": 1.0, "fact_reuse": 1.0, "rambling": 0.0},
    }
    verdict, lesson, major = stack2_adjudicate(native, usable=usable, d3_free=199, induction=0.42)
    assert verdict == "SURVIVE"
    assert major is True
    assert "199" in lesson
    assert mix_gate(native)[0] == "ADVANCE"


def test_english_death_kills_even_if_mix_moves() -> None:
    native = {
        "qa_2fact_heldout": {"first_top1": 0.40},
        "size_stop_heldout": {"free_exact": 0.20},
        "story_color_heldout": {"first_top1": 0.30},
        "mixed_2e_heldout": {"first_top1": 0.72},
        "fact_combine_heldout": {"first_top1": 0.62},
        "story_combine_heldout": {"first_top1": 0.56},
        "story_mixed_heldout": {"first_top1": 0.56},
    }
    verdict, _lesson, major = stack2_adjudicate(native, d3_free=202, induction=0.42)
    assert verdict == "KILL"
    assert major is False


def test_usable_collapse() -> None:
    usable = {
        "autoregressive": {"usable_turn": 0.40, "period_stop": 0.50, "fact_reuse": 0.20, "rambling": 0.4},
        "autoregressive_4turn": {"usable_turn": 0.40, "period_stop": 0.50, "fact_reuse": 0.20, "rambling": 0.4},
    }
    ok, lesson, collapsed = usable_holds(usable)
    assert ok is False and collapsed is True
    assert "collapse" in lesson


def test_s2_mix_items_are_gold_free() -> None:
    tokenizer = load_tokenizer()
    rng = random.Random(322001)
    families = []
    for mix in ("protect40_combine", "e13_replica", "combine_heavy"):
        for _ in range(40):
            item = sample_s2_item(rng, tokenizer, mix)
            assert item.get("query_position") is None
            assert "query_position" not in item["input"]
            families.append(item["family"])
            gen = len(item["input"]) - 1
            tiling_parse(item["input"], gen)
    assert "english_mixed" in families
    assert "fact_combine" in families
    assert "story_combine" in families or "story_mixed" in families


def _s2a_native(**overrides):
    native = {
        "qa_2fact_heldout": {"first_top1": 0.96875},
        "size_stop_heldout": {"free_exact": 1.0},
        "story_color_heldout": {"first_top1": 1.0},
        "mixed_2e_heldout": {"first_top1": 0.875},
        "fact_combine_heldout": {"first_top1": 0.65625},
        "story_combine_heldout": {"first_top1": 0.4375},
        "story_mixed_heldout": {"first_top1": 0.5},
        "dialogue_2fact_heldout": {"first_top1": 1.0},
    }
    native.update(overrides)
    return native


def test_s2a_mix_hold_and_drop() -> None:
    assert mix_holds_s2a(_s2a_native())[0] is True
    drop = _s2a_native(mixed_2e_heldout={"first_top1": 0.80})
    assert mix_holds_s2a(drop)[0] is False


def test_adjudicate_recover_advance_hold_kill() -> None:
    hold = {"native": _s2a_native(), "d3_slice": {"free_exact": S2A_D3_SLICE, "first_top1": 1.0}}
    assert adjudicate_recover(hold)[0] == "HOLD"
    plus = {"native": _s2a_native(), "d3_slice": {"free_exact": S2A_D3_SLICE + 0.05, "first_top1": 1.0}}
    assert adjudicate_recover(plus)[0] == "HOLD+"
    advance = {"native": _s2a_native(), "d3_slice": {"free_exact": 0.90, "first_top1": 1.0}}
    assert adjudicate_recover(advance)[0] == "ADVANCE"
    dead = {"native": _s2a_native(qa_2fact_heldout={"first_top1": 0.40}), "d3_slice": {"free_exact": 0.95, "first_top1": 1.0}}
    assert adjudicate_recover(dead)[0] == "KILL"


def test_remainder_span_mask_skips_first_target() -> None:
    mask = torch.tensor(
        [
            [False, True, True, True, False],
            [False, True, False, False, False],
            [True, True, False, False, False],
        ]
    )
    out = remainder_span_mask(mask)
    assert out[0].tolist() == [False, False, True, True, False]
    assert out[1].tolist() == [False, True, False, False, False]
    assert out[2].tolist() == [False, True, False, False, False]


def test_mixhold_lock_parents() -> None:
    assert parent_checkpoint(RECIPES["s2o"]) == S2I50_SURVIVOR
    assert parent_checkpoint(RECIPES["s2q"]) == S2I50_SURVIVOR
    assert parent_checkpoint(RECIPES["s2n"]) == S2I25_SURVIVOR
    assert parent_checkpoint(RECIPES["s2l"]) == S2A_SURVIVOR
    assert RECIPES["s2o"]["lock_from_drop"] is True
    assert RECIPES["s2o"]["language_p"] == 0.0
    assert RECIPES["s2o"]["structured_p"] == 0.0
    assert RECIPES["s2p"]["mix_remainder_span"] is True


def test_s3_compose_parents_and_items() -> None:
    assert parent_checkpoint(RECIPES["s3a"]) == S2M_SURVIVOR
    assert parent_checkpoint(RECIPES["s3b"]) == S2M_SURVIVOR
    assert RECIPES["s3a"]["compose"] is True
    tokenizer = load_tokenizer()
    rng = random.Random(324001)
    families = [sample_s2_item(rng, tokenizer, "compose_bind")["family"] for _ in range(40)]
    assert "who_bind" in families
    protect = [sample_s2_item(rng, tokenizer, "compose_bind_protect")["family"] for _ in range(50)]
    assert "who_bind" in protect
    lock = [sample_s2_item(rng, tokenizer, "compose_lock")["family"] for _ in range(60)]
    assert "who_bind" in lock
    assert parent_checkpoint(RECIPES["s3i"]) == S2M_SURVIVOR
    panels = build_s3_panels(tokenizer, seed=324001, n=32)
    golds = [item["value_text"] for item in panels["who_bind_2e_heldout"]]
    assert max(golds.count(g) for g in set(golds)) <= 8
    families_s = [sample_s2_item(rng, tokenizer, "compose_sent")["family"] for _ in range(40)]
    assert any(name.startswith("compose_sent") or name.startswith("compose_combine") for name in families_s)


def test_s3_mixed3_already_solved_is_not_a_win() -> None:
    from src.baby_v010.selection_stack2_s3 import adjudicate_compose

    parent = {"who_2e": 0.0, "who_3e": 0.0, "mixed_3e": 0.9375, "combine_3e": 0.25, "bare_sentence": 0.0, "sent_exact": 0.0, "prefix_sentence": 0.0}
    native = {
        "qa_2fact_heldout": {"first_top1": 0.97},
        "size_stop_heldout": {"free_exact": 1.0},
        "story_color_heldout": {"first_top1": 1.0},
        "mixed_2e_heldout": {"first_top1": 0.875},
        "fact_combine_heldout": {"first_top1": 0.719},
        "story_combine_heldout": {"first_top1": 0.562},
    }
    compose = {
        "who_bind_2e_heldout": {"first_top1": 0.0},
        "who_bind_3e_heldout": {"first_top1": 0.0},
        "mixed_3e_heldout": {"first_top1": 0.9375},
        "fact_combine_3e_heldout": {"first_top1": 0.25},
        "compose_sent_heldout": {"free_exact": 0.0},
    }
    row = {"native": native, "compose": compose, "d3_slice": {"free_exact": 0.925}}
    verdict, lesson = adjudicate_compose(row, parent=parent)
    assert verdict == "HOLD"
    assert "composition signal" not in lesson
    who_row = {
        "native": native,
        "compose": {**compose, "who_bind_2e_heldout": {"first_top1": 0.3125}},
        "d3_slice": {"free_exact": 0.925},
    }
    who_verdict, who_lesson = adjudicate_compose(who_row, parent=parent)
    assert who_verdict == "HOLD+"
    assert "composition signal" in who_lesson


def test_usable_holds_s2a() -> None:
    usable = {
        "autoregressive": {"usable_turn": 0.952, "period_stop": 1.0, "fact_reuse": 1.0, "rambling": 0.0},
        "autoregressive_4turn": {"usable_turn": 0.969, "period_stop": 1.0, "fact_reuse": 1.0, "rambling": 0.0},
    }
    assert usable_holds_s2a(usable)[0] is True
    drop = {
        "autoregressive": {"usable_turn": 0.80, "period_stop": 1.0, "fact_reuse": 1.0, "rambling": 0.0},
        "autoregressive_4turn": {"usable_turn": 0.80, "period_stop": 1.0, "fact_reuse": 1.0, "rambling": 0.0},
    }
    assert usable_holds_s2a(drop)[0] is False


def test_s4_sentence_parents_and_items() -> None:
    from src.baby_v010.selection_stack2_s4 import adjudicate_sentence, sample_s4_item

    assert parent_checkpoint(RECIPES["s4a"]) == S3S_SURVIVOR
    assert parent_checkpoint(RECIPES["s4c"]) == S3S_SURVIVOR
    assert RECIPES["s4a"]["sentence_p"] == 0.08
    assert RECIPES["s4c"]["sentence_p"] == 0.15
    assert RECIPES["s4a"]["sent_kind"] == "easy"
    tokenizer = load_tokenizer()
    rng = random.Random(325001)
    easy = sample_s4_item(rng, tokenizer, "easy")
    assert easy.get("query_position") is None
    assert easy["n_entities"] == 1
    assert easy["family"].startswith("compose_sent")
    bind = sample_s4_item(rng, tokenizer, "bind")
    assert bind["n_entities"] == 2
    native = {
        "qa_2fact_heldout": {"first_top1": 0.97},
        "size_stop_heldout": {"free_exact": 1.0},
        "story_color_heldout": {"first_top1": 1.0},
        "mixed_2e_heldout": {"first_top1": 0.97},
        "fact_combine_heldout": {"first_top1": 0.656},
        "story_combine_heldout": {"first_top1": 0.59},
    }
    compose = {
        "who_bind_2e_heldout": {"first_top1": 0.50},
        "who_bind_3e_heldout": {"first_top1": 0.47},
        "mixed_3e_heldout": {"first_top1": 0.94},
        "fact_combine_3e_heldout": {"first_top1": 0.22},
        "compose_sent_heldout": {"free_exact": 0.0},
        "compose_sent_1e_heldout": {"free_exact": 0.0},
    }
    parent = {"bare_sentence": 0.0, "sent_1e_exact": 0.0, "who_2e": 0.50, "who_3e": 0.47}
    zero = {
        "native": native,
        "compose": compose,
        "sentence": {"operators": {"bare": {"sentence_ok": 0.0, "one_word_color": 1.0}, "sent_prefix": {"sentence_ok": 0.0}}},
        "d3_slice": {"free_exact": 0.875},
    }
    assert adjudicate_sentence(zero, parent=parent)[0] == "HOLD"
    signal = {
        **zero,
        "sentence": {"operators": {"bare": {"sentence_ok": 0.375, "one_word_color": 0.5}, "sent_prefix": {"sentence_ok": 0.375}}},
        "compose": {**compose, "compose_sent_1e_heldout": {"free_exact": 0.31}},
    }
    assert adjudicate_sentence(signal, parent=parent)[0] == "ADVANCE"
    scaffold = {
        **zero,
        "sentence": {"operators": {"bare": {"sentence_ok": 0.0, "one_word_color": 1.0}, "sent_prefix": {"sentence_ok": 0.50}}},
    }
    assert adjudicate_sentence(scaffold, parent=parent)[0] == "HOLD"
    dead = {**zero, "native": {**native, "qa_2fact_heldout": {"first_top1": 0.40}}}
    assert adjudicate_sentence(dead, parent=parent)[0] == "KILL"


def test_s4_direct_fact_sentence_milestone_allows_style_mix() -> None:
    from src.baby_v010.selection_stack2_s4 import sentence_milestone

    native = {
        "qa_2fact_heldout": {"first_top1": 0.75},
        "size_stop_heldout": {"free_exact": 1.0},
        "story_color_heldout": {"first_top1": 0.91},
        "mixed_2e_heldout": {"first_top1": 0.8125},
        "fact_combine_heldout": {"first_top1": 0.6875},
        "story_combine_heldout": {"first_top1": 0.50},
    }
    compose = {
        "who_bind_2e_heldout": {"first_top1": 0.50},
        "who_bind_3e_heldout": {"first_top1": 0.50},
        "mixed_3e_heldout": {"first_top1": 0.94},
        "compose_sent_heldout": {"free_exact": 0.0},
        "compose_sent_1e_heldout": {"free_exact": 0.0},
    }
    sentence = {"operators": {"bare": {"sentence_ok": 0.375, "one_word_color": 0.375}, "sent_prefix": {"sentence_ok": 0.125}}}
    usable = {
        "autoregressive": {"usable_turn": 0.857, "period_stop": 1.0, "fact_reuse": 1.0, "rambling": 0.0},
        "autoregressive_4turn": {"usable_turn": 0.906, "period_stop": 1.0, "fact_reuse": 1.0, "rambling": 0.0},
    }
    ok, kind = sentence_milestone(185, native, usable, compose, sentence)
    assert ok is True
    assert kind == "direct-fact-sentence"
    dead = dict(native)
    dead["qa_2fact_heldout"] = {"first_top1": 0.40}
    assert sentence_milestone(185, dead, usable, compose, sentence)[0] is False
