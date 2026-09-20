from __future__ import annotations

import torch

from src.baby_v010.selection_rapid_treat import POLICY_A1, should_set_gen_index
from src.baby_v010.selection_rapid_treat_c import token_identity_src
from src.baby_v010.selection_rapid_treat_d import (
    C2_FIRST,
    C2_LONG,
    MASK_INVENTORY,
    MASK_MATCHED,
    apply_head_mask,
    d_canary_verdict,
    production_d_uses_query_position,
    tiling_matched_head,
    tiling_parse,
    tiling_value_heads,
)


def test_production_d_src_does_not_read_gold_fields() -> None:
    assert production_d_uses_query_position() is False
    assert "query_position" not in tiling_parse.__code__.co_names
    assert "candidate_heads" not in tiling_parse.__code__.co_names
    assert "query_position" not in tiling_value_heads.__code__.co_names
    assert "query_position" not in tiling_matched_head.__code__.co_names


def test_tiling_matches_c2_and_value_heads() -> None:
    # body records (key, v, v, SEP=7) twice, then unpaired key 10 as query
    tokens = [2, 9, 10, 20, 21, 7, 11, 22, 23, 7, 8, 10, 12]
    gen = 12
    parsed = tiling_parse(tokens, gen)
    assert parsed is not None
    assert parsed["query_src"] == token_identity_src(tokens, gen) == 11
    assert parsed["value_heads"] == [20, 22]
    assert tiling_value_heads(tokens, gen) == [20, 22]
    assert tiling_matched_head(tokens, gen) == 20
    assert parsed["matched_pos"] == 3


def test_apply_head_mask_forces_inventory_argmax() -> None:
    logits = torch.tensor([0.1, 4.0, 0.2, 3.0, 5.0])
    masked = apply_head_mask(logits, [1, 3])
    assert int(masked.argmax().item()) == 1
    assert torch.isneginf(masked[4])
    assert int(apply_head_mask(logits, [3]).argmax().item()) == 3


def test_d_canary_verdict_requires_jump_past_c2() -> None:
    base = {
        "long_gap": {"n": 215, "free_exact": 122, "first_correct": 126},
        "primitive_induction": {"first_top1": 0.296875},
        "primitive_keyed": {"first_top1": 0.96875},
        "query_swap_same_surface_novel": {"first_top1": 0.6666666666666666},
    }
    assert d_canary_verdict(base)[0] == "KILL"
    held = dict(base)
    held["long_gap"] = {"n": 215, "free_exact": C2_LONG, "first_correct": C2_FIRST}
    assert d_canary_verdict(held)[0] == "KILL"
    jump = dict(base)
    jump["long_gap"] = {"n": 215, "free_exact": 141, "first_correct": 200}
    assert d_canary_verdict(jump)[0] == "GRAD"
    mid = dict(base)
    mid["long_gap"] = {"n": 215, "free_exact": 136, "first_correct": 140}
    assert d_canary_verdict(mid)[0] == "ADVANCE"
    bind = dict(base)
    bind["long_gap"] = {"n": 215, "free_exact": 128, "first_correct": 130}
    bind["query_swap_same_surface_novel"] = {"first_top1": 0.80}
    assert d_canary_verdict(bind)[0] == "ADVANCE"


def test_d1_d3_do_not_arm_induction() -> None:
    induction = {"kind": "induction", "input": list(range(6)), "target": [9]}
    keyed = {"kind": "keyed", "input": list(range(6)), "target": [9]}
    assert should_set_gen_index(induction, POLICY_A1) is False
    assert should_set_gen_index(keyed, POLICY_A1) is True
    # mask heads are empty without a keyed tiling; induction prompts must not
    # accidentally look like a unique (key,value,SEP) inventory.
    assert tiling_parse(induction["input"], 5) is None


def test_mask_modes_are_distinct() -> None:
    tokens = [2, 9, 10, 20, 21, 7, 11, 22, 23, 7, 8, 10, 12]
    assert MASK_INVENTORY != MASK_MATCHED
    assert tiling_value_heads(tokens, 12) == [20, 22]
    assert tiling_matched_head(tokens, 12) == 20
