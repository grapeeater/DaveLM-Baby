from __future__ import annotations

import torch

from src.baby_v010.config import BabyVNextConfig, BindingConfig
from src.baby_v010.evaluate import score_items
from src.baby_v010.model import BabyVNextLM
from src.baby_v010.residual_overwrite import attach_overwrite
from src.baby_v010.selection_rapid_treat import (
    POLICY_A1,
    gen_index_for_item,
    greedy_decode,
    score_items_routed,
    should_set_gen_index,
)
from src.baby_v010.selection_rapid_treat_b import WRITE_GATED, WRITE_HARD
from src.baby_v010.selection_rapid_treat_c import (
    LOCATOR_AGREE,
    LOCATOR_HYBRID,
    LOCATOR_L0,
    LOCATOR_PREV2,
    LOCATOR_SLOT,
    LocatorOverwrite,
    LocatorRuntime,
    c_canary_verdict,
    production_forward_uses_query_position,
    src_from_attn,
    token_identity_src,
)


def _tiny_model(n_layers: int = 1) -> BabyVNextLM:
    config = BabyVNextConfig(
        vocab_size=32,
        context_length=32,
        d_model=32,
        n_heads=4,
        n_layers=n_layers,
        d_mlp=64,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        binding=BindingConfig(retrieval_dim=16),
    )
    model = BabyVNextLM(config)
    model.eval()
    return model


def _peaked_slot(src: int = 2) -> LocatorOverwrite:
    overwrite = LocatorOverwrite(32, gate_bias=12.0, gen_only=True)
    with torch.no_grad():
        overwrite.scorer.weight.zero_()
        overwrite.scorer.bias.zero_()
        overwrite.scorer.weight[0, 0] = 10.0
        overwrite.gate.weight.zero_()
        overwrite.gate.bias.fill_(12.0)
    overwrite.set_eval_write(WRITE_HARD, locator=LOCATOR_SLOT)
    return overwrite


def _inject_l0(overwrite: LocatorOverwrite, *, batch: int, heads: int, time: int, gen: int, src: int) -> None:
    weights = torch.zeros(batch, heads, time, time)
    for step in range(time):
        weights[:, :, step, : step + 1] = 1.0 / (step + 1)
    weights[:, :, gen, :] = 0.0
    weights[:, :, gen, src] = 1.0
    overwrite.attn_map.weights = weights


def test_production_src_does_not_read_query_position() -> None:
    assert production_forward_uses_query_position() is False
    assert "query_position" not in token_identity_src.__code__.co_names
    assert "query_position" not in src_from_attn.__code__.co_names


def test_c1_induction_gets_no_write() -> None:
    induction = {"kind": "induction", "input": list(range(6)), "target": [9], "query_position": 1}
    assert should_set_gen_index(induction, POLICY_A1) is False
    assert gen_index_for_item(induction, POLICY_A1, first_answer_only=True) is None
    overwrite = _peaked_slot(2)
    overwrite.set_eval_write(WRITE_HARD, locator=LOCATOR_L0)
    hidden = torch.zeros(1, 6, 32)
    hidden[0, 2, 0] = 5.0
    _inject_l0(overwrite, batch=1, heads=4, time=6, gen=5, src=3)
    overwrite.gen_index = None
    out = overwrite(hidden)
    assert torch.allclose(out, hidden, atol=1e-6)
    assert overwrite.last_src_index is None


def test_c1b_uses_l0_argmax_not_query_position() -> None:
    query_position = 1
    slot_src = 2
    l0_src = 3
    gen = 5
    overwrite = _peaked_slot(slot_src)
    overwrite.set_eval_write(WRITE_HARD, locator=LOCATOR_L0)
    hidden = torch.zeros(1, 6, 32)
    hidden[0, slot_src, 0] = 5.0
    hidden[0, query_position, 0] = 1.0
    hidden[0, l0_src, 1] = 9.0
    for t in range(6):
        hidden[0, t, 2:] = float(t + 1)
    _inject_l0(overwrite, batch=1, heads=4, time=6, gen=gen, src=l0_src)
    overwrite.gen_index = torch.tensor([gen])
    out = overwrite(hidden)
    assert int(overwrite.last_src_index[0].item()) == l0_src
    assert int(overwrite.last_src_index[0].item()) != query_position
    assert int(overwrite.last_slot_src[0].item()) == slot_src
    assert torch.allclose(out[0, gen], hidden[0, l0_src], atol=1e-5)
    assert not torch.allclose(out[0, gen], hidden[0, query_position], atol=1e-4)
    assert not torch.allclose(out[0, gen], hidden[0, slot_src], atol=1e-4)


def test_c1c_high_mass_keeps_slot_low_mass_uses_l0() -> None:
    slot_src = 2
    l0_src = 3
    gen = 5
    hidden = torch.zeros(1, 6, 32)
    hidden[0, slot_src, 0] = 5.0
    hidden[0, l0_src, 1] = 7.0
    for t in range(6):
        hidden[0, t, 2:] = float(t + 1)

    peaked = _peaked_slot(slot_src)
    peaked.set_eval_write(WRITE_HARD, locator=LOCATOR_HYBRID, conf_tau=0.3)
    _inject_l0(peaked, batch=1, heads=4, time=6, gen=gen, src=l0_src)
    peaked.gen_index = torch.tensor([gen])
    high = peaked(hidden)
    assert int(peaked.last_src_index[0].item()) == slot_src
    assert torch.allclose(high[0, gen], hidden[0, slot_src], atol=1e-4)

    flat = LocatorOverwrite(32, gate_bias=12.0, gen_only=True)
    with torch.no_grad():
        flat.scorer.weight.zero_()
        flat.scorer.bias.zero_()
        flat.gate.bias.fill_(12.0)
    flat.set_eval_write(WRITE_HARD, locator=LOCATOR_HYBRID, conf_tau=0.3)
    _inject_l0(flat, batch=1, heads=4, time=6, gen=gen, src=l0_src)
    flat.gen_index = torch.tensor([gen])
    low = flat(hidden)
    assert float(flat.last_pointer_mass[0].item()) < 0.3
    assert int(flat.last_src_index[0].item()) == l0_src
    assert torch.allclose(low[0, gen], hidden[0, l0_src], atol=1e-4)


def test_c1d_agree_hard_else_gated() -> None:
    slot_src = 2
    l0_src = 3
    gen = 5
    overwrite = _peaked_slot(slot_src)
    hidden = torch.zeros(1, 6, 32)
    hidden[0, slot_src, 0] = 5.0
    hidden[0, l0_src, 1] = 7.0
    for t in range(6):
        hidden[0, t, 2:] = float(t + 1)

    overwrite.set_eval_write(WRITE_HARD)
    overwrite.locator = LOCATOR_SLOT
    overwrite.gen_index = torch.tensor([gen])
    hard = overwrite(hidden)
    overwrite.set_eval_write(WRITE_GATED, locator=LOCATOR_SLOT)
    overwrite.gen_index = torch.tensor([gen])
    gated = overwrite(hidden)

    overwrite.set_eval_write(WRITE_HARD, locator=LOCATOR_AGREE)
    _inject_l0(overwrite, batch=1, heads=4, time=6, gen=gen, src=slot_src)
    overwrite.gen_index = torch.tensor([gen])
    agreed = overwrite(hidden)
    assert bool(overwrite.last_agree[0].item()) is True
    assert torch.allclose(agreed, hard, atol=1e-5)

    _inject_l0(overwrite, batch=1, heads=4, time=6, gen=gen, src=l0_src)
    overwrite.gen_index = torch.tensor([gen])
    disagreed = overwrite(hidden)
    assert bool(overwrite.last_agree[0].item()) is False
    assert torch.allclose(disagreed, gated, atol=1e-5)
    assert int(overwrite.last_src_index[0].item()) != 1


def test_c1a_is_gen_minus_two() -> None:
    overwrite = _peaked_slot(2)
    overwrite.set_eval_write(WRITE_HARD, locator=LOCATOR_PREV2)
    hidden = torch.zeros(1, 6, 32)
    for t in range(6):
        hidden[0, t] = float(t + 1)
    overwrite.gen_index = torch.tensor([5])
    out = overwrite(hidden)
    assert int(overwrite.last_src_index[0].item()) == 3
    assert torch.allclose(out[0, 5], hidden[0, 3], atol=1e-5)


def test_token_identity_later_of_unique_double() -> None:
    # body records (key, v, v, SEP=7) twice, then unpaired key 10 as query
    tokens = [2, 9, 10, 20, 21, 7, 11, 22, 23, 7, 8, 10, 12]
    gen = 12
    src = token_identity_src(tokens, gen)
    assert src == 11
    assert tokens[src] == 10
    assert src != 2
    # two different unpaired keys → not unique, refuse
    noisy = [2, 9, 10, 20, 21, 7, 11, 22, 23, 7, 8, 10, 11]
    assert token_identity_src(noisy, 13) is None


def test_a1_plus_c1b_greedy_induction_matches_off() -> None:
    torch.manual_seed(4)
    model = _tiny_model()
    overwrite = _peaked_slot(1)
    overwrite.set_eval_write(WRITE_HARD, locator=LOCATOR_L0)
    handle, _ = attach_overwrite(model, overwrite)
    runtime = LocatorRuntime(model, overwrite).install()
    induction = {"kind": "induction", "input": [2, 5, 6, 7], "target": [9, 10], "query_position": 0}
    keyed = {"kind": "keyed", "input": [2, 5, 6, 7], "target": [9, 10], "query_position": 0}
    device = torch.device("cpu")
    try:
        off_ind = greedy_decode(
            model, induction, device, 2, overwrite=overwrite, first_answer_only=False, policy=POLICY_A1
        )
        on_ind = greedy_decode(
            model, induction, device, 2, overwrite=overwrite, first_answer_only=True, policy=POLICY_A1
        )
        off_key = greedy_decode(
            model, keyed, device, 2, overwrite=overwrite, first_answer_only=False, policy=POLICY_A1
        )
        on_key = greedy_decode(
            model, keyed, device, 2, overwrite=overwrite, first_answer_only=True, policy=POLICY_A1
        )
    finally:
        runtime.remove()
        handle.remove()
    assert on_ind == off_ind
    assert on_key != off_key


def test_a1_mixed_batch_c1_induction_identity() -> None:
    torch.manual_seed(4)
    model = _tiny_model()
    overwrite = _peaked_slot(1)
    overwrite.set_eval_write(WRITE_HARD, locator=LOCATOR_L0)
    handle, _ = attach_overwrite(model, overwrite)
    runtime = LocatorRuntime(model, overwrite).install()
    induction = {"kind": "induction", "input": [2, 5, 6, 7, 8], "target": [9, 10], "query_position": 0}
    keyed = {"kind": "keyed", "input": [2, 5, 6, 7, 8], "target": [9, 10], "query_position": 0}
    device = torch.device("cpu")
    try:
        off = score_items(model, [induction, keyed], device, overwrite=overwrite, first_answer_only=False)
        routed = score_items_routed(
            model,
            [induction, keyed],
            device,
            overwrite=overwrite,
            first_answer_only=True,
            policy=POLICY_A1,
        )
    finally:
        runtime.remove()
        handle.remove()
    assert routed[0]["emitted"] == off[0]["emitted"]
    assert routed[0]["target_rank"] == off[0]["target_rank"]


def test_c_canary_verdict_bars_vs_b1() -> None:
    base = {
        "long_gap": {"n": 215, "free_exact": 109},
        "primitive_induction": {"first_top1": 0.296875},
        "primitive_keyed": {"first_top1": 0.96875},
        "query_swap_same_surface_novel": {"first_top1": 0.4375},
    }
    assert c_canary_verdict(base)[0] == "KILL"
    lift = dict(base)
    lift["long_gap"] = {"n": 215, "free_exact": 112}
    assert c_canary_verdict(lift)[0] == "ADVANCE"
    owner = dict(base)
    owner["long_gap"] = {"n": 215, "free_exact": 118}
    assert c_canary_verdict(owner)[0] == "OWNER"
    bind = dict(base)
    bind["query_swap_same_surface_novel"] = {"first_top1": 0.48}
    assert c_canary_verdict(bind)[0] == "ADVANCE"
    dead = dict(base)
    dead["long_gap"] = {"n": 215, "free_exact": 102}
    assert c_canary_verdict(dead)[0] == "KILL"
