from __future__ import annotations

import torch

from src.baby_v010.config import BabyVNextConfig, BindingConfig
from src.baby_v010.evaluate import score_items
from src.baby_v010.model import BabyVNextLM
from src.baby_v010.residual_overwrite import LocalSlotOverwrite, attach_overwrite
from src.baby_v010.selection_rapid_treat import (
    POLICY_A1,
    gen_index_for_item,
    greedy_decode,
    score_items_routed,
    should_set_gen_index,
)
from src.baby_v010.selection_rapid_treat_b import (
    WRITE_GATED,
    WRITE_HARD,
    WRITE_MIX,
    WRITE_CONF,
    TreatedOverwrite,
    b_canary_verdict,
)


def _tiny_model() -> BabyVNextLM:
    config = BabyVNextConfig(
        vocab_size=32,
        context_length=32,
        d_model=32,
        n_heads=4,
        n_layers=1,
        d_mlp=64,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        residual_dropout=0.0,
        binding=BindingConfig(retrieval_dim=16),
    )
    model = BabyVNextLM(config)
    model.eval()
    return model


def _peaked_overwrite(src: int = 2) -> TreatedOverwrite:
    """Scorer peaks at `src` via hidden[:, src, 0] (not query_position)."""
    overwrite = TreatedOverwrite(32, gate_bias=12.0, gen_only=True)
    with torch.no_grad():
        overwrite.scorer.weight.zero_()
        overwrite.scorer.bias.zero_()
        overwrite.scorer.weight[0, 0] = 10.0
        overwrite.gate.weight.zero_()
        overwrite.gate.bias.fill_(12.0)
    overwrite.set_eval_write(WRITE_HARD)
    return overwrite


def test_b1_induction_gets_no_write() -> None:
    induction = {"kind": "induction", "input": list(range(6)), "target": [9], "query_position": 1}
    assert should_set_gen_index(induction, POLICY_A1) is False
    assert gen_index_for_item(induction, POLICY_A1, first_answer_only=True) is None
    overwrite = _peaked_overwrite(2)
    hidden = torch.zeros(1, 6, 32)
    hidden[0, 2, 0] = 5.0
    for t in range(6):
        hidden[0, t, 1] = float(t + 1)
    overwrite.gen_index = None
    out = overwrite(hidden)
    assert torch.allclose(out, hidden, atol=1e-6)
    assert overwrite.last_src_index is None


def test_b1_keyed_uses_pointer_argmax_not_query_position() -> None:
    query_position = 1
    pointer_src = 2
    gen = 5
    overwrite = _peaked_overwrite(pointer_src)
    hidden = torch.zeros(1, 6, 32)
    hidden[0, pointer_src, 0] = 5.0
    hidden[0, query_position, 0] = 1.0
    for t in range(6):
        hidden[0, t, 1:] = float(t + 1)
    overwrite.gen_index = torch.tensor([gen])
    out = overwrite(hidden)
    assert overwrite.last_src_index is not None
    assert int(overwrite.last_src_index[0].item()) == pointer_src
    assert int(overwrite.last_src_index[0].item()) != query_position
    assert torch.allclose(out[0, gen], hidden[0, pointer_src], atol=1e-5)
    assert not torch.allclose(out[0, gen], hidden[0, query_position], atol=1e-4)
    assert torch.allclose(out[0, :gen], hidden[0, :gen], atol=1e-5)


def test_b1_skip_index_is_identity() -> None:
    overwrite = _peaked_overwrite(2)
    hidden = torch.randn(2, 6, 32)
    hidden = hidden.clone()
    hidden[0, 2, 0] = 8.0
    hidden[1, 2, 0] = 8.0
    overwrite.gen_index = torch.tensor([-1, 5])
    out = overwrite(hidden)
    assert torch.allclose(out[0], hidden[0], atol=1e-5)
    assert torch.allclose(out[1, 5], hidden[1, 2], atol=1e-4)


def test_gated_t1_matches_local_slot_overwrite() -> None:
    torch.manual_seed(0)
    parent = LocalSlotOverwrite(32, gate_bias=12.0, gen_only=True)
    treated = TreatedOverwrite(32, gate_bias=12.0, gen_only=True)
    treated.load_state_dict(parent.state_dict())
    treated.set_eval_write(WRITE_GATED, temperature=1.0)
    hidden = torch.randn(2, 8, 32)
    parent.gen_index = torch.tensor([7, 6])
    treated.gen_index = torch.tensor([7, 6])
    out_p = parent(hidden)
    out_t = treated(hidden)
    assert torch.allclose(out_p, out_t, atol=1e-6)


def test_mix_beta_endpoints() -> None:
    torch.manual_seed(1)
    overwrite = TreatedOverwrite(32, gate_bias=12.0, gen_only=True)
    with torch.no_grad():
        overwrite.scorer.weight.normal_(0, 0.5)
        overwrite.gate.bias.fill_(12.0)
    hidden = torch.randn(1, 6, 32)
    overwrite.set_eval_write(WRITE_GATED, 1.0, 0.0)
    overwrite.gen_index = torch.tensor([5])
    gated = overwrite(hidden)
    overwrite.set_eval_write(WRITE_HARD)
    overwrite.gen_index = torch.tensor([5])
    hard = overwrite(hidden)
    overwrite.set_eval_write(WRITE_MIX, mix_beta=0.0)
    overwrite.gen_index = torch.tensor([5])
    mix0 = overwrite(hidden)
    overwrite.set_eval_write(WRITE_MIX, mix_beta=1.0)
    overwrite.gen_index = torch.tensor([5])
    mix1 = overwrite(hidden)
    assert torch.allclose(mix0, gated, atol=1e-5)
    assert torch.allclose(mix1, hard, atol=1e-5)


def test_conf_hard_uses_hard_only_when_peak_high() -> None:
    pointer_src = 2
    gen = 5
    overwrite = _peaked_overwrite(pointer_src)
    hidden = torch.zeros(1, 6, 32)
    hidden[0, pointer_src, 0] = 5.0
    hidden[0, 1, 0] = 1.0
    for t in range(6):
        hidden[0, t, 1:] = float(t + 1)
    overwrite.set_eval_write(WRITE_HARD)
    overwrite.gen_index = torch.tensor([gen])
    hard = overwrite(hidden)
    overwrite.set_eval_write(WRITE_GATED)
    overwrite.gen_index = torch.tensor([gen])
    gated = overwrite(hidden)
    overwrite.set_eval_write(WRITE_CONF, conf_tau=0.0)
    overwrite.gen_index = torch.tensor([gen])
    always = overwrite(hidden)
    overwrite.set_eval_write(WRITE_CONF, conf_tau=1.1)
    overwrite.gen_index = torch.tensor([gen])
    never = overwrite(hidden)
    assert torch.allclose(always, hard, atol=1e-5)
    assert torch.allclose(never, gated, atol=1e-5)


def test_a1_plus_hard_greedy_induction_matches_off() -> None:
    torch.manual_seed(4)
    model = _tiny_model()
    overwrite = _peaked_overwrite(1)
    with torch.no_grad():
        overwrite.scorer.weight.normal_(0, 0.4)
    handle, _ = attach_overwrite(model, overwrite)
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
        handle.remove()
    assert on_ind == off_ind
    assert on_key != off_key


def test_a1_mixed_batch_hard_replace() -> None:
    torch.manual_seed(4)
    model = _tiny_model()
    overwrite = _peaked_overwrite(1)
    with torch.no_grad():
        overwrite.scorer.weight.normal_(0, 0.4)
    handle, _ = attach_overwrite(model, overwrite)
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
        handle.remove()
    assert routed[0]["emitted"] == off[0]["emitted"]
    assert routed[0]["target_rank"] == off[0]["target_rank"]
    assert routed[1]["emitted"] != off[1]["emitted"] or routed[1]["target_logit"] != off[1]["target_logit"]


def test_b_canary_verdict_bars() -> None:
    base = {
        "long_gap": {"n": 215, "free_exact": 102},
        "primitive_induction": {"first_top1": 0.296875},
        "primitive_keyed": {"first_top1": 0.984375},
        "query_swap_same_surface_novel": {"first_top1": 0.427},
    }
    assert b_canary_verdict(base)[0] == "KILL"
    lift = dict(base)
    lift["long_gap"] = {"n": 215, "free_exact": 112}
    assert b_canary_verdict(lift)[0] == "ADVANCE"
    owner = dict(base)
    owner["long_gap"] = {"n": 215, "free_exact": 118}
    assert b_canary_verdict(owner)[0] == "OWNER"
    bind = dict(base)
    bind["query_swap_same_surface_novel"] = {"first_top1": 0.48}
    assert b_canary_verdict(bind)[0] == "ADVANCE"
    dead = dict(base)
    dead["long_gap"] = {"n": 215, "free_exact": 71}
    assert b_canary_verdict(dead)[0] == "KILL"
