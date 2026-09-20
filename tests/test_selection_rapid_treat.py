from __future__ import annotations

import torch

from src.baby_v010.config import BabyVNextConfig, BindingConfig
from src.baby_v010.evaluate import score_items
from src.baby_v010.model import BabyVNextLM
from src.baby_v010.residual_overwrite import LocalSlotOverwrite, attach_overwrite
from src.baby_v010.selection_rapid_treat import (
    POLICY_A1,
    POLICY_A2,
    POLICY_A3,
    POLICY_KIND_BLIND,
    gen_index_for_item,
    greedy_decode,
    score_items_routed,
    should_set_gen_index,
    stage1_verdict,
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


def _hot_overwrite() -> LocalSlotOverwrite:
    overwrite = LocalSlotOverwrite(32, gate_bias=12.0, gen_only=True)
    with torch.no_grad():
        overwrite.scorer.weight.normal_(0, 0.5)
        overwrite.gate.weight.normal_(0, 0.5)
    return overwrite


def test_a1_skips_induction_keeps_keyed_and_kindless() -> None:
    induction = {"kind": "induction", "input": [1, 2, 3, 4], "target": [5]}
    keyed = {"kind": "keyed", "input": [1, 2, 3, 4, 5], "target": [6], "query_position": 1}
    kindless = {"input": [9, 8, 7], "target": [1]}
    assert should_set_gen_index(induction, POLICY_A1) is False
    assert should_set_gen_index(keyed, POLICY_A1) is True
    assert should_set_gen_index(kindless, POLICY_A1) is True
    assert gen_index_for_item(induction, POLICY_A1, first_answer_only=True) is None
    assert gen_index_for_item(keyed, POLICY_A1, first_answer_only=True) == 4
    assert gen_index_for_item(keyed, POLICY_A1, first_answer_only=False) is None
    assert should_set_gen_index(induction, POLICY_KIND_BLIND) is True


def test_a2_requires_long_gap_query() -> None:
    short = {"kind": "keyed", "input": list(range(8)), "query_position": 4, "target": [1]}
    long = {"kind": "keyed", "input": list(range(20)), "query_position": 1, "target": [1]}
    no_q = {"kind": "keyed", "input": list(range(20)), "target": [1]}
    induction = {"kind": "induction", "input": list(range(20)), "query_position": 1, "target": [1]}
    assert should_set_gen_index(short, POLICY_A2) is False
    assert should_set_gen_index(long, POLICY_A2) is True
    assert should_set_gen_index(no_q, POLICY_A2) is False
    assert should_set_gen_index(induction, POLICY_A2) is True


def test_a3_keyed_only() -> None:
    assert should_set_gen_index({"kind": "keyed", "input": [1, 2]}, POLICY_A3) is True
    assert should_set_gen_index({"kind": "induction", "input": [1, 2]}, POLICY_A3) is False
    assert should_set_gen_index({"input": [1, 2]}, POLICY_A3) is False


def test_a1_mixed_batch_induction_identity_keyed_can_write() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    overwrite = _hot_overwrite()
    handle, _ = attach_overwrite(model, overwrite)
    induction = {"kind": "induction", "input": [2, 5, 6, 7, 8], "target": [9, 10]}
    keyed = {"kind": "keyed", "input": [2, 5, 6, 7, 8], "target": [9, 10]}
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
        blind = score_items(model, [induction, keyed], device, overwrite=overwrite, first_answer_only=True)
    finally:
        handle.remove()
    assert routed[0]["emitted"] == off[0]["emitted"]
    assert routed[0]["target_rank"] == off[0]["target_rank"]
    assert routed[1]["emitted"] == blind[1]["emitted"]
    assert routed[1]["emitted"] != off[1]["emitted"] or routed[1]["target_logit"] != off[1]["target_logit"]


def test_a1_greedy_induction_matches_off() -> None:
    torch.manual_seed(1)
    model = _tiny_model()
    overwrite = _hot_overwrite()
    handle, _ = attach_overwrite(model, overwrite)
    induction = {"kind": "induction", "input": [2, 5, 6, 7], "target": [9, 10]}
    keyed = {"kind": "keyed", "input": [2, 5, 6, 7], "target": [9, 10]}
    device = torch.device("cpu")
    try:
        off_ind = greedy_decode(model, induction, device, 2, overwrite=overwrite, first_answer_only=False, policy=POLICY_A1)
        on_ind = greedy_decode(model, induction, device, 2, overwrite=overwrite, first_answer_only=True, policy=POLICY_A1)
        off_key = greedy_decode(model, keyed, device, 2, overwrite=overwrite, first_answer_only=False, policy=POLICY_A1)
        on_key = greedy_decode(model, keyed, device, 2, overwrite=overwrite, first_answer_only=True, policy=POLICY_A1)
    finally:
        handle.remove()
    assert on_ind == off_ind
    assert on_key != off_key


def test_a1_gate_identity_on_induction_write_on_keyed() -> None:
    torch.manual_seed(2)
    overwrite = _hot_overwrite()
    hidden = torch.randn(2, 6, 32)
    induction_idx = gen_index_for_item(
        {"kind": "induction", "input": list(range(6))}, POLICY_A1, first_answer_only=True
    )
    keyed_idx = gen_index_for_item(
        {"kind": "keyed", "input": list(range(6))}, POLICY_A1, first_answer_only=True
    )
    assert induction_idx is None
    assert keyed_idx == 5
    overwrite.gen_index = torch.tensor([keyed_idx])
    out_key = overwrite(hidden[:1])
    assert overwrite.gen_index is None
    assert not torch.allclose(out_key[0, 5], hidden[0, 5], atol=1e-4)
    overwrite.gen_index = None
    out_ind = overwrite(hidden[1:])
    assert torch.allclose(out_ind, hidden[1:], atol=1e-5)


def test_stage1_verdict_advance_and_kill() -> None:
    advance = {
        "long_gap": {"n": 215, "free_exact": 102, "n_overwrite_armed": 215},
        "primitive_induction": {"first_top1": 0.296875, "n_overwrite_armed": 0},
        "primitive_keyed": {"first_top1": 1.0},
        "mechanism": {"long_gap": {"wrote": True}},
    }
    assert stage1_verdict(advance)[0] == "ADVANCE"
    kill_ind = dict(advance)
    kill_ind["primitive_induction"] = {"first_top1": 0.15625, "n_overwrite_armed": 0}
    assert stage1_verdict(kill_ind)[0] == "KILL"
    kill_long = dict(advance)
    kill_long["long_gap"] = {"n": 215, "free_exact": 71, "n_overwrite_armed": 0}
    kill_long["mechanism"] = {"long_gap": {"wrote": False}}
    assert "long-gap" in stage1_verdict(kill_long)[1]
