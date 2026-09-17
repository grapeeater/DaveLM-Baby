from __future__ import annotations

import torch

from src.baby_v010.config import BabyVNextConfig, BindingConfig
from src.baby_v010.evaluate import score_items
from src.baby_v010.model import BabyVNextLM
from src.baby_v010.residual_overwrite import LocalSlotOverwrite, attach_overwrite
from src.baby_v010.selection_p11_u16000_runtime import (
    decide_verdict,
    greedy_decode,
    language_ce_fail,
    rest_lock_mean,
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


def test_decide_pass_and_regression() -> None:
    assert decide_verdict(benefit_ok=True, off_ok=True, on_ok=True) == ("PASS", "none")
    assert decide_verdict(benefit_ok=True, off_ok=True, on_ok=False) == (
        "REGRESSION",
        "overwrite_on_first_step",
    )
    assert decide_verdict(benefit_ok=False, off_ok=True, on_ok=True) == ("STOP_NO_BENEFIT", "none")
    assert decide_verdict(benefit_ok=True, off_ok=False, on_ok=True) == ("BLOCKED", "off_baseline_invalid")


def test_greedy_first_answer_only_can_differ_from_off() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    overwrite = LocalSlotOverwrite(32, gate_bias=12.0, gen_only=True)
    # Trainable overwrite weights so ON arm can diverge from inert OFF.
    with torch.no_grad():
        overwrite.scorer.weight.normal_(0, 0.5)
        overwrite.gate.weight.normal_(0, 0.5)
    handle, _ = attach_overwrite(model, overwrite)
    item = {"input": [2, 5, 6, 7, 8], "target": [9, 10, 11]}
    device = torch.device("cpu")
    try:
        off = greedy_decode(model, item, device, 3, overwrite=overwrite, first_answer_only=False)
        on = greedy_decode(model, item, device, 3, overwrite=overwrite, first_answer_only=True)
    finally:
        handle.remove()
    assert len(off) == 3
    assert len(on) == 3
    assert on[0] != off[0] or on != off


def test_overwrite_off_matches_no_hook_greedy() -> None:
    torch.manual_seed(1)
    model = _tiny_model()
    overwrite = LocalSlotOverwrite(32, gate_bias=8.0, gen_only=True)
    item = {"input": [2, 5, 6, 7], "target": [9, 10]}
    device = torch.device("cpu")
    plain = greedy_decode(model, item, device, 2)
    handle, _ = attach_overwrite(model, overwrite)
    try:
        off = greedy_decode(model, item, device, 2, overwrite=overwrite, first_answer_only=False)
    finally:
        handle.remove()
    assert off == plain


def test_language_ce_hard_stop() -> None:
    assert language_ce_fail(1.24, 1.43) is False
    assert language_ce_fail(1.24, 1.45) is True


def test_rest_lock_mean() -> None:
    rows = [{"all_ranks": [1, 1, 1, 2]}, {"all_ranks": [1, 2, 1, 1]}]
    items = [{"target_span": [0, 0, 0]}, {"target_span": [0, 0, 0]}]
    assert rest_lock_mean(rows, items) == 0.5


def test_first_answer_only_score_items() -> None:
    torch.manual_seed(2)
    model = _tiny_model()
    overwrite = LocalSlotOverwrite(32, gate_bias=8.0, gen_only=True)
    handle, _ = attach_overwrite(model, overwrite)
    item = {
        "input": [2, 5, 6, 7, 8],
        "target": [9, 10, 11],
        "kind": "keyed",
        "target_span": [9, 10],
    }
    device = torch.device("cpu")
    try:
        off = score_items(model, [item], device, overwrite=overwrite, first_answer_only=False)[0]
        on = score_items(model, [item], device, overwrite=overwrite, first_answer_only=True)[0]
    finally:
        handle.remove()
    assert off["emitted"] == score_items(model, [item], device)[0]["emitted"]
