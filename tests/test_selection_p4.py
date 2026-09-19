from __future__ import annotations

import pytest
import torch

from src.baby_v010.selection_p4 import copy_specs, decide_verdict, residual_copy_loss


def test_copy_specs_skip_short_gap() -> None:
    items = [
        {"kind": "keyed", "input": list(range(12)), "query_position": 2},
        {"kind": "keyed", "input": list(range(4)), "query_position": 2},
        {"kind": "induction", "input": list(range(12)), "query_position": 2},
    ]
    assert copy_specs(items) == [(0, 11, 2)]


def test_copy_loss_zero_when_gen_matches_query() -> None:
    hidden = torch.zeros(1, 6, 8)
    hidden[0, 5] = torch.ones(8)
    hidden[0, 1] = torch.ones(8)
    loss = residual_copy_loss(hidden, [(0, 5, 1)])
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_copy_loss_stopgrad_query() -> None:
    hidden = torch.randn(1, 5, 4, requires_grad=True)
    loss = residual_copy_loss(hidden, [(0, 4, 1)])
    loss.backward()
    assert hidden.grad is not None
    assert hidden.grad[0, 4].abs().sum() > 0
    assert hidden.grad[0, 1].abs().sum() == 0
    # #region agent log
    from src.baby_v010.selection_p4 import debug_log

    debug_log(
        "B",
        "test_selection_p4.py:test_copy_loss_stopgrad_query",
        "stopgrad_grads",
        {
            "gen_grad_l1": float(hidden.grad[0, 4].abs().sum()),
            "query_grad_l1": float(hidden.grad[0, 1].abs().sum()),
            "loss": float(loss.detach()),
        },
    )
    # #endregion


def test_decide_success() -> None:
    assert (
        decide_verdict(
            treatment_excess=0.12,
            control_excess=0.01,
            ci_lo=0.02,
            treatment_cosine=0.45,
            cosine_gain=0.41,
            regressions=[],
            copy_halved=False,
            negative_ok=True,
        )
        == "SUCCESS"
    )


def test_decide_regression_beats_success() -> None:
    assert (
        decide_verdict(
            treatment_excess=0.12,
            control_excess=0.01,
            ci_lo=0.02,
            treatment_cosine=0.45,
            cosine_gain=0.41,
            regressions=[{"panel": "primitive_induction"}],
            copy_halved=False,
            negative_ok=True,
        )
        == "REGRESSION"
    )


def test_decide_mechanism_and_null_and_falsified() -> None:
    assert (
        decide_verdict(
            treatment_excess=0.01,
            control_excess=0.0,
            ci_lo=-0.01,
            treatment_cosine=0.22,
            cosine_gain=0.18,
            regressions=[],
            copy_halved=False,
            negative_ok=True,
        )
        == "MECHANISM SUPPORTED, DOSE INSUFFICIENT"
    )
    assert (
        decide_verdict(
            treatment_excess=0.0,
            control_excess=0.0,
            ci_lo=-0.01,
            treatment_cosine=0.04,
            cosine_gain=0.01,
            regressions=[],
            copy_halved=False,
            negative_ok=True,
        )
        == "NULL"
    )
    assert (
        decide_verdict(
            treatment_excess=0.0,
            control_excess=0.0,
            ci_lo=-0.01,
            treatment_cosine=0.04,
            cosine_gain=0.0,
            regressions=[],
            copy_halved=True,
            negative_ok=True,
        )
        == "FALSIFIED"
    )
