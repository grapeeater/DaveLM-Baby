from __future__ import annotations

import pytest
import torch

from src.baby_v010.selection_p2 import (
    bind_fell_half,
    bind_specs,
    decide_verdict,
    residual_bind_loss,
    retention_failures,
)


def test_bind_specs_skip_induction_and_short_gap() -> None:
    items = [
        {
            "kind": "keyed",
            "input": list(range(10)),
            "query_position": 2,
            "body_key_positions": [0, 4],
        },
        {
            "kind": "induction",
            "input": list(range(10)),
            "query_position": 2,
            "body_key_positions": [0, 4],
        },
        {
            "kind": "keyed",
            "input": list(range(4)),
            "query_position": 2,
            "body_key_positions": [0, 1],
        },
    ]
    assert bind_specs(items) == [(0, 9, [2, 0, 4])]


def test_residual_bind_loss_is_near_zero_when_gen_matches_query() -> None:
    hidden = torch.zeros(1, 6, 8)
    hidden[0, 5] = torch.ones(8)
    hidden[0, 1] = torch.ones(8)
    hidden[0, 3] = -torch.ones(8)
    loss = residual_bind_loss(hidden, [(0, 5, [1, 3])], tau=0.10)
    assert float(loss) == pytest.approx(0.0, abs=1e-5)


def test_residual_bind_loss_is_high_when_gen_matches_competitor() -> None:
    hidden = torch.zeros(1, 6, 8)
    hidden[0, 5] = torch.ones(8)
    hidden[0, 1] = -torch.ones(8)
    hidden[0, 3] = torch.ones(8)
    loss = residual_bind_loss(hidden, [(0, 5, [1, 3])], tau=0.10)
    assert float(loss) > 2.0


def test_residual_bind_stopgrad_keys() -> None:
    hidden = torch.randn(1, 5, 4, requires_grad=True)
    loss = residual_bind_loss(hidden, [(0, 4, [1, 2, 3])], tau=0.10)
    loss.backward()
    assert hidden.grad is not None
    assert hidden.grad[0, 4].abs().sum() > 0
    assert hidden.grad[0, 1].abs().sum() == pytest.approx(0.0, abs=1e-8)
    assert hidden.grad[0, 2].abs().sum() == pytest.approx(0.0, abs=1e-8)
    assert hidden.grad[0, 3].abs().sum() == pytest.approx(0.0, abs=1e-8)


def test_bind_fell_half() -> None:
    assert bind_fell_half([1.0] * 50 + [0.4] * 50) is True
    assert bind_fell_half([1.0] * 50 + [0.9] * 50) is False
    assert bind_fell_half([1.0] * 40) is False


def test_decide_verdict_success_and_regression() -> None:
    success = decide_verdict(
        treatment_excess=0.12,
        control_excess=0.01,
        ci_lo=0.02,
        treatment_residual=0.98,
        parent_residual=0.9996,
        regressions=[],
        bind_halved=False,
        negative_ok=True,
    )
    assert success == "SUCCESS"
    regression = decide_verdict(
        treatment_excess=0.12,
        control_excess=0.01,
        ci_lo=0.02,
        treatment_residual=0.98,
        parent_residual=0.9996,
        regressions=[{"panel": "primitive_induction"}],
        bind_halved=False,
        negative_ok=True,
    )
    assert regression == "REGRESSION"


def test_decide_verdict_falsified_null_mechanism() -> None:
    falsified = decide_verdict(
        treatment_excess=0.0,
        control_excess=0.0,
        ci_lo=-0.02,
        treatment_residual=0.9995,
        parent_residual=0.9996,
        regressions=[],
        bind_halved=True,
        negative_ok=True,
    )
    assert falsified == "FALSIFIED"
    null = decide_verdict(
        treatment_excess=0.0,
        control_excess=0.0,
        ci_lo=-0.02,
        treatment_residual=0.9995,
        parent_residual=0.9996,
        regressions=[],
        bind_halved=False,
        negative_ok=True,
    )
    assert null == "NULL"
    mechanism = decide_verdict(
        treatment_excess=0.01,
        control_excess=0.0,
        ci_lo=-0.01,
        treatment_residual=0.98,
        parent_residual=0.9996,
        regressions=[],
        bind_halved=False,
        negative_ok=True,
    )
    assert mechanism == "MECHANISM SUPPORTED, DOSE INSUFFICIENT"


def test_retention_failures_trips_induction_drop() -> None:
    parent = {
        "summary": {"rest_lock": 0.98},
        "frozen": {
            "summaries": {
                "primitive_induction": {"first_top1": 0.30},
                "primitive_keyed": {"first_top1": 1.0},
                "short_keyed": {"free_exact": 0.84},
            }
        },
    }
    treat = {
        "summary": {"rest_lock": 0.98},
        "frozen": {
            "summaries": {
                "primitive_induction": {"first_top1": 0.18},
                "primitive_keyed": {"first_top1": 1.0},
                "short_keyed": {"free_exact": 0.83},
            }
        },
    }
    failures = retention_failures(parent, treat)
    assert len(failures) == 1
    assert failures[0]["panel"] == "primitive_induction"
