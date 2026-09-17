from __future__ import annotations

from src.baby_v010.selection_p5 import decide_verdict


def test_decide_success() -> None:
    assert (
        decide_verdict(
            treatment_excess=0.12,
            control_excess=0.01,
            ci_lo=0.02,
            treatment_cosine=0.45,
            cosine_gain=0.41,
            treatment_mass=0.9,
            regressions=[],
            ptr_halved=False,
            negative_ok=True,
        )
        == "SUCCESS"
    )


def test_decide_regression() -> None:
    assert (
        decide_verdict(
            treatment_excess=0.12,
            control_excess=0.01,
            ci_lo=0.02,
            treatment_cosine=0.45,
            cosine_gain=0.41,
            treatment_mass=0.9,
            regressions=[{"panel": "primitive_induction"}],
            ptr_halved=False,
            negative_ok=True,
        )
        == "REGRESSION"
    )


def test_decide_mechanism() -> None:
    assert (
        decide_verdict(
            treatment_excess=0.01,
            control_excess=0.0,
            ci_lo=-0.01,
            treatment_cosine=0.05,
            cosine_gain=0.01,
            treatment_mass=0.6,
            regressions=[],
            ptr_halved=False,
            negative_ok=True,
        )
        == "MECHANISM SUPPORTED, DOSE INSUFFICIENT"
    )
