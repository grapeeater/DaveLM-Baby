from __future__ import annotations

from src.baby_v010.query_presence_d2 import adjudicate_d2, arm_decision, headline


def _report(verdict_fields=None, **long_over) -> dict:
    long_gap = {
        "median_final_residual_cosine": 0.9999,
        "median_full_logit_cosine": 0.9995,
        "flip_toward_donor_final": 0.0,
        "flip_toward_donor_block": [0.0] * 12,
        "flip_toward_donor_attn": [0.0] * 12,
        "fraction_rows_any_query_tracking_head": 0.0,
        "g31_changed_final": 0.0,
    }
    long_gap.update(long_over)
    report = {
        "preflight": {
            "checkpoint_sha_match": True,
            "diagnostic_sha_match": True,
            "protected_material_opened": False,
            "twins_query_only": True,
            "sdpa_reference_max_abs": 1e-5,
            "arm": "treatment",
            "checkpoint_sha256": "abc",
        },
        "positive_control": {
            "median_final_residual_cosine": 0.96,
            "fraction_any_prev_tracking_head": 1.0,
            "eligible_flip_toward_donor_final": 0.32,
            "eligible_n_final": 270,
            "eligible_flip_toward_donor_best_block": 0.32,
            "eligible_n_best_block": 270,
            "flip_toward_donor_final": 0.24,
        },
        "long_gap": long_gap,
    }
    if verdict_fields:
        report["preflight"].update(verdict_fields)
    return report


def test_d2_weights_only_when_treatment_composition_control_B() -> None:
    treat = _report(fraction_rows_any_query_tracking_head=0.80)
    control = _report()
    decision = adjudicate_d2(treat, control)
    assert decision["treatment"]["verdict"] == "COMPOSITION"
    assert decision["control"]["verdict"] == "B"
    assert decision["headline"] == "WEIGHTS_ONLY"
    assert decision["hypothesis"]["H_WEIGHTS"] is True


def test_d2_residual_write_when_treatment_A() -> None:
    treat = _report(flip_toward_donor_final=0.40, median_full_logit_cosine=0.70)
    control = _report()
    decision = adjudicate_d2(treat, control)
    assert decision["headline"] == "RESIDUAL_WRITE"
    assert decision["treatment"]["verdict"] == "A"


def test_d2_still_b_when_both_B() -> None:
    treat = _report()
    control = _report()
    decision = adjudicate_d2(treat, control)
    assert decision["headline"] == "STILL_B"


def test_d2_invalid_if_checkpoint_hash_fails() -> None:
    treat = _report(verdict_fields={"checkpoint_sha_match": False})
    control = _report()
    decision = adjudicate_d2(treat, control)
    assert decision["headline"] == "INVALID"
    assert decision["treatment"]["valid"] is False


def test_d2_headline_helper() -> None:
    assert headline({"valid": True, "verdict": "COMPOSITION"}, {"valid": True, "verdict": "B"}) == "WEIGHTS_ONLY"
    assert headline({"valid": False, "verdict": "A"}, {"valid": True, "verdict": "B"}) == "INVALID"


def test_arm_decision_does_not_reopen_d1b() -> None:
    decision = arm_decision(_report(fraction_rows_any_query_tracking_head=0.80))
    assert decision["d1b_reopened"] is False
    assert decision["protocol"] == "V010_QUERY_PRESENCE_D2"
