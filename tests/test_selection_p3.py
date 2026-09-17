from __future__ import annotations

from src.baby_v010.selection_p3 import bind_specs, competitor_key_positions, decide_verdict


def _item() -> dict:
    return {
        "kind": "keyed",
        "input": [2, 701, 210, 211, 90, 702, 220, 221, 90, 701, 800, 801],
        "query_key": 701,
        "query_index": 0,
        "query_position": 9,
        "pair_count": 2,
        "candidate_heads": [210, 220],
        "source": [701, 210, 211, 702, 220, 221],
        "target": [210, 211, 90, 3],
        "target_span": [210, 211],
        "value_length": 2,
        "variant": "keyed_0",
    }


def test_competitor_keys_exclude_matching_token() -> None:
    item = _item()
    assert competitor_key_positions(item) == [5]
    assert bind_specs([item]) == [(0, 11, [9, 5])]


def test_bind_specs_skip_short_gap() -> None:
    item = _item()
    item["input"] = item["input"][:11]
    item["query_position"] = 9
    assert bind_specs([item], min_gap=2) == []


def test_decide_success_and_regression() -> None:
    success = decide_verdict(
        treatment_excess=0.12,
        control_excess=0.01,
        ci_lo=0.02,
        cosine_gain=0.15,
        early_track=0.1,
        early_track_gain=0.05,
        regressions=[],
        bind_halved=False,
        negative_ok=True,
    )
    assert success == "SUCCESS"
    regression = decide_verdict(
        treatment_excess=0.12,
        control_excess=0.01,
        ci_lo=0.02,
        cosine_gain=0.15,
        early_track=0.1,
        early_track_gain=0.05,
        regressions=[{"panel": "primitive_induction"}],
        bind_halved=False,
        negative_ok=True,
    )
    assert regression == "REGRESSION"


def test_decide_mechanism() -> None:
    out = decide_verdict(
        treatment_excess=0.01,
        control_excess=0.0,
        ci_lo=-0.01,
        cosine_gain=0.06,
        early_track=0.1,
        early_track_gain=0.05,
        regressions=[],
        bind_halved=False,
        negative_ok=True,
    )
    assert out == "MECHANISM SUPPORTED, DOSE INSUFFICIENT"
