from __future__ import annotations

import pytest

from src.baby_v010.query_splice_d3 import decide, eligible_item, splice_sources


def _cell(accuracy: float) -> dict:
    return {"accuracy": accuracy, "n": 100, "hit": int(round(accuracy * 100))}


def _item() -> dict:
    return {
        "kind": "keyed",
        "input": [2, 701, 210, 211, 90, 702, 220, 221, 90, 701, 800],
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


def test_splice_sources_finds_query_match_and_competitor() -> None:
    sources = splice_sources(_item())
    assert sources["query"] == 9
    assert sources["match_key"] == 1
    assert sources["competitor_key"] == 5
    assert eligible_item(_item()) is True


def test_eligible_item_rejects_missing_query() -> None:
    item = _item()
    del item["query_position"]
    assert eligible_item(item) is False


def _cells(**overrides) -> dict:
    base = {
        "as_is_long": _cell(0.34),
        "Q0R": _cell(0.34),
        "Q1R": _cell(0.34),
        "Q2R": _cell(0.34),
        "Q11R": _cell(0.34),
        "Q0M": _cell(0.34),
        "Q1M": _cell(0.34),
        "Q11M": _cell(0.34),
        "K0R": _cell(0.34),
        "K11R": _cell(0.34),
        "C11R": _cell(0.34),
        "QE": _cell(0.34),
    }
    base.update(overrides)
    return base


def test_decide_invalid_when_steer_fails() -> None:
    out = decide(_cells(), steer_miss_acc=0.5, n_miss=80)
    assert out["verdict"] == "INVALID"


def test_decide_early_query() -> None:
    out = decide(_cells(Q1R=_cell(0.50)), steer_miss_acc=1.0, n_miss=80)
    assert out["verdict"] == "EARLY_QUERY"
    assert out["early_query"] == pytest.approx(0.16)


def test_decide_late_query() -> None:
    out = decide(_cells(Q11R=_cell(0.50)), steer_miss_acc=1.0, n_miss=80)
    assert out["verdict"] == "LATE_QUERY"


def test_decide_both_query() -> None:
    out = decide(_cells(Q0R=_cell(0.50), Q11M=_cell(0.48)), steer_miss_acc=1.0, n_miss=80)
    assert out["verdict"] == "BOTH_QUERY"


def test_decide_match_key() -> None:
    out = decide(_cells(K11R=_cell(0.55)), steer_miss_acc=1.0, n_miss=80)
    assert out["verdict"] == "MATCH_KEY"


def test_decide_none() -> None:
    out = decide(_cells(), steer_miss_acc=1.0, n_miss=80)
    assert out["verdict"] == "NONE"


def test_decide_mixed_when_competitor_helps_gold() -> None:
    out = decide(_cells(C11R=_cell(0.45)), steer_miss_acc=1.0, n_miss=80)
    assert out["verdict"] == "MIXED"
