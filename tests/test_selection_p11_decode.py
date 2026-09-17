from __future__ import annotations

from src.baby_v010.selection_p11_decode import MIN_GAP, SUCCESS_DELTA, long_items


def test_decode_protocol_identity() -> None:
    assert MIN_GAP == 13
    assert SUCCESS_DELTA == 0.10
    assert long_items([{"input": [0] * 20, "query_position": 0}, {"input": [0] * 5, "query_position": 3}]) == [
        {"input": [0] * 20, "query_position": 0}
    ]
