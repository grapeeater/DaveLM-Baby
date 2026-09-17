from __future__ import annotations

import pytest

from src.baby_v010.query_locality import (
    ARMS,
    bucket,
    build_arm,
    candidate_rank,
    gap_of,
    other_key,
    render_keys,
)

SEP = 90
KEY_POOL = tuple(range(700, 760))
SPAN_POOL = tuple(range(200, 260))


def item(query_index: int = 0) -> dict:
    # body renders pair A (key 701 -> value 210,211) then pair B (702 -> 220,221).
    # `source` carries the pre-shuffle record order, which is what parse_records
    # reads and what candidate_heads / query_index are aligned to.
    body = [701, 210, 211, SEP, 702, 220, 221, SEP]
    inp = [2, 800, *body, 801, 701 + query_index, 802]
    return {
        "input": inp,
        "target": [210, 211, SEP, 3] if query_index == 0 else [220, 221, SEP, 3],
        "target_span": [210, 211] if query_index == 0 else [220, 221],
        "source": [701, 210, 211, 702, 220, 221],
        "kind": "keyed",
        "variant": "keyed_0",
        "query_key": 701 + query_index,
        "query_index": query_index,
        "query_position": len(inp) - 2,
        "pair_count": 2,
        "value_length": 2,
        "candidate_heads": [210, 220],
        "body_id": "T0",
    }


def build(arm: str, row: dict, seed: int = 7):
    return build_arm(row, arm, key_pool=KEY_POOL, span_pool=SPAN_POOL, seed=seed)


def test_gap_of_counts_tokens_after_query() -> None:
    assert gap_of(item()) == 1


def test_render_keys_are_in_render_order() -> None:
    assert render_keys(item()) == [701, 702]


def test_as_is_is_byte_identical() -> None:
    row = item()
    built = build("as_is", row)
    assert built["input"] == row["input"]
    assert built["gold_index"] == 0


def test_append_query_makes_gap_zero_and_keeps_gold() -> None:
    row = item()
    built = build("append_query", row)
    assert built["input"] == row["input"] + [701]
    assert built["gold_index"] == 0


def test_append_other_key_retargets_gold_to_that_pair() -> None:
    row = item(query_index=0)
    built = build("append_other_key", row)
    assert built["input"][-1] == 702
    # the appended key belongs to the second rendered pair
    assert built["gold_index"] == 1


def test_other_key_is_none_when_single_pair() -> None:
    row = item()
    row["input"] = [2, 800, 701, 210, 211, SEP, 801, 701, 802]
    row["query_position"] = 7
    row["candidate_heads"] = [210]
    row["source"] = [701, 210, 211]
    row["pair_count"] = 1
    assert other_key(row) is None
    assert build("append_other_key", row) is None


@pytest.mark.parametrize("arm,count", [("append_filler_key_8", 8), ("append_filler_key_32", 32)])
def test_filler_arms_extend_the_gap_by_the_named_count(arm: str, count: int) -> None:
    row = item()
    built = build(arm, row)
    assert len(built["input"]) == len(row["input"]) + count
    assert gap_of({**row, "input": built["input"]}) == gap_of(row) + count
    assert built["gold_index"] == 0


def test_filler_arms_never_reuse_tokens_already_in_the_item() -> None:
    row = item()
    present = set(row["input"]) | set(row["target"])
    for arm in ("append_filler_key_8", "append_filler_span_8", "append_unused_key"):
        built = build(arm, row)
        added = built["input"][len(row["input"]) :]
        assert not (set(added) & present), arm


def test_span_filler_comes_from_the_span_bank_and_key_filler_from_the_key_bank() -> None:
    row = item()
    assert set(build("append_filler_span_8", row)["input"][len(row["input"]) :]) <= set(SPAN_POOL)
    assert set(build("append_filler_key_8", row)["input"][len(row["input"]) :]) <= set(KEY_POOL)


def test_every_arm_preserves_the_original_prefix_and_inventory() -> None:
    row = item()
    for arm in ARMS:
        built = build(arm, row)
        assert built["input"][: len(row["input"])] == row["input"], arm


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_query_then_filler_places_query_first_then_n_nuisance_tokens(count: int) -> None:
    row = item()
    built = build(f"append_query_then_filler_{count}", row)
    tail = built["input"][len(row["input"]) :]
    assert tail[0] == row["query_key"]
    assert len(tail) == count + 1
    assert not (set(tail[1:]) & (set(row["input"]) | set(row["target"])))
    assert built["gold_index"] == row["query_index"]


def test_query_then_filler_zero_distance_case_matches_append_query() -> None:
    row = item()
    assert build("append_query_then_filler_1", row)["input"][: len(row["input"]) + 1] == (
        build("append_query", row)["input"]
    )


def test_unknown_arm_rejected() -> None:
    with pytest.raises(ValueError):
        build("nonsense", item())


def test_candidate_rank_is_one_based_and_counts_strict_betters() -> None:
    assert candidate_rank([1.0, 3.0, 2.0], 1) == 1
    assert candidate_rank([1.0, 3.0, 2.0], 2) == 2
    assert candidate_rank([1.0, 3.0, 2.0], 0) == 3


def test_bucket_boundaries() -> None:
    assert [bucket(g) for g in (0, 1, 2, 3, 4, 12, 13, 30, 31, 99)] == [
        "a_gap_0_1",
        "a_gap_0_1",
        "b_gap_2_3",
        "b_gap_2_3",
        "c_gap_4_12",
        "c_gap_4_12",
        "d_gap_13_30",
        "d_gap_13_30",
        "e_gap_31_plus",
        "e_gap_31_plus",
    ]
