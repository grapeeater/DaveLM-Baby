from __future__ import annotations

import pytest

from src.baby_v010.selection_t1 import (
    GAP_VARIANTS,
    MIN_GAP,
    allowed_gap,
    arm_items,
    body_length,
    insertion_index,
    natural_gap,
    set_gap,
    sorted_multiset,
)

SEP = 90
BOS = 2
BODY = [701, 210, 211, SEP, 702, 220, 221, SEP]  # K=2, value_length=2
FILLER_A = [750, 751, 752]
FILLER_B = [760, 761, 762, 763, 764]


def base(variant: str) -> dict:
    # The query key also opens the body, so the query slot is located from the
    # tail, exactly as `locate_query` does in production by excluding the
    # occurrence that is followed by the value span.
    if variant == "keyed_1":
        inp = [BOS, 800, *FILLER_A, *BODY, 801, 701, *FILLER_B]
        query_position = len(inp) - len(FILLER_B) - 1
    elif variant == "keyed_3":
        inp = [BOS, 800, *BODY, *FILLER_A, 701, 801, *FILLER_B]
        query_position = len(inp) - len(FILLER_B) - 2
    elif variant == "keyed_4":
        inp = [BOS, *BODY, 800, *FILLER_A, 801, 701, *FILLER_B]
        query_position = len(inp) - len(FILLER_B) - 1
    else:
        raise AssertionError(variant)
    assert inp[query_position] == 701
    return {
        "input": inp,
        "target": [210, 211, SEP, 3],
        "target_span": [210, 211],
        "source": [701, 210, 211, 702, 220, 221],
        "kind": "keyed",
        "difficulty": "full",
        "variant": variant,
        "query_key": 701,
        "query_index": 0,
        "query_position": query_position,
        "pair_count": 2,
        "value_length": 2,
        "candidate_heads": [210, 220],
    }


def test_body_length_matches_the_rendered_layout() -> None:
    assert body_length(base("keyed_1")) == len(BODY)


@pytest.mark.parametrize("variant", GAP_VARIANTS)
def test_natural_gap_counts_the_trailing_filler(variant: str) -> None:
    expected = len(FILLER_B) + (1 if variant == "keyed_3" else 0)
    assert natural_gap(base(variant)) == expected


@pytest.mark.parametrize("variant", GAP_VARIANTS)
def test_insertion_index_lands_inside_the_templates_own_filler_a(variant: str) -> None:
    row = base(variant)
    index = insertion_index(row)
    assert 1 <= index <= row["query_position"]
    # inserting there must not split the contiguous body
    body_start = row["input"].index(701)
    assert index <= body_start or index >= body_start + len(BODY)


@pytest.mark.parametrize("variant", GAP_VARIANTS)
def test_set_gap_hits_the_target_and_preserves_everything_else(variant: str) -> None:
    row = base(variant)
    for target in range(MIN_GAP[variant], natural_gap(row) + 1):
        moved = set_gap(row, target)
        assert moved is not None, (variant, target)
        assert natural_gap(moved) == target
        assert len(moved["input"]) == len(row["input"])
        assert sorted_multiset(moved["input"]) == sorted_multiset(row["input"])
        assert moved["input"][moved["query_position"]] == row["query_key"]
        assert moved["target"] == row["target"]
        assert moved["target_span"] == row["target_span"]
        assert moved["candidate_heads"] == row["candidate_heads"]
        assert moved["input"][0] == BOS


@pytest.mark.parametrize("variant", GAP_VARIANTS)
def test_set_gap_keeps_the_body_contiguous_and_the_queried_pair_unique(variant: str) -> None:
    row = base(variant)
    pattern = [701, 210, 211, SEP]
    for target in range(MIN_GAP[variant], natural_gap(row) + 1):
        moved = set_gap(row, target)["input"]
        hits = sum(moved[i : i + len(pattern)] == pattern for i in range(len(moved)))
        assert hits == 1, (variant, target)
        assert sum(moved[i : i + len(BODY)] == BODY for i in range(len(moved))) == 1


@pytest.mark.parametrize("variant", GAP_VARIANTS)
def test_set_gap_rejects_unreachable_targets(variant: str) -> None:
    row = base(variant)
    assert set_gap(row, MIN_GAP[variant] - 1) is None
    assert set_gap(row, natural_gap(row) + 1) is None


def test_set_gap_refuses_variants_without_a_free_gap_parameter() -> None:
    row = base("keyed_1")
    row["variant"] = "keyed_0"
    assert set_gap(row, 0) is None
    row["variant"] = "keyed_5"
    assert set_gap(row, 0) is None


def test_set_gap_is_a_copy_and_does_not_mutate_the_input_item() -> None:
    row = base("keyed_1")
    before = list(row["input"])
    set_gap(row, 0)
    assert row["input"] == before


def test_allowed_gap_follows_the_frozen_blocks() -> None:
    assert allowed_gap(1) == 1
    assert allowed_gap(150) == 1
    assert allowed_gap(151) == 3
    assert allowed_gap(300) == 3
    assert allowed_gap(301) == 8
    assert allowed_gap(450) == 8
    assert allowed_gap(451) == 20
    assert allowed_gap(600) == 20
    assert allowed_gap(601) is None
    assert allowed_gap(800) is None


def test_control_arm_uses_the_items_untouched() -> None:
    row = base("keyed_1")
    row["t1_gap"] = 0
    spec = {"items": [row]}
    assert arm_items(spec, "control")[0]["input"] == row["input"]


def test_treatment_arm_applies_the_scheduled_gap() -> None:
    row = base("keyed_1")
    row["t1_gap"] = 0
    out = arm_items({"items": [row]}, "treatment")[0]
    assert natural_gap(out) == 0
    assert sorted_multiset(out["input"]) == sorted_multiset(row["input"])


def test_treatment_arm_passes_through_rows_with_no_scheduled_gap() -> None:
    row = base("keyed_4")
    row["t1_gap"] = None
    out = arm_items({"items": [row]}, "treatment")[0]
    assert out["input"] == row["input"]


def test_both_arms_see_the_same_token_multiset_and_target_for_every_row() -> None:
    rows = []
    for variant in GAP_VARIANTS:
        row = base(variant)
        row["t1_gap"] = MIN_GAP[variant]
        rows.append(row)
    spec = {"items": rows}
    control = arm_items(spec, "control")
    treatment = arm_items(spec, "treatment")
    for c, t in zip(control, treatment):
        assert sorted_multiset(c["input"]) == sorted_multiset(t["input"])
        assert c["target"] == t["target"]
        assert natural_gap(t) <= natural_gap(c)
