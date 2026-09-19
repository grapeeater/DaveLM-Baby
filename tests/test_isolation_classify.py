from __future__ import annotations

from src.baby_v010.isolation_classify import (
    PAYLOAD_OFF_INVENTORY,
    PAYLOAD_ORIGINAL,
    PAYLOAD_OTHER_COMPETITOR,
    PAYLOAD_QUERIED_NEW,
    STAGE_CANNOT_IDENTIFY,
    STAGE_PAYLOAD_SUFFIX_FAIL,
    STAGE_SELECTED_NO_CONTINUE,
    STAGE_SUCCESS,
    STAGE_WEAK_IDENTIFY,
    classify_payload_origin,
    inventory_first_token_snapshot,
    mechanism_stage,
    summarize_scored,
    token_rank,
)


def _keyed_item(**updates) -> dict:
    item = {
        "kind": "keyed",
        "pair_count": 3,
        "value_length": 2,
        "query_key": 10,
        "target_span": [21, 22],
        "target": [21, 22, 76, 3],
        "source": [10, 21, 22, 11, 31, 32, 12, 41, 42],
        "input": [2, 10, 21, 22, 76, 11, 31, 32, 76, 12, 41, 42, 76, 10],
        "variant": "keyed_0",
        "isolation_transform": "query_swap",
        "original_query_key": 11,
        "original_target_span": [31, 32],
    }
    item.update(updates)
    return item


def test_token_rank_matches_strict_greater_convention():
    logits = [0.1, 4.0, 3.0, 4.0]
    assert token_rank(logits, 1) == 1
    assert token_rank(logits, 3) == 1
    assert token_rank(logits, 2) == 3
    assert token_rank(logits, 0) == 4


def test_payload_origin_splits_new_old_other_and_off():
    item = _keyed_item()
    follow = classify_payload_origin(item, {"emitted": [21, 22, 76, 3], "target_rank": 1, "all_ranks": [1, 1, 1, 1]})
    stuck = classify_payload_origin(item, {"emitted": [31, 32, 76, 3], "target_rank": 2, "all_ranks": [2, 1, 1, 1]})
    other = classify_payload_origin(item, {"emitted": [41, 42, 76, 3], "target_rank": 3, "all_ranks": [3, 1, 1, 1]})
    off = classify_payload_origin(item, {"emitted": [99, 98, 76, 3], "target_rank": 9, "all_ranks": [9, 1, 1, 1]})
    assert follow["payload_origin"] == PAYLOAD_QUERIED_NEW
    assert stuck["payload_origin"] == PAYLOAD_ORIGINAL
    assert stuck["emitted_original_value_span"] is True
    assert other["payload_origin"] == PAYLOAD_OTHER_COMPETITOR
    assert other["emitted_original_value_span"] is False
    assert off["payload_origin"] == PAYLOAD_OFF_INVENTORY


def test_summarize_scored_does_not_fold_original_into_other_competitor():
    item = _keyed_item()
    rows = [
        {"emitted": [21, 22, 76, 3], "target_rank": 1, "all_ranks": [1, 1, 1, 1], "free_exact": True, "tf_exact": True, "first_error": 4},
        {"emitted": [31, 32, 76, 3], "target_rank": 2, "all_ranks": [2, 1, 1, 1], "free_exact": False, "tf_exact": False, "first_error": 0},
        {"emitted": [41, 42, 76, 3], "target_rank": 3, "all_ranks": [3, 1, 1, 1], "free_exact": False, "tf_exact": False, "first_error": 0},
        {"emitted": [99, 98, 76, 3], "target_rank": 8, "all_ranks": [8, 1, 1, 1], "free_exact": False, "tf_exact": False, "first_error": 0},
    ]
    summary = summarize_scored([item] * 4, rows)
    assert summary["query_swap_follow_new_value"] == 1
    assert summary["emitted_original_value_span"] == 1
    assert summary["payload_origin"][PAYLOAD_QUERIED_NEW] == 1
    assert summary["payload_origin"][PAYLOAD_ORIGINAL] == 1
    assert summary["payload_origin"][PAYLOAD_OTHER_COMPETITOR] == 1
    assert summary["payload_origin"][PAYLOAD_OFF_INVENTORY] == 1
    assert summary["other_competitor_copy"] == 1
    assert summary["competitor_copy"] == 2
    assert summary["queried_copy"] == 1
    assert summary["off_inventory"] == 1
    assert summary["rest_value_tf_lock"] == 4
    assert summary["rest_value_tf_lock_defined"] == 4


def test_inventory_snapshot_detects_runner_up_signal():
    item = _keyed_item()
    logits = [0.0] * 80
    logits[31] = 5.0  # original first token wins
    logits[21] = 4.0  # queried new is runner-up
    logits[41] = 1.0
    snapshot = inventory_first_token_snapshot(item, logits, greedy_token=31)
    assert snapshot["queried_signal"] == "runner_up_inventory"
    assert snapshot["queried_inventory_rank"] == 2
    assert snapshot["original_inventory_rank"] == 1
    assert snapshot["greedy_role"] == PAYLOAD_ORIGINAL
    assert snapshot["margin_vs_best_other_inventory"] == -1.0
    origin = classify_payload_origin(item, {"emitted": [31, 32], "target_rank": 2, "all_ranks": [2, 1, 1, 1], "free_exact": False})
    row = {"emitted": [31, 32, 76, 3], "target_rank": 2, "all_ranks": [2, 1, 1, 1], "free_exact": False, "tf_exact": False, "first_error": 0}
    assert mechanism_stage(item, row, origin, snapshot) == STAGE_WEAK_IDENTIFY


def test_mechanism_stage_splits_selection_continuation_and_suffix():
    item = _keyed_item()
    origin = {"payload_origin": PAYLOAD_QUERIED_NEW, "emitted_original_value_span": False}
    success = {
        "emitted": [21, 22, 76, 3],
        "target_rank": 1,
        "all_ranks": [1, 1, 1, 1],
        "free_exact": True,
        "tf_exact": True,
        "first_error": 4,
    }
    suffix = {**success, "emitted": [21, 22, 99, 3], "free_exact": False, "first_error": 2}
    no_continue = {**success, "emitted": [21, 99, 76, 3], "free_exact": False, "first_error": 1}
    cannot = {
        "emitted": [7, 8, 76, 3],
        "target_rank": 40,
        "all_ranks": [40, 1, 1, 1],
        "free_exact": False,
        "tf_exact": False,
        "first_error": 0,
    }
    assert mechanism_stage(item, success, origin, {"queried_signal": "rank1_vocab"}) == STAGE_SUCCESS
    assert mechanism_stage(item, suffix, origin, {"queried_signal": "rank1_vocab"}) == STAGE_PAYLOAD_SUFFIX_FAIL
    assert mechanism_stage(item, no_continue, origin, {"queried_signal": "rank1_vocab"}) == STAGE_SELECTED_NO_CONTINUE
    assert mechanism_stage(item, cannot, origin, {"queried_signal": "missing_queried_candidate"}) == STAGE_CANNOT_IDENTIFY


def test_original_equal_to_target_is_not_stuck_on_old():
    item = _keyed_item(isolation_transform="sep_swap_keep_markers", original_target_span=[21, 22])
    row = {"emitted": [21, 22, 82, 3], "target_rank": 1, "all_ranks": [1, 1, 2, 1], "free_exact": False, "tf_exact": False, "first_error": 2}
    origin = classify_payload_origin(item, row)
    assert origin["payload_origin"] == PAYLOAD_QUERIED_NEW
    assert origin["emitted_original_value_span"] is False
    logits = [0.0] * 80
    logits[21] = 5.0
    logits[31] = 1.0
    logits[41] = 0.5
    snapshot = inventory_first_token_snapshot(item, logits, greedy_token=21)
    assert snapshot["queried_signal"] == "rank1_vocab"
    assert snapshot["original_token"] is None


def test_mixed_pair_count_does_not_use_a_single_chance_baseline():
    two = _keyed_item(
        pair_count=2,
        source=[10, 21, 22, 11, 31, 32],
        input=[2, 10, 21, 22, 76, 11, 31, 32, 76, 10],
    )
    four = _keyed_item(
        pair_count=4,
        source=[10, 21, 22, 11, 31, 32, 12, 41, 42, 13, 51, 52],
        input=[2, 10, 21, 22, 76, 11, 31, 32, 76, 12, 41, 42, 76, 13, 51, 52, 76, 10],
    )
    mixed = summarize_scored(
        [two, four],
        [
            {"emitted": [21, 22, 76, 3], "target_rank": 1, "all_ranks": [1, 1, 1, 1], "free_exact": True, "tf_exact": True, "first_error": 4},
            {"emitted": [41, 42, 76, 3], "target_rank": 3, "all_ranks": [3, 1, 1, 1], "free_exact": False, "tf_exact": False, "first_error": 0},
        ],
    )
    assert mixed["chance_1_over_k"] is None
    assert mixed["by_pair_count"]["2"]["n"] == 1
    assert mixed["by_pair_count"]["4"]["n"] == 1
    assert mixed["by_pair_count"]["2"]["chance_1_over_k"] == 0.5
    assert mixed["by_pair_count"]["4"]["chance_1_over_k"] == 0.25
