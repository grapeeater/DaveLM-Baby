from __future__ import annotations

from pathlib import Path

from src.baby_v010.autopsy_v2r4 import run_autopsy
from src.baby_v010.mechanism_census import binomial_two_sided

ROOT = Path(__file__).resolve().parents[1]
PANELS = ROOT / "data" / "generated" / "foundation_v2" / "panels.json"
METRICS = ROOT / "runs" / "structured_v2r4_seed106001_from6000_terminal" / "metrics.jsonl"


def test_tf_lock_splits_continuation_from_selection():
    report = run_autopsy(METRICS, PANELS)
    lock = report["mechanism"]["tf_continuation_lock"]["same_surface_novel"]["by_source"]
    assert lock["queried"]["n"] == 41
    assert lock["queried"]["rest_value_tf_lock"] == 41
    assert lock["queried"]["full_after_first_tf_lock"] == 41
    assert lock["competitor"]["n"] == 52
    assert lock["competitor"]["rest_defined"] == 52
    assert lock["competitor"]["rest_value_tf_lock"] == 51
    assert lock["competitor"]["full_after_first_tf_lock"] == 51
    held = report["mechanism"]["tf_continuation_lock"]["heldout_surface"]["by_source"]
    assert held["queried"]["n"] == 23
    assert held["queried"]["rest_value_tf_lock"] == 23
    assert held["queried"]["full_after_first_tf_lock"] == 0
    assert report["mechanism"]["tf_continuation_lock"]["same_surface_novel"]["greedy_first_in_inventory_firsts"] == 95
    assert report["mechanism"]["tf_continuation_lock"]["same_surface_novel"]["unique_value_first_tokens"] == 96


def test_two_pair_queried_copy_is_not_above_chance():
    report = run_autopsy(METRICS, PANELS)
    two = report["mechanism"]["chance_vs_queried"]["same_surface_novel"]["2"]
    three = report["mechanism"]["chance_vs_queried"]["same_surface_novel"]["3"]
    four = report["mechanism"]["chance_vs_queried"]["same_surface_novel"]["4"]
    assert two["n"] == 31 and two["queried"] == 19
    assert two["significant_0_05_two_sided"] is False
    assert two["two_sided_p"] == binomial_two_sided(31, 19, 0.5)
    assert three["queried"] == 7 and three["significant_0_05_two_sided"] is False
    assert four["queried"] == 15 and four["significant_0_05_two_sided"] is False
    short_two = report["mechanism"]["chance_vs_queried"]["short_keyed"]["2"]
    assert short_two["n"] == 28 and short_two["queried"] == 18
    assert short_two["significant_0_05_two_sided"] is False


def test_probe_limit_matched_slice_does_not_jump_at_terminal():
    report = run_autopsy(METRICS, PANELS)
    novel = report["mechanism"]["developmental"]["same_surface_novel"]
    assert novel["matched_limit"] == 16
    assert novel["copy_onset_update_matched_inventory_one"] == 12000
    by_update = {row["update"]: row for row in novel["updates"]}
    assert by_update[15500]["scored_n"] == 16
    assert by_update[16000]["scored_n"] == 96
    assert by_update[15500]["matched_slice"]["queried"] == 6
    assert by_update[16000]["matched_slice"]["queried"] == 5
    assert by_update[16000]["matched_slice"]["competitor"] == 11
    assert by_update[16000]["matched_slice"]["inventory_copy_rate"] == 1.0
    comparison = novel["terminal_matched_vs_preterminal_matched"]
    assert comparison["preterminal"]["queried"] == 6
    assert comparison["terminal"]["queried"] == 5


def test_induction_immediate_eos_is_trailing_own_separator():
    report = run_autopsy(METRICS, PANELS)
    block = report["mechanism"]["induction_trailing_separator_eos"]
    assert block["all_immediate_eos_are_own_separator"] is True
    primitive = block["primitive_induction"]
    full = block["same_surface_induction"]
    assert primitive["immediate_eos"] == 3
    assert full["immediate_eos"] == 19
    assert primitive["immediate_eos"] + full["immediate_eos"] == 22
    assert full["when_last_is_own_separator"]["n"] == 32
    assert full["when_last_is_own_separator"]["immediate_eos"] == 19
    assert full["when_last_is_not_own_separator"]["n"] == 64
    assert full["when_last_is_not_own_separator"]["immediate_eos"] == 0


def test_heldout_off_inventory_is_train_separator_glue():
    report = run_autopsy(METRICS, PANELS)
    held = report["mechanism"]["off_inventory"]["heldout_surface"]
    assert held["n_off_inventory"] == 41
    assert held["has_train_separator"] == 41
    lengths = report["mechanism"]["value_length"]["heldout_surface"]
    assert lengths["10"]["n"] == 12
    assert lengths["10"]["inventory"] == 0
    headlines = report["mechanism_headlines"]
    assert headlines["heldout_off_inventory_all_have_train_sep"] is True
    assert headlines["train_novel_2_pair_significant_0_05"] is False
    assert report["hypothesis_read"]["probe_limit_confound"] == "interim_evals_n_16_only_terminal_full_panel"
