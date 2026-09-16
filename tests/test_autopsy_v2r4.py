from __future__ import annotations

import json
from pathlib import Path

from src.baby_v010.autopsy_v2r4 import run_autopsy
from src.baby_v010.v2r4_provenance import (
    FROZEN_PANELS_SHA256,
    TERMINAL_METRICS_SHA256,
    UNIQUE_SCORED_PANELS,
)

ROOT = Path(__file__).resolve().parents[1]
PANELS = ROOT / "data" / "generated" / "foundation_v2" / "panels.json"
METRICS = ROOT / "runs" / "structured_v2r4_seed106001_from6000_terminal" / "metrics.jsonl"


def test_autopsy_locks_terminal_value_span_census():
    report = run_autopsy(METRICS, PANELS)
    assert report["panels_sha256"] == FROZEN_PANELS_SHA256
    assert report["metrics_sha256"] == TERMINAL_METRICS_SHA256
    assert report["gates_weakened"] is False
    assert report["protected_material_opened"] is False
    assert report["v2r5_status"] == "absent_in_github_unresolved_pending"

    primitive = report["panels"]["primitive_keyed"]
    assert primitive["n"] == 64
    assert primitive["first_ok"] == 64
    assert primitive["value_ok"] == 64
    assert primitive["free_exact"] == 62
    assert list(primitive["by_pair_count"].keys()) == ["1"]

    short = report["panels"]["short_keyed"]
    assert short["by_pair_count"]["1"]["n"] == 36
    assert short["by_pair_count"]["1"]["value_ok"] == 1.0
    assert short["by_pair_count"]["1"]["free_exact"] == 1.0
    assert short["by_pair_count"]["2"]["n"] == 28
    assert short["by_pair_count"]["2"]["value_ok"] == 18 / 28
    assert short["by_pair_count"]["2"]["free_exact"] == 18 / 28

    novel = report["panels"]["same_surface_novel"]
    assert novel["first_ok"] == 43
    assert novel["value_ok"] == 41
    assert novel["free_exact"] == 41
    assert novel["by_pair_count"]["2"]["n"] == 31
    assert novel["by_pair_count"]["2"]["value_ok"] == 19 / 31
    assert novel["by_pair_count"]["3"]["n"] == 21
    assert novel["by_pair_count"]["3"]["value_ok"] == 7 / 21
    assert novel["by_pair_count"]["3"]["chance_1_over_k"] == 1 / 3
    assert novel["by_pair_count"]["4"]["n"] == 44
    assert novel["by_pair_count"]["4"]["value_ok"] == 15 / 44

    hold = report["panels"]["heldout_surface"]
    assert hold["first_ok"] == 33
    assert hold["value_ok"] == 23
    assert hold["free_exact"] == 0
    assert hold["fail"]["value_ok_bad_sep"] == 15
    assert hold["fail"]["value_ok_bad_eos"] == 8

    broken = report["panels"]["broken_context"]
    assert broken["value_ok"] == 16
    assert broken["n"] == 64
    assert abs(report["value_vs_separator"]["heldout_value_rate"] - 23 / 96) < 1e-12
    assert abs(report["value_vs_separator"]["broken_context_value_rate"] - 16 / 64) < 1e-12
    assert report["value_vs_separator"]["tf_value_equals_free_value_all_unique_panels"] is True


def test_autopsy_alias_aware_gate_l_and_induction_eos():
    report = run_autopsy(METRICS, PANELS)
    aliased = report["alias_inflation"]["aliased_scored_rows"]
    unique = report["alias_inflation"]["unique_scored_rows"]
    assert aliased["n"] == 992
    assert aliased["immediate_eos"] == 41
    assert unique["n"] == 800
    assert unique["immediate_eos"] == 22
    assert unique["all_same_token"] == 0
    induction = report["panels"]["same_surface_induction"]
    primitive_ind = report["panels"]["primitive_induction"]
    assert induction["immediate_eos"] + primitive_ind["immediate_eos"] == 22
    assert induction["first_ok"] == 6
    assert induction["free_exact"] == 5
    assert primitive_ind["first_ok"] == 19
    assert primitive_ind["free_exact"] == 14


def test_autopsy_refuses_wrong_hash(tmp_path: Path):
    bogus = tmp_path / "panels.json"
    bogus.write_text("{}\n", encoding="utf-8")
    try:
        run_autopsy(METRICS, bogus)
    except RuntimeError as exc:
        assert "does not match frozen" in str(exc)
    else:
        raise AssertionError("expected hash refusal")
    for name in UNIQUE_SCORED_PANELS:
        assert name in json.loads((ROOT / "data" / "generated" / "foundation_v2" / "AUDIT.json").read_text())["panel_counts"]


def test_autopsy_emission_source_splits_copy_from_selection():
    report = run_autopsy(METRICS, PANELS)
    source = report["emission_source"]["panels"]
    novel = source["same_surface_novel"]
    assert novel["queried"] == 41
    assert novel["competitor"] == 52
    assert novel["off_inventory"] == 3
    assert novel["inventory_copy_rate"] == 93 / 96
    assert novel["rank1_when_competitor_copy"] == 0.0
    assert novel["median_target_rank_when_queried_copy"] == 1.0
    assert novel["median_target_rank_when_competitor_copy"] == 2.0
    short = source["short_keyed"]
    assert short["queried"] == 54
    assert short["competitor"] == 10
    assert short["off_inventory"] == 0
    hold = source["heldout_surface"]
    assert hold["queried"] == 23
    assert hold["competitor"] == 32
    assert hold["off_inventory"] == 41
    primitive = source["primitive_keyed"]
    assert primitive["queried"] == 64
    assert report["hypothesis_read"]["B_query_binding"] == "failed_competitor_copy_is_the_dominant_train_error"
    assert report["hypothesis_read"]["C_payload_copy"] == "supported_train_novel_inventory_copy_93_of_96"
