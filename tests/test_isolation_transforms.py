from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.baby_v010.isolation_transforms import (
    build_isolation_panels,
    extract_marker_tokens,
    harvest_markers,
    harvest_separators,
    locate_query,
    parse_records,
    query_swap,
    sep_swap_keep_markers,
    value_absent,
    write_isolation_panels,
)
from src.baby_v010.v2r4_provenance import FROZEN_PANELS_SHA256, identify_frozen_file

ROOT = Path(__file__).resolve().parents[1]
PANELS_PATH = ROOT / "data" / "generated" / "foundation_v2" / "panels.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_panels() -> dict:
    return json.loads(PANELS_PATH.read_text(encoding="utf-8"))


def test_frozen_panels_hash_is_unchanged_by_transforms(tmp_path: Path):
    identity = identify_frozen_file(PANELS_PATH, FROZEN_PANELS_SHA256)
    assert identity["match"] is True
    before = identity["working_tree_sha256"]
    write_isolation_panels(PANELS_PATH, tmp_path)
    after = _sha(PANELS_PATH)
    assert after == before
    assert (tmp_path / "ISOLATION_PANELS.json").exists()
    original = json.loads(PANELS_PATH.read_text(encoding="utf-8"))
    copies = json.loads((tmp_path / "ISOLATION_PANELS.json").read_text(encoding="utf-8"))
    assert original["same_surface_novel"][0]["input"] != copies["marker_swap_keep_sep_same_surface_novel"][0]["input"]


def test_harvested_surface_tokens_match_v2_allocation():
    panels = _load_panels()
    train_markers, heldout_markers = harvest_markers(panels)
    train_seps, heldout_seps = harvest_separators(panels)
    assert train_markers == tuple(sorted({tok for item in panels["same_surface_novel"] for tok in extract_marker_tokens(item)}))
    assert heldout_markers == tuple(range(64, 76))
    assert train_seps == tuple(range(76, 82))
    assert heldout_seps == tuple(range(82, 86))
    assert not set(train_markers) & set(heldout_markers)
    assert not set(train_seps) & set(heldout_seps)


def test_query_swap_follows_alternate_key_and_keeps_body():
    panels = _load_panels()
    item = next(row for row in panels["same_surface_novel"] if row["pair_count"] >= 2)
    swapped = query_swap(item)
    assert swapped["input"] != item["input"]
    assert swapped["query_key"] != item["query_key"]
    assert swapped["target_span"] != item["target_span"]
    assert swapped["original_query_key"] == item["query_key"]
    records = parse_records(item)
    new_value = dict(records)[swapped["query_key"]]
    assert swapped["target_span"] == list(new_value)
    _, old_query_pos = locate_query(item)
    assert item["input"][old_query_pos] == item["query_key"]
    assert swapped["input"][old_query_pos] == swapped["query_key"]
    # Body still contains both original values.
    for _, value in records:
        span = list(value)
        assert any(swapped["input"][index : index + len(span)] == span for index in range(len(swapped["input"]) - len(span) + 1))
    assert item["input"] == json.loads(PANELS_PATH.read_text(encoding="utf-8"))["same_surface_novel"][panels["same_surface_novel"].index(item)]["input"]


def test_sep_swap_changes_suffix_only():
    panels = _load_panels()
    train_seps, heldout_seps = harvest_separators(panels)
    item = panels["same_surface_novel"][0]
    swapped = sep_swap_keep_markers(item, train_seps, heldout_seps)
    assert swapped["target_span"] == item["target_span"]
    assert swapped["target"][-2] in heldout_seps
    assert swapped["target"][-2] != item["target"][-2]
    assert extract_marker_tokens(swapped) == extract_marker_tokens(item)


def test_value_absent_removes_original_span_and_keeps_target():
    panels = _load_panels()
    item = panels["same_surface_novel"][0]
    absent = value_absent(item)
    span = list(item["target_span"])
    assert absent["target_span"] == span
    assert absent["target"] == item["target"]
    still = any(absent["input"][index : index + len(span)] == span for index in range(len(absent["input"]) - len(span) + 1))
    assert still is False
    present_orig = any(item["input"][index : index + len(span)] == span for index in range(len(item["input"]) - len(span) + 1))
    assert present_orig is True


def test_isolation_bundle_counts_and_query_swap_coverage():
    panels = _load_panels()
    isolation = build_isolation_panels(panels)
    assert isolation["query_swap_same_surface_novel"]
    assert len(isolation["query_swap_same_surface_novel"]) == 96
    assert len(isolation["query_swap_short_keyed"]) == 28
    assert len(isolation["same_surface_novel_pair_count_2"]) == 31
    assert len(isolation["same_surface_novel_pair_count_3"]) == 21
    assert len(isolation["same_surface_novel_pair_count_4"]) == 44
    assert len(isolation["short_keyed_pair_count_1"]) == 36
    assert len(isolation["value_absent_broken_context"]) == 64
    assert isolation["meta"]["frozen_panels_mutated"] is False
    assert isolation["meta"]["diagnostic_only"] is True
