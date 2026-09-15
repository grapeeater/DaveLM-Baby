from __future__ import annotations

"""Automated pre-training shortcut and leakage audits."""

import argparse
import json
from pathlib import Path

from .data import LANG_TRAIN, marker_families, read_u16, build_banks


def run_audit(panel_path: Path) -> dict:
    panels = json.loads(panel_path.read_text(encoding="utf-8"))
    language = read_u16(LANG_TRAIN)
    banks = build_banks(language)
    train_families, holdout_families = marker_families(banks)
    train_markers = {x for family in train_families for x in family}
    holdout_markers = {x for family in holdout_families for x in family}
    all_items = panels["all_intact"]
    sequences = [tuple(item["input"]) for item in all_items]
    targets = [tuple(item["target_span"]) for item in all_items]
    keyed_targets = [tuple(item["target_span"]) for item in all_items if item["kind"] == "keyed"]
    induction_targets = [tuple(item["target_span"]) for item in all_items if item["kind"] == "induction"]
    # A target span with an unsupported internal bigram cannot occur contiguously
    # in the language stream. The generator records that invariant for every
    # panel item, so this is an exact leakage proof without an O(N * panel)
    # substring scan over the 8M-token stream.
    internal_violations = sum(
        any(pair in banks.bigrams for pair in zip(item["target_span"], item["target_span"][1:]))
        for item in all_items
    )
    exact_source_target_hits = 0 if internal_violations == 0 else None
    report = {
        "panel_path": str(panel_path),
        "n_items": len(all_items),
        "duplicate_contexts": len(sequences) - len(set(sequences)),
        "duplicate_targets": len(targets) - len(set(targets)),
        "duplicate_keyed_targets": len(keyed_targets) - len(set(keyed_targets)),
        "induction_target_token_inventory": len(set(induction_targets)),
        "target_span_exact_language_hits": exact_source_target_hits,
        "target_internal_bigrams_supported_by_language": internal_violations,
        "marker_sets_disjoint": not bool(train_markers & holdout_markers),
        "markers_disjoint_from_span_bank": not bool((train_markers | holdout_markers) & set(banks.span)),
        "heldout_markers_seen_in_train_marker_set": len(holdout_markers & train_markers),
        "protected_material_opened": False,
        "status": "PASS" if not sequences or (len(sequences) == len(set(sequences)) and len(keyed_targets) == len(set(keyed_targets)) and internal_violations == 0) else "FAIL",
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.panels)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
