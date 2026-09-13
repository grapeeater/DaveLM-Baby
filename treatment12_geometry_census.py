r"""Outcome-blind geometry census of the frozen T12/T11 train and retention pools.

Reads only the two structural pools (hash-checked). No checkpoint, logits,
hidden states, or model predictions are used. Describes the geometric universe
the frozen T12 model was trained and tested on.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from treatment12_config import RETENTION_POOL_PATH, TRAIN_POOL_PATH
from treatment12_model import correct_row_index, doc_retrieval_meta

OUT_DIR = Path(r"C:\DaveLM-CADAVER\treatment12_frozen_causal_validation")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(record: dict, keys) -> str:
    return "|".join(repr(record[k]) for k in keys)


LAYOUT_KEYS = [
    "sequence_length", "q", "answer_causal", "answer_index",
    "k0", "v0", "k1", "v1",
    "q_to_k0", "q_to_k1", "q_to_v0", "q_to_v1",
    "k0_to_v0", "k1_to_v1",
    "row0_to_row1_spacing", "row0_val_to_row1_val_spacing",
    "query_to_answer",
]
FULL_KEYS = LAYOUT_KEYS + ["query_slot", "orientation", "mapping_order"]


def geometry_record(doc) -> dict:
    meta = doc_retrieval_meta(doc)
    return {
        "doc_id": doc["doc_id"],
        "quartet_id": doc["quartet_id"],
        "orientation": int(doc["orientation"]),
        "query_slot": int(doc["query_slot"]),
        "mapping_order": int(doc.get("mapping_order", -1)),
        "query_key_token": int(doc["query_key_token"]),
        "sequence_length": len(doc["full_document_token_ids"]),
        "q": meta["qdp"],
        "answer_causal": meta["answer_causal_pos"],
        "answer_index": meta["answer_token_index"],
        "k0": meta["row0_src_pos"],
        "v0": meta["row0_val_pos"],
        "k1": meta["row1_src_pos"],
        "v1": meta["row1_val_pos"],
        "k0_src_token": meta["row0_src_token"],
        "v0_token": meta["row0_val_token"],
        "k1_src_token": meta["row1_src_token"],
        "v1_token": meta["row1_val_token"],
        "q_to_k0": meta["qdp"] - meta["row0_src_pos"],
        "q_to_k1": meta["qdp"] - meta["row1_src_pos"],
        "q_to_v0": meta["qdp"] - meta["row0_val_pos"],
        "q_to_v1": meta["qdp"] - meta["row1_val_pos"],
        "k0_to_v0": meta["row0_val_pos"] - meta["row0_src_pos"],
        "k1_to_v1": meta["row1_val_pos"] - meta["row1_src_pos"],
        "row0_to_row1_spacing": meta["row1_src_pos"] - meta["row0_src_pos"],
        "row0_val_to_row1_val_spacing": meta["row1_val_pos"] - meta["row0_val_pos"],
        "query_to_answer": meta["answer_causal_pos"] - meta["qdp"],
        "correct_row": correct_row_index(meta),
        "target_row_value_token": int(doc["target_value_token"]),
    }


def summarize_set(item: dict) -> str:
    raise NotImplementedError


def render_markdown(result: dict, train_docs: int, retention_docs: int) -> str:
    lines = []
    lines.append("# T12/T11 frozen geometry census")
    lines.append("")
    lines.append("Outcome-blind structural census. No model, logits, hidden states, or outcomes used.")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- train documents: {train_docs}")
    lines.append(f"- retention documents: {retention_docs}")
    lines.append(f"- train pool sha256: `{result['provenance']['train_pool_sha256']}`")
    lines.append(f"- retention pool sha256: `{result['provenance']['retention_pool_sha256']}`")
    lines.append("")
    ls = result["unique_layout_signatures"]
    lines.append("## Unique layout signatures")
    lines.append("")
    lines.append("| set | count |")
    lines.append("|---|---|")
    lines.append(f"| train layout signatures | {ls['train']} |")
    lines.append(f"| retention layout signatures | {ls['retention']} |")
    lines.append(f"| retention signatures also seen in train | {ls['retention_also_in_train']} |")
    lines.append(f"| retention signatures NOT seen in train | {ls['retention_not_in_train_count']} |")
    lines.append("")
    fs = result["unique_full_signatures"]
    lines.append("Full signatures (layout + query slot + orientation + mapping order): "
                 f"train={fs['train']}, retention={fs['retention']}, "
                 f"retention-not-in-train={fs['retention_not_in_train_count']}.")
    lines.append("")
    lines.append("## Absolute position reuse")
    lines.append("")
    lines.append("| variable | train distinct | retention distinct | retention missing from train |")
    lines.append("|---|---|---|---|")
    for var, cov in result["absolute_position_coverage"].items():
        missing = cov["retention_values_missing_from_train"]
        lines.append(f"| {var} | {cov['train_distinct']} | {cov['retention_distinct']} | {missing} |")
    lines.append("")
    lines.append(f"All absolute q/k/v/answer positions reused in retention: "
                 f"{result['all_absolute_positions_reused_in_retention']}")
    lines.append("")
    lines.append("## Relative distance reuse")
    lines.append("")
    lines.append("| variable | train distinct | retention distinct | retention missing |")
    lines.append("|---|---|---|---|")
    for var, cov in result["relative_distance_coverage"].items():
        missing = cov["retention_values_missing_from_train"]
        lines.append(f"| {var} | {cov['train_distinct']} | {cov['retention_distinct']} | {missing} |")
    lines.append("")
    lines.append("Relative-distance combination reuse: "
                 f"{result['relative_distance_combos_reused_in_retention']}")
    lines.append("")
    det = result["deterministic_relationships"]
    lines.append("## Deterministic geometry relationships (train only)")
    lines.append("")
    lines.append(f"- layout signatures total: {det['layout_signatures_total']}")
    lines.append(f"- signatures with a single query slot: {det['layout_signatures_with_single_query_slot']} "
                 f"-> query_slot deterministic given layout: "
                 f"{det['query_slot_deterministic_given_layout']}")
    lines.append(f"- signatures with a single orientation: "
                 f"{det['layout_signatures_with_single_orientation']} "
                 f"-> orientation deterministic given layout: "
                 f"{det['orientation_deterministic_given_layout']}")
    lines.append(f"- distinct target-row values per layout signature (set sizes): "
                 f"{det['target_row_set_sizes_given_layout']}")
    lines.append("")
    lines.append(f"Retention contains genuinely unseen layout geometry: "
                 f"{result['retention_has_genuinely_unseen_geometry']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    train_pool = read_json(TRAIN_POOL_PATH)
    retention_pool = read_json(RETENTION_POOL_PATH)
    train_sha = sha256_file(TRAIN_POOL_PATH)
    retention_sha = sha256_file(RETENTION_POOL_PATH)
    require(train_sha == "8f5d60de10fe2fdc8e772a9c1fc3e9f07861edd1583d7c413a095c2f55c6903c",
            "train pool hash mismatch")
    require(retention_sha == "c341d7308b145bd3c633b62d56e01391f4b05635bfcf2edfb146bcd9f2de4e69",
            "retention pool hash mismatch")

    train_docs = [doc for q in train_pool["quartets"] for doc in q["docs"]]
    retention_docs = [doc for q in retention_pool["quartets"] for doc in q["docs"]]
    train_recs = [geometry_record(d) for d in train_docs]
    retention_recs = [geometry_record(d) for d in retention_docs]

    train_layout = {fingerprint(r, LAYOUT_KEYS) for r in train_recs}
    train_full = {fingerprint(r, FULL_KEYS) for r in train_recs}
    ret_layout = {fingerprint(r, LAYOUT_KEYS) for r in retention_recs}
    ret_full = {fingerprint(r, FULL_KEYS) for r in retention_recs}

    abs_vars = ["q", "answer_causal", "k0", "v0", "k1", "v1"]
    coverage = {}
    all_abs_reused = True
    for var in abs_vars:
        t_vals = {r[var] for r in train_recs}
        r_vals = {r[var] for r in retention_recs}
        missing = sorted(r_vals - t_vals)
        coverage[var] = {
            "train_distinct": len(t_vals),
            "retention_distinct": len(r_vals),
            "retention_values_missing_from_train": missing,
        }
        if missing:
            all_abs_reused = False

    dist_vars = ["q_to_k0", "q_to_k1", "q_to_v0", "q_to_v1",
                 "k0_to_v0", "k1_to_v1", "row0_to_row1_spacing",
                 "row0_val_to_row1_val_spacing", "query_to_answer"]
    dist_report = {}
    rel_reused = True
    for var in dist_vars:
        t_vals = {r[var] for r in train_recs}
        r_vals = {r[var] for r in retention_recs}
        missing = sorted(r_vals - t_vals)
        dist_report[var] = {
            "train_distinct": len(t_vals),
            "retention_distinct": len(r_vals),
            "retention_values_missing_from_train": missing,
        }
        if missing:
            rel_reused = False

    combo_groups = [("q_to_k0", "q_to_k1"), ("q_to_v0", "q_to_v1"),
                    ("k0_to_v0", "k1_to_v1"),
                    ("row0_to_row1_spacing", "row0_val_to_row1_val_spacing")]
    combo_report = {}
    for var_group in combo_groups:
        name = "+".join(var_group)
        t_combos = {tuple(r[v] for v in var_group) for r in train_recs}
        r_combos = {tuple(r[v] for v in var_group) for r in retention_recs}
        combo_report[name] = {
            "train_distinct": len(t_combos),
            "retention_distinct": len(r_combos),
            "retention_combos_missing_from_train": sorted(r_combos - t_combos),
        }

    slot_by_layout = defaultdict(set)
    orient_by_layout = defaultdict(set)
    target_row_by_layout = defaultdict(set)
    for r in train_recs:
        key = fingerprint(r, LAYOUT_KEYS)
        slot_by_layout[key].add(r["query_slot"])
        orient_by_layout[key].add(r["orientation"])
        target_row_by_layout[key].add(r["correct_row"])

    slot_deterministic = sum(1 for s in slot_by_layout.values() if len(s) == 1)
    orient_deterministic = sum(1 for s in orient_by_layout.values() if len(s) == 1)
    target_row_sizes = sorted({len(v) for v in target_row_by_layout.values()})

    result = {
        "provenance": {
            "train_pool_sha256": train_sha,
            "retention_pool_sha256": retention_sha,
            "train_documents": len(train_docs),
            "retention_documents": len(retention_docs),
        },
        "unique_layout_signatures": {
            "train": len(train_layout),
            "retention": len(ret_layout),
            "retention_also_in_train": len(ret_layout & train_layout),
            "retention_not_in_train_count": len(ret_layout - train_layout),
            "retention_not_in_train": sorted(ret_layout - train_layout),
        },
        "unique_full_signatures": {
            "train": len(train_full),
            "retention": len(ret_full),
            "retention_also_in_train": len(ret_full & train_full),
            "retention_not_in_train_count": len(ret_full - train_full),
            "retention_not_in_train": sorted(ret_full - train_full),
        },
        "absolute_position_coverage": coverage,
        "all_absolute_positions_reused_in_retention": all_abs_reused,
        "relative_distance_coverage": dist_report,
        "relative_distance_combos_reused_in_retention": rel_reused,
        "relative_distance_combo_report": combo_report,
        "retention_has_genuinely_unseen_geometry": len(ret_layout - train_layout) > 0,
        "deterministic_relationships": {
            "layout_signatures_total": len(slot_by_layout),
            "layout_signatures_with_single_query_slot": slot_deterministic,
            "query_slot_deterministic_given_layout": slot_deterministic == len(slot_by_layout),
            "layout_signatures_with_single_orientation": orient_deterministic,
            "orientation_deterministic_given_layout": orient_deterministic == len(orient_by_layout),
            "target_row_set_sizes_given_layout": target_row_sizes,
            "target_row_deterministic_given_layout": all(
                len(v) == 1 for v in target_row_by_layout.values()),
        },
        "query_slot_counts": {
            "train": dict(sorted(Counter(r["query_slot"] for r in train_recs).items())),
            "retention": dict(sorted(Counter(r["query_slot"] for r in retention_recs).items())),
        },
        "orientation_counts": {
            "train": dict(sorted(Counter(r["orientation"] for r in train_recs).items())),
            "retention": dict(sorted(Counter(r["orientation"] for r in retention_recs).items())),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "geometry_census.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (OUT_DIR / "geometry_census_report.md").write_text(
        render_markdown(result, len(train_docs), len(retention_docs)), encoding="utf-8")
    summary = {
        "unique_layout_signatures": result["unique_layout_signatures"],
        "unique_full_signatures": result["unique_full_signatures"],
        "all_absolute_positions_reused_in_retention": all_abs_reused,
        "relative_distance_combos_reused_in_retention": rel_reused,
        "retention_has_genuinely_unseen_geometry": result["retention_has_genuinely_unseen_geometry"],
        "deterministic_relationships": result["deterministic_relationships"],
        "query_slot_counts": result["query_slot_counts"],
        "orientation_counts": result["orientation_counts"],
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"GEOMETRY CENSUS ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
