from __future__ import annotations

"""Classify greedy emissions as queried-copy, competitor-copy, or off-inventory.

This is a data-only diagnostic: it uses frozen panel records plus already-scored
`emitted` tokens. It does not load weights and does not change Gate L/C/R.

The scientific split is:

- inventory copy (queried or competitor) measures payload copy/emission
- queried copy versus competitor copy measures query-conditioned selection
- off-inventory measures collapse of even the copy-a-span recipe
"""

from collections import Counter, defaultdict
from statistics import median
from typing import Any

from .isolation_transforms import find_body_range, locate_query, recover_rendered_pairs

KEYED_SOURCE_PANELS = (
    "primitive_keyed",
    "short_keyed",
    "same_surface_novel",
    "heldout_surface",
    "unseen_length",
    "low_prior",
    "distractor",
    "broken_context",
)


def classify_emission(item: dict, row: dict) -> dict[str, Any]:
    if item.get("kind") != "keyed":
        raise RuntimeError("emission-source classification is defined for keyed rows")
    rendered = recover_rendered_pairs(item)
    span = [int(x) for x in item["target_span"]]
    emitted = [int(x) for x in row.get("emitted", [])]
    emitted_value = emitted[: len(span)]
    query_key = int(item["query_key"])
    source = "off_inventory"
    emit_render_index = None
    emit_key = None
    for pair in rendered:
        if emitted_value == pair["value"]:
            emit_render_index = pair["render_index"]
            emit_key = pair["key"]
            source = "queried" if pair["original_key"] == query_key else "competitor"
            break
    query_render_index = next(
        (pair["render_index"] for pair in rendered if pair["original_key"] == query_key),
        None,
    )
    _, query_pos = locate_query(item)
    body_range = find_body_range(item)
    query_side = None
    if query_pos is not None and body_range is not None:
        if query_pos < body_range[0]:
            query_side = "before_body"
        elif query_pos >= body_range[1]:
            query_side = "after_body"
        else:
            query_side = "inside_body"
    return {
        "source": source,
        "pair_count": item.get("pair_count"),
        "query_index": item.get("query_index"),
        "query_render_index": query_render_index,
        "emit_render_index": emit_render_index,
        "emit_key": emit_key,
        "query_side": query_side,
        "target_rank": int(row["target_rank"]),
        "first_ok": int(row["target_rank"]) == 1,
        "value_ok": source == "queried",
        "n_rendered": len(rendered),
    }


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    return float(median(values))


def summarize_keyed_panel(pairs: list[tuple[dict, dict]]) -> dict:
    classified = [classify_emission(item, row) for item, row in pairs]
    counts = Counter(row["source"] for row in classified)
    n = len(classified)
    by_pair: dict[str, dict] = {}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in classified:
        grouped[int(row["pair_count"])].append(row)
    for pair_count, rows in sorted(grouped.items()):
        ok_by_pos: dict[str, dict] = {}
        pos_groups: dict[int | None, list[dict]] = defaultdict(list)
        for row in rows:
            pos_groups[row["query_render_index"]].append(row)
        for pos, bucket in sorted(pos_groups.items(), key=lambda kv: (kv[0] is None, kv[0] if kv[0] is not None else -1)):
            ok_by_pos[str(pos)] = {
                "n": len(bucket),
                "queried": sum(item["source"] == "queried" for item in bucket),
                "queried_rate": sum(item["source"] == "queried" for item in bucket) / len(bucket),
            }
        queried_ranks = [row["target_rank"] for row in rows if row["source"] == "queried"]
        competitor_ranks = [row["target_rank"] for row in rows if row["source"] == "competitor"]
        by_pair[str(pair_count)] = {
            "n": len(rows),
            "queried": sum(row["source"] == "queried" for row in rows),
            "competitor": sum(row["source"] == "competitor" for row in rows),
            "off_inventory": sum(row["source"] == "off_inventory" for row in rows),
            "queried_rate": sum(row["source"] == "queried" for row in rows) / len(rows),
            "inventory_copy_rate": sum(row["source"] != "off_inventory" for row in rows) / len(rows),
            "chance_1_over_k": 1.0 / pair_count,
            "P_emit_first_pair": sum(row["emit_render_index"] == 0 for row in rows) / len(rows),
            "P_query_first_pair": sum(row["query_render_index"] == 0 for row in rows) / len(rows),
            "queried_given_render_index": ok_by_pos,
            "median_target_rank_when_queried_copy": _median(queried_ranks),
            "median_target_rank_when_competitor_copy": _median(competitor_ranks),
            "rank1_when_competitor_copy": (
                sum(rank == 1 for rank in competitor_ranks) / len(competitor_ranks) if competitor_ranks else None
            ),
        }
    return {
        "n": n,
        "queried": counts["queried"],
        "competitor": counts["competitor"],
        "off_inventory": counts["off_inventory"],
        "inventory_copy_rate": (counts["queried"] + counts["competitor"]) / n if n else None,
        "queried_rate": counts["queried"] / n if n else None,
        "median_target_rank_when_queried_copy": _median([row["target_rank"] for row in classified if row["source"] == "queried"]),
        "median_target_rank_when_competitor_copy": _median([row["target_rank"] for row in classified if row["source"] == "competitor"]),
        "rank1_when_competitor_copy": (
            sum(row["target_rank"] == 1 for row in classified if row["source"] == "competitor") / counts["competitor"]
            if counts["competitor"]
            else None
        ),
        "by_pair_count": by_pair,
    }


def emission_source_report(panels: dict, term: dict) -> dict:
    out = {}
    for name in KEYED_SOURCE_PANELS:
        pairs = list(zip(panels[name], term["rows"][name]))
        out[name] = summarize_keyed_panel(pairs)
    novel = out["same_surface_novel"]
    return {
        "status": "V2R4_EMISSION_SOURCE",
        "note": (
            "Inventory copy = greedy value span equals some in-context pair value. "
            "Queried copy vs competitor copy splits selection from payload copy. "
            "These are diagnostic metrics, not Gate C."
        ),
        "headline": {
            "same_surface_novel_inventory_copy": novel["inventory_copy_rate"],
            "same_surface_novel_queried": novel["queried"],
            "same_surface_novel_competitor": novel["competitor"],
            "same_surface_novel_off_inventory": novel["off_inventory"],
            "competitor_copies_are_not_rank1": novel["rank1_when_competitor_copy"] == 0.0,
        },
        "panels": out,
        "hypothesis_read": {
            "A_internal_identification": "partial_queried_token_is_rank1_or_runner_up_on_train_surface",
            "B_query_binding": "failed_queried_copy_not_above_1_over_k_including_2_pair",
            "C_payload_copy": "supported_inventory_copy_93_of_96_train_novel",
            "D_free_emission_of_selected_span": "supported_tf_equals_free_on_selected_span",
            "E_separator_eos": "heldout_exact_still_separator_ood",
            "F_surface": "heldout_inventory_copy_drops_and_first_pair_bias_appears",
        },
    }
