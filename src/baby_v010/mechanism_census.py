from __future__ import annotations

"""Mechanism census on frozen v2R4 metrics.

Data-only. Does not load weights, train, or change Gate L/C/R.

Splits that this module keeps separate:

- teacher-forced continuation of the gold value after the first token (A)
- greedy first-token selection among in-context values (B)
- exact inventory-span copy (C/D)
- separator/EOS suffix (E)
- probe-limit-matched developmental slices (so U16000 n=96 is not
  compared naively to earlier n=16 probes)
"""

from collections import Counter, defaultdict
from math import comb
from typing import Any

from .emission_source import KEYED_SOURCE_PANELS, classify_emission
from .isolation_transforms import recover_rendered_pairs
from .v2r4_provenance import EOS_TOKEN, TERMINAL_UPDATE

TRAIN_SEPARATORS = frozenset(range(76, 82))
PROBE_LIMIT = 16


def binomial_sf(n: int, k: int, p: float) -> float:
    """Exact P(X >= k) for X ~ Binomial(n, p)."""
    if n < 0 or k < 0:
        raise RuntimeError("binomial_sf requires non-negative n, k")
    if k > n:
        return 0.0
    return sum(comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i)) for i in range(k, n + 1))


def binomial_two_sided(n: int, k: int, p: float) -> float:
    """Exact two-sided test: sum P(X=i) for |i - np| >= |k - np|."""
    target = abs(k - n * p)
    total = 0.0
    for i in range(n + 1):
        if abs(i - n * p) + 1e-12 >= target:
            total += comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
    return total


def rest_value_tf_lock(item: dict, row: dict) -> bool | None:
    """Whether gold value tokens after the first are all rank-1 under TF.

    Undefined (None) when the value span is a single token.
    """
    span = [int(x) for x in item["target_span"]]
    if len(span) <= 1:
        return None
    ranks = [int(x) for x in row.get("all_ranks", [])]
    return len(ranks) >= len(span) and all(rank == 1 for rank in ranks[1 : len(span)])


def full_after_first_tf_lock(item: dict, row: dict) -> bool:
    """Whether every gold token after the first, including sep/EOS, is rank-1 under TF."""
    ranks = [int(x) for x in row.get("all_ranks", [])]
    return len(ranks) > 1 and all(rank == 1 for rank in ranks[1:])


def greedy_first_in_inventory_firsts(item: dict, row: dict) -> bool:
    emitted = [int(x) for x in row.get("emitted", [])]
    if not emitted:
        return False
    firsts = {pair["value"][0] for pair in recover_rendered_pairs(item) if pair["value"]}
    return emitted[0] in firsts


def lcp(left: list[int], right: list[int]) -> int:
    n = 0
    for a, b in zip(left, right):
        if a != b:
            break
        n += 1
    return n


def off_inventory_mode(item: dict, row: dict) -> str:
    rendered = recover_rendered_pairs(item)
    span = [int(x) for x in item["target_span"]]
    emitted = [int(x) for x in row.get("emitted", [])]
    emitted_value = emitted[: len(span)]
    query_key = int(item["query_key"])
    best = 0
    best_kind = "no_prefix"
    for pair in rendered:
        length = lcp(emitted_value, pair["value"])
        if length > best:
            best = length
            best_kind = "queried_prefix" if pair["original_key"] == query_key else "competitor_prefix"
    if emitted and emitted[0] == EOS_TOKEN:
        return "immediate_eos"
    if best >= 1:
        return best_kind
    if len(emitted) >= 2 and emitted[1] in TRAIN_SEPARATORS:
        return "first_then_train_sep"
    if any(token in TRAIN_SEPARATORS for token in emitted):
        return "train_sep_elsewhere"
    return "other"


def _lock_bucket() -> dict[str, int]:
    return {
        "n": 0,
        "rest_defined": 0,
        "rest_value_tf_lock": 0,
        "full_after_first_tf_lock": 0,
    }


def tf_lock_table(pairs: list[tuple[dict, dict]]) -> dict:
    by_source = {name: _lock_bucket() for name in ("queried", "competitor", "off_inventory")}
    first_token_hits = 0
    unique_first = 0
    for item, row in pairs:
        classified = classify_emission(item, row)
        bucket = by_source[classified["source"]]
        bucket["n"] += 1
        rest = rest_value_tf_lock(item, row)
        if rest is not None:
            bucket["rest_defined"] += 1
            bucket["rest_value_tf_lock"] += int(rest)
        bucket["full_after_first_tf_lock"] += int(full_after_first_tf_lock(item, row))
        first_token_hits += int(greedy_first_in_inventory_firsts(item, row))
        firsts = [pair["value"][0] for pair in recover_rendered_pairs(item) if pair["value"]]
        unique_first += int(len(set(firsts)) == len(firsts))
    n = len(pairs)
    out = {}
    for name, bucket in by_source.items():
        out[name] = {
            **bucket,
            "rest_value_tf_lock_rate": (
                bucket["rest_value_tf_lock"] / bucket["rest_defined"] if bucket["rest_defined"] else None
            ),
            "full_after_first_tf_lock_rate": (
                bucket["full_after_first_tf_lock"] / bucket["n"] if bucket["n"] else None
            ),
        }
    return {
        "n": n,
        "greedy_first_in_inventory_firsts": first_token_hits,
        "unique_value_first_tokens": unique_first,
        "by_source": out,
    }


def chance_tests_from_emission(panel_summary: dict) -> dict:
    out = {}
    for pair_count, block in panel_summary.get("by_pair_count", {}).items():
        n = int(block["n"])
        k = int(block["queried"])
        p = float(block["chance_1_over_k"])
        out[str(pair_count)] = {
            "n": n,
            "queried": k,
            "queried_rate": k / n if n else None,
            "chance_1_over_k": p,
            "one_sided_p_ge": binomial_sf(n, k, p),
            "two_sided_p": binomial_two_sided(n, k, p),
            "significant_0_05_two_sided": binomial_two_sided(n, k, p) < 0.05,
        }
    return out


def value_length_table(pairs: list[tuple[dict, dict]]) -> dict:
    grouped: dict[int, list[str]] = defaultdict(list)
    for item, row in pairs:
        grouped[int(item["value_length"])].append(classify_emission(item, row)["source"])
    out = {}
    for length, sources in sorted(grouped.items()):
        n = len(sources)
        queried = sum(src == "queried" for src in sources)
        inventory = sum(src != "off_inventory" for src in sources)
        out[str(length)] = {
            "n": n,
            "queried": queried,
            "inventory": inventory,
            "queried_rate": queried / n,
            "inventory_copy_rate": inventory / n,
        }
    return out


def query_side_table(pairs: list[tuple[dict, dict]]) -> dict:
    grouped: dict[str, list[str]] = defaultdict(list)
    for item, row in pairs:
        classified = classify_emission(item, row)
        grouped[str(classified["query_side"])].append(classified["source"])
    out = {}
    for side, sources in sorted(grouped.items()):
        n = len(sources)
        queried = sum(src == "queried" for src in sources)
        out[side] = {"n": n, "queried": queried, "queried_rate": queried / n}
    return out


def off_inventory_table(pairs: list[tuple[dict, dict]]) -> dict:
    modes: Counter[str] = Counter()
    has_train_sep = 0
    first_then_train_sep = 0
    n_off = 0
    for item, row in pairs:
        if classify_emission(item, row)["source"] != "off_inventory":
            continue
        n_off += 1
        emitted = [int(x) for x in row.get("emitted", [])]
        if any(token in TRAIN_SEPARATORS for token in emitted):
            has_train_sep += 1
        if len(emitted) >= 2 and emitted[1] in TRAIN_SEPARATORS:
            first_then_train_sep += 1
        modes[off_inventory_mode(item, row)] += 1
    return {
        "n_off_inventory": n_off,
        "has_train_separator": has_train_sep,
        "first_then_train_sep": first_then_train_sep,
        "modes": dict(modes),
    }


def induction_trailing_sep_eos(pairs: list[tuple[dict, dict]]) -> dict:
    last_is_own_sep_eos = 0
    last_is_own_sep_n = 0
    last_not_own_sep_eos = 0
    last_not_own_sep_n = 0
    eos_total = 0
    eos_last_is_own_sep = 0
    by_variant: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "eos": 0, "correct": 0})
    for item, row in pairs:
        if item.get("kind") != "induction":
            raise RuntimeError("induction_trailing_sep_eos requires induction items")
        last = int(item["input"][-1])
        own_sep = int(item["target"][-2])
        emitted = [int(x) for x in row.get("emitted", [])]
        eos = bool(emitted) and emitted[0] == EOS_TOKEN
        correct = bool(emitted) and emitted[0] == int(item["target_span"][0])
        variant = str(item.get("variant"))
        by_variant[variant]["n"] += 1
        by_variant[variant]["eos"] += int(eos)
        by_variant[variant]["correct"] += int(correct)
        if eos:
            eos_total += 1
            eos_last_is_own_sep += int(last == own_sep)
        if last == own_sep:
            last_is_own_sep_n += 1
            last_is_own_sep_eos += int(eos)
        else:
            last_not_own_sep_n += 1
            last_not_own_sep_eos += int(eos)
    return {
        "n": len(pairs),
        "immediate_eos": eos_total,
        "immediate_eos_last_token_is_own_separator": eos_last_is_own_sep,
        "when_last_is_own_separator": {
            "n": last_is_own_sep_n,
            "immediate_eos": last_is_own_sep_eos,
            "rate": last_is_own_sep_eos / last_is_own_sep_n if last_is_own_sep_n else None,
        },
        "when_last_is_not_own_separator": {
            "n": last_not_own_sep_n,
            "immediate_eos": last_not_own_sep_eos,
            "rate": last_not_own_sep_eos / last_not_own_sep_n if last_not_own_sep_n else None,
        },
        "by_variant": dict(by_variant),
    }


def _source_counts(pairs: list[tuple[dict, dict]]) -> dict[str, int]:
    counts = Counter(classify_emission(item, row)["source"] for item, row in pairs)
    n = len(pairs)
    queried = counts["queried"]
    competitor = counts["competitor"]
    off_inventory = counts["off_inventory"]
    return {
        "n": n,
        "queried": queried,
        "competitor": competitor,
        "off_inventory": off_inventory,
        "inventory_copy_rate": (queried + competitor) / n if n else None,
        "queried_rate": queried / n if n else None,
    }


def developmental_emission_source(
    panels: dict,
    records: list[dict],
    panel_name: str,
    matched_limit: int = PROBE_LIMIT,
) -> dict:
    rows_out = []
    for record in records:
        scored = record["rows"][panel_name]
        inferred_limit = len(scored)
        matched_n = min(matched_limit, inferred_limit, len(panels[panel_name]))
        matched = list(zip(panels[panel_name][:matched_n], scored[:matched_n]))
        full = list(zip(panels[panel_name][:inferred_limit], scored))
        rows_out.append({
            "update": record.get("update"),
            "stage": record.get("stage"),
            "scored_n": inferred_limit,
            "probe_limit_inferred": inferred_limit,
            "matched_slice": _source_counts(matched),
            "scored_as_logged": _source_counts(full),
            "language_dev_ce": record.get("language_dev_ce"),
        })
    terminal = [row for row in rows_out if int(row["update"]) == TERMINAL_UPDATE]
    preterminal = [row for row in rows_out if int(row["update"]) == TERMINAL_UPDATE - 500]
    return {
        "panel": panel_name,
        "matched_limit": matched_limit,
        "note": (
            "Interim v2R4 evals used probe_limit=16; only U16000 scored the full panel. "
            "Compare matched_slice across updates. Do not read the U16000 n=96 jump as a "
            "late capability explosion."
        ),
        "copy_onset_update_matched_inventory_one": next(
            (
                row["update"]
                for row in rows_out
                if row["matched_slice"]["inventory_copy_rate"] == 1.0
            ),
            None,
        ),
        "terminal_matched_vs_preterminal_matched": {
            "preterminal": preterminal[0]["matched_slice"] if preterminal else None,
            "terminal": terminal[0]["matched_slice"] if terminal else None,
        },
        "updates": rows_out,
    }


def mechanism_census(panels: dict, records: list[dict], term: dict, emission_source: dict) -> dict:
    def keyed_pairs(name: str) -> list[tuple[dict, dict]]:
        return list(zip(panels[name], term["rows"][name]))

    keyed = {name: keyed_pairs(name) for name in KEYED_SOURCE_PANELS}
    novel = keyed["same_surface_novel"]
    held = keyed["heldout_surface"]
    novel_chance = chance_tests_from_emission(emission_source["panels"]["same_surface_novel"])
    short_chance = chance_tests_from_emission(emission_source["panels"]["short_keyed"])
    primitive_ind = list(zip(panels["primitive_induction"], term["rows"]["primitive_induction"]))
    full_ind = list(zip(panels["same_surface_induction"], term["rows"]["same_surface_induction"]))
    primitive_ind_eos = induction_trailing_sep_eos(primitive_ind)
    full_ind_eos = induction_trailing_sep_eos(full_ind)
    return {
        "status": "V2R4_MECHANISM_CENSUS",
        "tf_continuation_lock": {name: tf_lock_table(pairs) for name, pairs in keyed.items()},
        "chance_vs_queried": {
            "same_surface_novel": novel_chance,
            "short_keyed": short_chance,
        },
        "query_side": {
            "same_surface_novel": query_side_table(novel),
            "heldout_surface": query_side_table(held),
        },
        "value_length": {
            "same_surface_novel": value_length_table(novel),
            "heldout_surface": value_length_table(held),
        },
        "off_inventory": {
            "same_surface_novel": off_inventory_table(novel),
            "heldout_surface": off_inventory_table(held),
        },
        "induction_trailing_separator_eos": {
            "primitive_induction": primitive_ind_eos,
            "same_surface_induction": full_ind_eos,
            "all_immediate_eos_are_own_separator": (
                primitive_ind_eos["immediate_eos_last_token_is_own_separator"]
                + full_ind_eos["immediate_eos_last_token_is_own_separator"]
                == primitive_ind_eos["immediate_eos"] + full_ind_eos["immediate_eos"]
            ),
        },
        "developmental": {
            "same_surface_novel": developmental_emission_source(panels, records, "same_surface_novel"),
            "short_keyed": developmental_emission_source(panels, records, "short_keyed"),
            "primitive_keyed": developmental_emission_source(panels, records, "primitive_keyed"),
            "heldout_surface": developmental_emission_source(panels, records, "heldout_surface"),
        },
        "hypothesis_read": {
            "A_tf_continuation_after_gold_first_token": "supported_train_novel_51_of_52_competitor_rows_lock",
            "B_query_binding": "not_above_chance_on_queried_copy_including_2_pair",
            "C_payload_copy": "supported_train_inventory_copy",
            "D_curriculum": "supported_1pair_then_copy_span_then_selection_never_lifts_on_matched_slice",
            "E_separator_eos": "heldout_off_inventory_is_train_sep_glue_and_induction_eos_is_trailing_sep",
            "F_surface": "heldout_inventory_copy_collapses_at_value_length_ge_5",
            "probe_limit_confound": "interim_n_16_only_terminal_is_full_panel",
        },
    }


def mechanism_headlines(census: dict) -> dict[str, Any]:
    novel_lock = census["tf_continuation_lock"]["same_surface_novel"]
    held_off = census["off_inventory"]["heldout_surface"]
    novel_dev = census["developmental"]["same_surface_novel"]
    two_pair = census["chance_vs_queried"]["same_surface_novel"]["2"]
    induction = census["induction_trailing_separator_eos"]
    return {
        "train_novel_competitor_rest_value_tf_lock": (
            novel_lock["by_source"]["competitor"]["rest_value_tf_lock"],
            novel_lock["by_source"]["competitor"]["rest_defined"],
        ),
        "train_novel_2_pair_two_sided_p": two_pair["two_sided_p"],
        "train_novel_2_pair_significant_0_05": two_pair["significant_0_05_two_sided"],
        "heldout_off_inventory_all_have_train_sep": (
            held_off["n_off_inventory"] == held_off["has_train_separator"]
        ),
        "induction_eos_all_trailing_own_sep": induction["all_immediate_eos_are_own_separator"],
        "matched_slice_terminal_queried": novel_dev["terminal_matched_vs_preterminal_matched"]["terminal"]["queried"],
        "matched_slice_preterminal_queried": novel_dev["terminal_matched_vs_preterminal_matched"]["preterminal"]["queried"],
        "copy_onset_update": novel_dev["copy_onset_update_matched_inventory_one"],
    }
