from __future__ import annotations

"""Read-only classification for v2R4 isolation probes.

No training. These helpers split greedy emissions into:

- queried payload (new gold after a query-swap)
- previous/original payload
- another in-context competitor payload
- off-inventory

and, when first-token logits are supplied, rank competing in-context value
first tokens so a runner-up queried candidate is not collapsed into
"cannot identify."
"""

from collections import Counter
from statistics import median
from typing import Any

from .autopsy_v2r4 import classify_item_row
from .emission_source import classify_emission
from .isolation_transforms import recover_rendered_pairs
from .mechanism_census import binomial_two_sided, full_after_first_tf_lock, rest_value_tf_lock

PAYLOAD_QUERIED_NEW = "queried_new"
PAYLOAD_ORIGINAL = "original"
PAYLOAD_OTHER_COMPETITOR = "other_competitor"
PAYLOAD_OFF_INVENTORY = "off_inventory"
PAYLOAD_ORIGINAL_ABSENT = "original_absent_copy"

STAGE_CANNOT_IDENTIFY = "cannot_identify_queried"
STAGE_WEAK_IDENTIFY = "weak_identify_other_wins"
STAGE_IDENTIFIED_OFF_VOCAB = "identified_inventory_off_vocab_wins"
STAGE_SELECTED_NO_CONTINUE = "selected_cannot_continue_payload"
STAGE_PAYLOAD_SUFFIX_FAIL = "payload_ok_suffix_fail"
STAGE_SUCCESS = "copies_and_closes"
STAGE_OTHER = "other"


def token_rank(logits: list[float], token: int) -> int:
    if token < 0 or token >= len(logits):
        raise RuntimeError(("token out of logit range", token, len(logits)))
    value = logits[token]
    return 1 + sum(1 for other in logits if other > value)


def _span_prefix_match(emitted: list[int], span: list[int]) -> bool:
    return bool(span) and emitted[: len(span)] == list(span)


def classify_payload_origin(item: dict, row: dict) -> dict[str, Any]:
    """Classify the greedy value span without mixing original into competitor."""
    emitted = [int(tok) for tok in row.get("emitted", [])]
    original = [int(tok) for tok in item["original_target_span"]] if item.get("original_target_span") else None
    queried_span = [int(tok) for tok in item["target_span"]] if item.get("target_span") is not None else []
    original_is_previous = bool(original) and original != queried_span
    rendered = recover_rendered_pairs(item) if item.get("kind") == "keyed" else []
    query_key = int(item["query_key"]) if item.get("query_key") is not None else None
    matched = None
    for pair in rendered:
        if _span_prefix_match(emitted, pair["value"]):
            matched = pair
            break
    stuck_old = original_is_previous and _span_prefix_match(emitted, original)
    replacement = [int(tok) for tok in item["replacement_span"]] if item.get("replacement_span") else None
    copied_replacement = bool(replacement) and _span_prefix_match(emitted, replacement)
    if matched is None:
        origin = PAYLOAD_ORIGINAL_ABSENT if stuck_old else PAYLOAD_OFF_INVENTORY
    elif query_key is not None and matched["original_key"] == query_key:
        origin = PAYLOAD_QUERIED_NEW
    elif original_is_previous and matched["value"] == original:
        origin = PAYLOAD_ORIGINAL
    else:
        origin = PAYLOAD_OTHER_COMPETITOR
    return {
        "payload_origin": origin,
        "emitted_original_value_span": stuck_old,
        "copied_replacement_span": copied_replacement,
        "emit_render_index": None if matched is None else matched["render_index"],
        "emit_key": None if matched is None else matched["original_key"],
        "n_rendered": len(rendered),
    }


def inventory_first_candidates(item: dict) -> list[dict[str, Any]]:
    original = [int(tok) for tok in item["original_target_span"]] if item.get("original_target_span") else None
    queried_span = [int(tok) for tok in item["target_span"]] if item.get("target_span") is not None else []
    original_is_previous = bool(original) and original != queried_span
    query_key = int(item["query_key"])
    out = []
    for pair in recover_rendered_pairs(item):
        if not pair["value"]:
            continue
        role = PAYLOAD_QUERIED_NEW if pair["original_key"] == query_key else PAYLOAD_OTHER_COMPETITOR
        if original_is_previous and pair["value"] == original:
            role = PAYLOAD_ORIGINAL
        out.append({
            "role": role,
            "token": int(pair["value"][0]),
            "key": int(pair["original_key"]),
            "render_index": pair["render_index"],
            "value": list(pair["value"]),
        })
    return out


def inventory_first_token_snapshot(item: dict, logits: list[float], greedy_token: int) -> dict[str, Any]:
    candidates = inventory_first_candidates(item)
    scored = []
    for candidate in candidates:
        token = candidate["token"]
        scored.append({
            **{key: value for key, value in candidate.items() if key != "value"},
            "logit": float(logits[token]),
            "vocab_rank": token_rank(logits, token),
        })
    scored.sort(key=lambda row: (-row["logit"], row["render_index"]))
    for index, row in enumerate(scored):
        row["inventory_rank"] = index + 1
    queried = next((row for row in scored if row["role"] == PAYLOAD_QUERIED_NEW), None)
    original = next((row for row in scored if row["role"] == PAYLOAD_ORIGINAL), None)
    others = [row for row in scored if row["role"] != PAYLOAD_QUERIED_NEW]
    best_other = others[0] if others else None
    queried_logit = None if queried is None else queried["logit"]
    best_other_logit = None if best_other is None else best_other["logit"]
    greedy_role = PAYLOAD_OFF_INVENTORY
    greedy_match = next((row for row in scored if row["token"] == greedy_token), None)
    if greedy_match is not None:
        greedy_role = greedy_match["role"]
    if queried is None:
        signal = "missing_queried_candidate"
    elif queried["vocab_rank"] == 1:
        signal = "rank1_vocab"
    elif queried["inventory_rank"] == 1:
        signal = "rank1_inventory_not_vocab"
    elif queried["inventory_rank"] == 2:
        signal = "runner_up_inventory"
    else:
        signal = "present_not_competitive"
    return {
        "greedy_token": int(greedy_token),
        "greedy_role": greedy_role,
        "n_inventory_firsts": len(scored),
        "unique_inventory_firsts": len({row["token"] for row in scored}),
        "queried_token": None if queried is None else queried["token"],
        "queried_vocab_rank": None if queried is None else queried["vocab_rank"],
        "queried_inventory_rank": None if queried is None else queried["inventory_rank"],
        "queried_logit": queried_logit,
        "original_token": None if original is None else original["token"],
        "original_vocab_rank": None if original is None else original["vocab_rank"],
        "original_inventory_rank": None if original is None else original["inventory_rank"],
        "best_other_role": None if best_other is None else best_other["role"],
        "best_other_token": None if best_other is None else best_other["token"],
        "margin_vs_best_other_inventory": (
            None if queried_logit is None or best_other_logit is None else queried_logit - best_other_logit
        ),
        "queried_signal": signal,
        "candidates": scored,
    }


def mechanism_stage(item: dict, row: dict, origin: dict, snapshot: dict | None) -> str:
    info = classify_item_row(item, row)
    if info["value_ok"] and info["free_exact"]:
        return STAGE_SUCCESS
    if info["value_ok"] and not info["free_exact"]:
        return STAGE_PAYLOAD_SUFFIX_FAIL
    if info["first_ok"] and not info["value_ok"]:
        return STAGE_SELECTED_NO_CONTINUE
    if snapshot:
        signal = snapshot.get("queried_signal")
        if signal == "rank1_inventory_not_vocab":
            return STAGE_IDENTIFIED_OFF_VOCAB
        if signal in {"runner_up_inventory", "present_not_competitive"}:
            return STAGE_WEAK_IDENTIFY
        if signal in {"missing_queried_candidate", "rank1_vocab"}:
            if signal == "rank1_vocab":
                return STAGE_SELECTED_NO_CONTINUE
            return STAGE_CANNOT_IDENTIFY
        return STAGE_CANNOT_IDENTIFY
    if origin["payload_origin"] in {PAYLOAD_ORIGINAL, PAYLOAD_OTHER_COMPETITOR, PAYLOAD_ORIGINAL_ABSENT}:
        return STAGE_WEAK_IDENTIFY
    return STAGE_OTHER


def compact_query_swap_row(item: dict, row: dict, origin: dict, snapshot: dict | None) -> dict[str, Any]:
    info = classify_item_row(item, row)
    rest = rest_value_tf_lock(item, row)
    return {
        "variant": item.get("variant"),
        "pair_count": item.get("pair_count"),
        "value_length": item.get("value_length"),
        "isolation_transform": item.get("isolation_transform"),
        "payload_origin": origin["payload_origin"],
        "emitted_original_value_span": origin["emitted_original_value_span"],
        "first_ok": info["first_ok"],
        "value_ok": info["value_ok"],
        "free_exact": info["free_exact"],
        "fail": info["fail"],
        "target_rank": info["target_rank"],
        "rest_value_tf_lock": rest,
        "full_after_first_tf_lock": full_after_first_tf_lock(item, row),
        "mechanism_stage": mechanism_stage(item, row, origin, snapshot),
        "queried_signal": None if snapshot is None else snapshot.get("queried_signal"),
        "queried_vocab_rank": None if snapshot is None else snapshot.get("queried_vocab_rank"),
        "queried_inventory_rank": None if snapshot is None else snapshot.get("queried_inventory_rank"),
        "original_vocab_rank": None if snapshot is None else snapshot.get("original_vocab_rank"),
        "original_inventory_rank": None if snapshot is None else snapshot.get("original_inventory_rank"),
        "greedy_role": None if snapshot is None else snapshot.get("greedy_role"),
        "margin_vs_best_other_inventory": None if snapshot is None else snapshot.get("margin_vs_best_other_inventory"),
        "emit_render_index": origin["emit_render_index"],
        "query_index": item.get("query_index"),
        "original_query_key": item.get("original_query_key"),
        "query_key": item.get("query_key"),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(median(values))


def summarize_scored(items: list[dict], rows: list[dict], *, _nested: bool = False) -> dict:
    classified = [classify_item_row(item, row) for item, row in zip(items, rows)]
    origins = [classify_payload_origin(item, row) for item, row in zip(items, rows)]
    snapshots = [row.get("inventory_first_tokens") for row in rows]
    follow = 0
    stuck_old = 0
    queried = competitor = off_inventory = 0
    rest_defined = rest_lock = full_lock = 0
    origin_counts = Counter(origin["payload_origin"] for origin in origins)
    stages = Counter()
    signals = Counter()
    queried_vocab_ranks: list[int] = []
    queried_inventory_ranks: list[int] = []
    original_vocab_ranks: list[int] = []
    margins: list[float] = []
    queried_rank1_when_not_followed = 0
    not_followed = 0
    for item, row, info, origin, snapshot in zip(items, rows, classified, origins, snapshots):
        if origin["emitted_original_value_span"]:
            stuck_old += 1
        if item.get("isolation_transform") == "query_swap" and info["value_ok"]:
            follow += 1
        if item.get("kind") == "keyed":
            source = classify_emission(item, row)["source"]
            queried += int(source == "queried")
            competitor += int(source == "competitor")
            off_inventory += int(source == "off_inventory")
            rest = rest_value_tf_lock(item, row)
            if rest is not None:
                rest_defined += 1
                rest_lock += int(rest)
            full_lock += int(full_after_first_tf_lock(item, row))
        stage = mechanism_stage(item, row, origin, snapshot)
        stages[stage] += 1
        if snapshot:
            signal = snapshot.get("queried_signal")
            if signal:
                signals[signal] += 1
            rank = snapshot.get("queried_vocab_rank")
            inv = snapshot.get("queried_inventory_rank")
            orig_rank = snapshot.get("original_vocab_rank")
            margin = snapshot.get("margin_vs_best_other_inventory")
            if rank is not None:
                queried_vocab_ranks.append(int(rank))
            if inv is not None:
                queried_inventory_ranks.append(int(inv))
            if orig_rank is not None:
                original_vocab_ranks.append(int(orig_rank))
            if margin is not None:
                margins.append(float(margin))
            if item.get("isolation_transform") == "query_swap" and not info["value_ok"]:
                not_followed += 1
                queried_rank1_when_not_followed += int(rank == 1)
    n = len(classified)
    pair_counts = {int(item["pair_count"]) for item in items if item.get("kind") == "keyed" and item.get("pair_count")}
    chance = (1.0 / next(iter(pair_counts))) if len(pair_counts) == 1 else None
    value_ok = sum(info["value_ok"] for info in classified)
    p_value = None
    if chance is not None and n:
        p_value = binomial_two_sided(n, value_ok, chance)
    is_query_swap = any(item.get("isolation_transform") == "query_swap" for item in items)
    has_original = any(
        item.get("original_target_span") and list(item["original_target_span"]) != list(item.get("target_span") or [])
        for item in items
    )
    by_pair = None
    if not _nested and len(pair_counts) > 1:
        by_pair = {}
        for pair_count in sorted(pair_counts):
            indices = [i for i, item in enumerate(items) if int(item.get("pair_count", -1)) == pair_count]
            by_pair[str(pair_count)] = summarize_scored(
                [items[i] for i in indices],
                [rows[i] for i in indices],
                _nested=True,
            )
    return {
        "n": n,
        "first_ok": sum(info["first_ok"] for info in classified),
        "tf_value": sum(info["tf_value"] for info in classified),
        "value_ok": value_ok,
        "free_exact": sum(info["free_exact"] for info in classified),
        "tf_exact": sum(info["tf_exact"] for info in classified),
        "fail": {key: sum(info["fail"] == key for info in classified) for key in sorted({info["fail"] for info in classified})},
        "query_swap_follow_new_value": follow if is_query_swap else None,
        "emitted_original_value_span": stuck_old if has_original else None,
        "payload_origin": {
            PAYLOAD_QUERIED_NEW: origin_counts[PAYLOAD_QUERIED_NEW],
            PAYLOAD_ORIGINAL: origin_counts[PAYLOAD_ORIGINAL],
            PAYLOAD_OTHER_COMPETITOR: origin_counts[PAYLOAD_OTHER_COMPETITOR],
            PAYLOAD_OFF_INVENTORY: origin_counts[PAYLOAD_OFF_INVENTORY],
            PAYLOAD_ORIGINAL_ABSENT: origin_counts[PAYLOAD_ORIGINAL_ABSENT],
        },
        "queried_copy": queried,
        "competitor_copy": competitor,
        "other_competitor_copy": origin_counts[PAYLOAD_OTHER_COMPETITOR],
        "off_inventory": off_inventory,
        "inventory_copy_rate": ((queried + competitor) / n) if n else None,
        "rest_value_tf_lock": rest_lock,
        "rest_value_tf_lock_defined": rest_defined,
        "full_after_first_tf_lock": full_lock,
        "chance_1_over_k": chance,
        "value_ok_rate": (value_ok / n) if n else None,
        "free_exact_rate": (sum(info["free_exact"] for info in classified) / n) if n else None,
        "binomial_two_sided_p_vs_1_over_k": p_value,
        "mechanism_stage": dict(stages),
        "queried_signal": dict(signals),
        "median_queried_vocab_rank": _median(queried_vocab_ranks),
        "median_queried_inventory_rank": _median(queried_inventory_ranks),
        "median_original_vocab_rank": _median(original_vocab_ranks),
        "median_margin_vs_best_other_inventory": _median(margins),
        "queried_vocab_rank1_when_not_followed": queried_rank1_when_not_followed if is_query_swap else None,
        "query_swap_not_followed": not_followed if is_query_swap else None,
        "copied_replacement_span": sum(origin["copied_replacement_span"] for origin in origins),
        "by_pair_count": by_pair,
    }
