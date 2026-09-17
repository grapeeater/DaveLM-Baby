"""D1 query-presence localization: pure helpers plus read-only torch tracing.

No optimizer. Default ``BabyVNextLM.forward`` is unchanged. Tracing uses forward
hooks and the existing reference attention backend.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from .query_locality import gap_of

SITES_FINAL = ("final",)
N_LAYERS = 12
IDENTITY_MAX_ABS = 1e-5
UNIFORM_TRACK_MULTIPLIER = 5.0
SMOKE_SDPA_MAX_ABS = 1e-3
POSITIVE_COSINE_MAX = 0.995
POSITIVE_FLIP_MIN = 0.25
LONG_RESID_COSINE_MIN = 0.999
LONG_LOGIT_COSINE_MIN = 0.995
LONG_FLIP_B_MAX = 0.05
LONG_FLIP_A_MIN = 0.20
COMPOSITION_TRACK_MIN = 0.50


def loc_class(item: dict) -> str:
    inp = [int(t) for t in item["input"]]
    query_key = int(item["query_key"])
    if inp[-1] == query_key:
        return "last_is_query"
    if len(inp) > 1 and inp[-2] == query_key:
        return "prev_is_query"
    return "query_deeper"


def gap_bucket(gap: int) -> str:
    if gap <= 1:
        return "g0_1"
    if gap <= 3:
        return "g2_3"
    if gap <= 12:
        return "g4_12"
    if gap <= 30:
        return "g13_30"
    return "g31p"


def twin_diffs(left: dict, right: dict) -> list[int]:
    a = [int(t) for t in left["input"]]
    b = [int(t) for t in right["input"]]
    if len(a) != len(b):
        return list(range(max(len(a), len(b))))
    return [i for i, (x, y) in enumerate(zip(a, b)) if x != y]


def assert_query_only_twins(items: Iterable[dict]) -> None:
    bodies: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        bodies[item["body_id"]].append(item)
    for body_id, group in bodies.items():
        if len(group) != int(group[0]["pair_count"]):
            raise RuntimeError(("incomplete body", body_id, len(group)))
        query_pos = int(group[0]["query_position"])
        for left in group:
            if int(left["query_position"]) != query_pos:
                raise RuntimeError(("query_position moved inside body", body_id))
            for right in group:
                if left is right:
                    continue
                diffs = twin_diffs(left, right)
                if diffs != [query_pos]:
                    raise RuntimeError(("twins are not query-token only", body_id, diffs))


def directed_pairs(group: list[dict]) -> list[tuple[dict, dict]]:
    ordered = sorted(group, key=lambda row: int(row["query_index"]))
    pairs = []
    for donor in ordered:
        for recipient in ordered:
            if donor is recipient:
                continue
            pairs.append((donor, recipient))
    return pairs


def body_key_positions(item: dict) -> tuple[list[int], int | None]:
    from .isolation_transforms import locate_query, recover_rendered_pairs

    rendered = recover_rendered_pairs(item)
    positions = [int(row["start"]) for row in rendered]
    _body_pos, _query_pos = locate_query(item)
    queried = None
    query_key = int(item["query_key"])
    for row in rendered:
        if int(row["original_key"]) == query_key or int(row["key"]) == query_key:
            queried = int(row["start"])
            break
    return positions, queried


def attention_index_masses(
    gen_row: list[float],
    *,
    gen_pos: int,
    query_pos: int,
    body_keys: list[int],
    queried_body: int | None,
) -> dict:
    """``gen_row`` is attention mass from generation position over keys 0..T-1."""
    time = gen_pos + 1
    if len(gen_row) < time:
        raise ValueError("gen_row shorter than causal prefix")
    prefix = gen_row[:time]
    total = sum(prefix)
    uniform = 1.0 / time
    def at(index: int) -> float:
        if 0 <= index < time:
            return float(prefix[index])
        return 0.0

    body_mass = sum(at(index) for index in body_keys)
    return {
        "sum": float(total),
        "uniform": uniform,
        "query_slot": at(query_pos),
        "prev": at(gen_pos - 1) if gen_pos > 0 else 0.0,
        "self": at(gen_pos),
        "queried_body_key": at(queried_body) if queried_body is not None else 0.0,
        "all_body_keys": float(body_mass),
        "tracks_query": at(query_pos) >= UNIFORM_TRACK_MULTIPLIER * uniform,
        "tracks_prev": (at(gen_pos - 1) if gen_pos > 0 else 0.0)
        >= UNIFORM_TRACK_MULTIPLIER * uniform,
    }


def cosine_l2(left: list[float], right: list[float]) -> dict:
    if len(left) != len(right):
        raise ValueError("vector length mismatch")
    dot = 0.0
    n1 = 0.0
    n2 = 0.0
    acc = 0.0
    max_abs = 0.0
    for a, b in zip(left, right):
        dot += a * b
        n1 += a * a
        n2 += b * b
        diff = a - b
        acc += diff * diff
        max_abs = max(max_abs, abs(diff))
    denom = math.sqrt(n1) * math.sqrt(n2)
    cosine = 0.0 if denom == 0.0 else dot / denom
    return {"cosine": cosine, "l2": math.sqrt(acc), "max_abs": max_abs}


def skip_identity(max_abs: float) -> bool:
    return max_abs < IDENTITY_MAX_ABS


def flip_toward_donor(before: int, after: int, donor_gold: int) -> bool:
    return after == donor_gold and before != donor_gold


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def adjudicate(report: dict) -> dict:
    """Apply the frozen D1 rules. ``report`` is the M1–M3 summary object."""
    validity = []
    invalid_reasons = []
    preflight = report.get("preflight", {})
    if not preflight.get("parent_sha_match"):
        invalid_reasons.append("parent_sha")
    if not preflight.get("diagnostic_sha_match"):
        invalid_reasons.append("diagnostic_sha")
    if preflight.get("protected_material_opened"):
        invalid_reasons.append("protected")
    if not preflight.get("twins_query_only"):
        invalid_reasons.append("twins")
    sdpa = preflight.get("sdpa_reference_max_abs")
    if sdpa is None or sdpa >= SMOKE_SDPA_MAX_ABS:
        invalid_reasons.append("sdpa_reference")
    validity.append(not invalid_reasons)

    pos = report["positive_control"]
    pos_cosine_ok = pos["median_full_logit_cosine"] <= POSITIVE_COSINE_MAX
    pos_flip = max(pos["flip_toward_donor_final"], pos["flip_toward_donor_best_block"])
    pos_flip_ok = pos_flip >= POSITIVE_FLIP_MIN
    if not pos_cosine_ok:
        invalid_reasons.append("positive_cosine")
    if not pos_flip_ok:
        invalid_reasons.append("positive_flip")
    valid = not invalid_reasons

    long = report["long_gap"]
    flip_sites = [long["flip_toward_donor_final"], *long["flip_toward_donor_block"], *long["flip_toward_donor_attn"]]
    b = (
        long["median_final_residual_cosine"] >= LONG_RESID_COSINE_MIN
        and long["median_full_logit_cosine"] >= LONG_LOGIT_COSINE_MIN
        and all(rate <= LONG_FLIP_B_MAX for rate in flip_sites)
    )
    a = long["flip_toward_donor_final"] >= LONG_FLIP_A_MIN
    composition = (
        (not a)
        and long["fraction_rows_any_query_tracking_head"] >= COMPOSITION_TRACK_MIN
        and long["median_final_residual_cosine"] >= LONG_RESID_COSINE_MIN
    )
    if not valid:
        verdict = "INVALID"
    elif a:
        verdict = "A"
    elif composition:
        verdict = "COMPOSITION"
    elif b:
        verdict = "B"
    else:
        verdict = "MIXED"

    return {
        "protocol": "V010_QUERY_PRESENCE_D1",
        "valid": valid,
        "invalid_reasons": invalid_reasons,
        "positive_control": {
            "median_full_logit_cosine_ok": pos_cosine_ok,
            "flip_ok": pos_flip_ok,
            "flip_used": pos_flip,
        },
        "verdict": verdict,
        "hypothesis": {
            "H_B": verdict == "B",
            "H_A": verdict == "A",
            "H_C": verdict == "COMPOSITION",
        },
        "trained": False,
        "gates_changed": False,
        "protected_material_opened": False,
        "authoritative_parent_unchanged": True,
    }
