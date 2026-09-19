"""Query-locality causal probe: read-only re-renderings of frozen S2 bodies.

The frozen artifacts show that queried first-token selection is above chance only
in the ``keyed_0``/``keyed_2`` renderings, which are exactly the renderings whose
template places the query key one token before the generation position. Variant
identity and query-to-generation distance are almost perfectly confounded in the
frozen panels, so this module builds paired re-renderings that move the distance
while holding the body, the query key, and the candidate inventory fixed.

Nothing here trains, mutates frozen panels, or opens protected material. The
transforms are pure functions over item dictionaries so they can be unit tested
without a GPU.
"""
from __future__ import annotations

import random
from typing import Sequence

ARMS = (
    "as_is",
    "append_query",
    "append_other_key",
    "append_unused_key",
    "append_filler_key_8",
    "append_filler_span_8",
    "append_filler_key_32",
    # Dose-response on transport distance from a known-good starting point: the
    # query key is appended adjacent to the generation position, then pushed away
    # by n nuisance tokens. Pure distance, one arm family, gold unchanged.
    "append_query_then_filler_1",
    "append_query_then_filler_2",
    "append_query_then_filler_4",
    "append_query_then_filler_8",
)


def gap_of(item: dict) -> int:
    """Tokens strictly between the query key and the generation position."""
    return len(item["input"]) - 1 - int(item["query_position"])


def render_keys(item: dict) -> list[int]:
    """Keys of the rendered pairs, aligned with ``candidate_heads``."""
    from .isolation_transforms import parse_records

    return [int(key) for key, _value in parse_records(item)]


def _forbidden(item: dict) -> set[int]:
    return {int(token) for token in item["input"]} | {
        int(token) for token in item["target"]
    }


def other_key(item: dict) -> tuple[int, int] | None:
    """A rendered key that is not the queried one, with its candidate index."""
    keys = render_keys(item)
    query_key = int(item["query_key"])
    for index, key in enumerate(keys):
        if key != query_key:
            return key, index
    return None


def build_arm(
    item: dict,
    arm: str,
    *,
    key_pool: Sequence[int],
    span_pool: Sequence[int],
    seed: int,
) -> dict | None:
    """Return ``{"input", "gold_index"}`` for ``arm``, or ``None`` if inapplicable.

    ``gold_index`` indexes ``item["candidate_heads"]``: it is the candidate the
    arm's query *should* select. Every arm keeps the body, the separators, and the
    candidate inventory byte-identical; only material appended after the original
    input differs. Appending is used instead of in-place moves so that no arm can
    accidentally delete or reorder the inventory being selected from.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    inp = [int(token) for token in item["input"]]
    query_index = int(item["query_index"])
    rng = random.Random(seed)
    banned = _forbidden(item)

    if arm == "as_is":
        return {"input": inp, "gold_index": query_index}
    if arm == "append_query":
        return {"input": inp + [int(item["query_key"])], "gold_index": query_index}
    if arm == "append_other_key":
        found = other_key(item)
        if found is None:
            return None
        key, index = found
        return {"input": inp + [key], "gold_index": index}
    if arm == "append_unused_key":
        pool = [token for token in key_pool if token not in banned]
        if not pool:
            return None
        return {"input": inp + [rng.choice(pool)], "gold_index": query_index}

    count = int(arm.rsplit("_", 1)[1])
    pool = [
        token
        for token in (span_pool if "span" in arm else key_pool)
        if token not in banned
    ]
    if len(pool) < count:
        return None
    prefix = [int(item["query_key"])] if arm.startswith("append_query_then") else []
    return {
        "input": inp + prefix + rng.sample(pool, count),
        "gold_index": query_index,
    }


def candidate_rank(candidate_logits: Sequence[float], index: int) -> int:
    """1-based rank of ``index`` among the candidate heads."""
    value = candidate_logits[index]
    return 1 + sum(1 for other in candidate_logits if other > value)


def bucket(gap: int) -> str:
    if gap <= 1:
        return "a_gap_0_1"
    if gap <= 3:
        return "b_gap_2_3"
    if gap <= 12:
        return "c_gap_4_12"
    if gap <= 30:
        return "d_gap_13_30"
    return "e_gap_31_plus"
