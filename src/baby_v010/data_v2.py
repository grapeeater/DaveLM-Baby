from __future__ import annotations

"""Baby v0.10 foundation v2 generator.

v1R2 made the final retrieval problem active from the first structured update.
This module keeps the same anti-shortcut invariants but exposes a genuine
scaffold: one-token induction first, short one/two-key retrieval second, and
the full multi-key/multi-token task only after the primitive is trained.
Targets are sampled from the frozen language-derived span bank on every item;
there is no finite answer inventory.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from .data import (
    BOS,
    EOS,
    LANG_TRAIN,
    VOCAB,
    Banks,
    build_banks,
    marker_families,
    read_u16,
    sample_span,
    separator_sets,
)


def _sample(rng: random.Random, pool: list[int] | tuple[int, ...], n: int) -> list[int]:
    if n < 0 or n > len(pool):
        raise RuntimeError(f"invalid sample n={n} pool={len(pool)}")
    return rng.sample(list(pool), n)


def _surface(banks: Banks, heldout: bool, rng: random.Random) -> tuple[tuple[int, ...], int]:
    train, holdout = marker_families(banks)
    families = holdout if heldout else train
    family = rng.choice(families)
    train_sep, holdout_sep = separator_sets(banks)
    sep = rng.choice(holdout_sep if heldout else train_sep)
    return family, sep


def _reserved_surface_tokens(banks: Banks) -> set[int]:
    train, holdout = marker_families(banks)
    train_sep, holdout_sep = separator_sets(banks)
    return {x for family in train + holdout for x in family} | set(train_sep) | set(holdout_sep)


def _filler(rng: random.Random, banks: Banks, n: int, banned: set[int]) -> list[int]:
    pool = [x for x in banks.key if x not in banned | _reserved_surface_tokens(banks)]
    return _sample(rng, pool, n)


def _length_choices(difficulty: str) -> tuple[int, ...]:
    if difficulty == "primitive":
        return (23, 31, 47, 63)
    if difficulty == "short":
        return (31, 47, 63, 79, 95)
    if difficulty == "full":
        return (47, 63, 79, 95, 127, 161)
    raise ValueError(difficulty)


def _induction(
    rng: random.Random,
    banks: Banks,
    *,
    difficulty: str,
    heldout: bool,
    length: int | None = None,
    low_prior: bool = False,
    broken: str | None = None,
) -> dict:
    if difficulty == "primitive":
        source_len = rng.choice((8, 12, 16, 24, 32))
        filler_total = rng.choice((0, 1, 2, 4))
    elif difficulty == "short":
        source_len = rng.choice((12, 16, 24, 32, 48))
        filler_total = rng.choice((2, 4, 8, 12))
    else:
        source_len = rng.choice((16, 24, 32, 48, 64, 80))
        filler_total = rng.choice((4, 8, 12, 20, 28))

    source = sample_span(rng, banks, source_len, low_prior=low_prior)
    j = rng.randrange(1, source_len)
    query = source[:j]
    context_source = source[:]
    if broken == "context":
        context_source = sample_span(rng, banks, source_len, low_prior=low_prior)
    elif broken == "order":
        context_source = source[:]
        rng.shuffle(context_source)
        if context_source == source:
            context_source = context_source[1:] + context_source[:1]

    markers, sep = _surface(banks, heldout, rng)
    banned = set(markers) | {sep}
    filler = _filler(rng, banks, filler_total, banned)
    left = rng.randrange(filler_total + 1)
    prefix, suffix = filler[:left], filler[left:]

    # Every variant puts the source before the query, but changes wrappers,
    # separators, and nuisance-token placement.
    variant = rng.randrange(6)
    if variant == 0:
        row = [BOS, *prefix, markers[0], *context_source, markers[1], *query, markers[2], *suffix]
    elif variant == 1:
        row = [BOS, markers[3], *prefix, *context_source, sep, *query, markers[4], *suffix]
    elif variant == 2:
        row = [BOS, *prefix, markers[1], *context_source, *suffix, markers[5], *query, sep]
    elif variant == 3:
        row = [BOS, markers[2], *context_source, *prefix, markers[0], *query, *suffix, sep]
    elif variant == 4:
        row = [BOS, *prefix, *context_source, markers[4], sep, *query, markers[1], *suffix]
    else:
        row = [BOS, markers[5], *prefix, *context_source, markers[3], *query, sep, *suffix]

    target_span = [source[j]]
    target = target_span + [sep, EOS]
    if length is not None and len(row) + len(target) > length:
        raise RuntimeError("induction construction exceeded requested length")
    return {
        "input": row,
        "target": target,
        "kind": "induction",
        "surface": "heldout" if heldout else "train",
        "difficulty": difficulty,
        "variant": f"induction_{variant}",
        "source": context_source,
        "target_span": target_span,
        "source_length": source_len,
        "decision_index": j,
        "low_prior": low_prior,
        "broken": broken,
    }


def _keyed(
    rng: random.Random,
    banks: Banks,
    *,
    difficulty: str,
    heldout: bool,
    length: int | None = None,
    low_prior: bool = False,
    broken: str | None = None,
    distractor: bool = False,
) -> dict:
    if difficulty == "primitive":
        pair_count = 1
        value_len = rng.choice((1, 2))
        default_length = rng.choice((23, 31, 47, 63))
    elif difficulty == "short":
        pair_count = rng.choice((1, 2))
        value_len = rng.choice((1, 2, 3, 4))
        default_length = rng.choice((31, 47, 63, 79, 95))
    else:
        pair_count = rng.choice((2, 3, 4))
        value_len = rng.choice((2, 3, 4, 5, 6, 8, 10))
        default_length = rng.choice((47, 63, 79, 95, 127, 161))
    if distractor:
        pair_count = 4

    markers, sep = _surface(banks, heldout, rng)
    key_pool = [x for x in banks.key if x not in _reserved_surface_tokens(banks)]
    keys = _sample(rng, key_pool, pair_count)
    records: list[tuple[int, list[int]]] = [
        (key, sample_span(rng, banks, value_len, low_prior=low_prior)) for key in keys
    ]
    query_idx = rng.randrange(pair_count)
    query_key, answer = records[query_idx]
    rendered = [(key, list(value)) for key, value in records]
    rng.shuffle(rendered)
    if broken == "context":
        replacement_pool = [x for x in key_pool if x not in keys]
        replacement = rng.choice(replacement_pool)
        rendered = [(replacement if key == query_key else key, value) for key, value in rendered]
    elif broken == "order":
        rendered = [(key, list(reversed(value)) if key == query_key else value) for key, value in rendered]

    used = set(markers) | {sep} | set(keys)
    used.update(x for _, value in rendered for x in value)
    target_len = value_len + 2
    query_wrapper = [markers[rng.randrange(6)], query_key, markers[rng.randrange(6)]]
    body: list[int] = []
    for key, value in rendered:
        body.extend([key, *value, sep])
    fixed = 1 + len(body) + len(query_wrapper) + target_len
    desired = length if length is not None else max(default_length, fixed)
    if desired < fixed:
        raise RuntimeError(("requested keyed length is infeasible", desired, fixed, pair_count, value_len))
    filler_total = max(0, desired - fixed)
    # Keep fixed panels within context while retaining variable answer position.
    if filler_total > 0:
        filler = _filler(rng, banks, filler_total, used)
        left = rng.randrange(filler_total + 1)
        filler_a, filler_b = filler[:left], filler[left:]
    else:
        filler_a, filler_b = [], []

    variant = rng.randrange(6)
    if variant == 0:
        row = [BOS, *filler_a, *body, *filler_b, markers[0], query_key, markers[1]]
    elif variant == 1:
        row = [BOS, markers[2], *filler_a, *body, markers[3], query_key, *filler_b]
    elif variant == 2:
        row = [BOS, *filler_a, markers[4], *body, *filler_b, query_key, markers[5]]
    elif variant == 3:
        row = [BOS, markers[5], *body, *filler_a, query_key, markers[0], *filler_b]
    elif variant == 4:
        row = [BOS, *body, markers[1], *filler_a, markers[3], query_key, *filler_b]
    else:
        row = [BOS, *filler_a, query_key, markers[2], *body, markers[4], *filler_b]
    target = answer + [sep, EOS]
    if len(row) + len(target) > 255:
        raise RuntimeError("keyed construction exceeds model context")
    if length is not None and len(row) + len(target) != length:
        raise RuntimeError(("keyed length mismatch", len(row) + len(target), length))
    return {
        "input": row,
        "target": target,
        "kind": "keyed",
        "surface": "heldout" if heldout else "train",
        "difficulty": difficulty,
        "variant": f"keyed_{variant}",
        "source": [x for key, value in records for x in ([key] + value)],
        "target_span": answer,
        "query_key": query_key,
        "query_index": query_idx,
        "pair_count": pair_count,
        "value_length": value_len,
        "low_prior": low_prior,
        "broken": broken,
        "distractor": distractor,
    }


def make_item(
    rng: random.Random,
    banks: Banks,
    *,
    difficulty: str = "full",
    kind: str | None = None,
    heldout: bool = False,
    length: int | None = None,
    low_prior: bool = False,
    broken: str | None = None,
    distractor: bool = False,
) -> dict:
    kind = kind or ("induction" if rng.random() < 0.30 else "keyed")
    # Keyed rows compute a feasible nuisance length after sampling their record
    # count and value length; forcing a length before that would create an
    # accidental construction failure. Explicit panel lengths remain exact.
    if kind == "induction":
        item = _induction(rng, banks, difficulty=difficulty, heldout=heldout, length=length, low_prior=low_prior, broken=broken)
    elif kind == "keyed":
        item = _keyed(rng, banks, difficulty=difficulty, heldout=heldout, length=length, low_prior=low_prior, broken=broken, distractor=distractor)
    else:
        raise ValueError(kind)
    if len(item["input"]) + len(item["target"]) > 255:
        raise RuntimeError("v2 item exceeds model context")
    return item


def _unique_keyed(rng: random.Random, banks: Banks, n: int, **kwargs) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > n * 100:
            raise RuntimeError("could not construct unique keyed target panel" )
        item = make_item(rng, banks, kind="keyed", **kwargs)
        target = tuple(item["target_span"] )
        if target in seen:
            continue
        seen.add(target)
        out.append(item)
    return out

def build_panels(banks: Banks, seed: int = 102000) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    panels: dict[str, list[dict]] = {}
    panels["primitive_induction"] = [make_item(rng, banks, difficulty="primitive", kind="induction") for _ in range(64)]
    panels["primitive_keyed"] = _unique_keyed(rng, banks, 64, difficulty="primitive")
    panels["short_keyed"] = _unique_keyed(rng, banks, 64, difficulty="short")
    panels["same_surface_novel"] = _unique_keyed(rng, banks, 96, difficulty="full")
    panels["same_surface_induction"] = [make_item(rng, banks, difficulty="full", kind="induction") for _ in range(96)]
    panels["novel"] = panels["same_surface_novel"]
    panels["induction"] = panels["same_surface_induction"]
    panels["heldout_surface"] = _unique_keyed(rng, banks, 96, difficulty="full", heldout=True)
    panels["unseen_length"] = _unique_keyed(rng, banks, 64, difficulty="full", heldout=True, length=161)
    panels["low_prior"] = _unique_keyed(rng, banks, 64, difficulty="full", heldout=True, low_prior=True)
    panels["distractor"] = _unique_keyed(rng, banks, 64, difficulty="full", heldout=True, distractor=True)
    panels["broken_context"] = _unique_keyed(rng, banks, 64, difficulty="full", heldout=True, broken="context")
    panels["broken_order"] = _unique_keyed(rng, banks, 64, difficulty="full", heldout=True, broken="order")
    panels["all_intact"] = (
        panels["primitive_induction"]
        + panels["primitive_keyed"]
        + panels["short_keyed"]
        + panels["same_surface_novel"]
        + panels["same_surface_induction"]
        + panels["heldout_surface"]
        + panels["unseen_length"]
        + panels["low_prior"]
        + panels["distractor"]
    )
    return panels


def audit_panels(panels: dict[str, list[dict]], banks: Banks) -> dict:
    train_families, heldout_families = marker_families(banks)
    train_markers = {x for family in train_families for x in family}
    heldout_markers = {x for family in heldout_families for x in family}
    intact = panels["all_intact"]
    target_spans = [tuple(x["target_span"]) for x in intact]
    keyed_targets = [tuple(x["target_span"]) for x in intact if x["kind"] == "keyed"]
    input_sequences = [tuple(x["input"]) for x in intact]
    violations = sum(
        any(pair in banks.bigrams for pair in zip(x["target_span"], x["target_span"][1:]))
        for x in intact
    )
    heldout_rows = panels["heldout_surface"] + panels["unseen_length"] + panels["low_prior"] + panels["distractor"] + panels["broken_context"] + panels["broken_order"]
    train_rows = panels["primitive_induction"] + panels["primitive_keyed"] + panels["short_keyed"] + panels["same_surface_novel"] + panels["same_surface_induction"]
    def marker_overlap(row: dict, marker_set: set[int]) -> bool:
        return bool(set(row["input"]) & marker_set)
    train_surface_overlap = sum(marker_overlap(row, heldout_markers) for row in train_rows)
    heldout_surface_overlap = sum(marker_overlap(row, train_markers) for row in heldout_rows)
    return {
        "panel_counts": {key: len(value) for key, value in panels.items()},
        "duplicate_input_sequences": len(input_sequences) - len(set(input_sequences)),
        "duplicate_target_spans": len(target_spans) - len(set(target_spans)),
        "duplicate_keyed_target_spans": len(keyed_targets) - len(set(keyed_targets)),
        "target_internal_language_bigram_violations": violations,
        "marker_sets_disjoint": not bool(train_markers & heldout_markers),
        "train_marker_count": len(train_markers),
        "heldout_marker_count": len(heldout_markers),
        "train_rows_using_heldout_markers": train_surface_overlap,
        "heldout_rows_using_train_markers": heldout_surface_overlap,
        "markers_disjoint_from_span_bank": not bool((train_markers | heldout_markers) & set(banks.span)),
        "span_bank_size": len(banks.span),
        "key_bank_size": len(banks.key),
        "language_sha256": hashlib.sha256(LANG_TRAIN.read_bytes()).hexdigest(),
        "protected_material_opened": False,
    }


def save_panels(out: Path, seed: int = 102000) -> None:
    language = read_u16(LANG_TRAIN)
    banks = build_banks(language)
    panels = build_panels(banks, seed)
    audit = audit_panels(panels, banks)
    if audit["duplicate_input_sequences"] or audit["duplicate_keyed_target_spans"] or audit["target_internal_language_bigram_violations"]:
        raise RuntimeError(json.dumps(audit, indent=2))
    if audit["train_rows_using_heldout_markers"] or audit["heldout_rows_using_train_markers"]:
        raise RuntimeError(json.dumps(audit, indent=2))
    out.mkdir(parents=True, exist_ok=True)
    panels_path = out / "panels.json"
    panels_path.write_text(json.dumps(panels, indent=2) + "\n", encoding="utf-8")
    (out / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "protocol": "BABY_V010_FOUNDATION_V2",
        "panel_seed": seed,
        "language_stream": str(LANG_TRAIN),
        "language_sha256": audit["language_sha256"],
        "protected_material_opened": False,
        "panels_sha256": hashlib.sha256(panels_path.read_bytes()).hexdigest(),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PANELS_FROZEN", "audit": audit, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=102000)
    args = parser.parse_args()
    save_panels(args.out, args.seed)
