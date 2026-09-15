from __future__ import annotations

"""Fresh structured-data generator for Baby v0.10 foundation v1.

The generator deliberately uses token IDs rather than natural-language strings
for structured rows. This lets the experiment measure contextual identity
transport without accidentally making a finite word inventory or a particular
English phrase the answer key. The language stream remains a separate retention
source.
"""

import argparse
import hashlib
import json
import random
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BOS, EOS, VOCAB = 2, 3, 1024
MIN_TOKEN = 5
LANG_TRAIN = Path(r"C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\LANGUAGE_TRAIN_STREAM.u16")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_u16(path: Path) -> list[int]:
    values = array("H")
    with path.open("rb") as f:
        values.fromfile(f, path.stat().st_size // 2)
    if __import__("sys").byteorder != "little":
        values.byteswap()
    return list(values)


def token_counts(tokens: Iterable[int]) -> list[int]:
    counts = [0] * VOCAB
    for token in tokens:
        if 0 <= token < VOCAB:
            counts[token] += 1
    return counts


def bigram_set(tokens: list[int]) -> set[tuple[int, int]]:
    return set(zip(tokens, tokens[1:]))


@dataclass(frozen=True)
class Banks:
    span: tuple[int, ...]
    key: tuple[int, ...]
    counts: tuple[int, ...]
    bigrams: frozenset[tuple[int, int]]


def build_banks(language_tokens: list[int]) -> Banks:
    counts = token_counts(language_tokens)
    span = tuple(i for i in range(MIN_TOKEN, VOCAB) if 8 <= counts[i] <= 5000)
    key = tuple(i for i in range(MIN_TOKEN, VOCAB) if i not in set(span))
    if len(span) < 80 or len(key) < 80:
        raise RuntimeError(f"insufficient token banks: span={len(span)} key={len(key)}")
    return Banks(span, key, tuple(counts), frozenset(bigram_set(language_tokens)))


def marker_families(banks: Banks) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Allocate markers from the key bank so they can never be value tokens."""
    pool = list(banks.key)
    needed = 8 * 6 + 10
    if len(pool) < needed:
        raise RuntimeError("key bank too small for disjoint marker/separator allocation")
    train = tuple(tuple(pool[i * 6 : (i + 1) * 6]) for i in range(6))
    holdout_start = 36
    heldout = tuple(tuple(pool[holdout_start + i * 6 : holdout_start + (i + 1) * 6]) for i in range(2))
    return train, heldout


def separator_sets(banks: Banks) -> tuple[tuple[int, ...], tuple[int, ...]]:
    train, heldout = marker_families(banks)
    used = {x for family in train + heldout for x in family}
    pool = [x for x in banks.key if x not in used]
    return tuple(pool[:6]), tuple(pool[6:10])


def _sample_unique(rng: random.Random, pool: tuple[int, ...], n: int) -> list[int]:
    if n > len(pool):
        raise RuntimeError("sample larger than pool")
    return rng.sample(pool, n)


def sample_span(
    rng: random.Random,
    banks: Banks,
    length: int,
    *,
    low_prior: bool = False,
) -> list[int]:
    """Sample a novel span with no language-supported internal bigram."""
    candidates = [x for x in banks.span if (banks.counts[x] <= 60 if low_prior else True)]
    if len(candidates) < max(32, length * 2):
        candidates = list(banks.span)
    for _ in range(400):
        span = _sample_unique(rng, tuple(candidates), length)
        if all((a, b) not in banks.bigrams for a, b in zip(span, span[1:])):
            return span
    # The fallback is still checked; failing here is a generator error, not a
    # reason to weaken the audit.
    raise RuntimeError("could not draw an out-of-prior target span")


def _family(rng: random.Random, banks: Banks, heldout: bool) -> tuple[int, ...]:
    train, holdout = marker_families(banks)
    families = holdout if heldout else train
    return rng.choice(families)


def _separator(rng: random.Random, banks: Banks, heldout: bool) -> int:
    train, holdout = separator_sets(banks)
    return rng.choice(holdout if heldout else train)


def _filler(rng: random.Random, banks: Banks, n: int, forbidden: set[int]) -> list[int]:
    pool = tuple(x for x in banks.key if x not in forbidden)
    return _sample_unique(rng, pool, n)


def _induction(
    rng: random.Random,
    banks: Banks,
    *,
    length: int,
    heldout: bool,
    low_prior: bool,
    broken: str | None = None,
) -> dict:
    markers = _family(rng, banks, heldout)
    sep = _separator(rng, banks, heldout)
    # A long source followed by a repeated prefix. The visible wrapper varies
    # so answer onset cannot be tied to one marker identity.
    source_len = max(8, min(64, (length - 5) // 2))
    prefix_len = max(1, min(source_len - 1, length - source_len - 4))
    source = sample_span(rng, banks, source_len, low_prior=low_prior)
    query = source[:prefix_len]
    target = [source[prefix_len]]
    if broken == "order":
        source = source[:]
        rng.shuffle(source)
    if broken == "context":
        source = sample_span(rng, banks, source_len, low_prior=low_prior)
    variant = rng.randrange(4)
    if variant == 0:
        row = [BOS, markers[0], *source, markers[1], *query]
    elif variant == 1:
        row = [BOS, markers[2], *source, sep, markers[3], *query, markers[4]]
    elif variant == 2:
        row = [BOS, *source, markers[0], *query, sep, markers[5]]
    else:
        row = [BOS, markers[1], sep, *source, markers[4], *query]
    return {
        "input": row,
        "target": target + [sep, EOS],
        "kind": "induction",
        "surface": "heldout" if heldout else "train",
        "variant": f"induction_{variant}",
        "source": source,
        "target_span": target,
        "low_prior": low_prior,
        "broken": broken,
    }


def _keyed(
    rng: random.Random,
    banks: Banks,
    *,
    length: int,
    heldout: bool,
    low_prior: bool,
    broken: str | None = None,
) -> dict:
    markers = _family(rng, banks, heldout)
    sep = _separator(rng, banks, heldout)
    pair_count = rng.choice((2, 3, 4))
    value_len = rng.choice((2, 3, 4, 5, 6, 8, 10))
    keys = _sample_unique(rng, banks.key, pair_count + 3)
    records: list[tuple[int, list[int]]] = []
    for i in range(pair_count):
        value = sample_span(rng, banks, value_len, low_prior=low_prior)
        records.append((keys[i], value))
    query_idx = rng.randrange(pair_count)
    query_key, answer = records[query_idx]
    render_records = records[:]
    rng.shuffle(render_records)
    if broken == "context":
        replacement = _sample_unique(rng, banks.key, 1)[0]
        render_records = [(replacement if key == query_key else key, value) for key, value in render_records]
    if broken == "order":
        render_records = [
            (key, list(reversed(value)) if key == query_key else value)
            for key, value in render_records
        ]
    used = {x for key, value in render_records for x in ([key] + value)} | set(markers)
    filler_len = max(2, min(24, length - (pair_count * (value_len + 2) + value_len + 8)))
    filler_a = _filler(rng, banks, filler_len, used)
    filler_b = _filler(rng, banks, filler_len // 2, used | set(filler_a))
    intro = _sample_unique(rng, tuple(x for x in banks.key if x not in used), rng.choice((1, 2)))
    body: list[int] = []
    for key, value in render_records:
        body.extend((key, *value, sep))
    variant = rng.randrange(6)
    if variant == 0:
        row = [BOS, markers[0], *intro, *filler_a, *body, *filler_b, markers[1], query_key, markers[2]]
    elif variant == 1:
        row = [BOS, markers[3], *body, *filler_a, markers[4], query_key, markers[5], *filler_b]
    elif variant == 2:
        row = [BOS, *filler_a, markers[1], *body, markers[0], *filler_b, markers[5], query_key]
    elif variant == 3:
        row = [BOS, markers[2], *filler_a, *body, query_key, markers[4], *filler_b, markers[0]]
    elif variant == 4:
        row = [BOS, *body, markers[5], *filler_a, markers[3], query_key, *filler_b]
    else:
        row = [BOS, markers[4], *filler_a, query_key, markers[1], *body, markers[0], *filler_b]
    return {
        "input": row,
        "target": answer + [sep, EOS],
        "kind": "keyed",
        "surface": "heldout" if heldout else "train",
        "variant": f"keyed_{variant}",
        "source": [x for key, value in records for x in ([key] + value)],
        "target_span": answer,
        "query_key": query_key,
        "query_index": query_idx,
        "low_prior": low_prior,
        "broken": broken,
    }


def make_item(
    rng: random.Random,
    banks: Banks,
    *,
    mode: str = "train",
    length: int | None = None,
    kind: str | None = None,
    low_prior: bool = False,
    broken: str | None = None,
) -> dict:
    heldout = mode in {"heldout_surface", "unseen_length", "distractor"}
    if length is None:
        length = rng.choice((31, 47, 63, 79, 95, 127)) if heldout else rng.choice((23, 31, 47, 63, 79, 95))
    kind = kind or ("induction" if rng.random() < 0.25 else "keyed")
    if kind == "induction":
        item = _induction(rng, banks, length=length, heldout=heldout, low_prior=low_prior, broken=broken)
    elif kind == "keyed":
        item = _keyed(rng, banks, length=length, heldout=heldout, low_prior=low_prior, broken=broken)
    else:
        raise ValueError(kind)
    if len(item["input"]) + len(item["target"]) > 255:
        raise RuntimeError("structured item exceeds model context")
    return item


def build_panels(banks: Banks, seed: int = 101000) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    panels: dict[str, list[dict]] = {}
    panels["novel"] = [make_item(rng, banks, mode="heldout_surface", kind="keyed") for _ in range(96)]
    panels["induction"] = [make_item(rng, banks, mode="heldout_surface", kind="induction") for _ in range(96)]
    panels["heldout_surface"] = [make_item(rng, banks, mode="heldout_surface", kind="keyed") for _ in range(96)]
    panels["unseen_length"] = [make_item(rng, banks, mode="unseen_length", length=127, kind="keyed") for _ in range(64)]
    panels["low_prior"] = [make_item(rng, banks, mode="heldout_surface", kind="keyed", low_prior=True) for _ in range(64)]
    panels["distractor"] = [make_item(rng, banks, mode="distractor", kind="keyed") for _ in range(64)]
    panels["broken_context"] = [make_item(rng, banks, mode="heldout_surface", kind="keyed", broken="context") for _ in range(64)]
    panels["broken_order"] = [make_item(rng, banks, mode="heldout_surface", kind="keyed", broken="order") for _ in range(64)]
    panels["all_intact"] = panels["novel"] + panels["induction"] + panels["heldout_surface"] + panels["unseen_length"] + panels["low_prior"] + panels["distractor"]
    return panels


def audit_panels(panels: dict[str, list[dict]], banks: Banks) -> dict:
    train_families, heldout_families = marker_families(banks)
    train_markers = {x for family in train_families for x in family}
    heldout_markers = {x for family in heldout_families for x in family}
    target_spans = [tuple(x["target_span"]) for x in panels["all_intact"]]
    keyed_targets = [tuple(x["target_span"]) for x in panels["all_intact"] if x["kind"] == "keyed"]
    induction_targets = [tuple(x["target_span"]) for x in panels["all_intact"] if x["kind"] == "induction"]
    input_sequences = [tuple(x["input"]) for x in panels["all_intact"]]
    internal_bigram_violations = sum(
        any(pair in banks.bigrams for pair in zip(x["target_span"], x["target_span"][1:]))
        for x in panels["all_intact"]
    )
    return {
        "panel_counts": {key: len(value) for key, value in panels.items()},
        "duplicate_input_sequences": len(input_sequences) - len(set(input_sequences)),
        "duplicate_target_spans": len(target_spans) - len(set(target_spans)),
        "duplicate_keyed_target_spans": len(keyed_targets) - len(set(keyed_targets)),
        "induction_target_token_inventory": len(set(induction_targets)),
        "target_internal_language_bigram_violations": internal_bigram_violations,
        "marker_sets_disjoint": not bool(train_markers & heldout_markers),
        "train_marker_count": len(train_markers),
        "heldout_marker_count": len(heldout_markers),
        "markers_disjoint_from_span_bank": not bool((train_markers | heldout_markers) & set(banks.span)),
        "span_bank_size": len(banks.span),
        "key_bank_size": len(banks.key),
        "language_sha256": sha256(LANG_TRAIN),
        "protected_material_opened": False,
    }


def save_panels(out: Path, seed: int = 101000) -> None:
    language = read_u16(LANG_TRAIN)
    banks = build_banks(language)
    panels = build_panels(banks, seed)
    audit = audit_panels(panels, banks)
    out.mkdir(parents=True, exist_ok=True)
    (out / "panels.json").write_text(json.dumps(panels, indent=2) + "\n", encoding="utf-8")
    (out / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "protocol": "BABY_V010_FOUNDATION_V1",
        "panel_seed": seed,
        "language_stream": str(LANG_TRAIN),
        "language_sha256": audit["language_sha256"],
        "protected_material_opened": False,
        "panels_sha256": sha256(out / "panels.json"),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PANELS_FROZEN", "audit": audit, "manifest": manifest}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101000)
    args = parser.parse_args()
    save_panels(args.out, args.seed)
