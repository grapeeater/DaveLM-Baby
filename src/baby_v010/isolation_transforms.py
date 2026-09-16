from __future__ import annotations

"""Copy-only isolation transforms of frozen foundation_v2 keyed items.

These transforms never write to `data/generated/foundation_v2/panels.json`.
They exist so a later checkpoint probe can separate:

- query-conditioned binding versus copy-a-present-span
- marker identity versus separator OOD
- value-present broken-key versus value-absent controls
- pair-count slices versus 1/K chance

Diagnostic panels are not graduation gates.
"""

import copy
import json
from pathlib import Path
from typing import Iterable

from .data import EOS, VOCAB
from .v2r4_provenance import BOS_TOKEN, FROZEN_PANELS_SHA256, require_frozen_file

MIN_TOKEN = 5
TRAIN_KEYED_PANELS = ("primitive_keyed", "short_keyed", "same_surface_novel")
HELDOUT_KEYED_PANELS = ("heldout_surface", "unseen_length", "low_prior", "distractor", "broken_context", "broken_order")


def assert_frozen_panels(path: Path) -> dict:
    return require_frozen_file(path, FROZEN_PANELS_SHA256, "panels")


def parse_records(item: dict) -> list[tuple[int, list[int]]]:
    pair_count = int(item["pair_count"])
    value_length = int(item["value_length"])
    source = [int(x) for x in item["source"]]
    expected = pair_count * (1 + value_length)
    if len(source) != expected:
        raise RuntimeError(("source length mismatch", len(source), expected, item.get("variant")))
    records: list[tuple[int, list[int]]] = []
    cursor = 0
    for _ in range(pair_count):
        key = source[cursor]
        value = source[cursor + 1 : cursor + 1 + value_length]
        records.append((key, value))
        cursor += 1 + value_length
    return records


def locate_query(item: dict) -> tuple[int | None, int | None]:
    inp = [int(x) for x in item["input"]]
    query_key = int(item["query_key"])
    span = [int(x) for x in item["target_span"]]
    positions = [index for index, token in enumerate(inp) if token == query_key]
    body_positions = [index for index in positions if inp[index + 1 : index + 1 + len(span)] == span]
    query_positions = [index for index in positions if index not in body_positions]
    body_pos = body_positions[0] if len(body_positions) == 1 else None
    query_pos = query_positions[0] if len(query_positions) == 1 else None
    return body_pos, query_pos


def find_body_range(item: dict) -> tuple[int, int] | None:
    inp = [int(x) for x in item["input"]]
    sep = int(item["target"][-2])
    ranges: list[tuple[int, int]] = []
    for key, value in parse_records(item):
        pattern = [key, *value, sep]
        found = False
        for index in range(0, len(inp) - len(pattern) + 1):
            if inp[index : index + len(pattern)] == pattern:
                ranges.append((index, index + len(pattern)))
                found = True
                break
        if not found:
            value_pattern = [*value, sep]
            for index in range(1, len(inp) - len(value_pattern) + 1):
                if inp[index : index + len(value_pattern)] == value_pattern:
                    ranges.append((index - 1, index + len(value_pattern)))
                    break
    if not ranges:
        return None
    return min(start for start, _ in ranges), max(end for _, end in ranges)


def extract_marker_tokens(item: dict) -> tuple[int, int]:
    if item.get("kind") != "keyed":
        raise RuntimeError("marker extraction is defined for keyed rows")
    inp = [int(x) for x in item["input"]]
    variant = item.get("variant")
    _, query_pos = locate_query(item)
    body_range = find_body_range(item)
    if query_pos is None or body_range is None:
        raise RuntimeError(("cannot extract markers", variant, query_pos, body_range))
    start, end = body_range
    if variant == "keyed_0":
        tokens = (inp[query_pos - 1], inp[query_pos + 1])
    elif variant == "keyed_1":
        tokens = (inp[1], inp[query_pos - 1])
    elif variant == "keyed_2":
        tokens = (inp[start - 1], inp[query_pos + 1])
    elif variant == "keyed_3":
        tokens = (inp[1], inp[query_pos + 1])
    elif variant == "keyed_4":
        tokens = (inp[end], inp[query_pos - 1])
    elif variant == "keyed_5":
        tokens = (inp[query_pos + 1], inp[end])
    else:
        raise RuntimeError(f"unknown keyed variant {variant}")
    return tokens


def harvest_separators(panels: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    train = sorted({int(item["target"][-2]) for name in TRAIN_KEYED_PANELS for item in panels[name]})
    heldout = sorted({int(item["target"][-2]) for name in ("heldout_surface",) for item in panels[name]})
    return tuple(train), tuple(heldout)


def harvest_markers(panels: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    train = sorted({tok for name in TRAIN_KEYED_PANELS for item in panels[name] for tok in extract_marker_tokens(item)})
    heldout = sorted({tok for item in panels["heldout_surface"] for tok in extract_marker_tokens(item)})
    return tuple(train), tuple(heldout)


def _copy_item(item: dict, **updates) -> dict:
    copied = copy.deepcopy(item)
    copied.update(updates)
    copied["diagnostic_only"] = True
    copied["frozen_gate_item"] = False
    return copied


def query_swap(item: dict) -> dict:
    if item.get("kind") != "keyed":
        raise RuntimeError("query_swap requires keyed items")
    if item.get("broken"):
        raise RuntimeError("query_swap is defined on intact items")
    records = parse_records(item)
    if len(records) < 2:
        raise RuntimeError("query_swap requires pair_count >= 2")
    query_key = int(item["query_key"])
    others = [(key, value) for key, value in records if key != query_key]
    if not others:
        raise RuntimeError("no alternate key")
    others.sort(key=lambda pair: pair[0])
    new_key, new_value = others[0]
    _, query_pos = locate_query(item)
    if query_pos is None:
        raise RuntimeError("could not locate query token")
    inp = [int(x) for x in item["input"]]
    inp[query_pos] = new_key
    sep = int(item["target"][-2])
    return _copy_item(
        item,
        input=inp,
        query_key=new_key,
        query_index=[key for key, _ in records].index(new_key),
        target_span=list(new_value),
        target=list(new_value) + [sep, EOS],
        isolation_transform="query_swap",
        original_query_key=query_key,
        original_target_span=list(item["target_span"]),
    )


def _map_markers(tokens: Iterable[int], train_markers: tuple[int, ...], heldout_markers: tuple[int, ...]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    used: set[int] = set()
    for token in tokens:
        if token in mapping:
            continue
        if token in train_markers:
            index = train_markers.index(token)
        else:
            index = token
        candidate = heldout_markers[index % len(heldout_markers)]
        if candidate in used and len(used) < len(heldout_markers):
            for option in heldout_markers:
                if option not in used:
                    candidate = option
                    break
        mapping[token] = candidate
        used.add(candidate)
    return mapping


def marker_swap_keep_sep(
    item: dict,
    train_markers: tuple[int, ...],
    heldout_markers: tuple[int, ...],
) -> dict:
    if item.get("surface") != "train":
        raise RuntimeError("marker_swap_keep_sep is defined on train-surface items")
    markers = extract_marker_tokens(item)
    mapping = _map_markers(markers, train_markers, heldout_markers)
    sep = int(item["target"][-2])
    records = parse_records(item)
    protected = {BOS_TOKEN, EOS, sep, int(item["query_key"])} | {key for key, _ in records} | {tok for _, value in records for tok in value}
    inp = [int(x) for x in item["input"]]
    rewritten = []
    for token in inp:
        if token in mapping and token not in protected:
            rewritten.append(mapping[token])
        else:
            rewritten.append(token)
    if rewritten == inp:
        raise RuntimeError("marker_swap_keep_sep made no change")
    return _copy_item(
        item,
        input=rewritten,
        isolation_transform="marker_swap_keep_sep",
        marker_map={str(src): dst for src, dst in mapping.items()},
        original_input=list(item["input"]),
    )


def sep_swap_keep_markers(
    item: dict,
    source_seps: tuple[int, ...],
    dest_seps: tuple[int, ...],
) -> dict:
    sep = int(item["target"][-2])
    if sep not in source_seps:
        raise RuntimeError(("separator not in source set", sep, source_seps))
    new_sep = dest_seps[source_seps.index(sep) % len(dest_seps)]
    if new_sep == sep:
        raise RuntimeError("sep_swap_keep_markers mapped to the same separator")
    inp = [new_sep if token == sep else int(token) for token in item["input"]]
    target = [int(x) for x in item["target"]]
    target[-2] = new_sep
    return _copy_item(
        item,
        input=inp,
        target=target,
        isolation_transform="sep_swap_keep_markers",
        original_separator=sep,
        new_separator=new_sep,
        original_target_span=list(item["target_span"]),
    )


def substitute_span(span: list[int], banned: set[int]) -> list[int]:
    if len(span) >= 2:
        reversed_span = list(reversed(span))
        if reversed_span != span and not set(reversed_span) & (banned - set(span)):
            return reversed_span
    replacement: list[int] = []
    occupied = set(banned)
    for token in span:
        candidate = token
        for _ in range(VOCAB):
            candidate = MIN_TOKEN + ((candidate + 1 - MIN_TOKEN) % (VOCAB - MIN_TOKEN))
            if candidate not in occupied and candidate not in {BOS_TOKEN, EOS}:
                replacement.append(candidate)
                occupied.add(candidate)
                break
        else:
            raise RuntimeError("could not substitute value span")
    if replacement == span:
        raise RuntimeError("substitute_span was a no-op")
    return replacement


def value_absent(item: dict) -> dict:
    if item.get("kind") != "keyed":
        raise RuntimeError("value_absent requires keyed items")
    inp = [int(x) for x in item["input"]]
    span = [int(x) for x in item["target_span"]]
    sep = int(item["target"][-2])
    records = parse_records(item)
    start = None
    for index in range(0, len(inp) - len(span) + 1):
        if inp[index : index + len(span)] == span and index + len(span) < len(inp) and inp[index + len(span)] == sep:
            start = index
            break
    if start is None:
        raise RuntimeError("could not find value span followed by separator")
    banned = {BOS_TOKEN, EOS, sep, int(item["query_key"])} | {key for key, _ in records} | set(span)
    new_span = substitute_span(span, banned)
    rewritten = inp[:]
    rewritten[start : start + len(span)] = new_span
    still_present = any(
        rewritten[index : index + len(span)] == span
        for index in range(0, len(rewritten) - len(span) + 1)
    )
    if still_present:
        raise RuntimeError("value_absent failed to remove the original span")
    return _copy_item(
        item,
        input=rewritten,
        isolation_transform="value_absent",
        replacement_span=new_span,
        original_target_span=span,
        broken="value_absent",
    )


def pair_count_slice(items: list[dict], pair_count: int) -> list[dict]:
    return [
        _copy_item(item, isolation_transform=f"pair_count_{pair_count}")
        for item in items
        if item.get("kind") == "keyed" and int(item["pair_count"]) == pair_count
    ]


def _safe_map(items: list[dict], fn, **kwargs) -> list[dict]:
    out = []
    for item in items:
        try:
            out.append(fn(item, **kwargs) if kwargs else fn(item))
        except RuntimeError:
            continue
    return out


def build_isolation_panels(panels: dict) -> dict[str, list[dict]]:
    train_markers, heldout_markers = harvest_markers(panels)
    train_seps, heldout_seps = harvest_separators(panels)
    novel = panels["same_surface_novel"]
    short = panels["short_keyed"]
    heldout = panels["heldout_surface"]
    broken = panels["broken_context"]
    isolation = {
        "query_swap_same_surface_novel": _safe_map([item for item in novel if int(item["pair_count"]) >= 2], query_swap),
        "query_swap_short_keyed": _safe_map([item for item in short if int(item["pair_count"]) >= 2], query_swap),
        "marker_swap_keep_sep_same_surface_novel": _safe_map(
            novel, marker_swap_keep_sep, train_markers=train_markers, heldout_markers=heldout_markers
        ),
        "sep_swap_keep_markers_same_surface_novel": _safe_map(
            novel, sep_swap_keep_markers, source_seps=train_seps, dest_seps=heldout_seps
        ),
        "value_absent_same_surface_novel": _safe_map(novel, value_absent),
        "value_absent_heldout_surface": _safe_map(heldout, value_absent),
        "value_absent_broken_context": _safe_map(broken, value_absent),
        "same_surface_novel_pair_count_2": pair_count_slice(novel, 2),
        "same_surface_novel_pair_count_3": pair_count_slice(novel, 3),
        "same_surface_novel_pair_count_4": pair_count_slice(novel, 4),
        "short_keyed_pair_count_1": pair_count_slice(short, 1),
        "short_keyed_pair_count_2": pair_count_slice(short, 2),
    }
    isolation["meta"] = {
        "train_markers": list(train_markers),
        "heldout_markers": list(heldout_markers),
        "train_separators": list(train_seps),
        "heldout_separators": list(heldout_seps),
        "protected_material_opened": False,
        "frozen_panels_mutated": False,
        "diagnostic_only": True,
        "panel_counts": {key: len(value) if isinstance(value, list) else value for key, value in isolation.items() if key != "meta"},
    }
    return isolation


def write_isolation_panels(panels_path: Path, out: Path) -> dict:
    before = assert_frozen_panels(panels_path)
    original = json.loads(panels_path.read_text(encoding="utf-8"))
    isolation = build_isolation_panels(original)
    after = assert_frozen_panels(panels_path)
    if after["working_tree_sha256"] != before["working_tree_sha256"]:
        raise RuntimeError("frozen panels.json changed during isolation transform construction")
    out.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in isolation.items() if key != "meta"}
    (out / "ISOLATION_PANELS.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "ISOLATION_META.json").write_text(json.dumps(isolation["meta"], indent=2) + "\n", encoding="utf-8")
    return isolation["meta"]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write copy-only isolation panels; never mutate frozen panels.json")
    parser.add_argument("--panels", type=Path, default=Path("data/generated/foundation_v2/panels.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    meta = write_isolation_panels(args.panels, args.out)
    print(json.dumps({"status": "ISOLATION_PANELS_WRITTEN", "out": str(args.out), "meta": meta["panel_counts"]}, indent=2))


if __name__ == "__main__":
    main()

