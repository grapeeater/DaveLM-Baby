r"""Materialize the structural-gloss companion diagnostic (Fork B).

For each factual-arm Primary item, replace the two English fact sentences with
two structural rows (N <name> P <predicate> O <description>) while keeping the
original English query unchanged. Retain both competing mappings, q0/q1 query
selection, and a balanced row order. 24 families x 4 items = 96 total.

No model is loaded and no inference is run.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(r"C:\DaveLM-CADAVER")
V8 = ROOT / "fact_supervision_87001_corrected_v8"
OUT = ROOT / "fact_supervision_87001_structural_gloss_seed87002"
ITEMS_SRC = V8 / "ITEMS.jsonl"
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")

EXPECTED_ITEMS_SHA = "2dab70739e9c0ffe1ad3154076ec613a10a4964fa1138e9c7b40d8a5e55f518d"
EXPECTED_TOKENIZER_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"

NAME_SET = {"Alex", "Mia", "Nora", "Owen"}
QUERY_PREDS = ("found", "carried")
MAPPING_RE = re.compile(r"([A-Z][a-zA-Z]*) ([a-z]+) the ([a-z]+ [a-z]+)\.")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def parse_source(prompt: str):
    facts_line, query_line = prompt.split("\n", 1)
    mappings = [(m[0], m[1], m[2]) for m in MAPPING_RE.findall(facts_line)]
    require(len(mappings) == 2, f"two mappings not found: {facts_line!r}")
    qwords = query_line.split()
    require(qwords[:3] == ["The", "person", "who"] and qwords[4] == "the" and qwords[-1] == "was",
            f"bad query: {query_line!r}")
    pred = qwords[3]
    desc = " ".join(qwords[5:-1])
    return mappings, query_line, pred, desc


def main() -> int:
    require(sha256_file(ITEMS_SRC) == EXPECTED_ITEMS_SHA, "ITEMS hash")
    require(sha256_file(TOKENIZER) == EXPECTED_TOKENIZER_SHA, "tokenizer hash")
    tok = Tokenizer.from_file(str(TOKENIZER))

    rows = [json.loads(l) for l in ITEMS_SRC.read_text(encoding="utf-8").splitlines()]
    factual_primary = [r for r in rows if r["partition"] == "primary" and r["arm"] == "factual"]
    require(len(factual_primary) == 192, f"factual primary != 192 ({len(factual_primary)})")
    by_family = defaultdict(list)
    for r in factual_primary:
        by_family[r["family_id"]].append(r)
    require(len(by_family) == 24 and all(len(v) == 8 for v in by_family.values()), "family structure")

    # record tokenizer behavior for structural markers
    marker_tokens = {}
    for m in ("N", "P", "O"):
        marker_tokens[m] = tok.encode(m).ids

    items = []
    for family_id in sorted(by_family):
        group = by_family[family_id]
        stratum = group[0]["stratum"]
        reps = {}
        for r in group:
            if r["assignment"] == 0 and r["fact_order"] == 0:
                reps[r["query"]] = r
        require(set(reps) == {0, 1}, f"{family_id}: missing query reps")
        cands = reps[0]["candidates"]
        cand_ids = reps[0]["candidate_token_ids"]
        require(len(cands) == 2 and len(cand_ids) == 2, "candidates")
        for q in (0, 1):
            r = reps[q]
            mappings, query_line, pred, desc = parse_source(r["prompt"])
            # correct mapping = the row whose (pred, obj) matches the query
            correct_idx = next(i for i, (n, p, o) in enumerate(mappings) if (p, o) == (pred, desc))
            correct_name = mappings[correct_idx][0]
            other_name = mappings[1 - correct_idx][0]
            require(correct_name in NAME_SET and other_name in NAME_SET, "name set")
            require(correct_name != other_name, "same name")
            correct_index = [c.strip(" .") for c in cands].index(correct_name)
            row_correct = f"N {correct_name} P {mappings[correct_idx][1]} O {mappings[correct_idx][2]}"
            row_other = f"N {other_name} P {mappings[1 - correct_idx][1]} O {mappings[1 - correct_idx][2]}"
            for row_order in (0, 1):
                if row_order == 0:
                    facts_line = f"{row_correct}\n{row_other}"
                else:
                    facts_line = f"{row_other}\n{row_correct}"
                prompt = f"{facts_line}\n{query_line}"
                # context validation: BOS + prompt + candidate + EOS <= 256
                pids = tok.encode(prompt).ids
                cand_len = max(len(c) for c in cand_ids)
                require(1 + len(pids) + cand_len + 1 <= 256, f"{family_id}: context > 256")
                items.append({
                    "id": f"gloss:{family_id}:q{q}_o{row_order}",
                    "family_id": family_id,
                    "stratum": stratum,
                    "query": q,
                    "row_order": row_order,
                    "correct_index": correct_index,
                    "prompt": prompt,
                    "candidates": list(cands),
                    "candidate_token_ids": cand_ids,
                    "correct_name": correct_name,
                    "other_name": other_name,
                    "mappings": [list(m) for m in mappings],
                    "source_item_id": r["id"],
                })
    require(len(items) == 96, f"items != 96 ({len(items)})")

    # balancing
    ci = Counter(i["correct_index"] for i in items)
    name = Counter(i["correct_name"] for i in items)
    order = Counter(i["row_order"] for i in items)
    strat = Counter(i["stratum"] for i in items)
    fam = Counter(i["family_id"] for i in items)
    require(ci == {0: 48, 1: 48}, f"correct_index balance {dict(ci)}")
    require(name == {"Alex": 24, "Mia": 24, "Nora": 24, "Owen": 24}, f"name balance {dict(name)}")
    require(order == {0: 48, 1: 48}, f"row order balance {dict(order)}")
    require(strat == {"object": 48, "predicate": 48}, f"stratum balance {dict(strat)}")
    require(len(fam) == 24 and all(v == 4 for v in fam.values()), "family size")

    # overlap
    two_fact_norm = {norm(r["prompt"]) for r in rows}
    overlap = [i["id"] for i in items if norm(i["prompt"]) in two_fact_norm]
    require(not overlap, f"two-fact overlap {overlap[:5]}")
    ts_norm = set()
    for p in (ROOT / "language_pilot_0_tinystories_seed8380" / "language_train.jsonl",
              ROOT / "language_pilot_0_tinystories_seed8380" / "language_dev.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            ts_norm.add(norm(json.loads(line)["text"]))
    ts_hits = []
    for i in items:
        pn = norm(i["prompt"])
        for t in ts_norm:
            if pn in t:
                ts_hits.append(i["id"])
                break
    require(not ts_hits, f"TinyStories overlap {ts_hits[:5]}")
    p7 = norm((ROOT / "language_compositional_p7" / "TRAIN_TEXT.txt").read_text(encoding="utf-8"))
    p7_hits = [i["id"] for i in items if norm(i["prompt"]) in p7]
    require(not p7_hits, f"P7 overlap {p7_hits[:5]}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ITEMS.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    build_report = {
        "status": "STRUCTURAL_GLOSS_BUILD_COMPLETE",
        "fork": "B (facts-gloss, English query unchanged)",
        "counts": {"items": len(items), "families": 24, "items_per_family": 4},
        "balance": {"correct_index": dict(ci), "correct_name": dict(name),
                    "row_order": dict(order), "stratum": dict(strat)},
        "marker_tokens": marker_tokens,
        "overlap": {"two_fact_hits": len(overlap), "tinystories_hits": len(ts_hits),
                    "p7_hits": len(p7_hits),
                    "note": "structural re-encoding of factual-arm Primary; not fresh lexical material"},
    }
    (OUT / "BUILD_REPORT.json").write_text(json.dumps(build_report, indent=1), encoding="utf-8")
    print(json.dumps(build_report, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"BUILD ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
