r"""Materialize the Fork L2 minimal-lexical paired diagnostic (ONE + TWO).

ONE: emit only the correct mapping row (24 families x q0/q1 = 48 items).
TWO: emit both competing mapping rows (24 families x q0/q1 x row_order = 96).
Row syntax (article-retained): <name> <predicate> the <description>
Query unchanged: The person who <predicate> the <description> was
No symbolic role-label tokens. No model is loaded and no inference is run.
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
OUT = ROOT / "fact_supervision_87001_minimal_lexical_pair_seed87002"
ITEMS_SRC = V8 / "ITEMS.jsonl"
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")

EXPECTED_ITEMS_SHA = "2dab70739e9c0ffe1ad3154076ec613a10a4964fa1138e9c7b40d8a5e55f518d"
EXPECTED_TOKENIZER_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"

NAME_SET = {"Alex", "Mia", "Nora", "Owen"}
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


def row_text(name, pred, desc):
    return f"{name} {pred} the {desc}"


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

    items_one, items_two = [], []
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
            correct_idx = next(i for i, (n, p, o) in enumerate(mappings) if (p, o) == (pred, desc))
            correct_name = mappings[correct_idx][0]
            other_name = mappings[1 - correct_idx][0]
            correct_index = [c.strip(" .") for c in cands].index(correct_name)
            # ONE: only the correct mapping row
            one_prompt = f"{row_text(correct_name, pred, desc)}\n{query_line}"
            pids = tok.encode(one_prompt).ids
            require(1 + len(pids) + max(len(c) for c in cand_ids) + 1 <= 256, "ONE context")
            items_one.append({
                "id": f"lexone:{family_id}:q{q}",
                "family_id": family_id, "stratum": stratum, "query": q,
                "correct_index": correct_index, "prompt": one_prompt,
                "candidates": list(cands), "candidate_token_ids": cand_ids,
                "correct_name": correct_name, "other_name": other_name,
                "source_item_id": r["id"],
            })
            # TWO: both rows
            c_row = row_text(correct_name, mappings[correct_idx][1], mappings[correct_idx][2])
            o_row = row_text(other_name, mappings[1 - correct_idx][1], mappings[1 - correct_idx][2])
            for row_order in (0, 1):
                facts_line = f"{c_row}\n{o_row}" if row_order == 0 else f"{o_row}\n{c_row}"
                two_prompt = f"{facts_line}\n{query_line}"
                pids = tok.encode(two_prompt).ids
                require(1 + len(pids) + max(len(c) for c in cand_ids) + 1 <= 256, "TWO context")
                items_two.append({
                    "id": f"lextwo:{family_id}:q{q}_o{row_order}",
                    "family_id": family_id, "stratum": stratum, "query": q,
                    "row_order": row_order, "correct_index": correct_index, "prompt": two_prompt,
                    "candidates": list(cands), "candidate_token_ids": cand_ids,
                    "correct_name": correct_name, "other_name": other_name,
                    "mappings": [list(m) for m in mappings], "source_item_id": r["id"],
                })
    require(len(items_one) == 48, f"ONE != 48 ({len(items_one)})")
    require(len(items_two) == 96, f"TWO != 96 ({len(items_two)})")

    def bal(items, keys):
        return {k: dict(sorted(Counter(i[k] for i in items).items())) for k in keys}
    b_one = bal(items_one, ["correct_index", "correct_name", "stratum"])
    b_two = bal(items_two, ["correct_index", "correct_name", "row_order", "stratum"])
    require(b_one["correct_index"] == {0: 24, 1: 24}, "ONE ci")
    require(b_one["correct_name"] == {"Alex": 12, "Mia": 12, "Nora": 12, "Owen": 12}, "ONE name")
    require(b_one["stratum"] == {"object": 24, "predicate": 24}, "ONE stratum")
    require(b_two["correct_index"] == {0: 48, 1: 48}, "TWO ci")
    require(b_two["correct_name"] == {"Alex": 24, "Mia": 24, "Nora": 24, "Owen": 24}, "TWO name")
    require(b_two["row_order"] == {0: 48, 1: 48}, "TWO row order")
    require(b_two["stratum"] == {"object": 48, "predicate": 48}, "TWO stratum")
    fam_one = Counter(i["family_id"] for i in items_one)
    fam_two = Counter(i["family_id"] for i in items_two)
    require(len(fam_one) == 24 and all(v == 2 for v in fam_one.values()), "ONE family size")
    require(len(fam_two) == 24 and all(v == 4 for v in fam_two.values()), "TWO family size")

    # overlap checks
    two_fact_norm = {norm(r["prompt"]) for r in rows}
    all_items = items_one + items_two
    hits = [i["id"] for i in all_items if norm(i["prompt"]) in two_fact_norm]
    require(not hits, f"two-fact overlap {hits[:5]}")
    ts_norm = set()
    for p in (ROOT / "language_pilot_0_tinystories_seed8380" / "language_train.jsonl",
              ROOT / "language_pilot_0_tinystories_seed8380" / "language_dev.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            ts_norm.add(norm(json.loads(line)["text"]))
    ts_hits = [i["id"] for i in all_items if any(norm(i["prompt"]) in t for t in ts_norm)]
    require(not ts_hits, f"TinyStories overlap {ts_hits[:5]}")
    p7 = norm((ROOT / "language_compositional_p7" / "TRAIN_TEXT.txt").read_text(encoding="utf-8"))
    p7_hits = [i["id"] for i in all_items if norm(i["prompt"]) in p7]
    require(not p7_hits, f"P7 overlap {p7_hits[:5]}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ITEMS_ONE.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items_one), encoding="utf-8")
    (OUT / "ITEMS_TWO.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items_two), encoding="utf-8")
    build_report = {
        "status": "MINIMAL_LEXICAL_PAIR_BUILD_COMPLETE",
        "fork": "L2 (article-retained minimal lexical rows; query unchanged)",
        "row_syntax": "<name> <predicate> the <description>",
        "counts": {"ONE": len(items_one), "TWO": len(items_two)},
        "balance": {"ONE": b_one, "TWO": b_two},
        "overlap": {"two_fact_hits": len(hits), "tinystories_hits": len(ts_hits),
                    "p7_hits": len(p7_hits),
                    "note": "lexical re-encoding of factual-arm Primary; not fresh lexical material"},
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
