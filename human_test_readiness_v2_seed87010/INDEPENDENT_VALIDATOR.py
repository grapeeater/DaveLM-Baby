"""Independent structural validator for the prospective readiness DEV v2 bundle.

This validator deliberately re-derives factual answers from rendered prompt text;
it does not import the construction script or trust builder answer-key metadata.
It is intended to run before the bundle's manifest and receipt are sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer


TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
TOKENIZER_SHA256 = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


FACT = re.compile(
    r"^([A-Z][a-z]+) (found|carried) the ([a-z ]+)\. "
    r"([A-Z][a-z]+) (found|carried) the ([a-z ]+)\. "
    r"The ([a-z ]+) was (found|carried) by$"
)


def derive_fact_answer(prompt: str) -> str:
    """Derive the response name strictly from rendered factual text."""
    m = FACT.fullmatch(prompt)
    if not m:
        raise AssertionError(f"factual prompt does not satisfy grammar: {prompt!r}")
    a, r1, o1, b, r2, o2, query_object, query_relation = m.groups()
    if r1 != r2 or r1 != query_relation:
        raise AssertionError("query relation does not match both factual relations")
    owners = {o1: a, o2: b}
    if query_object not in owners:
        raise AssertionError("query object is absent from rendered facts")
    return owners[query_object]


def validate_fact(row: dict, tok: Tokenizer) -> None:
    expected = " " + derive_fact_answer(row["prompt"]) + "."
    assert row["correct_candidate"] == expected, (row["id"], expected, row["correct_candidate"])
    assert row["candidates"][row["correct_index"]] == expected
    assert len(row["candidates"]) == 2
    ids = [tok.encode(c).ids for c in row["candidates"]]
    assert ids == row["candidate_token_ids"]
    assert len(ids[0]) == len(ids[1])
    prompt_ids = tok.encode(row["prompt"]).ids
    for candidate, candidate_ids in zip(row["candidates"], ids):
        assert tok.decode(candidate_ids) == candidate
        assert tok.encode(row["prompt"] + candidate).ids == prompt_ids + candidate_ids


def validate_negative_tests(facts: list[dict], tok: Tokenizer) -> list[str]:
    def rejected(fn) -> None:
        try:
            fn()
        except AssertionError:
            return
        raise AssertionError("negative validator case was accepted")

    original = facts[0]
    # Wrong stored key must be detected against unchanged prompt semantics.
    altered = dict(original)
    altered["correct_candidate"] = altered["candidates"][1 - altered["correct_index"]]
    rejected(lambda: validate_fact(altered, tok))
    # A changed predicate invalidates the rendered relation grammar.
    rejected(lambda: derive_fact_answer(original["prompt"].replace(" found ", " carried ", 1)))
    # A query object with no antecedent must be rejected.
    rejected(lambda: derive_fact_answer(re.sub(r"The [a-z ]+ was", "The missing object was", original["prompt"])))
    # A reversal group with only one assignment is invalid.
    reverse = [r for r in facts if r["reversal_pair_id"] == facts[0]["reversal_pair_id"]]
    rejected(lambda: _assert_complete_reversal(reverse[:1]))
    # Unequal candidate token sequences are invalid.
    rejected(lambda: _assert_equal_lengths(tok.encode(" Ash.").ids, tok.encode(" Ash Ash Ash Ash Ash Ash.").ids))
    return ["wrong_answer_key_rejected", "wrong_predicate_rejected", "missing_query_object_rejected", "broken_reversal_rejected", "unequal_candidate_length_rejected"]


def _assert_complete_reversal(members: list[dict]) -> None:
    assert len(members) == 2
    assert {r["assignment"] for r in members} == {0, 1}


def _assert_equal_lengths(left: list[int], right: list[int]) -> None:
    assert len(left) == len(right)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    args = ap.parse_args()
    bundle = args.bundle
    assert sha(TOKENIZER) == TOKENIZER_SHA256
    tok = Tokenizer.from_file(str(TOKENIZER))

    facts = rows(bundle / "DEV_FACTS.jsonl")
    instructions = rows(bundle / "DEV_INSTRUCTIONS.jsonl")
    continuity = rows(bundle / "DEV_CONTINUITY.jsonl")
    generations = rows(bundle / "DEV_GENERATION.jsonl")
    all_items = rows(bundle / "DEV_ITEMS.jsonl")
    assert len(facts) == 24 and len(instructions) == 12 and len(continuity) == 12 and len(generations) == 20
    assert len(all_items) == 68
    assert [r["id"] for r in all_items] == [r["id"] for r in facts + instructions + continuity + generations]
    assert len({norm(r["prompt"]) for r in all_items}) == len(all_items)

    for row in facts:
        validate_fact(row, tok)

    family = defaultdict(list)
    reverse = defaultdict(list)
    for row in facts:
        family[row["family_id"]].append(row)
        reverse[row["reversal_pair_id"]].append(row)
    assert len(family) == 6 and all(len(x) == 4 for x in family.values())
    assert len(reverse) == 12 and all(len(x) == 2 for x in reverse.values())
    for members in family.values():
        assert {x["assignment"] for x in members} == {0, 1}
        assert {x["query"] for x in members} == {0, 1}
        assert len({x["fact_order"] for x in members}) == 1
    for members in reverse.values():
        assert {x["assignment"] for x in members} == {0, 1}
        assert members[0]["candidates"] == members[1]["candidates"]
        assert {x["correct_index"] for x in members} == {0, 1}
    assert Counter(r["correct_index"] for r in facts) == Counter({0: 12, 1: 12})
    assert Counter(r["fact_order"] for r in facts) == Counter({0: 12, 1: 12})

    for row in instructions + continuity:
        assert row["correct_candidate"] == row["candidates"][row["correct_index"]]
        ids = [tok.encode(c).ids for c in row["candidates"]]
        assert ids == row["candidate_token_ids"] and len(ids[0]) == len(ids[1])
        pids = tok.encode(row["prompt"]).ids
        for candidate, candidate_ids in zip(row["candidates"], ids):
            assert tok.decode(candidate_ids) == candidate
            assert tok.encode(row["prompt"] + candidate).ids == pids + candidate_ids

    # Independently parse continuity's two supported surface forms.
    for row in continuity:
        if "My name is" in row["prompt"]:
            expected = re.search(r"My name is ([A-Z][a-z]+)\.", row["prompt"]).group(1)
        else:
            expected = re.search(r"Please call me ([A-Z][a-z]+)\.", row["prompt"]).group(1)
        assert row["correct_candidate"] == f" {expected}."

    for row in all_items:
        assert tok.decode(tok.encode(row["prompt"]).ids) == row["prompt"]
        assert 1 + len(tok.encode(row["prompt"]).ids) + row.get("generation_max_new_tokens", 0) <= 256

    negatives = validate_negative_tests(facts, tok)
    output = {
        "status": "PASS_INDEPENDENT_SEMANTIC_VALIDATION",
        "validator": Path(__file__).name,
        "validator_sha256": sha(Path(__file__)),
        "tokenizer_sha256": sha(TOKENIZER),
        "counts": {"fact": len(facts), "instruction": len(instructions), "continuity": len(continuity), "generation": len(generations), "total": len(all_items)},
        "fact_family_count": len(family),
        "reversal_pair_count": len(reverse),
        "unique_normalized_prompt_count": len({norm(r["prompt"]) for r in all_items}),
        "negative_tests": negatives,
        "final_accessed": False,
        "sacred_accessed": False,
    }
    (bundle / "INDEPENDENT_VALIDATION.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
