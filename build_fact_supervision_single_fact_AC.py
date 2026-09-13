r"""Materialize the paired single-fact A+C diagnostic freeze (seed-87002 study).

Derives A (literal single fact) and C (one fact + one neutral name-mention)
items ONLY from the factual-arm Primary partition (24 families). No model is
loaded and no inference is run. Builds items, runs semantic validation,
balancing, and study-local overlap checks, then writes an additive
PREEXECUTION_ONLY freeze.

Neutral template (fixed before outcomes): "<name> sat down."
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
OUT = ROOT / "fact_supervision_87001_single_fact_AC_seed87002"
ITEMS_SRC = V8 / "ITEMS.jsonl"
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")

EXPECTED_ITEMS_SHA = "2dab70739e9c0ffe1ad3154076ec613a10a4964fa1138e9c7b40d8a5e55f518d"
EXPECTED_TOKENIZER_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"

NEUTRAL_TMPL = "{name} sat down."
QUERY_PREDS = ("found", "carried")
NAME_SET = {"Alex", "Mia", "Nora", "Owen"}

SENTENCE_RE = re.compile(r"([A-Z][a-zA-Z]* [a-z]+ the [a-z]+ [a-z]+)\.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def query_parts(query_line: str):
    qwords = query_line.split()
    require(qwords[:3] == ["The", "person", "who"] and qwords[4] == "the" and qwords[-1] == "was",
            f"unexpected query line: {query_line!r}")
    return qwords[3], " ".join(qwords[5:-1])


def factual_sentences(facts_line: str):
    return [m + "." for m in SENTENCE_RE.findall(facts_line)]


def main() -> int:
    require(sha256_file(ITEMS_SRC) == EXPECTED_ITEMS_SHA, "ITEMS.jsonl hash mismatch")
    require(sha256_file(TOKENIZER) == EXPECTED_TOKENIZER_SHA, "tokenizer hash mismatch")
    tok = Tokenizer.from_file(str(TOKENIZER))

    rows = [json.loads(line) for line in ITEMS_SRC.read_text(encoding="utf-8").splitlines()]
    factual_primary = [r for r in rows if r["partition"] == "primary" and r["arm"] == "factual"]
    require(len(factual_primary) == 192, f"factual primary != 192 ({len(factual_primary)})")

    by_family = defaultdict(list)
    for r in factual_primary:
        by_family[r["family_id"]].append(r)
    require(len(by_family) == 24 and all(len(v) == 8 for v in by_family.values()),
            "factual primary family structure != 24x8")

    # verify neutral template is tokenizable and contains no query vocabulary
    for name in sorted(NAME_SET):
        neutral = NEUTRAL_TMPL.format(name=name)
        ids = tok.encode(neutral).ids
        require(len(ids) > 0, "neutral template failed to tokenize")
        low = neutral.casefold()
        for pred in QUERY_PREDS:
            require(pred not in low, "neutral template contains a queried predicate")

    items_a, items_c = [], []
    for family_id in sorted(by_family):
        group = by_family[family_id]
        stratum = group[0]["stratum"]
        # representative rows per query (assignment 0, order 0)
        reps = {}
        for r in group:
            if r["assignment"] == 0 and r["fact_order"] == 0:
                reps[r["query"]] = r
        require(set(reps) == {0, 1}, f"{family_id}: missing query reps")
        cands = reps[0]["candidates"]
        cand_ids = reps[0]["candidate_token_ids"]
        require(len(cands) == 2 and len(cand_ids) == 2, "candidates != 2")
        for q in (0, 1):
            r = reps[q]
            facts_line, query_line = r["prompt"].split("\n", 1)
            sentences = factual_sentences(facts_line)
            require(len(sentences) == 2, f"{family_id}: two facts not found in {facts_line!r}")
            pred, desc = query_parts(query_line)
            queried = next(s for s in sentences if f"{pred} the {desc}" in s)
            correct_name = queried.split()[0]
            require(correct_name in NAME_SET, f"{family_id}: bad correct name {correct_name}")
            other_name = next(c for c in cands if c.strip(" .") != correct_name).strip(" .")
            correct_index = int(r["correct_index"])
            # A
            a_prompt = f"{queried}\n{query_line}"
            items_a.append({
                "id": f"singlefact:A:{family_id}:q{q}",
                "family_id": family_id,
                "stratum": stratum,
                "query": q,
                "correct_index": correct_index,
                "prompt": a_prompt,
                "candidates": list(cands),
                "candidate_token_ids": cand_ids,
                "correct_name": correct_name,
                "other_name": other_name,
                "source_item_id": r["id"],
            })
            # C (two neutral positions)
            for pos in (0, 1):
                neutral = NEUTRAL_TMPL.format(name=other_name)
                facts_line = (neutral + " " + queried) if pos == 1 else (queried + " " + neutral)
                c_prompt = f"{facts_line}\n{query_line}"
                items_c.append({
                    "id": f"singlefact:C:{family_id}:q{q}_p{pos}",
                    "family_id": family_id,
                    "stratum": stratum,
                    "query": q,
                    "neutral_position": pos,
                    "correct_index": correct_index,
                    "prompt": c_prompt,
                    "candidates": list(cands),
                    "candidate_token_ids": cand_ids,
                    "correct_name": correct_name,
                    "other_name": other_name,
                    "source_item_id": r["id"],
                })
    require(len(items_a) == 48 and len(items_c) == 96, "item counts")

    # semantic validation (inline)
    for item in items_a + items_c:
        facts_line, query_line = item["prompt"].split("\n", 1)
        pred, desc = query_parts(query_line)
        sentences = factual_sentences(facts_line)
        require(len(sentences) == 1, f"{item['id']}: expected exactly one factual sentence")
        queried = sentences[0]
        require(queried.split()[0] == item["correct_name"], f"{item['id']}: answer key not entailed")
        # correct candidate must be the correct name
        cand_names = [c.strip(" .") for c in item["candidates"]]
        require(cand_names[item["correct_index"]] == item["correct_name"],
                f"{item['id']}: correct_index does not point at correct name")
    for item in items_c:
        facts_line = item["prompt"].split("\n", 1)[0]
        neutral = NEUTRAL_TMPL.format(name=item["other_name"])
        require(neutral in facts_line, f"{item['id']}: neutral sentence missing")
        # neutral cannot answer or contradict the query
        low = neutral.casefold()
        require(not any(p in low for p in QUERY_PREDS), f"{item['id']}: neutral leaks predicate")
        pred, desc = query_parts(item["prompt"].split("\n", 1)[1])
        require(desc not in low and pred not in low, f"{item['id']}: neutral contains query content")

    # balancing (exact)
    def balance(items):
        n = len(items)
        fam = Counter(i["family_id"] for i in items)
        ci = Counter(i["correct_index"] for i in items)
        name = Counter(i["correct_name"] for i in items)
        return {"n": n, "families": len(fam), "items_per_family": sorted(set(fam.values())),
                "correct_index": dict(sorted(ci.items())), "correct_name": dict(sorted(name.items()))}
    bal_a = balance(items_a)
    bal_c = balance(items_c)
    require(bal_a["families"] == 24 and bal_a["items_per_family"] == [2], "A family balance")
    require(bal_c["families"] == 24 and bal_c["items_per_family"] == [4], "C family balance")
    require(bal_a["correct_index"] == {0: 24, 1: 24}, "A correct_index not balanced")
    require(bal_c["correct_index"] == {0: 48, 1: 48}, "C correct_index not balanced")
    # C neutral-position balance: 2 each per family
    pos = Counter(i["neutral_position"] for i in items_c)
    require(pos == {0: 48, 1: 48}, "C neutral position not balanced")

    # overlap checks (study-local)
    two_fact_prompts = {norm(r["prompt"]) for r in rows}
    two_fact_factual_primary = {norm(r["prompt"]) for r in factual_primary}
    overlap_twofact = sum(1 for i in items_a + items_c if norm(i["prompt"]) in two_fact_prompts)
    # derivation: each A prompt is a substring of its source primary prompt
    src_by_id = {r["id"]: r for r in factual_primary}
    sub_a = sum(1 for i in items_a if norm(i["prompt"]) in norm(src_by_id[i["source_item_id"]]["prompt"]))
    sub_c = sum(1 for i in items_c if norm(i["prompt"]) in norm(src_by_id[i["source_item_id"]]["prompt"]))

    # TinyStories overlap (from pilot0 train/dev text)
    ts_texts = []
    for p in (ROOT / "language_pilot_0_tinystories_seed8380" / "language_train.jsonl",
              ROOT / "language_pilot_0_tinystories_seed8380" / "language_dev.jsonl"):
        for line in p.read_text(encoding="utf-8").splitlines():
            ts_texts.append(json.loads(line)["text"])
    ts_norm = set(norm(t) for t in ts_texts)
    ts_hits = []
    for i in items_a + items_c:
        pnorm = norm(i["prompt"])
        for t in ts_norm:
            if pnorm in t:
                ts_hits.append(i["id"])
                break
    require(not ts_hits, f"TinyStories overlap hits: {ts_hits[:5]}")
    p7 = (ROOT / "language_compositional_p7" / "TRAIN_TEXT.txt")
    p7_text = norm(p7.read_text(encoding="utf-8"))
    p7_hits = [i["id"] for i in items_a + items_c if norm(i["prompt"]) in p7_text]
    require(not p7_hits, f"P7 TRAIN_TEXT overlap hits: {p7_hits[:5]}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ITEMS_A.jsonl").write_text(
        "".join(json.dumps(i) + "\n" for i in items_a), encoding="utf-8")
    (OUT / "ITEMS_C.jsonl").write_text(
        "".join(json.dumps(i) + "\n" for i in items_c), encoding="utf-8")

    payload = {
        "status": "SINGLE_FACT_AC_BUILD_COMPLETE",
        "neutral_template": NEUTRAL_TMPL,
        "source": {"items_sha256": EXPECTED_ITEMS_SHA, "partition": "primary", "arm": "factual",
                   "families": 24},
        "counts": {"A": len(items_a), "C": len(items_c)},
        "balance": {"A": bal_a, "C": bal_c},
        "overlap": {
            "exact_two_fact_prompt_hits": overlap_twofact,
            "A_prompt_is_substring_of_source": sub_a,
            "C_prompt_is_substring_of_source": sub_c,
            "tinystories_hits": len(ts_hits),
            "p7_train_text_hits": len(p7_hits),
            "note": ("A/C prompts are derived substrings of factual Primary two-fact prompts by "
                     "construction; no fresh lexical novelty is claimed."),
        },
    }
    (OUT / "BUILD_REPORT.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"BUILD ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
