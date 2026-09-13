r"""Independent validation + additive PREEXECUTION_ONLY freeze for the single-fact
A+C diagnostic (seed-87002 study). Reads the materialized items, re-validates
semantics and balance WITHOUT importing the builder, then writes the freeze
receipts. No checkpoint is loaded and no inference is run.
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
OUT = ROOT / "fact_supervision_87001_single_fact_AC_seed87002"
ITEMS_A = OUT / "ITEMS_A.jsonl"
ITEMS_C = OUT / "ITEMS_C.jsonl"
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")

EXPECTED_TOKENIZER_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
NEUTRAL_TMPL = "{name} sat down."
NAME_SET = {"Alex", "Mia", "Nora", "Owen"}
QUERY_PREDS = ("found", "carried")
SENTENCE_RE = re.compile(r"([A-Z][a-zA-Z]* [a-z]+ the [a-z]+ [a-z]+)\.")

# Frozen identity anchors for the later authorized execution (not verified here).
CHECKPOINTS = {
    "Pilot1 parent": "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb",
    "factual-500": "3ae30ae847d3129d7715173ddbfbbaf6b7490772dc850e361a5a2d785b991123",
    "control-500": "c888e3b4cf20860b8d1d0651e418be931da7e8db2acc1c32625edb29701f1530",
}
SCORER = {
    "evaluator": "fact_supervision_87001_eval_v1/EVALUATE.py",
    "evaluator_sha256": "3563cb58334cd9753ca8ebff7fe49987a0f0ccb535abdfdf9a149cc1541a497b",
    "pinned_binding_impl": "fact_supervision_87001_eval_v1/PINNED_PILOT1_BINDING_IMPLEMENTATION.py",
    "pinned_binding_impl_sha256": "5a29ef7f4e96dde93e2feb8199669cb4f69d86609081334ad40a99159e7feda5",
    "semantics": ("candidate conditional LL over response tokens + EOS (English base_model path); "
                  "signed margin; correct iff margin>0; tie==0; greedy argmax<=32 stop EOS"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def query_parts(query_line):
    qwords = query_line.split()
    require(qwords[:3] == ["The", "person", "who"] and qwords[4] == "the" and qwords[-1] == "was",
            f"bad query: {query_line!r}")
    return qwords[3], " ".join(qwords[5:-1])


def validate_set(items, tok, tag, expect_per_family, has_neutral):
    require(len(items) == 24 * expect_per_family, f"{tag}: count")
    fam = Counter(i["family_id"] for i in items)
    require(len(fam) == 24 and all(v == expect_per_family for v in fam.values()), f"{tag}: family size")
    for item in items:
        facts_line, query_line = item["prompt"].split("\n", 1)
        pred, desc = query_parts(query_line)
        sentences = [m + "." for m in SENTENCE_RE.findall(facts_line)]
        require(len(sentences) == 1, f"{tag} {item['id']}: factual sentence count != 1")
        factual = sentences[0]
        require(factual.split()[0] == item["correct_name"], f"{tag} {item['id']}: entailment")
        require(item["correct_name"] in NAME_SET, f"{tag} {item['id']}: bad name")
        cand_names = [c.strip(" .") for c in item["candidates"]]
        require(cand_names == sorted(cand_names) or len(set(cand_names)) == 2, f"{tag} {item['id']}: candidates")
        require(cand_names[item["correct_index"]] == item["correct_name"],
                f"{tag} {item['id']}: correct_index")
        require(item["correct_name"] != item["other_name"], f"{tag} {item['id']}: same name")
        for c, cid in zip(item["candidates"], item["candidate_token_ids"]):
            require(tok.encode(c).ids == cid, f"{tag} {item['id']}: candidate token mismatch")
        if has_neutral:
            neutral = NEUTRAL_TMPL.format(name=item["other_name"])
            require(neutral in facts_line, f"{tag} {item['id']}: neutral missing")
            low = neutral.casefold()
            require(pred not in low and desc not in low, f"{tag} {item['id']}: neutral leaks query")

    ci = Counter(i["correct_index"] for i in items)
    name = Counter(i["correct_name"] for i in items)
    return {"tag": tag, "n": len(items), "families": len(fam),
            "correct_index": dict(sorted(ci.items())), "correct_name": dict(sorted(name.items()))}


def main() -> int:
    require(sha256_file(TOKENIZER) == EXPECTED_TOKENIZER_SHA, "tokenizer hash")
    tok = Tokenizer.from_file(str(TOKENIZER))
    items_a = [json.loads(line) for line in ITEMS_A.read_text(encoding="utf-8").splitlines()]
    items_c = [json.loads(line) for line in ITEMS_C.read_text(encoding="utf-8").splitlines()]
    require(len(items_a) == 48 and len(items_c) == 96, "item counts")

    res_a = validate_set(items_a, tok, "A", 2, has_neutral=False)
    res_c = validate_set(items_c, tok, "C", 4, has_neutral=True)
    require(res_a["correct_index"] == {0: 24, 1: 24}, "A ci balance")
    require(res_c["correct_index"] == {0: 48, 1: 48}, "C ci balance")
    require(res_a["correct_name"] == {"Alex": 12, "Mia": 12, "Nora": 12, "Owen": 12}, "A name balance")
    require(res_c["correct_name"] == {"Alex": 24, "Mia": 24, "Nora": 24, "Owen": 24}, "C name balance")
    pos = Counter(i["neutral_position"] for i in items_c)
    require(pos == {0: 48, 1: 48}, "C neutral position balance")

    validation = {
        "status": "VALIDATION_PASS",
        "neutral_template": NEUTRAL_TMPL,
        "A": res_a,
        "C": res_c,
        "C_neutral_position_balance": dict(sorted(pos.items())),
        "entailment": "every A/C correct candidate is the actor of the retained factual sentence",
        "neutral_contract": "C neutral sentence mentions only the other name and cannot answer/contradict the query",
        "candidates_valid": "candidate strings round-trip to their frozen token ids",
        "original_reversal_query_do_not_transfer": (
            "the original two-fact relational reversal and the original query dimension do not "
            "transfer unchanged; candidate-index/name balance is reported instead"),
    }
    (OUT / "VALIDATION_REPORT.json").write_text(json.dumps(validation, indent=1), encoding="utf-8")

    # freeze receipts
    hashes = {
        "ITEMS_A.jsonl": sha256_file(ITEMS_A),
        "ITEMS_C.jsonl": sha256_file(ITEMS_C),
    }
    manifest = {
        "protocol_id": "fact_supervision_87001_single_fact_AC_seed87002",
        "status": "PREEXECUTION_ONLY",
        "construction": "paired A (literal single fact) + C (one fact + one neutral name-mention)",
        "source": "factual-arm Primary only (24 families); DEV/TRAIN/Confirmation/control/sacred excluded",
        "neutral_template": NEUTRAL_TMPL,
        "counts": {"A": len(items_a), "C": len(items_c)},
        "checkpoint_set": CHECKPOINTS,
        "scorer": SCORER,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
        "checkpoint_loaded": False,
        "inference_run": False,
        "sacred_access": False,
        "confirmation_access": False,
        "seed_87003_access": False,
    }
    protocol = (
        "# Single-fact A+C diagnostic (seed-87002) — pre-execution freeze\n\n"
        "A: retain the queried factual sentence and original query/candidates; remove the competing factual sentence.\n"
        "C: retain the queried factual sentence, add one neutral sentence mentioning the other candidate name "
        "(fixed template `{name} sat down.`), balanced so the factual sentence appears first for half and second for half.\n\n"
        "Scoring: reuse EVALUATE.py candidate conditional LL over response tokens + EOS; signed margin; correct iff "
        "margin>0, tie==0; greedy argmax<=32 stopping at EOS. No normalization, ranks/top-k, perplexity, or thresholds.\n\n"
        "Report A and C separately and their paired difference: exact correct/total, ties, object/predicate strata, "
        "candidate-index/name balance, frozen margin summaries, greedy exact, and (C) sentence-order balance. The original "
        "two-fact relational reversal and query dimensions do not transfer unchanged.\n\n"
        "No checkpoint was loaded and no inference was run during this freeze.\n"
    )
    (OUT / "PROTOCOL.md").write_text(protocol, encoding="utf-8")
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "PREEXECUTION.json").write_text(json.dumps({
        "status": "PREEXECUTION_ONLY",
        "checkpoint_loaded": False,
        "inference_run": False,
        "optimizer_created": False,
        "sacred_access": False,
        "confirmation_access": False,
        "seed_87003_access": False,
        "checkpoints": CHECKPOINTS,
        "scorer": SCORER,
    }, indent=1), encoding="utf-8")

    receipt = {
        "status": "SINGLE_FACT_AC_FREEZE_COMPLETE_PREEXECUTION_ONLY",
        "bundle": "fact_supervision_87001_single_fact_AC_seed87002",
        "items_a_sha256": hashes["ITEMS_A.jsonl"],
        "items_c_sha256": hashes["ITEMS_C.jsonl"],
        "validation_sha256": sha256_file(OUT / "VALIDATION_REPORT.json"),
        "manifest_sha256": sha256_file(OUT / "MANIFEST.json"),
        "protocol_sha256": sha256_file(OUT / "PROTOCOL.md"),
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
        "checkpoint_loaded": False,
        "inference_run": False,
        "sacred_access": False,
        "confirmation_access": False,
        "seed_87003_access": False,
    }
    (OUT / "FREEZE_RECEIPT.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    (OUT / "FREEZE_RECEIPT.sha256").write_text(
        sha256_file(OUT / "FREEZE_RECEIPT.json") + "  FREEZE_RECEIPT.json\n", encoding="utf-8")

    sums = []
    for name in sorted(x.name for x in OUT.iterdir() if x.is_file()):
        if name == "SHA256SUMS.txt":
            continue
        sums.append(f"{sha256_file(OUT / name)}  {name}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(sorted(sums)) + "\n", encoding="utf-8")

    print(json.dumps({"status": receipt["status"], "validation": validation["status"],
                      "hashes": hashes}, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"FREEZE ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
