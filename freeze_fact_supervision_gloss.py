r"""Independent validation + additive PREEXECUTION_ONLY freeze for the structural-gloss
(Fork B) diagnostic. Re-validates items WITHOUT importing the builder, then writes the
freeze receipts. No checkpoint is loaded and no inference is run.
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
OUT = ROOT / "fact_supervision_87001_structural_gloss_seed87002"
ITEMS = OUT / "ITEMS.jsonl"
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")

EXPECTED_TOKENIZER_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
NAME_SET = {"Alex", "Mia", "Nora", "Owen"}
QUERY_PREDS = ("found", "carried")

CHECKPOINTS = {
    "Pilot1 parent": "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb",
    "factual-500": "3ae30ae847d3129d7715173ddbfbbaf6b7490772dc850e361a5a2d785b991123",
    "control-500": "c888e3b4cf20860b8d1d0651e418be931da7e8db2acc1c32625edb29701f1530",
}
SCORER = {
    "evaluator_sha256": "3563cb58334cd9753ca8ebff7fe49987a0f0ccb535abdfdf9a149cc1541a497b",
    "semantics": ("candidate conditional LL over response tokens + EOS (English base_model path); "
                  "signed margin; correct iff margin>0; tie==0; greedy argmax<=32 stop EOS"),
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    require(sha256_file(TOKENIZER) == EXPECTED_TOKENIZER_SHA, "tokenizer hash")
    tok = Tokenizer.from_file(str(TOKENIZER))
    items = [json.loads(l) for l in ITEMS.read_text(encoding="utf-8").splitlines()]
    require(len(items) == 96, f"items != 96 ({len(items)})")

    checks = []
    for item in items:
        lines = item["prompt"].split("\n")
        require(len(lines) == 3, f"{item['id']}: expected 3 lines")
        row0, row1, query_line = lines
        def parse_row(line):
            m = re.match(r"N ([A-Z][a-zA-Z]*) P ([a-z]+) O ([a-z]+ [a-z]+)$", line)
            require(m is not None, f"{item['id']}: bad row {line!r}")
            return m.group(1), m.group(2), m.group(3)
        n0, p0, o0 = parse_row(row0)
        n1, p1, o1 = parse_row(row1)
        qwords = query_line.split()
        require(qwords[:3] == ["The", "person", "who"] and qwords[4] == "the" and qwords[-1] == "was",
                f"{item['id']}: bad query {query_line!r}")
        qp, qo = qwords[3], " ".join(qwords[5:-1])
        # both candidate names appear as the two row names
        require({n0, n1} == {item["correct_name"], item["other_name"]},
                f"{item['id']}: row names != candidate names")
        require(n0 != n1 and n0 in NAME_SET and n1 in NAME_SET, f"{item['id']}: name set")
        # answer determined ONLY by matching query (pred,desc) to one row
        matches = [i for i, (n, p, o) in enumerate(((n0, p0, o0), (n1, p1, o1))) if (p, o) == (qp, qo)]
        require(len(matches) == 1, f"{item['id']}: query matches {len(matches)} rows")
        correct_name = (n0, n1)[matches[0]]
        require(correct_name == item["correct_name"], f"{item['id']}: answer key not from query match")
        # row order does not change the answer key
        require(item["correct_name"] in (n0, n1) and item["correct_name"] != item["other_name"],
                f"{item['id']}: order invariance")
        cand_names = [c.strip(" .") for c in item["candidates"]]
        require(cand_names[item["correct_index"]] == item["correct_name"],
                f"{item['id']}: correct_index")
        for c, cid in zip(item["candidates"], item["candidate_token_ids"]):
            require(tok.encode(c).ids == cid, f"{item['id']}: candidate round-trip")
        pids = tok.encode(item["prompt"]).ids
        require(1 + len(pids) + max(len(c) for c in item["candidate_token_ids"]) + 1 <= 256,
                f"{item['id']}: context > 256")
        checks.append(item["id"])

    ci = Counter(i["correct_index"] for i in items)
    name = Counter(i["correct_name"] for i in items)
    order = Counter(i["row_order"] for i in items)
    strat = Counter(i["stratum"] for i in items)
    require(ci == {0: 48, 1: 48}, "correct_index balance")
    require(name == {"Alex": 24, "Mia": 24, "Nora": 24, "Owen": 24}, "name balance")
    require(order == {0: 48, 1: 48}, "row order balance")
    require(strat == {"object": 48, "predicate": 48}, "stratum balance")

    validation = {
        "status": "VALIDATION_PASS",
        "items": len(items),
        "entailment": "correct answer equals the unique row matching the queried predicate/description",
        "both_names_present": True,
        "order_invariance": "row order does not change the answer key",
        "candidate_roundtrip": True,
        "context_within_256": True,
        "balance": {"correct_index": dict(ci), "correct_name": dict(name),
                    "row_order": dict(order), "stratum": dict(strat)},
        "original_relational_reversal_does_not_transfer": (
            "the original two-fact relational reversal/query dimensions do not transfer "
            "unchanged; row order and query are reported as balanced descriptive factors"),
    }
    (OUT / "VALIDATION_REPORT.json").write_text(json.dumps(validation, indent=1), encoding="utf-8")

    manifest = {
        "protocol_id": "fact_supervision_87001_structural_gloss_seed87002",
        "fork": "B (facts-gloss, English query unchanged)",
        "status": "PREEXECUTION_ONLY",
        "source": "factual-arm Primary only (24 families x 4 = 96 items)",
        "syntax": "N <name> P <predicate> O <description> (two rows) + original English query",
        "counts": {"items": len(items), "families": 24, "items_per_family": 4},
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
        "# Structural-gloss companion diagnostic (Fork B) — pre-execution freeze\n\n"
        "Replace the two English fact sentences with structural rows `N <name> P <predicate> O <description>` "
        "and keep the original English query unchanged (`The person who <predicate> the <description> was`).\n\n"
        "Both competing mappings and both candidate names are present; the correct actor is the unique row whose "
        "(predicate, description) equals the query. Row order is balanced and does not change the answer key.\n\n"
        "Scoring: reuse the frozen candidate conditional LL over response tokens + EOS; signed margin; correct iff "
        "margin>0, tie==0; greedy argmax<=32 stopping at EOS. No normalization, ranks/top-k, perplexity, thresholds, "
        "significance tests, or mechanistic intervention.\n\n"
        "Report (per checkpoint, overall and object/predicate strata): exact correct/total, ties, families-complete, "
        "correct-index/name slices, row-order slice, frozen margin summaries, greedy exact; plus a descriptive "
        "gloss-vs-A / gloss-vs-C difference using the already-frozen A/C aggregates. The original two-fact relational "
        "reversal and query dimensions do not transfer unchanged.\n\n"
        "No checkpoint was loaded and no inference was run during this freeze.\n"
    )
    (OUT / "PROTOCOL.md").write_text(protocol, encoding="utf-8")
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "PREEXECUTION.json").write_text(json.dumps({
        "status": "PREEXECUTION_ONLY",
        "checkpoint_loaded": False, "inference_run": False, "optimizer_created": False,
        "sacred_access": False, "confirmation_access": False, "seed_87003_access": False,
        "checkpoints": CHECKPOINTS, "scorer": SCORER, "tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
    }, indent=1), encoding="utf-8")

    receipt = {
        "status": "STRUCTURAL_GLOSS_FREEZE_COMPLETE_PREEXECUTION_ONLY",
        "bundle": "fact_supervision_87001_structural_gloss_seed87002",
        "items_sha256": sha256_file(ITEMS),
        "validation_sha256": sha256_file(OUT / "VALIDATION_REPORT.json"),
        "manifest_sha256": sha256_file(OUT / "MANIFEST.json"),
        "protocol_sha256": sha256_file(OUT / "PROTOCOL.md"),
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
        "checkpoint_loaded": False, "inference_run": False,
        "sacred_access": False, "confirmation_access": False, "seed_87003_access": False,
    }
    (OUT / "FREEZE_RECEIPT.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    (OUT / "FREEZE_RECEIPT.sha256").write_text(
        sha256_file(OUT / "FREEZE_RECEIPT.json") + "  FREEZE_RECEIPT.json\n", encoding="utf-8")

    sums = sorted(f"{sha256_file(OUT / f.name)}  {f.name}"
                  for f in OUT.iterdir() if f.is_file() and f.name != "SHA256SUMS.txt")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")

    print(json.dumps({"status": receipt["status"], "validation": validation["status"],
                      "items_sha256": receipt["items_sha256"]}, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"FREEZE ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
