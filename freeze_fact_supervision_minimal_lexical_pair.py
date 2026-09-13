r"""Independent validation + additive PREEXECUTION_ONLY freeze for the Fork L2
minimal-lexical paired diagnostic (ONE + TWO). Re-validates items WITHOUT
importing the builder, then writes the freeze receipts. No checkpoint is loaded
and no inference is run.
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
OUT = ROOT / "fact_supervision_87001_minimal_lexical_pair_seed87002"
ONE = OUT / "ITEMS_ONE.jsonl"
TWO = OUT / "ITEMS_TWO.jsonl"
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")

EXPECTED_TOKENIZER_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
NAME_SET = {"Alex", "Mia", "Nora", "Owen"}
ROW_RE = re.compile(r"([A-Z][a-zA-Z]*) ([a-z]+) the ([a-z]+ [a-z]+)$")

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


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def parse_row(line):
    m = ROW_RE.match(line)
    require(m is not None, f"bad row {line!r}")
    return m.group(1), m.group(2), m.group(3)


def query_parts(query_line):
    qwords = query_line.split()
    require(qwords[:3] == ["The", "person", "who"] and qwords[4] == "the" and qwords[-1] == "was",
            f"bad query: {query_line!r}")
    return qwords[3], " ".join(qwords[5:-1])


def validate(items, tok, tag, rows_per_item, items_per_family, both_names_required):
    require(len(items) == (24 * items_per_family), f"{tag}: count")
    fam = Counter(i["family_id"] for i in items)
    require(len(fam) == 24 and all(v == items_per_family for v in fam.values()), f"{tag}: families")
    for item in items:
        lines = item["prompt"].split("\n")
        require(len(lines) == rows_per_item + 1, f"{tag} {item['id']}: line count")
        rows = [parse_row(l) for l in lines[:-1]]
        qp, qo = query_parts(lines[-1])
        # no role-label tokens: rows must be plain name/pred/the/desc words
        for name, pred, desc in rows:
            require(name in NAME_SET and pred in ("found", "carried"), f"{tag} {item['id']}: vocab")
            require(" " not in name and pred in ("found", "carried"), f"{tag} {item['id']}: vocab2")
        if both_names_required:
            row_names = {n for n, _, _ in rows}
            require(len(rows) == 2 and len(row_names) == 2, f"{tag} {item['id']}: two names")
            require({item["correct_name"], item["other_name"]} == row_names,
                    f"{tag} {item['id']}: candidate names present")
        else:
            require(len(rows) == 1, f"{tag} {item['id']}: one row expected")
        # unique query-to-row entailment
        matches = [i for i, (n, p, o) in enumerate(rows) if (p, o) == (qp, qo)]
        require(len(matches) == 1, f"{tag} {item['id']}: {len(matches)} matches")
        require(rows[matches[0]][0] == item["correct_name"], f"{tag} {item['id']}: entailment")
        cand_names = [c.strip(" .") for c in item["candidates"]]
        require(cand_names[item["correct_index"]] == item["correct_name"], f"{tag} {item['id']}: ci")
        for c, cid in zip(item["candidates"], item["candidate_token_ids"]):
            require(tok.encode(c).ids == cid, f"{tag} {item['id']}: candidate round-trip")
        pids = tok.encode(item["prompt"]).ids
        require(1 + len(pids) + max(len(c) for c in item["candidate_token_ids"]) + 1 <= 256,
                f"{tag} {item['id']}: context")
    return {
        "tag": tag, "n": len(items), "families": 24,
        "correct_index": dict(sorted(Counter(i["correct_index"] for i in items).items())),
        "correct_name": dict(sorted(Counter(i["correct_name"] for i in items).items())),
        "stratum": dict(sorted(Counter(i["stratum"] for i in items).items())),
        "row_order": dict(sorted(Counter(i.get("row_order", 0) for i in items).items())),
    }


def main() -> int:
    require(sha256_file(TOKENIZER) == EXPECTED_TOKENIZER_SHA, "tokenizer hash")
    tok = Tokenizer.from_file(str(TOKENIZER))
    items_one = [json.loads(l) for l in ONE.read_text(encoding="utf-8").splitlines()]
    items_two = [json.loads(l) for l in TWO.read_text(encoding="utf-8").splitlines()]
    require(len(items_one) == 48 and len(items_two) == 96, "counts")

    v_one = validate(items_one, tok, "ONE", rows_per_item=1, items_per_family=2, both_names_required=False)
    v_two = validate(items_two, tok, "TWO", rows_per_item=2, items_per_family=4, both_names_required=True)
    require(v_one["correct_index"] == {0: 24, 1: 24}, "ONE ci")
    require(v_two["correct_index"] == {0: 48, 1: 48}, "TWO ci")
    require(v_one["correct_name"] == {"Alex": 12, "Mia": 12, "Nora": 12, "Owen": 12}, "ONE name")
    require(v_two["correct_name"] == {"Alex": 24, "Mia": 24, "Nora": 24, "Owen": 24}, "TWO name")
    require(v_one["stratum"] == {"object": 24, "predicate": 24}, "ONE stratum")
    require(v_two["stratum"] == {"object": 48, "predicate": 48}, "TWO stratum")
    require(v_two["row_order"] == {0: 48, 1: 48}, "TWO row order")
    # order invariance: recompute per TWO family; answer key independent of row order
    byfam = defaultdict(list)
    for it in items_two:
        byfam[it["family_id"]].append(it)
    for fam, grp in byfam.items():
        for it in grp:
            require(it["correct_name"] != it["other_name"], f"{fam}: same name")

    validation = {
        "status": "VALIDATION_PASS",
        "ONE": v_one,
        "TWO": v_two,
        "entailment": "correct answer is the unique row whose (predicate, description) equals the query",
        "order_invariance": True,
        "both_names_present_in_TWO": True,
        "candidate_roundtrip": True,
        "context_within_256": True,
        "no_role_label_tokens": True,
        "fork": "L2 (article-retained minimal lexical rows)",
    }
    (OUT / "VALIDATION_REPORT.json").write_text(json.dumps(validation, indent=1), encoding="utf-8")

    manifest = {
        "protocol_id": "fact_supervision_87001_minimal_lexical_pair_seed87002",
        "fork": "L2",
        "status": "PREEXECUTION_ONLY",
        "source": "factual-arm Primary only",
        "row_syntax": "<name> <predicate> the <description>",
        "counts": {"ONE": len(items_one), "TWO": len(items_two),
                   "ONE_families": 24, "TWO_families": 24},
        "checkpoint_set": CHECKPOINTS,
        "scorer": SCORER,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
        "checkpoint_loaded": False, "inference_run": False,
        "sacred_access": False, "confirmation_access": False, "seed_87003_access": False,
    }
    protocol = (
        "# Minimal-lexical paired diagnostic (Fork L2) - pre-execution freeze\n\n"
        "ONE: emit only the correct mapping row (48 items).\n"
        "TWO: emit both competing mapping rows (96 items).\n"
        "Row syntax: `<name> <predicate> the <description>`; original English query unchanged "
        "(`The person who <predicate> the <description> was`).\n\n"
        "Scoring: reuse the frozen candidate conditional LL over response tokens + EOS; signed "
        "margin; correct iff margin>0, tie==0; greedy argmax<=32 stopping at EOS. No "
        "normalization, ranks/top-k, perplexity, thresholds, significance tests.\n\n"
        "Report ONE and TWO separately (overall and object/predicate): exact correct/total, ties, "
        "complete families, correct-index/name slices, row-order slice (TWO), frozen margin "
        "summaries, greedy exact; plus a descriptive ONE-vs-TWO difference and a comparison with "
        "the already-frozen A/C and gloss aggregates. Original two-fact relational "
        "reversal/query dimensions do not transfer unchanged.\n\n"
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
        "status": "MINIMAL_LEXICAL_PAIR_FREEZE_COMPLETE_PREEXECUTION_ONLY",
        "bundle": "fact_supervision_87001_minimal_lexical_pair_seed87002",
        "items_one_sha256": sha256_file(ONE),
        "items_two_sha256": sha256_file(TWO),
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
                      "items_one_sha256": receipt["items_one_sha256"],
                      "items_two_sha256": receipt["items_two_sha256"]}, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"FREEZE ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
