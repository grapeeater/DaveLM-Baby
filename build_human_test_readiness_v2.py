"""Outcome-blind construction of the replacement HUMAN_TEST_READY DEV battery.

This script intentionally imports no model code and opens no FINAL artifact.
It materializes a fresh nonsacred DEV battery, its independent validator, and a
non-circular freeze receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


ROOT = Path(r"C:\DaveLM-CADAVER")
OUT = ROOT / "human_test_readiness_v2_seed87010"
SEED = 87010
TOK_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
PILOT1 = ROOT / "language_pilot_1_early_block_protection_seed8380" / "pilot_run" / "checkpoints" / "seed_8380" / "latest.pt"
PILOT1_SHA = "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb"
TOK_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.write_text(text.replace("\r\n", "\n"), encoding="utf-8", newline="\n")


def write_json(path: Path, obj: Any) -> None:
    write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def strings_from_json(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for v in value.values():
            out.extend(strings_from_json(v))
        return out
    if isinstance(value, list):
        out: list[str] = []
        for v in value:
            out.extend(strings_from_json(v))
        return out
    return []


def load_strings(path: Path) -> list[str]:
    suffix = path.suffix.casefold()
    if suffix == ".jsonl":
        out: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.extend(strings_from_json(json.loads(line)))
        return out
    if suffix == ".json":
        return strings_from_json(json.loads(path.read_text(encoding="utf-8")))
    return [path.read_text(encoding="utf-8")]


def fact_items(tok: Tokenizer) -> list[dict[str, Any]]:
    # All candidate pairs have equal tokenizer length including the period.
    families = [
        ("F01", "Ash", "Cal", "found", "striped drum", "tiny boat", 0),
        ("F02", "Cam", "Chip", "carried", "silver key", "round map", 0),
        ("F03", "Cory", "Ike", "found", "orange shell", "green brush", 0),
        ("F04", "Ira", "Neil", "carried", "wool scarf", "paper crown", 1),
        ("F05", "Ned", "Ron", "found", "glass bead", "wooden spoon", 1),
        ("F06", "Sid", "Will", "carried", "purple flag", "square tile", 1),
    ]
    rows: list[dict[str, Any]] = []
    for fid, a, b, rel, x, y, fact_order in families:
        for assignment in (0, 1):
            mapping = {a: x, b: y} if assignment == 0 else {a: y, b: x}
            for query in (0, 1):
                query_obj = x if query == 0 else y
                actor = next(name for name, obj in mapping.items() if obj == query_obj)
                facts = [(a, rel, mapping[a]), (b, rel, mapping[b])]
                if fact_order == 1:
                    facts.reverse()
                rendered_facts = " ".join(f"{name} {predicate} the {obj}." for name, predicate, obj in facts)
                prompt = f"{rendered_facts} The {query_obj} was {rel} by"
                # Same candidate order in each matched reversal pair; query changes order.
                candidates = [f" {a}.", f" {b}."] if query == 0 else [f" {b}.", f" {a}."]
                correct_index = candidates.index(f" {actor}.")
                row = {
                    "id": f"fact:{fid}:a{assignment}:q{query}",
                    "category": "fact",
                    "family_id": f"fact:{fid}",
                    "reversal_pair_id": f"fact:{fid}:q{query}",
                    "assignment": assignment,
                    "query": query,
                    "fact_order": fact_order,
                    "relation": rel,
                    "facts": [{"subject": n, "relation": r, "object": o} for n, r, o in facts],
                    "query_object": query_obj,
                    "prompt": prompt,
                    "candidates": candidates,
                    "candidate_token_ids": [tok.encode(c).ids for c in candidates],
                    "correct_index": correct_index,
                    "correct_candidate": candidates[correct_index],
                    "generation_max_new_tokens": 32,
                    "scoring": "sum candidate-token conditional log probabilities; final EOS excluded; positive signed margin wins; tie fails",
                }
                rows.append(row)
    return rows


def paired_item(tok: Tokenizer, ident: str, category: str, prompt: str, candidates: list[str], correct_index: int, **extra: Any) -> dict[str, Any]:
    return {
        "id": ident,
        "category": category,
        "prompt": prompt,
        "candidates": candidates,
        "candidate_token_ids": [tok.encode(c).ids for c in candidates],
        "correct_index": correct_index,
        "correct_candidate": candidates[correct_index],
        "generation_max_new_tokens": 32,
        "scoring": "sum candidate-token conditional log probabilities; final EOS excluded; positive signed margin wins; tie fails; greedy exact requires candidate tokens then EOS",
        **extra,
    }


def instruction_items(tok: Tokenizer) -> list[dict[str, Any]]:
    specs = [
        ("yesno", "Answer with exactly one word: Is snow cold?", [" yes.", " no."], 0),
        ("yesno", "Answer with exactly one word: Is a stone an animal?", [" yes.", " no."], 1),
        ("yesno", "Reply with one word, yes or no: Does a bird have wings?", [" yes.", " no."], 0),
        ("yesno", "Reply with one word, yes or no: Is the moon a sandwich?", [" yes.", " no."], 1),
        ("mood", "Use one word, happy or sad: A child got a gift. The child feels", [" happy.", " sad."], 0),
        ("mood", "Use one word, happy or sad: A toy was lost. The child feels", [" happy.", " sad."], 1),
        ("mood", "Use one word, happy or sad: A puppy met its friend. The puppy feels", [" happy.", " sad."], 0),
        ("mood", "Use one word, happy or sad: A cup broke. The child feels", [" happy.", " sad."], 1),
        ("temperature", "Use one word, hot or cold: Soup from a stove is", [" hot.", " cold."], 0),
        ("temperature", "Use one word, hot or cold: Ice from a freezer is", [" hot.", " cold."], 1),
        ("temperature", "Use one word, hot or cold: Tea in a warm mug is", [" hot.", " cold."], 0),
        ("temperature", "Use one word, hot or cold: Snow on the ground is", [" hot.", " cold."], 1),
    ]
    return [paired_item(tok, f"instruction:{i:02d}", "instruction", prompt, candidates, correct, subtype=subtype) for i, (subtype, prompt, candidates, correct) in enumerate(specs, 1)]


def continuity_items(tok: Tokenizer) -> list[dict[str, Any]]:
    pairs = [("Ash", "Cal"), ("Cam", "Chip"), ("Cory", "Ike"), ("Ira", "Neil"), ("Ned", "Ron"), ("Sid", "Will")]
    rows: list[dict[str, Any]] = []
    for i, (a, b) in enumerate(pairs, 1):
        candidates = [f" {a}.", f" {b}."]
        rows.append(paired_item(tok, f"continuity:{i:02d}:a", "continuity", f"User: My name is {a}. Assistant: Hello. User: What is my name? Assistant:", candidates, 0, family_id=f"continuity:{i:02d}", memory_name=a))
        rows.append(paired_item(tok, f"continuity:{i:02d}:b", "continuity", f"User: Please call me {b}. Assistant: Okay. User: What name should I use for you? Assistant:", candidates, 1, family_id=f"continuity:{i:02d}", memory_name=b))
    return rows


def generation_items() -> list[dict[str, Any]]:
    prompts = [
        "A small bird landed on the fence and",
        "The child found a note under the pillow. The note said",
        "Mara dropped her paintbrush, so",
        "A dog heard a noise outside and",
        "The lamp went out during dinner. Everyone",
        "When the rain stopped, the children",
        "A boy gave his friend a paper star. His friend",
        "The kitten climbed onto the chair and",
        "After the cake was baked, the family",
        "A girl lost her mitten at the park. She",
        "What should a child do after spilling water on the floor?",
        "How can a person be kind to a new friend?",
        "What is one safe thing to do before crossing a street?",
        "Why might someone smile after receiving a letter?",
        "Tell a short sentence about a turtle.",
        "Say hello to a new neighbor.",
        "User: I feel tired today. Assistant:",
        "User: I found a lost pencil. What should I do? Assistant:",
        "User: My favorite color is green. What color do I like? Assistant:",
        "User: The bird is in the tree. Where is the bird? Assistant:",
    ]
    # The order is frozen by a declared seed, not by checkpoint behavior.
    rng = random.Random(SEED)
    rng.shuffle(prompts)
    return [{"id": f"generation:{i:02d}", "category": "generation", "prompt": p, "generation_max_new_tokens": 32, "scoring": "greedy generation only; raw output retained verbatim; automatic and blinded human rubric fields frozen in protocol"} for i, p in enumerate(prompts, 1)]


def parse_fact(prompt: str) -> tuple[list[tuple[str, str, str]], str, str]:
    pattern = re.compile(r"^([A-Z][a-z]+) (found|carried) the ([a-z ]+)\. ([A-Z][a-z]+) (found|carried) the ([a-z ]+)\. The ([a-z ]+) was (found|carried) by$")
    m = pattern.fullmatch(prompt)
    if not m:
        raise AssertionError(f"unparseable factual prompt: {prompt}")
    a, r1, o1, b, r2, o2, qo, qr = m.groups()
    if r1 != r2 or r1 != qr:
        raise AssertionError("relation mismatch")
    if qo not in {o1, o2}:
        raise AssertionError("query object absent")
    return [(a, r1, o1), (b, r2, o2)], qo, qr


def overlap_audit(items: list[dict[str, Any]]) -> dict[str, Any]:
    # Explicitly nonsacred, nonfinal sources. No wildcard source discovery is used.
    sources = [
        ROOT / "language_pilot_0_tinystories_seed8380" / "language_train.jsonl",
        ROOT / "language_pilot_0_tinystories_seed8380" / "language_dev.jsonl",
        ROOT / "human_readiness_hr1_causal_aligned_seed87006_v1" / "data" / "ENGLISH_TRAIN.jsonl",
        ROOT / "human_readiness_hr1_causal_aligned_seed87006_v1" / "data" / "ENGLISH_DEV.jsonl",
        ROOT / "human_test_readiness_v1" / "DEV_ITEMS.jsonl",
        ROOT / "post_p7_language_report_card_v2_seed8391" / "ITEMS.jsonl",
        ROOT / "post_p7_language_report_card_v3d_seed8391" / "ITEMS.jsonl",
        ROOT / "english_context_characterization_v1_seed8380" / "ITEMS.jsonl",
        ROOT / "english_context_characterization_v1_seed8380" / "PRIORS.jsonl",
        ROOT / "language_compositional_p7" / "TRAIN_TEXT.txt",
    ]
    prompts = {row["id"]: norm(row["prompt"]) for row in items}
    results: list[dict[str, Any]] = []
    hits: list[dict[str, str]] = []
    for source in sources:
        assert source.exists(), source
        ss = [norm(x) for x in load_strings(source) if x.strip()]
        joined = "\n".join(ss)
        local: list[str] = []
        for ident, prompt in prompts.items():
            if prompt in joined:
                local.append(ident)
                hits.append({"source": str(source), "item_id": ident})
        results.append({"path": str(source), "sha256": sha(source), "string_count": len(ss), "exact_contiguous_prompt_hits": local})
    return {
        "normalization": "Unicode casefolding plus whitespace collapse; contiguous decoded-text prompt match",
        "sources": results,
        "hits": hits,
        "limitation": "Exact decoded-text non-overlap does not rule out semantic or near-duplicate exposure. FINAL_ITEMS and sacred material were not opened or scanned.",
        "final_accessed": False,
        "sacred_accessed": False,
    }


def payload_manifest(out: Path) -> list[dict[str, Any]]:
    excluded = {"SHA256SUMS.txt", "FREEZE_RECEIPT.json", "FREEZE_RECEIPT.sha256"}
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in excluded)
    return [{"path": p.relative_to(out).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size} for p in files]


def make_readonly(out: Path) -> None:
    for p in out.rglob("*"):
        if p.is_file():
            p.chmod(p.stat().st_mode & ~stat.S_IWRITE)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing bundle: {OUT}")
    OUT.mkdir(parents=True)
    tok = Tokenizer.from_file(str(TOK_PATH))
    assert sha(TOK_PATH) == TOK_SHA
    assert sha(PILOT1) == PILOT1_SHA

    facts = fact_items(tok)
    instructions = instruction_items(tok)
    continuity = continuity_items(tok)
    generations = generation_items()
    all_items = facts + instructions + continuity + generations

    # Independent structural checks before any file is frozen.
    assert len(facts) == 24 and len(instructions) == 12 and len(continuity) == 12 and len(generations) == 20
    assert len({x["prompt"] for x in all_items}) == len(all_items)
    for row in facts + instructions + continuity:
        assert len(row["candidate_token_ids"][0]) == len(row["candidate_token_ids"][1])
        for candidate, ids in zip(row["candidates"], row["candidate_token_ids"]):
            assert tok.decode(ids) == candidate
            assert tok.encode(row["prompt"] + candidate).ids == tok.encode(row["prompt"]).ids + ids
        if row["category"] == "fact":
            parsed, query_obj, rel = parse_fact(row["prompt"])
            answer = next(name for name, predicate, obj in parsed if predicate == rel and obj == query_obj)
            assert row["correct_candidate"] == f" {answer}."
    assert sum(x["correct_index"] == 0 for x in facts) == 12
    assert sum(x["correct_index"] == 1 for x in facts) == 12
    assert sum(x["fact_order"] == 0 for x in facts) == 12
    assert sum(x["fact_order"] == 1 for x in facts) == 12
    assert len({x["reversal_pair_id"] for x in facts}) == 12
    for rp in {x["reversal_pair_id"] for x in facts}:
        vals = [x for x in facts if x["reversal_pair_id"] == rp]
        assert len(vals) == 2 and {x["assignment"] for x in vals} == {0, 1}
        assert vals[0]["candidates"] == vals[1]["candidates"]
        assert {x["correct_index"] for x in vals} == {0, 1}
    for row in all_items:
        ids = tok.encode(row["prompt"]).ids
        assert tok.decode(ids) == row["prompt"]
        assert 1 + len(ids) + row.get("generation_max_new_tokens", 0) <= 256

    audit = overlap_audit(all_items)
    if audit["hits"]:
        raise RuntimeError(f"prohibited exact overlap: {audit['hits']}")

    erratum = """# HUMAN_TEST_READY v1 historical erratum\n\nThe historical `human_test_readiness_v1/PROTOCOL.md` specified 20 short generation prompts and 20 controlled fact items. Its preserved `DEV_ITEMS.jsonl` instead contains 26 items: 8 generation, 10 fact, 4 instruction, and 4 continuity. The historical artifact and all prior measurements remain unchanged. No result from that artifact is retroactively classified under this replacement protocol.\n\nThis erratum creates no claim about any checkpoint. It records why `human_test_readiness_v2_seed87010` is a new prospective development battery for future candidates only. The sealed FINAL readiness battery was not opened, inspected, derived from, or scored.\n"""
    protocol = """# HUMAN_TEST_READY development protocol v2\n\nThis is a prospective, nonsacred development battery built outcome-blindly with construction seed 87010. It is not the sealed FINAL battery. It must be frozen before evaluating any future candidate.\n\n## Battery\n\n- 20 ordinary short generation prompts, greedy decoding only, max 32 tokens.\n- 24 factual cloze items: 6 families x 4 assignment/query cells, with 12 matched reversal pairs. Candidate pairs are equal-token-length under the approved tokenizer.\n- 12 elementary instruction items, balanced across yes/no, happy/sad, and hot/cold exact responses.\n- 12 two-turn continuity items, balanced across six name pairs.\n\n## Controlled scoring\n\nFor fact, instruction, and continuity items, score each full candidate token sequence conditionally on BOS plus the exact prompt. Sum only candidate-token log probabilities; exclude EOS; apply no length normalization or prior subtraction. A strictly positive correct-minus-incorrect margin is correct. A zero margin fails. Greedy exact success requires the exact correct candidate tokens followed immediately by EOS within 32 tokens.\n\n## Generation rubric\n\nPreserve every greedy response verbatim. Automatic failure fields are immediate EOS, any three-identical-token run, any repeated decoded trigram, and a duplicate normalized non-EOS response shared by more than three prompts. Two blinded reviewers independently mark grammatical completeness and prompt relevance using `HUMAN_REVIEW_RUBRIC.md`; a disagreement is not a pass.\n\n## Advancement gates\n\nA development candidate must pass all gates: aligned TinyStories DEV loss <= 3.25 on the fixed 128-record causal slice; fact candidate correctness >= 20/24, >= 10/12 strict reversal pairs, >= 4/6 complete families, and >= 18/24 greedy exact; instruction candidate correctness >= 10/12 and greedy exact >= 9/12; continuity candidate correctness >= 10/12 and greedy exact >= 9/12; generation non-immediate-EOS >= 16/20, automatic non-degenerate >= 16/20, reviewer-complete >= 14/20, reviewer-relevant >= 14/20, and both reviewer fields passing >= 12/20. Both nonsacred binding pools independently require answer >= 76/80, BOTH_DISTINCT >= 76/80, and zero collapse.\n\nLoss is a safety diagnostic and cannot substitute for behavior. No candidate may access FINAL unless all development gates pass.\n"""
    rubric = """# Frozen generation review rubric\n\nFor each raw response, two reviewers work independently without seeing checkpoint identity.\n\n- `complete`: 1 only if the response contains a grammatical, completed English sentence ending in `.`, `?`, or `!`, without truncation or an unfinished trailing fragment.\n- `relevant`: 1 only if the response answers the prompt or continues its stated event while preserving an appropriate prompted entity, relation, question answer, or social act. A response dominated by an unrelated newly introduced actor/event, contradiction, prompt copying, or generic template receives 0.\n- `automatic_non_degenerate`: computed, not judged. It is 1 only if the response is non-EOS, has no three-identical-token run, no repeated decoded trigram, and is not a fourth-or-later duplicate normalized non-EOS response.\n\nA reviewer disagreement is scored 0 for the corresponding advancement field. Raw text is never repaired before review.\n"""
    lexicon = {
        "construction_seed": SEED,
        "tokenizer_sha256": TOK_SHA,
        "fact_name_pairs": [["Ash", "Cal"], ["Cam", "Chip"], ["Cory", "Ike"], ["Ira", "Neil"], ["Ned", "Ron"], ["Sid", "Will"]],
        "relations": ["found", "carried"],
        "candidate_length_rule": "each within-item pair must have equal token sequence length, including leading space and terminal period",
    }
    construction = {
        "seed": SEED,
        "outcome_blind": True,
        "checkpoint_loaded": False,
        "model_behavior_accessed": False,
        "final_accessed": False,
        "sacred_accessed": False,
        "algorithm": "fixed semantic family list plus deterministic generation-prompt shuffle using Python random.Random(87010); answer keys are derived from rendered factual prompt semantics",
    }
    preflight = {
        "status": "PASS_PROSPECTIVE_READINESS_DEV_FREEZE",
        "counts": {"generation": 20, "fact": 24, "fact_families": 6, "fact_reversal_pairs": 12, "instruction": 12, "continuity": 12, "total": len(all_items)},
        "tokenizer_sha256": TOK_SHA,
        "pilot1_parent_sha256": PILOT1_SHA,
        "unique_prompts": len({x["prompt"] for x in all_items}),
        "final_accessed": False,
        "sacred_accessed": False,
        "negative_validator_tests": ["wrong factual answer", "wrong factual predicate", "missing query object", "broken reversal pair", "unequal candidate token length"],
    }

    write_jsonl(OUT / "DEV_FACTS.jsonl", facts)
    write_jsonl(OUT / "DEV_INSTRUCTIONS.jsonl", instructions)
    write_jsonl(OUT / "DEV_CONTINUITY.jsonl", continuity)
    write_jsonl(OUT / "DEV_GENERATION.jsonl", generations)
    write_jsonl(OUT / "DEV_ITEMS.jsonl", all_items)
    write_json(OUT / "LEXICON.json", lexicon)
    write_json(OUT / "CONSTRUCTION.json", construction)
    write_json(OUT / "OVERLAP_AUDIT.json", audit)
    write_json(OUT / "PREFLIGHT.json", preflight)
    write_text(OUT / "ERRATUM.md", erratum)
    write_text(OUT / "PROTOCOL.md", protocol)
    write_text(OUT / "HUMAN_REVIEW_RUBRIC.md", rubric)
    shutil.copy2(Path(__file__), OUT / "BUILD_READINESS_V2.py")
    validator = ROOT / "validate_human_test_readiness_v2.py"
    if not validator.exists():
        raise RuntimeError(f"missing independent validator: {validator}")
    shutil.copy2(validator, OUT / "INDEPENDENT_VALIDATOR.py")
    subprocess.run([sys.executable, str(OUT / "INDEPENDENT_VALIDATOR.py"), "--bundle", str(OUT)], check=True)

    manifest = {"bundle": OUT.name, "payload": payload_manifest(OUT)}
    write_json(OUT / "MANIFEST.json", manifest)
    sums = payload_manifest(OUT)
    write_text(OUT / "SHA256SUMS.txt", "".join(f"{x['sha256']}  {x['path']}\n" for x in sums))
    receipt = {
        "status": "FROZEN_PROSPECTIVE_READINESS_DEV_V2",
        "bundle": str(OUT),
        "construction_seed": SEED,
        "manifest_sha256": sha(OUT / "MANIFEST.json"),
        "sha256sums_sha256": sha(OUT / "SHA256SUMS.txt"),
        "tokenizer_sha256": TOK_SHA,
        "parent_sha256": PILOT1_SHA,
        "historical_erratum": "ERRATUM.md",
        "final_accessed": False,
        "sacred_accessed": False,
    }
    write_json(OUT / "FREEZE_RECEIPT.json", receipt)
    write_text(OUT / "FREEZE_RECEIPT.sha256", sha(OUT / "FREEZE_RECEIPT.json") + "  FREEZE_RECEIPT.json\n")
    make_readonly(OUT)
    print(json.dumps({"status": receipt["status"], "bundle": str(OUT), "counts": preflight["counts"], "receipt_sha256": sha(OUT / "FREEZE_RECEIPT.json")}, indent=2))


if __name__ == "__main__":
    main()
