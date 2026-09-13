"""Independent fail-closed verification of the sealed English characterization.

verify() reads files and distribution metadata only. It never imports model code,
deserializes checkpoints, executes the frozen builder, or writes a file. The CLI
writes its completed receipt to this execution directory unless --stdout is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
from itertools import combinations, product
import json
from pathlib import Path
import random
import re
import sys
import time


EXAM = Path(r"C:\DaveLM-CADAVER\english_context_characterization_v1_seed8380")
EXECUTION = Path(__file__).resolve().parent
RECEIPT_SHA256 = "26cf767af6f85738c52e506f6dcfe2a34fbcce40a9b2829101408038cfa81e1b"
PAYLOADS = {
    "FAMILIES.json", "FREEZE_RECEIPT.md", "ITEMS.jsonl", "LEXICON.json",
    "MANIFEST.json", "PREFLIGHT.json", "PRIORS.jsonl", "PROTOCOL.md",
    "implementation/build_and_validate.py", "implementation/english_scoring.py",
}
WORDS = {
    "names": ["Ben", "Sam", "Tim", "Tom"],
    "animals": ["bear", "cat", "fish"],
    "objects": ["bag", "ball", "book", "car", "hat", "toy"],
    "places": ["bed", "cave", "room", "shop", "yard"],
}
PAIRS = {
    "names": list(combinations(WORDS["names"], 2)),
    "animals": list(combinations(WORDS["animals"], 2)),
    "objects": [("bag", "book"), ("bag", "hat"), ("ball", "car"),
                ("ball", "toy"), ("book", "hat"), ("car", "toy")],
    "places": [("bed", "room"), ("cave", "shop"), ("cave", "yard"), ("shop", "yard")],
}


def require(condition, description):
    if not condition:
        raise RuntimeError("INPUT VERIFICATION FAILED: " + description)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_ready(value):
    return json.loads(json.dumps(value))


def norm(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def token_arrays(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if (key in {"full_document_token_ids", "token_ids", "document_token_ids"}
                    and isinstance(value, list) and value
                    and all(type(x) is int for x in value)):
                yield tuple(value)
            else:
                yield from token_arrays(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from token_arrays(value)


def verify():
    """Return a JSON-ready PASS receipt; raise on the first mismatch. No writes."""
    started = time.monotonic()
    require(sys.version_info[:2] == (3, 12), "Python 3.12 is required")
    versions = {name: metadata.version(name) for name in ("tokenizers", "torch")}
    require(versions["tokenizers"] == "0.23.1", "tokenizers version changed")
    require(versions["torch"] == "2.9.1+rocmsdk20260116", "torch distribution version changed")
    receipt_path = EXAM / "SHA256SUMS.txt"
    require(sha(receipt_path) == RECEIPT_SHA256, "detached receipt SHA-256 changed")
    payload_hashes = {}
    for line in receipt_path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\r\n]+)", line)
        require(match is not None, "malformed detached receipt line")
        digest, name = match.groups()
        require(name in PAYLOADS and name not in payload_hashes, "unexpected or duplicate sealed path")
        path = EXAM / name
        actual = sha(path)
        require(actual == digest, "sealed payload hash mismatch: " + name)
        data = path.read_bytes()
        require(not data.startswith(b"\xef\xbb\xbf") and b"\r" not in data,
                "sealed payload must use LF and no BOM: " + name)
        data.decode("utf-8")
        payload_hashes[name] = {"sha256": actual, "bytes": len(data), "path": str(path)}
    require(set(payload_hashes) == PAYLOADS, "receipt must seal exactly ten payloads")
    manifest = read_json(EXAM / "MANIFEST.json")
    frozen = read_json(EXAM / "PREFLIGHT.json")
    lexicon = read_json(EXAM / "LEXICON.json")
    family_data = read_json(EXAM / "FAMILIES.json")
    items = [json.loads(s) for s in (EXAM / "ITEMS.jsonl").read_text(encoding="utf-8").splitlines()]
    priors = [json.loads(s) for s in (EXAM / "PRIORS.jsonl").read_text(encoding="utf-8").splitlines()]
    verified_files = {}
    nested_records = 0

    def check_nested(obj):
        nonlocal nested_records
        if isinstance(obj, dict):
            if "path" in obj and "sha256" in obj:
                path = Path(obj["path"])
                require(not any(word in path.name.casefold() for word in ("retention", "sacred")),
                        "forbidden exam/retention path")
                key = str(path.resolve())
                if key not in verified_files:
                    verified_files[key] = {"sha256": sha(path), "bytes": path.stat().st_size}
                current = verified_files[key]
                require(current["sha256"] == obj["sha256"], "nested hash mismatch: " + key)
                if "bytes" in obj:
                    require(current["bytes"] == obj["bytes"], "nested byte count mismatch: " + key)
                if obj.get("expected_sha256") is not None:
                    require(current["sha256"] == obj["expected_sha256"], "expected hash mismatch: " + key)
                nested_records += 1
            for value in obj.values():
                check_nested(value)
        elif isinstance(obj, list):
            for value in obj:
                check_nested(value)

    check_nested(manifest)
    check_nested(frozen)
    require(manifest["seed"] == family_data["seed"] == 8380, "sampling seed changed")
    require(frozen["status"] == "PASS_PRE_EXECUTION_MATERIAL_VALIDATION", "frozen preflight status")
    require(manifest["runtime"]["construction_python"] == ".".join(map(str, sys.version_info[:3])),
            "construction Python patch version changed")
    require(manifest["runtime"]["tokenizers"] == versions["tokenizers"], "manifest tokenizer runtime")
    require(manifest["runtime"]["torch_distribution_version_not_imported"] == versions["torch"],
            "manifest torch metadata")

    # Tokenizers is the only non-stdlib import; no model/checkpoint module is imported.
    from tokenizers import Tokenizer
    tokenizer_path = Path(manifest["source_inputs"]["tokenizer"]["path"])
    tokenizer_json = read_json(tokenizer_path)
    require(tokenizer_json["model"]["byte_fallback"] is False, "byte fallback differs")
    require(tokenizer_json["pre_tokenizer"]["type"] == "ByteLevel", "ByteLevel differs")
    tok = Tokenizer.from_file(str(tokenizer_path))
    enc = lambda text: tok.encode(text, add_special_tokens=False).ids
    dec = lambda ids: tok.decode(ids, skip_special_tokens=False)
    require(tok.get_vocab_size() == 1024 and tok.token_to_id("<bos>") == 2
            and tok.token_to_id("<eos>") == 3, "tokenizer vocabulary/special IDs")
    require(lexicon["eligible_pairs"] == json_ready(PAIRS), "explicit lexical pair lists")
    families = {}
    family_rows = []
    for task, ek, vk, count, size in (("naming", "animals", "names", 8, 18),
                                    ("possession", "names", "objects", 16, 36),
                                    ("location", "names", "places", 16, 24)):
        universe = sorted(product(PAIRS[ek], PAIRS[vk]))
        chosen = sorted(random.Random(8380).sample(universe, count))
        record = family_data["tasks"][task]
        require(len(universe) == size == record["eligible_universe_size"], "eligible universe size")
        require(record["sample_size"] == count, "sample size")
        require(record["eligible_universe"] == [{"entities": list(e), "values": list(v)}
                for e, v in universe], "eligible universe enumeration")
        expected_families = []
        for index, (entities, values) in enumerate(chosen):
            fid = f"{task}:f{index:03d}"
            rec = {"family_id": fid, "task": task, "entities": list(entities), "values": list(values),
                   "candidates": list(entities if task == "possession" else values),
                   "eligible_universe_index": universe.index((entities, values)),
                   "frames": ["core"] if task == "naming" else ["core", "frame_holdout", "qa"]}
            expected_families.append(rec)
            families[fid] = rec
            family_rows.append(rec)
        require(record["selected_families"] == expected_families, "seed-8380 selected identities")
    require(frozen["selected_family_identities"] == family_rows, "preflight selected families")
    require(len(items) == 832 and len(priors) == 208 and len(families) == 40, "material counts")
    require(len({x["item_id"] for x in items}) == len({x["prompt"] for x in items}) == 832,
            "duplicate item ID/prompt")
    prior_map = {x["prior_id"]: x for x in priors}
    by_id = {x["item_id"]: x for x in items}
    require(len(prior_map) == 208, "duplicate prior IDs")
    groups, reversals = defaultdict(list), defaultdict(list)
    for x in items:
        f = families[x["family_id"]]
        e, v, task = f["entities"], f["values"], f["task"]
        a, q, o, frame = x["assignment"], x["query_index"], x["fact_order"], x["frame"]
        require(all(type(z) is int and z in (0, 1) for z in (a, q, o)), "binary factor type/value")
        require(frame in f["frames"], "unapproved frame")
        require(x["task"] == task and x["entities"] == e and x["values"] == v
                and x["candidates"] == f["candidates"], "item-family inconsistency")
        mapping = {e[i]: v[i ^ a] for i in range(2)}
        require(x["mapping"] == mapping, "mapping differs from assignment")
        def fact(i):
            if task == "naming":
                return f"The {e[i]} is called {mapping[e[i]]}."
            if task == "possession":
                return f"{e[i]} has the {mapping[e[i]]}."
            return f"{e[i]} is in the {mapping[e[i]]}."
        facts = [fact(o), fact(1 - o)]
        if task == "naming":
            query = f"The name of the {e[q]} is"
        elif task == "possession":
            query = {"core": f"The person with the {v[q]} is", "frame_holdout": f"The {v[q]} is with",
                     "qa": f"Who has the {v[q]}?"}[frame]
        else:
            query = {"core": f"{e[q]} is in the", "frame_holdout": f"{e[q]} can be found in the",
                     "qa": f"Where is {e[q]}?"}[frame]
        require(x["ordered_facts"] == facts and x["query"] == query
                and x["prompt"] == " ".join(facts) + "\n" + query, "rendered facts/query/prompt")
        answer = next(key for key, val in mapping.items() if val == v[q]) if task == "possession" else mapping[e[q]]
        require(x["correct_index"] == q ^ a and x["correct_candidate"] == answer
                and f["candidates"][q ^ a] == answer, "XOR and independent answer lookup")
        mention = sorted(range(2), key=lambda i: " ".join(facts).index(f["candidates"][i]))
        require(x["candidate_mention_order"] == mention
                and x["correct_candidate_mention_rank"] == mention.index(q ^ a), "candidate recency metadata")
        require(x["queried_fact_rank"] == [o, 1-o].index((q ^ a) if task == "possession" else q),
                "queried fact position")
        for candidate in f["candidates"]:
            require(len(re.findall(r"(?<![A-Za-z])" + re.escape(candidate) + r"(?![A-Za-z])",
                                   " ".join(facts))) == 1, "candidate fact occurrences")
        prefix = f"{x['family_id']}:{frame}"
        require(x["item_id"] == f"{prefix}:a{a}q{q}o{o}" and x["prior_id"] == f"{prefix}:q{q}:prior"
                and x["reversal_pair_id"] == f"{prefix}:q{q}o{o}", "stable item/link IDs")
        require(x["role"] == ("primary" if task != "naming" and frame == "core" else "diagnostic"),
                "primary/diagnostic separation")
        prior = prior_map[x["prior_id"]]
        require(prior["prompt"] == query and prior["candidates"] == x["candidates"], "prior query/candidates")
        groups[x["family_id"], frame].append(x)
        reversals[x["reversal_pair_id"]].append(x)
    require(len(groups) == 104 and len(reversals) == 416, "family/frame and reversal counts")
    balances = []
    for (fid, frame), rows in groups.items():
        require([(x["assignment"], x["query_index"], x["fact_order"]) for x in rows]
                == list(product(range(2), repeat=3)), "eight-member enumeration")
        binary = {key: dict(sorted(Counter(x[key] for x in rows).items())) for key in
                  ("assignment", "query_index", "fact_order", "correct_index",
                   "correct_candidate_mention_rank", "queried_fact_rank")}
        require(all(c == {0: 4, 1: 4} for c in binary.values()), "binary factor balance")
        mentions = {str(i): dict(Counter(x["candidate_mention_order"].index(i) for x in rows)) for i in (0, 1)}
        require(all(c == {0: 4, 1: 4} for c in mentions.values()), "mention balance")
        roles = Counter((e, v) for x in rows for e, v in x["mapping"].items())
        require(len(roles) == 4 and set(roles.values()) == {4}, "entity-value role balance")
        balances.append({"family_id": fid, "frame": frame, "items": 8, "reversal_pairs": 4,
                         "binary_factor_counts": binary, "candidate_mention_rank_counts": mentions,
                         "candidate_correct_incorrect_counts": {c: {"correct": 4, "incorrect": 4}
                                                               for c in rows[0]["candidates"]},
                         "entity_value_role_counts": [{"entity": e, "value": v, "count": n}
                                                      for (e, v), n in sorted(roles.items())]})
    require(json_ready(balances) == frozen["per_family_frame_balancing"], "preflight balancing records")
    for rows in reversals.values():
        require(len(rows) == 2 and {x["assignment"] for x in rows} == {0, 1}
                and {x["correct_index"] for x in rows} == {0, 1}, "reversal completeness")
        require(all(rows[0][key] == rows[1][key] for key in ("query", "fact_order", "candidates")),
                "unmatched reversal")
    for prior in priors:
        linked = [by_id[item_id] for item_id in prior["linked_item_ids"]]
        require(len(linked) == 4 and prior["correct_index"] is None, "logical prior metadata")
        require(prior["linked_item_ids"] == [x["item_id"] for x in items if x["prior_id"] == prior["prior_id"]],
                "prior links/order")
        require({(x["assignment"], x["fact_order"]) for x in linked} == set(product(range(2), repeat=2)),
                "prior assignment/order coverage")
        require(all(x[key] == prior[key] for x in linked for key in
                    ("family_id", "frame", "query_index", "task", "candidates", "candidate_completions", "candidate_token_ids")),
                "prior/context metadata")
        require(prior["role"] == "query_only_prior_diagnostic", "prior role")
    boundaries, roundtrips, lengths, contextual_lengths = 0, 0, [], []
    for row in items + priors:
        prompt, ids = row["prompt"], row["prompt_token_ids"]
        require(row["bos_token_id"] == 2 and not prompt[-1].isspace() and enc(prompt) == ids
                and dec(ids) == prompt, "prompt tokenization/roundtrip")
        require(row["candidate_completions"] == [" " + c + "." for c in row["candidates"]]
                and len(row["candidate_token_ids"]) == 2, "candidate serialization")
        roundtrips += 1
        for completion, candidate_ids in zip(row["candidate_completions"], row["candidate_token_ids"]):
            require(len(candidate_ids) == 3 and candidate_ids[-1] == 18
                    and enc(completion) == candidate_ids and enc(prompt + completion) == ids + candidate_ids,
                    "completion tokens/boundary")
            require(dec(candidate_ids) == completion and dec(ids + candidate_ids) == prompt + completion,
                    "completion roundtrips")
            require(not set(ids + candidate_ids) & set(range(5)), "unexpected special token")
            length = 1 + len(ids) + len(candidate_ids)
            require(length <= 256, "completed context overflow")
            lengths.append(length)
            if "item_id" in row:
                contextual_lengths.append(length)
            boundaries += 1
            roundtrips += 2
    observed = {"contextual_items": len(items), "logical_priors": len(priors),
                "unique_contextual_prompts": len({x["prompt"] for x in items}),
                "unique_prior_prompts": len({x["prompt"] for x in priors}),
                "unique_prior_prompt_candidate_pairs": len({(x["prompt"], tuple(x["candidate_completions"])) for x in priors}),
                "family_frame_groups": len(groups), "items_per_family_frame": 8,
                "reversal_pairs_per_family_frame": 4, "total_reversal_pairs": len(reversals),
                "item_counts_by_task_frame": dict(sorted(Counter(f"{x['task']}/{x['frame']}" for x in items).items())),
                "prior_counts_by_task_frame": dict(sorted(Counter(f"{x['task']}/{x['frame']}" for x in priors).items()))}
    require(all(frozen[key] == value for key, value in observed.items()), "aggregate material preflight")
    token_receipt = {"byte_fallback": False, "pre_tokenizer": "ByteLevel", "vocabulary_size": 1024,
                     "bos_id": 2, "eos_id": 3, "candidate_completed_token_lengths": [3],
                     "candidate_word_token_lengths": [2], "boundary_checks": boundaries,
                     "encode_decode_checks": roundtrips, "boundary_failures": 0, "roundtrip_failures": 0,
                     "max_completed_input_including_bos": max(lengths), "min_completed_input_including_bos": min(lengths),
                     "max_contextual_completed_input_including_bos": max(contextual_lengths)}
    require(token_receipt == frozen["tokenizer"] and min(contextual_lengths) == 26, "tokenizer preflight reproduction")
    inputs = manifest["source_inputs"]
    corpora = {split: [json.loads(s) for s in Path(inputs[split]["path"]).read_text(encoding="utf-8").splitlines()]
               for split in ("train", "dev")}
    texts = {split: [x["text"] for x in rows] for split, rows in corpora.items()}
    require(len(texts["train"]) == 9000 and len(texts["dev"]) == 1000, "corpus document counts")
    raw = Path(inputs["source_corpus"]["path"]).read_text(encoding="utf-8")
    stories = [s.strip().replace("\r\n", "\n").replace("\r", "\n")
               for s in raw.split("<|endoftext|>") if s.strip()]
    unique = {hashlib.sha256(s.encode("utf-8")).hexdigest(): s for s in stories}
    chosen = [unique[h] for h in sorted(unique)[:10000]]
    require(texts["train"] == chosen[:9000] and texts["dev"] == chosen[9000:], "source corpus partition")
    require(not set(texts["train"]) & set(texts["dev"]), "exact train/dev story overlap")
    for rows in corpora.values():
        for row in rows:
            require(enc(row["text"]) == row["token_ids"], "stored corpus token array")
    require(set(lexicon["words"]) == {w for words in WORDS.values() for w in words}, "lexicon inventory")
    for word, record in lexicon["words"].items():
        regex = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])")
        for split, documents in texts.items():
            counts = [len(regex.findall(text)) for text in documents]
            require(record["frequencies"][split] == {"occurrences": sum(counts), "stories": sum(c > 0 for c in counts)},
                    "word frequency: " + word + "/" + split)
        require(record["frequencies"]["train"]["occurrences"] >= 100
                and record["frequencies"]["train"]["stories"] >= 50, "lexical accessibility")
        require(record["strings"] == {"bare": word, "leading_space": " " + word, "completed": " " + word + "."},
                "lexical strings")
        for kind, text in record["strings"].items():
            ids = enc(text)
            require(ids == record["token_ids"][kind] and dec(ids) == text and not set(ids) & set(range(5)),
                    "lexical token IDs/roundtrip")
            require(tok.encode(text, add_special_tokens=False).tokens == record["tokens"][kind], "lexical token strings")
        require(len(record["token_ids"]["leading_space"]) == 2 and record["eligible"] is True,
                "lexical token-length eligibility")
    for category, words in WORDS.items():
        eligible = [pair for pair in combinations(words, 2)
                    if max(lexicon["words"][w]["frequencies"]["train"]["occurrences"] for w in pair)
                    <= 2 * min(lexicon["words"][w]["frequencies"]["train"]["occurrences"] for w in pair)]
        require(eligible == PAIRS[category], "frequency-matched pair universe")
    normalized = {split: [norm(s) for s in stories] for split, stories in texts.items()}
    strings = {"atomic_fact": {norm(s) for x in items for s in x["ordered_facts"]},
               "two_fact_context": {norm(" ".join(x["ordered_facts"])) for x in items},
               "whole_prompt": {norm(x["prompt"]) for x in items},
               "whole_prompt_plus_either_candidate": {norm(x["prompt"] + c) for x in items for c in x["candidate_completions"]}}
    overlap = {}
    # Exact absence of each complete atomic fact implies absence of each containing
    # longer test string. Verify containment explicitly, avoiding redundant scans.
    for split, documents in normalized.items():
        for atom in strings["atomic_fact"]:
            require(not any(atom in text for text in documents), "atomic-fact corpus overlap")
    for level, values in strings.items():
        if level != "atomic_fact":
            require(all(any(atom in text for atom in strings["atomic_fact"]) for text in values),
                    "longer-string absence proof lacks constituent atom")
        overlap[level] = {"distinct_strings": len(values), "results": {
            split: {"matched_test_strings": 0, "matches": []} for split in normalized}}
    require(overlap == frozen["corpus"]["overlap"], "exact normalized corpus overlap preflight")
    for record in frozen["corpus"]["whitelist_template_counts"].values():
        for split, docs in normalized.items():
            require(sum(len(re.findall(record["regex"], s)) for s in docs) == record["counts"][split],
                    "whitelist template frequency")
    for phrase, counts in frozen["corpus"]["ordinary_phrase_counts"].items():
        pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
        for split, docs in normalized.items():
            require(sum(len(re.findall(pattern, s)) for s in docs) == counts[split], "ordinary phrase frequency")
    binding = frozen["common_binding_reference"]
    devsets = []
    for record in binding["references"].values():
        obj = read_json(record["path"])
        quartets = obj["quartets"]
        docs = [d for q in quartets for d in q["docs"]]
        arrays = {tuple(d["full_document_token_ids"]) for d in docs}
        require(len(quartets) == 20 and len(docs) == len(arrays) == 80, "binding reference counts")
        require(all(len(q["docs"]) == 4 and {d["member"] for d in q["docs"]}
                    == {"o1_k0", "o1_k1", "o2_k0", "o2_k1"} for q in quartets), "binding quartet completeness")
        require(dict(Counter(d["layout_combo"] for d in docs)) == record["layouts"]
                == {"p56_b12": 40, "p58_b13": 40}, "binding layouts")
        require([q["quartet_id"] for q in quartets] == record["quartet_ids"], "binding quartet identities")
        devsets.append(arrays)
    require(len(devsets) == 2 and not devsets[0] & devsets[1], "binding DEV disjointness")
    require(len(binding["training_checks"]) == 6, "known nonsacred training references")
    for record in binding["training_checks"]:
        path = Path(record["path"])
        require("retention" not in path.name.casefold() and "sacred" not in path.name.casefold(), "forbidden binding file")
        arrays = set(token_arrays(read_json(path)))
        require(len(arrays) == record["unique_arrays"], "training token-array counts")
        require([len(ds & arrays) for ds in devsets] == [record["pilot0_dev_overlap"], record["pilot1_dev_overlap"]]
                == [0, 0], "binding training disjointness")
    # Verify every byte input again after material inspection; no mutations accepted.
    for path, record in verified_files.items():
        require(sha(path) == record["sha256"], "input changed during verification: " + path)
    for name, record in payload_hashes.items():
        require(sha(EXAM / name) == record["sha256"], "sealed payload changed during verification")
    require(sha(receipt_path) == RECEIPT_SHA256, "receipt changed during verification")
    return {"status": "PASS_EXECUTION_INPUT_VERIFICATION", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started, "exam_directory": str(EXAM),
            "detached_receipt_sha256": RECEIPT_SHA256, "sealed_payload_count": len(payload_hashes),
            "sealed_payloads": payload_hashes, "verified_nested_hash_records": nested_records,
            "verified_unique_input_files": len(verified_files), "current_actual_file_hashes": verified_files,
            "runtime": {"python": sys.version, "executable": sys.executable, **versions,
                        "torch_imported_by_verifier": False}, "material": observed, "tokenizer": token_receipt,
            "corpus": {"source_partition_reproduced": True, "stored_story_token_arrays_verified": 10000,
                       "lexicon_words_verified": 18, "overlap": overlap,
                       "overlap_method": "Exact atomic-fact scans plus verified constituent containment for longer strings",
                       "whitelist_and_ordinary_phrase_counts_reproduced": True},
            "nonsacred_binding": {"dev_documents": [80, 80], "dev_intersection": 0,
                                  "known_training_sets_checked": 6, "training_intersections": [0] * 12},
            "checkpoint_deserializations": 0, "model_instantiations": 0, "model_forward_calls": 0,
            "optimizer_creations": 0, "backward_passes": 0, "behavioral_scores_produced": 0,
            "sacred_exam_opened": False, "input_files_modified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdout", action="store_true", help="Print receipt instead of writing VERIFICATION.json")
    args = parser.parse_args()
    result = verify()
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.stdout:
        print(serialized, end="")
    else:
        target = EXECUTION / "VERIFICATION.json"
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(serialized)
        print("PASS_EXECUTION_INPUT_VERIFICATION " + str(target))
