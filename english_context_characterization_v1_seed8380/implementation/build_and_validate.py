"""Materialize and validate the approved exam. Never loads or runs a model.

Run with the existing Python 3.12/tokenizers runtime and -B. Writes only the
new exam directory; all project sources/checkpoints/corpora are read as bytes.
SHA256SUMS.txt is a detached receipt, avoiding circular/self hashes.
"""
from __future__ import annotations

import ast
import collections
import hashlib
import importlib.metadata
import itertools
import json
import platform
import random
import re
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(r"C:\DaveLM-CADAVER")
OUT = ROOT / "english_context_characterization_v1_seed8380"
P0 = ROOT / "language_pilot_0_tinystories_seed8380"
P1 = ROOT / "language_pilot_1_early_block_protection_seed8380"
TOK = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
NAMES = ["Ben", "Sam", "Tim", "Tom"]
ANIMALS = ["bear", "cat", "fish"]
OBJECTS = ["bag", "ball", "book", "car", "hat", "toy"]
PLACES = ["bed", "cave", "room", "shop", "yard"]
PAIRS = {
    "names": list(itertools.combinations(NAMES, 2)),
    "animals": list(itertools.combinations(ANIMALS, 2)),
    "objects": [("bag", "book"), ("bag", "hat"), ("ball", "car"),
                ("ball", "toy"), ("book", "hat"), ("car", "toy")],
    "places": [("bed", "room"), ("cave", "shop"), ("cave", "yard"), ("shop", "yard")],
}
TASKS = {"naming": ("animals", "names", 8, 18),
         "possession": ("names", "objects", 16, 36),
         "location": ("names", "places", 16, 24)}
MODES = {"naming": ["core"], "possession": ["core", "frame_holdout", "qa"],
         "location": ["core", "frame_holdout", "qa"]}
EXPECTED_FREQ = {
    "Ben": (4168, 546, 344, 46), "Sam": (2136, 313, 189, 29),
    "Tim": (2453, 360, 288, 40), "Tom": (3991, 536, 531, 65),
    "bear": (1278, 353, 76, 29), "cat": (1121, 333, 125, 35),
    "fish": (680, 202, 58, 23), "bag": (307, 170, 30, 19),
    "ball": (1376, 388, 142, 38), "book": (373, 140, 42, 16),
    "car": (1461, 450, 166, 51), "hat": (380, 195, 51, 18),
    "toy": (1700, 777, 187, 80), "bed": (683, 408, 50, 31),
    "cave": (183, 64, 13, 5), "room": (1306, 801, 113, 72),
    "shop": (149, 77, 31, 13), "yard": (225, 137, 20, 17),
}
EXPECTED_SPACED = {
    "Ben": [532, 274], "Sam": [527, 339], "Tim": [525, 337], "Tom": [525, 295],
    "bear": [319, 285], "cat": [268, 265], "fish": [291, 795],
    "bag": [281, 408], "ball": [281, 452], "book": [281, 485],
    "car": [268, 285], "hat": [356, 265], "toy": [304, 93],
    "bed": [281, 278], "cave": [511, 403], "room": [942, 295],
    "shop": [373, 1001], "yard": [715, 609],
}
CHECKPOINTS = {
    "graduate": (ROOT / "treatment13_orthogonal_shared_unbounded_seed8380/checkpoints/orthogonal_shared_unbounded/seed_8380/latest.pt", "fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430"),
    "pilot0": (P0 / "pilot_run/checkpoints/seed_8380/latest.pt", "769bd01efd28e7888064c0d5fe1dd0d85a4344e2039aef04bcbf7d9e906f47f5"),
    "pilot1": (P1 / "pilot_run/checkpoints/seed_8380/latest.pt", "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb"),
}
CORE = ["PROTOCOL.md", "MANIFEST.json", "LEXICON.json", "FAMILIES.json",
        "ITEMS.jsonl", "PRIORS.jsonl", "PREFLIGHT.json"]
MECHANICAL_CORRECTIONS = []


def require(test, message):
    if not test:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path, expected=None):
    path = Path(path)
    actual = sha(path)
    if expected is not None:
        require(actual == expected, f"PROVENANCE FAILURE: {path}: {actual} != {expected}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size,
            "expected_sha256": expected, "expected_hash_matched": expected is None or actual == expected}


def write_json(name, value):
    (OUT / name).write_bytes((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))


def write_jsonl(name, rows):
    (OUT / name).write_bytes("".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode("utf-8"))


def norm(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def query(task, mode, entities, values, q):
    if task == "naming":
        return f"The name of the {entities[q]} is"
    if task == "possession":
        return {"core": f"The person with the {values[q]} is",
                "frame_holdout": f"The {values[q]} is with",
                "qa": f"Who has the {values[q]}?"}[mode]
    return {"core": f"{entities[q]} is in the",
            "frame_holdout": f"{entities[q]} can be found in the",
            "qa": f"Where is {entities[q]}?"}[mode]


def facts(task, entities, values, a):
    if task == "naming":
        return [f"The {entities[i]} is called {values[i ^ a]}." for i in range(2)]
    if task == "possession":
        return [f"{entities[i]} has the {values[i ^ a]}." for i in range(2)]
    return [f"{entities[i]} is in the {values[i ^ a]}." for i in range(2)]


def arrays(obj):
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("full_document_token_ids", "token_ids", "document_token_ids") and isinstance(v, list) and v and all(isinstance(x, int) for x in v):
                found.append(tuple(v))
            else:
                found.extend(arrays(v))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(arrays(v))
    return found


def score_source_audit():
    path = OUT / "implementation/english_scoring.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    call_names = [ast.unparse(n.func) for n in calls]
    require("model.base_model" in call_names, "Missing isolated base-model forward")
    require("model" not in call_names, "English scorer invokes wrapper")
    forbidden = ("backward", "optimizer", "optim.", "torch.load", "localizer", "retrieval", "model.wq", "model.wk", "model.wv", "model.wo")
    for name in call_names:
        require(not any(term in name.lower() for term in forbidden), f"Forbidden scoring call: {name}")
    require(all(isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)) or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)) for n in tree.body), "Unexpected scoring-module top-level executable statement")
    require(not any(isinstance(n, ast.Call) for top in tree.body if not isinstance(top, ast.FunctionDef) for n in ast.walk(top)), "Import-time scoring-module execution")
    return {"status": "PASS_STATIC_SOURCE_AUDIT", "forward_expression": "model.base_model(input_ids)",
            "wrapper_forward_calls": 0, "optimizer_or_backward_or_load_calls": 0,
            "top_level_execution_calls": 0, "source": file_record(path),
            "base_forward_source": file_record(Path(r"C:\DaveLM-v0.9\v0_7\model.py")),
            "wrapper_source": file_record(ROOT / "treatment13_model.py"),
            "scope": "AST and source inspection; scorer not imported or executed; no dynamic checkpoint validation claimed",
            "base_forward_review": "v0_7.model.DaveLM.forward lines105-114 returns language_head(final_norm(blocks(embeddings))); T13 final_norm capture hook returns None and does not modify output."}


def main():
    require(sys.version_info[:2] == (3, 12), "Sampling requires Python3.12")
    require(not (OUT / "SHA256SUMS.txt").exists(), "Detached freeze receipt exists; refusing to overwrite frozen package")
    require((OUT / "PROTOCOL.md").is_file(), "Approved PROTOCOL.md required first")
    require((OUT / "implementation/english_scoring.py").is_file(), "Scoring source required for static path audit")
    p0m = json.loads((P0 / "CORPUS_MANIFEST.json").read_text(encoding="utf-8"))
    p1m = json.loads((P1 / "PILOT1_MATERIAL_MANIFEST.json").read_text(encoding="utf-8"))
    expected_train = "450f78bdb437bde98f9a1ff48364b835df517b7d8e24a4f52256c114fa34c41c"
    expected_dev = "deff4fc7ed18e6e1f0b6f32faccc80f1eb44d0e58f3f38d512ffdfab798d6ac4"
    inputs = {"tokenizer": file_record(TOK, "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"),
              "source_corpus": file_record(P0 / "TinyStories-valid.txt", p0m["source"]["source_sha256"]),
              "train": file_record(P0 / "language_train.jsonl", expected_train),
              "dev": file_record(P0 / "language_dev.jsonl", expected_dev),
              "pilot1_train_copy": file_record(P1 / "language_train.jsonl", expected_train),
              "pilot1_dev_copy": file_record(P1 / "language_dev.jsonl", expected_dev),
              "pilot0_manifest": file_record(P0 / "CORPUS_MANIFEST.json"),
              "pilot1_manifest": file_record(P1 / "PILOT1_MATERIAL_MANIFEST.json")}
    checkpoints = {k: file_record(p, h) for k, (p, h) in CHECKPOINTS.items()}
    print("Input/checkpoint byte hashes verified; no checkpoint deserialization.", flush=True)
    tokobj = json.loads(TOK.read_text(encoding="utf-8"))
    require(tokobj["model"]["byte_fallback"] is False, "Tokenizer implementation changed")
    require(tokobj["pre_tokenizer"]["type"] == "ByteLevel", "Not the specified ByteLevel tokenizer")
    tok = Tokenizer.from_file(str(TOK))
    require(tok.get_vocab_size() == 1024 and tok.token_to_id("<bos>") == 2 and tok.token_to_id("<eos>") == 3, "Tokenizer contract mismatch")
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    corpora = {s: [json.loads(x) for x in (P0 / f"language_{s}.jsonl").read_text(encoding="utf-8").splitlines()] for s in ("train", "dev")}
    texts = {s: [x["text"] for x in rs] for s, rs in corpora.items()}
    require(len(texts["train"]) == 9000 and len(texts["dev"]) == 1000, "Corpus counts changed")
    raw = (P0 / "TinyStories-valid.txt").read_text(encoding="utf-8")
    stories = [s.strip().replace("\r\n", "\n").replace("\r", "\n") for s in raw.split("<|endoftext|>") if s.strip()]
    uniq = {hashlib.sha256(s.encode("utf-8")).hexdigest(): s for s in stories}
    chosen_stories = [uniq[h] for h in sorted(uniq)[:10000]]
    require(texts["train"] == chosen_stories[:9000] and texts["dev"] == chosen_stories[9000:], "Source corpus partition not reproducible")
    require(not set(texts["train"]) & set(texts["dev"]), "Exact corpus train/dev overlap")
    for rs in corpora.values():
        for rec in rs:
            require(enc(rec["text"]) == rec["token_ids"], "Stored corpus token IDs inconsistent")
    lexicon = {}
    for word in sorted(EXPECTED_FREQ):
        regex = re.compile(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])")
        freq = {}
        for split, docs in texts.items():
            counts = [len(regex.findall(d)) for d in docs]
            freq[split] = {"occurrences": sum(counts), "stories": sum(c > 0 for c in counts)}
        require(tuple(freq[s][key] for s in ("train", "dev") for key in ("occurrences", "stories")) == EXPECTED_FREQ[word], f"Frequency discrepancy: {word}")
        require(freq["train"]["occurrences"] >= 100 and freq["train"]["stories"] >= 50, f"Lexical eligibility: {word}")
        variants = {"bare": word, "leading_space": " " + word, "completed": " " + word + "."}
        ids = {k: enc(v) for k, v in variants.items()}
        require(ids["leading_space"] == EXPECTED_SPACED[word] and ids["completed"] == EXPECTED_SPACED[word] + [18], f"Token IDs changed: {word}")
        for key, value in variants.items():
            require(tok.decode(ids[key], skip_special_tokens=False) == value, f"Round trip: {word}/{key}")
            require(not set(ids[key]) & set(range(5)), f"Special token in word: {word}")
        lexicon[word] = {"strings": variants, "token_ids": ids, "tokens": {k: tok.encode(v, add_special_tokens=False).tokens for k, v in variants.items()},
                         "frequencies": freq, "answer_candidate": word in NAMES + PLACES, "eligible": True}
    for cat, words in (("names", NAMES), ("animals", ANIMALS), ("objects", OBJECTS), ("places", PLACES)):
        eligible = [pair for pair in itertools.combinations(words, 2) if max(lexicon[w]["frequencies"]["train"]["occurrences"] for w in pair) <= 2 * min(lexicon[w]["frequencies"]["train"]["occurrences"] for w in pair)]
        require(eligible == PAIRS[cat], f"Eligible pair universe discrepancy: {cat}")
    print("Corpus partition, 10000 stored token arrays and lexical rules verified.", flush=True)
    family_data = {}; family_rows = []; items = []; priors = []
    for task, (ek, vk, n, expected_n) in TASKS.items():
        universe = sorted(itertools.product(PAIRS[ek], PAIRS[vk]))
        require(len(universe) == expected_n, f"Universe size: {task}")
        selected = sorted(random.Random(8380).sample(universe, n))
        selected_rows = []
        for fi, (ee, vv) in enumerate(selected):
            fid = f"{task}:f{fi:03d}"
            candidates = list(ee if task == "possession" else vv)
            completions = [" " + c + "." for c in candidates]
            fam = {"family_id": fid, "task": task, "entities": list(ee), "values": list(vv), "candidates": candidates,
                   "eligible_universe_index": universe.index((ee, vv)), "frames": MODES[task]}
            selected_rows.append(fam); family_rows.append(fam)
            for a, q, o in itertools.product(range(2), repeat=3):
                fs = facts(task, ee, vv, a); ordered = [fs[o], fs[1-o]]
                correct = q ^ a
                target_entity = correct if task == "possession" else q
                mention_order = [candidates.index(ee[i] if task == "possession" else vv[i ^ a]) for i in (o, 1-o)]
                correct_rank = mention_order.index(correct)
                for mode in MODES[task]:
                    query_text = query(task, mode, ee, vv, q)
                    prompt = ordered[0] + " " + ordered[1] + "\n" + query_text
                    item_id = f"{fid}:{mode}:a{a}q{q}o{o}"
                    items.append({"item_id": item_id, "family_id": fid, "task": task, "frame": mode,
                                  "role": "primary" if task in ("possession", "location") and mode == "core" else "diagnostic",
                                  "assignment": a, "query_index": q, "fact_order": o,
                                  "entities": list(ee), "values": list(vv), "mapping": {ee[i]: vv[i ^ a] for i in range(2)},
                                  "ordered_facts": ordered, "query": query_text, "prompt": prompt,
                                  "prompt_token_ids": enc(prompt), "bos_token_id": 2,
                                  "candidates": candidates, "candidate_completions": completions,
                                  "candidate_token_ids": [enc(c) for c in completions],
                                  "correct_index": correct, "correct_candidate": candidates[correct],
                                  "correct_candidate_mention_rank": correct_rank,
                                  "candidate_mention_order": mention_order,
                                  "queried_fact_rank": (o, 1-o).index(target_entity),
                                  "reversal_pair_id": f"{fid}:{mode}:q{q}o{o}",
                                  "prior_id": f"{fid}:{mode}:q{q}:prior"})
            for mode in MODES[task]:
                for q in range(2):
                    qp = query(task, mode, ee, vv, q)
                    priors.append({"prior_id": f"{fid}:{mode}:q{q}:prior", "family_id": fid, "task": task,
                                   "frame": mode, "query_index": q, "prompt": qp, "prompt_token_ids": enc(qp),
                                   "bos_token_id": 2, "candidates": candidates, "candidate_completions": completions,
                                   "candidate_token_ids": [enc(c) for c in completions], "correct_index": None,
                                   "role": "query_only_prior_diagnostic", "linked_item_ids": [x["item_id"] for x in items if x["prior_id"] == f"{fid}:{mode}:q{q}:prior"]})
        family_data[task] = {"eligible_universe_size": len(universe), "sample_size": n,
                             "eligible_universe": [{"entities": list(e), "values": list(v)} for e, v in universe],
                             "selected_families": selected_rows}
    require(len(items) == 832 and len(priors) == 208, "Wrong materialized counts")
    require(len({x["item_id"] for x in items}) == 832 and len({x["prompt"] for x in items}) == 832, "Context item duplicates")
    require(len({x["prior_id"] for x in priors}) == 208, "Prior IDs duplicate")
    boundary_checks = 0; lengths = []; roundtrips = 0
    for rec in items + priors:
        p = rec["prompt"]; pi = rec["prompt_token_ids"]
        require(not p[-1].isspace() and enc(p) == pi, "Prompt serialization")
        require(tok.decode(pi, skip_special_tokens=False) == p, "Prompt round trip")
        roundtrips += 1
        for c, ci in zip(rec["candidate_completions"], rec["candidate_token_ids"]):
            require(len(ci) == 3 and ci[-1] == 18, "Unequal candidate completion length")
            require(enc(p+c) == pi+ci and enc(c) == ci, "Prompt/candidate boundary failure")
            require(tok.decode(ci, skip_special_tokens=False) == c and tok.decode(pi+ci, skip_special_tokens=False) == p+c, "Completed round trip")
            require(not set(pi+ci) & set(range(5)), "Unexpected encoded special token")
            require(1+len(pi)+len(ci) <= 256, "Context overflow")
            lengths.append(1+len(pi)+len(ci)); boundary_checks += 1; roundtrips += 2
    balances = []
    by_group = collections.defaultdict(list)
    for item in items:
        by_group[(item["family_id"], item["frame"])].append(item)
    for (fid, mode), group in by_group.items():
        require(len(group) == 8 and {(x["assignment"], x["query_index"], x["fact_order"]) for x in group} == set(itertools.product(range(2), repeat=3)), "Family incompleteness")
        counts = {key: dict(sorted(collections.Counter(x[key] for x in group).items())) for key in
                  ("assignment", "query_index", "fact_order", "correct_index", "correct_candidate_mention_rank", "queried_fact_rank")}
        require(all(v == {0: 4, 1: 4} for v in counts.values()), "Family balance failure")
        mentions = {str(c): dict(collections.Counter(x["candidate_mention_order"].index(c) for x in group)) for c in (0,1)}
        require(all(v == {0:4, 1:4} for v in mentions.values()), "Candidate first/second mention balance")
        role_counts = collections.Counter((e, v) for x in group for e, v in x["mapping"].items())
        require(len(role_counts) == 4 and set(role_counts.values()) == {4}, "Entity/value association-role balance")
        rp = collections.defaultdict(list)
        for x in group:
            rp[x["reversal_pair_id"]].append(x)
            require(x["correct_index"] == (x["query_index"] ^ x["assignment"]), "Assignment equation answer mismatch")
            if x["task"] == "possession":
                answer = next(e for e, v in x["mapping"].items() if v == x["values"][x["query_index"]])
            else:
                answer = x["mapping"][x["entities"][x["query_index"]]]
            require(answer == x["correct_candidate"], "Independent map lookup answer mismatch")
            for candidate in x["candidates"]:
                require(sum(len(re.findall(r"(?<![A-Za-z])"+re.escape(candidate)+r"(?![A-Za-z])", f)) for f in x["ordered_facts"]) == 1, "Candidate occurrence imbalance")
            found_order = sorted(((' '.join(x['ordered_facts'])).index(c), i) for i, c in enumerate(x["candidates"]))
            require([i for _, i in found_order] == x["candidate_mention_order"], "Mention metadata disagrees with text")
        require(len(rp) == 4, "Wrong reversal pair count")
        for pair in rp.values():
            require(len(pair) == 2 and {x["assignment"] for x in pair} == {0,1} and {x["correct_index"] for x in pair} == {0,1}, "Invalid matched reversal")
            require(pair[0]["query"] == pair[1]["query"] and pair[0]["candidates"] == pair[1]["candidates"] and pair[0]["fact_order"] == pair[1]["fact_order"], "Unmatched reversal")
        balances.append({"family_id": fid, "frame": mode, "items": 8, "reversal_pairs": 4,
                         "binary_factor_counts": counts, "candidate_mention_rank_counts": mentions,
                         "candidate_correct_incorrect_counts": {c:{"correct":4,"incorrect":4} for c in group[0]["candidates"]},
                         "entity_value_role_counts": [{"entity":e,"value":v,"count":c} for (e,v),c in sorted(role_counts.items())]})
    for prior in priors:
        require(len(prior["linked_item_ids"]) == 4, "Prior linkage count")
    print("832 items,208 priors, all family balances/answer keys/boundaries verified.", flush=True)
    normalized = {s: [norm(d) for d in ds] for s, ds in texts.items()}
    overlap_levels = {
        "atomic_fact": {f for x in items for f in x["ordered_facts"]},
        "two_fact_context": {" ".join(x["ordered_facts"]) for x in items},
        "whole_prompt": {x["prompt"] for x in items},
        "whole_prompt_plus_either_candidate": {x["prompt"]+c for x in items for c in x["candidate_completions"]}}
    overlap = {}
    for level, strings in overlap_levels.items():
        queries = sorted({norm(s) for s in strings}); result = {}
        for split, docs in normalized.items():
            hits = [{"normalized_string":s,"story_indices":[i for i,d in enumerate(docs) if s in d]} for s in queries if any(s in d for d in docs)]
            require(not hits, f"MATERIAL INTEGRITY: newly detected {level} overlap in {split}")
            result[split] = {"matched_test_strings":len(hits),"matches":hits}
        overlap[level] = {"distinct_strings":len(queries),"results":result}
    patterns = {
        "naming_fact":r"the (bear|cat|fish) is called (ben|sam|tim|tom)\.",
        "possession_fact":r"(ben|sam|tim|tom) has the (bag|ball|book|car|hat|toy)\.",
        "location_fact":r"(ben|sam|tim|tom) is in the (bed|cave|room|shop|yard)\.",
        "naming_query":r"the name of the (bear|cat|fish) is",
        "possession_core_query":r"the person with the (bag|ball|book|car|hat|toy) is",
        "possession_holdout_query":r"the (bag|ball|book|car|hat|toy) is with",
        "possession_qa":r"who has the (bag|ball|book|car|hat|toy)\?",
        "location_core_query":r"(ben|sam|tim|tom) is in the",
        "location_holdout_query":r"(ben|sam|tim|tom) can be found in the",
        "location_qa":r"where is (ben|sam|tim|tom)\?"}
    template_counts = {name:{"regex":r"(?<![a-z])"+p+r"(?![a-z])", "counts":{s:sum(len(re.findall(r"(?<![a-z])"+p+r"(?![a-z])",d)) for d in ds) for s,ds in normalized.items()}} for name,p in patterns.items()}
    phrase_counts = {p:{s:sum(len(re.findall(r"(?<![a-z])"+re.escape(p)+r"(?![a-z])",d)) for d in ds) for s,ds in normalized.items()} for p in ("is called","the name of","has the","the person with","is with","is in the","can be found in the","where is","who has")}
    binding_refs = {}
    dev_sets = []
    for key, folder, expected in (("pilot0_dev", P0, p0m["binding_dev"]["sha256"]), ("pilot1_dev", P1, p1m["binding_dev_sha256"])):
        path = folder / "binding_dev.json"; info = file_record(path, expected)
        obj = json.loads(path.read_text(encoding="utf-8")); qs = obj["quartets"]
        ds = [d for q in qs for d in q["docs"]]; ar = [tuple(d["full_document_token_ids"]) for d in ds]
        require(len(qs)==20 and len(ds)==80 and len(set(ar))==80, "Binding reference duplicate/count failure")
        require(all(len(q["docs"])==4 and {d["member"] for d in q["docs"]}=={"o1_k0","o1_k1","o2_k0","o2_k1"} for q in qs), "Binding quartet completeness")
        require(dict(collections.Counter(d["layout_combo"] for d in ds))=={"p56_b12":40,"p58_b13":40}, "Binding layouts changed")
        binding_refs[key] = {**info,"documents":80,"quartets":20,"layouts":{"p56_b12":40,"p58_b13":40},"quartet_ids":[q["quartet_id"] for q in qs],"status":"nonsacred_existing_development_reference"}
        dev_sets.append(set(ar))
    require(not dev_sets[0]&dev_sets[1], "Binding DEV references intersect")
    train_paths = [P0 / "binding_rehearsal.json", P1 / "binding_rehearsal.json",
        ROOT / "treatment10_strict_counterfactual_binding_seed8380/treatment10_training_quartet_pool.json",
        ROOT / "treatment11_answer_only_strict_counterfactual_binding_seed8380/treatment11_training_quartet_pool.json",
        ROOT / "treatment13_learned_mapping_row_localization_seed8380/treatment13_training_quartet_pool.json",
        ROOT / "treatment5_counterfactual_pairs_seed8382/treatment5_full_document_pair_pool.json"]
    expected_training = {str(P0 / "binding_rehearsal.json"):p0m["binding_rehearsal"]["sha256"],
                         str(P1 / "binding_rehearsal.json"):p1m["binding_rehearsal_sha256"],
                         str(train_paths[4]):p0m["prior_t13_training_pool_sha256"]}
    training_checks = []
    for path in train_paths:
        require("retention" not in path.name, "Forbidden sacred/retention path")
        ar = set(arrays(json.loads(path.read_text(encoding="utf-8"))))
        require(ar, f"No arrays found in prior training reference {path}")
        intersections = [len(dev & ar) for dev in dev_sets]
        require(intersections == [0,0], f"Binding DEV overlaps known training: {path}")
        training_checks.append({**file_record(path,expected_training.get(str(path))), "unique_arrays":len(ar),"pilot0_dev_overlap":intersections[0],"pilot1_dev_overlap":intersections[1]})
    require(p1m["prior_token_array_overlaps"] == {}, "Pilot1 recorded provenance failure")
    binding_provenance = {"references":binding_refs,"dev_intersection":0,"training_checks":training_checks,
                          "sacred_exam_opened":False,"sacred_behavior_evaluated":False,
                          "historical_manifest_sacred_disjointness_claim":"Recorded previously; not reopened or recomputed in this freeze",
                          "report_separately":True,"binding_evaluation_performed":False}
    source_audit = score_source_audit()
    counts = dict(sorted(collections.Counter(f"{x['task']}/{x['frame']}" for x in items).items()))
    prior_counts = dict(sorted(collections.Counter(f"{x['task']}/{x['frame']}" for x in priors).items()))
    write_json("LEXICON.json", {"schema_version":1,"count_rule":"case-sensitive (?<![A-Za-z])WORD(?![A-Za-z]); stories counted once", "eligibility":{"train_occurrences_min":100,"train_stories_min":50,"spaced_word_tokens":2,"within_pair_max_frequency_ratio":2,"dev_used_for_selection":False},"words":lexicon,"eligible_pairs":PAIRS})
    write_json("FAMILIES.json", {"schema_version":1,"seed":8380,"sampling":"Fresh Python3.12 random.Random(8380).sample(sorted Cartesian product,n) per task; sorted sample", "tasks":family_data})
    write_jsonl("ITEMS.jsonl",items); write_jsonl("PRIORS.jsonl",priors)
    implementation_files = sorted((OUT/"implementation").glob("*.py"))
    noncyclic_hashes = {name:file_record(OUT/name) for name in ("PROTOCOL.md","LEXICON.json","FAMILIES.json","ITEMS.jsonl","PRIORS.jsonl")}
    noncyclic_hashes.update({str(p.relative_to(OUT)).replace("\\","/"):file_record(p) for p in implementation_files})
    manifest = {"schema_version":1,"protocol_id":"english_context_characterization_v1_seed8380","status":"VALIDATED_PRE_EXECUTION",
        "seed":8380,"scientific_artifacts":CORE,"artifact_hash_records":noncyclic_hashes,
        "hash_contract":"Exact UTF-8/LF bytes without BOM; MANIFEST and PREFLIGHT final hashes are in detached SHA256SUMS.txt; no self/circular hashes",
        "source_inputs":inputs,"checkpoint_registry":checkpoints,"source_corpus_revision":p0m["source"],
        "common_binding_reference":binding_provenance,
        "sampling_universes":{"naming":18,"possession":36,"location":24},
        "item_counts":counts,"prior_counts":prior_counts,"total_contextual_items":832,"total_logical_priors":208,
        "runtime":{"construction_python":platform.python_version(),"construction_python_executable":sys.executable,
                   "tokenizers":importlib.metadata.version("tokenizers"),"torch_distribution_version_not_imported":importlib.metadata.version("torch"),
                   "english_model_path":"model.base_model(input_ids)","device":"same device for all three checkpoints; record at execution",
                   "model_dtype":"float32","log_softmax_and_accumulation_dtype":"float64","autocast":False,"tf32":False,
                   "eval_mode":True,"inference_mode":True,"unpadded_one_sequence_per_candidate":True,
                   "bos_id":2,"eos_appended":False,"doc_marker_appended":False,"generation":False,"optimizer":False,"backward":False},
        "scoring":{"candidate":"one ASCII space + word + period","candidate_tokens":3,"word_tokens":2,"period_id":18,
                   "sequence_log_likelihood":"sum_j log p(t_j | BOS,prompt,t_<j); natural logs; correctly shifted causal positions",
                   "margin":"correct log-likelihood minus incorrect log-likelihood","item_correct":"margin > 0; exact ties incorrect",
                   "reversal_correct":"both assignments correct at fixed family/query/order/frame","family_correct":"all eight assignment/query/order items correct",
                   "length_normalization":False,"prior_subtraction":False,"word_only_excludes_final_period":True},
        "reporting":{"headline":"raw complete-family counts/proportions, reversal profiles and likelihood-margin summaries",
                     "primary_endpoints":["possession/core complete-family proportion","possession/core mean within-family reversal success","location/core complete-family proportion","location/core mean within-family reversal success"],
                     "ci":"secondary 95% exact hypergeometric interval only over each enumerated eligible lexical-family universe; never uncertainty over general English capability",
                     "diagnostic_strata":["naming/core","possession/frame_holdout","location/frame_holdout","possession/qa","location/qa","query-only priors","common synthetic binding references"],
                     "no_primary_pooling_across_relations_or_qa":True,"pass_fail_capability_gates":False,"automatic_parent_selection":False},
        "static_path_isolation":source_audit,"mechanical_corrections":MECHANICAL_CORRECTIONS,
        "no_checkpoint_deserialization_or_inference_this_turn":True,"no_behavioral_scores":True}
    write_json("MANIFEST.json",manifest)
    checkpoint_hashes_after = {k:sha(p) for k,(p,_) in CHECKPOINTS.items()}
    require(all(checkpoint_hashes_after[k] == h for k,(_,h) in CHECKPOINTS.items()), "Checkpoint changed during preflight")
    preflight = {"schema_version":1,"status":"PASS_PRE_EXECUTION_MATERIAL_VALIDATION",
        "headline":"Material integrity only; zero checkpoint/model evaluation and zero behavioral scores",
        "source_hashes":inputs,"checkpoint_hashes_before":checkpoints,"checkpoint_hashes_after":checkpoint_hashes_after,
        "selected_family_identities":family_rows,"item_counts_by_task_frame":counts,"prior_counts_by_task_frame":prior_counts,
        "contextual_items":832,"logical_priors":208,"unique_contextual_prompts":len({x["prompt"] for x in items}),
        "unique_prior_prompts":len({x["prompt"] for x in priors}),
        "unique_prior_prompt_candidate_pairs":len({(x["prompt"],tuple(x["candidate_completions"])) for x in priors}),
        "family_frame_groups":len(by_group),"items_per_family_frame":8,"reversal_pairs_per_family_frame":4,
        "total_reversal_pairs":len(by_group)*4,"per_family_frame_balancing":balances,
        "answer_key_consistency":{"xor_equation_items_checked":832,"independent_mapping_lookup_items_checked":832,"failures":0},
        "tokenizer":{"byte_fallback":False,"pre_tokenizer":"ByteLevel","vocabulary_size":1024,"bos_id":2,"eos_id":3,
                     "candidate_completed_token_lengths":[3],"candidate_word_token_lengths":[2],"boundary_checks":boundary_checks,
                     "encode_decode_checks":roundtrips,"boundary_failures":0,"roundtrip_failures":0,
                     "max_completed_input_including_bos":max(lengths),"min_completed_input_including_bos":min(lengths),
                     "max_contextual_completed_input_including_bos":max(1+len(x["prompt_token_ids"])+3 for x in items)},
        "corpus":{"source_partition_reproduced":True,"stored_story_token_arrays_verified":10000,"train_dev_exact_story_overlap":0,
                  "normalization":"re.sub(r'\\s+', ' ', text).strip().casefold()","overlap":overlap,
                  "whitelist_template_counts":template_counts,"ordinary_phrase_counts":phrase_counts,
                  "scope":"Exact normalized strings and explicit whitelist regexes; no claim of semantic or near-duplicate absence"},
        "common_binding_reference":binding_provenance,"ordinary_path_isolation":source_audit,
        "checkpoint_deserializations":0,"model_instantiations":0,"model_forward_calls":0,"optimizer_creations":0,
        "backward_passes":0,"behavioral_scores_produced":0,"sacred_exam_opened":False,
        "mechanical_corrections":MECHANICAL_CORRECTIONS,
        "validated_artifact_hashes_excluding_this_file":{**noncyclic_hashes,"MANIFEST.json":file_record(OUT/"MANIFEST.json")}}
    write_json("PREFLIGHT.json",preflight)
    for name in CORE + [str(p.relative_to(OUT)) for p in implementation_files]:
        data=(OUT/name).read_bytes(); data.decode("utf-8")
        require(not data.startswith(b"\xef\xbb\xbf") and b"\r" not in data and data.endswith(b"\n"),f"UTF8/LF byte failure: {name}")
    # The separate seal step follows independent review, hashing these actual bytes.
    print(json.dumps({"status":"PASS_PENDING_DETACHED_SEAL","counts":counts,"items":832,"priors":208,
                      "family_frame_groups":len(by_group),"max_completed_tokens":max(lengths),
                      "behavioral_scores":0,"files":{name:sha(OUT/name) for name in CORE}},indent=2))


if __name__ == "__main__":
    main()
