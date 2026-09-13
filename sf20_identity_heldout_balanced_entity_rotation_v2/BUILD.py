"""Build prospective SF20 v2 without loading a model or creating an optimizer."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tokenizers import Tokenizer


H = Path(__file__).resolve().parent
ROOT = H.parent
SF13 = ROOT / "sf13_broad_coverage_kl_retention_v1"
SF20_V1_STOP = ROOT / "sf20_identity_heldout_balanced_entity_rotation_v1"
LANGUAGE_TRAIN = ROOT / "language_pilot_1_early_block_protection_seed8380" / "language_train.jsonl"
TOKENIZER_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
TOKENIZER_SHA256 = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
LANGUAGE_TRAIN_SHA256 = "450f78bdb437bde98f9a1ff48364b835df517b7d8e24a4f52256c114fa34c41c"

ORIGINAL_NAMES = ["Alex", "Owen", "Mia", "Nora"]
COMMON_GIVEN_NAMES = (
    "Adam Alice Amy Ava Ben Beth Billy Bob Brian Carl Cathy Chad Charlie Chloe Clare Colin "
    "Daniel David Ella Emma Emily Ethan Eva Faye Fiona Frank Fred George Grace Harry Henry "
    "Jack Jacob James Jane John Julia Kate Leo Liam Lily Lucy Luke Mandy Mark Mary Max Megan "
    "Mia Mike Mila Molly Nathan Neil Nina Nora Oliver Owen Paul Peter Ralph Ravi Remy Rick "
    "Riley Rita Rob Rose Ruby Ruth Ryan Sam Sara Sarah Sean Sophia Sophie Steve Susan Tara Ted "
    "Tilly Tim Tina Tom Tony Wendy William Willow Zoe"
).split()

# Frozen nonsacred lexicons only. Item files and outcomes are not selection inputs.
FORBIDDEN_LEXICONS = [
    ROOT / "english_context_characterization_v1_seed8380" / "LEXICON.json",
    ROOT / "post_p7_language_report_card_v1_seed8380" / "LEXICON.json",
    ROOT / "human_test_readiness_v2_seed87010" / "LEXICON.json",
]

# Deterministic pair geometry over the rank-ordered 16 selected names.
SURFACE_PAIR_INDEXES = [(0, 1), (2, 3), (4, 5), (6, 8), (7, 9), (10, 11), (12, 13), (14, 15)]
ORDER_PAIR_INDEXES = [(0, 3), (1, 4), (2, 6), (5, 8), (7, 10), (9, 13), (11, 14), (12, 15)]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def verify_sf13() -> None:
    assert sha(SF13 / "FREEZE_RECEIPT.json") == (SF13 / "FREEZE_RECEIPT.sha256").read_text().split()[0]
    receipt = read(SF13 / "FREEZE_RECEIPT.json")
    assert receipt["status"] == "SF13_PROSPECTIVE_PREFLIGHT_PASS"
    assert sha(SF13 / "SHA256SUMS.txt") == receipt["manifest_sha256"]
    for line in (SF13 / "SHA256SUMS.txt").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert sha(SF13 / relative) == expected, relative


def forbidden_identities() -> tuple[set[str], list[dict]]:
    forbidden = set(ORIGINAL_NAMES)
    provenance = []
    for path in FORBIDDEN_LEXICONS:
        data = read(path)
        found = set()
        for key in ("names", "subjects"):
            found.update(data.get(key, []))
        for pair in data.get("fact_name_pairs", []):
            found.update(pair)
        if data.get("distractor_entity"):
            found.add(data["distractor_entity"])
        forbidden.update(found)
        provenance.append({"path": str(path), "sha256": sha(path), "identities": sorted(found)})
    return forbidden, provenance


def select_names(tokenizer: Tokenizer, forbidden: set[str]) -> tuple[list[str], dict]:
    counts = Counter()
    story_ids = defaultdict(set)
    allowed = set(COMMON_GIVEN_NAMES)
    for line in LANGUAGE_TRAIN.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        words = re.findall(r"(?<![A-Za-z])([A-Z][a-z]{1,14})(?![A-Za-z])", row["text"])
        for name in words:
            if name in allowed:
                counts[name] += 1
                story_ids[name].add(row["id"])

    original_first_ids = {tokenizer.encode(" " + name + ".").ids[0] for name in ORIGINAL_NAMES}
    census = []
    for name in COMMON_GIVEN_NAMES:
        start = tokenizer.encode(name).ids
        leading = tokenizer.encode(" " + name).ids
        completed = tokenizer.encode(" " + name + ".").ids
        geometry = [len(start), len(leading), len(completed)]
        exact_roundtrip = (tokenizer.decode(start) == name and tokenizer.decode(leading) == " " + name
                           and tokenizer.decode(completed) == " " + name + ".")
        eligible = (geometry == [3, 3, 4] and exact_roundtrip and counts[name] >= 5
                    and name not in forbidden and completed[0] not in original_first_ids)
        census.append({
            "name": name,
            "train_occurrences": counts[name],
            "train_stories": len(story_ids[name]),
            "sentence_start_ids": start,
            "leading_space_ids": leading,
            "candidate_with_period_ids": completed,
            "geometry": geometry,
            "roundtrip": exact_roundtrip,
            "forbidden_identity": name in forbidden,
            "original_answer_first_token_collision": bool(completed and completed[0] in original_first_ids),
            "eligible": eligible,
        })
    eligible_rows = [row for row in census if row["eligible"]]
    eligible_rows.sort(key=lambda row: (-row["train_stories"], -row["train_occurrences"], row["name"]))
    assert len(eligible_rows) >= 16
    selected = [row["name"] for row in eligible_rows[:16]]
    return selected, {
        "selection_rule": (
            "From a fixed conservative common-given-name list, require exact 3/3/4 roundtrip geometry, "
            ">=5 case-sensitive TinyStories TRAIN occurrences, no frozen-lexicon identity collision, and "
            "no collision with the four original answer first-token IDs; rank by descending TRAIN story "
            "count, then occurrences, then name. Select first 16."
        ),
        "source": str(LANGUAGE_TRAIN),
        "source_sha256": sha(LANGUAGE_TRAIN),
        "original_answer_first_token_ids": sorted(original_first_ids),
        "eligible_count": len(eligible_rows),
        "selected": selected,
        "eligible_ranked": eligible_rows,
        "full_reference_census": census,
    }


def identity_occurrences(text: str, names: list[str], tokenizer: Tokenizer) -> list[dict]:
    encoding = tokenizer.encode(text)
    hits = []
    for slot, name in enumerate(names):
        for match in re.finditer(r"(?<![A-Za-z])" + re.escape(name) + r"(?![A-Za-z])", text):
            begin, end = match.span()
            span_begin = begin if begin == 0 else begin - 1
            assert begin == 0 or text[begin - 1] == " "
            indexes = [i for i, (a, b) in enumerate(encoding.offsets) if a < end and b > span_begin]
            expected = tokenizer.encode(("" if begin == 0 else " ") + name).ids
            actual = [encoding.ids[i] for i in indexes]
            assert actual == expected and len(actual) == 3
            hits.append({"slot": slot, "name": name, "char_span": [span_begin, end],
                         "token_indexes": indexes, "token_ids": actual})
    hits.sort(key=lambda row: row["char_span"])
    assert all(a["char_span"][1] <= b["char_span"][0] for a, b in zip(hits, hits[1:]))
    return hits


def skeleton(text: str, names: list[str], tokenizer: Tokenizer) -> list:
    encoding = tokenizer.encode(text)
    hits = identity_occurrences(text, names, tokenizer)
    by_start = {row["token_indexes"][0]: row for row in hits}
    skip = {i for row in hits for i in row["token_indexes"][1:]}
    result = []
    for index, token_id in enumerate(encoding.ids):
        if index in by_start:
            result.append(f"<NAME{by_start[index]['slot']}>")
        elif index not in skip:
            result.append(token_id)
    return result


def replace_names(text: str, old_pair: list[str], new_pair: list[str]) -> str:
    result = text
    placeholders = ["__SF20_NAME_SLOT_0__", "__SF20_NAME_SLOT_1__"]
    for old, placeholder in zip(old_pair, placeholders):
        result = re.sub(r"(?<![A-Za-z])" + re.escape(old) + r"(?![A-Za-z])", placeholder, result)
    for placeholder, new in zip(placeholders, new_pair):
        result = result.replace(placeholder, new)
    return result


def main() -> None:
    assert not (H / "PROTOCOL.json").exists(), "Do not overwrite a built SF20 v2"
    verify_sf13()
    assert sha(TOKENIZER_PATH) == TOKENIZER_SHA256
    assert sha(LANGUAGE_TRAIN) == LANGUAGE_TRAIN_SHA256
    assert (SF20_V1_STOP / "PREFLIGHT_STOP.json").exists()
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    forbidden, forbidden_sources = forbidden_identities()
    selected, census = select_names(tokenizer, forbidden)

    source_items = read(SF13 / "TRAIN.json")
    source_new = [row for row in source_items if row["id"].startswith("SF9:")]
    preservation = [row for row in source_items if row["id"].startswith("TRAIN:")]
    assert len(source_new) == 32 and len(preservation) == 16
    surface_keys = list(dict.fromkeys(":".join(row["id"].split(":")[:-1]) for row in source_new
                                      if row["id"].startswith("SF9:surface:")))
    order_keys = list(dict.fromkeys(":".join(row["id"].split(":")[:-1]) for row in source_new
                                    if row["id"].startswith("SF9:competing:")))
    assert len(surface_keys) == len(order_keys) == 8
    surface_pairs = [[selected[a], selected[b]] for a, b in SURFACE_PAIR_INDEXES]
    order_pairs = [[selected[a], selected[b]] for a, b in ORDER_PAIR_INDEXES]
    pair_map = dict(zip(surface_keys, surface_pairs)) | dict(zip(order_keys, order_pairs))
    first_ids = {name: tokenizer.encode(" " + name + ".").ids[0] for name in selected}
    assert all(first_ids[a] != first_ids[b] for a, b in pair_map.values())

    new_items = []
    source_to_new = {}
    tokenization_checks = []
    assignment_rows = []
    for source in source_new:
        pair_key = ":".join(source["id"].split(":")[:-1])
        new_pair = pair_map[pair_key]
        old_pair = [candidate.strip()[0:-1] for candidate in source["candidates"]]
        assert old_pair in [["Alex", "Owen"], ["Mia", "Nora"]]
        mapping = dict(zip(old_pair, new_pair))
        prompt = replace_names(source["prompt"], old_pair, new_pair)
        old_skeleton = skeleton(source["prompt"], old_pair, tokenizer)
        new_skeleton = skeleton(prompt, new_pair, tokenizer)
        assert old_skeleton == new_skeleton
        assert len(tokenizer.encode(prompt).ids) == len(source["prompt_token_ids"])
        assert tokenizer.decode(tokenizer.encode(prompt).ids) == prompt
        candidates = [" " + name + "." for name in new_pair]
        candidate_ids = [tokenizer.encode(candidate).ids for candidate in candidates]
        assert all(len(ids) == 4 and ids[-1] == tokenizer.encode(".").ids[0] for ids in candidate_ids)
        actor = mapping[source["actor"]]
        new_id = "SF20:" + ":".join(source["id"].split(":")[1:-1]) + ":" + actor
        row = dict(source)
        row.update({
            "id": new_id,
            "source_id": source["id"],
            "family_id": "SF20:" + ":".join(source["family_id"].split(":")[1:]),
            "pair_id": "SF20:" + ":".join(source["pair_id"].split(":")[1:]),
            "prompt": prompt,
            "prompt_token_ids": tokenizer.encode(prompt).ids,
            "candidates": candidates,
            "candidate_token_ids": candidate_ids,
            "actor": actor,
        })
        assert row["correct_index"] == source["correct_index"]
        new_items.append(row)
        source_to_new[source["id"]] = new_id
        correct_name = new_pair[row["correct_index"]]
        distractor_name = new_pair[1 - row["correct_index"]]
        assignment_rows.append({"source_id": source["id"], "item_id": new_id, "pair_key": pair_key,
                                "stratum": "Surface" if ":surface:" in source["id"] else "Order",
                                "subgroup": source["subgroup"], "candidate_names": new_pair,
                                "correct": correct_name, "distractor": distractor_name,
                                "correct_index": row["correct_index"]})
        tokenization_checks.append({
            "source_id": source["id"],
            "item_id": new_id,
            "source_prompt_tokens": len(source["prompt_token_ids"]),
            "new_prompt_tokens": len(row["prompt_token_ids"]),
            "source_identity_occurrences": identity_occurrences(source["prompt"], old_pair, tokenizer),
            "new_identity_occurrences": identity_occurrences(prompt, new_pair, tokenizer),
            "surrounding_tokenization_unchanged": old_skeleton == new_skeleton,
            "candidate_token_ids": candidate_ids,
            "answer_first_token_ids": [ids[0] for ids in candidate_ids],
            "first_answer_tokens_distinct": candidate_ids[0][0] != candidate_ids[1][0],
            "prompt_length_unchanged": len(source["prompt_token_ids"]) == len(row["prompt_token_ids"]),
        })

    items = preservation + new_items
    assert len(items) == 48 and len({row["id"] for row in items}) == 48
    assert len({norm(row["prompt"]) for row in items}) == 48

    schedule = []
    for source_update in read(SF13 / "SCHEDULE.json"):
        update = dict(source_update)
        if update["kind"] == "english":
            update["ids"] = [source_to_new.get(item_id, item_id) for item_id in source_update["ids"]]
        schedule.append(update)
    exposure = Counter(item_id for update in schedule if update["kind"] == "english" for item_id in update["ids"])
    assert len(schedule) == 200 and Counter(row["kind"] for row in schedule) == {"english": 180, "binding": 20}
    assert all(exposure[row["id"]] == (45 if row["id"].startswith("TRAIN:") else 180) for row in items)

    appearances = Counter()
    correct = Counter()
    distractor = Counter()
    per_stratum = defaultdict(lambda: {"candidate": Counter(), "correct": Counter(), "distractor": Counter()})
    for row in assignment_rows:
        for name in row["candidate_names"]:
            appearances[name] += 1
            per_stratum[row["stratum"]]["candidate"][name] += 1
        correct[row["correct"]] += 1
        distractor[row["distractor"]] += 1
        per_stratum[row["stratum"]]["correct"][row["correct"]] += 1
        per_stratum[row["stratum"]]["distractor"][row["distractor"]] += 1
    assert all(appearances[name] == 4 and correct[name] == 2 and distractor[name] == 2 for name in selected)
    assert all(per_stratum[s][role][name] == (2 if role == "candidate" else 1)
               for s in ["Surface", "Order"] for role in ["candidate", "correct", "distractor"] for name in selected)

    # Copy authoritative payloads. DEV and TRAIN16 are byte-identical; no item outcomes are consulted.
    copied = []
    for relative in [
        "D3_SELECTION.json", "DEV_ORDER.json", "DEV_SURFACE.json", "EXTERNAL_INPUTS.json",
        "KL_POOL_MANIFEST.json", "KL_POOL.json", "SF13_KL_SCHEDULE.json", "SF2_ENGINE.py",
        "SF2_PROTOCOL.json", "TRAIN16_RETENTION.json",
    ]:
        destination = H / relative
        shutil.copyfile(SF13 / relative, destination)
        assert sha(destination) == sha(SF13 / relative)
        copied.append({"source": str(SF13 / relative), "destination": relative, "sha256": sha(destination)})
    for subdir in ["data", "sources"]:
        for source in (SF13 / subdir).glob("*"):
            if source.is_file():
                destination = H / subdir / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                assert sha(destination) == sha(source)
                copied.append({"source": str(source), "destination": destination.relative_to(H).as_posix(),
                               "sha256": sha(destination)})

    write(H / "TRAIN.json", items)
    write(H / "SCHEDULE.json", schedule)
    write(H / "NAME_CENSUS.json", census)
    write(H / "FORBIDDEN_IDENTITIES.json", {"identities": sorted(forbidden), "lexicon_sources": forbidden_sources})
    write(H / "IDENTITY_ASSIGNMENT.json", {
        "selected_names": selected,
        "surface_pair_keys": dict(zip(surface_keys, surface_pairs)),
        "order_pair_keys": dict(zip(order_keys, order_pairs)),
        "first_answer_token_ids": first_ids,
        "records": assignment_rows,
        "balance": {
            "candidate_appearances": dict(appearances),
            "correct": dict(correct),
            "distractor": dict(distractor),
            "per_stratum": {s: {role: dict(values) for role, values in roles.items()}
                             for s, roles in per_stratum.items()},
        },
    })
    write(H / "TOKENIZATION_MATCH.json", {
        "required_abstract_geometry": {"sentence_start": 3, "leading_space": 3, "candidate_with_period": 4},
        "all_surrounding_tokenization_unchanged": all(row["surrounding_tokenization_unchanged"] for row in tokenization_checks),
        "all_prompt_lengths_unchanged": all(row["prompt_length_unchanged"] for row in tokenization_checks),
        "all_first_answer_tokens_distinct_within_item": all(row["first_answer_tokens_distinct"] for row in tokenization_checks),
        "records": tokenization_checks,
    })

    old_protocol = read(SF13 / "PROTOCOL.json")
    runs = []
    for seed, source_run in zip([87053, 87054, 87055], old_protocol["runs"]):
        run = dict(source_run)
        run["seed"] = seed
        runs.append(run)
    protocol = {key: old_protocol[key] for key in [
        "parent_anchor", "parent_sha256", "tokenizer", "tokenizer_sha256", "runtime", "optimizer",
        "kl_data", "binding", "scope", "evaluation_updates", "sampling", "margin", "dev", "gates",
        "evaluation", "stopping", "persistence",
    ]}
    protocol.update({
        "study": "SF20_IDENTITY_HELDOUT_BALANCED_ENTITY_ROTATION_V2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "predecessor_stop": str(SF20_V1_STOP),
        "predecessor_stop_manifest_sha256": sha(SF20_V1_STOP / "SHA256SUMS.txt"),
        "prospective_amendment": "Replace the invalid single-token-name rule with exact 3/3/4 and exact-template token-span matching to the original identities.",
        "hypothesis": "Replacing the four repeated widening identities with 16 balanced, tokenization-matched new identities reduces privileged answer-identity shortcuts and may improve D3 retention and source/order switching.",
        "sole_scientific_variable": "Identity set in the 32 widening records: Alex/Owen/Mia/Nora -> fixed balanced 16-name set. The 16 TRAIN16 preservation records remain byte-identical.",
        "selected_names": selected,
        "selection_rule": census["selection_rule"],
        "curriculum": "Same 16 preservation and 32 widening logical records as SF13. Only identity strings/token IDs in the 32 widening records change.",
        "english_objective": "Byte-equivalent SF13 full answer+EOS causal CE + 1.0 full forward parent KL on frozen broad KL160 + 0.25 first-answer-token pairwise hinge at M=1.0. No SF12 R_name, SF17 projection, SF18 head freeze, or SF19 membership hinge.",
        "runs": runs,
        "seeds": [87053, 87054, 87055],
        "transfer": "Historical transfer panels remain LOCKED/UNSCORED. FINAL and sacred material are prohibited.",
        "optional_new_name_telemetry": "OMITTED to avoid unnecessary complexity; no new telemetry panel is part of SF20.",
        "classification": {
            "priority1": "MECHANICAL_INCOMPLETE if execution/integrity prevents valid classification",
            "priority2": "RETENTION_REGRESSION if any TRAIN16/language/binding retention gate fails",
            "priority3": "IDENTITY_ROTATION_SUPPORTED if >=2/3 branches reach U200 and pass every frozen endpoint gate including D3",
            "priority4": "IDENTITY_ROTATION_INSUFFICIENT if >=2/3 branches fail D3 while retention and widening survive",
            "otherwise": "IDENTITY_ROTATION_WIDENING_STALLED if >=2/3 keep D3/retention but widening does not materially advance; otherwise MIXED_OR_UNRESOLVED",
        },
    })
    write(H / "PROTOCOL.json", protocol)
    external = read(H / "EXTERNAL_INPUTS.json")
    external[str(LANGUAGE_TRAIN)] = LANGUAGE_TRAIN_SHA256
    for source in forbidden_sources:
        external[source["path"]] = source["sha256"]
    write(H / "EXTERNAL_INPUTS.json", external)

    source_controller = (SF13 / "CONTROLLER.py").read_text(encoding="utf-8")
    controller = source_controller.replace("SF13", "SF20")
    controller = controller.replace("broad-coverage KL retention at fixed retention compute (sampling geometry change)",
                                    "identity-heldout balanced entity rotation with SF13 objective")
    controller = controller.replace(
        "ONE scientific variable vs SF11/SF12: retention position coverage / sampling geometry\n(contiguous sequential 160-entry slice -> frozen row-distributed 160-entry schedule).\nAll loss terms, curriculum, optimizer, scope, gates and persistence are unchanged from SF11.",
        "ONE scientific variable vs SF13: widening identity set. Full CE, broad KL160, margin,\noptimizer, scope, schedule, gates, binding, evaluation and persistence remain unchanged."
    )
    (H / "CONTROLLER.py").write_text(controller, encoding="utf-8", newline="\n")

    write(H / "PROVENANCE.json", {
        "study": protocol["study"],
        "sf13_receipt_sha256": sha(SF13 / "FREEZE_RECEIPT.json"),
        "sf13_manifest_sha256": sha(SF13 / "SHA256SUMS.txt"),
        "source_controller_sha256": sha(SF13 / "CONTROLLER.py"),
        "copied_payloads": copied,
        "selection_source": str(LANGUAGE_TRAIN),
        "selection_source_sha256": LANGUAGE_TRAIN_SHA256,
        "forbidden_lexicon_sources": forbidden_sources,
        "model_outputs_used_for_selection": False,
        "treatment_outcomes_exist": False,
        "checkpoint_loaded": False,
        "optimizer_created": False,
        "updates": 0,
        "final_or_sacred_accessed": False,
    })
    (H / "AMENDMENT.md").write_text(
        "# SF20 prospective eligibility amendment\n\n"
        "The preserved v1 pre-seal stop correctly applied the then-written single-token rule. Before any "
        "SF20 model access or update, that rule was prospectively replaced with exact token-span matching "
        "to the actual 3/3/4 original-name geometry. The SF20 scientific variable remains widening identity geometry.\n",
        encoding="utf-8", newline="\n")
    print(json.dumps({"status": "BUILT", "selected_names": selected, "items": len(items),
                      "english_updates": 180, "binding_updates": 20}))


if __name__ == "__main__":
    main()
