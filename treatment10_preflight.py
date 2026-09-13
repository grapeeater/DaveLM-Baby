r"""Treatment-10 structural preflight.

Reads the generated training/retention quartet pools, verifies the frozen
structural and anti-shortcut contract outcome-blind, builds the balanced
frozen training schedule (whole-quartet batches, exact equal exposure), and
writes the preflight result. No model and no checkpoint are touched.
"""

from __future__ import annotations

import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

from treatment10_config import (
    BATCH_SIZE,
    GENERATOR_RESULT_PATH,
    MAX_STEPS,
    NAME,
    PREFLIGHT_RESULT_PATH,
    QUARTETS_PER_STEP,
    SCHEDULE_PATH,
    START_CHECKPOINT_PATH,
    START_CHECKPOINT_SHA256,
    SEED,
)
from treatment10_common import Treatment10Error, load_json, require, sha256_file, write_json


def _digest(ids) -> str:
    return ",".join(str(int(t)) for t in ids)


def flatten_docs(pool) -> List[Dict[str, Any]]:
    docs = []
    for quartet in pool["quartets"]:
        docs.extend(quartet["docs"])
    return docs


def validate_anti_shortcut(docs: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    orient = Counter(doc["orientation"] for doc in docs)
    slots = Counter(doc["query_slot"] for doc in docs)
    by_query: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        by_query[doc["query_key_token"]].append(doc)

    per_query = []
    for qid in sorted(by_query):
        group = by_query[qid]
        targets = set(doc["target_value_token"] for doc in group)
        slots_q = set(doc["query_slot"] for doc in group)
        orient_q = Counter(doc["orientation"] for doc in group)
        cand_sets = set(tuple(doc["candidate_pair_sorted"]) for doc in group)
        per_query.append({
            "query_key_token": qid,
            "query_word": group[0]["query_key_word"],
            "document_count": len(group),
            "distinct_target_values": sorted(targets),
            "distinct_target_count": len(targets),
            "query_slots": sorted(slots_q),
            "orientation_counts": {str(k): v for k, v in sorted(orient_q.items())},
            "distinct_candidate_sets": sorted([list(c) for c in cand_sets]),
        })

    require(orient.get(1, 0) == orient.get(2, 0),
            f"{label}: orientation counts unequal.")
    require(slots.get(0, 0) == slots.get(1, 0),
            f"{label}: query slot counts unequal.")
    for row in per_query:
        require(row["distinct_target_count"] >= 2,
                f"{label}: query {row['query_key_token']} has fewer than 2 distinct targets.")
        require(set(row["query_slots"]) == {0, 1},
                f"{label}: query {row['query_key_token']} not seen in both slots.")

    return {
        "unit": "unique training documents",
        "orientation_document_counts": {str(k): v for k, v in sorted(orient.items())},
        "query_slot_document_counts": {str(k): v for k, v in sorted(slots.items())},
        "query_identity_count": len(per_query),
        "per_query_identity": per_query,
        "checks_passed": True,
    }


def validate_matched_quartets(pool, label: str) -> None:
    expected_diff = {
        ("o1_k0", "o1_k1"): 2,
        ("o2_k0", "o2_k1"): 2,
        ("o1_k0", "o2_k0"): 3,
        ("o1_k1", "o2_k1"): 3,
    }
    pairs = [
        ("o1_k0", "o1_k1"),
        ("o2_k0", "o2_k1"),
        ("o1_k0", "o2_k0"),
        ("o1_k1", "o2_k1"),
    ]
    for quartet in pool["quartets"]:
        by_member = {doc["member"]: doc for doc in quartet["docs"]}
        for ma, mb in pairs:
            ids_a = by_member[ma]["full_document_token_ids"]
            ids_b = by_member[mb]["full_document_token_ids"]
            require(len(ids_a) == len(ids_b), "member lengths differ")
            diff = sum(1 for x, y in zip(ids_a, ids_b) if x != y)
            require(diff == expected_diff[(ma, mb)],
                    f"{label} {quartet['quartet_id']}: {ma} vs {mb} diff={diff}, "
                    f"expected {expected_diff[(ma, mb)]}.")


def build_schedule(quartets: List[Dict[str, Any]]) -> Dict[str, Any]:
    quartet_ids = [q["quartet_id"] for q in quartets]
    q = len(quartet_ids)
    require(MAX_STEPS * QUARTETS_PER_STEP % q == 0,
            "steps*quartets_per_step not divisible by quartet count")
    presentations = (MAX_STEPS * QUARTETS_PER_STEP) // q
    steps_data = []
    rng = random.Random(900_000 + SEED)
    for _round in range(presentations):
        order = list(range(q))
        rng.shuffle(order)
        for start in range(0, q, QUARTETS_PER_STEP):
            ids = [quartet_ids[i] for i in order[start:start + QUARTETS_PER_STEP]]
            require(len(set(ids)) == QUARTETS_PER_STEP, "duplicate quartet in a step")
            steps_data.append({"quartet_ids": ids})
    require(len(steps_data) == MAX_STEPS, "step count mismatch")
    for step_index, step in enumerate(steps_data):
        step["step"] = step_index + 1
    exposure = Counter()
    for step in steps_data:
        for qid in step["quartet_ids"]:
            exposure[qid] += 1
    require(set(exposure) == set(quartet_ids), "schedule does not cover all quartets")
    require(min(exposure.values()) == max(exposure.values()),
            "quartet exposure not exactly balanced")

    return {
        "artifact_type": "treatment10_frozen_schedule",
        "experiment": NAME,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "quartets_per_step": QUARTETS_PER_STEP,
        "docs_per_step": BATCH_SIZE,
        "steps": MAX_STEPS,
        "quartet_count": q,
        "presentations_per_quartet": presentations,
        "schedule_document_events": MAX_STEPS * BATCH_SIZE,
        "exposure_min": min(exposure.values()),
        "exposure_max": max(exposure.values()),
        "steps_data": steps_data,
    }


def main() -> int:
    generator_result = load_json(GENERATOR_RESULT_PATH)
    train_pool = load_json(__import__("treatment10_config", fromlist=["TRAIN_POOL_PATH"]).TRAIN_POOL_PATH)
    retention_pool = load_json(__import__("treatment10_config", fromlist=["RETENTION_POOL_PATH"]).RETENTION_POOL_PATH)

    require(sha256_file(__import__("treatment10_config", fromlist=["TRAIN_POOL_PATH"]).TRAIN_POOL_PATH)
            == generator_result["train_pool_sha256"], "train pool hash mismatch vs generator.")
    require(sha256_file(__import__("treatment10_config", fromlist=["RETENTION_POOL_PATH"]).RETENTION_POOL_PATH)
            == generator_result["retention_pool_sha256"], "retention pool hash mismatch vs generator.")

    train_docs = flatten_docs(train_pool)
    retention_docs = flatten_docs(retention_pool)
    train_digests = {doc["doc_id"]: _digest(doc["full_document_token_ids"]) for doc in train_docs}
    retention_digests = {doc["doc_id"]: _digest(doc["full_document_token_ids"]) for doc in retention_docs}
    require(not (set(train_digests.values()) & set(retention_digests.values())),
            "retention/training document token overlap detected")

    validate_matched_quartets(train_pool, "train")
    validate_matched_quartets(retention_pool, "retention")

    anti = validate_anti_shortcut(train_docs, "train")

    candidate_orient = Counter()
    for doc in train_docs:
        key = tuple(doc["candidate_pair_sorted"])
        candidate_orient[(key, doc["orientation"])] += 1
    candidate_balance = {}
    grouped = defaultdict(dict)
    for (key, orient), count in candidate_orient.items():
        grouped[key][orient] = count
    for key, counts in grouped.items():
        candidate_balance[str(list(key))] = {str(o): counts.get(o, 0) for o in (1, 2)}

    require(all(counts.get(1) == counts.get(2) for counts in grouped.values()),
            "candidate-set orientation counts unequal.")

    schedule = build_schedule(train_pool["quartets"])
    write_json(SCHEDULE_PATH, schedule)

    schedule_sha = sha256_file(SCHEDULE_PATH)
    preflight_result = {
        "treatment": NAME,
        "status": "TREATMENT10_PREFLIGHT_PASS",
        "generator_result_path": str(GENERATOR_RESULT_PATH),
        "train_pool_sha256": generator_result["train_pool_sha256"],
        "retention_pool_sha256": generator_result["retention_pool_sha256"],
        "schedule_sha256": schedule_sha,
        "start_checkpoint": {
            "path": str(START_CHECKPOINT_PATH),
            "expected_sha256": START_CHECKPOINT_SHA256,
            "observed_sha256": sha256_file(START_CHECKPOINT_PATH),
        },
        "train": {
            "quartets": train_pool["quartet_count"],
            "documents": train_pool["document_count"],
            "orientation_document_counts": anti["orientation_document_counts"],
            "query_slot_document_counts": anti["query_slot_document_counts"],
            "per_query_identity": anti["per_query_identity"],
        },
        "retention": {
            "quartets": retention_pool["quartet_count"],
            "documents": retention_pool["document_count"],
            "orientation_document_counts": dict(Counter(doc["orientation"] for doc in retention_docs)),
            "query_slot_document_counts": dict(Counter(doc["query_slot"] for doc in retention_docs)),
            "overlap_with_training": 0,
        },
        "schedule": {
            "steps": schedule["steps"],
            "batch_size": schedule["batch_size"],
            "quartets_per_step": schedule["quartets_per_step"],
            "presentations_per_quartet": schedule["presentations_per_quartet"],
            "document_events": schedule["schedule_document_events"],
            "exposure_min": schedule["exposure_min"],
            "exposure_max": schedule["exposure_max"],
        },
        "anti_shortcut_contract": anti,
        "candidate_set_orientation_balance": candidate_balance,
        "matched_quartet_diff_validation": "passed for train and retention pools",
        "note": "Structural preflight only. No model, checkpoint, or behavioral outcome used.",
    }
    write_json(PREFLIGHT_RESULT_PATH, preflight_result)
    print({k: v for k, v in preflight_result.items() if k != "start_checkpoint"})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Treatment10Error as exc:
        print(f"PREFLIGHT ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
