r"""Outcome-blind structural preflight and frozen schedule for Treatment 11."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from treatment11_config import (
    BATCH_SIZE,
    EXPECTED_DOCUMENT_LENGTH,
    EXPECTED_RETENTION_DOCUMENTS,
    EXPECTED_RETENTION_QUARTETS,
    EXPECTED_TRAIN_DOCUMENTS,
    EXPECTED_TRAIN_QUARTETS,
    GENERATOR_RESULT_PATH,
    MAX_STEPS,
    NAME,
    PREFLIGHT_RESULT_PATH,
    QUARTETS_PER_STEP,
    RETENTION_POOL_PATH,
    SCHEDULE_PATH,
    SEED,
    START_CHECKPOINT_PATH,
    START_CHECKPOINT_SHA256,
    TRAIN_POOL_PATH,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def flatten(pool):
    return [doc for quartet in pool["quartets"] for doc in quartet["docs"]]


def digest_ids(ids) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def validate_pool(pool, axis: str, expected_quartets: int, expected_docs: int):
    require(pool["quartet_count"] == expected_quartets, f"{axis}: quartet count")
    docs = flatten(pool)
    require(len(docs) == expected_docs == pool["document_count"], f"{axis}: document count")
    orientation = Counter(doc["orientation"] for doc in docs)
    slots = Counter(doc["query_slot"] for doc in docs)
    require(orientation[1] == orientation[2], f"{axis}: orientation imbalance")
    require(slots[0] == slots[1], f"{axis}: slot imbalance")

    by_query = defaultdict(list)
    candidate_orientation = Counter()
    for doc in docs:
        require(len(doc["full_document_token_ids"]) == EXPECTED_DOCUMENT_LENGTH,
                f"{axis}: document length")
        by_query[doc["query_key_token"]].append(doc)
        candidate_orientation[(tuple(doc["candidate_pair_sorted"]), doc["orientation"])] += 1
    per_query = []
    for qid in sorted(by_query):
        group = by_query[qid]
        targets = sorted({doc["target_value_token"] for doc in group})
        query_slots = sorted({doc["query_slot"] for doc in group})
        orient = Counter(doc["orientation"] for doc in group)
        require(len(targets) > 1, f"{axis}: query deterministically predicts target")
        require(query_slots == [0, 1], f"{axis}: query deterministically predicts slot")
        require(orient[1] == orient[2], f"{axis}: per-query orientation imbalance")
        per_query.append({
            "query_key_token": qid,
            "query_word": group[0]["query_key_word"],
            "documents": len(group),
            "target_values": targets,
            "query_slots": query_slots,
            "orientation_counts": {"1": orient[1], "2": orient[2]},
        })

    for candidate in {key for key, _ in candidate_orientation}:
        require(candidate_orientation[(candidate, 1)] == candidate_orientation[(candidate, 2)],
                f"{axis}: candidate orientation imbalance")

    members = {"o1_k0", "o1_k1", "o2_k0", "o2_k1"}
    for quartet in pool["quartets"]:
        by_member = {doc["member"]: doc for doc in quartet["docs"]}
        require(set(by_member) == members, f"{axis}: quartet members")
        for key in ("k0", "k1"):
            o1 = by_member[f"o1_{key}"]
            o2 = by_member[f"o2_{key}"]
            require(o1["query_key_token"] == o2["query_key_token"], "query mismatch across reversal")
            require(o1["candidate_pair_sorted"] == o2["candidate_pair_sorted"], "candidate mismatch")
            require(o1["target_value_token"] == o2["distractor_value_token"], "o1 target != o2 distractor")
            require(o2["target_value_token"] == o1["distractor_value_token"], "o2 target != o1 distractor")
            require(o1["query_slot"] == o2["query_slot"], "slot changed across reversal")
            require(o1["qdp"] == o2["qdp"], "qdp changed across reversal")
            require(o1["answer_token_index"] == o2["answer_token_index"], "answer position changed")
    return docs, per_query, orientation, slots


def build_schedule(quartets):
    quartet_ids = [q["quartet_id"] for q in quartets]
    total_presentations = MAX_STEPS * QUARTETS_PER_STEP
    require(total_presentations % len(quartet_ids) == 0, "schedule cannot be exactly balanced")
    rounds = total_presentations // len(quartet_ids)
    steps = []
    rng = random.Random(1_100_000 + SEED)
    for _ in range(rounds):
        order = list(quartet_ids)
        rng.shuffle(order)
        for start in range(0, len(order), QUARTETS_PER_STEP):
            ids = order[start:start + QUARTETS_PER_STEP]
            require(len(ids) == QUARTETS_PER_STEP and len(set(ids)) == QUARTETS_PER_STEP,
                    "invalid quartet batch")
            steps.append({"step": len(steps) + 1, "quartet_ids": ids})
    require(len(steps) == MAX_STEPS, "step count mismatch")
    exposure = Counter(qid for step in steps for qid in step["quartet_ids"])
    require(min(exposure.values()) == max(exposure.values()) == rounds, "exposure imbalance")
    return {
        "artifact_type": "treatment11_frozen_schedule",
        "experiment": NAME,
        "seed": SEED,
        "steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "quartets_per_step": QUARTETS_PER_STEP,
        "documents_per_step": BATCH_SIZE,
        "answer_decisions_per_step": BATCH_SIZE,
        "presentations_per_quartet": rounds,
        "schedule_document_events": MAX_STEPS * BATCH_SIZE,
        "schedule_answer_decisions": MAX_STEPS * BATCH_SIZE,
        "steps_data": steps,
    }


def main() -> int:
    generator = read_json(GENERATOR_RESULT_PATH)
    require(generator["status"] == "TREATMENT11_GENERATOR_PASS", "generator status")
    require(sha256_file(TRAIN_POOL_PATH) == generator["train_pool_sha256"], "train hash")
    require(sha256_file(RETENTION_POOL_PATH) == generator["retention_pool_sha256"], "retention hash")

    train = read_json(TRAIN_POOL_PATH)
    retention = read_json(RETENTION_POOL_PATH)
    train_docs, per_query, train_orientation, train_slots = validate_pool(
        train, "train", EXPECTED_TRAIN_QUARTETS, EXPECTED_TRAIN_DOCUMENTS)
    retention_docs, _, retention_orientation, retention_slots = validate_pool(
        retention, "retention", EXPECTED_RETENTION_QUARTETS, EXPECTED_RETENTION_DOCUMENTS)
    require(not ({digest_ids(d["full_document_token_ids"]) for d in train_docs}
                 & {digest_ids(d["full_document_token_ids"]) for d in retention_docs}),
            "train/retention overlap")

    checkpoint_hash = sha256_file(START_CHECKPOINT_PATH)
    require(checkpoint_hash == START_CHECKPOINT_SHA256, "baseline checkpoint hash")
    schedule = build_schedule(train["quartets"])
    write_json(SCHEDULE_PATH, schedule)

    result = {
        "status": "TREATMENT11_PREFLIGHT_PASS",
        "experiment": NAME,
        "train_pool_sha256": generator["train_pool_sha256"],
        "retention_pool_sha256": generator["retention_pool_sha256"],
        "schedule_sha256": sha256_file(SCHEDULE_PATH),
        "start_checkpoint": {
            "path": str(START_CHECKPOINT_PATH),
            "expected_sha256": START_CHECKPOINT_SHA256,
            "observed_sha256": checkpoint_hash,
        },
        "objective": {
            "type": "answer_only_causal_cross_entropy",
            "supervised_answer_decisions_per_document": 1,
            "supervised_answer_decisions_per_step": BATCH_SIZE,
            "other_supervised_positions_per_step": 0,
        },
        "train": {
            "quartets": train["quartet_count"],
            "documents": len(train_docs),
            "orientation_counts": dict(sorted(train_orientation.items())),
            "query_slot_counts": dict(sorted(train_slots.items())),
            "per_query_identity": per_query,
        },
        "retention": {
            "quartets": retention["quartet_count"],
            "documents": len(retention_docs),
            "orientation_counts": dict(sorted(retention_orientation.items())),
            "query_slot_counts": dict(sorted(retention_slots.items())),
            "training_overlap": 0,
        },
        "schedule": {
            "steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "quartets_per_step": QUARTETS_PER_STEP,
            "presentations_per_quartet": schedule["presentations_per_quartet"],
            "answer_decisions": schedule["schedule_answer_decisions"],
        },
        "anti_shortcut_contract_passed": True,
        "behavioral_inputs_used": False,
    }
    write_json(PREFLIGHT_RESULT_PATH, result)
    print(result)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"PREFLIGHT ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
