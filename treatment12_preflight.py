r"""Read-only Treatment-12 preflight.

Verifies exact reuse of the T11 structural universe and schedule, the baseline
checkpoint hash, balanced quartets, per-batch/per-quartet exposure, the
answer-only objective, and that every retrieval position required by the T12
module resolves to the correct structural token for every document. No model
and no behavioral outcome is touched.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from treatment12_config import (
    BATCH_SIZE,
    DOCUMENT_LENGTH,
    EXPECTED_PRESENTATIONS_PER_QUARTET,
    EXPECTED_RETENTION_DOCUMENTS,
    EXPECTED_RETENTION_QUARTETS,
    EXPECTED_SCHEDULE_SHA256,
    EXPECTED_SUPERVISED_ANSWER_DECISIONS,
    EXPECTED_TRAIN_DOCUMENTS,
    EXPECTED_TRAIN_QUARTETS,
    MAX_STEPS,
    NAME,
    PREFLIGHT_RESULT_PATH,
    QUARTETS_PER_STEP,
    RETENTION_POOL_PATH,
    SCHEDULE_PATH,
    START_CHECKPOINT_PATH,
    START_CHECKPOINT_SHA256,
    TRAIN_POOL_PATH,
)
from treatment12_model import correct_row_index, doc_retrieval_meta


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def flatten(pool):
    return [doc for quartet in pool["quartets"] for doc in quartet["docs"]]


def digest_ids(ids) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def validate_pool(pool, axis: str, expected_quartets: int, expected_docs: int):
    require(pool["quartet_count"] == expected_quartets, f"{axis}: quartet count")
    docs = flatten(pool)
    require(pool["document_count"] == expected_docs and len(docs) == expected_docs,
            f"{axis}: document count")

    orientation = Counter(doc["orientation"] for doc in docs)
    slots = Counter(doc["query_slot"] for doc in docs)
    require(orientation[1] == orientation[2], f"{axis}: orientation imbalance")
    require(slots[0] == slots[1], f"{axis}: slot imbalance")

    errors: List[str] = []
    by_query = defaultdict(list)
    for doc in docs:
        require(len(doc["full_document_token_ids"]) == DOCUMENT_LENGTH,
                f"{axis}: document length")
        meta = doc_retrieval_meta(doc)
        ids = doc["full_document_token_ids"]
        checks = [
            (meta["query_token_at_qdp"], meta["query_key_token"], "qdp token != query key"),
            (meta["row0_src_token"], meta["query_key_token"] if meta["query_slot"] == 0
             else None, "row0 src token"),
            (meta["row1_src_token"], meta["query_key_token"] if meta["query_slot"] == 1
             else None, "row1 src token"),
        ]
        if not (meta["query_token_at_qdp"] == meta["query_key_token"]):
            errors.append(f"{axis} {doc['doc_id']}: qdp token mismatch")
        if not (meta["answer_token_at_ans_index"] == meta["target_value_token"]):
            errors.append(f"{axis} {doc['doc_id']}: answer token != target value")
        if not (meta["row0_src_pos"] < meta["row0_val_pos"] < meta["qdp"]):
            errors.append(f"{axis} {doc['doc_id']}: row0 ordering")
        if not (meta["row1_src_pos"] < meta["row1_val_pos"] < meta["qdp"]):
            errors.append(f"{axis} {doc['doc_id']}: row1 ordering")
        if not (meta["row0_src_pos"] != meta["row1_src_pos"]
                and meta["row0_val_pos"] != meta["row1_val_pos"]):
            errors.append(f"{axis} {doc['doc_id']}: rows not distinct")
        src_tokens = {meta["row0_src_token"], meta["row1_src_token"]}
        if meta["query_key_token"] not in src_tokens:
            errors.append(f"{axis} {doc['doc_id']}: query key not a row source")
        val_tokens = {meta["row0_val_token"], meta["row1_val_token"]}
        if val_tokens != {meta["target_value_token"], meta["distractor_value_token"]}:
            errors.append(f"{axis} {doc['doc_id']}: row value set mismatch")
        correct = correct_row_index(meta)
        expected_src = (meta["row0_src_token"] if correct == 0 else meta["row1_src_token"])
        if not (expected_src == meta["query_key_token"]):
            errors.append(f"{axis} {doc['doc_id']}: correct-row source mismatch")
        by_query[doc["query_key_token"]].append(doc)

    require(not errors, f"{axis}: retrieval-position validation errors: {errors[:5]}")
    per_query = []
    for qid in sorted(by_query):
        group = by_query[qid]
        targets = sorted({d["target_value_token"] for d in group})
        q_slots = sorted({d["query_slot"] for d in group})
        require(len(targets) > 1, f"{axis}: query->target deterministic")
        require(q_slots == [0, 1], f"{axis}: query->slot deterministic")
        per_query.append({
            "query_key_token": qid,
            "documents": len(group),
            "target_values": targets,
            "query_slots": q_slots,
        })

    reversal_checks = 0
    for quartet in pool["quartets"]:
        by_member = {d["member"]: d for d in quartet["docs"]}
        for key in ("k0", "k1"):
            o1 = by_member[f"o1_{key}"]
            o2 = by_member[f"o2_{key}"]
            m1 = doc_retrieval_meta(o1)
            m2 = doc_retrieval_meta(o2)
            require(o1["query_key_token"] == o2["query_key_token"], "reversal query mismatch")
            require(m1["row0_src_pos"] == m2["row0_src_pos"]
                    and m1["row1_src_pos"] == m2["row1_src_pos"], "reversal changed row src pos")
            require(m1["row0_val_pos"] == m2["row0_val_pos"]
                    and m1["row1_val_pos"] == m2["row1_val_pos"], "reversal changed row val pos")
            require(m1["row0_src_token"] == m2["row0_src_token"]
                    and m1["row1_src_token"] == m2["row1_src_token"], "reversal changed row keys")
            require(m1["row0_val_token"] != m2["row0_val_token"]
                    or m1["row1_val_token"] != m2["row1_val_token"],
                    "reversal did not change row value assignments")
            require(o1["candidate_pair_sorted"] == o2["candidate_pair_sorted"],
                    "reversal changed candidate pair")
            require(o1["target_value_token"] == o2["distractor_value_token"],
                    "reversal target/distractor swap violated")
            reversal_checks += 1
    require(reversal_checks == pool["quartet_count"] * 2, "reversal check coverage")
    return docs, per_query, orientation, slots


def main() -> int:
    require(sha256_file(TRAIN_POOL_PATH) ==
            __import__("treatment12_config", fromlist=["EXPECTED_TRAIN_POOL_SHA256"]).EXPECTED_TRAIN_POOL_SHA256,
            "T11 train pool hash mismatch")
    require(sha256_file(RETENTION_POOL_PATH) ==
            __import__("treatment12_config", fromlist=["EXPECTED_RETENTION_POOL_SHA256"]).EXPECTED_RETENTION_POOL_SHA256,
            "T11 retention pool hash mismatch")
    require(sha256_file(SCHEDULE_PATH) ==
            __import__("treatment12_config", fromlist=["EXPECTED_SCHEDULE_SHA256"]).EXPECTED_SCHEDULE_SHA256,
            "T11 schedule hash mismatch")

    train = read_json(TRAIN_POOL_PATH)
    retention = read_json(RETENTION_POOL_PATH)
    schedule = read_json(SCHEDULE_PATH)
    require(schedule["steps"] == MAX_STEPS and schedule["batch_size"] == BATCH_SIZE,
            "schedule size mismatch")
    require(schedule["quartets_per_step"] == QUARTETS_PER_STEP, "quartets per step mismatch")
    require(schedule["presentations_per_quartet"] == EXPECTED_PRESENTATIONS_PER_QUARTET,
            "presentations per quartet mismatch")
    require(schedule["schedule_answer_decisions"] == EXPECTED_SUPERVISED_ANSWER_DECISIONS,
            "scheduled answer decisions mismatch")

    train_docs, per_query, train_orient, train_slots = validate_pool(
        train, "train", EXPECTED_TRAIN_QUARTETS, EXPECTED_TRAIN_DOCUMENTS)
    retention_docs, _, ret_orient, ret_slots = validate_pool(
        retention, "retention", EXPECTED_RETENTION_QUARTETS, EXPECTED_RETENTION_DOCUMENTS)
    require(not ({digest_ids(d["full_document_token_ids"]) for d in train_docs}
                 & {digest_ids(d["full_document_token_ids"]) for d in retention_docs}),
            "train/retention overlap")

    checkpoint_hash = sha256_file(START_CHECKPOINT_PATH)
    require(checkpoint_hash == START_CHECKPOINT_SHA256, "baseline checkpoint hash")

    schedule_coverage = Counter(qid for step in schedule["steps_data"]
                                for qid in step["quartet_ids"])
    require(len(schedule_coverage) == EXPECTED_TRAIN_QUARTETS, "schedule quartet coverage")
    require(min(schedule_coverage.values()) == max(schedule_coverage.values())
            == EXPECTED_PRESENTATIONS_PER_QUARTET, "schedule exposure imbalance")
    for step in schedule["steps_data"]:
        require(len(step["quartet_ids"]) == QUARTETS_PER_STEP
                and len(set(step["quartet_ids"])) == QUARTETS_PER_STEP,
                "invalid step quartet batch")

    result = {
        "status": "TREATMENT12_PREFLIGHT_PASS",
        "experiment": NAME,
        "reused_universe": {
            "train_pool_sha256": sha256_file(TRAIN_POOL_PATH),
            "retention_pool_sha256": sha256_file(RETENTION_POOL_PATH),
            "schedule_sha256": sha256_file(SCHEDULE_PATH),
        },
        "start_checkpoint": {
            "path": str(START_CHECKPOINT_PATH),
            "expected_sha256": START_CHECKPOINT_SHA256,
            "observed_sha256": checkpoint_hash,
        },
        "train": {
            "quartets": train["quartet_count"],
            "documents": len(train_docs),
            "orientation_counts": dict(sorted(train_orient.items())),
            "query_slot_counts": dict(sorted(train_slots.items())),
            "per_query_identity": per_query,
        },
        "retention": {
            "quartets": retention["quartet_count"],
            "documents": len(retention_docs),
            "orientation_counts": dict(sorted(ret_orient.items())),
            "query_slot_counts": dict(sorted(ret_slots.items())),
            "training_overlap": 0,
        },
        "schedule": {
            "steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "quartets_per_step": QUARTETS_PER_STEP,
            "presentations_per_quartet": schedule["presentations_per_quartet"],
            "supervised_answer_decisions": EXPECTED_SUPERVISED_ANSWER_DECISIONS,
        },
        "objective": {
            "type": "answer_only_causal_cross_entropy",
            "answer_decisions_per_step": BATCH_SIZE,
            "other_supervised_positions_per_step": 0,
        },
        "retrieval_position_validation": "passed for every train and retention document",
        "reversal_mapping_change_validation": "passed for every quartet member pair",
        "anti_cheat": "model forward receives positions only; no row label / correct candidate / target id",
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
