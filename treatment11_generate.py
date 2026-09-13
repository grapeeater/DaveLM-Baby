r"""Create T11-owned immutable copies of the validated strict-reversal universe.

This is an objective-only controlled treatment. It reads only the frozen T10
structural pools, verifies their hashes, preserves every token array, assigns
new T11-owned record identifiers, and writes no behavioral information.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

from treatment11_config import (
    GENERATOR_RESULT_PATH,
    NAME,
    OUT_ROOT,
    RETENTION_POOL_PATH,
    SEED,
    SOURCE_T10_RETENTION_POOL,
    SOURCE_T10_RETENTION_POOL_SHA256,
    SOURCE_T10_TRAIN_POOL,
    SOURCE_T10_TRAIN_POOL_SHA256,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def clone_pool(source, axis: str):
    pool = copy.deepcopy(source)
    pool["artifact_type"] = f"treatment11_{axis}_quartet_pool"
    pool["experiment"] = NAME
    pool["seed"] = SEED
    for index, quartet in enumerate(pool["quartets"]):
        new_qid = f"t11_{axis}:qt_{index:06d}"
        quartet["quartet_id"] = new_qid
        quartet["axis"] = f"t11_{axis}"
        for doc in quartet["docs"]:
            doc["axis"] = f"t11_{axis}"
            doc["quartet_id"] = new_qid
            doc["doc_id"] = f"{new_qid}:{doc['member']}"
    return pool


def digest_ids(ids) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def main() -> int:
    require(sha256_file(SOURCE_T10_TRAIN_POOL) == SOURCE_T10_TRAIN_POOL_SHA256,
            "source T10 training structural pool hash mismatch")
    require(sha256_file(SOURCE_T10_RETENTION_POOL) == SOURCE_T10_RETENTION_POOL_SHA256,
            "source T10 retention structural pool hash mismatch")

    train = clone_pool(read_json(SOURCE_T10_TRAIN_POOL), "train")
    retention = clone_pool(read_json(SOURCE_T10_RETENTION_POOL), "retention")
    train_digests = {
        digest_ids(doc["full_document_token_ids"])
        for quartet in train["quartets"] for doc in quartet["docs"]
    }
    retention_digests = {
        digest_ids(doc["full_document_token_ids"])
        for quartet in retention["quartets"] for doc in quartet["docs"]
    }
    require(not train_digests.intersection(retention_digests), "T11 train/retention token overlap")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(TRAIN_POOL_PATH, train)
    write_json(RETENTION_POOL_PATH, retention)
    result = {
        "status": "TREATMENT11_GENERATOR_PASS",
        "experiment": NAME,
        "seed": SEED,
        "source_contract": "T10 strict-reversal structural construction; token arrays unchanged",
        "source_train_pool_sha256": SOURCE_T10_TRAIN_POOL_SHA256,
        "source_retention_pool_sha256": SOURCE_T10_RETENTION_POOL_SHA256,
        "train_pool_sha256": sha256_file(TRAIN_POOL_PATH),
        "retention_pool_sha256": sha256_file(RETENTION_POOL_PATH),
        "train_quartets": train["quartet_count"],
        "train_documents": train["document_count"],
        "retention_quartets": retention["quartet_count"],
        "retention_documents": retention["document_count"],
        "train_retention_overlap": 0,
        "behavioral_inputs_used": False,
    }
    write_json(GENERATOR_RESULT_PATH, result)
    print(result)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"GENERATOR ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
