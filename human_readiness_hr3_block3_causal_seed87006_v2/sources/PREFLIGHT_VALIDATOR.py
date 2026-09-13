"""Independent no-checkpoint preflight validator for HR-3 block-3 relaxation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path

import torch
from tokenizers import __version__ as tokenizers_version


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_external_receipt(bundle: Path, expected_status: str) -> dict:
    expected = (bundle / "FREEZE_RECEIPT.sha256").read_text(encoding="utf-8").strip().split()[0]
    assert sha(bundle / "FREEZE_RECEIPT.json") == expected
    receipt = read_json(bundle / "FREEZE_RECEIPT.json")
    assert receipt["status"] == expected_status
    assert sha(bundle / "MANIFEST.json") == receipt["manifest_sha256"]
    assert sha(bundle / "SHA256SUMS.txt") == receipt["sha256sums_sha256"]
    for line in (bundle / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            expected_hash, rel = line.split("  ", 1)
            assert sha(bundle / rel) == expected_hash
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    args = ap.parse_args()
    bundle = args.bundle
    protocol = read_json(bundle / "HR3_PROTOCOL.json")
    assert protocol["status"] == "HR3_BLOCK3_PROSPECTIVE_PROTOCOL"
    assert protocol["parent_sha256"] == "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb"
    parent = Path(protocol["parent_path"])
    assert sha(parent) == protocol["parent_sha256"]
    assert sha(Path(protocol["tokenizer_path"])) == protocol["tokenizer_sha256"]
    assert tuple(map(int, platform.python_version().split(".")[:2])) == (3, 12)
    assert torch.__version__ == "2.12.0+rocm7.14.0"
    assert tokenizers_version == "0.23.1"
    assert torch.cuda.is_available()

    for rel, expected in protocol["frozen_payload_hashes"].items():
        assert sha(bundle / rel) == expected, rel
    train = [json.loads(x) for x in (bundle / protocol["data"]["english_train"]).read_text(encoding="utf-8").splitlines() if x]
    dev = [json.loads(x) for x in (bundle / protocol["data"]["english_dev"]).read_text(encoding="utf-8").splitlines() if x]
    schedule = read_json(bundle / protocol["data"]["english_schedule"])["batches"]
    binding_schedule = read_json(bundle / protocol["data"]["binding_schedule"])["batches"]
    rehearsal = read_json(bundle / protocol["data"]["binding_rehearsal"])["quartets"]
    ids = {r["id"] for r in train}
    assert len(train) == 149014 and len(dev) == 15866
    assert len(schedule) == 450 and all(len(batch["record_ids"]) == 64 for batch in schedule)
    assert all(record_id in ids for batch in schedule for record_id in batch["record_ids"])
    assert len(rehearsal) == 80 and sum(len(q["docs"]) for q in rehearsal) == 320
    assert len(binding_schedule) == 50 and all(len(batch["quartet_ids"]) == 8 for batch in binding_schedule)
    assert all(isinstance(qid, int) and 0 <= qid < len(rehearsal) for batch in binding_schedule for qid in batch["quartet_ids"])
    for batch in binding_schedule:
        expected_docs = [doc["doc_id"] for qid in batch["quartet_ids"] for doc in rehearsal[qid]["docs"]]
        assert expected_docs == batch["documents"], "literal binding schedule disagrees with canonical indexed rehearsal source"

    # The static source check establishes the only scientific delta: block 3 is
    # newly trainable on English; blocks 0--2 and all T13 components remain frozen.
    runtime = (bundle / "sources" / "hr3_block3_runtime.py").read_text(encoding="utf-8")
    trainer = (bundle / "sources" / "TRAIN_HR3.py").read_text(encoding="utf-8")
    evaluator = (bundle / "sources" / "EVALUATE_HR3.py").read_text(encoding="utf-8")
    assert "for index in range(3):" in runtime
    assert "for index in range(4):" not in runtime
    assert "x[i, : len(z) - 1] = torch.tensor(z[:-1]" in runtime
    assert "y[i, : len(z) - 1] = torch.tensor(z[1:]" in runtime
    assert "aligned_tensors" in trainer and "aligned_dev_loss" in evaluator
    assert "AdamW" in trainer and "lr=5e-5" in trainer
    assert "range(start, 501)" in trainer
    assert "FINAL_ITEMS" not in trainer.upper() and "FINAL_ITEMS" not in evaluator.upper()
    assert "human_test_readiness_v1" not in trainer and "human_test_readiness_v1" not in evaluator
    assert "binding_eval" in trainer and "binding_eval" in evaluator
    assert "checkpoint_500" in trainer and "for update in (100, 250, 500):" in evaluator

    readiness = Path(protocol["readiness_dev"]["path"])
    readiness_receipt = check_external_receipt(readiness, "FROZEN_PROSPECTIVE_READINESS_DEV_V2")
    assert sha(readiness / "FREEZE_RECEIPT.json") == protocol["readiness_dev"]["receipt_sha256"]
    assert (readiness / "DEV_ITEMS.jsonl").exists()
    # Never enumerate or inspect the sealed FINAL item path.

    z = [2, 37, 268, 18, 3]
    x = z[:-1]
    y = z[1:]
    assert list(zip(x, y)) == [(2, 37), (37, 268), (268, 18), (18, 3)]
    assert x[-1] != y[-1] and y[-1] == 3
    output = {
        "status": "PASS_HR3_BLOCK3_PREFLIGHT_NO_CHECKPOINT_NO_OPTIMIZER",
        "parent_sha256": protocol["parent_sha256"],
        "tokenizer_sha256": protocol["tokenizer_sha256"],
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "tokenizers": tokenizers_version, "cuda_available": torch.cuda.is_available()},
        "train_records": len(train),
        "dev_records": len(dev),
        "english_batches": len(schedule),
        "binding_batches": len(binding_schedule),
        "binding_quartets": len(rehearsal),
        "scope": "English trainable blocks 3-7; blocks 0-2 and T13 localization/retrieval frozen. Binding full scope.",
        "alignment_transitions": list(zip(x, y)),
        "readiness_dev_receipt_sha256": sha(readiness / "FREEZE_RECEIPT.json"),
        "readiness_dev_status": readiness_receipt["status"],
        "final_accessed": False,
        "sacred_accessed": False,
        "checkpoint_loaded": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
    }
    (bundle / "PREFLIGHT.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
