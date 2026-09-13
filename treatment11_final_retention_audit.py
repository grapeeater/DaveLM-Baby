r"""Read-only Treatment-11 strict-counterfactual retention audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from treatment11_config import (
    EXPECTED_VOCAB_SIZE,
    FINAL_AUDIT_RESULT_PATH,
    FINAL_CHECKPOINT_PATH,
    PREFLIGHT_RESULT_PATH,
    RETENTION_POOL_PATH,
)

SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
from v0_8_2.model import build_model


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


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    for key in ("model_state_dict", "model_state", "model", "state_dict"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    raise RuntimeError("unsupported checkpoint")


def summarize(rows):
    margins = [row["margin"] for row in rows]
    return {
        "n": len(rows),
        "answer_accuracy": sum(row["answer_correct"] for row in rows) / len(rows),
        "target_gt_distractor_rate": sum(row["margin"] > 0 for row in rows) / len(rows),
        "mean_margin": statistics.mean(margins),
        "median_margin": statistics.median(margins),
        "mean_candidate_mass": statistics.mean(row["candidate_mass"] for row in rows),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    require(torch.cuda.is_available() if args.device == "cuda" else True, "GPU unavailable")
    device = torch.device(args.device)

    preflight = read_json(PREFLIGHT_RESULT_PATH)
    require(sha256_file(RETENTION_POOL_PATH) == preflight["retention_pool_sha256"],
            "retention pool hash")
    require(FINAL_CHECKPOINT_PATH.is_file(), "missing final checkpoint")
    pool = read_json(RETENTION_POOL_PATH)
    docs = [doc for quartet in pool["quartets"] for doc in quartet["docs"]]

    model = build_model("untied").to(device)
    model.load_state_dict(extract_state_dict(
        torch.load(FINAL_CHECKPOINT_PATH, map_location=device)))
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(docs), args.batch_size):
            chunk = docs[start:start + args.batch_size]
            input_ids = torch.tensor(
                [doc["full_document_token_ids"] for doc in chunk],
                dtype=torch.long, device=device)
            output = model(input_ids)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            require(int(logits.shape[-1]) == EXPECTED_VOCAB_SIZE, "vocab dimension")
            for row_index, doc in enumerate(chunk):
                position = int(doc["answer_token_index"]) - 1
                vector = logits[row_index, position, :].float()
                target = int(doc["target_value_token"])
                distractor = int(doc["distractor_value_token"])
                target_logit = float(vector[target])
                distractor_logit = float(vector[distractor])
                probs = torch.softmax(vector, dim=-1)
                preferred = target if target_logit > distractor_logit else distractor
                rows.append({
                    "doc_id": doc["doc_id"],
                    "quartet_id": doc["quartet_id"],
                    "member": doc["member"],
                    "query_key_token": doc["query_key_token"],
                    "target_value_token": target,
                    "distractor_value_token": distractor,
                    "orientation": doc["orientation"],
                    "query_slot": doc["query_slot"],
                    "answer_correct": int(vector.argmax().item()) == target,
                    "margin": target_logit - distractor_logit,
                    "candidate_mass": float(probs[target] + probs[distractor]),
                    "preferred_candidate": preferred,
                })

    by_quartet = defaultdict(list)
    by_slot = defaultdict(list)
    by_orientation = defaultdict(list)
    for row in rows:
        by_quartet[row["quartet_id"]].append(row)
        by_slot[row["query_slot"]].append(row)
        by_orientation[row["orientation"]].append(row)
    quartet_success = sum(len(group) == 4 and all(r["answer_correct"] for r in group)
                          for group in by_quartet.values())

    reversal_pairs = 0
    preference_changed = 0
    follows_reversal = 0
    for group in by_quartet.values():
        by_member = {row["member"]: row for row in group}
        for key in ("k0", "k1"):
            o1 = by_member[f"o1_{key}"]
            o2 = by_member[f"o2_{key}"]
            reversal_pairs += 1
            changed = o1["preferred_candidate"] != o2["preferred_candidate"]
            preference_changed += changed
            follows_reversal += changed and o1["margin"] > 0 and o2["margin"] > 0

    result = {
        "status": "TREATMENT11_FINAL_RETENTION_AUDIT",
        "checkpoint_sha256": sha256_file(FINAL_CHECKPOINT_PATH),
        "retention_documents": len(rows),
        "retention_quartets": len(by_quartet),
        "summary": summarize(rows),
        "complete_quartet_success_count": quartet_success,
        "complete_quartet_success_rate": quartet_success / len(by_quartet),
        "per_query_slot": {str(k): summarize(v) for k, v in sorted(by_slot.items())},
        "per_reversal_orientation": {
            str(k): summarize(v) for k, v in sorted(by_orientation.items())},
        "reversal_following": {
            "source_orientation_pairs": reversal_pairs,
            "preferred_candidate_changed_count": preference_changed,
            "preferred_candidate_changed_rate": preference_changed / reversal_pairs,
            "changed_and_followed_local_mapping_count": follows_reversal,
            "changed_and_followed_local_mapping_rate": follows_reversal / reversal_pairs,
        },
        "per_document_rows": rows,
        "interpretation": (
            "Strong held-out T11 performance supports: The existing Baby architecture can learn "
            "strict document-local binding when optimization directly targets the answer decision. "
            "Weak/chance performance supports moving forward architecturally because both "
            "counterfactual data and direct answer-decision supervision were insufficient."
        ),
    }
    FINAL_AUDIT_RESULT_PATH.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print({k: v for k, v in result.items() if k not in ("per_document_rows", "interpretation")})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"AUDIT ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
