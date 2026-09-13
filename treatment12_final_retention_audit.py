r"""Treatment-12 final retention audit (read-only).

Evaluates the SAME 160 frozen retention documents / 40 strict counterfactual
quartets used by T11, on the T12 retrieval-augmented model. Reports answer and
retrieval diagnostics split by query slot and reversal orientation.
"""

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

from treatment12_config import (
    DOCUMENT_LENGTH,
    FINAL_AUDIT_RESULT_PATH,
    FINAL_CHECKPOINT_PATH,
    PREFLIGHT_RESULT_PATH,
    RETENTION_POOL_PATH,
    VOCAB_SIZE,
)
from treatment12_model import (
    Treatment12Model,
    correct_row_index,
    doc_retrieval_meta,
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


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    for key in ("model_state_dict", "model_state", "model", "state_dict"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    raise RuntimeError("unsupported checkpoint")


def summarize(rows):
    margins = [row["margin"] for row in rows]
    n = len(rows)
    return {
        "n": n,
        "answer_accuracy": sum(row["answer_correct"] for row in rows) / n,
        "target_gt_distractor_rate": sum(row["margin"] > 0 for row in rows) / n,
        "mean_margin": statistics.mean(margins),
        "median_margin": statistics.median(margins),
        "mean_candidate_mass": statistics.mean(row["candidate_mass"] for row in rows),
        "retrieval_argmax_matches_correct_row_rate": statistics.mean(
            row["retrieval_correct"] for row in rows),
        "mean_correct_row_attention_weight": statistics.mean(
            row["correct_row_weight"] for row in rows),
        "mean_incorrect_row_attention_weight": statistics.mean(
            row["incorrect_row_weight"] for row in rows),
        "mean_retrieval_score_margin": statistics.mean(
            row["retrieval_margin"] for row in rows),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    require(torch.cuda.is_available() if args.device == "cuda" else True, "GPU unavailable")
    device = torch.device(args.device)

    preflight = read_json(PREFLIGHT_RESULT_PATH)
    require(preflight["status"] == "TREATMENT12_PREFLIGHT_PASS", "preflight")
    require(sha256_file(RETENTION_POOL_PATH) == preflight["reused_universe"]["retention_pool_sha256"],
            "retention pool hash")
    require(FINAL_CHECKPOINT_PATH.is_file(), "missing final checkpoint")

    pool = read_json(RETENTION_POOL_PATH)
    docs = [doc for quartet in pool["quartets"] for doc in quartet["docs"]]

    model = Treatment12Model()
    model.to(device)
    checkpoint = torch.load(FINAL_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint))
    model.eval()

    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(docs), args.batch_size):
            chunk = docs[start:start + args.batch_size]
            metas = [doc_retrieval_meta(doc) for doc in chunk]
            input_ids = torch.tensor(
                [doc["full_document_token_ids"] for doc in chunk],
                dtype=torch.long, device=device)
            require(input_ids.shape == (len(chunk), DOCUMENT_LENGTH), "batch input shape")
            positions = {
                "q": torch.tensor([m["qdp"] for m in metas], dtype=torch.long, device=device),
                "k0": torch.tensor([m["row0_src_pos"] for m in metas], dtype=torch.long, device=device),
                "k1": torch.tensor([m["row1_src_pos"] for m in metas], dtype=torch.long, device=device),
                "v0": torch.tensor([m["row0_val_pos"] for m in metas], dtype=torch.long, device=device),
                "v1": torch.tensor([m["row1_val_pos"] for m in metas], dtype=torch.long, device=device),
                "answer": torch.tensor([m["answer_causal_pos"] for m in metas],
                                       dtype=torch.long, device=device),
            }
            out_logits, extras = model(input_ids, positions)
            require(out_logits.shape[1:] == (DOCUMENT_LENGTH, VOCAB_SIZE), "logit shape")
            probs_logits = out_logits.float()
            for i, (doc, meta) in enumerate(zip(chunk, metas)):
                vector = probs_logits[i, meta["answer_causal_pos"], :]
                target = doc["target_value_token"]
                distractor = doc["distractor_value_token"]
                target_logit = float(vector[target])
                distractor_logit = float(vector[distractor])
                probs = torch.softmax(vector, dim=-1)
                correct_row = correct_row_index(meta)
                weights = extras["retrieval_weights"][i]
                scores = extras["retrieval_scores"][i]
                rows.append({
                    "doc_id": doc["doc_id"],
                    "quartet_id": doc["quartet_id"],
                    "member": doc["member"],
                    "query_key_token": doc["query_key_token"],
                    "target_value_token": int(doc["target_value_token"]),
                    "distractor_value_token": int(doc["distractor_value_token"]),
                    "orientation": doc["orientation"],
                    "query_slot": doc["query_slot"],
                    "answer_correct": int(vector.argmax().item()) == int(target),
                    "margin": target_logit - distractor_logit,
                    "candidate_mass": float(probs[int(target)] + probs[int(distractor)]),
                    "retrieval_correct": int(weights.argmax().item()) == correct_row,
                    "correct_row_weight": float(weights[correct_row].item()),
                    "incorrect_row_weight": float(weights[1 - correct_row].item()),
                    "retrieval_margin": float(
                        (scores[correct_row] - scores[1 - correct_row]).item()),
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
    changed = 0
    followed = 0
    for group in by_quartet.values():
        by_member = {row["member"]: row for row in group}
        for key in ("k0", "k1"):
            o1 = by_member[f"o1_{key}"]
            o2 = by_member[f"o2_{key}"]
            reversal_pairs += 1
            pref1 = o1["target_value_token"] if o1["margin"] > 0 else o1["distractor_value_token"]
            pref2 = o2["target_value_token"] if o2["margin"] > 0 else o2["distractor_value_token"]
            row_changed = pref1 != pref2
            changed += row_changed
            followed += row_changed and o1["margin"] > 0 and o2["margin"] > 0

    result = {
        "status": "TREATMENT12_FINAL_RETENTION_AUDIT",
        "checkpoint_sha256": sha256_file(FINAL_CHECKPOINT_PATH),
        "retention_documents": len(rows),
        "retention_quartets": len(by_quartet),
        "overall": summarize(rows),
        "complete_quartet": {
            "success_count": quartet_success,
            "success_rate": quartet_success / len(by_quartet),
        },
        "per_query_slot": {str(k): summarize(v) for k, v in sorted(by_slot.items())},
        "per_reversal_orientation": {
            str(o): summarize(v) for o, v in sorted(by_orientation.items())},
        "reversal_following": {
            "source_orientation_pairs": reversal_pairs,
            "preferred_candidate_changed_count": changed,
            "preferred_candidate_changed_rate": changed / reversal_pairs,
            "changed_and_followed_local_mapping_count": followed,
            "changed_and_followed_local_mapping_rate": followed / reversal_pairs,
        },
        "retrieval_diagnostics_by_query_slot": {
            str(k): {
                "retrieval_argmax_matches_correct_row_rate": statistics.mean(
                    row["retrieval_correct"] for row in v),
                "mean_correct_row_attention_weight": statistics.mean(
                    row["correct_row_weight"] for row in v),
                "mean_incorrect_row_attention_weight": statistics.mean(
                    row["incorrect_row_weight"] for row in v),
                "mean_retrieval_score_margin": statistics.mean(
                    row["retrieval_margin"] for row in v),
            } for k, v in sorted(by_slot.items())
        },
        "retrieval_diagnostics_by_reversal_orientation": {
            str(o): {
                "retrieval_argmax_matches_correct_row_rate": statistics.mean(
                    row["retrieval_correct"] for row in v),
                "mean_correct_row_attention_weight": statistics.mean(
                    row["correct_row_weight"] for row in v),
                "mean_incorrect_row_attention_weight": statistics.mean(
                    row["incorrect_row_weight"] for row in v),
                "mean_retrieval_score_margin": statistics.mean(
                    row["retrieval_margin"] for row in v),
            } for o, v in sorted(by_orientation.items())
        },
        "per_document_rows": rows,
        "mechanistic_interpretation_note": (
            "If answer performance rises but retrieval-row accuracy stays at chance, flag the "
            "result as mechanistically ambiguous. If retrieval-row accuracy rises strongly but "
            "answer performance does not, report that key-match learned while value transfer/"
            "readout remained insufficient. Candidate mass alone is not binding."
        ),
    }
    FINAL_AUDIT_RESULT_PATH.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print({k: v for k, v in result.items()
           if k not in ("per_document_rows", "mechanistic_interpretation_note")})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"AUDIT ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
