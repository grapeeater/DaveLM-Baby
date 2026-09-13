r"""Treatment-10 final retention audit.

Read-only inference audit of the trained ordinary-Baby checkpoint on the NEW
held-out strict counterfactual retention pool. Reports answer accuracy,
target>distractor rates, margins, quartet success, per-slot and per-orientation
results, candidate mass, and schedule/exposure balance. No training, no
positive controls, no sealed evaluation.

Not executed during implementation review; run with:
    py -3.12 .\treatment10_final_retention_audit.py --device cuda
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from treatment10_config import (
    FINAL_AUDIT_RESULT_PATH,
    FINAL_CHECKPOINT_PATH,
    RETENTION_POOL_PATH,
)
from treatment10_common import Treatment10Error, load_json, require, sha256_file, write_json

SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from v0_8_2.model import build_model  # trusted frozen model builder


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model_state", "model", "state_dict"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise Treatment10Error("Unsupported checkpoint container.")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    require(Path(FINAL_CHECKPOINT_PATH).is_file(), "Missing final checkpoint.")
    retention_pool = load_json(Path(RETENTION_POOL_PATH))
    docs = []
    for quartet in retention_pool["quartets"]:
        docs.extend(quartet["docs"])

    model = build_model("untied")
    model.to(device)
    model.eval()
    state = torch.load(FINAL_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(extract_state_dict(state))

    rows = []
    with torch.no_grad():
        for start in range(0, len(docs), args.batch_size):
            chunk = docs[start:start + args.batch_size]
            input_ids = torch.tensor(
                [doc["full_document_token_ids"] for doc in chunk], dtype=torch.long, device=device)
            out = model(input_ids)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            logits = logits.float()
            for doc, ids in zip(chunk, input_ids.tolist()):
                pos = doc["answer_token_index"] - 1  # causal position predicting the answer token
                logit_vector = logits[chunk.index(doc), pos, :]
                target = doc["target_value_token"]
                distractor = doc["distractor_value_token"]
                target_logit = float(logit_vector[target])
                distractor_logit = float(logit_vector[distractor])
                correct = target_logit > distractor_logit
                margin = target_logit - distractor_logit
                probs = torch.softmax(logit_vector, dim=-1)
                candidate_mass = float(probs[target] + probs[distractor])
                rows.append({
                    "doc_id": doc["doc_id"],
                    "quartet_id": doc["quartet_id"],
                    "member": doc["member"],
                    "orientation": doc["orientation"],
                    "query_slot": doc["query_slot"],
                    "correct": bool(correct),
                    "target_logit": target_logit,
                    "distractor_logit": distractor_logit,
                    "margin": margin,
                    "candidate_mass": candidate_mass,
                })

    n = len(rows)
    correct_n = sum(1 for r in rows if r["correct"])
    per_slot = {}
    per_orientation = {}
    for r in rows:
        per_slot.setdefault(r["query_slot"], []).append(r)
        per_orientation.setdefault(r["orientation"], []).append(r)

    def summarize(subrows: List[Dict[str, Any]]) -> Dict[str, Any]:
        acc = sum(1 for r in subrows if r["correct"]) / len(subrows)
        margins = [r["margin"] for r in subrows]
        mass = [r["candidate_mass"] for r in subrows]
        return {
            "n": len(subrows),
            "answer_accuracy": acc,
            "target_gt_distractor_rate": acc,
            "mean_margin": sum(margins) / len(margins),
            "median_margin": sorted(margins)[len(margins) // 2],
            "mean_candidate_mass": sum(mass) / len(mass),
        }

    # Quartet success: all four members correct.
    by_quartet: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_quartet.setdefault(r["quartet_id"], []).append(r)
    quartet_ok = sum(1 for q, qrows in by_quartet.items() if len(qrows) == 4 and all(x["correct"] for x in qrows))

    result = {
        "status": "FINAL_RETENTION_AUDIT",
        "final_checkpoint_sha256": sha256_file(Path(FINAL_CHECKPOINT_PATH)),
        "retention_documents": n,
        "retention_quartets": len(by_quartet),
        "summary": summarize(rows),
        "complete_quartet_success_rate": quartet_ok / len(by_quartet),
        "complete_quartet_success_count": quartet_ok,
        "per_query_slot": {str(k): summarize(v) for k, v in sorted(per_slot.items())},
        "per_reversal_orientation": {str(k): summarize(v) for k, v in sorted(per_orientation.items())},
        "per_document_rows": rows,
        "interpretation": (
            "Strong result: the existing Baby architecture can learn document-local binding when "
            "the training distribution explicitly requires counterfactual binding discrimination. "
            "It does not establish a universal symbolic algorithm. Weak result: supports moving "
            "forward to an architectural Treatment 11."
        ),
        "note": "Outcome-blind structural universe; retention examples are new base records with "
                "strict counterfactual reversals.",
    }
    write_json(Path(FINAL_AUDIT_RESULT_PATH), result)
    print({k: v for k, v in result.items() if k not in ("per_document_rows", "interpretation")})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Treatment10Error as exc:
        print(f"AUDIT ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
