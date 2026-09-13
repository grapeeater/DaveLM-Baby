r"""Treatment-12 trainer: Baby + query-conditioned mapping retrieval.

Architecture-only comparison vs T11. Same baseline checkpoint, same T11
pools/schedule, same answer-only causal CE objective, same optimizer/LR/batch/
steps/seed. Only the forward model gains the small retrieval module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from treatment12_config import (
    BATCH_SIZE,
    CHECKPOINT_ROOT,
    DOCUMENT_LENGTH,
    EXPECTED_SUPERVISED_ANSWER_DECISIONS,
    FINAL_CHECKPOINT_PATH,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    MAX_STEPS,
    NAME,
    PREFLIGHT_RESULT_PATH,
    SEED,
    START_CHECKPOINT_PATH,
    START_CHECKPOINT_SHA256,
    TRAINING_METRICS_PATH,
    TRAINING_RESULT_PATH,
    TRAIN_POOL_PATH,
    VOCAB_SIZE,
    WEIGHT_DECAY,
    SCHEDULE_PATH,
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


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model_state", "model", "state_dict"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
        if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise RuntimeError("Unsupported checkpoint container")


def load_baseline(model: Treatment12Model, device: torch.device) -> str:
    observed = sha256_file(START_CHECKPOINT_PATH)
    require(observed == START_CHECKPOINT_SHA256, "baseline checkpoint hash mismatch")
    checkpoint = torch.load(START_CHECKPOINT_PATH, map_location=device)
    model.base_model.load_state_dict(extract_state_dict(checkpoint))
    return observed


def build_batch(step, docs_by_quartet, device):
    docs = []
    for quartet_id in step["quartet_ids"]:
        docs.extend(docs_by_quartet[quartet_id])
    require(len(docs) == BATCH_SIZE, "scheduled document count != batch size")
    metas = [doc_retrieval_meta(doc) for doc in docs]
    input_ids = torch.tensor(
        [doc["full_document_token_ids"] for doc in docs], dtype=torch.long, device=device)
    positions = {
        "q": torch.tensor([meta["qdp"] for meta in metas], dtype=torch.long, device=device),
        "k0": torch.tensor([meta["row0_src_pos"] for meta in metas], dtype=torch.long, device=device),
        "k1": torch.tensor([meta["row1_src_pos"] for meta in metas], dtype=torch.long, device=device),
        "v0": torch.tensor([meta["row0_val_pos"] for meta in metas], dtype=torch.long, device=device),
        "v1": torch.tensor([meta["row1_val_pos"] for meta in metas], dtype=torch.long, device=device),
        "answer": torch.tensor([meta["answer_causal_pos"] for meta in metas],
                               dtype=torch.long, device=device),
    }
    targets = torch.tensor(
        [int(doc["target_value_token"]) for doc in docs], dtype=torch.long, device=device)
    distractors = torch.tensor(
        [int(doc["distractor_value_token"]) for doc in docs], dtype=torch.long, device=device)
    require(input_ids.shape == (BATCH_SIZE, DOCUMENT_LENGTH), "input shape mismatch")
    for key, tensor in positions.items():
        require(tensor.shape == (BATCH_SIZE,), f"position {key} shape mismatch")
    require(targets.shape == (BATCH_SIZE,) and distractors.shape == (BATCH_SIZE,),
            "target/distractor shape mismatch")
    return docs, metas, input_ids, positions, targets, distractors


def compute_retrieval_metrics(extras, metas, device) -> Dict[str, Any]:
    weights = extras["retrieval_weights"]
    scores = extras["retrieval_scores"]
    rows = torch.arange(weights.shape[0], device=device)
    correct = torch.tensor([correct_row_index(meta) for meta in metas],
                           dtype=torch.long, device=device)
    incorrect = 1 - correct
    argmax_matches = (weights.argmax(-1) == correct).float().mean().item()
    mean_correct = weights[rows, correct].mean().item()
    mean_incorrect = weights[rows, incorrect].mean().item()
    margin = scores[rows, correct] - scores[rows, incorrect]
    return {
        "retrieval_argmax_matches_correct_row_rate": float(argmax_matches),
        "mean_correct_row_attention_weight": float(mean_correct),
        "mean_incorrect_row_attention_weight": float(mean_incorrect),
        "mean_retrieval_score_margin": float(margin.mean().item()),
    }


def train_step(model, optimizer, batch):
    docs, metas, input_ids, positions, targets, distractors = batch
    optimizer.zero_grad()
    out_logits, extras = model(input_ids, positions)
    require(out_logits.shape == (BATCH_SIZE, DOCUMENT_LENGTH, VOCAB_SIZE),
            "full logit shape mismatch")
    rows = torch.arange(BATCH_SIZE, device=input_ids.device)
    answer_logits = out_logits[rows, positions["answer"].long(), :]
    require(answer_logits.shape == (BATCH_SIZE, VOCAB_SIZE), "answer logit shape mismatch")
    require(targets.shape == (BATCH_SIZE,), "target shape mismatch")
    loss = torch.nn.functional.cross_entropy(answer_logits, targets)
    require(bool(torch.isfinite(loss).item()), "non-finite answer CE")

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
    require(bool(torch.isfinite(grad_norm).item()), "non-finite gradient norm")
    require(all(torch.isfinite(p.grad).all().item()
                for p in model.parameters() if p.grad is not None), "non-finite gradient")
    optimizer.step()
    require(all(torch.isfinite(p).all().item() for p in model.parameters()),
            "non-finite parameter after optimizer step")

    target_logits = answer_logits[rows, targets]
    distractor_logits = answer_logits[rows, distractors]
    margins = target_logits - distractor_logits
    metrics = compute_retrieval_metrics(extras, metas, input_ids.device)
    metrics.update({
        "answer_ce_loss": float(loss.item()),
        "answer_accuracy": float((answer_logits.argmax(-1) == targets).float().mean().item()),
        "target_gt_distractor_rate": float((margins > 0).float().mean().item()),
        "mean_target_distractor_margin": float(margins.mean().item()),
        "supervised_answer_decisions": int(targets.numel()),
        "selected_logits_shape": list(answer_logits.shape),
        "target_shape": list(targets.shape),
    })
    return metrics


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    require(torch.cuda.is_available() if args.device == "cuda" else True,
            "requested GPU backend unavailable")
    device = torch.device(args.device)

    preflight = read_json(PREFLIGHT_RESULT_PATH)
    require(preflight["status"] == "TREATMENT12_PREFLIGHT_PASS", "preflight not passed")
    require(sha256_file(TRAIN_POOL_PATH) == preflight["reused_universe"]["train_pool_sha256"],
            "train pool hash")
    require(sha256_file(SCHEDULE_PATH) == preflight["reused_universe"]["schedule_sha256"],
            "schedule hash")
    train_pool = read_json(TRAIN_POOL_PATH)
    schedule = read_json(SCHEDULE_PATH)
    docs_by_quartet = {q["quartet_id"]: q["docs"] for q in train_pool["quartets"]}

    random.seed(SEED)
    torch.manual_seed(SEED)
    model = Treatment12Model()
    model.to(device)
    model.train()
    baseline_hash = load_baseline(model, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if args.smoke_test:
        batch = build_batch(schedule["steps_data"][0], docs_by_quartet, device)
        metrics = train_step(model, optimizer, batch)
        print({
            "status": "TREATMENT12_SMOKE_PASS",
            "experiment": NAME,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0),
            "baseline_checkpoint_sha256": baseline_hash,
            "batch_size": BATCH_SIZE,
            "document_shape": list(batch[2].shape),
            "metrics": metrics,
            "authoritative_outputs_written": False,
        })
        return 0

    start = time.time()
    metrics = []
    for index, step in enumerate(schedule["steps_data"]):
        batch = build_batch(step, docs_by_quartet, device)
        record = train_step(model, optimizer, batch)
        record["step"] = index + 1
        record["elapsed_seconds"] = time.time() - start
        metrics.append(record)
        if (index + 1) % 50 == 0:
            print(record, flush=True)

    TRAINING_METRICS_PATH.write_text(
        "".join(json.dumps(row) + "\n" for row in metrics), encoding="utf-8")
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "step": MAX_STEPS,
        "seed": SEED,
        "objective": "answer_only_causal_cross_entropy_with_query_conditioned_mapping_retrieval",
        "supervised_answer_decisions_per_step": BATCH_SIZE,
    }, FINAL_CHECKPOINT_PATH)
    result = {
        "status": "TREATMENT12_TRAINED",
        "steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "objective": "answer_only_causal_cross_entropy_with_query_conditioned_mapping_retrieval",
        "supervised_answer_decisions": EXPECTED_SUPERVISED_ANSWER_DECISIONS,
        "final_checkpoint_sha256": sha256_file(FINAL_CHECKPOINT_PATH),
        "final_step": metrics[-1],
    }
    write_json(TRAINING_RESULT_PATH, result)
    print(result)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"TRAIN ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
