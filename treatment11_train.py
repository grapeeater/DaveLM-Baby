r"""Treatment-11 answer-only strict-counterfactual trainer.

The sole intervention from T10 is the objective: one causal CE decision at the
registered answer prediction position per document. No other token contributes
to loss. Architecture, baseline checkpoint, optimizer, LR, batch size, steps,
seed, and complete-quartet scheduling remain unchanged.
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
import torch.nn as nn

from treatment11_config import (
    BATCH_SIZE,
    CHECKPOINT_ROOT,
    EXPECTED_VOCAB_SIZE,
    FINAL_CHECKPOINT_PATH,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    MAX_STEPS,
    PREFLIGHT_RESULT_PATH,
    SCHEDULE_PATH,
    SEED,
    START_CHECKPOINT_PATH,
    START_CHECKPOINT_SHA256,
    TRAINING_METRICS_PATH,
    TRAINING_RESULT_PATH,
    TRAIN_POOL_PATH,
    WEIGHT_DECAY,
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


def resolve_vocab_size(model: nn.Module) -> int:
    if hasattr(model, "token_embedding") and hasattr(model.token_embedding, "num_embeddings"):
        return int(model.token_embedding.num_embeddings)
    return EXPECTED_VOCAB_SIZE


def load_baseline(model: nn.Module, device: torch.device) -> str:
    observed = sha256_file(START_CHECKPOINT_PATH)
    require(observed == START_CHECKPOINT_SHA256, "baseline checkpoint hash mismatch")
    checkpoint = torch.load(START_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint))
    return observed


def build_batch(step, docs_by_quartet, device):
    docs = []
    for quartet_id in step["quartet_ids"]:
        docs.extend(docs_by_quartet[quartet_id])
    require(len(docs) == BATCH_SIZE, "scheduled document count != batch size")
    input_ids = torch.tensor(
        [doc["full_document_token_ids"] for doc in docs], dtype=torch.long, device=device)
    positions = torch.tensor(
        [int(doc["answer_token_index"]) - 1 for doc in docs], dtype=torch.long, device=device)
    targets = torch.tensor(
        [int(doc["target_value_token"]) for doc in docs], dtype=torch.long, device=device)
    distractors = torch.tensor(
        [int(doc["distractor_value_token"]) for doc in docs], dtype=torch.long, device=device)
    require(input_ids.shape == (BATCH_SIZE, 193), "input shape mismatch")
    require(positions.numel() == targets.numel() == distractors.numel() == BATCH_SIZE,
            "answer-decision count != batch size")
    return docs, input_ids, positions, targets, distractors


def train_step(model, optimizer, batch):
    docs, input_ids, positions, targets, distractors = batch
    optimizer.zero_grad()
    out = model(input_ids)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    require(logits.shape[:2] == input_ids.shape, "logit batch/sequence shape mismatch")
    vocab_size = resolve_vocab_size(model)
    require(int(logits.shape[-1]) == vocab_size == EXPECTED_VOCAB_SIZE,
            "logit vocabulary dimension mismatch")
    rows = torch.arange(BATCH_SIZE, device=input_ids.device)
    selected_logits = logits[rows, positions, :]
    require(selected_logits.shape == (BATCH_SIZE, vocab_size),
            "selected answer-logit tensor shape mismatch")
    require(targets.shape == (BATCH_SIZE,), "selected answer-target shape mismatch")
    require(selected_logits.shape[0] == targets.numel() == BATCH_SIZE,
            "selected answer-decision count mismatch")

    loss = nn.functional.cross_entropy(selected_logits, targets)
    require(bool(torch.isfinite(loss).item()), "non-finite answer CE")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
    require(bool(torch.isfinite(grad_norm).item()), "non-finite gradient norm")
    require(all(torch.isfinite(p.grad).all().item()
                for p in model.parameters() if p.grad is not None), "non-finite gradient")
    optimizer.step()
    require(all(torch.isfinite(p).all().item() for p in model.parameters()),
            "non-finite parameter after optimizer step")

    target_logits = selected_logits[rows, targets]
    distractor_logits = selected_logits[rows, distractors]
    margins = target_logits - distractor_logits
    return {
        "answer_ce_loss": float(loss.detach().item()),
        "answer_accuracy": float((selected_logits.argmax(-1) == targets).float().mean().item()),
        "target_gt_distractor_rate": float((margins > 0).float().mean().item()),
        "mean_target_distractor_margin": float(margins.mean().detach().item()),
        "supervised_answer_decisions": int(targets.numel()),
        "selected_logits_shape": list(selected_logits.shape),
        "target_shape": list(targets.shape),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)

    require(torch.cuda.is_available() if args.device == "cuda" else True,
            "requested GPU backend unavailable")
    device = torch.device(args.device)
    preflight = read_json(PREFLIGHT_RESULT_PATH)
    require(preflight["status"] == "TREATMENT11_PREFLIGHT_PASS", "preflight not passed")
    require(sha256_file(TRAIN_POOL_PATH) == preflight["train_pool_sha256"], "train pool hash")
    require(sha256_file(SCHEDULE_PATH) == preflight["schedule_sha256"], "schedule hash")
    train_pool = read_json(TRAIN_POOL_PATH)
    schedule = read_json(SCHEDULE_PATH)
    docs_by_quartet = {q["quartet_id"]: q["docs"] for q in train_pool["quartets"]}

    random.seed(SEED)
    torch.manual_seed(SEED)
    model = build_model("untied").to(device)
    model.train()
    baseline_hash = load_baseline(model, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if args.smoke_test:
        record = train_step(
            model, optimizer,
            build_batch(schedule["steps_data"][0], docs_by_quartet, device),
        )
        print({
            "status": "TREATMENT11_SMOKE_PASS",
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0),
            "baseline_checkpoint_sha256": baseline_hash,
            "batch_size": BATCH_SIZE,
            "document_shape": [BATCH_SIZE, 193],
            "metrics": record,
            "authoritative_outputs_written": False,
        })
        return 0

    start = time.time()
    metrics = []
    for index, step in enumerate(schedule["steps_data"]):
        record = train_step(model, optimizer, build_batch(step, docs_by_quartet, device))
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
        "objective": "answer_only_causal_cross_entropy",
        "supervised_answer_decisions_per_step": BATCH_SIZE,
    }, FINAL_CHECKPOINT_PATH)
    result = {
        "status": "TREATMENT11_TRAINED",
        "steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "objective": "answer_only_causal_cross_entropy",
        "supervised_answer_decisions": MAX_STEPS * BATCH_SIZE,
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
