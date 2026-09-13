r"""Treatment-10 trainer.

Loads the authoritative common pre-treatment Baby checkpoint (frozen one-map
baseline), then trains on the NEW Treatment-10 strict counterfactual quartet
schedule with an ordinary autoregressive full-vocabulary causal CE objective.
No architectural modification is made.

Not executed during implementation review; run with:
    py -3.12 .\treatment10_train.py --device cuda
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from treatment10_config import (
    BATCH_SIZE,
    FINAL_CHECKPOINT_PATH,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    MAX_STEPS,
    PREFLIGHT_RESULT_PATH,
    SCHEDULE_PATH,
    SEED_CHECKPOINT_ROOT,
    START_CHECKPOINT_PATH,
    START_CHECKPOINT_SHA256,
    TRAIN_POOL_PATH,
    TRAINING_METRICS_PATH,
    TRAINING_RESULT_PATH,
    WEIGHT_DECAY,
)
from treatment10_common import Treatment10Error, load_json, require, sha256_file, write_json

SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from v0_8_2.model import build_model  # trusted frozen model builder


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)


def resolve_vocab_size(model: nn.Module) -> int:
    if hasattr(model, "token_embedding") and hasattr(model.token_embedding, "num_embeddings"):
        return int(model.token_embedding.num_embeddings)
    return 1024


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model_state", "model", "state_dict"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise Treatment10Error("Unsupported checkpoint container.")


def load_start_checkpoint(model: nn.Module, device: torch.device) -> Dict[str, str]:
    require(Path(START_CHECKPOINT_PATH).is_file(), "Missing start checkpoint.")
    observed = sha256_file(Path(START_CHECKPOINT_PATH))
    require(observed == START_CHECKPOINT_SHA256,
            f"Start checkpoint SHA256 mismatch: {observed}")
    state = torch.load(START_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(extract_state_dict(state))
    return {"expected": START_CHECKPOINT_SHA256, "observed": observed}


def _run_smoke(model, optimizer, device, schedule, docs_by_quartet) -> Dict[str, Any]:
    """Non-authoritative single-step smoke test. Writes nothing."""
    step = schedule["steps_data"][0]
    docs = []
    for quartet_id in step["quartet_ids"]:
        docs.extend(docs_by_quartet[quartet_id])
    require(len(docs) == BATCH_SIZE, "step doc count != batch size")
    seq_lengths = {len(doc["full_document_token_ids"]) for doc in docs}
    require(len(seq_lengths) == 1, "inconsistent document lengths in batch")

    input_ids = torch.tensor(
        [doc["full_document_token_ids"] for doc in docs], dtype=torch.long, device=device)

    optimizer.zero_grad()
    out = model(input_ids)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    require(int(logits.shape[-1]) == resolve_vocab_size(model),
            "logits vocab dimension != resolved vocab size")
    targets = input_ids[:, 1:].contiguous()
    logits_shifted = logits[:, :-1, :].contiguous()
    require(logits_shifted.shape[0] == BATCH_SIZE, "batch dim mismatch")
    require(logits_shifted.shape[1] == targets.shape[1], "sequence dim mismatch")

    loss = nn.functional.cross_entropy(
        logits_shifted.reshape(-1, logits_shifted.shape[-1]),
        targets.reshape(-1),
    )
    loss_finite = bool(torch.isfinite(loss).item())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
    optimizer.step()

    grads_exist = any(p.grad is not None for p in model.parameters())
    grad_finite = bool(all(
        torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None))
    param_finite = bool(all(
        torch.isfinite(p).all().item() for p in model.parameters()))

    cuda_available = torch.cuda.is_available()
    return {
        "start_checkpoint_sha256_match": True,
        "device": "cuda" if cuda_available else "cpu",
        "cuda_available": cuda_available,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        "model_loaded_from_start_checkpoint": True,
        "batch_size": BATCH_SIZE,
        "sequence_length": next(iter(seq_lengths)),
        "input_ids_shape": list(input_ids.shape),
        "logits_shape": list(logits.shape),
        "targets_shape": list(targets.shape),
        "loss": float(loss.item()),
        "loss_finite": loss_finite,
        "backward_ok": True,
        "optimizer_step_ok": True,
        "gradients_exist": grads_exist,
        "gradients_finite": grad_finite,
        "parameters_finite": param_finite,
        "wrote_authoritative_outputs": False,
    }


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a single non-authoritative optimizer step and stop without writing outputs.")
    args = parser.parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    preflight = load_json(Path(PREFLIGHT_RESULT_PATH))
    train_pool = load_json(Path(TRAIN_POOL_PATH))
    schedule = load_json(Path(SCHEDULE_PATH))

    require(preflight["status"] == "TREATMENT10_PREFLIGHT_PASS", "preflight not passed.")

    require(sha256_file(Path(TRAIN_POOL_PATH)) == preflight["train_pool_sha256"],
            "train pool hash mismatch vs preflight.")
    require(sha256_file(Path(SCHEDULE_PATH)) == preflight["schedule_sha256"],
            "schedule hash mismatch vs preflight.")

    docs_by_quartet = {}
    for quartet in train_pool["quartets"]:
        docs_by_quartet[quartet["quartet_id"]] = quartet["docs"]

    set_seed(8380)
    model = build_model("untied")
    model.to(device)
    model.train()
    load_start_checkpoint(model, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    if args.smoke_test:
        facts = _run_smoke(model, optimizer, device, schedule, docs_by_quartet)
        for key, value in facts.items():
            print(f"SMOKE {key}: {value}", flush=True)
        return 0

    total_tokens = 0
    metrics_lines = []
    start_time = time.time()
    step_metrics = []

    vocab_size = resolve_vocab_size(model)

    for step_index, step in enumerate(schedule["steps_data"]):
        docs = []
        for quartet_id in step["quartet_ids"]:
            docs.extend(docs_by_quartet[quartet_id])
        require(len(docs) == BATCH_SIZE, "step doc count != batch size")
        input_ids = torch.tensor(
            [doc["full_document_token_ids"] for doc in docs], dtype=torch.long, device=device)

        optimizer.zero_grad()
        out = model(input_ids)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        require(int(logits.shape[-1]) == vocab_size,
                "logits vocab dimension != resolved vocab size")
        targets = input_ids[:, 1:].contiguous()
        logits_shifted = logits[:, :-1, :].contiguous()
        loss = nn.functional.cross_entropy(
            logits_shifted.reshape(-1, logits_shifted.shape[-1]),
            targets.reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()

        tokens = targets.numel()
        total_tokens += tokens
        correct = (logits_shifted.argmax(-1) == targets).sum().item()
        acc = correct / tokens
        record = {
            "step": step_index + 1,
            "loss": float(loss.detach()),
            "token_accuracy": acc,
            "supervised_tokens": tokens,
            "elapsed_seconds": time.time() - start_time,
        }
        step_metrics.append(record)
        if (step_index + 1) % 50 == 0:
            print(record, flush=True)

    metrics_path = Path(TRAINING_METRICS_PATH)
    with metrics_path.open("w", encoding="utf-8") as handle:
        for rec in step_metrics:
            handle.write(__import__("json").dumps(rec) + "\n")

    SEED_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "step": MAX_STEPS,
                "seed": 8380,
                "objective": "ordinary_autoregressive_causal_ce"},
               FINAL_CHECKPOINT_PATH)

    result = {
        "status": "TRAINED",
        "steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "final_checkpoint_sha256": sha256_file(Path(FINAL_CHECKPOINT_PATH)),
        "total_supervised_tokens": total_tokens,
        "final_step": step_metrics[-1],
        "average_loss": sum(r["loss"] for r in step_metrics) / len(step_metrics),
        "average_token_accuracy": sum(r["token_accuracy"] for r in step_metrics) / len(step_metrics),
        "note": "Ordinary autoregressive objective; no architectural change.",
    }
    write_json(Path(TRAINING_RESULT_PATH), result)
    print({k: v for k, v in result.items() if k != "note"})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Treatment10Error as exc:
        print(f"TRAIN ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
