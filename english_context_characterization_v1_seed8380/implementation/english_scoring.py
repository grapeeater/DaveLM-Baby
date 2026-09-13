"""Frozen English candidate scoring; importing this module performs no inference.

Only ``model.base_model(input_ids)`` is used for English inference. The caller
must independently verify checkpoint hashes, construct/load the reviewed model,
and pass the frozen materialized token arrays. This module does not load models,
create optimizers, generate text, calibrate scores, or aggregate study endpoints.

Static source verification establishes the intended call path and causal shift;
it does not establish successful execution or checkpoint behavior. No forward
pass is needed to import or inspect this file. The inherited T13 final-norm hook
may capture a tensor but returns None and does not modify base-model outputs.
"""

from __future__ import annotations

import math


BOS_TOKEN_ID = 2
PERIOD_TOKEN_ID = 18
VOCAB_SIZE = 1024
CONTEXT_SIZE = 256
CANDIDATE_TOKEN_COUNT = 3
WORD_TOKEN_COUNT = 2


def configure_runtime(device: str) -> dict:
    """Configure the common FP32 runtime, only when explicitly invoked later.

    ``device`` must be identical for the three checkpoints. Require an explicit
    CUDA index, if using CUDA. We disable TF32, reduced-precision matmul, and
    cuDNN autotuning; no model is accessed by this function. The protocol does
    not claim bitwise identity across distinct hardware/library installations.
    """
    import torch

    requested_device = torch.device(device)
    if requested_device.type not in {"cpu", "cuda"}:
        raise ValueError("The frozen evaluator supports one CPU or CUDA device.")
    if requested_device.type == "cuda" and requested_device.index is None:
        raise ValueError("Use an explicit CUDA device index, such as cuda:0.")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "device": str(requested_device),
        "torch_version": str(torch.__version__),
        "model_dtype": "float32",
        "log_softmax_and_sum_dtype": "float64",
        "autocast": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "sequence_batch_size": 1,
        "padding": False,
        "truncation": False,
        "english_forward_path": "model.base_model(input_ids)",
    }


def _validate_token_array(tokens, label: str) -> tuple[int, ...]:
    if not isinstance(tokens, (list, tuple)) or not tokens:
        raise ValueError(f"{label} must be a nonempty token list or tuple.")
    if any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in tokens):
        raise ValueError(f"{label} must contain integer vocabulary IDs only.")
    return tuple(tokens)


def score_candidate_pair(
    model,
    prompt_token_ids,
    candidate_token_ids,
    *,
    device: str,
    correct_candidate_index: int | None = None,
) -> dict:
    """Score two frozen completions by their full conditional log-likelihood.

    ``prompt_token_ids`` excludes BOS; this function prepends exactly one BOS.
    Each candidate must contain the two leading-space word tokens and period.
    ``correct_candidate_index=None`` is used for query-only prior comparisons.
    Index 0/1 determines fixed candidate ordering for d = logp(c0) - logp(c1).
    Calling this function performs inference; it must not be called during the
    materialization-only freeze turn.
    """
    import torch

    prompt = _validate_token_array(prompt_token_ids, "prompt_token_ids")
    if BOS_TOKEN_ID in prompt:
        raise ValueError("The frozen prompt token array must exclude BOS.")
    if not isinstance(candidate_token_ids, (list, tuple)) or len(candidate_token_ids) != 2:
        raise ValueError("Exactly two matched candidates are required.")
    candidates = tuple(
        _validate_token_array(candidate, "candidate_token_ids")
        for candidate in candidate_token_ids
    )
    if candidates[0] == candidates[1]:
        raise ValueError("Matched candidates must be distinct.")
    if any(len(candidate) != CANDIDATE_TOKEN_COUNT or candidate[-1] != PERIOD_TOKEN_ID
           for candidate in candidates):
        raise ValueError("Each completion must be two word tokens plus period ID 18.")
    if correct_candidate_index is not None and (
        type(correct_candidate_index) is not int or correct_candidate_index not in (0, 1)
    ):
        raise ValueError("Correct candidate index must be 0, 1, or None for priors.")
    prefix = (BOS_TOKEN_ID,) + prompt
    if len(prefix) + CANDIDATE_TOKEN_COUNT > CONTEXT_SIZE:
        raise ValueError("Frozen completed input exceeds context 256; do not truncate.")

    runtime = configure_runtime(device)
    requested_device = torch.device(runtime["device"])
    base_model = model.base_model
    if int(base_model.context_size) != CONTEXT_SIZE:
        raise ValueError("Unexpected base-model context size.")
    for name, tensor in list(base_model.named_parameters()) + list(base_model.named_buffers()):
        if tensor.device != requested_device:
            raise ValueError(f"Base-model tensor {name} is on the wrong device.")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ValueError(f"Base-model tensor {name} must already be FP32.")
    base_model.eval()

    scores = []
    with torch.inference_mode(), torch.autocast(device_type=requested_device.type, enabled=False):
        for candidate in candidates:
            # Every completion gets its own unpadded [1, T] sequence. There is
            # no wrapper forward, answer-position metadata, or retrieval call.
            input_ids = torch.tensor([prefix + candidate], dtype=torch.long, device=requested_device)
            logits = model.base_model(input_ids)
            expected_shape = (1, len(prefix) + len(candidate), VOCAB_SIZE)
            if tuple(logits.shape) != expected_shape or logits.dtype != torch.float32:
                raise ValueError("Base-model output shape or FP32 dtype is incorrect.")
            # Candidate t_j at zero-based input position len(prefix)+j is
            # predicted by logits at len(prefix)+j-1, using only earlier tokens.
            prediction_logits = logits[0, len(prefix) - 1:len(prefix) + len(candidate) - 1]
            log_probabilities = torch.log_softmax(prediction_logits.to(torch.float64), dim=-1)
            targets = torch.tensor(candidate, dtype=torch.long, device=requested_device)
            token_log_probabilities = log_probabilities.gather(1, targets[:, None]).squeeze(1)
            if not bool(torch.isfinite(token_log_probabilities).all().item()):
                raise ValueError("Nonfinite candidate log-probability; do not emit scores.")
            scores.append({
                "candidate_token_ids": list(candidate),
                "token_log_probabilities": token_log_probabilities.cpu().tolist(),
                "conditional_log_likelihood": float(token_log_probabilities.sum(dtype=torch.float64).item()),
                "word_only_log_likelihood": float(token_log_probabilities[:WORD_TOKEN_COUNT].sum(dtype=torch.float64).item()),
            })

    full = [score["conditional_log_likelihood"] for score in scores]
    word = [score["word_only_log_likelihood"] for score in scores]
    difference = full[0] - full[1]
    word_difference = word[0] - word[1]
    largest = max(full)
    pair_log_mass = largest + math.log(math.exp(full[0] - largest) + math.exp(full[1] - largest))
    margin = None if correct_candidate_index is None else full[correct_candidate_index] - full[1 - correct_candidate_index]
    word_margin = None if correct_candidate_index is None else word[correct_candidate_index] - word[1 - correct_candidate_index]
    return {
        "candidates": scores,
        "candidate0_minus_candidate1": difference,
        "word_only_candidate0_minus_candidate1": word_difference,
        "correct_candidate_index": correct_candidate_index,
        "correct_minus_incorrect_margin": margin,
        "word_only_correct_minus_incorrect_margin": word_margin,
        "item_correct": None if margin is None else margin > 0.0,
        "word_only_item_correct": None if word_margin is None else word_margin > 0.0,
        "tie": difference == 0.0,
        "word_only_tie": word_difference == 0.0,
        "candidate_pair_log_probability_mass": pair_log_mass,
        "candidate_pair_probability_mass": math.exp(pair_log_mass),
        "completed_input_length_including_bos": len(prefix) + CANDIDATE_TOKEN_COUNT,
        "runtime": runtime,
    }
