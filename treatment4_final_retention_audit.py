#!/usr/bin/env python3
"""
DaveLM v0.9 — Treatment #4 Final Retention Audit

READ-ONLY DIAGNOSTIC.

Purpose
-------
Determine whether the FINAL Treatment #4 checkpoint retains the behavior
that appeared during training on the exact frozen 32,000 historical
answer events.

This script:

1. Verifies the exact final candidate checkpoint SHA256.
2. Imports Treatment #4's own frozen machinery.
3. Reconstructs the exact original training corpus.
4. Replays the exact historical 1000-step schedule using SCHEDULE_SEED.
5. Uses the exact Treatment #4 answer-position calculation.
6. Re-verifies target identity and native-prefix identity for every event.
7. Runs the FINAL checkpoint in inference mode only.
8. Measures:
   - top-1 correct
   - top-5 target presence
   - correct > distractor
   - correct - distractor margin >= 0.5
   - candidate-pair probability mass
   - membership loss
   - selector loss
   - replacement loss
   - correct-token rank

NO optimizer.
NO backward().
NO gradients.
NO training.
NO positive-control evaluation.
NO sealed evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import treatment4_train as t4


# =============================================================================
# FROZEN EXPECTATIONS
# =============================================================================

SEED = 8380
SCHEDULE_SEED = 8381

MAX_STEPS = 1000
BATCH_SIZE = 32
CONTEXT_SIZE = 256
EXPECTED_EVENTS = 32_000

MARGIN = 0.5

EXPECTED_FINAL_CHECKPOINT_SHA256 = (
    "2a3f7a24b793016d952a05ec32d84b57b0f6fd84af45d730a61c4b0a294b1476"
)

EXPECTED_SCHEDULE_SHA256 = (
    "a34bb91d258f11c6f7cb6b6cd89c79d8e6fa12d64af15dd14b028fb7ff675cff"
)

ROOT = Path(
    r"C:\DaveLM-CADAVER\two_mapping_membership_margin_manual_seed8380"
)

FINAL_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "membership_margin"
    / "seed_8380"
    / "latest.pt"
)

OUTPUT_PATH = (
    ROOT
    / "treatment4_final_retention_audit.json"
)


# =============================================================================
# UTILS
# =============================================================================

def header(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)
    print()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )


def mean(values):
    if not values:
        return None

    return sum(values) / len(values)


# =============================================================================
# PROVENANCE
# =============================================================================

def verify_provenance() -> dict:

    header(
        "1. PROVENANCE / FROZEN-CONTRACT AUDIT"
    )

    if not FINAL_CHECKPOINT.exists():
        raise RuntimeError(
            f"Final checkpoint not found:\n{FINAL_CHECKPOINT}"
        )

    observed_checkpoint_sha = sha256_file(
        FINAL_CHECKPOINT
    )

    if (
        observed_checkpoint_sha
        != EXPECTED_FINAL_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "\nFINAL CHECKPOINT SHA256 MISMATCH.\n"
            f"Expected: {EXPECTED_FINAL_CHECKPOINT_SHA256}\n"
            f"Observed: {observed_checkpoint_sha}"
        )

    print(
        "[PASS] Final candidate checkpoint SHA256 exact"
    )

    constant_checks = {
        "SEED": (
            int(t4.SEED),
            SEED,
        ),
        "SCHEDULE_SEED": (
            int(t4.SCHEDULE_SEED),
            SCHEDULE_SEED,
        ),
        "MAX_STEPS": (
            int(t4.MAX_STEPS),
            MAX_STEPS,
        ),
        "BATCH_SIZE": (
            int(t4.BATCH_SIZE),
            BATCH_SIZE,
        ),
        "CONTEXT_SIZE": (
            int(t4.CONTEXT_SIZE),
            CONTEXT_SIZE,
        ),
        "MARGIN": (
            float(t4.MARGIN),
            MARGIN,
        ),
    }

    for name, (
        observed,
        expected,
    ) in constant_checks.items():

        if observed != expected:
            raise RuntimeError(
                f"{name} mismatch: "
                f"expected {expected}, observed {observed}"
            )

        print(
            f"[PASS] {name} = {observed}"
        )

    return {
        "checkpoint_path":
            str(FINAL_CHECKPOINT),

        "checkpoint_sha256":
            observed_checkpoint_sha,

        "expected_checkpoint_sha256":
            EXPECTED_FINAL_CHECKPOINT_SHA256,

        "seed":
            SEED,

        "schedule_seed":
            SCHEDULE_SEED,

        "max_steps":
            MAX_STEPS,

        "batch_size":
            BATCH_SIZE,

        "context_size":
            CONTEXT_SIZE,

        "margin":
            MARGIN,
    }


# =============================================================================
# BUILD FINAL MODEL — NO OPTIMIZER
# =============================================================================

def load_final_model(device: torch.device):

    header(
        "2. BUILD BABY + LOAD FINAL TREATMENT #4 CHECKPOINT — READ ONLY"
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )

    # Exact architecture source used by Treatment #4 itself.
    build_model_fn = t4.require_callable(
        t4.original_run,
        "build_model",
    )

    model = build_model_fn(
        "untied"
    ).to(
        device
    )

    checkpoint = torch.load(
        FINAL_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model_state = t4.extract_model_state(
        checkpoint
    )

    missing, unexpected = model.load_state_dict(
        model_state,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "\nFINAL CHECKPOINT STATE-DICT MISMATCH.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}"
        )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        f"[PASS] Model architecture: original_run.build_model('untied')"
    )

    print(
        f"[PASS] Final checkpoint loaded"
    )

    print(
        f"[PASS] State dict exact"
    )

    print(
        f"Parameters: {parameter_count:,}"
    )

    print(
        f"Device:     {device}"
    )

    print(
        "Optimizer:  NONE"
    )

    print(
        "Gradients:  DISABLED"
    )

    return (
        model,
        parameter_count,
    )


# =============================================================================
# EXACT TRAINING CORPUS
# =============================================================================

def reconstruct_training():

    header(
        "3. RECONSTRUCT EXACT TREATMENT #4 TRAINING CORPUS"
    )

    (
        tokenizer,
        records,
        store,
    ) = t4.reconstruct_training()

    if len(records) != int(
        t4.EXPECTED_RECORDS
    ):
        raise RuntimeError(
            "Training-record count mismatch."
        )

    print()
    print(
        "[PASS] Treatment #4 native reconstruction returned:"
    )
    print(
        f"       tokenizer"
    )
    print(
        f"       records = {len(records)}"
    )
    print(
        f"       DocumentTokenStore"
    )

    return (
        tokenizer,
        records,
        store,
    )


# =============================================================================
# EXACT FROZEN RETENTION REPLAY
# =============================================================================

@torch.inference_mode()
def audit_retention(
    model,
    records,
    store,
    device,
):

    header(
        "4. REPLAY EXACT 32,000 HISTORICAL ANSWER EVENTS"
    )

    model.eval()

    generator = (
        torch.Generator()
        .manual_seed(
            SCHEDULE_SEED
        )
    )

    schedule_digest = hashlib.sha256()

    event_count = 0
    target_matches = 0
    prefix_matches = 0

    top1_correct_count = 0
    top5_correct_count = 0

    correct_gt_distractor_count = 0
    margin_satisfied_count = 0

    candidate_masses = []
    margins = []
    membership_losses = []
    selector_losses = []
    replacement_losses = []
    correct_ranks = []

    first_events = []
    first_failures = []

    per_document = {}

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        (
            inputs,
            targets,
            _,
            meta,
        ) = t4.sample_document_batch(
            store,
            BATCH_SIZE,
            CONTEXT_SIZE,
            generator,
            device,
        )

        document_indices = [
            int(x)
            for x in meta[
                "document_indices"
            ]
        ]

        starts = [
            int(x)
            for x in meta[
                "within_document_starts"
            ]
        ]

        schedule_digest.update(
            t4.strict_json_dumps(
                {
                    "document_indices":
                        document_indices,

                    "within_document_starts":
                        starts,
                }
            ).encode(
                "ascii"
            )
        )

        answer_positions = []
        correct_ids = []
        distractor_ids = []

        event_metadata = []

        # -------------------------------------------------------------
        # Reproduce Treatment #4's exact native answer-position audit.
        # -------------------------------------------------------------

        for batch_index in range(
            BATCH_SIZE
        ):

            document_index = (
                document_indices[
                    batch_index
                ]
            )

            row = records[
                document_index
            ]

            start = starts[
                batch_index
            ]

            prefix_ids = [
                int(x)
                for x in row[
                    "layout"
                ][
                    "prefix_ids"
                ]
            ]

            position = (
                len(prefix_ids)
                - start
                - 1
            )

            if not (
                0
                <= position
                < targets.shape[1]
            ):
                raise RuntimeError(
                    f"Step {step}, batch {batch_index}: "
                    "answer outside sampled window."
                )

            expected_target = int(
                row[
                    "target_token_id"
                ]
            )

            actual_target = int(
                targets[
                    batch_index,
                    position,
                ].item()
            )

            if (
                actual_target
                != expected_target
            ):
                raise RuntimeError(
                    f"Step {step}, batch {batch_index}: "
                    "sampled target mismatch."
                )

            target_matches += 1

            if actual_target == -100:
                raise RuntimeError(
                    f"Step {step}, batch {batch_index}: "
                    "answer is not supervised."
                )

            sampled_prefix = [
                int(x)
                for x in inputs[
                    batch_index,
                    :position + 1,
                ].tolist()
            ]

            if (
                sampled_prefix
                != prefix_ids
            ):
                raise RuntimeError(
                    f"Step {step}, batch {batch_index}: "
                    "native prefix identity mismatch."
                )

            prefix_matches += 1

            distractor_id = int(
                row[
                    "distractor_token_id"
                ]
            )

            if (
                expected_target
                == distractor_id
            ):
                raise RuntimeError(
                    f"Step {step}, batch {batch_index}: "
                    "target equals distractor."
                )

            answer_positions.append(
                position
            )

            correct_ids.append(
                expected_target
            )

            distractor_ids.append(
                distractor_id
            )

            event_metadata.append(
                {
                    "step":
                        step,

                    "batch_index":
                        batch_index,

                    "document_index":
                        document_index,

                    "within_document_start":
                        start,

                    "answer_position":
                        position,

                    "correct_token_id":
                        expected_target,

                    "distractor_token_id":
                        distractor_id,
                }
            )

        # -------------------------------------------------------------
        # EXACT Treatment #4 forward interface.
        # -------------------------------------------------------------

        logits = model(
            inputs
        )

        if (
            logits.ndim != 3
            or logits.shape[0]
            != BATCH_SIZE
        ):
            raise RuntimeError(
                f"Unexpected logits shape: "
                f"{tuple(logits.shape)}"
            )

        batch_indices_tensor = torch.arange(
            BATCH_SIZE,
            device=device,
            dtype=torch.long,
        )

        position_tensor = torch.tensor(
            answer_positions,
            device=device,
            dtype=torch.long,
        )

        correct_tensor = torch.tensor(
            correct_ids,
            device=device,
            dtype=torch.long,
        )

        distractor_tensor = torch.tensor(
            distractor_ids,
            device=device,
            dtype=torch.long,
        )

        answer_logits = logits[
            batch_indices_tensor,
            position_tensor,
            :
        ]

        correct_logits = answer_logits.gather(
            1,
            correct_tensor.unsqueeze(1),
        ).squeeze(1)

        distractor_logits = answer_logits.gather(
            1,
            distractor_tensor.unsqueeze(1),
        ).squeeze(1)

        answer_probs = F.softmax(
            answer_logits,
            dim=-1,
        )

        correct_probs = answer_probs.gather(
            1,
            correct_tensor.unsqueeze(1),
        ).squeeze(1)

        distractor_probs = answer_probs.gather(
            1,
            distractor_tensor.unsqueeze(1),
        ).squeeze(1)

        candidate_mass_tensor = (
            correct_probs
            + distractor_probs
        )

        margin_tensor = (
            correct_logits
            - distractor_logits
        )

        membership_tensor = (
            torch.logsumexp(
                answer_logits,
                dim=-1,
            )
            -
            torch.logsumexp(
                torch.stack(
                    [
                        correct_logits,
                        distractor_logits,
                    ],
                    dim=-1,
                ),
                dim=-1,
            )
        )

        selector_tensor = F.relu(
            MARGIN
            - margin_tensor
        )

        replacement_tensor = (
            membership_tensor
            + selector_tensor
        )

        top1_tensor = torch.argmax(
            answer_logits,
            dim=-1,
        )

        top5_tensor = torch.topk(
            answer_logits,
            k=min(
                5,
                answer_logits.shape[-1],
            ),
            dim=-1,
        ).indices

        correct_rank_tensor = (
            (
                answer_logits
                > correct_logits.unsqueeze(1)
            )
            .sum(
                dim=-1
            )
            + 1
        )

        # -------------------------------------------------------------
        # Accumulate exact-event metrics.
        # -------------------------------------------------------------

        for batch_index in range(
            BATCH_SIZE
        ):

            event_count += 1

            correct_id = correct_ids[
                batch_index
            ]

            distractor_id = distractor_ids[
                batch_index
            ]

            top1_id = int(
                top1_tensor[
                    batch_index
                ].item()
            )

            top5_ids = [
                int(x)
                for x in top5_tensor[
                    batch_index
                ].tolist()
            ]

            top1_correct = (
                top1_id
                == correct_id
            )

            top5_correct = (
                correct_id
                in top5_ids
            )

            correct_gt_distractor = bool(
                margin_tensor[
                    batch_index
                ].item()
                > 0.0
            )

            margin_satisfied = bool(
                margin_tensor[
                    batch_index
                ].item()
                >= MARGIN
            )

            candidate_mass = float(
                candidate_mass_tensor[
                    batch_index
                ].item()
            )

            margin_value = float(
                margin_tensor[
                    batch_index
                ].item()
            )

            membership_loss = float(
                membership_tensor[
                    batch_index
                ].item()
            )

            selector_loss = float(
                selector_tensor[
                    batch_index
                ].item()
            )

            replacement_loss = float(
                replacement_tensor[
                    batch_index
                ].item()
            )

            correct_rank = int(
                correct_rank_tensor[
                    batch_index
                ].item()
            )

            if top1_correct:
                top1_correct_count += 1

            if top5_correct:
                top5_correct_count += 1

            if correct_gt_distractor:
                correct_gt_distractor_count += 1

            if margin_satisfied:
                margin_satisfied_count += 1

            candidate_masses.append(
                candidate_mass
            )

            margins.append(
                margin_value
            )

            membership_losses.append(
                membership_loss
            )

            selector_losses.append(
                selector_loss
            )

            replacement_losses.append(
                replacement_loss
            )

            correct_ranks.append(
                correct_rank
            )

            metadata = dict(
                event_metadata[
                    batch_index
                ]
            )

            metadata.update(
                {
                    "top1_token_id":
                        top1_id,

                    "top5_token_ids":
                        top5_ids,

                    "top1_correct":
                        top1_correct,

                    "top5_correct":
                        top5_correct,

                    "correct_gt_distractor":
                        correct_gt_distractor,

                    "margin_satisfied":
                        margin_satisfied,

                    "margin":
                        margin_value,

                    "candidate_mass":
                        candidate_mass,

                    "membership_loss":
                        membership_loss,

                    "selector_loss":
                        selector_loss,

                    "replacement_loss":
                        replacement_loss,

                    "correct_rank":
                        correct_rank,
                }
            )

            if len(first_events) < 10:
                first_events.append(
                    metadata
                )

            if (
                not top1_correct
                and len(first_failures) < 20
            ):
                first_failures.append(
                    metadata
                )

            document_index = metadata[
                "document_index"
            ]

            bucket = per_document.setdefault(
                str(document_index),
                {
                    "events":
                        0,

                    "top1_correct":
                        0,

                    "correct_gt_distractor":
                        0,

                    "margin_satisfied":
                        0,
                },
            )

            bucket[
                "events"
            ] += 1

            bucket[
                "top1_correct"
            ] += int(
                top1_correct
            )

            bucket[
                "correct_gt_distractor"
            ] += int(
                correct_gt_distractor
            )

            bucket[
                "margin_satisfied"
            ] += int(
                margin_satisfied
            )

        if (
            step == 1
            or step % 100 == 0
        ):
            running_selection = (
                correct_gt_distractor_count
                / event_count
            )

            running_margin = (
                margin_satisfied_count
                / event_count
            )

            running_top1 = (
                top1_correct_count
                / event_count
            )

            print(
                f"Step {step:4d}/{MAX_STEPS} | "
                f"events={event_count:5d} | "
                f"top1={running_top1:.4f} | "
                f"correct>distractor={running_selection:.4f} | "
                f"margin={running_margin:.4f}"
            )

    # =========================================================================
    # FINAL SCHEDULE INTEGRITY
    # =========================================================================

    observed_schedule_sha = (
        schedule_digest.hexdigest()
    )

    if (
        observed_schedule_sha
        != EXPECTED_SCHEDULE_SHA256
    ):
        raise RuntimeError(
            "\nFROZEN SCHEDULE SHA256 MISMATCH.\n"
            f"Expected: {EXPECTED_SCHEDULE_SHA256}\n"
            f"Observed: {observed_schedule_sha}"
        )

    if event_count != EXPECTED_EVENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_EVENTS} events; "
            f"observed {event_count}."
        )

    if target_matches != EXPECTED_EVENTS:
        raise RuntimeError(
            "Not all targets matched."
        )

    if prefix_matches != EXPECTED_EVENTS:
        raise RuntimeError(
            "Not all native prefixes matched."
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    top1_accuracy = (
        top1_correct_count
        / event_count
    )

    top5_accuracy = (
        top5_correct_count
        / event_count
    )

    selection_accuracy = (
        correct_gt_distractor_count
        / event_count
    )

    margin_accuracy = (
        margin_satisfied_count
        / event_count
    )

    mean_candidate_mass = mean(
        candidate_masses
    )

    mean_margin = mean(
        margins
    )

    mean_membership_loss = mean(
        membership_losses
    )

    mean_selector_loss = mean(
        selector_losses
    )

    mean_replacement_loss = mean(
        replacement_losses
    )

    mean_correct_rank = mean(
        correct_ranks
    )

    sorted_ranks = sorted(
        correct_ranks
    )

    median_correct_rank = (
        sorted_ranks[
            len(sorted_ranks) // 2
        ]
    )

    # Diagnostic only.
    # These are NOT experimental positive-control gates.
    if (
        selection_accuracy >= 0.95
        and margin_accuracy >= 0.90
        and mean_candidate_mass >= 0.85
    ):
        classification = (
            "STRONG_FINAL_TRAINING_RETENTION"
        )

    elif (
        selection_accuracy <= 0.65
        or margin_accuracy <= 0.50
    ):
        classification = (
            "FINAL_TRAINING_RETENTION_FAILURE"
        )

    else:
        classification = (
            "PARTIAL_OR_AMBIGUOUS_FINAL_RETENTION"
        )

    result = {
        "experiment":
            "DaveLM v0.9 Treatment #4 final retention audit",

        "classification":
            classification,

        "diagnostic_only":
            True,

        "safety": {
            "read_only":
                True,

            "optimizer_created":
                False,

            "backward_calls":
                0,

            "gradients_enabled":
                False,

            "training_performed":
                False,

            "positive_controls_evaluated":
                False,

            "sealed_evaluation_opened":
                False,
        },

        "schedule": {
            "expected_sha256":
                EXPECTED_SCHEDULE_SHA256,

            "observed_sha256":
                observed_schedule_sha,

            "sha256_exact":
                True,

            "events":
                event_count,

            "target_matches":
                target_matches,

            "prefix_matches":
                prefix_matches,
        },

        "metrics": {
            "top1_correct":
                top1_correct_count,

            "top1_accuracy":
                top1_accuracy,

            "top5_correct":
                top5_correct_count,

            "top5_accuracy":
                top5_accuracy,

            "correct_gt_distractor":
                correct_gt_distractor_count,

            "correct_gt_distractor_rate":
                selection_accuracy,

            "margin_satisfied":
                margin_satisfied_count,

            "margin_satisfied_rate":
                margin_accuracy,

            "mean_candidate_mass":
                mean_candidate_mass,

            "mean_correct_minus_distractor_logit":
                mean_margin,

            "mean_membership_loss":
                mean_membership_loss,

            "mean_selector_loss":
                mean_selector_loss,

            "mean_answer_replacement_loss":
                mean_replacement_loss,

            "mean_correct_rank":
                mean_correct_rank,

            "median_correct_rank":
                median_correct_rank,
        },

        "first_10_events":
            first_events,

        "first_20_top1_failures":
            first_failures,

        "per_document":
            per_document,
    }

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cpu",
        choices=[
            "cpu",
            "cuda",
        ],
        help=(
            "Inference device. "
            "Default is cpu for conservative reproducibility."
        ),
    )

    args = parser.parse_args()

    header(
        "DaveLM v0.9 — TREATMENT #4 FINAL RETENTION AUDIT"
    )

    device = torch.device(
        args.device
    )

    if (
        args.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA/ROCm torch device requested, "
            "but torch.cuda.is_available() is False."
        )

    provenance = verify_provenance()

    (
        model,
        parameter_count,
    ) = load_final_model(
        device
    )

    (
        tokenizer,
        records,
        store,
    ) = reconstruct_training()

    result = audit_retention(
        model,
        records,
        store,
        device,
    )

    result[
        "provenance"
    ] = provenance

    result[
        "parameter_count"
    ] = parameter_count

    write_json(
        OUTPUT_PATH,
        result,
    )

    header(
        "5. FINAL RETENTION RESULT"
    )

    metrics = result[
        "metrics"
    ]

    print(
        f"Classification:          {result['classification']}"
    )

    print(
        f"Exact events:            {result['schedule']['events']:,}"
    )

    print(
        f"Schedule SHA exact:      {result['schedule']['sha256_exact']}"
    )

    print(
        f"Target matches:          {result['schedule']['target_matches']:,}"
    )

    print(
        f"Prefix matches:          {result['schedule']['prefix_matches']:,}"
    )

    print()

    print(
        f"Top-1 accuracy:          {metrics['top1_accuracy']:.6f}"
    )

    print(
        f"Top-5 accuracy:          {metrics['top5_accuracy']:.6f}"
    )

    print(
        f"Correct > distractor:    "
        f"{metrics['correct_gt_distractor_rate']:.6f}"
    )

    print(
        f"Margin >= {MARGIN}:         "
        f"{metrics['margin_satisfied_rate']:.6f}"
    )

    print(
        f"Mean candidate mass:     "
        f"{metrics['mean_candidate_mass']:.6f}"
    )

    print(
        f"Mean membership loss:    "
        f"{metrics['mean_membership_loss']:.6f}"
    )

    print(
        f"Mean selector loss:      "
        f"{metrics['mean_selector_loss']:.6f}"
    )

    print(
        f"Mean replacement loss:   "
        f"{metrics['mean_answer_replacement_loss']:.6f}"
    )

    print(
        f"Mean correct rank:       "
        f"{metrics['mean_correct_rank']:.3f}"
    )

    print(
        f"Median correct rank:     "
        f"{metrics['median_correct_rank']}"
    )

    print()

    print(
        f"JSON report:\n{OUTPUT_PATH}"
    )

    print()

    if (
        result["classification"]
        == "STRONG_FINAL_TRAINING_RETENTION"
    ):
        print(
            "INTERPRETATION:"
        )
        print(
            "The final checkpoint still retains the Treatment #4 "
            "training-event behavior strongly."
        )
        print(
            "Catastrophic forgetting / schedule-wide retention failure "
            "is therefore disfavored."
        )
        print(
            "Next diagnostic: representation / query-separability probe."
        )

    elif (
        result["classification"]
        == "FINAL_TRAINING_RETENTION_FAILURE"
    ):
        print(
            "INTERPRETATION:"
        )
        print(
            "The behavior seen during training did not survive strongly "
            "in the final checkpoint across the frozen historical events."
        )
        print(
            "Next diagnostic: interference / forgetting dynamics, "
            "not hidden-state representation probing yet."
        )

    else:
        print(
            "INTERPRETATION:"
        )
        print(
            "Final retention is intermediate or mixed."
        )
        print(
            "Inspect per-document failures and retention distribution "
            "before choosing the next experimental branch."
        )


if __name__ == "__main__":
    main()