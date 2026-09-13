from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


# =============================================================================
# DaveLM v0.9 — Treatment #5
# Paired Counterfactual Query Binding
#
# FINAL TRAINER
#
# SCIENTIFIC CHANGE FROM T4:
#   Training distribution only.
#
# Every batch contains 16 COMPLETE matched counterfactual pairs:
#
#   Twin A:
#       same base document/context/mappings
#       query slot 0
#       target slot 0
#
#   Twin B:
#       same base document/context/mappings
#       query slot 1
#       target slot 1
#
# Answer objective is the same Treatment #4 replacement objective:
#
#   membership =
#       logsumexp(full_vocab_logits)
#       - logsumexp([correct_logit, distractor_logit])
#
#   selector =
#       relu(0.5 - (correct_logit - distractor_logit))
#
#   answer replacement =
#       membership + selector
#
# Original answer CE is removed.
# Every non-answer causal position uses ordinary full-vocabulary CE.
#
# IMPORTANT:
#   - NO schedule regeneration.
#   - NO positive-control evaluation.
#   - NO sealed evaluation.
#   - NO modification under C:\DaveLM-v0.9.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

SOURCE_ROOT = Path(
    r"C:\DaveLM-v0.9"
)

SCHEDULE_PATH = Path(
    r"C:\DaveLM-CADAVER"
    r"\treatment5_counterfactual_pairs_seed8382"
    r"\treatment5_full_document_frozen_schedule.json"
)

PAIR_POOL_PATH = Path(
    r"C:\DaveLM-CADAVER"
    r"\treatment5_counterfactual_pairs_seed8382"
    r"\treatment5_full_document_pair_pool.json"
)

PREFLIGHT_RESULT_PATH = Path(
    r"C:\DaveLM-CADAVER"
    r"\treatment5_counterfactual_pairs_seed8382"
    r"\treatment5_full_document_preflight_result.json"
)

START_CHECKPOINT = Path(
    r"C:\DaveLM-v0.9"
    r"\experiments"
    r"\minimal_contextual_binding"
    r"\checkpoints"
    r"\treatment_one_mapping"
    r"\seed_8380"
    r"\latest.pt"
)

OUTPUT_ROOT = Path(
    r"C:\DaveLM-CADAVER"
    r"\treatment5_paired_counterfactual_query_binding_seed8380"
)

CHECKPOINT_ROOT = (
    OUTPUT_ROOT
    / "checkpoints"
    / "paired_counterfactual"
    / "seed_8380"
)

LATEST_CHECKPOINT = (
    CHECKPOINT_ROOT
    / "latest.pt"
)

TRAINING_RESULT_PATH = (
    OUTPUT_ROOT
    / "treatment5_training_result.json"
)

TRAINING_METRICS_PATH = (
    OUTPUT_ROOT
    / "treatment5_training_metrics.jsonl"
)


# =============================================================================
# FROZEN HASHES
# =============================================================================

EXPECTED_SCHEDULE_SHA256 = (
    "a4e64fd9d1b934aa440a5d036a3bec0a"
    "c7c9ac97d58c72fb45df6d8576bb7c12"
)

EXPECTED_PAIR_POOL_SHA256 = (
    "0b31e39361f7a45dc4182c49ab4b4967"
    "bffe539cd2ff12e7b196eabfb3dac307"
)

EXPECTED_START_CHECKPOINT_SHA256 = (
    "345984c52a06db5f988aaf4cd47963ee"
    "a0e9d77489cee94dbb10816af2f5443e"
)


# =============================================================================
# FROZEN EXPERIMENT CONSTANTS
# =============================================================================

SEED = 8380

SCHEDULE_SEED = 8382

MAX_STEPS = 1000

BATCH_SIZE = 32

PAIRS_PER_BATCH = 16

MARGIN = 0.5

LEARNING_RATE = 3.0e-4

WEIGHT_DECAY = 0.05

GRAD_CLIP = 2.0

EXPECTED_PARAMETER_COUNT = 10_594_944

EXPECTED_STEPS = 1000

EXPECTED_PAIR_PRESENTATIONS = 16_000

EXPECTED_EXAMPLES = 32_000

EXPECTED_SLOT0 = 16_000

EXPECTED_SLOT1 = 16_000

EXPECTED_ANSWER_SUBSTITUTIONS = 32_000

EXPECTED_TOTAL_SUPERVISED = 6_144_000

EXPECTED_NONANSWER_SUPERVISED = 6_112_000

EXPECTED_PAIR_EXPOSURE_MIN = 10

EXPECTED_PAIR_EXPOSURE_MAX = 11

EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH = 193

EXPECTED_CAUSAL_SEQUENCE_LENGTH = 192


# =============================================================================
# LOGGING / SAVING
# =============================================================================

LOG_STEPS = {
    1,
    10,
    25,
    50,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
}

CHECKPOINT_STEPS = {
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def header(title: str) -> None:

    print()

    print(
        "=" * 100
    )

    print(
        title
    )

    print(
        "=" * 100
    )

    print()


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def append_jsonl(
    path: Path,
    payload: dict[str, Any],
) -> None:

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:

        handle.write(
            json.dumps(
                payload,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
        )

        handle.write(
            "\n"
        )


def set_seed(
    seed: int,
) -> None:

    random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# =============================================================================
# IMPORT EXACT NATIVE TWO-MAP MACHINERY
# =============================================================================

def import_native_module():

    if str(
        SOURCE_ROOT
    ) not in sys.path:

        sys.path.insert(
            0,
            str(
                SOURCE_ROOT
            ),
        )

    from experiments.two_mapping_contextual_binding import run as original_run

    return original_run


# =============================================================================
# VERIFY ARTIFACTS
# =============================================================================

def load_and_verify_schedule() -> dict[str, Any]:

    header(
        "VERIFYING FROZEN TREATMENT #5 ARTIFACTS"
    )

    for path in (
        SCHEDULE_PATH,
        PAIR_POOL_PATH,
        PREFLIGHT_RESULT_PATH,
        START_CHECKPOINT,
    ):

        if not path.is_file():

            raise FileNotFoundError(
                f"Required artifact not found:\n"
                f"{path}"
            )

    schedule_sha = sha256_file(
        SCHEDULE_PATH
    )

    pair_pool_sha = sha256_file(
        PAIR_POOL_PATH
    )

    checkpoint_sha = sha256_file(
        START_CHECKPOINT
    )

    print(
        "Frozen schedule SHA256:"
    )

    print(
        f"  {schedule_sha}"
    )

    print()

    print(
        "Expected:"
    )

    print(
        f"  {EXPECTED_SCHEDULE_SHA256}"
    )

    if schedule_sha != (
        EXPECTED_SCHEDULE_SHA256
    ):

        raise RuntimeError(
            "Frozen T5 schedule SHA mismatch. "
            "Training refused."
        )

    print()

    print(
        "Frozen schedule integrity: PASS"
    )

    print()

    print(
        "Pair-pool SHA256:"
    )

    print(
        f"  {pair_pool_sha}"
    )

    print()

    print(
        "Expected:"
    )

    print(
        f"  {EXPECTED_PAIR_POOL_SHA256}"
    )

    if pair_pool_sha != (
        EXPECTED_PAIR_POOL_SHA256
    ):

        raise RuntimeError(
            "Frozen T5 pair-pool SHA mismatch. "
            "Training refused."
        )

    print()

    print(
        "Pair-pool integrity: PASS"
    )

    print()

    print(
        "Starting checkpoint SHA256:"
    )

    print(
        f"  {checkpoint_sha}"
    )

    print()

    print(
        "Expected:"
    )

    print(
        f"  {EXPECTED_START_CHECKPOINT_SHA256}"
    )

    if checkpoint_sha != (
        EXPECTED_START_CHECKPOINT_SHA256
    ):

        raise RuntimeError(
            "Starting checkpoint SHA mismatch. "
            "Training refused."
        )

    print()

    print(
        "Starting checkpoint integrity: PASS"
    )

    schedule = json.loads(
        SCHEDULE_PATH.read_text(
            encoding="utf-8"
        )
    )

    return schedule


# =============================================================================
# AUDIT FROZEN SCHEDULE AGAIN BEFORE TRAINING
# =============================================================================

def audit_schedule(
    schedule: dict[str, Any],
) -> None:

    header(
        "AUDITING FROZEN TRAINING SCHEDULE"
    )

    expected_values = {
        "schedule_seed": (
            SCHEDULE_SEED
        ),
        "steps": (
            EXPECTED_STEPS
        ),
        "batch_size": (
            BATCH_SIZE
        ),
        "pairs_per_batch": (
            PAIRS_PER_BATCH
        ),
        "pair_presentations": (
            EXPECTED_PAIR_PRESENTATIONS
        ),
        "scheduled_examples": (
            EXPECTED_EXAMPLES
        ),
        "slot_0_presentations": (
            EXPECTED_SLOT0
        ),
        "slot_1_presentations": (
            EXPECTED_SLOT1
        ),
        "pair_exposure_min": (
            EXPECTED_PAIR_EXPOSURE_MIN
        ),
        "pair_exposure_max": (
            EXPECTED_PAIR_EXPOSURE_MAX
        ),
        "answer_substitutions": (
            EXPECTED_ANSWER_SUBSTITUTIONS
        ),
        "total_supervised_tokens": (
            EXPECTED_TOTAL_SUPERVISED
        ),
        "nonanswer_supervised_tokens": (
            EXPECTED_NONANSWER_SUPERVISED
        ),
    }

    for key, expected in (
        expected_values.items()
    ):

        if key not in schedule:

            raise RuntimeError(
                f"Frozen schedule missing "
                f"{key!r}."
            )

        observed = int(
            schedule[
                key
            ]
        )

        if observed != int(
            expected
        ):

            raise RuntimeError(
                f"Frozen schedule field "
                f"{key!r} mismatch: "
                f"expected {expected}, "
                f"observed {observed}."
            )

    steps_data = schedule.get(
        "steps_data"
    )

    if not isinstance(
        steps_data,
        list,
    ):

        raise RuntimeError(
            "steps_data is not a list."
        )

    if len(
        steps_data
    ) != MAX_STEPS:

        raise RuntimeError(
            f"Expected {MAX_STEPS} steps; "
            f"found {len(steps_data)}."
        )

    observed_examples = 0

    observed_slot0 = 0

    observed_slot1 = 0

    observed_supervised = 0

    observed_nonanswer = 0

    observed_answers = 0

    pair_exposures: dict[
        str,
        int,
    ] = {}

    for expected_step, step_data in enumerate(
        steps_data,
        start=1,
    ):

        observed_step = int(
            step_data[
                "step"
            ]
        )

        if observed_step != expected_step:

            raise RuntimeError(
                f"Step ordering mismatch: "
                f"expected {expected_step}, "
                f"got {observed_step}."
            )

        examples = step_data.get(
            "examples"
        )

        if not isinstance(
            examples,
            list,
        ):

            raise RuntimeError(
                f"Step {expected_step}: "
                f"examples is not a list."
            )

        if len(
            examples
        ) != BATCH_SIZE:

            raise RuntimeError(
                f"Step {expected_step}: "
                f"expected {BATCH_SIZE} "
                f"examples; got "
                f"{len(examples)}."
            )

        # ---------------------------------------------------------------------
        # Every adjacent two examples must be one complete A/B pair.
        # ---------------------------------------------------------------------

        for offset in range(
            0,
            BATCH_SIZE,
            2,
        ):

            a = examples[
                offset
            ]

            b = examples[
                offset + 1
            ]

            if str(
                a[
                    "pair_id"
                ]
            ) != str(
                b[
                    "pair_id"
                ]
            ):

                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"pair adjacency failure."
                )

            if str(
                a[
                    "twin"
                ]
            ) != "A":

                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"first pair member "
                    f"is not Twin A."
                )

            if str(
                b[
                    "twin"
                ]
            ) != "B":

                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"second pair member "
                    f"is not Twin B."
                )

            if int(
                a[
                    "query_slot"
                ]
            ) != 0:

                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"Twin A query slot "
                    f"is not 0."
                )

            if int(
                b[
                    "query_slot"
                ]
            ) != 1:

                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"Twin B query slot "
                    f"is not 1."
                )

            if int(
                a[
                    "target_token_id"
                ]
            ) != int(
                b[
                    "distractor_token_id"
                ]
            ):

                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"A target != "
                    f"B distractor."
                )

            if int(
                b[
                    "target_token_id"
                ]
            ) != int(
                a[
                    "distractor_token_id"
                ]
            ):

                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"B target != "
                    f"A distractor."
                )

            pair_id = str(
                a[
                    "pair_id"
                ]
            )

            pair_exposures[
                pair_id
            ] = (
                pair_exposures.get(
                    pair_id,
                    0,
                )
                + 1
            )

        for example in examples:

            ids = [
                int(x)
                for x in example[
                    "full_document_token_ids"
                ]
            ]

            if len(
                ids
            ) != (
                EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH
            ):

                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"model-visible document "
                    f"length {len(ids)}; "
                    f"expected "
                    f"{EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH}."
                )

            supervised_count = int(
                example[
                    "supervised_token_count"
                ]
            )

            nonanswer_count = int(
                example[
                    "nonanswer_supervised_token_count"
                ]
            )

            if supervised_count != (
                len(ids)
                - 1
            ):

                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"supervised-token "
                    f"count mismatch."
                )

            if nonanswer_count != (
                supervised_count
                - 1
            ):

                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"non-answer-token "
                    f"count mismatch."
                )

            answer_index = int(
                example[
                    "answer_token_index"
                ]
            )

            answer_position = int(
                example[
                    "answer_causal_position"
                ]
            )

            if answer_position != (
                answer_index
                - 1
            ):

                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"answer causal-position "
                    f"mismatch."
                )

            target = int(
                example[
                    "target_token_id"
                ]
            )

            distractor = int(
                example[
                    "distractor_token_id"
                ]
            )

            if target == distractor:

                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"target == distractor."
                )

            if int(
                ids[
                    answer_index
                ]
            ) != target:

                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"frozen answer token "
                    f"does not equal target."
                )

            query_slot = int(
                example[
                    "query_slot"
                ]
            )

            if query_slot == 0:

                observed_slot0 += 1

            elif query_slot == 1:

                observed_slot1 += 1

            else:

                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"invalid query slot."
                )

            observed_examples += 1

            observed_answers += 1

            observed_supervised += (
                supervised_count
            )

            observed_nonanswer += (
                nonanswer_count
            )

    if observed_examples != (
        EXPECTED_EXAMPLES
    ):

        raise RuntimeError(
            "Scheduled example count mismatch."
        )

    if observed_slot0 != (
        EXPECTED_SLOT0
    ):

        raise RuntimeError(
            "Slot-0 schedule count mismatch."
        )

    if observed_slot1 != (
        EXPECTED_SLOT1
    ):

        raise RuntimeError(
            "Slot-1 schedule count mismatch."
        )

    if observed_answers != (
        EXPECTED_ANSWER_SUBSTITUTIONS
    ):

        raise RuntimeError(
            "Answer substitution count mismatch."
        )

    if observed_supervised != (
        EXPECTED_TOTAL_SUPERVISED
    ):

        raise RuntimeError(
            "Supervised-token count mismatch."
        )

    if observed_nonanswer != (
        EXPECTED_NONANSWER_SUPERVISED
    ):

        raise RuntimeError(
            "Non-answer-token count mismatch."
        )

    exposure_values = list(
        pair_exposures.values()
    )

    if min(
        exposure_values
    ) != (
        EXPECTED_PAIR_EXPOSURE_MIN
    ):

        raise RuntimeError(
            "Minimum pair exposure mismatch."
        )

    if max(
        exposure_values
    ) != (
        EXPECTED_PAIR_EXPOSURE_MAX
    ):

        raise RuntimeError(
            "Maximum pair exposure mismatch."
        )

    print(
        f"Steps: "
        f"{MAX_STEPS}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Complete pairs / batch: "
        f"{PAIRS_PER_BATCH}"
    )

    print(
        f"Scheduled examples: "
        f"{observed_examples}"
    )

    print(
        f"Slot 0: "
        f"{observed_slot0}"
    )

    print(
        f"Slot 1: "
        f"{observed_slot1}"
    )

    print(
        f"Answer substitutions: "
        f"{observed_answers}"
    )

    print(
        f"Total supervised positions: "
        f"{observed_supervised}"
    )

    print(
        f"Ordinary non-answer CE positions: "
        f"{observed_nonanswer}"
    )

    print(
        f"Pair exposure range: "
        f"{min(exposure_values)}-"
        f"{max(exposure_values)}"
    )

    print()

    print(
        "Frozen schedule audit: PASS"
    )


# =============================================================================
# EXACT CHECKPOINT STATE EXTRACTION USED BY T4
# =============================================================================

def extract_model_state(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Starting checkpoint "
            "is not a dictionary."
        )

    if "model_state" in checkpoint:

        candidate = checkpoint[
            "model_state"
        ]

        if isinstance(
            candidate,
            dict,
        ):

            return candidate

    if "model_state_dict" in checkpoint:

        candidate = checkpoint[
            "model_state_dict"
        ]

        if isinstance(
            candidate,
            dict,
        ):

            return candidate

    if "model" in checkpoint:

        candidate = checkpoint[
            "model"
        ]

        if isinstance(
            candidate,
            dict,
        ):

            return candidate

    if checkpoint and all(
        torch.is_tensor(
            value
        )
        for value
        in checkpoint.values()
    ):

        return checkpoint

    raise RuntimeError(
        "Could not identify "
        "model state in "
        "starting checkpoint."
    )


# =============================================================================
# BUILD BABY — EXACT SAME METHOD AS T4
# =============================================================================

def build_model_and_optimizer(
    original_run,
    device: torch.device,
):

    header(
        "BUILDING BABY AND LOADING VERIFIED ONE-MAP CHECKPOINT"
    )

    set_seed(
        SEED
    )

    # -------------------------------------------------------------------------
    # THIS is the important fix.
    #
    # Treatment #4 does NOT search for DaveLMV082 as a class.
    #
    # It calls the native two-map experiment's actual model builder:
    #
    #     original_run.build_model("untied")
    #
    # We do exactly the same thing here.
    # -------------------------------------------------------------------------

    build_model_fn = getattr(
        original_run,
        "build_model",
        None,
    )

    if not callable(
        build_model_fn
    ):

        raise RuntimeError(
            "Native two-map run module "
            "does not expose callable "
            "build_model()."
        )

    model = build_model_fn(
        "untied"
    ).to(
        device
    )

    checkpoint = torch.load(
        START_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model_state = extract_model_state(
        checkpoint
    )

    missing, unexpected = (
        model.load_state_dict(
            model_state,
            strict=False,
        )
    )

    if missing or unexpected:

        raise RuntimeError(
            "\nSTART CHECKPOINT "
            "STATE-DICT MISMATCH.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}\n"
        )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    if parameter_count != (
        EXPECTED_PARAMETER_COUNT
    ):

        raise RuntimeError(
            f"Parameter-count mismatch.\n"
            f"Expected: "
            f"{EXPECTED_PARAMETER_COUNT:,}\n"
            f"Observed: "
            f"{parameter_count:,}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        f"Parameters: "
        f"{parameter_count:,}"
    )

    print(
        f"Device:     "
        f"{device}"
    )

    print(
        "Model builder: "
        'original_run.build_model("untied")'
    )

    print(
        "One-map model_state: PASS"
    )

    print(
        "Optimizer: fresh AdamW"
    )

    print(
        f"LR:         "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay: "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Grad clip:    "
        f"{GRAD_CLIP}"
    )

    return (
        model,
        optimizer,
        parameter_count,
    )


# =============================================================================
# NORMALIZE MODEL OUTPUT
# =============================================================================

def extract_logits(
    output: Any,
) -> torch.Tensor:

    if torch.is_tensor(
        output
    ):

        return output

    if isinstance(
        output,
        dict,
    ):

        candidate = output.get(
            "logits"
        )

        if torch.is_tensor(
            candidate
        ):

            return candidate

    if isinstance(
        output,
        (
            tuple,
            list,
        ),
    ):

        for candidate in output:

            if (
                torch.is_tensor(
                    candidate
                )
                and candidate.ndim == 3
            ):

                return candidate

    candidate = getattr(
        output,
        "logits",
        None,
    )

    if torch.is_tensor(
        candidate
    ):

        return candidate

    raise RuntimeError(
        "Could not extract logits "
        "from Baby's forward output."
    )


# =============================================================================
# BUILD ONE FROZEN BATCH
# =============================================================================

def build_batch(
    step_data: dict[str, Any],
    device: torch.device,
):

    examples = step_data[
        "examples"
    ]

    if len(
        examples
    ) != BATCH_SIZE:

        raise RuntimeError(
            f"Step {step_data['step']}: "
            f"batch-size mismatch."
        )

    documents = [
        [
            int(token_id)
            for token_id
            in example[
                "full_document_token_ids"
            ]
        ]
        for example
        in examples
    ]

    lengths = {
        len(document)
        for document
        in documents
    }

    if lengths != {
        EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH
    }:

        raise RuntimeError(
            f"Step {step_data['step']}: "
            f"unexpected document lengths: "
            f"{sorted(lengths)}."
        )

    full_ids = torch.tensor(
        documents,
        dtype=torch.long,
        device=device,
    )

    inputs = (
        full_ids[
            :,
            :-1,
        ]
        .contiguous()
    )

    targets = (
        full_ids[
            :,
            1:,
        ]
        .contiguous()
    )

    if inputs.shape != (
        BATCH_SIZE,
        EXPECTED_CAUSAL_SEQUENCE_LENGTH,
    ):

        raise RuntimeError(
            f"Step {step_data['step']}: "
            f"unexpected input shape "
            f"{tuple(inputs.shape)}."
        )

    if targets.shape != (
        BATCH_SIZE,
        EXPECTED_CAUSAL_SEQUENCE_LENGTH,
    ):

        raise RuntimeError(
            f"Step {step_data['step']}: "
            f"unexpected target shape "
            f"{tuple(targets.shape)}."
        )

    answer_positions = torch.tensor(
        [
            int(
                example[
                    "answer_causal_position"
                ]
            )
            for example
            in examples
        ],
        dtype=torch.long,
        device=device,
    )

    correct_ids = torch.tensor(
        [
            int(
                example[
                    "target_token_id"
                ]
            )
            for example
            in examples
        ],
        dtype=torch.long,
        device=device,
    )

    distractor_ids = torch.tensor(
        [
            int(
                example[
                    "distractor_token_id"
                ]
            )
            for example
            in examples
        ],
        dtype=torch.long,
        device=device,
    )

    rows = torch.arange(
        BATCH_SIZE,
        device=device,
    )

    observed_answer_targets = (
        targets[
            rows,
            answer_positions,
        ]
    )

    if not torch.equal(
        observed_answer_targets,
        correct_ids,
    ):

        raise RuntimeError(
            f"Step {step_data['step']}: "
            f"causal answer labels "
            f"do not equal frozen "
            f"correct-token IDs."
        )

    return (
        inputs,
        targets,
        answer_positions,
        correct_ids,
        distractor_ids,
        examples,
    )


# =============================================================================
# EXACT T4-STYLE PER-TOKEN REPLACEMENT LOSS
# =============================================================================

def treatment5_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    answer_positions: torch.Tensor,
    correct_ids: torch.Tensor,
    distractor_ids: torch.Tensor,
):

    if logits.shape[:2] != (
        targets.shape
    ):

        raise RuntimeError(
            f"Logit/target shape mismatch: "
            f"{tuple(logits.shape)} vs "
            f"{tuple(targets.shape)}."
        )

    batch_size, sequence_length, vocab_size = (
        logits.shape
    )

    if batch_size != BATCH_SIZE:

        raise RuntimeError(
            "Unexpected model batch size."
        )

    # -------------------------------------------------------------------------
    # Ordinary tokenwise full-vocabulary CE everywhere.
    # -------------------------------------------------------------------------

    token_ce = F.cross_entropy(
        logits.reshape(
            -1,
            vocab_size,
        ),
        targets.reshape(
            -1
        ),
        reduction="none",
    ).reshape(
        batch_size,
        sequence_length,
    )

    rows = torch.arange(
        batch_size,
        dtype=torch.long,
        device=logits.device,
    )

    answer_logits = logits[
        rows,
        answer_positions,
        :
    ]

    correct_logits = answer_logits[
        rows,
        correct_ids,
    ]

    distractor_logits = answer_logits[
        rows,
        distractor_ids,
    ]

    # -------------------------------------------------------------------------
    # Membership term.
    # -------------------------------------------------------------------------

    all_vocab_partition = (
        torch.logsumexp(
            answer_logits,
            dim=-1,
        )
    )

    candidate_partition = (
        torch.logsumexp(
            torch.stack(
                (
                    correct_logits,
                    distractor_logits,
                ),
                dim=-1,
            ),
            dim=-1,
        )
    )

    membership_loss = (
        all_vocab_partition
        - candidate_partition
    )

    # -------------------------------------------------------------------------
    # Strict correct-vs-distractor margin.
    # -------------------------------------------------------------------------

    logit_delta = (
        correct_logits
        - distractor_logits
    )

    selector_loss = F.relu(
        MARGIN
        - logit_delta
    )

    replacement_loss = (
        membership_loss
        + selector_loss
    )

    # -------------------------------------------------------------------------
    # Literally replace original answer CE.
    # -------------------------------------------------------------------------

    substituted_losses = (
        token_ce.clone()
    )

    original_answer_ce = (
        substituted_losses[
            rows,
            answer_positions,
        ].clone()
    )

    substituted_losses[
        rows,
        answer_positions,
    ] = replacement_loss

    # Same style of reduction:
    # mean over every supervised causal position.
    total_loss = (
        substituted_losses.mean()
    )

    # -------------------------------------------------------------------------
    # Training diagnostics.
    # -------------------------------------------------------------------------

    with torch.no_grad():

        answer_probabilities = (
            torch.softmax(
                answer_logits,
                dim=-1,
            )
        )

        correct_probability = (
            answer_probabilities[
                rows,
                correct_ids,
            ]
        )

        distractor_probability = (
            answer_probabilities[
                rows,
                distractor_ids,
            ]
        )

        candidate_mass = (
            correct_probability
            + distractor_probability
        )

        correct_gt = (
            logit_delta > 0.0
        ).float()

        margin_satisfied = (
            logit_delta >= MARGIN
        ).float()

        top1 = (
            answer_logits.argmax(
                dim=-1
            )
            == correct_ids
        ).float()

        ranks = (
            (
                answer_logits
                > correct_logits.unsqueeze(
                    -1
                )
            )
            .sum(
                dim=-1
            )
            + 1
        ).float()

        nonanswer_mask = torch.ones(
            (
                batch_size,
                sequence_length,
            ),
            dtype=torch.bool,
            device=logits.device,
        )

        nonanswer_mask[
            rows,
            answer_positions,
        ] = False

        mean_nonanswer_ce = (
            token_ce[
                nonanswer_mask
            ].mean()
        )

    metrics = {
        "loss": float(
            total_loss.detach().item()
        ),
        "mean_membership_loss": float(
            membership_loss.mean().item()
        ),
        "mean_selector_loss": float(
            selector_loss.mean().item()
        ),
        "mean_answer_replacement_loss": float(
            replacement_loss.mean().item()
        ),
        "mean_original_answer_ce_removed": float(
            original_answer_ce.mean().item()
        ),
        "correct_gt_distractor_fraction": float(
            correct_gt.mean().item()
        ),
        "margin_satisfied_fraction": float(
            margin_satisfied.mean().item()
        ),
        "mean_candidate_mass": float(
            candidate_mass.mean().item()
        ),
        "mean_correct_probability": float(
            correct_probability.mean().item()
        ),
        "mean_distractor_probability": float(
            distractor_probability.mean().item()
        ),
        "mean_logit_delta": float(
            logit_delta.mean().item()
        ),
        "full_vocab_top1_fraction": float(
            top1.mean().item()
        ),
        "mean_target_rank": float(
            ranks.mean().item()
        ),
        "median_target_rank": float(
            ranks.median().item()
        ),
        "mean_nonanswer_ce": float(
            mean_nonanswer_ce.item()
        ),
    }

    return (
        total_loss,
        metrics,
    )


# =============================================================================
# SLOT-SPECIFIC METRICS
# =============================================================================

@torch.no_grad()
def slot_metrics(
    logits: torch.Tensor,
    answer_positions: torch.Tensor,
    correct_ids: torch.Tensor,
    distractor_ids: torch.Tensor,
    examples: list[dict[str, Any]],
) -> dict[str, float]:

    rows = torch.arange(
        BATCH_SIZE,
        device=logits.device,
    )

    answer_logits = logits[
        rows,
        answer_positions,
        :
    ]

    correct_logits = answer_logits[
        rows,
        correct_ids,
    ]

    distractor_logits = answer_logits[
        rows,
        distractor_ids,
    ]

    delta = (
        correct_logits
        - distractor_logits
    )

    output: dict[
        str,
        float,
    ] = {}

    for slot in (
        0,
        1,
    ):

        indices = [
            index
            for index, example
            in enumerate(
                examples
            )
            if int(
                example[
                    "query_slot"
                ]
            ) == slot
        ]

        if len(
            indices
        ) != PAIRS_PER_BATCH:

            raise RuntimeError(
                f"Expected "
                f"{PAIRS_PER_BATCH} "
                f"slot-{slot} examples "
                f"in batch; "
                f"got {len(indices)}."
            )

        index_tensor = torch.tensor(
            indices,
            dtype=torch.long,
            device=logits.device,
        )

        slot_delta = delta[
            index_tensor
        ]

        output[
            f"slot{slot}_correct_gt_distractor_fraction"
        ] = float(
            (
                slot_delta > 0.0
            )
            .float()
            .mean()
            .item()
        )

        output[
            f"slot{slot}_margin_satisfied_fraction"
        ] = float(
            (
                slot_delta >= MARGIN
            )
            .float()
            .mean()
            .item()
        )

        output[
            f"slot{slot}_mean_logit_delta"
        ] = float(
            slot_delta.mean().item()
        )

    return output


# =============================================================================
# SAVE CHECKPOINT
# =============================================================================

def save_checkpoint(
    model,
    optimizer,
    parameter_count: int,
    step: int,
    metrics: dict[str, Any],
    path: Path,
) -> str:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "experiment": (
            "DaveLM v0.9 Treatment #5 "
            "paired counterfactual "
            "query binding"
        ),
        "step": int(
            step
        ),
        "seed": int(
            SEED
        ),
        "schedule_seed": int(
            SCHEDULE_SEED
        ),
        "parameter_count": int(
            parameter_count
        ),
        "start_checkpoint_sha256": (
            EXPECTED_START_CHECKPOINT_SHA256
        ),
        "pair_pool_sha256": (
            EXPECTED_PAIR_POOL_SHA256
        ),
        "schedule_sha256": (
            EXPECTED_SCHEDULE_SHA256
        ),
        "objective": {
            "answer_loss": (
                "membership_loss + "
                "selector_loss"
            ),
            "membership_loss": (
                "logsumexp(full_vocab_logits) "
                "- logsumexp("
                "[correct_logit,"
                "distractor_logit])"
            ),
            "selector_loss": (
                "relu(0.5 - "
                "(correct_logit - "
                "distractor_logit))"
            ),
            "margin": (
                MARGIN
            ),
            "original_answer_ce_retained": (
                False
            ),
            "nonanswer_loss": (
                "ordinary_full_vocabulary_ce"
            ),
        },
        "optimizer": {
            "name": (
                "AdamW"
            ),
            "learning_rate": (
                LEARNING_RATE
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "grad_clip": (
                GRAD_CLIP
            ),
        },
        "metrics": (
            metrics
        ),

        # Use T4-compatible model-state naming.
        "model_state": (
            model.state_dict()
        ),

        "optimizer_state": (
            optimizer.state_dict()
        ),
    }

    temporary_path = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    torch.save(
        payload,
        temporary_path,
    )

    temporary_path.replace(
        path
    )

    return sha256_file(
        path
    )


# =============================================================================
# TRAIN
# =============================================================================

def train(
    device_name: str,
) -> None:

    run_start_time = (
        time.perf_counter()
    )

    schedule = (
        load_and_verify_schedule()
    )

    audit_schedule(
        schedule
    )

    # =========================================================================
    # DEVICE
    # =========================================================================

    header(
        "DEVICE"
    )

    if (
        device_name == "cuda"
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "CUDA/ROCm device requested "
            "but torch.cuda.is_available() "
            "returned False."
        )

    device = torch.device(
        device_name
    )

    print(
        f"Training device: "
        f"{device}"
    )

    if device.type == "cuda":

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # =========================================================================
    # CREATE OUTPUT DIRECTORIES
    # =========================================================================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Fresh invocation = fresh log.
    TRAINING_METRICS_PATH.write_text(
        "",
        encoding="utf-8",
    )

    # =========================================================================
    # IMPORT NATIVE MODEL MACHINERY
    # =========================================================================

    original_run = (
        import_native_module()
    )

    # =========================================================================
    # BUILD MODEL + LOAD VERIFIED START + FRESH ADAMW
    # =========================================================================

    (
        model,
        optimizer,
        parameter_count,
    ) = build_model_and_optimizer(
        original_run,
        device,
    )

    model.train()

    # =========================================================================
    # CONTRACT REPORT
    # =========================================================================

    header(
        "TREATMENT #5 TRAINING CONTRACT"
    )

    print(
        f"Seed: "
        f"{SEED}"
    )

    print(
        f"Schedule seed: "
        f"{SCHEDULE_SEED}"
    )

    print(
        f"Frozen schedule SHA:"
    )

    print(
        f"  "
        f"{EXPECTED_SCHEDULE_SHA256}"
    )

    print()

    print(
        f"Steps: "
        f"{MAX_STEPS}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Complete twin pairs / batch: "
        f"{PAIRS_PER_BATCH}"
    )

    print(
        f"Margin: "
        f"{MARGIN}"
    )

    print()

    print(
        "Architecture changes: NO"
    )

    print(
        "Optimizer changes: NO"
    )

    print(
        "Answer objective changes from T4: NO"
    )

    print(
        "Training-distribution change: YES"
    )

    print(
        "Positive-control evaluation: NO"
    )

    print(
        "Sealed evaluation: NO"
    )

    print()

    print(
        "TRAINING STARTING NOW"
    )

    # =========================================================================
    # COUNTERS
    # =========================================================================

    cumulative_answer_substitutions = 0

    cumulative_supervised_tokens = 0

    cumulative_nonanswer_tokens = 0

    final_metrics: dict[
        str,
        Any,
    ] | None = None

    # =========================================================================
    # TRAIN LOOP
    # =========================================================================

    for step_data in schedule[
        "steps_data"
    ]:

        step = int(
            step_data[
                "step"
            ]
        )

        step_start = (
            time.perf_counter()
        )

        (
            inputs,
            targets,
            answer_positions,
            correct_ids,
            distractor_ids,
            examples,
        ) = build_batch(
            step_data,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        output = model(
            inputs
        )

        logits = extract_logits(
            output
        )

        (
            loss,
            metrics,
        ) = treatment5_loss(
            logits,
            targets,
            answer_positions,
            correct_ids,
            distractor_ids,
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                f"Step {step}: "
                f"non-finite loss "
                f"{loss.item()}."
            )

        loss.backward()

        grad_norm_raw = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )
        )

        grad_norm = float(
            grad_norm_raw.item()
            if torch.is_tensor(
                grad_norm_raw
            )
            else grad_norm_raw
        )

        if not math.isfinite(
            grad_norm
        ):

            raise RuntimeError(
                f"Step {step}: "
                f"non-finite gradient norm "
                f"{grad_norm}."
            )

        optimizer.step()

        metrics.update(
            slot_metrics(
                logits.detach(),
                answer_positions,
                correct_ids,
                distractor_ids,
                examples,
            )
        )

        step_supervised_tokens = sum(
            int(
                example[
                    "supervised_token_count"
                ]
            )
            for example
            in examples
        )

        step_nonanswer_tokens = sum(
            int(
                example[
                    "nonanswer_supervised_token_count"
                ]
            )
            for example
            in examples
        )

        step_answer_substitutions = len(
            examples
        )

        cumulative_supervised_tokens += (
            step_supervised_tokens
        )

        cumulative_nonanswer_tokens += (
            step_nonanswer_tokens
        )

        cumulative_answer_substitutions += (
            step_answer_substitutions
        )

        step_seconds = (
            time.perf_counter()
            - step_start
        )

        metrics.update(
            {
                "step": int(
                    step
                ),
                "gradient_norm_pre_clip": float(
                    grad_norm
                ),
                "step_seconds": float(
                    step_seconds
                ),
                "cumulative_supervised_tokens": int(
                    cumulative_supervised_tokens
                ),
                "cumulative_nonanswer_tokens": int(
                    cumulative_nonanswer_tokens
                ),
                "cumulative_answer_substitutions": int(
                    cumulative_answer_substitutions
                ),
            }
        )

        append_jsonl(
            TRAINING_METRICS_PATH,
            metrics,
        )

        final_metrics = (
            metrics
        )

        # ---------------------------------------------------------------------
        # Human-readable log.
        # ---------------------------------------------------------------------

        if step in LOG_STEPS:

            print()

            print(
                f"[STEP "
                f"{step:4d}/"
                f"{MAX_STEPS}]"
            )

            print(
                f"  total loss:             "
                f"{metrics['loss']:.6f}"
            )

            print(
                f"  non-answer CE:          "
                f"{metrics['mean_nonanswer_ce']:.6f}"
            )

            print(
                f"  replacement loss:       "
                f"{metrics['mean_answer_replacement_loss']:.6f}"
            )

            print(
                f"  membership loss:        "
                f"{metrics['mean_membership_loss']:.6f}"
            )

            print(
                f"  selector loss:          "
                f"{metrics['mean_selector_loss']:.6f}"
            )

            print(
                f"  candidate mass:         "
                f"{metrics['mean_candidate_mass']:.6f}"
            )

            print(
                f"  correct > distractor:   "
                f"{metrics['correct_gt_distractor_fraction']:.6f}"
            )

            print(
                f"  margin satisfied:       "
                f"{metrics['margin_satisfied_fraction']:.6f}"
            )

            print(
                f"  mean logit delta:       "
                f"{metrics['mean_logit_delta']:+.6f}"
            )

            print(
                f"  full-vocab answer top1: "
                f"{metrics['full_vocab_top1_fraction']:.6f}"
            )

            print(
                f"  target rank mean:       "
                f"{metrics['mean_target_rank']:.3f}"
            )

            print(
                f"  slot0 pair-win:         "
                f"{metrics['slot0_correct_gt_distractor_fraction']:.6f}"
            )

            print(
                f"  slot1 pair-win:         "
                f"{metrics['slot1_correct_gt_distractor_fraction']:.6f}"
            )

            print(
                f"  slot0 margin:           "
                f"{metrics['slot0_margin_satisfied_fraction']:.6f}"
            )

            print(
                f"  slot1 margin:           "
                f"{metrics['slot1_margin_satisfied_fraction']:.6f}"
            )

            print(
                f"  grad norm pre-clip:     "
                f"{grad_norm:.6f}"
            )

            print(
                f"  step seconds:           "
                f"{step_seconds:.3f}"
            )

            print(
                f"  cumulative answers:     "
                f"{cumulative_answer_substitutions}"
            )

            print(
                f"  cumulative supervised:  "
                f"{cumulative_supervised_tokens}"
            )

        # ---------------------------------------------------------------------
        # Save periodic checkpoints.
        # ---------------------------------------------------------------------

        if step in CHECKPOINT_STEPS:

            numbered_path = (
                CHECKPOINT_ROOT
                / f"step_{step:04d}.pt"
            )

            numbered_sha = (
                save_checkpoint(
                    model,
                    optimizer,
                    parameter_count,
                    step,
                    metrics,
                    numbered_path,
                )
            )

            latest_sha = (
                save_checkpoint(
                    model,
                    optimizer,
                    parameter_count,
                    step,
                    metrics,
                    LATEST_CHECKPOINT,
                )
            )

            print()

            print(
                f"  Saved checkpoint: "
                f"{numbered_path}"
            )

            print(
                f"  SHA256: "
                f"{numbered_sha}"
            )

            print(
                f"  latest.pt SHA256: "
                f"{latest_sha}"
            )

    # =========================================================================
    # FINAL ACCOUNTING
    # =========================================================================

    if final_metrics is None:

        raise RuntimeError(
            "Training loop completed "
            "zero steps."
        )

    if cumulative_answer_substitutions != (
        EXPECTED_ANSWER_SUBSTITUTIONS
    ):

        raise RuntimeError(
            "Final answer-substitution "
            "count mismatch."
        )

    if cumulative_supervised_tokens != (
        EXPECTED_TOTAL_SUPERVISED
    ):

        raise RuntimeError(
            "Final supervised-token "
            "count mismatch."
        )

    if cumulative_nonanswer_tokens != (
        EXPECTED_NONANSWER_SUPERVISED
    ):

        raise RuntimeError(
            "Final non-answer-token "
            "count mismatch."
        )

    final_checkpoint_sha = (
        sha256_file(
            LATEST_CHECKPOINT
        )
    )

    total_runtime = (
        time.perf_counter()
        - run_start_time
    )

    result = {
        "experiment": (
            "DaveLM v0.9 Treatment #5"
        ),
        "treatment": (
            "paired counterfactual "
            "query binding"
        ),
        "classification": (
            "TRAINING COMPLETE — "
            "POSITIVE CONTROLS "
            "NOT YET EVALUATED"
        ),
        "seed": (
            SEED
        ),
        "schedule_seed": (
            SCHEDULE_SEED
        ),
        "steps": (
            MAX_STEPS
        ),
        "batch_size": (
            BATCH_SIZE
        ),
        "pairs_per_batch": (
            PAIRS_PER_BATCH
        ),
        "pair_presentations": (
            EXPECTED_PAIR_PRESENTATIONS
        ),
        "scheduled_examples": (
            EXPECTED_EXAMPLES
        ),
        "supervised_tokens": int(
            cumulative_supervised_tokens
        ),
        "nonanswer_supervised_tokens": int(
            cumulative_nonanswer_tokens
        ),
        "answer_substitutions": int(
            cumulative_answer_substitutions
        ),
        "starting_checkpoint": {
            "path": str(
                START_CHECKPOINT
            ),
            "sha256": (
                EXPECTED_START_CHECKPOINT_SHA256
            ),
        },
        "pair_pool": {
            "path": str(
                PAIR_POOL_PATH
            ),
            "sha256": (
                EXPECTED_PAIR_POOL_SHA256
            ),
        },
        "frozen_schedule": {
            "path": str(
                SCHEDULE_PATH
            ),
            "sha256": (
                EXPECTED_SCHEDULE_SHA256
            ),
        },
        "parameter_count": int(
            parameter_count
        ),
        "architecture_construction": (
            'original_run.build_model("untied")'
        ),
        "optimizer": {
            "name": (
                "AdamW"
            ),
            "learning_rate": (
                LEARNING_RATE
            ),
            "weight_decay": (
                WEIGHT_DECAY
            ),
            "gradient_clip": (
                GRAD_CLIP
            ),
            "scheduler": None,
        },
        "objective": {
            "answer": (
                "membership + selector"
            ),
            "membership": (
                "logsumexp(full vocab) - "
                "logsumexp(correct,distractor)"
            ),
            "selector": (
                "relu(0.5 - "
                "(correct-distractor))"
            ),
            "original_answer_ce_retained": (
                False
            ),
            "nonanswer": (
                "ordinary full-vocabulary CE"
            ),
        },
        "runtime_seconds": float(
            total_runtime
        ),
        "final_step_metrics": (
            final_metrics
        ),
        "candidate_checkpoint": {
            "path": str(
                LATEST_CHECKPOINT
            ),
            "sha256": (
                final_checkpoint_sha
            ),
        },
        "training_metrics": str(
            TRAINING_METRICS_PATH
        ),
        "positive_controls_evaluated": (
            False
        ),
        "sealed_axes_loaded": (
            False
        ),
        "sealed_evaluation": (
            False
        ),
    }

    write_json(
        TRAINING_RESULT_PATH,
        result,
    )

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    header(
        "TREATMENT #5 TRAINING COMPLETE"
    )

    print(
        f"Runtime: "
        f"{total_runtime:.2f} seconds"
    )

    print()

    print(
        "FINAL STEP METRICS"
    )

    print(
        f"  Loss: "
        f"{final_metrics['loss']:.6f}"
    )

    print(
        f"  Correct > distractor: "
        f"{final_metrics['correct_gt_distractor_fraction']:.6f}"
    )

    print(
        f"  Margin satisfied: "
        f"{final_metrics['margin_satisfied_fraction']:.6f}"
    )

    print(
        f"  Candidate mass: "
        f"{final_metrics['mean_candidate_mass']:.6f}"
    )

    print(
        f"  Membership loss: "
        f"{final_metrics['mean_membership_loss']:.6f}"
    )

    print(
        f"  Selector loss: "
        f"{final_metrics['mean_selector_loss']:.6f}"
    )

    print(
        f"  Replacement loss: "
        f"{final_metrics['mean_answer_replacement_loss']:.6f}"
    )

    print(
        f"  Mean target rank: "
        f"{final_metrics['mean_target_rank']:.3f}"
    )

    print(
        f"  Median target rank: "
        f"{final_metrics['median_target_rank']:.3f}"
    )

    print(
        f"  Slot-0 pair-win: "
        f"{final_metrics['slot0_correct_gt_distractor_fraction']:.6f}"
    )

    print(
        f"  Slot-1 pair-win: "
        f"{final_metrics['slot1_correct_gt_distractor_fraction']:.6f}"
    )

    print()

    print(
        "FINAL CHECKPOINT:"
    )

    print(
        f"  {LATEST_CHECKPOINT}"
    )

    print()

    print(
        "FINAL CHECKPOINT SHA256:"
    )

    print(
        f"  {final_checkpoint_sha}"
    )

    print()

    print(
        "TRAINING RESULT:"
    )

    print(
        f"  {TRAINING_RESULT_PATH}"
    )

    print()

    print(
        "METRICS LOG:"
    )

    print(
        f"  {TRAINING_METRICS_PATH}"
    )

    print()

    print(
        "Positive controls evaluated: NO"
    )

    print(
        "Sealed evaluation opened: NO"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "  1. Final training-retention audit."
    )

    print(
        "  2. Same authorized T4 "
        "positive-control gates."
    )

    print(
        "  3. Novel/shortcut pools remain "
        "sealed unless both gates pass."
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "DaveLM v0.9 Treatment #5 "
            "paired counterfactual "
            "query-binding trainer."
        )
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=(
            "cuda",
            "cpu",
        ),
        help=(
            "Training device. "
            "ROCm PyTorch uses "
            "'cuda' for AMD GPUs."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    train(
        args.device
    )


if __name__ == "__main__":

    main()