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
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# DaveLM v0.9 — Treatment #8
#
# CONTEXTUALIZED QUERY-TO-LOGIT PATHWAY
#
# GRANDPA-REVISED T8
#
# HYPOTHESIS
#
# T7 materially improved reciprocal selection, but its query branch used a
# context-free raw query-token embedding.
#
# T8 changes exactly ONE substantive thing:
#
#     T7:
#         q = raw token embedding at query position
#
#     T8:
#         q = final contextualized hidden state at query position
#
# The output pathway remains T7's additive mechanism:
#
#     query_bias = Linear(320 -> 1024, bias=True)(q)
#     adjusted_answer_logits = base_answer_logits + query_bias
#
# EVERYTHING ELSE REMAINS FROZEN:
#
# - same original one-map starting checkpoint
# - same paired counterfactual pool
# - same exact frozen schedule
# - same 1000 steps
# - same batch size 32 / 16 complete twin pairs
# - same AdamW
# - LR 3e-4
# - weight decay 0.05
# - grad clip 2.0
# - same membership + selector objective
# - same selector margin 0.5
# - same answer weight lambda 191
# - same reduction over all 192 causal positions
#
# NO POSITIVE CONTROLS.
# NO SEALED EVALUATION.
#
# Final classification comes ONLY from the exact 32,000-event retention audit.
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

SOURCE_ROOT = Path(
    r"C:\DaveLM-v0.9"
)

CADAVER_ROOT = Path(
    r"C:\DaveLM-CADAVER"
)

PAIR_POOL_PATH = (
    CADAVER_ROOT
    / "treatment5_counterfactual_pairs_seed8382"
    / "treatment5_full_document_pair_pool.json"
)

SCHEDULE_PATH = (
    CADAVER_ROOT
    / "treatment5_counterfactual_pairs_seed8382"
    / "treatment5_full_document_frozen_schedule.json"
)

START_CHECKPOINT = (
    SOURCE_ROOT
    / "experiments"
    / "minimal_contextual_binding"
    / "checkpoints"
    / "treatment_one_mapping"
    / "seed_8380"
    / "latest.pt"
)

PREFLIGHT_PATH = (
    CADAVER_ROOT
    / "treatment8_contextualized_query_logit_pathway_seed8380"
    / "treatment8_preflight_result.json"
)

OUTPUT_ROOT = (
    CADAVER_ROOT
    / "treatment8_contextualized_query_logit_pathway_seed8380"
)

CHECKPOINT_ROOT = (
    OUTPUT_ROOT
    / "checkpoints"
    / "contextualized_query_logit_pathway"
    / "seed_8380"
)

LATEST_CHECKPOINT = (
    CHECKPOINT_ROOT
    / "latest.pt"
)

METRICS_PATH = (
    OUTPUT_ROOT
    / "treatment8_training_metrics.jsonl"
)

RESULT_PATH = (
    OUTPUT_ROOT
    / "treatment8_training_result.json"
)


# =============================================================================
# FROZEN HASHES
# =============================================================================

EXPECTED_PAIR_POOL_SHA256 = (
    "0b31e39361f7a45dc4182c49ab4b4967"
    "bffe539cd2ff12e7b196eabfb3dac307"
)

EXPECTED_SCHEDULE_SHA256 = (
    "a4e64fd9d1b934aa440a5d036a3bec0a"
    "c7c9ac97d58c72fb45df6d8576bb7c12"
)

EXPECTED_START_CHECKPOINT_SHA256 = (
    "345984c52a06db5f988aaf4cd47963ee"
    "a0e9d77489cee94dbb10816af2f5443e"
)


# =============================================================================
# FROZEN TRAINING CONTRACT
# =============================================================================

SEED = 8380
SCHEDULE_SEED = 8382

MAX_STEPS = 1000
BATCH_SIZE = 32
PAIRS_PER_BATCH = 16

ANSWER_WEIGHT = 191.0
MARGIN = 0.5

LEARNING_RATE = 3.0e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 2.0

EXPECTED_BASE_PARAMETER_COUNT = 10_594_944
EXPECTED_QUERY_PROJECTION_PARAMETER_COUNT = 328_704
EXPECTED_TOTAL_PARAMETER_COUNT = 10_923_648

EXPECTED_HIDDEN_SIZE = 320
EXPECTED_VOCAB_SIZE = 1024

EXPECTED_PAIR_COUNT = 1_536
EXPECTED_PAIR_PRESENTATIONS = 16_000
EXPECTED_EXAMPLES = 32_000

EXPECTED_SLOT0 = 16_000
EXPECTED_SLOT1 = 16_000

EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH = 193
EXPECTED_CAUSAL_SEQUENCE_LENGTH = 192

EXPECTED_ANSWER_SUBSTITUTIONS = 32_000
EXPECTED_TOTAL_SUPERVISED = 6_144_000
EXPECTED_NONANSWER_SUPERVISED = 6_112_000

EXPECTED_PAIR_EXPOSURE_MIN = 10
EXPECTED_PAIR_EXPOSURE_MAX = 11


# =============================================================================
# HELPERS
# =============================================================================

def banner(
    title: str,
) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print()


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(
            message
        )


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


def verify_sha(
    path: Path,
    expected: str,
    label: str,
) -> str:
    require(
        path.is_file(),
        f"Missing {label}:\n{path}",
    )

    observed = sha256_file(
        path
    )

    print(
        f"{label}:"
    )

    print(
        f"  observed: {observed}"
    )

    print(
        f"  expected: {expected}"
    )

    require(
        observed == expected,
        f"{label} SHA mismatch.",
    )

    print(
        "  PASS"
    )

    print()

    return observed


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def append_jsonl(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )


def import_native_module():
    if str(
        SOURCE_ROOT
    ) not in sys.path:
        sys.path.insert(
            0,
            str(SOURCE_ROOT),
        )

    from experiments.two_mapping_contextual_binding import run as original_run

    return original_run


def extract_model_state(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    require(
        isinstance(
            checkpoint,
            dict,
        ),
        "Starting checkpoint is not a dictionary.",
    )

    for key in (
        "model_state",
        "model_state_dict",
        "model",
    ):
        candidate = checkpoint.get(
            key
        )

        if isinstance(
            candidate,
            dict,
        ):
            return candidate

    if checkpoint and all(
        torch.is_tensor(
            value
        )
        for value in checkpoint.values()
    ):
        return checkpoint

    raise RuntimeError(
        "Could not identify model state in starting checkpoint."
    )


def extract_logits(
    output: Any,
) -> torch.Tensor:
    if torch.is_tensor(
        output
    ):
        logits = output

    elif isinstance(
        output,
        dict,
    ) and torch.is_tensor(
        output.get(
            "logits"
        )
    ):
        logits = output[
            "logits"
        ]

    elif isinstance(
        output,
        (
            tuple,
            list,
        ),
    ):
        logits = None

        for candidate in output:
            if (
                torch.is_tensor(
                    candidate
                )
                and candidate.ndim == 3
            ):
                logits = candidate
                break

        require(
            logits is not None,
            "Could not locate logits in model output.",
        )

    elif torch.is_tensor(
        getattr(
            output,
            "logits",
            None,
        )
    ):
        logits = output.logits

    else:
        raise RuntimeError(
            "Could not extract logits from Baby output."
        )

    require(
        logits.ndim == 3,
        f"Expected logits [batch,time,vocab], got {tuple(logits.shape)}.",
    )

    return logits


def set_all_seeds(
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
# T8 MODEL WRAPPER
# =============================================================================

class Treatment8ContextualQueryModel(
    nn.Module
):
    def __init__(
        self,
        base_model: nn.Module,
    ):
        super().__init__()

        self.base_model = (
            base_model
        )

        token_embedding = getattr(
            base_model,
            "token_embedding",
            None,
        )

        require(
            isinstance(
                token_embedding,
                nn.Embedding,
            ),
            "Base model does not expose expected token_embedding.",
        )

        require(
            int(
                token_embedding.embedding_dim
            ) == EXPECTED_HIDDEN_SIZE,
            "Hidden-size mismatch.",
        )

        require(
            int(
                token_embedding.num_embeddings
            ) == EXPECTED_VOCAB_SIZE,
            "Vocabulary-size mismatch.",
        )

        final_norm = getattr(
            base_model,
            "final_norm",
            None,
        )

        require(
            isinstance(
                final_norm,
                nn.Module,
            ),
            "Base model does not expose expected final_norm.",
        )

        language_head = getattr(
            base_model,
            "language_head",
            None,
        )

        require(
            isinstance(
                language_head,
                nn.Linear,
            ),
            "Base model does not expose expected language_head.",
        )

        require(
            int(
                language_head.in_features
            ) == EXPECTED_HIDDEN_SIZE,
            "language_head hidden-size mismatch.",
        )

        require(
            int(
                language_head.out_features
            ) == EXPECTED_VOCAB_SIZE,
            "language_head vocabulary-size mismatch.",
        )

        # ---------------------------------------------------------------------
        # SAME ADDITIVE OUTPUT MODULE AS T7.
        #
        # Only its INPUT changes:
        #
        # T7:
        #     raw query-token embedding
        #
        # T8:
        #     final contextualized query hidden state
        # ---------------------------------------------------------------------

        self.query_projection = nn.Linear(
            EXPECTED_HIDDEN_SIZE,
            EXPECTED_VOCAB_SIZE,
            bias=True,
        )

        self._captured_final_hidden: torch.Tensor | None = None

        self._final_norm_hook = (
            self.base_model.final_norm.register_forward_hook(
                self._capture_final_hidden
            )
        )

    def _capture_final_hidden(
        self,
        module: nn.Module,
        module_inputs: tuple[Any, ...],
        module_output: Any,
    ) -> None:
        require(
            torch.is_tensor(
                module_output
            ),
            "final_norm output is not a tensor.",
        )

        require(
            module_output.ndim == 3,
            "final_norm output must be [batch,time,hidden].",
        )

        require(
            int(
                module_output.shape[
                    -1
                ]
            ) == EXPECTED_HIDDEN_SIZE,
            "Captured hidden size mismatch.",
        )

        # IMPORTANT:
        # Do NOT detach.
        #
        # Gradient from the query projection is allowed to flow back through
        # the contextualized query representation into Baby, exactly as T7's
        # attached raw embedding branch allowed gradient into Baby's embedding.
        self._captured_final_hidden = (
            module_output
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> Any:
        self._captured_final_hidden = (
            None
        )

        output = self.base_model(
            input_ids
        )

        require(
            self._captured_final_hidden
            is not None,
            "final_norm hook failed to capture hidden states.",
        )

        return output

    def get_contextual_query_hidden(
        self,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden = (
            self._captured_final_hidden
        )

        require(
            hidden is not None,
            "No contextual hidden state is currently captured.",
        )

        batch_size = int(
            hidden.shape[
                0
            ]
        )

        require(
            tuple(
                query_positions.shape
            ) == (
                batch_size,
            ),
            "query_positions shape mismatch.",
        )

        rows = torch.arange(
            batch_size,
            dtype=torch.long,
            device=hidden.device,
        )

        contextual_query_hidden = hidden[
            rows,
            query_positions,
            :
        ]

        require(
            tuple(
                contextual_query_hidden.shape
            ) == (
                batch_size,
                EXPECTED_HIDDEN_SIZE,
            ),
            "Contextual query-hidden shape mismatch.",
        )

        return contextual_query_hidden

    def get_query_bias(
        self,
        query_positions: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        contextual_query_hidden = (
            self.get_contextual_query_hidden(
                query_positions
            )
        )

        query_bias = (
            self.query_projection(
                contextual_query_hidden
            )
        )

        require(
            tuple(
                query_bias.shape
            ) == (
                int(
                    contextual_query_hidden.shape[
                        0
                    ]
                ),
                EXPECTED_VOCAB_SIZE,
            ),
            "Query-bias shape mismatch.",
        )

        return (
            query_bias,
            contextual_query_hidden,
        )


# =============================================================================
# FROZEN ARTIFACT AUDIT
# =============================================================================

def load_and_verify_frozen_artifacts() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, str],
]:
    banner(
        "VERIFYING FROZEN ARTIFACTS"
    )

    pair_pool_sha = verify_sha(
        PAIR_POOL_PATH,
        EXPECTED_PAIR_POOL_SHA256,
        "Pair pool SHA256",
    )

    schedule_sha = verify_sha(
        SCHEDULE_PATH,
        EXPECTED_SCHEDULE_SHA256,
        "Frozen schedule SHA256",
    )

    checkpoint_sha = verify_sha(
        START_CHECKPOINT,
        EXPECTED_START_CHECKPOINT_SHA256,
        "Starting checkpoint SHA256",
    )

    require(
        PREFLIGHT_PATH.is_file(),
        f"Missing passed T8 preflight:\n{PREFLIGHT_PATH}",
    )

    preflight = json.loads(
        PREFLIGHT_PATH.read_text(
            encoding="utf-8"
        )
    )

    require(
        preflight.get(
            "status"
        ) == "PREFLIGHT_PASS",
        "T8 preflight is not marked PREFLIGHT_PASS.",
    )

    pair_pool = json.loads(
        PAIR_POOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    schedule = json.loads(
        SCHEDULE_PATH.read_text(
            encoding="utf-8"
        )
    )

    return (
        pair_pool,
        schedule,
        {
            "pair_pool_sha256": (
                pair_pool_sha
            ),
            "schedule_sha256": (
                schedule_sha
            ),
            "start_checkpoint_sha256": (
                checkpoint_sha
            ),
        },
    )


# =============================================================================
# FROZEN SCHEDULE AUDIT
# =============================================================================

def audit_schedule(
    pair_pool: dict[str, Any],
    schedule: dict[str, Any],
) -> None:
    banner(
        "VERIFYING EXACT T8 TRAINING CONTRACT"
    )

    pairs = pair_pool.get(
        "pairs"
    )

    require(
        isinstance(
            pairs,
            list,
        ),
        "Pair pool 'pairs' is not a list.",
    )

    require(
        len(
            pairs
        ) == EXPECTED_PAIR_COUNT,
        "Pair count mismatch.",
    )

    require(
        int(
            schedule[
                "schedule_seed"
            ]
        ) == SCHEDULE_SEED,
        "Schedule seed mismatch.",
    )

    require(
        int(
            schedule[
                "steps"
            ]
        ) == MAX_STEPS,
        "Step count mismatch.",
    )

    require(
        int(
            schedule[
                "batch_size"
            ]
        ) == BATCH_SIZE,
        "Batch size mismatch.",
    )

    require(
        int(
            schedule[
                "pairs_per_batch"
            ]
        ) == PAIRS_PER_BATCH,
        "Pairs/batch mismatch.",
    )

    require(
        int(
            schedule[
                "pair_presentations"
            ]
        ) == EXPECTED_PAIR_PRESENTATIONS,
        "Pair-presentation count mismatch.",
    )

    require(
        int(
            schedule[
                "scheduled_examples"
            ]
        ) == EXPECTED_EXAMPLES,
        "Scheduled example count mismatch.",
    )

    require(
        int(
            schedule[
                "slot_0_presentations"
            ]
        ) == EXPECTED_SLOT0,
        "Slot-0 count mismatch.",
    )

    require(
        int(
            schedule[
                "slot_1_presentations"
            ]
        ) == EXPECTED_SLOT1,
        "Slot-1 count mismatch.",
    )

    require(
        int(
            schedule[
                "answer_substitutions"
            ]
        ) == EXPECTED_ANSWER_SUBSTITUTIONS,
        "Answer-substitution count mismatch.",
    )

    require(
        int(
            schedule[
                "total_supervised_tokens"
            ]
        ) == EXPECTED_TOTAL_SUPERVISED,
        "Total supervised-position count mismatch.",
    )

    require(
        int(
            schedule[
                "nonanswer_supervised_tokens"
            ]
        ) == EXPECTED_NONANSWER_SUPERVISED,
        "Nonanswer-position count mismatch.",
    )

    steps_data = schedule.get(
        "steps_data"
    )

    require(
        isinstance(
            steps_data,
            list,
        ),
        "steps_data is not a list.",
    )

    require(
        len(
            steps_data
        ) == MAX_STEPS,
        "steps_data length mismatch.",
    )

    observed_events = 0
    observed_slot0 = 0
    observed_slot1 = 0

    pair_exposures: dict[
        str,
        int,
    ] = {}

    for expected_step, step_data in enumerate(
        steps_data,
        start=1,
    ):
        require(
            int(
                step_data[
                    "step"
                ]
            ) == expected_step,
            "Frozen step ordering mismatch.",
        )

        examples = step_data.get(
            "examples"
        )

        require(
            isinstance(
                examples,
                list,
            ),
            f"Step {expected_step}: examples missing.",
        )

        require(
            len(
                examples
            ) == BATCH_SIZE,
            f"Step {expected_step}: batch size mismatch.",
        )

        for offset in range(
            0,
            BATCH_SIZE,
            2,
        ):
            twin_a = examples[
                offset
            ]

            twin_b = examples[
                offset + 1
            ]

            require(
                str(
                    twin_a[
                        "pair_id"
                    ]
                ) == str(
                    twin_b[
                        "pair_id"
                    ]
                ),
                f"Step {expected_step}: pair split.",
            )

            require(
                str(
                    twin_a[
                        "twin"
                    ]
                ) == "A",
                f"Step {expected_step}: expected Twin A.",
            )

            require(
                str(
                    twin_b[
                        "twin"
                    ]
                ) == "B",
                f"Step {expected_step}: expected Twin B.",
            )

            require(
                int(
                    twin_a[
                        "query_slot"
                    ]
                ) == 0,
                f"Step {expected_step}: Twin A slot mismatch.",
            )

            require(
                int(
                    twin_b[
                        "query_slot"
                    ]
                ) == 1,
                f"Step {expected_step}: Twin B slot mismatch.",
            )

            require(
                int(
                    twin_a[
                        "target_token_id"
                    ]
                ) == int(
                    twin_b[
                        "distractor_token_id"
                    ]
                ),
                f"Step {expected_step}: A target != B distractor.",
            )

            require(
                int(
                    twin_b[
                        "target_token_id"
                    ]
                ) == int(
                    twin_a[
                        "distractor_token_id"
                    ]
                ),
                f"Step {expected_step}: B target != A distractor.",
            )

            pair_id = str(
                twin_a[
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
            require(
                len(
                    example[
                        "full_document_token_ids"
                    ]
                ) == EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH,
                f"Step {expected_step}: document length mismatch.",
            )

            require(
                int(
                    example[
                        "supervised_token_count"
                    ]
                ) == EXPECTED_CAUSAL_SEQUENCE_LENGTH,
                f"Step {expected_step}: supervised count mismatch.",
            )

            require(
                int(
                    example[
                        "nonanswer_supervised_token_count"
                    ]
                ) == 191,
                f"Step {expected_step}: nonanswer count mismatch.",
            )

            require(
                int(
                    example[
                        "answer_causal_position"
                    ]
                ) == (
                    int(
                        example[
                            "answer_token_index"
                        ]
                    )
                    - 1
                ),
                f"Step {expected_step}: answer causal position mismatch.",
            )

            require(
                int(
                    example[
                        "query_difference_position"
                    ]
                ) < int(
                    example[
                        "answer_token_index"
                    ]
                ),
                f"Step {expected_step}: query not before answer.",
            )

            require(
                int(
                    example[
                        "target_token_id"
                    ]
                ) != int(
                    example[
                        "distractor_token_id"
                    ]
                ),
                f"Step {expected_step}: target == distractor.",
            )

            slot = int(
                example[
                    "query_slot"
                ]
            )

            if slot == 0:
                observed_slot0 += 1

            elif slot == 1:
                observed_slot1 += 1

            else:
                raise RuntimeError(
                    "Invalid query slot."
                )

            observed_events += 1

    require(
        observed_events == EXPECTED_EXAMPLES,
        "Observed event count mismatch.",
    )

    require(
        observed_slot0 == EXPECTED_SLOT0,
        "Observed slot-0 count mismatch.",
    )

    require(
        observed_slot1 == EXPECTED_SLOT1,
        "Observed slot-1 count mismatch.",
    )

    require(
        len(
            pair_exposures
        ) == EXPECTED_PAIR_COUNT,
        "Not every pair receives exposure.",
    )

    require(
        min(
            pair_exposures.values()
        ) == EXPECTED_PAIR_EXPOSURE_MIN,
        "Minimum pair exposure mismatch.",
    )

    require(
        max(
            pair_exposures.values()
        ) == EXPECTED_PAIR_EXPOSURE_MAX,
        "Maximum pair exposure mismatch.",
    )

    print(
        f"Steps: {MAX_STEPS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Complete pairs / batch: {PAIRS_PER_BATCH}"
    )

    print(
        f"Events: {observed_events}"
    )

    print(
        f"Slot 0: {observed_slot0}"
    )

    print(
        f"Slot 1: {observed_slot1}"
    )

    print(
        f"Pair exposure range: "
        f"{min(pair_exposures.values())}-"
        f"{max(pair_exposures.values())}"
    )

    print(
        f"Total supervised positions: "
        f"{EXPECTED_TOTAL_SUPERVISED}"
    )

    print(
        f"Ordinary non-answer CE positions: "
        f"{EXPECTED_NONANSWER_SUPERVISED}"
    )

    print()

    print(
        "Frozen training contract: PASS"
    )


# =============================================================================
# BUILD EXACT T8 MODEL
# =============================================================================

def build_t8_model(
    device: torch.device,
) -> Treatment8ContextualQueryModel:
    banner(
        "BUILDING T8 MODEL"
    )

    original_run = import_native_module()

    build_model = getattr(
        original_run,
        "build_model",
        None,
    )

    require(
        callable(
            build_model
        ),
        "Native build_model() unavailable.",
    )

    # IMPORTANT:
    #
    # Same deterministic ordering as passed preflight:
    #
    # 1. seed
    # 2. build base model
    # 3. load frozen base checkpoint
    # 4. construct T8 query projection
    #
    # Loading weights does not consume RNG, so the new projection receives the
    # deterministic initialization implied by this exact sequence.
    set_all_seeds(
        SEED
    )

    base_model = build_model(
        "untied"
    )

    checkpoint = torch.load(
        START_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    start_state = extract_model_state(
        checkpoint
    )

    missing, unexpected = (
        base_model.load_state_dict(
            start_state,
            strict=False,
        )
    )

    require(
        not missing,
        f"Missing starting checkpoint keys: {missing}",
    )

    require(
        not unexpected,
        f"Unexpected starting checkpoint keys: {unexpected}",
    )

    base_parameter_count = sum(
        p.numel()
        for p in base_model.parameters()
    )

    require(
        base_parameter_count == EXPECTED_BASE_PARAMETER_COUNT,
        "Base parameter count mismatch.",
    )

    model = Treatment8ContextualQueryModel(
        base_model
    )

    projection_parameter_count = sum(
        p.numel()
        for p in model.query_projection.parameters()
    )

    total_parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    require(
        projection_parameter_count
        == EXPECTED_QUERY_PROJECTION_PARAMETER_COUNT,
        "Query projection parameter count mismatch.",
    )

    require(
        total_parameter_count == EXPECTED_TOTAL_PARAMETER_COUNT,
        "Total parameter count mismatch.",
    )

    model = model.to(
        device
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(
        f"Base parameters: {base_parameter_count:,}"
    )

    print(
        f"Query-projection parameters: "
        f"{projection_parameter_count:,}"
    )

    print(
        f"Total parameters: {total_parameter_count:,}"
    )

    print()

    print(
        'Base builder: original_run.build_model("untied")'
    )

    print(
        "Start checkpoint: VERIFIED / LOADED"
    )

    print(
        "Context source: final_norm output"
    )

    print(
        "Query representation: contextual hidden[row, query_position, :]"
    )

    print(
        "Projection: Linear(320 -> 1024, bias=True)"
    )

    print(
        "Adjusted answer logits: base_answer_logits + query_bias"
    )

    print()

    print(
        "T8 model construction: PASS"
    )

    return model


# =============================================================================
# BATCH CONSTRUCTION
# =============================================================================

def build_batch(
    examples: list[
        dict[str, Any]
    ],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    require(
        len(
            examples
        ) == BATCH_SIZE,
        "Training batch does not contain exactly 32 examples.",
    )

    documents = torch.tensor(
        [
            [
                int(x)
                for x in example[
                    "full_document_token_ids"
                ]
            ]
            for example in examples
        ],
        dtype=torch.long,
        device=device,
    )

    require(
        tuple(
            documents.shape
        ) == (
            BATCH_SIZE,
            EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH,
        ),
        "Full-document batch shape mismatch.",
    )

    inputs = documents[
        :,
        :-1,
    ].contiguous()

    labels = documents[
        :,
        1:,
    ].contiguous()

    query_positions = torch.tensor(
        [
            int(
                example[
                    "query_difference_position"
                ]
            )
            for example in examples
        ],
        dtype=torch.long,
        device=device,
    )

    answer_positions = torch.tensor(
        [
            int(
                example[
                    "answer_causal_position"
                ]
            )
            for example in examples
        ],
        dtype=torch.long,
        device=device,
    )

    target_ids = torch.tensor(
        [
            int(
                example[
                    "target_token_id"
                ]
            )
            for example in examples
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
            for example in examples
        ],
        dtype=torch.long,
        device=device,
    )

    query_slots = torch.tensor(
        [
            int(
                example[
                    "query_slot"
                ]
            )
            for example in examples
        ],
        dtype=torch.long,
        device=device,
    )

    rows = torch.arange(
        BATCH_SIZE,
        dtype=torch.long,
        device=device,
    )

    require(
        bool(
            torch.equal(
                labels[
                    rows,
                    answer_positions,
                ],
                target_ids,
            )
        ),
        "Causal training target does not match frozen target ID.",
    )

    return {
        "inputs": (
            inputs
        ),
        "labels": (
            labels
        ),
        "query_positions": (
            query_positions
        ),
        "answer_positions": (
            answer_positions
        ),
        "target_ids": (
            target_ids
        ),
        "distractor_ids": (
            distractor_ids
        ),
        "query_slots": (
            query_slots
        ),
    }


# =============================================================================
# LOSS + METRICS
# =============================================================================

def compute_training_loss_and_metrics(
    model: Treatment8ContextualQueryModel,
    batch: dict[str, torch.Tensor],
) -> tuple[
    torch.Tensor,
    dict[str, float],
]:
    inputs = batch[
        "inputs"
    ]

    labels = batch[
        "labels"
    ]

    query_positions = batch[
        "query_positions"
    ]

    answer_positions = batch[
        "answer_positions"
    ]

    target_ids = batch[
        "target_ids"
    ]

    distractor_ids = batch[
        "distractor_ids"
    ]

    query_slots = batch[
        "query_slots"
    ]

    batch_size = int(
        inputs.shape[
            0
        ]
    )

    require(
        batch_size == BATCH_SIZE,
        "Unexpected batch size.",
    )

    output = model(
        inputs
    )

    base_logits = extract_logits(
        output
    )

    require(
        tuple(
            base_logits.shape
        ) == (
            BATCH_SIZE,
            EXPECTED_CAUSAL_SEQUENCE_LENGTH,
            EXPECTED_VOCAB_SIZE,
        ),
        "Base logits shape mismatch.",
    )

    query_bias, contextual_query_hidden = (
        model.get_query_bias(
            query_positions
        )
    )

    rows = torch.arange(
        batch_size,
        dtype=torch.long,
        device=inputs.device,
    )

    base_answer_logits = base_logits[
        rows,
        answer_positions,
        :
    ]

    adjusted_answer_logits = (
        base_answer_logits
        + query_bias
    )

    # -------------------------------------------------------------------------
    # ORDINARY FULL-VOCAB CAUSAL CE OVER ALL 192 POSITIONS.
    # -------------------------------------------------------------------------

    token_ce = F.cross_entropy(
        base_logits.reshape(
            -1,
            EXPECTED_VOCAB_SIZE,
        ),
        labels.reshape(
            -1,
        ),
        reduction="none",
    ).view(
        batch_size,
        EXPECTED_CAUSAL_SEQUENCE_LENGTH,
    )

    # -------------------------------------------------------------------------
    # ANSWER REPLACEMENT OBJECTIVE.
    #
    # membership:
    #
    #   logsumexp(full vocab)
    #       -
    #   logsumexp(correct,distractor)
    #
    # selector:
    #
    #   relu(
    #       margin - (correct_logit - distractor_logit)
    #   )
    # -------------------------------------------------------------------------

    correct_logits = adjusted_answer_logits[
        rows,
        target_ids,
    ]

    distractor_logits = adjusted_answer_logits[
        rows,
        distractor_ids,
    ]

    candidate_logits = torch.stack(
        [
            correct_logits,
            distractor_logits,
        ],
        dim=1,
    )

    membership_loss_per_example = (
        torch.logsumexp(
            adjusted_answer_logits,
            dim=1,
        )
        - torch.logsumexp(
            candidate_logits,
            dim=1,
        )
    )

    logit_delta = (
        correct_logits
        - distractor_logits
    )

    selector_loss_per_example = F.relu(
        MARGIN
        - logit_delta
    )

    raw_replacement_per_example = (
        membership_loss_per_example
        + selector_loss_per_example
    )

    weighted_replacement_per_example = (
        ANSWER_WEIGHT
        * raw_replacement_per_example
    )

    # -------------------------------------------------------------------------
    # EXACT T6/T7 REDUCTION:
    #
    # Start with ordinary CE over all 192 causal positions.
    #
    # At each answer position:
    #
    #   remove ordinary answer CE
    #   insert lambda * (membership + selector)
    #
    # Then mean over entire B x 192 position table.
    # -------------------------------------------------------------------------

    loss_table = token_ce.clone()

    original_answer_ce = loss_table[
        rows,
        answer_positions,
    ]

    loss_table[
        rows,
        answer_positions,
    ] = weighted_replacement_per_example

    total_loss = loss_table.mean()

    # -------------------------------------------------------------------------
    # OBSERVATIONAL METRICS ONLY.
    # They do NOT change training.
    # -------------------------------------------------------------------------

    with torch.no_grad():
        nonanswer_mask = torch.ones_like(
            token_ce,
            dtype=torch.bool,
        )

        nonanswer_mask[
            rows,
            answer_positions,
        ] = False

        nonanswer_ce = token_ce[
            nonanswer_mask
        ].mean()

        ordinary_answer_ce = (
            original_answer_ce.mean()
        )

        membership_loss = (
            membership_loss_per_example.mean()
        )

        selector_loss = (
            selector_loss_per_example.mean()
        )

        raw_replacement_loss = (
            raw_replacement_per_example.mean()
        )

        weighted_replacement_loss = (
            weighted_replacement_per_example.mean()
        )

        correct_gt = (
            logit_delta > 0.0
        ).float().mean()

        margin_pass = (
            logit_delta >= MARGIN
        ).float().mean()

        mean_delta = (
            logit_delta.mean()
        )

        probabilities = F.softmax(
            adjusted_answer_logits,
            dim=1,
        )

        target_probability = probabilities[
            rows,
            target_ids,
        ]

        distractor_probability = probabilities[
            rows,
            distractor_ids,
        ]

        candidate_mass = (
            target_probability
            + distractor_probability
        ).mean()

        top1_ids = adjusted_answer_logits.argmax(
            dim=1
        )

        top1 = (
            top1_ids == target_ids
        ).float().mean()

        target_logit_column = correct_logits.unsqueeze(
            1
        )

        target_rank = (
            (
                adjusted_answer_logits
                > target_logit_column
            )
            .sum(
                dim=1
            )
            + 1
        ).float()

        rank_mean = (
            target_rank.mean()
        )

        rank_median = (
            target_rank.median()
        )

        slot0_mask = (
            query_slots == 0
        )

        slot1_mask = (
            query_slots == 1
        )

        require(
            int(
                slot0_mask.sum().item()
            ) == 16,
            "Batch does not contain exactly 16 slot-0 examples.",
        )

        require(
            int(
                slot1_mask.sum().item()
            ) == 16,
            "Batch does not contain exactly 16 slot-1 examples.",
        )

        slot0_correct = (
            (
                logit_delta[
                    slot0_mask
                ] > 0.0
            )
            .float()
            .mean()
        )

        slot1_correct = (
            (
                logit_delta[
                    slot1_mask
                ] > 0.0
            )
            .float()
            .mean()
        )

        slot0_margin = (
            (
                logit_delta[
                    slot0_mask
                ] >= MARGIN
            )
            .float()
            .mean()
        )

        slot1_margin = (
            (
                logit_delta[
                    slot1_mask
                ] >= MARGIN
            )
            .float()
            .mean()
        )

        mean_abs_query_bias = (
            query_bias.abs().mean()
        )

        query_bias_l2 = (
            torch.linalg.vector_norm(
                query_bias,
                dim=1,
            ).mean()
        )

        contextual_query_l2 = (
            torch.linalg.vector_norm(
                contextual_query_hidden,
                dim=1,
            ).mean()
        )

        # Exact contribution to final scalar loss.
        total_positions = float(
            BATCH_SIZE
            * EXPECTED_CAUSAL_SEQUENCE_LENGTH
        )

        nonanswer_contribution = (
            token_ce[
                nonanswer_mask
            ].sum()
            / total_positions
        )

        weighted_answer_contribution = (
            weighted_replacement_per_example.sum()
            / total_positions
        )

    metrics = {
        "total_loss": (
            float(
                total_loss.detach().item()
            )
        ),
        "nonanswer_ce": (
            float(
                nonanswer_ce.item()
            )
        ),
        "ordinary_answer_ce": (
            float(
                ordinary_answer_ce.item()
            )
        ),
        "raw_answer_replacement": (
            float(
                raw_replacement_loss.item()
            )
        ),
        "weighted_answer_replacement": (
            float(
                weighted_replacement_loss.item()
            )
        ),
        "nonanswer_contribution": (
            float(
                nonanswer_contribution.item()
            )
        ),
        "weighted_answer_contribution": (
            float(
                weighted_answer_contribution.item()
            )
        ),
        "membership_loss": (
            float(
                membership_loss.item()
            )
        ),
        "selector_loss": (
            float(
                selector_loss.item()
            )
        ),
        "candidate_mass": (
            float(
                candidate_mass.item()
            )
        ),
        "correct_gt_distractor": (
            float(
                correct_gt.item()
            )
        ),
        "margin_pass": (
            float(
                margin_pass.item()
            )
        ),
        "mean_logit_delta": (
            float(
                mean_delta.item()
            )
        ),
        "full_vocab_top1": (
            float(
                top1.item()
            )
        ),
        "target_rank_mean": (
            float(
                rank_mean.item()
            )
        ),
        "target_rank_median": (
            float(
                rank_median.item()
            )
        ),
        "slot0_correct_gt_distractor": (
            float(
                slot0_correct.item()
            )
        ),
        "slot1_correct_gt_distractor": (
            float(
                slot1_correct.item()
            )
        ),
        "slot0_margin_pass": (
            float(
                slot0_margin.item()
            )
        ),
        "slot1_margin_pass": (
            float(
                slot1_margin.item()
            )
        ),
        "mean_abs_query_bias": (
            float(
                mean_abs_query_bias.item()
            )
        ),
        "query_bias_l2": (
            float(
                query_bias_l2.item()
            )
        ),
        "contextual_query_hidden_l2": (
            float(
                contextual_query_l2.item()
            )
        ),
    }

    return (
        total_loss,
        metrics,
    )


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(
    model: Treatment8ContextualQueryModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    hashes: dict[str, str],
    trainer_sha256: str,
) -> str:
    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "treatment": (
            "DaveLM v0.9 Treatment #8 — "
            "contextualized query-to-logit pathway"
        ),
        "step": (
            int(
                step
            )
        ),
        "model_state": (
            model.state_dict()
        ),
        "optimizer_state": (
            optimizer.state_dict()
        ),
        "seed": (
            SEED
        ),
        "schedule_seed": (
            SCHEDULE_SEED
        ),
        "trainer_sha256": (
            trainer_sha256
        ),
        "pair_pool_sha256": (
            hashes[
                "pair_pool_sha256"
            ]
        ),
        "schedule_sha256": (
            hashes[
                "schedule_sha256"
            ]
        ),
        "start_checkpoint_sha256": (
            hashes[
                "start_checkpoint_sha256"
            ]
        ),
        "base_parameter_count": (
            EXPECTED_BASE_PARAMETER_COUNT
        ),
        "query_projection_parameter_count": (
            EXPECTED_QUERY_PROJECTION_PARAMETER_COUNT
        ),
        "parameter_count": (
            EXPECTED_TOTAL_PARAMETER_COUNT
        ),
        "query_representation": (
            "base_model.final_norm output at frozen query position"
        ),
        "query_projection": (
            "Linear(320 -> 1024, bias=True)"
        ),
        "answer_logit_adjustment": (
            "base_answer_logits + query_bias"
        ),
        "answer_weight": (
            ANSWER_WEIGHT
        ),
        "margin": (
            MARGIN
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
        "positive_controls_evaluated": (
            False
        ),
        "sealed_evaluation_opened": (
            False
        ),
    }

    torch.save(
        payload,
        LATEST_CHECKPOINT,
    )

    return sha256_file(
        LATEST_CHECKPOINT
    )


# =============================================================================
# LOGGING
# =============================================================================

def print_step_metrics(
    step: int,
    metrics: dict[str, float],
    grad_norm: float,
    cumulative_answers: int,
    cumulative_supervised: int,
) -> None:
    print()
    print(
        f"STEP {step}/{MAX_STEPS}"
    )

    print(
        f"  total loss:                    "
        f"{metrics['total_loss']:.6f}"
    )

    print(
        f"  nonanswer CE:                  "
        f"{metrics['nonanswer_ce']:.6f}"
    )

    print(
        f"  raw answer replacement:        "
        f"{metrics['raw_answer_replacement']:.6f}"
    )

    print(
        f"  weighted answer replacement:   "
        f"{metrics['weighted_answer_replacement']:.6f}"
    )

    print(
        f"  nonanswer contribution:        "
        f"{metrics['nonanswer_contribution']:.6f}"
    )

    print(
        f"  weighted answer contribution:  "
        f"{metrics['weighted_answer_contribution']:.6f}"
    )

    print(
        f"  membership loss:               "
        f"{metrics['membership_loss']:.6f}"
    )

    print(
        f"  selector loss:                 "
        f"{metrics['selector_loss']:.6f}"
    )

    print(
        f"  candidate mass:                "
        f"{metrics['candidate_mass']:.6f}"
    )

    print(
        f"  correct > distractor:          "
        f"{metrics['correct_gt_distractor']:.6f}"
    )

    print(
        f"  margin >= {MARGIN}:             "
        f"{metrics['margin_pass']:.6f}"
    )

    print(
        f"  mean logit delta:              "
        f"{metrics['mean_logit_delta']:+.6f}"
    )

    print(
        f"  full-vocab top1:               "
        f"{metrics['full_vocab_top1']:.6f}"
    )

    print(
        f"  target rank mean:              "
        f"{metrics['target_rank_mean']:.3f}"
    )

    print(
        f"  target rank median:            "
        f"{metrics['target_rank_median']:.3f}"
    )

    print(
        f"  slot0 pair-win:                "
        f"{metrics['slot0_correct_gt_distractor']:.6f}"
    )

    print(
        f"  slot1 pair-win:                "
        f"{metrics['slot1_correct_gt_distractor']:.6f}"
    )

    print(
        f"  slot0 margin:                  "
        f"{metrics['slot0_margin_pass']:.6f}"
    )

    print(
        f"  slot1 margin:                  "
        f"{metrics['slot1_margin_pass']:.6f}"
    )

    print(
        f"  mean |query bias|:             "
        f"{metrics['mean_abs_query_bias']:.6f}"
    )

    print(
        f"  query-bias L2:                 "
        f"{metrics['query_bias_l2']:.6f}"
    )

    print(
        f"  contextual-query hidden L2:    "
        f"{metrics['contextual_query_hidden_l2']:.6f}"
    )

    print(
        f"  grad norm:                     "
        f"{grad_norm:.6f}"
    )

    print(
        f"  cumulative answers:            "
        f"{cumulative_answers}"
    )

    print(
        f"  cumulative supervised:         "
        f"{cumulative_supervised}"
    )

    print(
        flush=True
    )


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "DaveLM v0.9 Treatment #8 — "
            "contextualized query-to-logit pathway"
        )
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=[
            "cuda",
            "cpu",
        ],
    )

    args = parser.parse_args()

    start_wall = time.perf_counter()

    banner(
        "DAVELM v0.9 — TREATMENT #8"
    )

    print(
        "CONTEXTUALIZED QUERY-TO-LOGIT PATHWAY"
    )

    print()

    print(
        "Grandpa prescription:"
    )

    print(
        "  Give the query representation context."
    )

    print()

    print(
        "No positive controls."
    )

    print(
        "No sealed evaluation."
    )

    print(
        "Final classification requires exact 32,000-event retention audit."
    )

    # =========================================================================
    # DEVICE
    # =========================================================================

    if args.device == "cuda":
        require(
            torch.cuda.is_available(),
            "CUDA/ROCm device requested but torch.cuda.is_available() is false.",
        )

        device = torch.device(
            "cuda"
        )

    else:
        device = torch.device(
            "cpu"
        )

    banner(
        "DEVICE"
    )

    print(
        f"Training device: {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # =========================================================================
    # ARTIFACTS
    # =========================================================================

    (
        pair_pool,
        schedule,
        hashes,
    ) = load_and_verify_frozen_artifacts()

    audit_schedule(
        pair_pool,
        schedule,
    )

    # =========================================================================
    # TRAINER IDENTITY
    # =========================================================================

    trainer_path = Path(
        __file__
    ).resolve()

    trainer_sha256 = sha256_file(
        trainer_path
    )

    banner(
        "TRAINER IDENTITY"
    )

    print(
        f"Trainer:"
    )

    print(
        f"  {trainer_path}"
    )

    print()

    print(
        f"Trainer SHA256:"
    )

    print(
        f"  {trainer_sha256}"
    )

    # =========================================================================
    # CLEAN OUTPUT METRICS
    # =========================================================================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # A new exact run starts a new metrics log.
    METRICS_PATH.write_text(
        "",
        encoding="utf-8",
    )

    # =========================================================================
    # MODEL
    # =========================================================================

    model = build_t8_model(
        device
    )

    model.train()

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    banner(
        "OPTIMIZER"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        "Optimizer: AdamW"
    )

    print(
        f"LR: {LEARNING_RATE}"
    )

    print(
        f"Weight decay: {WEIGHT_DECAY}"
    )

    print(
        f"Grad clip: {GRAD_CLIP}"
    )

    print(
        "Parameters optimized: ALL T8 parameters"
    )

    print(
        "  base Baby: YES"
    )

    print(
        "  query projection: YES"
    )

    # =========================================================================
    # CONTRACT
    # =========================================================================

    banner(
        "FROZEN T8 TRAINING CONTRACT"
    )

    print(
        f"Seed: {SEED}"
    )

    print(
        f"Schedule seed: {SCHEDULE_SEED}"
    )

    print(
        f"Steps: {MAX_STEPS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Complete pairs/batch: {PAIRS_PER_BATCH}"
    )

    print(
        f"Answer weight lambda: {ANSWER_WEIGHT}"
    )

    print(
        f"Selector margin: {MARGIN}"
    )

    print()

    print(
        "Query source:"
    )

    print(
        "  FINAL contextual hidden state at frozen query position"
    )

    print()

    print(
        "Answer objective:"
    )

    print(
        "  membership + selector"
    )

    print()

    print(
        "Per-position replacement:"
    )

    print(
        "  ordinary answer CE -> 191 * (membership + selector)"
    )

    print()

    print(
        "Reduction:"
    )

    print(
        "  mean across exact 32 x 192 causal position table"
    )

    print()

    print(
        "Positive controls: NO"
    )

    print(
        "Sealed evaluation: NO"
    )

    print()

    print(
        "BEGIN TRAINING"
    )

    # =========================================================================
    # TRAIN
    # =========================================================================

    training_start = time.perf_counter()

    cumulative_answers = 0
    cumulative_supervised = 0

    last_metrics: dict[
        str,
        float,
    ] | None = None

    last_grad_norm = float(
        "nan"
    )

    steps_data = schedule[
        "steps_data"
    ]

    for step_index, step_data in enumerate(
        steps_data,
        start=1,
    ):
        require(
            int(
                step_data[
                    "step"
                ]
            ) == step_index,
            "Frozen schedule step mismatch during training.",
        )

        examples = step_data[
            "examples"
        ]

        batch = build_batch(
            examples,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss, metrics = (
            compute_training_loss_and_metrics(
                model,
                batch,
            )
        )

        require(
            bool(
                torch.isfinite(
                    total_loss
                ).item()
            ),
            f"Non-finite loss at step {step_index}.",
        )

        total_loss.backward()

        grad_norm_tensor = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRAD_CLIP,
            )
        )

        grad_norm = float(
            grad_norm_tensor.detach().item()
            if torch.is_tensor(
                grad_norm_tensor
            )
            else grad_norm_tensor
        )

        require(
            math.isfinite(
                grad_norm
            ),
            f"Non-finite gradient norm at step {step_index}.",
        )

        optimizer.step()

        cumulative_answers += (
            BATCH_SIZE
        )

        cumulative_supervised += (
            BATCH_SIZE
            * EXPECTED_CAUSAL_SEQUENCE_LENGTH
        )

        log_record = {
            "step": (
                step_index
            ),
            **metrics,
            "grad_norm": (
                grad_norm
            ),
            "cumulative_answers": (
                cumulative_answers
            ),
            "cumulative_supervised_positions": (
                cumulative_supervised
            ),
        }

        append_jsonl(
            METRICS_PATH,
            log_record,
        )

        last_metrics = (
            metrics
        )

        last_grad_norm = (
            grad_norm
        )

        if (
            step_index == 1
            or step_index % 100 == 0
            or step_index == MAX_STEPS
        ):
            print_step_metrics(
                step_index,
                metrics,
                grad_norm,
                cumulative_answers,
                cumulative_supervised,
            )

        # Same practical checkpoint cadence as the prior treatment series.
        if (
            step_index % 100 == 0
            or step_index == MAX_STEPS
        ):
            save_checkpoint(
                model,
                optimizer,
                step_index,
                hashes,
                trainer_sha256,
            )

    training_runtime = (
        time.perf_counter()
        - training_start
    )

    # =========================================================================
    # FINAL CHECKPOINT
    # =========================================================================

    require(
        last_metrics is not None,
        "Training completed without metrics.",
    )

    require(
        cumulative_answers
        == EXPECTED_EXAMPLES,
        "Final answer-event count mismatch.",
    )

    require(
        cumulative_supervised
        == EXPECTED_TOTAL_SUPERVISED,
        "Final supervised-position count mismatch.",
    )

    final_checkpoint_sha256 = (
        save_checkpoint(
            model,
            optimizer,
            MAX_STEPS,
            hashes,
            trainer_sha256,
        )
    )

    total_runtime = (
        time.perf_counter()
        - start_wall
    )

    # =========================================================================
    # RESULT
    # =========================================================================

    result = {
        "treatment": (
            "DaveLM v0.9 Treatment #8 — "
            "contextualized query-to-logit pathway"
        ),
        "status": (
            "TRAINING_COMPLETE_REQUIRES_FINAL_RETENTION_AUDIT"
        ),
        "scientific_hypothesis": (
            "T7's query-only additive path improved selection but "
            "used a context-free raw query embedding. T8 replaces "
            "that representation with Baby's final contextualized "
            "hidden state at the query position while preserving "
            "the additive query-to-logit mechanism and all other "
            "training conditions."
        ),
        "trainer": {
            "path": (
                str(
                    trainer_path
                )
            ),
            "sha256": (
                trainer_sha256
            ),
        },
        "frozen_artifacts": (
            hashes
        ),
        "model": {
            "base_parameters": (
                EXPECTED_BASE_PARAMETER_COUNT
            ),
            "query_projection_parameters": (
                EXPECTED_QUERY_PROJECTION_PARAMETER_COUNT
            ),
            "total_parameters": (
                EXPECTED_TOTAL_PARAMETER_COUNT
            ),
            "context_source": (
                "base_model.final_norm output"
            ),
            "query_representation": (
                "hidden[row, query_position, :]"
            ),
            "projection": (
                "Linear(320 -> 1024, bias=True)"
            ),
            "answer_adjustment": (
                "base_answer_logits + query_bias"
            ),
        },
        "training": {
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
            "answer_weight": (
                ANSWER_WEIGHT
            ),
            "margin": (
                MARGIN
            ),
            "optimizer": (
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
            "answer_events": (
                cumulative_answers
            ),
            "supervised_positions": (
                cumulative_supervised
            ),
            "nonanswer_positions": (
                EXPECTED_NONANSWER_SUPERVISED
            ),
        },
        "final_minibatch_metrics": {
            **last_metrics,
            "grad_norm": (
                last_grad_norm
            ),
        },
        "checkpoint": {
            "path": (
                str(
                    LATEST_CHECKPOINT
                )
            ),
            "sha256": (
                final_checkpoint_sha256
            ),
        },
        "training_runtime_seconds": (
            training_runtime
        ),
        "total_runtime_seconds": (
            total_runtime
        ),
        "classification": (
            "NOT YET CLASSIFIED — "
            "RUN EXACT 32,000-EVENT FINAL RETENTION AUDIT"
        ),
        "positive_controls_evaluated": (
            False
        ),
        "sealed_evaluation_opened": (
            False
        ),
    }

    write_json(
        RESULT_PATH,
        result,
    )

    # =========================================================================
    # DONE
    # =========================================================================

    banner(
        "TREATMENT #8 TRAINING COMPLETE"
    )

    print(
        f"Training runtime: "
        f"{training_runtime:.2f} seconds"
    )

    print(
        f"Total runtime: "
        f"{total_runtime:.2f} seconds"
    )

    print()

    print(
        "Final checkpoint:"
    )

    print(
        f"  {LATEST_CHECKPOINT}"
    )

    print()

    print(
        "Final checkpoint SHA256:"
    )

    print(
        f"  {final_checkpoint_sha256}"
    )

    print()

    print(
        "Training result:"
    )

    print(
        f"  {RESULT_PATH}"
    )

    print()

    print(
        "Metrics:"
    )

    print(
        f"  {METRICS_PATH}"
    )

    print()

    print(
        "FINAL CLASSIFICATION:"
    )

    print(
        "  NOT YET."
    )

    print()

    print(
        "Run exact all-32,000 final retention audit next."
    )

    print(
        "Do NOT run positive controls."
    )

    print(
        "Do NOT open sealed evaluation."
    )

    print()

    print(
        "Optimizer created: YES"
    )

    print(
        "Backward performed: YES"
    )

    print(
        "Training performed: YES"
    )

    print(
        "Positive controls evaluated: NO"
    )

    print(
        "Sealed evaluation opened: NO"
    )


if __name__ == "__main__":
    main()