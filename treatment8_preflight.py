from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# =============================================================================
# DaveLM v0.9 — Treatment #8 Preflight
#
# CONTEXTUALIZED QUERY-TO-LOGIT PATHWAY
#
# GRANDPA-REVISED T8
#
# SCIENTIFIC HYPOTHESIS
#
# T7's direct query-to-logit pathway materially improved exact training
# retention:
#
#     T5 correct>distractor: ~0.501
#     T6 correct>distractor: ~0.547
#     T7 correct>distractor: ~0.757
#
# But T7's query branch received only:
#
#     token_embedding(query_token_id)
#
# That vector is context-free.
#
# The same query token may appear in multiple records where its correct answer
# depends on the current mappings/context.
#
# T8 therefore changes exactly ONE substantive thing from T7:
#
# T7:
#
#     q = raw token embedding at query position
#
# T8:
#
#     q = FINAL CONTEXTUALIZED HIDDEN STATE at query position
#
# The output mechanism remains T7's additive pathway:
#
#     query_bias = Linear(320 -> 1024, bias=True)(q)
#
#     adjusted_answer_logits =
#         base_answer_logits + query_bias
#
# EVERYTHING ELSE REMAINS IDENTICAL TO T7:
# - same original one-map starting checkpoint
# - same base Baby architecture
# - same frozen pair pool
# - same exact schedule
# - same 1000 steps
# - same batch size / complete-pair composition
# - same AdamW / LR / weight decay / grad clip
# - same membership + selector objective
# - same answer weight lambda = 191
# - same selector margin = 0.5
# - same final retention gate
#
# IMPORTANT:
#
# Baby's native forward is NOT rewritten.
#
# We capture the output of base_model.final_norm with a forward hook.
# That tensor is Baby's final contextual representation immediately before
# the language head.
#
# The T8 branch receives ONLY the hidden vector at the frozen query position.
#
# It does NOT receive:
# - target IDs
# - distractor IDs
# - candidate IDs
# - query-slot labels
# - mapping metadata
#
# PREFLIGHT ONLY.
# NO OPTIMIZER.
# NO BACKWARD.
# NO TRAINING.
# NO POSITIVE CONTROLS.
# NO SEALED EVALUATION.
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

OUTPUT_ROOT = (
    CADAVER_ROOT
    / "treatment8_contextualized_query_logit_pathway_seed8380"
)

RESULT_PATH = (
    OUTPUT_ROOT
    / "treatment8_preflight_result.json"
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
# FROZEN EXPERIMENT CONTRACT
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

EXPECTED_HIDDEN_SIZE = 320
EXPECTED_VOCAB_SIZE = 1024

# Same Linear(320 -> 1024, bias=True) parameter count as T7.
EXPECTED_QUERY_PROJECTION_PARAMETER_COUNT = 328_704

EXPECTED_TOTAL_PARAMETER_COUNT = (
    EXPECTED_BASE_PARAMETER_COUNT
    + EXPECTED_QUERY_PROJECTION_PARAMETER_COUNT
)

EXPECTED_PAIR_COUNT = 1_536
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
EXPECTED_NONANSWER_PER_EXAMPLE = 191


# =============================================================================
# PRE-REGISTERED FINAL RETENTION GATE
# =============================================================================

RETENTION_CORRECT_GT_GATE = 0.95
RETENTION_MARGIN_VALUE = 0.5
RETENTION_MARGIN_FRACTION_GATE = 0.90


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


# =============================================================================
# IMPORT EXACT NATIVE BABY BUILDER
# =============================================================================

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


# =============================================================================
# START CHECKPOINT STATE EXTRACTION
# =============================================================================

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


# =============================================================================
# NORMALIZE BABY OUTPUT
# =============================================================================

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
            "Could not find 3D logits tensor in Baby output.",
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
        f"Expected 3D logits; got {tuple(logits.shape)}.",
    )

    return logits


# =============================================================================
# T8 WRAPPER
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
            "Base hidden-size mismatch.",
        )

        require(
            int(
                token_embedding.num_embeddings
            ) == EXPECTED_VOCAB_SIZE,
            "Base vocabulary-size mismatch.",
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
            "Base model does not expose expected final_norm module.",
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
            "Base model does not expose expected language_head Linear.",
        )

        require(
            int(
                language_head.in_features
            ) == EXPECTED_HIDDEN_SIZE,
            "language_head input size mismatch.",
        )

        require(
            int(
                language_head.out_features
            ) == EXPECTED_VOCAB_SIZE,
            "language_head output size mismatch.",
        )

        # ---------------------------------------------------------------------
        # SAME T7 OUTPUT MODULE.
        #
        # The module itself does NOT change.
        #
        # The only substantive T8 change is what feeds this projection:
        #
        #   T7 = raw token embedding
        #   T8 = contextualized final hidden state at query position
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
            f"Expected final_norm output [batch,time,hidden]; "
            f"got {tuple(module_output.shape)}.",
        )

        require(
            int(
                module_output.shape[
                    -1
                ]
            ) == EXPECTED_HIDDEN_SIZE,
            "final_norm hidden dimension mismatch.",
        )

        self._captured_final_hidden = (
            module_output
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ):
        self._captured_final_hidden = (
            None
        )

        output = self.base_model(
            input_ids
        )

        require(
            self._captured_final_hidden
            is not None,
            "final_norm hook did not capture a hidden-state tensor.",
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
            "No contextual hidden state has been captured.",
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
    ) -> torch.Tensor:
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

        return query_bias


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    banner(
        "TREATMENT #8 PREFLIGHT — CONTEXTUALIZED QUERY-TO-LOGIT PATHWAY"
    )

    # =========================================================================
    # FROZEN ARTIFACTS
    # =========================================================================

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

    # =========================================================================
    # PAIR POOL
    # =========================================================================

    banner(
        "AUDITING FROZEN COUNTERFACTUAL PAIR POOL"
    )

    pairs = pair_pool.get(
        "pairs"
    )

    require(
        isinstance(
            pairs,
            list,
        ),
        "Pair-pool 'pairs' field is not a list.",
    )

    require(
        len(
            pairs
        ) == EXPECTED_PAIR_COUNT,
        "Pair-count mismatch.",
    )

    pair_lookup: dict[
        str,
        dict[str, Any],
    ] = {}

    query_positions_seen: set[
        int
    ] = set()

    for pair in pairs:
        pair_id = str(
            pair[
                "pair_id"
            ]
        )

        require(
            pair_id not in pair_lookup,
            f"Duplicate pair_id: {pair_id}",
        )

        twin_a = pair[
            "twin_a"
        ]

        twin_b = pair[
            "twin_b"
        ]

        require(
            str(
                twin_a[
                    "twin"
                ]
            ) == "A",
            f"{pair_id}: twin_a != A.",
        )

        require(
            str(
                twin_b[
                    "twin"
                ]
            ) == "B",
            f"{pair_id}: twin_b != B.",
        )

        require(
            int(
                twin_a[
                    "query_slot"
                ]
            ) == 0,
            f"{pair_id}: Twin A slot mismatch.",
        )

        require(
            int(
                twin_b[
                    "query_slot"
                ]
            ) == 1,
            f"{pair_id}: Twin B slot mismatch.",
        )

        a_target = int(
            twin_a[
                "target_token_id"
            ]
        )

        a_distractor = int(
            twin_a[
                "distractor_token_id"
            ]
        )

        b_target = int(
            twin_b[
                "target_token_id"
            ]
        )

        b_distractor = int(
            twin_b[
                "distractor_token_id"
            ]
        )

        require(
            a_target != a_distractor,
            f"{pair_id}: A target == distractor.",
        )

        require(
            b_target != b_distractor,
            f"{pair_id}: B target == distractor.",
        )

        require(
            a_target == b_distractor,
            f"{pair_id}: A target != B distractor.",
        )

        require(
            b_target == a_distractor,
            f"{pair_id}: B target != A distractor.",
        )

        query_position = int(
            pair[
                "query_difference_position"
            ]
        )

        answer_index = int(
            pair[
                "answer_token_index"
            ]
        )

        answer_position = int(
            pair[
                "answer_causal_position"
            ]
        )

        require(
            answer_position == (
                answer_index - 1
            ),
            f"{pair_id}: causal answer-position mismatch.",
        )

        require(
            query_position < answer_index,
            f"{pair_id}: query does not precede answer.",
        )

        prefix_a = [
            int(x)
            for x in twin_a[
                "prefix_token_ids"
            ]
        ]

        prefix_b = [
            int(x)
            for x in twin_b[
                "prefix_token_ids"
            ]
        ]

        prefix_differences = [
            index
            for index, (
                a,
                b,
            ) in enumerate(
                zip(
                    prefix_a,
                    prefix_b,
                )
            )
            if a != b
        ]

        require(
            prefix_differences == [
                query_position
            ],
            f"{pair_id}: prefixes do not differ only at query position.",
        )

        full_a = [
            int(x)
            for x in twin_a[
                "full_document_token_ids"
            ]
        ]

        full_b = [
            int(x)
            for x in twin_b[
                "full_document_token_ids"
            ]
        ]

        require(
            len(
                full_a
            ) == EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH,
            f"{pair_id}: Twin A document length mismatch.",
        )

        require(
            len(
                full_b
            ) == EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH,
            f"{pair_id}: Twin B document length mismatch.",
        )

        full_differences = [
            index
            for index, (
                a,
                b,
            ) in enumerate(
                zip(
                    full_a,
                    full_b,
                )
            )
            if a != b
        ]

        require(
            full_differences == sorted(
                [
                    query_position,
                    answer_index,
                ]
            ),
            f"{pair_id}: full twin differences mismatch.",
        )

        require(
            full_a[
                answer_index
            ] == a_target,
            f"{pair_id}: Twin A answer != target.",
        )

        require(
            full_b[
                answer_index
            ] == b_target,
            f"{pair_id}: Twin B answer != target.",
        )

        require(
            int(
                pair[
                    "supervised_token_count"
                ]
            ) == EXPECTED_CAUSAL_SEQUENCE_LENGTH,
            f"{pair_id}: supervised count mismatch.",
        )

        require(
            int(
                pair[
                    "nonanswer_supervised_token_count"
                ]
            ) == EXPECTED_NONANSWER_PER_EXAMPLE,
            f"{pair_id}: nonanswer count mismatch.",
        )

        query_positions_seen.add(
            query_position
        )

        pair_lookup[
            pair_id
        ] = pair

    print(
        f"Pairs: {len(pair_lookup)}"
    )

    print(
        f"Query positions: "
        f"{sorted(query_positions_seen)}"
    )

    print(
        "Twin reciprocity: PASS"
    )

    print(
        "Query-only prefix difference: PASS"
    )

    print(
        "Query+answer full-document difference: PASS"
    )

    print(
        "Frozen pair-pool structural audit: PASS"
    )

    # =========================================================================
    # SCHEDULE
    # =========================================================================

    banner(
        "AUDITING FROZEN TRAINING SCHEDULE"
    )

    expected_schedule_values = {
        "schedule_seed": SCHEDULE_SEED,
         "steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "pairs_per_batch": PAIRS_PER_BATCH,
        "pair_presentations": EXPECTED_PAIR_PRESENTATIONS,
        "scheduled_examples": EXPECTED_EXAMPLES,
        "slot_0_presentations": EXPECTED_SLOT0,
        "slot_1_presentations": EXPECTED_SLOT1,
        "pair_exposure_min": EXPECTED_PAIR_EXPOSURE_MIN,
        "pair_exposure_max": EXPECTED_PAIR_EXPOSURE_MAX,
        "answer_substitutions": EXPECTED_ANSWER_SUBSTITUTIONS,
        "total_supervised_tokens": EXPECTED_TOTAL_SUPERVISED,
        "nonanswer_supervised_tokens": EXPECTED_NONANSWER_SUPERVISED,
    }

    for key, expected in expected_schedule_values.items():
        require(
            key in schedule,
            f"Schedule missing {key!r}.",
        )

        require(
            int(
                schedule[
                    key
                ]
            ) == int(
                expected
            ),
            f"Schedule field {key!r} mismatch.",
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
        "Step-count mismatch.",
    )

    observed_events = 0
    observed_slot0 = 0
    observed_slot1 = 0
    observed_supervised = 0
    observed_nonanswer = 0

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
            "Step ordering mismatch.",
        )

        examples = step_data.get(
            "examples"
        )

        require(
            isinstance(
                examples,
                list,
            ),
            f"Step {expected_step}: examples is not a list.",
        )

        require(
            len(
                examples
            ) == BATCH_SIZE,
            f"Step {expected_step}: batch-size mismatch.",
        )

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

            require(
                str(
                    a[
                        "pair_id"
                    ]
                ) == str(
                    b[
                        "pair_id"
                    ]
                ),
                f"Step {expected_step}: incomplete pair.",
            )

            require(
                str(
                    a[
                        "twin"
                    ]
                ) == "A"
                and str(
                    b[
                        "twin"
                    ]
                ) == "B",
                f"Step {expected_step}: A/B ordering mismatch.",
            )

            require(
                int(
                    a[
                        "query_slot"
                    ]
                ) == 0
                and int(
                    b[
                        "query_slot"
                    ]
                ) == 1,
                f"Step {expected_step}: query-slot ordering mismatch.",
            )

            require(
                int(
                    a[
                        "target_token_id"
                    ]
                ) == int(
                    b[
                        "distractor_token_id"
                    ]
                ),
                f"Step {expected_step}: A target != B distractor.",
            )

            require(
                int(
                    b[
                        "target_token_id"
                    ]
                ) == int(
                    a[
                        "distractor_token_id"
                    ]
                ),
                f"Step {expected_step}: B target != A distractor.",
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
            pair_id = str(
                example[
                    "pair_id"
                ]
            )

            require(
                pair_id in pair_lookup,
                f"Step {expected_step}: unknown pair_id.",
            )

            twin_name = str(
                example[
                    "twin"
                ]
            )

            frozen_twin = (
                pair_lookup[
                    pair_id
                ][
                    "twin_a"
                ]
                if twin_name == "A"
                else pair_lookup[
                    pair_id
                ][
                    "twin_b"
                ]
            )

            schedule_ids = [
                int(x)
                for x in example[
                    "full_document_token_ids"
                ]
            ]

            frozen_ids = [
                int(x)
                for x in frozen_twin[
                    "full_document_token_ids"
                ]
            ]

            require(
                schedule_ids == frozen_ids,
                f"Step {expected_step}: "
                f"schedule/pair document mismatch.",
            )

            query_position = int(
                example[
                    "query_difference_position"
                ]
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

            require(
                query_position == int(
                    pair_lookup[
                        pair_id
                    ][
                        "query_difference_position"
                    ]
                ),
                f"Step {expected_step}: query-position mismatch.",
            )

            require(
                answer_index == int(
                    pair_lookup[
                        pair_id
                    ][
                        "answer_token_index"
                    ]
                ),
                f"Step {expected_step}: answer-index mismatch.",
            )

            require(
                answer_position == (
                    answer_index - 1
                ),
                f"Step {expected_step}: causal-position mismatch.",
            )

            require(
                query_position < answer_index,
                f"Step {expected_step}: query does not precede answer.",
            )

            require(
                int(
                    example[
                        "supervised_token_count"
                    ]
                ) == EXPECTED_CAUSAL_SEQUENCE_LENGTH,
                f"Step {expected_step}: supervised-count mismatch.",
            )

            require(
                int(
                    example[
                        "nonanswer_supervised_token_count"
                    ]
                ) == EXPECTED_NONANSWER_PER_EXAMPLE,
                f"Step {expected_step}: nonanswer-count mismatch.",
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

            observed_supervised += int(
                example[
                    "supervised_token_count"
                ]
            )

            observed_nonanswer += int(
                example[
                    "nonanswer_supervised_token_count"
                ]
            )

    require(
        observed_events == EXPECTED_EXAMPLES,
        "Scheduled example-count mismatch.",
    )

    require(
        observed_slot0 == EXPECTED_SLOT0,
        "Slot-0 count mismatch.",
    )

    require(
        observed_slot1 == EXPECTED_SLOT1,
        "Slot-1 count mismatch.",
    )

    require(
        observed_supervised == EXPECTED_TOTAL_SUPERVISED,
        "Supervised-position count mismatch.",
    )

    require(
        observed_nonanswer == EXPECTED_NONANSWER_SUPERVISED,
        "Nonanswer-position count mismatch.",
    )

    require(
        len(
            pair_exposures
        ) == EXPECTED_PAIR_COUNT,
        "Not every pair receives frozen exposure.",
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
        f"Complete pairs/batch: {PAIRS_PER_BATCH}"
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
        f"Supervised positions: {observed_supervised}"
    )

    print(
        f"Nonanswer positions: {observed_nonanswer}"
    )

    print(
        f"Pair exposure: "
        f"{min(pair_exposures.values())}-"
        f"{max(pair_exposures.values())}"
    )

    print()

    print(
        "Frozen schedule/pair contract: PASS"
    )

    # =========================================================================
    # EXACT NATIVE MODEL + VERIFIED CHECKPOINT
    # =========================================================================

    banner(
        "VERIFYING CONTEXTUAL QUERY PATHWAY"
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
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
        "Base parameter-count mismatch.",
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
        "Query-projection parameter-count mismatch.",
    )

    require(
        total_parameter_count == EXPECTED_TOTAL_PARAMETER_COUNT,
        "Total T8 parameter-count mismatch.",
    )

    require(
        model.query_projection.bias is not None,
        "T8 query projection unexpectedly lacks its T7-compatible bias.",
    )

    require(
        tuple(
            model.query_projection.weight.shape
        ) == (
            EXPECTED_VOCAB_SIZE,
            EXPECTED_HIDDEN_SIZE,
        ),
        "query_projection.weight shape mismatch.",
    )

    require(
        tuple(
            model.query_projection.bias.shape
        ) == (
            EXPECTED_VOCAB_SIZE,
        ),
        "query_projection.bias shape mismatch.",
    )

    # =========================================================================
    # REAL FROZEN-BATCH FORWARD CONTRACT TEST
    # =========================================================================

    first_examples = schedule[
        "steps_data"
    ][
        0
    ][
        "examples"
    ]

    documents = [
        [
            int(x)
            for x in example[
                "full_document_token_ids"
            ]
        ]
        for example in first_examples
    ]

    full_ids = torch.tensor(
        documents,
        dtype=torch.long,
    )

    inputs = full_ids[
        :,
        :-1,
    ].contiguous()

    query_positions = torch.tensor(
        [
            int(
                example[
                    "query_difference_position"
                ]
            )
            for example in first_examples
        ],
        dtype=torch.long,
    )

    answer_positions = torch.tensor(
        [
            int(
                example[
                    "answer_causal_position"
                ]
            )
            for example in first_examples
        ],
        dtype=torch.long,
    )

    model.eval()

    with torch.no_grad():
        output = model(
            inputs
        )

        base_logits = extract_logits(
            output
        )

        contextual_query_hidden = (
            model.get_contextual_query_hidden(
                query_positions
            )
        )

        query_bias = (
            model.get_query_bias(
                query_positions
            )
        )

        rows = torch.arange(
            BATCH_SIZE,
            dtype=torch.long,
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

    require(
        tuple(
            contextual_query_hidden.shape
        ) == (
            BATCH_SIZE,
            EXPECTED_HIDDEN_SIZE,
        ),
        "Contextual query hidden shape mismatch.",
    )

    require(
        tuple(
            query_bias.shape
        ) == (
            BATCH_SIZE,
            EXPECTED_VOCAB_SIZE,
        ),
        "Query bias shape mismatch.",
    )

    require(
        tuple(
            adjusted_answer_logits.shape
        ) == (
            BATCH_SIZE,
            EXPECTED_VOCAB_SIZE,
        ),
        "Adjusted answer logits shape mismatch.",
    )

    print(
        f"Base parameters: "
        f"{base_parameter_count:,}"
    )

    print(
        f"Query-projection parameters: "
        f"{projection_parameter_count:,}"
    )

    print(
        f"Total T8 parameters: "
        f"{total_parameter_count:,}"
    )

    print()

    print(
        'Base builder: original_run.build_model("untied")'
    )

    print(
        "Starting checkpoint state load: PASS"
    )

    print(
        "Context source: base_model.final_norm output"
    )

    print(
        "Contextual query extraction: hidden[row, query_position, :]"
    )

    print(
        "Projection: Linear(320 -> 1024, bias=True)"
    )

    print(
        "Adjusted answer logits:"
    )

    print(
        "  base_answer_logits + query_bias"
    )

    print()

    print(
        "Native Baby forward rewritten: NO"
    )

    print(
        "Target IDs enter query branch: NO"
    )

    print(
        "Distractor IDs enter query branch: NO"
    )

    print(
        "Candidate IDs enter query branch: NO"
    )

    print(
        "Query-slot labels enter query branch: NO"
    )

    print(
        "Mapping metadata enters query branch: NO"
    )

    print()

    print(
        "Contextual query hidden capture: PASS"
    )

    print(
        "Answer-only adjusted-logit construction: PASS"
    )

    print(
        "T8 model contract: PASS"
    )

    # =========================================================================
    # PRE-REGISTERED EXPERIMENT
    # =========================================================================

    banner(
        "TREATMENT #8 PRE-REGISTERED CONTRACT"
    )

    print(
        "ONE substantive change from T7:"
    )

    print(
        "  Query projection input changes from:"
    )

    print(
        "    RAW token embedding"
    )

    print(
        "  to:"
    )

    print(
        "    FINAL CONTEXTUALIZED hidden state at query position"
    )

    print()

    print(
        "T7 additive output mechanism retained: YES"
    )

    print(
        "Linear projection dimensions retained: YES"
    )

    print(
        "Linear projection bias=True retained: YES"
    )

    print()

    print(
        f"Answer weight lambda: {ANSWER_WEIGHT}"
    )

    print(
        f"Selector margin: {MARGIN}"
    )

    print(
        f"AdamW LR: {LEARNING_RATE}"
    )

    print(
        f"Weight decay: {WEIGHT_DECAY}"
    )

    print(
        f"Grad clip: {GRAD_CLIP}"
    )

    print(
        f"Steps: {MAX_STEPS}"
    )

    print()

    print(
        "Starting checkpoint unchanged: YES"
    )

    print(
        "Pair pool unchanged: YES"
    )

    print(
        "Schedule unchanged: YES"
    )

    print(
        "Batch composition unchanged: YES"
    )

    print(
        "Membership objective unchanged: YES"
    )

    print(
        "Selector objective unchanged: YES"
    )

    print(
        "Lambda=191 unchanged: YES"
    )

    print(
        "Optimizer contract unchanged: YES"
    )

    print(
        "Retention gate unchanged: YES"
    )

    print()

    print(
        "Positive controls: NOT RUN"
    )

    print(
        "Sealed evaluation: NOT OPENED"
    )

    print(
        "Optimizer: NOT CREATED"
    )

    print(
        "Backward: NOT CALLED"
    )

    print(
        "Training: NOT PERFORMED"
    )

    result = {
        "treatment": (
            "DaveLM v0.9 Treatment #8 — "
            "contextualized query-to-logit pathway"
        ),
        "status": (
            "PREFLIGHT_PASS"
        ),
        "scientific_hypothesis": (
            "T7 materially improved reciprocal selection but "
            "its direct query pathway used a context-free raw "
            "query-token embedding. T8 tests whether replacing "
            "that representation with Baby's final contextualized "
            "hidden state at the query position is sufficient to "
            "make the paired counterfactual training task masterable."
        ),
        "single_substantive_change": {
            "t7_query_representation": (
                "base_model.token_embedding(query_token_id)"
            ),
            "t8_query_representation": (
                "base_model.final_norm output at frozen query position"
            ),
            "unchanged_output_path": (
                "Linear(320->1024,bias=True) then additive "
                "answer-logit bias"
            ),
        },
        "context_capture": {
            "module": (
                "base_model.final_norm"
            ),
            "method": (
                "forward hook"
            ),
            "query_selection": (
                "captured_hidden[row, query_position, :]"
            ),
            "expected_hidden_size": (
                EXPECTED_HIDDEN_SIZE
            ),
        },
        "frozen_artifacts": {
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
        "frozen_training": {
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
        "model": {
            "base_parameters": (
                base_parameter_count
            ),
            "query_projection_parameters": (
                projection_parameter_count
            ),
            "total_parameters": (
                total_parameter_count
            ),
        },
        "retention_gate": {
            "correct_gt_distractor_fraction_min": (
                RETENTION_CORRECT_GT_GATE
            ),
            "required_logit_margin": (
                RETENTION_MARGIN_VALUE
            ),
            "margin_satisfied_fraction_min": (
                RETENTION_MARGIN_FRACTION_GATE
            ),
        },
        "optimizer_created": (
            False
        ),
        "backward_called": (
            False
        ),
        "training_performed": (
            False
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

    banner(
        "TREATMENT #8 PREFLIGHT PASS"
    )

    print(
        "Grandpa-revised scientific question is now frozen."
    )

    print()

    print(
        "T8 contextual query representation: PASS"
    )

    print(
        "T7 additive output pathway preserved: PASS"
    )

    print(
        "Frozen experiment contract preserved: PASS"
    )

    print()

    print(
        "T8 trainer may now be written."
    )

    print()

    print(
        "Result:"
    )

    print(
        f"  {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()