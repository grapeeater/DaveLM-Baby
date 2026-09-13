import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# DAVELM v0.9 — TREATMENT #9 TRAINER
# ANSWER-POSITION CONTENT-ADDRESSABLE RETRIEVAL
# =============================================================================
#
# FINAL TREATMENT ATTEMPT.
#
# Frozen hypothesis:
#
#   Baby's remaining reciprocal-binding failure is caused by the absence of
#   a reliable content-addressable retrieval operation at the answer position.
#
# SINGLE ARCHITECTURAL INTERVENTION:
#
#   Add one learned single-head retrieval module after Baby's final
#   contextual hidden states.
#
#   Query:
#       final hidden at answer_causal_position
#
#   Keys / values:
#       final hiddens STRICTLY BEFORE answer_causal_position
#
#   Retrieval:
#       nn.MultiheadAttention(
#           embed_dim=320,
#           num_heads=1,
#           batch_first=True,
#           bias=True,
#       )
#
#   Residual:
#       augmented_answer_hidden =
#           base_answer_hidden + retrieved_context
#
#   Output:
#       adjusted_answer_logits =
#           base_model.language_head(augmented_answer_hidden)
#
# IMPORTANT:
#
#   T8 query->logit projection is NOT retained.
#
#   Retrieval receives NO:
#       target ID
#       distractor ID
#       candidate pair
#       query slot
#       query position
#       mapping metadata
#       mapping key/value positions
#
# Frozen training contract remains Treatment #6/#7/#8:
#
#   ordinary non-answer objective:
#       native full-vocabulary CE
#
#   answer membership:
#       logsumexp(all logits)
#       -
#       logsumexp(correct,distractor)
#
#   answer selector:
#       relu(0.5 - (correct - distractor))
#
#   replacement:
#       membership + selector
#
#   lambda:
#       191
#
#   reduction:
#       form ordinary CE over all B x 192 causal positions
#       replace each answer CE with 191 * replacement
#       mean entire table
#
# NO positive controls.
# NO sealed evaluation.
#
# Final result is NOT classified by minibatch metrics.
# Exact all-32,000 final retention audit is mandatory afterward.
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

CADAVER_ROOT = Path(r"C:\DaveLM-CADAVER")
PROTECTED_ROOT = Path(r"C:\DaveLM-v0.9")

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

START_CHECKPOINT_PATH = (
    PROTECTED_ROOT
    / "experiments"
    / "minimal_contextual_binding"
    / "checkpoints"
    / "treatment_one_mapping"
    / "seed_8380"
    / "latest.pt"
)

PREFLIGHT_RESULT_PATH = (
    CADAVER_ROOT
    / "treatment9_answer_position_retrieval_seed8380"
    / "treatment9_preflight_result.json"
)

OUTPUT_ROOT = (
    CADAVER_ROOT
    / "treatment9_answer_position_retrieval_seed8380"
)

CHECKPOINT_ROOT = (
    OUTPUT_ROOT
    / "checkpoints"
    / "answer_position_retrieval"
    / "seed_8380"
)

LATEST_CHECKPOINT_PATH = (
    CHECKPOINT_ROOT
    / "latest.pt"
)

METRICS_PATH = (
    OUTPUT_ROOT
    / "treatment9_training_metrics.jsonl"
)

TRAINING_RESULT_PATH = (
    OUTPUT_ROOT
    / "treatment9_training_result.json"
)


# =============================================================================
# FROZEN ARTIFACT HASHES
# =============================================================================

EXPECTED_PAIR_POOL_SHA256 = (
    "0b31e39361f7a45dc4182c49ab4b4967bffe539cd2ff12e7b196eabfb3dac307"
)

EXPECTED_SCHEDULE_SHA256 = (
    "a4e64fd9d1b934aa440a5d036a3bec0ac7c9ac97d58c72fb45df6d8576bb7c12"
)

EXPECTED_START_CHECKPOINT_SHA256 = (
    "345984c52a06db5f988aaf4cd47963eea0e9d77489cee94dbb10816af2f5443e"
)


# =============================================================================
# FROZEN TRAINING CONTRACT
# =============================================================================

TRAIN_SEED = 8380
SCHEDULE_SEED = 8382

MAX_STEPS = 1000
BATCH_SIZE = 32
PAIRS_PER_BATCH = 16

EXPECTED_PAIR_COUNT = 1536
EXPECTED_EVENTS = 32000
EXPECTED_SLOT0_EVENTS = 16000
EXPECTED_SLOT1_EVENTS = 16000

EXPECTED_PAIR_PRESENTATIONS = 16000
EXPECTED_PAIR_EXPOSURE_MIN = 10
EXPECTED_PAIR_EXPOSURE_MAX = 11

MODEL_VISIBLE_DOCUMENT_LENGTH = 193
CAUSAL_SEQUENCE_LENGTH = 192

EXPECTED_TOTAL_SUPERVISED_POSITIONS = 6_144_000
EXPECTED_ANSWER_REPLACEMENTS = 32_000
EXPECTED_NONANSWER_POSITIONS = 6_112_000

VOCAB_SIZE = 1024
HIDDEN_SIZE = 320

BASE_PARAMETER_COUNT = 10_594_944
RETRIEVAL_PARAMETER_COUNT = 410_880
TOTAL_PARAMETER_COUNT = 11_005_824

ANSWER_WEIGHT = 191.0
SELECTOR_MARGIN = 0.5

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 2.0

RETENTION_CORRECT_GATE = 0.95
RETENTION_MARGIN_GATE = 0.90

LOG_INTERVAL = 100
CHECKPOINT_INTERVAL = 100


# =============================================================================
# UTILITIES
# =============================================================================

def heading(text):
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
        )


def append_jsonl(path, obj):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                obj,
                sort_keys=True,
            )
        )

        f.write("\n")


def parameter_count(module):
    return sum(
        p.numel()
        for p in module.parameters()
    )


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def set_all_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        require(
            torch.cuda.is_available(),
            (
                "CUDA/ROCm requested, but "
                "torch.cuda.is_available() is False."
            ),
        )

        return torch.device("cuda")

    raise ValueError(
        f"Unsupported device: {device_arg}"
    )


def global_grad_norm(parameters):
    squared_sum = 0.0

    for parameter in parameters:
        if parameter.grad is None:
            continue

        grad_norm = float(
            parameter.grad.detach().norm(2).item()
        )

        squared_sum += grad_norm * grad_norm

    return math.sqrt(squared_sum)


# =============================================================================
# IMPORT EXACT NATIVE BABY BUILDER
# =============================================================================

def import_original_run():
    protected_root_string = str(
        PROTECTED_ROOT
    )

    if protected_root_string not in sys.path:
        sys.path.insert(
            0,
            protected_root_string,
        )

    import experiments.two_mapping_contextual_binding.run as original_run

    return original_run


# =============================================================================
# T9 MODEL
# =============================================================================

class Treatment9RetrievalModel(nn.Module):
    """
    Native Baby + ONE answer-position retrieval module.

    The native forward is not rewritten.

    A forward hook captures final contextual hidden states:

        base_model.final_norm output
        shape [B, T, 320]

    Retrieval query:

        hidden[row, answer_causal_position, :]

    Retrieval keys / values:

        hidden[row, position, :]
        for position STRICTLY BEFORE answer_causal_position

    The resulting vector is added to the base answer hidden:

        augmented = answer_hidden + retrieved_context

    Existing native language head produces answer logits:

        adjusted_logits = base_model.language_head(augmented)

    Ordinary non-answer logits remain the native base logits.
    """

    def __init__(self, base_model):
        super().__init__()

        self.base_model = base_model

        self.retrieval = nn.MultiheadAttention(
            embed_dim=HIDDEN_SIZE,
            num_heads=1,
            bias=True,
            batch_first=True,
        )

        self._captured_final_hidden = None

        self._hook_handle = (
            self.base_model.final_norm.register_forward_hook(
                self._capture_final_hidden
            )
        )

    def _capture_final_hidden(
        self,
        module,
        inputs,
        output,
    ):
        self._captured_final_hidden = output

    def forward(self, input_ids):
        self._captured_final_hidden = None

        base_logits = self.base_model(
            input_ids
        )

        require(
            self._captured_final_hidden is not None,
            "final_norm hook captured no hidden states.",
        )

        return base_logits

    def retrieve_for_answers(
        self,
        answer_positions,
        need_weights=False,
    ):
        hidden = self._captured_final_hidden

        require(
            hidden is not None,
            "No contextual hidden states captured.",
        )

        require(
            hidden.ndim == 3,
            (
                "Expected final hidden [B,T,D], got "
                f"{tuple(hidden.shape)}."
            ),
        )

        batch_size, sequence_length, hidden_size = (
            hidden.shape
        )

        require(
            hidden_size == HIDDEN_SIZE,
            (
                f"Expected hidden size {HIDDEN_SIZE}, "
                f"got {hidden_size}."
            ),
        )

        require(
            answer_positions.shape == (batch_size,),
            (
                "answer_positions shape mismatch: "
                f"{tuple(answer_positions.shape)}."
            ),
        )

        rows = torch.arange(
            batch_size,
            device=hidden.device,
        )

        answer_hidden = hidden[
            rows,
            answer_positions,
            :
        ]

        # [B,D] -> [B,1,D]
        retrieval_query = (
            answer_hidden.unsqueeze(1)
        )

        # The memory tensor contains the whole causal input,
        # but the key-padding mask forbids the retrieval module
        # from using answer/current/future positions.
        memory = hidden

        positions = torch.arange(
            sequence_length,
            device=hidden.device,
        ).unsqueeze(0)

        key_padding_mask = (
            positions
            >= answer_positions.unsqueeze(1)
        )

        visible_counts = (
            (~key_padding_mask)
            .sum(dim=1)
        )

        require(
            bool(
                (visible_counts > 0)
                .all()
                .item()
            ),
            (
                "At least one example has no legal "
                "retrieval-memory positions."
            ),
        )

        retrieved, attention_weights = self.retrieval(
            query=retrieval_query,
            key=memory,
            value=memory,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )

        retrieved_context = (
            retrieved[:, 0, :]
        )

        augmented_answer_hidden = (
            answer_hidden
            + retrieved_context
        )

        adjusted_answer_logits = (
            self.base_model.language_head(
                augmented_answer_hidden
            )
        )

        return {
            "answer_hidden": answer_hidden,
            "retrieved_context": retrieved_context,
            "augmented_answer_hidden": augmented_answer_hidden,
            "adjusted_answer_logits": adjusted_answer_logits,
            "attention_weights": attention_weights,
            "key_padding_mask": key_padding_mask,
        }


# =============================================================================
# FROZEN ARTIFACT VERIFICATION
# =============================================================================

def verify_frozen_artifacts():
    heading(
        "VERIFYING FROZEN ARTIFACTS"
    )

    for path in (
        PAIR_POOL_PATH,
        SCHEDULE_PATH,
        START_CHECKPOINT_PATH,
        PREFLIGHT_RESULT_PATH,
    ):
        require(
            path.exists(),
            f"Missing required artifact: {path}",
        )

        print(f"PASS: {path}")

    pair_sha = sha256_file(
        PAIR_POOL_PATH
    )

    schedule_sha = sha256_file(
        SCHEDULE_PATH
    )

    start_sha = sha256_file(
        START_CHECKPOINT_PATH
    )

    require(
        pair_sha == EXPECTED_PAIR_POOL_SHA256,
        "Pair-pool SHA mismatch.",
    )

    require(
        schedule_sha == EXPECTED_SCHEDULE_SHA256,
        "Frozen schedule SHA mismatch.",
    )

    require(
        start_sha == EXPECTED_START_CHECKPOINT_SHA256,
        "Start-checkpoint SHA mismatch.",
    )

    print()
    print(
        f"Pair pool SHA: {pair_sha}"
    )

    print(
        f"Schedule SHA:  {schedule_sha}"
    )

    print(
        f"Start SHA:     {start_sha}"
    )

    print()
    print("Frozen artifact identity: PASS")

    return (
        pair_sha,
        schedule_sha,
        start_sha,
    )


# =============================================================================
# VERIFY PREFLIGHT CONTRACT
# =============================================================================

def verify_preflight_contract():
    heading(
        "VERIFYING T9 PREFLIGHT CONTRACT"
    )

    preflight = load_json(
        PREFLIGHT_RESULT_PATH
    )

    require(
        isinstance(preflight, dict),
        "T9 preflight result is not a dict.",
    )

    require(
        int(preflight["treatment"]) == 9,
        "Preflight treatment number mismatch.",
    )

    require(
        preflight["treatment_name"]
        == "answer_position_content_addressable_retrieval",
        "Preflight treatment name mismatch.",
    )

    require(
        preflight["t8_query_logit_projection_retained"]
        is False,
        (
            "Preflight unexpectedly retains "
            "the T8 query-logit branch."
        ),
    )

    architecture = preflight[
        "architecture"
    ]

    require(
        int(
            architecture[
                "base_parameter_count"
            ]
        )
        == BASE_PARAMETER_COUNT,
        "Preflight base parameter count mismatch.",
    )

    require(
        int(
            architecture[
                "retrieval_parameter_count"
            ]
        )
        == RETRIEVAL_PARAMETER_COUNT,
        (
            "Preflight retrieval parameter "
            "count mismatch."
        ),
    )

    require(
        int(
            architecture[
                "total_parameter_count"
            ]
        )
        == TOTAL_PARAMETER_COUNT,
        "Preflight total parameter count mismatch.",
    )

    require(
        architecture[
            "retrieval_active_only_at_answer_positions"
        ]
        is True,
        (
            "Preflight does not require "
            "answer-only retrieval."
        ),
    )

    require(
        architecture[
            "retrieval_memory_strictly_before_answer_position"
        ]
        is True,
        (
            "Preflight does not require "
            "strictly-prior retrieval memory."
        ),
    )

    require(
        architecture[
            "normal_language_head_reused"
        ]
        is True,
        (
            "Preflight does not require native "
            "language-head reuse."
        ),
    )

    require(
        architecture[
            "native_nonanswer_logits_modified"
        ]
        is False,
        (
            "Preflight unexpectedly allows "
            "non-answer logit modification."
        ),
    )

    training = preflight[
        "training"
    ]

    expected_training = {
        "train_seed": TRAIN_SEED,
        "schedule_seed": SCHEDULE_SEED,
        "steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "pairs_per_batch": PAIRS_PER_BATCH,
        "scheduled_examples": EXPECTED_EVENTS,
        "slot0_presentations": EXPECTED_SLOT0_EVENTS,
        "slot1_presentations": EXPECTED_SLOT1_EVENTS,
    }

    for key, expected in (
        expected_training.items()
    ):
        require(
            int(training[key]) == expected,
            (
                f"Preflight training field "
                f"{key!r} mismatch."
            ),
        )

    require(
        math.isclose(
            float(training["learning_rate"]),
            LEARNING_RATE,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Preflight learning rate mismatch.",
    )

    require(
        math.isclose(
            float(training["weight_decay"]),
            WEIGHT_DECAY,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Preflight weight decay mismatch.",
    )

    require(
        math.isclose(
            float(training["grad_clip"]),
            GRAD_CLIP,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Preflight grad clip mismatch.",
    )

    require(
        math.isclose(
            float(training["answer_weight_lambda"]),
            ANSWER_WEIGHT,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Preflight answer weight mismatch.",
    )

    require(
        math.isclose(
            float(training["selector_margin"]),
            SELECTOR_MARGIN,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Preflight selector margin mismatch.",
    )

    print(
        "Preflight architecture contract: PASS"
    )

    print(
        "Preflight training contract:     PASS"
    )

    print(
        "T8 query-logit branch removed:   PASS"
    )


# =============================================================================
# SCHEDULE VERIFICATION
# =============================================================================

def verify_schedule_and_get_steps():
    heading(
        "VERIFYING EXACT FROZEN TRAINING SCHEDULE"
    )

    schedule = load_json(
        SCHEDULE_PATH
    )

    require(
        isinstance(schedule, dict),
        "Frozen schedule must be a dict.",
    )

    exact_values = {
        "steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "pairs_per_batch": PAIRS_PER_BATCH,
        "schedule_seed": SCHEDULE_SEED,
        "scheduled_examples": EXPECTED_EVENTS,
        "slot_0_presentations": EXPECTED_SLOT0_EVENTS,
        "slot_1_presentations": EXPECTED_SLOT1_EVENTS,
        "pair_presentations": EXPECTED_PAIR_PRESENTATIONS,
        "pair_exposure_min": EXPECTED_PAIR_EXPOSURE_MIN,
        "pair_exposure_max": EXPECTED_PAIR_EXPOSURE_MAX,
        "total_supervised_tokens": EXPECTED_TOTAL_SUPERVISED_POSITIONS,
        "answer_substitutions": EXPECTED_ANSWER_REPLACEMENTS,
        "nonanswer_supervised_tokens": EXPECTED_NONANSWER_POSITIONS,
    }

    for key, expected in (
        exact_values.items()
    ):
        require(
            key in schedule,
            f"Schedule missing {key!r}.",
        )

        require(
            int(schedule[key]) == expected,
            (
                f"Schedule field {key!r}: "
                f"{schedule[key]} != {expected}."
            ),
        )

    steps_data = schedule[
        "steps_data"
    ]

    require(
        isinstance(steps_data, list),
        "steps_data must be a list.",
    )

    require(
        len(steps_data) == MAX_STEPS,
        (
            f"Expected {MAX_STEPS} steps; "
            f"found {len(steps_data)}."
        ),
    )

    total_events = 0
    slot0_events = 0
    slot1_events = 0

    for index, step in enumerate(
        steps_data
    ):
        expected_step_number = (
            index + 1
        )

        require(
            int(step["step"])
            == expected_step_number,
            (
                f"Expected step "
                f"{expected_step_number}; "
                f"found {step['step']}."
            ),
        )

        examples = step[
            "examples"
        ]

        require(
            len(examples) == BATCH_SIZE,
            (
                f"Step {expected_step_number}: "
                "wrong number of examples."
            ),
        )

        require(
            int(step["pair_count"])
            == PAIRS_PER_BATCH,
            (
                f"Step {expected_step_number}: "
                "pair count mismatch."
            ),
        )

        pair_slots = {}

        for event in examples:
            total_events += 1

            slot = int(
                event["query_slot"]
            )

            require(
                slot in (0, 1),
                f"Invalid query slot {slot}.",
            )

            if slot == 0:
                slot0_events += 1
            else:
                slot1_events += 1

            pair_id = event[
                "pair_id"
            ]

            pair_slots.setdefault(
                pair_id,
                set(),
            )

            require(
                slot not in pair_slots[pair_id],
                (
                    f"Step {expected_step_number}: "
                    f"duplicate slot for {pair_id}."
                ),
            )

            pair_slots[pair_id].add(
                slot
            )

            document = event[
                "full_document_token_ids"
            ]

            require(
                len(document)
                == MODEL_VISIBLE_DOCUMENT_LENGTH,
                (
                    f"{pair_id}: document "
                    "length mismatch."
                ),
            )

            answer_position = int(
                event[
                    "answer_causal_position"
                ]
            )

            query_position = int(
                event[
                    "query_difference_position"
                ]
            )

            target_id = int(
                event[
                    "target_token_id"
                ]
            )

            distractor_id = int(
                event[
                    "distractor_token_id"
                ]
            )

            require(
                0
                <= query_position
                < answer_position,
                (
                    f"{pair_id}: query is not "
                    "strictly before answer position."
                ),
            )

            require(
                1
                <= answer_position
                < CAUSAL_SEQUENCE_LENGTH,
                (
                    f"{pair_id}: invalid "
                    "answer position."
                ),
            )

            require(
                target_id != distractor_id,
                (
                    f"{pair_id}: target equals "
                    "distractor."
                ),
            )

            require(
                int(
                    event[
                        "answer_token_index"
                    ]
                )
                == answer_position + 1,
                (
                    f"{pair_id}: answer token "
                    "index mismatch."
                ),
            )

            require(
                int(
                    document[
                        answer_position + 1
                    ]
                )
                == target_id,
                (
                    f"{pair_id}: causal label "
                    "does not equal target."
                ),
            )

        require(
            len(pair_slots)
            == PAIRS_PER_BATCH,
            (
                f"Step {expected_step_number}: "
                "unique pair count mismatch."
            ),
        )

        for pair_id, slots in (
            pair_slots.items()
        ):
            require(
                slots == {0, 1},
                (
                    f"Step {expected_step_number}: "
                    f"{pair_id} does not contain "
                    "both reciprocal twins."
                ),
            )

    require(
        total_events == EXPECTED_EVENTS,
        "Total schedule event count mismatch.",
    )

    require(
        slot0_events == EXPECTED_SLOT0_EVENTS,
        "Slot0 schedule count mismatch.",
    )

    require(
        slot1_events == EXPECTED_SLOT1_EVENTS,
        "Slot1 schedule count mismatch.",
    )

    print(
        f"Steps:                        "
        f"{len(steps_data)}"
    )

    print(
        f"Events:                       "
        f"{total_events}"
    )

    print(
        f"Slot 0:                       "
        f"{slot0_events}"
    )

    print(
        f"Slot 1:                       "
        f"{slot1_events}"
    )

    print(
        f"Complete pairs / batch:       "
        f"{PAIRS_PER_BATCH}"
    )

    print(
        "Frozen schedule integrity:    PASS"
    )

    return steps_data


# =============================================================================
# START CHECKPOINT
# =============================================================================

def load_start_state():
    checkpoint = safe_torch_load(
        START_CHECKPOINT_PATH,
        map_location="cpu",
    )

    require(
        isinstance(checkpoint, dict),
        "Start checkpoint is not a dict.",
    )

    require(
        "model_state" in checkpoint,
        (
            "Start checkpoint missing "
            "'model_state'."
        ),
    )

    require(
        isinstance(
            checkpoint["model_state"],
            dict,
        ),
        (
            "Start checkpoint model_state "
            "is not a dict."
        ),
    )

    return checkpoint[
        "model_state"
    ]


# =============================================================================
# CHECKPOINT SAVING
# =============================================================================

def save_checkpoint(
    model,
    optimizer,
    step,
    pair_sha,
    schedule_sha,
    start_sha,
    trainer_sha,
):
    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "treatment": 9,

        "treatment_name": (
            "answer_position_content_addressable_retrieval"
        ),

        "step": int(step),

        "model_state": (
            model.state_dict()
        ),

        "optimizer_state": (
            optimizer.state_dict()
        ),

        "train_seed": TRAIN_SEED,

        "schedule_seed": SCHEDULE_SEED,

        "pair_pool_sha256": pair_sha,

        "schedule_sha256": schedule_sha,

        "start_checkpoint_sha256": start_sha,

        "trainer_sha256": trainer_sha,

        "base_parameter_count": (
            BASE_PARAMETER_COUNT
        ),

        "retrieval_parameter_count": (
            RETRIEVAL_PARAMETER_COUNT
        ),

        "parameter_count": (
            TOTAL_PARAMETER_COUNT
        ),

        "answer_weight": ANSWER_WEIGHT,

        "margin": SELECTOR_MARGIN,

        "learning_rate": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "grad_clip": GRAD_CLIP,

        "architecture": {
            "retrieval_module": (
                "nn.MultiheadAttention("
                "embed_dim=320,num_heads=1,"
                "batch_first=True,bias=True)"
            ),

            "query_source": (
                "final hidden at "
                "answer_causal_position"
            ),

            "memory_source": (
                "final hiddens strictly before "
                "answer_causal_position"
            ),

            "residual": (
                "answer_hidden + retrieved_context"
            ),

            "output": (
                "base_model.language_head("
                "augmented_answer_hidden)"
            ),

            "t8_query_logit_projection_retained": (
                False
            ),
        },
    }

    step_path = (
        CHECKPOINT_ROOT
        / f"step_{step:04d}.pt"
    )

    torch.save(
        checkpoint,
        step_path,
    )

    torch.save(
        checkpoint,
        LATEST_CHECKPOINT_PATH,
    )

    return step_path


# =============================================================================
# TRAIN ONE STEP
# =============================================================================

def train_step(
    model,
    optimizer,
    examples,
    device,
):
    require(
        len(examples) == BATCH_SIZE,
        (
            f"Expected batch size "
            f"{BATCH_SIZE}, got "
            f"{len(examples)}."
        ),
    )

    documents = torch.tensor(
        [
            event[
                "full_document_token_ids"
            ]
            for event in examples
        ],
        dtype=torch.long,
        device=device,
    )

    require(
        documents.shape
        == (
            BATCH_SIZE,
            MODEL_VISIBLE_DOCUMENT_LENGTH,
        ),
        (
            "Document tensor shape mismatch: "
            f"{tuple(documents.shape)}."
        ),
    )

    inputs = documents[
        :,
        :-1,
    ]

    labels = documents[
        :,
        1:,
    ]

    answer_positions = torch.tensor(
        [
            int(
                event[
                    "answer_causal_position"
                ]
            )
            for event in examples
        ],
        dtype=torch.long,
        device=device,
    )

    target_ids = torch.tensor(
        [
            int(
                event[
                    "target_token_id"
                ]
            )
            for event in examples
        ],
        dtype=torch.long,
        device=device,
    )

    distractor_ids = torch.tensor(
        [
            int(
                event[
                    "distractor_token_id"
                ]
            )
            for event in examples
        ],
        dtype=torch.long,
        device=device,
    )

    slots = torch.tensor(
        [
            int(
                event[
                    "query_slot"
                ]
            )
            for event in examples
        ],
        dtype=torch.long,
        device=device,
    )

    rows = torch.arange(
        BATCH_SIZE,
        device=device,
    )

    observed_answer_labels = labels[
        rows,
        answer_positions,
    ]

    require(
        torch.equal(
            observed_answer_labels,
            target_ids,
        ),
        (
            "Runtime answer-label / target "
            "alignment failed."
        ),
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    # -------------------------------------------------------------------------
    # NATIVE BABY FORWARD
    # -------------------------------------------------------------------------

    base_logits = model(
        inputs
    )

    require(
        base_logits.shape
        == (
            BATCH_SIZE,
            CAUSAL_SEQUENCE_LENGTH,
            VOCAB_SIZE,
        ),
        (
            "Unexpected native logit shape: "
            f"{tuple(base_logits.shape)}."
        ),
    )

    # -------------------------------------------------------------------------
    # T9 RETRIEVAL — ANSWER ONLY
    # -------------------------------------------------------------------------

    retrieval_output = (
        model.retrieve_for_answers(
            answer_positions,
            need_weights=True,
        )
    )

    adjusted_answer_logits = (
        retrieval_output[
            "adjusted_answer_logits"
        ]
    )

    require(
        adjusted_answer_logits.shape
        == (
            BATCH_SIZE,
            VOCAB_SIZE,
        ),
        (
            "Adjusted answer-logit "
            "shape mismatch."
        ),
    )

    # -------------------------------------------------------------------------
    # ORDINARY FULL-VOCAB CE TABLE
    # -------------------------------------------------------------------------

    ordinary_ce = F.cross_entropy(
        base_logits.reshape(
            -1,
            VOCAB_SIZE,
        ),
        labels.reshape(-1),
        reduction="none",
    ).reshape(
        BATCH_SIZE,
        CAUSAL_SEQUENCE_LENGTH,
    )

    # This is the native answer CE which will be replaced.
    native_answer_ce = ordinary_ce[
        rows,
        answer_positions,
    ]

    # -------------------------------------------------------------------------
    # ANSWER MEMBERSHIP + SELECTOR
    # -------------------------------------------------------------------------

    target_logits = adjusted_answer_logits[
        rows,
        target_ids,
    ]

    distractor_logits = adjusted_answer_logits[
        rows,
        distractor_ids,
    ]

    logit_delta = (
        target_logits
        - distractor_logits
    )

    full_log_partition = torch.logsumexp(
        adjusted_answer_logits,
        dim=-1,
    )

    candidate_log_partition = torch.logsumexp(
        torch.stack(
            (
                target_logits,
                distractor_logits,
            ),
            dim=-1,
        ),
        dim=-1,
    )

    membership_loss = (
        full_log_partition
        - candidate_log_partition
    )

    selector_loss = F.relu(
        SELECTOR_MARGIN
        - logit_delta
    )

    raw_replacement = (
        membership_loss
        + selector_loss
    )

    weighted_replacement = (
        ANSWER_WEIGHT
        * raw_replacement
    )

    # -------------------------------------------------------------------------
    # EXACT FROZEN REDUCTION
    # -------------------------------------------------------------------------

    loss_table = ordinary_ce.clone()

    loss_table[
        rows,
        answer_positions,
    ] = weighted_replacement

    total_loss = loss_table.mean()

    # Useful decomposition.
    nonanswer_sum = (
        ordinary_ce.sum()
        - native_answer_ce.sum()
    )

    nonanswer_contribution = (
        nonanswer_sum
        / (
            BATCH_SIZE
            * CAUSAL_SEQUENCE_LENGTH
        )
    )

    weighted_answer_contribution = (
        weighted_replacement.sum()
        / (
            BATCH_SIZE
            * CAUSAL_SEQUENCE_LENGTH
        )
    )

    # -------------------------------------------------------------------------
    # BACKWARD
    # -------------------------------------------------------------------------

    total_loss.backward()

    preclip_grad_norm = global_grad_norm(
        model.parameters()
    )

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP,
    )

    optimizer.step()

    # -------------------------------------------------------------------------
    # READ-ONLY ONLINE METRICS FROM THIS TRAINING BATCH
    # -------------------------------------------------------------------------

    with torch.no_grad():
        probabilities = F.softmax(
            adjusted_answer_logits,
            dim=-1,
        )

        target_prob = probabilities[
            rows,
            target_ids,
        ]

        distractor_prob = probabilities[
            rows,
            distractor_ids,
        ]

        candidate_mass = (
            target_prob
            + distractor_prob
        )

        correct_over_distractor = (
            logit_delta > 0.0
        )

        margin_success = (
            logit_delta
            >= SELECTOR_MARGIN
        )

        top1_ids = torch.argmax(
            adjusted_answer_logits,
            dim=-1,
        )

        top1_correct = (
            top1_ids
            == target_ids
        )

        ranks = (
            adjusted_answer_logits
            > target_logits.unsqueeze(-1)
        ).sum(
            dim=-1
        ) + 1

        slot0_mask = (
            slots == 0
        )

        slot1_mask = (
            slots == 1
        )

        retrieved_context = (
            retrieval_output[
                "retrieved_context"
            ]
        )

        answer_hidden = (
            retrieval_output[
                "answer_hidden"
            ]
        )

        augmented_hidden = (
            retrieval_output[
                "augmented_answer_hidden"
            ]
        )

        attention_weights = (
            retrieval_output[
                "attention_weights"
            ]
        )

        require(
            attention_weights is not None,
            "Expected attention weights for logging.",
        )

        # Shape expected:
        # [B, 1, T]
        require(
            attention_weights.shape
            == (
                BATCH_SIZE,
                1,
                CAUSAL_SEQUENCE_LENGTH,
            ),
            (
                "Unexpected attention-weight "
                f"shape {tuple(attention_weights.shape)}."
            ),
        )

        attention_vector = (
            attention_weights[:, 0, :]
        )

        # Since masked positions are zero,
        # standard entropy works directly.
        attention_entropy = -(
            attention_vector
            * torch.log(
                attention_vector.clamp_min(
                    1e-12
                )
            )
        ).sum(
            dim=-1
        )

        max_attention_weight = (
            attention_vector.max(
                dim=-1
            ).values
        )

        metrics = {
            "total_loss": float(
                total_loss.item()
            ),

            "nonanswer_contribution": float(
                nonanswer_contribution.item()
            ),

            "weighted_answer_contribution": float(
                weighted_answer_contribution.item()
            ),

            "raw_replacement": float(
                raw_replacement.mean().item()
            ),

            "weighted_replacement": float(
                weighted_replacement.mean().item()
            ),

            "membership_loss": float(
                membership_loss.mean().item()
            ),

            "selector_loss": float(
                selector_loss.mean().item()
            ),

            "candidate_mass": float(
                candidate_mass.mean().item()
            ),

            "correct_over_distractor": float(
                correct_over_distractor
                .float()
                .mean()
                .item()
            ),

            "fraction_margin_ge_0_5": float(
                margin_success
                .float()
                .mean()
                .item()
            ),

            "mean_logit_delta": float(
                logit_delta.mean().item()
            ),

            "full_vocab_top1": float(
                top1_correct
                .float()
                .mean()
                .item()
            ),

            "target_probability": float(
                target_prob.mean().item()
            ),

            "distractor_probability": float(
                distractor_prob.mean().item()
            ),

            "target_rank_mean": float(
                ranks
                .float()
                .mean()
                .item()
            ),

            "target_rank_median": float(
                ranks
                .float()
                .median()
                .item()
            ),

            "slot0_correct_over_distractor": float(
                correct_over_distractor[
                    slot0_mask
                ]
                .float()
                .mean()
                .item()
            ),

            "slot1_correct_over_distractor": float(
                correct_over_distractor[
                    slot1_mask
                ]
                .float()
                .mean()
                .item()
            ),

            "slot0_fraction_margin_ge_0_5": float(
                margin_success[
                    slot0_mask
                ]
                .float()
                .mean()
                .item()
            ),

            "slot1_fraction_margin_ge_0_5": float(
                margin_success[
                    slot1_mask
                ]
                .float()
                .mean()
                .item()
            ),

            "retrieved_context_l2": float(
                torch.linalg.vector_norm(
                    retrieved_context,
                    ord=2,
                    dim=-1,
                )
                .mean()
                .item()
            ),

            "answer_hidden_l2": float(
                torch.linalg.vector_norm(
                    answer_hidden,
                    ord=2,
                    dim=-1,
                )
                .mean()
                .item()
            ),

            "augmented_answer_hidden_l2": float(
                torch.linalg.vector_norm(
                    augmented_hidden,
                    ord=2,
                    dim=-1,
                )
                .mean()
                .item()
            ),

            "attention_entropy": float(
                attention_entropy
                .mean()
                .item()
            ),

            "max_attention_weight": float(
                max_attention_weight
                .mean()
                .item()
            ),

            "preclip_grad_norm": float(
                preclip_grad_norm
            ),
        }

    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        default="cuda",
        choices=(
            "cuda",
            "cpu",
        ),
    )

    args = parser.parse_args()

    device = resolve_device(
        args.device
    )

    heading(
        "DAVELM v0.9 — TREATMENT #9"
    )

    print(
        "ANSWER-POSITION CONTENT-ADDRESSABLE RETRIEVAL"
    )

    print()

    print(
        "THIS IS THE FINAL TREATMENT ATTEMPT."
    )

    print()

    print(
        "Positive controls: CLOSED"
    )

    print(
        "Sealed evaluation: CLOSED"
    )

    print(
        "Adaptive tuning: NO"
    )

    print(
        "Extra steps: NO"
    )

    print(
        "T8 query-logit branch: REMOVED"
    )

    # -------------------------------------------------------------------------
    # DEVICE
    # -------------------------------------------------------------------------

    heading(
        "DEVICE"
    )

    print(
        f"Device: {device}"
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # -------------------------------------------------------------------------
    # VERIFY EVERYTHING BEFORE TRAINING
    # -------------------------------------------------------------------------

    pair_sha, schedule_sha, start_sha = (
        verify_frozen_artifacts()
    )

    verify_preflight_contract()

    steps_data = (
        verify_schedule_and_get_steps()
    )

    # -------------------------------------------------------------------------
    # TRAINER IDENTITY
    # -------------------------------------------------------------------------

    trainer_path = Path(__file__).resolve()

    trainer_sha = sha256_file(
        trainer_path
    )

    heading(
        "TRAINER IDENTITY"
    )

    print(
        f"Trainer path: {trainer_path}"
    )

    print(
        f"Trainer SHA256: {trainer_sha}"
    )

    # -------------------------------------------------------------------------
    # DETERMINISTIC MODEL CONSTRUCTION
    # -------------------------------------------------------------------------

    heading(
        "BUILDING T9 MODEL"
    )

    set_all_seeds(
        TRAIN_SEED
    )

    original_run = (
        import_original_run()
    )

    base_model = (
        original_run.build_model(
            "untied"
        )
    )

    observed_base_params = (
        parameter_count(
            base_model
        )
    )

    require(
        observed_base_params
        == BASE_PARAMETER_COUNT,
        (
            f"Base params "
            f"{observed_base_params:,} != "
            f"{BASE_PARAMETER_COUNT:,}."
        ),
    )

    # Exact frozen original one-mapping starting state.
    start_state = (
        load_start_state()
    )

    base_model.load_state_dict(
        start_state,
        strict=True,
    )

    # IMPORTANT:
    # Construct new retrieval module AFTER native Baby build + exact
    # starting-state load, preserving the deterministic random stream
    # frozen by the preflight contract.
    model = Treatment9RetrievalModel(
        base_model
    )

    observed_retrieval_params = (
        parameter_count(
            model.retrieval
        )
    )

    observed_total_params = (
        parameter_count(
            model
        )
    )

    require(
        observed_retrieval_params
        == RETRIEVAL_PARAMETER_COUNT,
        (
            f"Retrieval params "
            f"{observed_retrieval_params:,} != "
            f"{RETRIEVAL_PARAMETER_COUNT:,}."
        ),
    )

    require(
        observed_total_params
        == TOTAL_PARAMETER_COUNT,
        (
            f"Total params "
            f"{observed_total_params:,} != "
            f"{TOTAL_PARAMETER_COUNT:,}."
        ),
    )

    model.to(
        device
    )

    model.train()

    print(
        f"Base parameters:       "
        f"{observed_base_params:,}"
    )

    print(
        f"Retrieval parameters:  "
        f"{observed_retrieval_params:,}"
    )

    print(
        f"Total parameters:      "
        f"{observed_total_params:,}"
    )

    print()

    print(
        "Native start checkpoint strict load: PASS"
    )

    print(
        "Single retrieval module:             PASS"
    )

    print(
        "T8 direct query-logit branch:        ABSENT"
    )

    # -------------------------------------------------------------------------
    # FRESH OPTIMIZER
    # -------------------------------------------------------------------------

    heading(
        "FRESH OPTIMIZER"
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
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Weight decay: {WEIGHT_DECAY}"
    )

    print(
        f"Gradient clip: {GRAD_CLIP}"
    )

    print(
        "Parameters: ALL T9 parameters"
    )

    # -------------------------------------------------------------------------
    # PREP OUTPUT
    # -------------------------------------------------------------------------

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # A clean treatment run should not append to stale metrics.
    if METRICS_PATH.exists():
        METRICS_PATH.unlink()

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------

    heading(
        "TRAINING TREATMENT #9"
    )

    print(
        f"Steps:              {MAX_STEPS}"
    )

    print(
        f"Batch size:         {BATCH_SIZE}"
    )

    print(
        f"Complete pairs:     {PAIRS_PER_BATCH}/batch"
    )

    print(
        f"Answer events:      {EXPECTED_EVENTS}"
    )

    print(
        f"Answer lambda:      {ANSWER_WEIGHT}"
    )

    print(
        f"Selector margin:    {SELECTOR_MARGIN}"
    )

    print()

    print(
        "DO NOT SELECT/COPY LIVE POWERSHELL OUTPUT."
    )

    print(
        "Screenshots are safe."
    )

    print()

    run_start_time = (
        time.perf_counter()
    )

    cumulative_answer_events = 0
    cumulative_supervised_positions = 0

    final_metrics = None

    for step_index, step_record in enumerate(
        steps_data,
        start=1,
    ):
        require(
            int(step_record["step"])
            == step_index,
            (
                f"Schedule drift at step "
                f"{step_index}."
            ),
        )

        examples = (
            step_record["examples"]
        )

        step_start_time = (
            time.perf_counter()
        )

        metrics = train_step(
            model=model,
            optimizer=optimizer,
            examples=examples,
            device=device,
        )

        step_runtime = (
            time.perf_counter()
            - step_start_time
        )

        cumulative_answer_events += (
            BATCH_SIZE
        )

        cumulative_supervised_positions += (
            BATCH_SIZE
            * CAUSAL_SEQUENCE_LENGTH
        )

        metrics_record = {
            "step": step_index,

            "epoch": int(
                examples[0]["epoch"]
            ),

            "answer_events_this_step": (
                BATCH_SIZE
            ),

            "cumulative_answer_events": (
                cumulative_answer_events
            ),

            "supervised_positions_this_step": (
                BATCH_SIZE
                * CAUSAL_SEQUENCE_LENGTH
            ),

            "cumulative_supervised_positions": (
                cumulative_supervised_positions
            ),

            "step_runtime_seconds": (
                step_runtime
            ),

            **metrics,
        }

        append_jsonl(
            METRICS_PATH,
            metrics_record,
        )

        final_metrics = metrics_record

        # ---------------------------------------------------------------------
        # LOG
        # ---------------------------------------------------------------------

        if (
            step_index == 1
            or step_index % LOG_INTERVAL == 0
            or step_index == MAX_STEPS
        ):
            print()
            print(
                f"STEP {step_index:4d}/{MAX_STEPS}"
            )

            print(
                f"  total loss:                  "
                f"{metrics['total_loss']:.6f}"
            )

            print(
                f"  nonanswer contribution:      "
                f"{metrics['nonanswer_contribution']:.6f}"
            )

            print(
                f"  raw replacement:             "
                f"{metrics['raw_replacement']:.6f}"
            )

            print(
                f"  weighted replacement:        "
                f"{metrics['weighted_replacement']:.6f}"
            )

            print(
                f"  weighted answer contribution:"
                f" {metrics['weighted_answer_contribution']:.6f}"
            )

            print(
                f"  membership:                  "
                f"{metrics['membership_loss']:.6f}"
            )

            print(
                f"  selector:                    "
                f"{metrics['selector_loss']:.6f}"
            )

            print(
                f"  candidate mass:              "
                f"{metrics['candidate_mass']:.6f}"
            )

            print(
                f"  correct > distractor:        "
                f"{metrics['correct_over_distractor']:.6f}"
            )

            print(
                f"  margin >= 0.5:               "
                f"{metrics['fraction_margin_ge_0_5']:.6f}"
            )

            print(
                f"  mean delta:                  "
                f"{metrics['mean_logit_delta']:+.6f}"
            )

            print(
                f"  full-vocab top1:             "
                f"{metrics['full_vocab_top1']:.6f}"
            )

            print(
                f"  rank mean / median:          "
                f"{metrics['target_rank_mean']:.3f}"
                f" / "
                f"{metrics['target_rank_median']:.3f}"
            )

            print(
                f"  slot0 correct / margin:      "
                f"{metrics['slot0_correct_over_distractor']:.6f}"
                f" / "
                f"{metrics['slot0_fraction_margin_ge_0_5']:.6f}"
            )

            print(
                f"  slot1 correct / margin:      "
                f"{metrics['slot1_correct_over_distractor']:.6f}"
                f" / "
                f"{metrics['slot1_fraction_margin_ge_0_5']:.6f}"
            )

            print(
                f"  retrieved context L2:        "
                f"{metrics['retrieved_context_l2']:.6f}"
            )

            print(
                f"  answer hidden L2:            "
                f"{metrics['answer_hidden_l2']:.6f}"
            )

            print(
                f"  augmented hidden L2:         "
                f"{metrics['augmented_answer_hidden_l2']:.6f}"
            )

            print(
                f"  attention entropy:           "
                f"{metrics['attention_entropy']:.6f}"
            )

            print(
                f"  max attention weight:        "
                f"{metrics['max_attention_weight']:.6f}"
            )

            print(
                f"  preclip grad norm:           "
                f"{metrics['preclip_grad_norm']:.6f}"
            )

            print(
                f"  cumulative answers:          "
                f"{cumulative_answer_events}"
            )

            print(
                f"  cumulative supervised:       "
                f"{cumulative_supervised_positions}"
            )

        # ---------------------------------------------------------------------
        # CHECKPOINT
        # ---------------------------------------------------------------------

        if (
            step_index % CHECKPOINT_INTERVAL == 0
            or step_index == MAX_STEPS
        ):
            checkpoint_path = (
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step_index,
                    pair_sha=pair_sha,
                    schedule_sha=schedule_sha,
                    start_sha=start_sha,
                    trainer_sha=trainer_sha,
                )
            )

            print(
                f"  checkpoint: {checkpoint_path}"
            )

    # -------------------------------------------------------------------------
    # FINAL COUNTS
    # -------------------------------------------------------------------------

    require(
        cumulative_answer_events
        == EXPECTED_EVENTS,
        (
            f"Final answer-event count "
            f"{cumulative_answer_events} != "
            f"{EXPECTED_EVENTS}."
        ),
    )

    require(
        cumulative_supervised_positions
        == EXPECTED_TOTAL_SUPERVISED_POSITIONS,
        (
            "Final supervised-position "
            "count mismatch."
        ),
    )

    total_runtime = (
        time.perf_counter()
        - run_start_time
    )

    # -------------------------------------------------------------------------
    # FINAL CHECKPOINT IDENTITY
    # -------------------------------------------------------------------------

    require(
        LATEST_CHECKPOINT_PATH.exists(),
        "Final latest.pt was not written.",
    )

    final_checkpoint_sha = (
        sha256_file(
            LATEST_CHECKPOINT_PATH
        )
    )

    # -------------------------------------------------------------------------
    # TRAINING RESULT
    # -------------------------------------------------------------------------

    training_result = {
        "treatment": 9,

        "treatment_name": (
            "answer_position_content_addressable_retrieval"
        ),

        "status": (
            "TRAINING_COMPLETE_REQUIRES_FINAL_RETENTION_AUDIT"
        ),

        "formal_classification": (
            "NOT_YET_CLASSIFIED"
        ),

        "message": (
            "Run exact all-32,000 final retention audit next. "
            "Do NOT run positive controls. "
            "Do NOT open sealed evaluation."
        ),

        "trainer_path": str(
            trainer_path
        ),

        "trainer_sha256": trainer_sha,

        "pair_pool_path": str(
            PAIR_POOL_PATH
        ),

        "pair_pool_sha256": pair_sha,

        "schedule_path": str(
            SCHEDULE_PATH
        ),

        "schedule_sha256": schedule_sha,

        "start_checkpoint_path": str(
            START_CHECKPOINT_PATH
        ),

        "start_checkpoint_sha256": start_sha,

        "final_checkpoint_path": str(
            LATEST_CHECKPOINT_PATH
        ),

        "final_checkpoint_sha256": (
            final_checkpoint_sha
        ),

        "metrics_path": str(
            METRICS_PATH
        ),

        "runtime_seconds": (
            total_runtime
        ),

        "cumulative_answer_events": (
            cumulative_answer_events
        ),

        "cumulative_supervised_positions": (
            cumulative_supervised_positions
        ),

        "architecture": {
            "base_parameter_count": (
                BASE_PARAMETER_COUNT
            ),

            "retrieval_parameter_count": (
                RETRIEVAL_PARAMETER_COUNT
            ),

            "total_parameter_count": (
                TOTAL_PARAMETER_COUNT
            ),

            "retrieval_module": (
                "single-head nn.MultiheadAttention "
                "over strictly-prior final hidden states"
            ),

            "query_source": (
                "answer-position final hidden"
            ),

            "retrieved_value_destination": (
                "residual addition to answer hidden"
            ),

            "output": (
                "native language_head"
            ),

            "t8_query_logit_projection_retained": (
                False
            ),
        },

        "training_contract": {
            "seed": TRAIN_SEED,

            "schedule_seed": (
                SCHEDULE_SEED
            ),

            "steps": MAX_STEPS,

            "batch_size": (
                BATCH_SIZE
            ),

            "pairs_per_batch": (
                PAIRS_PER_BATCH
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

            "answer_weight_lambda": (
                ANSWER_WEIGHT
            ),

            "selector_margin": (
                SELECTOR_MARGIN
            ),
        },

        "final_training_batch_metrics": (
            final_metrics
        ),

        "required_next_step": (
            "EXACT_ALL_32000_FINAL_RETENTION_AUDIT"
        ),

        "positive_controls_run": False,

        "sealed_evaluation_opened": False,
    }

    save_json(
        TRAINING_RESULT_PATH,
        training_result,
    )

    # -------------------------------------------------------------------------
    # FINAL PRINT
    # -------------------------------------------------------------------------

    heading(
        "TREATMENT #9 TRAINING COMPLETE"
    )

    print(
        f"Runtime:                 "
        f"{total_runtime:.2f} sec"
    )

    print(
        f"Answer events:           "
        f"{cumulative_answer_events}"
    )

    print(
        f"Supervised positions:    "
        f"{cumulative_supervised_positions}"
    )

    print()

    print(
        f"Final checkpoint:"
    )

    print(
        LATEST_CHECKPOINT_PATH
    )

    print()

    print(
        f"Final checkpoint SHA256:"
    )

    print(
        final_checkpoint_sha
    )

    print()

    print(
        f"Training result:"
    )

    print(
        TRAINING_RESULT_PATH
    )

    print()

    print(
        f"Metrics:"
    )

    print(
        METRICS_PATH
    )

    print()

    print(
        "FORMAL RESULT: NOT YET."
    )

    print()

    print(
        "Run the exact all-32,000 final "
        "retention audit next."
    )

    print()

    print(
        "DO NOT run positive controls."
    )

    print(
        "DO NOT open sealed evaluation."
    )

    print(
        "DO NOT add steps."
    )

    print(
        "DO NOT tune T9."
    )

    print()

    print(
        "This was the final treatment attempt."
    )


if __name__ == "__main__":
    main()