import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# DAVELM v0.9 — TREATMENT #8 FINAL RETENTION AUDIT
# =============================================================================
#
# READ-ONLY AUDIT.
#
# NO optimizer.
# NO backward.
# NO gradients.
# NO training.
# NO positive controls.
# NO sealed evaluation.
#
# Purpose:
# Evaluate the final Treatment #8 checkpoint on the EXACT frozen
# 32,000 scheduled paired-counterfactual training events.
#
# Preregistered gate:
#
#   correct > distractor              >= 0.95
#   fraction(raw logit delta >= 0.5)  >= 0.90
#
# PASS:
#   STRONG_FINAL_TRAINING_RETENTION
#
# FAIL:
#   FINAL_TRAINING_RETENTION_BELOW_GATE
#
# IMPORTANT:
# --batch-size controls AUDIT INFERENCE batching only.
# It does NOT change, regenerate, shuffle, or otherwise modify
# the frozen training schedule.
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

T8_ROOT = (
    CADAVER_ROOT
    / "treatment8_contextualized_query_logit_pathway_seed8380"
)

FINAL_CHECKPOINT_PATH = (
    T8_ROOT
    / "checkpoints"
    / "contextualized_query_logit_pathway"
    / "seed_8380"
    / "latest.pt"
)

TRAINING_RESULT_PATH = (
    T8_ROOT
    / "treatment8_training_result.json"
)

AUDIT_RESULT_PATH = (
    T8_ROOT
    / "treatment8_final_retention_audit.json"
)


# =============================================================================
# FROZEN IDENTITIES
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

EXPECTED_FINAL_CHECKPOINT_SHA256 = (
    "4548f48773033abdba9b0ddd82b7df27ddbabf3bc5090bb2fc2ac60e1ad967a3"
)

EXPECTED_TRAINER_SHA256 = (
    "d6ac1d0670415667553c78e89f2004c3dc4dd612c8f43e4ed92dbb204923a470"
)


# =============================================================================
# FROZEN CONTRACT
# =============================================================================

SEED = 8380
SCHEDULE_SEED = 8382

MAX_STEPS = 1000
TRAIN_BATCH_SIZE = 32
PAIRS_PER_TRAIN_BATCH = 16

EXPECTED_PAIR_COUNT = 1536
EXPECTED_EVENT_COUNT = 32000

EXPECTED_SLOT0_COUNT = 16000
EXPECTED_SLOT1_COUNT = 16000

EXPECTED_PAIR_PRESENTATIONS = 16000

EXPECTED_PAIR_EXPOSURE_MIN = 10
EXPECTED_PAIR_EXPOSURE_MAX = 11

MODEL_VISIBLE_DOCUMENT_LENGTH = 193
CAUSAL_POSITIONS_PER_DOCUMENT = 192

EXPECTED_TOTAL_SUPERVISED_POSITIONS = 6_144_000
EXPECTED_ANSWER_SUBSTITUTIONS = 32_000
EXPECTED_NONANSWER_SUPERVISED_POSITIONS = 6_112_000

VOCAB_SIZE = 1024
HIDDEN_SIZE = 320

BASE_PARAMETER_COUNT = 10_594_944
QUERY_PROJECTION_PARAMETER_COUNT = 328_704
TOTAL_PARAMETER_COUNT = 10_923_648

ANSWER_WEIGHT = 191.0
SELECTOR_MARGIN = 0.5

CORRECT_GATE = 0.95
MARGIN_GATE = 0.90


# =============================================================================
# BASIC UTILITIES
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
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
        )


def parameter_count(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
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


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        require(
            torch.cuda.is_available(),
            "CUDA/ROCm requested but torch.cuda.is_available() is False.",
        )

        return torch.device("cuda")

    raise ValueError(
        f"Unsupported device: {device_arg}"
    )


# =============================================================================
# IMPORT NATIVE BABY BUILDER
# =============================================================================

def import_original_run():
    protected_root_string = str(PROTECTED_ROOT)

    if protected_root_string not in sys.path:
        sys.path.insert(
            0,
            protected_root_string,
        )

    import experiments.two_mapping_contextual_binding.run as original_run

    return original_run


# =============================================================================
# EXACT T8 WRAPPER
# =============================================================================

class Treatment8ContextualQueryModel(nn.Module):
    """
    Exact Treatment #8 substantive architecture.

    Native Baby is left unchanged.

    During the native forward pass we capture:

        base_model.final_norm output

    Then for each example:

        q = contextual_hidden[row, query_difference_position, :]

        query_bias =
            Linear(320 -> 1024, bias=True)(q)

        adjusted_answer_logits =
            base_answer_logits + query_bias

    The branch is used only for the answer-position metrics.
    """

    def __init__(self, base_model):
        super().__init__()

        self.base_model = base_model

        self.query_projection = nn.Linear(
            HIDDEN_SIZE,
            VOCAB_SIZE,
            bias=True,
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
            "final_norm forward hook captured nothing.",
        )

        return base_logits

    def get_adjusted_answer_logits(
        self,
        base_logits,
        query_positions,
        answer_positions,
    ):
        hidden = self._captured_final_hidden

        require(
            hidden is not None,
            "No contextual hidden state available.",
        )

        require(
            hidden.ndim == 3,
            (
                "Expected final hidden shape [B,T,D], "
                f"got {tuple(hidden.shape)}"
            ),
        )

        require(
            base_logits.ndim == 3,
            (
                "Expected base logits shape [B,T,V], "
                f"got {tuple(base_logits.shape)}"
            ),
        )

        batch_size = base_logits.shape[0]

        require(
            query_positions.shape == (batch_size,),
            "query_positions shape mismatch.",
        )

        require(
            answer_positions.shape == (batch_size,),
            "answer_positions shape mismatch.",
        )

        rows = torch.arange(
            batch_size,
            device=base_logits.device,
        )

        query_hidden = hidden[
            rows,
            query_positions,
            :
        ]

        base_answer_logits = base_logits[
            rows,
            answer_positions,
            :
        ]

        query_bias = self.query_projection(
            query_hidden
        )

        adjusted_answer_logits = (
            base_answer_logits
            + query_bias
        )

        return {
            "adjusted_answer_logits": adjusted_answer_logits,
            "base_answer_logits": base_answer_logits,
            "query_hidden": query_hidden,
            "query_bias": query_bias,
        }


# =============================================================================
# VERIFY FROZEN PAIR POOL
# =============================================================================

def verify_pair_pool(pair_pool):
    require(
        isinstance(pair_pool, dict),
        "Pair pool must be a JSON object.",
    )

    require(
        "pairs" in pair_pool,
        "Pair pool missing top-level 'pairs'.",
    )

    pairs = pair_pool["pairs"]

    require(
        isinstance(pairs, list),
        "pair_pool['pairs'] must be a list.",
    )

    require(
        len(pairs) == EXPECTED_PAIR_COUNT,
        (
            f"Expected {EXPECTED_PAIR_COUNT} frozen pairs, "
            f"found {len(pairs)}."
        ),
    )

    if "pair_count" in pair_pool:
        require(
            int(pair_pool["pair_count"])
            == EXPECTED_PAIR_COUNT,
            "pair_pool pair_count mismatch.",
        )

    pair_by_id = {}

    for index, pair in enumerate(pairs):

        require(
            isinstance(pair, dict),
            f"Pair index {index} is not a dict.",
        )

        for field in (
            "pair_id",
            "base_record_index",
            "candidate_pair",
            "query_difference_position",
            "answer_causal_position",
            "twin_a",
            "twin_b",
        ):
            require(
                field in pair,
                (
                    f"Pair index {index} missing "
                    f"required field {field!r}."
                ),
            )

        pair_id = pair["pair_id"]

        require(
            pair_id not in pair_by_id,
            f"Duplicate pair_id {pair_id!r}.",
        )

        candidate_pair = pair["candidate_pair"]

        require(
            isinstance(candidate_pair, list)
            and len(candidate_pair) == 2,
            (
                f"{pair_id}: invalid candidate_pair "
                f"{candidate_pair!r}"
            ),
        )

        twin_a = pair["twin_a"]
        twin_b = pair["twin_b"]

        require(
            twin_a["query_slot"] == 0,
            f"{pair_id}: twin_a query_slot is not 0.",
        )

        require(
            twin_b["query_slot"] == 1,
            f"{pair_id}: twin_b query_slot is not 1.",
        )

        require(
            twin_a["target_token_id"]
            == twin_b["distractor_token_id"],
            (
                f"{pair_id}: reciprocal target/distractor "
                "invariant failed A->B."
            ),
        )

        require(
            twin_b["target_token_id"]
            == twin_a["distractor_token_id"],
            (
                f"{pair_id}: reciprocal target/distractor "
                "invariant failed B->A."
            ),
        )

        require(
            sorted(
                [
                    int(twin_a["target_token_id"]),
                    int(twin_b["target_token_id"]),
                ]
            )
            == sorted(
                [int(x) for x in candidate_pair]
            ),
            (
                f"{pair_id}: candidate pair does not match "
                "twin targets."
            ),
        )

        for twin_name, twin in (
            ("twin_a", twin_a),
            ("twin_b", twin_b),
        ):
            full_document = twin[
                "full_document_token_ids"
            ]

            require(
                len(full_document)
                == MODEL_VISIBLE_DOCUMENT_LENGTH,
                (
                    f"{pair_id}/{twin_name}: expected "
                    f"{MODEL_VISIBLE_DOCUMENT_LENGTH} tokens, "
                    f"found {len(full_document)}."
                ),
            )

            target_id = int(
                twin["target_token_id"]
            )

            answer_position = int(
                pair["answer_causal_position"]
            )

            observed_answer_label = int(
                full_document[
                    answer_position + 1
                ]
            )

            require(
                observed_answer_label
                == target_id,
                (
                    f"{pair_id}/{twin_name}: "
                    "answer label does not equal target."
                ),
            )

        pair_by_id[pair_id] = pair

    return pair_by_id


# =============================================================================
# VERIFY EXACT FROZEN SCHEDULE
# =============================================================================

def verify_and_flatten_schedule(
    schedule,
    pair_by_id,
):
    require(
        isinstance(schedule, dict),
        "Schedule must be a JSON object.",
    )

    required_top_level = (
        "batch_size",
        "pairs_per_batch",
        "schedule_seed",
        "scheduled_examples",
        "slot_0_presentations",
        "slot_1_presentations",
        "steps",
        "steps_data",
        "pair_presentations",
        "pair_exposure_min",
        "pair_exposure_max",
        "total_supervised_tokens",
        "answer_substitutions",
        "nonanswer_supervised_tokens",
    )

    for key in required_top_level:
        require(
            key in schedule,
            f"Schedule missing top-level field {key!r}.",
        )

    require(
        int(schedule["steps"]) == MAX_STEPS,
        "Schedule steps mismatch.",
    )

    require(
        int(schedule["batch_size"])
        == TRAIN_BATCH_SIZE,
        "Schedule batch_size mismatch.",
    )

    require(
        int(schedule["pairs_per_batch"])
        == PAIRS_PER_TRAIN_BATCH,
        "Schedule pairs_per_batch mismatch.",
    )

    require(
        int(schedule["schedule_seed"])
        == SCHEDULE_SEED,
        "Schedule seed mismatch.",
    )

    require(
        int(schedule["scheduled_examples"])
        == EXPECTED_EVENT_COUNT,
        "Schedule scheduled_examples mismatch.",
    )

    require(
        int(schedule["slot_0_presentations"])
        == EXPECTED_SLOT0_COUNT,
        "Schedule slot_0_presentations mismatch.",
    )

    require(
        int(schedule["slot_1_presentations"])
        == EXPECTED_SLOT1_COUNT,
        "Schedule slot_1_presentations mismatch.",
    )

    require(
        int(schedule["pair_presentations"])
        == EXPECTED_PAIR_PRESENTATIONS,
        "Schedule pair_presentations mismatch.",
    )

    require(
        int(schedule["pair_exposure_min"])
        == EXPECTED_PAIR_EXPOSURE_MIN,
        "Schedule pair_exposure_min mismatch.",
    )

    require(
        int(schedule["pair_exposure_max"])
        == EXPECTED_PAIR_EXPOSURE_MAX,
        "Schedule pair_exposure_max mismatch.",
    )

    require(
        int(schedule["total_supervised_tokens"])
        == EXPECTED_TOTAL_SUPERVISED_POSITIONS,
        "Schedule total_supervised_tokens mismatch.",
    )

    require(
        int(schedule["answer_substitutions"])
        == EXPECTED_ANSWER_SUBSTITUTIONS,
        "Schedule answer_substitutions mismatch.",
    )

    require(
        int(schedule["nonanswer_supervised_tokens"])
        == EXPECTED_NONANSWER_SUPERVISED_POSITIONS,
        "Schedule nonanswer_supervised_tokens mismatch.",
    )

    steps_data = schedule["steps_data"]

    require(
        isinstance(steps_data, list),
        "schedule['steps_data'] must be a list.",
    )

    require(
        len(steps_data) == MAX_STEPS,
        (
            f"Expected {MAX_STEPS} step dictionaries, "
            f"found {len(steps_data)}."
        ),
    )

    flat_events = []

    slot_counts = Counter()
    pair_exposures = Counter()

    for step_list_index, step in enumerate(
        steps_data
    ):
        require(
            isinstance(step, dict),
            (
                f"steps_data[{step_list_index}] "
                "is not a dictionary."
            ),
        )

        expected_step_number = (
            step_list_index + 1
        )

        require(
            int(step["step"])
            == expected_step_number,
            (
                f"Expected step number "
                f"{expected_step_number}, "
                f"found {step['step']}."
            ),
        )

        require(
            int(step["example_count"])
            == TRAIN_BATCH_SIZE,
            (
                f"Step {expected_step_number}: "
                "example_count mismatch."
            ),
        )

        require(
            int(step["pair_count"])
            == PAIRS_PER_TRAIN_BATCH,
            (
                f"Step {expected_step_number}: "
                "pair_count mismatch."
            ),
        )

        examples = step["examples"]

        require(
            isinstance(examples, list),
            (
                f"Step {expected_step_number}: "
                "'examples' is not a list."
            ),
        )

        require(
            len(examples)
            == TRAIN_BATCH_SIZE,
            (
                f"Step {expected_step_number}: "
                f"expected {TRAIN_BATCH_SIZE} examples, "
                f"found {len(examples)}."
            ),
        )

        step_pair_slots = {}

        for example_index, event in enumerate(
            examples
        ):
            required_event_fields = (
                "answer_causal_position",
                "answer_token_index",
                "base_record_id",
                "base_record_index",
                "candidate_pair",
                "distractor_token_id",
                "epoch",
                "full_document_token_ids",
                "model_visible_document_length",
                "nonanswer_supervised_token_count",
                "pair_id",
                "query_difference_position",
                "query_slot",
                "supervised_token_count",
                "target_token_id",
                "twin",
            )

            for field in required_event_fields:
                require(
                    field in event,
                    (
                        f"Step {expected_step_number}, "
                        f"example {example_index}: "
                        f"missing {field!r}."
                    ),
                )

            pair_id = event["pair_id"]

            require(
                pair_id in pair_by_id,
                (
                    f"Schedule references unknown "
                    f"pair_id {pair_id!r}."
                ),
            )

            pair = pair_by_id[pair_id]

            slot = int(
                event["query_slot"]
            )

            require(
                slot in (0, 1),
                (
                    f"{pair_id}: invalid query slot "
                    f"{slot}."
                ),
            )

            expected_twin_label = (
                "A" if slot == 0 else "B"
            )

            require(
                event["twin"]
                == expected_twin_label,
                (
                    f"{pair_id}: slot/twin mismatch "
                    f"({slot}, {event['twin']!r})."
                ),
            )

            pool_twin = (
                pair["twin_a"]
                if slot == 0
                else pair["twin_b"]
            )

            # -------------------------------------------------------------
            # EXACT EVENT <-> PAIR-POOL IDENTITY
            # -------------------------------------------------------------

            require(
                int(event["base_record_index"])
                == int(pair["base_record_index"]),
                f"{pair_id}: base_record_index mismatch.",
            )

            require(
                event["base_record_id"]
                == pair["base_record_id"],
                f"{pair_id}: base_record_id mismatch.",
            )

            require(
                [int(x) for x in event["candidate_pair"]]
                == [int(x) for x in pair["candidate_pair"]],
                f"{pair_id}: candidate_pair mismatch.",
            )

            require(
                int(event["query_difference_position"])
                == int(pair["query_difference_position"]),
                f"{pair_id}: query position mismatch.",
            )

            require(
                int(event["answer_causal_position"])
                == int(pair["answer_causal_position"]),
                f"{pair_id}: answer position mismatch.",
            )

            require(
                int(event["answer_token_index"])
                == int(pair["answer_token_index"]),
                f"{pair_id}: answer token index mismatch.",
            )

            require(
                int(event["target_token_id"])
                == int(pool_twin["target_token_id"]),
                f"{pair_id}: target token mismatch.",
            )

            require(
                int(event["distractor_token_id"])
                == int(pool_twin["distractor_token_id"]),
                f"{pair_id}: distractor token mismatch.",
            )

            require(
                [
                    int(x)
                    for x in event[
                        "full_document_token_ids"
                    ]
                ]
                == [
                    int(x)
                    for x in pool_twin[
                        "full_document_token_ids"
                    ]
                ],
                (
                    f"{pair_id}: scheduled full document "
                    "does not exactly equal frozen pool twin."
                ),
            )

            # -------------------------------------------------------------
            # TOKEN / POSITION CONTRACT
            # -------------------------------------------------------------

            full_document = [
                int(x)
                for x in event[
                    "full_document_token_ids"
                ]
            ]

            require(
                int(
                    event[
                        "model_visible_document_length"
                    ]
                )
                == MODEL_VISIBLE_DOCUMENT_LENGTH,
                (
                    f"{pair_id}: model visible "
                    "document length field mismatch."
                ),
            )

            require(
                len(full_document)
                == MODEL_VISIBLE_DOCUMENT_LENGTH,
                (
                    f"{pair_id}: actual full document "
                    "length mismatch."
                ),
            )

            require(
                int(event["supervised_token_count"])
                == CAUSAL_POSITIONS_PER_DOCUMENT,
                (
                    f"{pair_id}: supervised_token_count "
                    "mismatch."
                ),
            )

            require(
                int(
                    event[
                        "nonanswer_supervised_token_count"
                    ]
                )
                == (
                    CAUSAL_POSITIONS_PER_DOCUMENT
                    - 1
                ),
                (
                    f"{pair_id}: "
                    "nonanswer_supervised_token_count "
                    "mismatch."
                ),
            )

            query_position = int(
                event["query_difference_position"]
            )

            answer_position = int(
                event["answer_causal_position"]
            )

            answer_token_index = int(
                event["answer_token_index"]
            )

            target_id = int(
                event["target_token_id"]
            )

            distractor_id = int(
                event["distractor_token_id"]
            )

            require(
                0
                <= query_position
                < CAUSAL_POSITIONS_PER_DOCUMENT,
                (
                    f"{pair_id}: invalid query position "
                    f"{query_position}."
                ),
            )

            require(
                0
                <= answer_position
                < CAUSAL_POSITIONS_PER_DOCUMENT,
                (
                    f"{pair_id}: invalid answer position "
                    f"{answer_position}."
                ),
            )

            require(
                answer_token_index
                == answer_position + 1,
                (
                    f"{pair_id}: answer_token_index "
                    "is not answer_causal_position + 1."
                ),
            )

            require(
                target_id != distractor_id,
                (
                    f"{pair_id}: target equals "
                    "distractor."
                ),
            )

            observed_answer_label = int(
                full_document[
                    answer_position + 1
                ]
            )

            require(
                observed_answer_label
                == target_id,
                (
                    f"{pair_id}: causal answer label "
                    f"{observed_answer_label} != "
                    f"target {target_id}."
                ),
            )

            # -------------------------------------------------------------
            # COMPLETE-PAIR INVARIANT WITHIN EACH TRAINING STEP
            # -------------------------------------------------------------

            if pair_id not in step_pair_slots:
                step_pair_slots[pair_id] = set()

            require(
                slot
                not in step_pair_slots[pair_id],
                (
                    f"Step {expected_step_number}: "
                    f"duplicate slot {slot} "
                    f"for {pair_id}."
                ),
            )

            step_pair_slots[pair_id].add(
                slot
            )

            slot_counts[slot] += 1

            flat_events.append(
                {
                    "step": expected_step_number,
                    "epoch": int(event["epoch"]),
                    "pair_id": pair_id,
                    "base_record_index": int(
                        event["base_record_index"]
                    ),
                    "query_slot": slot,
                    "twin": event["twin"],
                    "candidate_pair": [
                        int(x)
                        for x in event[
                            "candidate_pair"
                        ]
                    ],
                    "full_document_token_ids": (
                        full_document
                    ),
                    "query_difference_position": (
                        query_position
                    ),
                    "answer_causal_position": (
                        answer_position
                    ),
                    "target_token_id": (
                        target_id
                    ),
                    "distractor_token_id": (
                        distractor_id
                    ),
                }
            )

        require(
            len(step_pair_slots)
            == PAIRS_PER_TRAIN_BATCH,
            (
                f"Step {expected_step_number}: "
                f"expected {PAIRS_PER_TRAIN_BATCH} "
                f"unique pair IDs, found "
                f"{len(step_pair_slots)}."
            ),
        )

        for pair_id, slots in (
            step_pair_slots.items()
        ):
            require(
                slots == {0, 1},
                (
                    f"Step {expected_step_number}: "
                    f"{pair_id} does not contain "
                    f"both reciprocal slots: {slots}"
                ),
            )

            pair_exposures[pair_id] += 1

        if "pair_ids" in step:
            require(
                set(step["pair_ids"])
                == set(step_pair_slots.keys()),
                (
                    f"Step {expected_step_number}: "
                    "pair_ids list does not match "
                    "actual scheduled examples."
                ),
            )

    require(
        len(flat_events)
        == EXPECTED_EVENT_COUNT,
        (
            f"Expected {EXPECTED_EVENT_COUNT} "
            f"frozen events, found "
            f"{len(flat_events)}."
        ),
    )

    require(
        slot_counts[0]
        == EXPECTED_SLOT0_COUNT,
        (
            f"Expected {EXPECTED_SLOT0_COUNT} "
            f"slot0 events, found "
            f"{slot_counts[0]}."
        ),
    )

    require(
        slot_counts[1]
        == EXPECTED_SLOT1_COUNT,
        (
            f"Expected {EXPECTED_SLOT1_COUNT} "
            f"slot1 events, found "
            f"{slot_counts[1]}."
        ),
    )

    require(
        len(pair_exposures)
        == EXPECTED_PAIR_COUNT,
        (
            "Not all frozen pairs were scheduled: "
            f"{len(pair_exposures)} / "
            f"{EXPECTED_PAIR_COUNT}."
        ),
    )

    exposure_values = list(
        pair_exposures.values()
    )

    exposure_min = min(
        exposure_values
    )

    exposure_max = max(
        exposure_values
    )

    require(
        exposure_min
        == EXPECTED_PAIR_EXPOSURE_MIN,
        (
            f"Observed minimum pair exposure "
            f"{exposure_min}, expected "
            f"{EXPECTED_PAIR_EXPOSURE_MIN}."
        ),
    )

    require(
        exposure_max
        == EXPECTED_PAIR_EXPOSURE_MAX,
        (
            f"Observed maximum pair exposure "
            f"{exposure_max}, expected "
            f"{EXPECTED_PAIR_EXPOSURE_MAX}."
        ),
    )

    return {
        "events": flat_events,
        "slot0_count": slot_counts[0],
        "slot1_count": slot_counts[1],
        "pair_exposure_min": exposure_min,
        "pair_exposure_max": exposure_max,
    }


# =============================================================================
# METRIC ACCUMULATOR
# =============================================================================

class MetricAccumulator:
    def __init__(self):
        self.events = 0

        self.top1_correct = 0
        self.top5_correct = 0

        self.correct_over_distractor = 0
        self.margin_success = 0

        self.candidate_mass_sum = 0.0

        self.membership_loss_sum = 0.0
        self.selector_loss_sum = 0.0
        self.raw_replacement_sum = 0.0

        self.target_probability_sum = 0.0
        self.distractor_probability_sum = 0.0

        self.delta_sum = 0.0

        self.rank_sum = 0.0
        self.ranks = []

        self.query_bias_abs_sum = 0.0
        self.query_bias_element_count = 0

        self.query_bias_l2_sum = 0.0
        self.query_hidden_l2_sum = 0.0

    def update(
        self,
        logits,
        target_ids,
        distractor_ids,
        query_bias,
        query_hidden,
    ):
        batch_size = logits.shape[0]

        require(
            logits.shape
            == (batch_size, VOCAB_SIZE),
            (
                "Unexpected answer-logit shape: "
                f"{tuple(logits.shape)}"
            ),
        )

        rows = torch.arange(
            batch_size,
            device=logits.device,
        )

        target_logits = logits[
            rows,
            target_ids,
        ]

        distractor_logits = logits[
            rows,
            distractor_ids,
        ]

        deltas = (
            target_logits
            - distractor_logits
        )

        full_log_partition = torch.logsumexp(
            logits,
            dim=-1,
        )

        candidate_logits = torch.stack(
            (
                target_logits,
                distractor_logits,
            ),
            dim=-1,
        )

        candidate_log_partition = (
            torch.logsumexp(
                candidate_logits,
                dim=-1,
            )
        )

        membership_loss = (
            full_log_partition
            - candidate_log_partition
        )

        selector_loss = F.relu(
            SELECTOR_MARGIN
            - deltas
        )

        raw_replacement = (
            membership_loss
            + selector_loss
        )

        probabilities = F.softmax(
            logits,
            dim=-1,
        )

        target_probabilities = probabilities[
            rows,
            target_ids,
        ]

        distractor_probabilities = (
            probabilities[
                rows,
                distractor_ids,
            ]
        )

        candidate_mass = (
            target_probabilities
            + distractor_probabilities
        )

        top1_ids = torch.argmax(
            logits,
            dim=-1,
        )

        top5_ids = torch.topk(
            logits,
            k=5,
            dim=-1,
        ).indices

        top5_hits = (
            top5_ids
            == target_ids.unsqueeze(-1)
        ).any(
            dim=-1
        )

        # Rank = 1 + number of vocabulary logits
        # strictly greater than target.
        ranks = (
            logits
            > target_logits.unsqueeze(-1)
        ).sum(
            dim=-1
        ) + 1

        self.events += batch_size

        self.top1_correct += int(
            (
                top1_ids
                == target_ids
            ).sum().item()
        )

        self.top5_correct += int(
            top5_hits.sum().item()
        )

        self.correct_over_distractor += int(
            (
                deltas > 0.0
            ).sum().item()
        )

        self.margin_success += int(
            (
                deltas
                >= SELECTOR_MARGIN
            ).sum().item()
        )

        self.candidate_mass_sum += float(
            candidate_mass.sum().item()
        )

        self.membership_loss_sum += float(
            membership_loss.sum().item()
        )

        self.selector_loss_sum += float(
            selector_loss.sum().item()
        )

        self.raw_replacement_sum += float(
            raw_replacement.sum().item()
        )

        self.target_probability_sum += float(
            target_probabilities.sum().item()
        )

        self.distractor_probability_sum += float(
            distractor_probabilities.sum().item()
        )

        self.delta_sum += float(
            deltas.sum().item()
        )

        self.rank_sum += float(
            ranks.float().sum().item()
        )

        self.ranks.extend(
            int(x)
            for x in ranks.detach().cpu().tolist()
        )

        self.query_bias_abs_sum += float(
            query_bias.abs().sum().item()
        )

        self.query_bias_element_count += int(
            query_bias.numel()
        )

        self.query_bias_l2_sum += float(
            torch.linalg.vector_norm(
                query_bias,
                ord=2,
                dim=-1,
            ).sum().item()
        )

        self.query_hidden_l2_sum += float(
            torch.linalg.vector_norm(
                query_hidden,
                ord=2,
                dim=-1,
            ).sum().item()
        )

    def summary(self):
        require(
            self.events > 0,
            "Cannot summarize zero events.",
        )

        sorted_ranks = sorted(
            self.ranks
        )

        rank_count = len(
            sorted_ranks
        )

        if rank_count % 2 == 1:
            rank_median = float(
                sorted_ranks[
                    rank_count // 2
                ]
            )

        else:
            rank_median = (
                sorted_ranks[
                    rank_count // 2 - 1
                ]
                + sorted_ranks[
                    rank_count // 2
                ]
            ) / 2.0

        return {
            "events": self.events,

            "full_vocab_top1": (
                self.top1_correct
                / self.events
            ),

            "full_vocab_top5": (
                self.top5_correct
                / self.events
            ),

            "correct_over_distractor": (
                self.correct_over_distractor
                / self.events
            ),

            "fraction_margin_ge_0_5": (
                self.margin_success
                / self.events
            ),

            "candidate_mass": (
                self.candidate_mass_sum
                / self.events
            ),

            "membership_loss": (
                self.membership_loss_sum
                / self.events
            ),

            "selector_loss": (
                self.selector_loss_sum
                / self.events
            ),

            "raw_answer_replacement": (
                self.raw_replacement_sum
                / self.events
            ),

            "target_probability": (
                self.target_probability_sum
                / self.events
            ),

            "distractor_probability": (
                self.distractor_probability_sum
                / self.events
            ),

            "mean_logit_delta": (
                self.delta_sum
                / self.events
            ),

            "target_rank_mean": (
                self.rank_sum
                / self.events
            ),

            "target_rank_median": (
                rank_median
            ),

            "mean_abs_query_bias": (
                self.query_bias_abs_sum
                / self.query_bias_element_count
            ),

            "mean_query_bias_l2": (
                self.query_bias_l2_sum
                / self.events
            ),

            "mean_contextual_query_hidden_l2": (
                self.query_hidden_l2_sum
                / self.events
            ),
        }


# =============================================================================
# PRINTING
# =============================================================================

def print_metric_summary(
    title,
    metrics,
):
    heading(title)

    print(
        f"Events:                         "
        f"{metrics['events']}"
    )

    print(
        f"Full-vocab top1:                "
        f"{metrics['full_vocab_top1']:.6f}"
    )

    print(
        f"Full-vocab top5:                "
        f"{metrics['full_vocab_top5']:.6f}"
    )

    print(
        f"Correct > distractor:           "
        f"{metrics['correct_over_distractor']:.6f}"
    )

    print(
        f"Fraction margin >= 0.5:         "
        f"{metrics['fraction_margin_ge_0_5']:.6f}"
    )

    print(
        f"Candidate mass:                 "
        f"{metrics['candidate_mass']:.6f}"
    )

    print(
        f"Membership loss:                "
        f"{metrics['membership_loss']:.6f}"
    )

    print(
        f"Selector loss:                  "
        f"{metrics['selector_loss']:.6f}"
    )

    print(
        f"Raw answer replacement:         "
        f"{metrics['raw_answer_replacement']:.6f}"
    )

    print(
        f"Target probability:             "
        f"{metrics['target_probability']:.6f}"
    )

    print(
        f"Distractor probability:         "
        f"{metrics['distractor_probability']:.6f}"
    )

    print(
        f"Mean logit delta:               "
        f"{metrics['mean_logit_delta']:+.6f}"
    )

    print(
        f"Target rank mean:               "
        f"{metrics['target_rank_mean']:.6f}"
    )

    print(
        f"Target rank median:             "
        f"{metrics['target_rank_median']:.3f}"
    )

    print(
        f"Mean |query bias|:              "
        f"{metrics['mean_abs_query_bias']:.6f}"
    )

    print(
        f"Mean query-bias L2:             "
        f"{metrics['mean_query_bias_l2']:.6f}"
    )

    print(
        f"Mean contextual-query hidden L2:"
        f" {metrics['mean_contextual_query_hidden_l2']:.6f}"
    )


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
        help=(
            "Read-only audit device."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help=(
            "Inference-only audit batch size. "
            "Does NOT alter the frozen training schedule."
        ),
    )

    args = parser.parse_args()

    require(
        args.batch_size >= 1,
        "--batch-size must be at least 1.",
    )

    # -------------------------------------------------------------------------
    # HEADER
    # -------------------------------------------------------------------------

    heading(
        "DAVELM v0.9 — TREATMENT #8 FINAL RETENTION AUDIT"
    )

    print(
        "CONTEXTUALIZED QUERY-TO-LOGIT PATHWAY"
    )

    print()

    print(
        "READ-ONLY EXACT ALL-32,000 TRAINING RETENTION AUDIT"
    )

    print()

    print(
        "Optimizer created: NO"
    )

    print(
        "Backward performed: NO"
    )

    print(
        "Gradients enabled: NO"
    )

    print(
        "Training performed: NO"
    )

    print(
        "Positive controls evaluated: NO"
    )

    print(
        "Sealed evaluation opened: NO"
    )

    # -------------------------------------------------------------------------
    # DEVICE
    # -------------------------------------------------------------------------

    device = resolve_device(
        args.device
    )

    heading("DEVICE")

    print(
        f"Audit device: {device}"
    )

    print(
        f"Audit inference batch size: "
        f"{args.batch_size}"
    )

    print(
        "Frozen training batch size remains: "
        f"{TRAIN_BATCH_SIZE}"
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # -------------------------------------------------------------------------
    # REQUIRED ARTIFACTS
    # -------------------------------------------------------------------------

    heading(
        "VERIFYING REQUIRED ARTIFACTS"
    )

    required_paths = (
        PAIR_POOL_PATH,
        SCHEDULE_PATH,
        START_CHECKPOINT_PATH,
        FINAL_CHECKPOINT_PATH,
        TRAINING_RESULT_PATH,
    )

    for path in required_paths:
        require(
            path.exists(),
            (
                "Required artifact does not "
                f"exist: {path}"
            ),
        )

        print(
            f"PASS: {path}"
        )

    # -------------------------------------------------------------------------
    # FROZEN HASHES
    # -------------------------------------------------------------------------

    heading(
        "VERIFYING FROZEN HASHES"
    )

    pair_pool_sha256 = sha256_file(
        PAIR_POOL_PATH
    )

    schedule_sha256 = sha256_file(
        SCHEDULE_PATH
    )

    start_checkpoint_sha256 = sha256_file(
        START_CHECKPOINT_PATH
    )

    final_checkpoint_sha256 = sha256_file(
        FINAL_CHECKPOINT_PATH
    )

    print(
        "Pair pool SHA256:"
    )

    print(
        "observed:",
        pair_pool_sha256,
    )

    print(
        "expected:",
        EXPECTED_PAIR_POOL_SHA256,
    )

    require(
        pair_pool_sha256
        == EXPECTED_PAIR_POOL_SHA256,
        "Pair-pool SHA256 mismatch.",
    )

    print("PASS")
    print()

    print(
        "Frozen schedule SHA256:"
    )

    print(
        "observed:",
        schedule_sha256,
    )

    print(
        "expected:",
        EXPECTED_SCHEDULE_SHA256,
    )

    require(
        schedule_sha256
        == EXPECTED_SCHEDULE_SHA256,
        "Frozen schedule SHA256 mismatch.",
    )

    print("PASS")
    print()

    print(
        "Starting checkpoint SHA256:"
    )

    print(
        "observed:",
        start_checkpoint_sha256,
    )

    print(
        "expected:",
        EXPECTED_START_CHECKPOINT_SHA256,
    )

    require(
        start_checkpoint_sha256
        == EXPECTED_START_CHECKPOINT_SHA256,
        "Starting checkpoint SHA256 mismatch.",
    )

    print("PASS")
    print()

    print(
        "Final T8 checkpoint SHA256:"
    )

    print(
        "observed:",
        final_checkpoint_sha256,
    )

    print(
        "expected:",
        EXPECTED_FINAL_CHECKPOINT_SHA256,
    )

    require(
        final_checkpoint_sha256
        == EXPECTED_FINAL_CHECKPOINT_SHA256,
        "Final T8 checkpoint SHA256 mismatch.",
    )

    print("PASS")

    # -------------------------------------------------------------------------
    # TRAINING RESULT
    # -------------------------------------------------------------------------

    heading(
        "VERIFYING T8 TRAINING RESULT"
    )

    training_result = load_json(
        TRAINING_RESULT_PATH
    )

    require(
        isinstance(
            training_result,
            dict,
        ),
        "Training result JSON is not a dictionary.",
    )

    training_status = (
        training_result.get(
            "status"
        )
    )

    print(
        "Training-result status:",
        training_status,
    )

    require(
        training_status
        == (
            "TRAINING_COMPLETE_REQUIRES_"
            "FINAL_RETENTION_AUDIT"
        ),
        (
            "Unexpected T8 training-result "
            f"status: {training_status!r}"
        ),
    )

    print("PASS")

    # Training-result schemas have varied slightly.
    # If identities are present, verify them.
    if "trainer_sha256" in training_result:
        print()
        print(
            "Training-result trainer SHA256:"
        )

        print(
            "observed:",
            training_result[
                "trainer_sha256"
            ],
        )

        print(
            "expected:",
            EXPECTED_TRAINER_SHA256,
        )

        require(
            training_result[
                "trainer_sha256"
            ]
            == EXPECTED_TRAINER_SHA256,
            "Training-result trainer SHA mismatch.",
        )

        print("PASS")

    # -------------------------------------------------------------------------
    # LOAD / VERIFY FROZEN JSON CONTRACT
    # -------------------------------------------------------------------------

    heading(
        "VERIFYING EXACT FROZEN 32,000-EVENT CONTRACT"
    )

    pair_pool = load_json(
        PAIR_POOL_PATH
    )

    schedule = load_json(
        SCHEDULE_PATH
    )

    pair_by_id = verify_pair_pool(
        pair_pool
    )

    normalized_schedule = (
        verify_and_flatten_schedule(
            schedule,
            pair_by_id,
        )
    )

    frozen_events = (
        normalized_schedule[
            "events"
        ]
    )

    print(
        f"Unique frozen pairs:            "
        f"{len(pair_by_id)}"
    )

    print(
        f"Frozen steps:                   "
        f"{MAX_STEPS}"
    )

    print(
        f"Frozen train batch size:        "
        f"{TRAIN_BATCH_SIZE}"
    )

    print(
        f"Complete pairs / train batch:   "
        f"{PAIRS_PER_TRAIN_BATCH}"
    )

    print(
        f"Exact scheduled events:         "
        f"{len(frozen_events)}"
    )

    print(
        f"Slot 0 events:                  "
        f"{normalized_schedule['slot0_count']}"
    )

    print(
        f"Slot 1 events:                  "
        f"{normalized_schedule['slot1_count']}"
    )

    print(
        f"Pair exposure range:            "
        f"{normalized_schedule['pair_exposure_min']}"
        f"-"
        f"{normalized_schedule['pair_exposure_max']}"
    )

    print(
        f"Total supervised positions:     "
        f"{EXPECTED_TOTAL_SUPERVISED_POSITIONS}"
    )

    print(
        f"Answer substitutions:           "
        f"{EXPECTED_ANSWER_SUBSTITUTIONS}"
    )

    print(
        f"Ordinary non-answer positions:  "
        f"{EXPECTED_NONANSWER_SUPERVISED_POSITIONS}"
    )

    print()

    print(
        "Exact schedule <-> pair-pool identity: PASS"
    )

    print(
        "Complete reciprocal-pair batching: PASS"
    )

    print(
        "Target/answer causal-position identity: PASS"
    )

    # -------------------------------------------------------------------------
    # BUILD EXACT MODEL
    # -------------------------------------------------------------------------

    heading(
        "BUILDING EXACT T8 MODEL"
    )

    original_run = (
        import_original_run()
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )

    base_model = (
        original_run.build_model(
            "untied"
        )
    )

    observed_base_parameter_count = (
        parameter_count(
            base_model
        )
    )

    require(
        observed_base_parameter_count
        == BASE_PARAMETER_COUNT,
        (
            "Base parameter-count mismatch: "
            f"{observed_base_parameter_count:,} "
            f"!= {BASE_PARAMETER_COUNT:,}"
        ),
    )

    model = (
        Treatment8ContextualQueryModel(
            base_model
        )
    )

    observed_projection_parameter_count = (
        parameter_count(
            model.query_projection
        )
    )

    observed_total_parameter_count = (
        parameter_count(
            model
        )
    )

    require(
        observed_projection_parameter_count
        == QUERY_PROJECTION_PARAMETER_COUNT,
        (
            "Query projection parameter-count "
            "mismatch."
        ),
    )

    require(
        observed_total_parameter_count
        == TOTAL_PARAMETER_COUNT,
        (
            "Total T8 parameter-count "
            "mismatch."
        ),
    )

    print(
        f"Base parameters:                "
        f"{observed_base_parameter_count:,}"
    )

    print(
        f"Query-projection parameters:    "
        f"{observed_projection_parameter_count:,}"
    )

    print(
        f"Total T8 parameters:            "
        f"{observed_total_parameter_count:,}"
    )

    print()

    print(
        'Base builder: original_run.build_model("untied")'
    )

    print(
        "Context source: base_model.final_norm output"
    )

    print(
        "Query representation:"
    )

    print(
        "  contextual_hidden[row, "
        "query_difference_position, :]"
    )

    print(
        "Projection:"
    )

    print(
        "  Linear(320 -> 1024, bias=True)"
    )

    print(
        "Adjusted answer logits:"
    )

    print(
        "  base_answer_logits + query_bias"
    )

    # -------------------------------------------------------------------------
    # LOAD FINAL CHECKPOINT
    # -------------------------------------------------------------------------

    heading(
        "LOADING FINAL T8 CHECKPOINT"
    )

    checkpoint = safe_torch_load(
        FINAL_CHECKPOINT_PATH,
        map_location="cpu",
    )

    require(
        isinstance(
            checkpoint,
            dict,
        ),
        "Final T8 checkpoint is not a dictionary.",
    )

    require(
        "model_state"
        in checkpoint,
        "Final checkpoint missing model_state.",
    )

    require(
        int(
            checkpoint["step"]
        )
        == MAX_STEPS,
        (
            f"Expected final checkpoint step "
            f"{MAX_STEPS}, found "
            f"{checkpoint.get('step')!r}."
        ),
    )

    # Verify checkpoint provenance where recorded.
    if "trainer_sha256" in checkpoint:
        require(
            checkpoint["trainer_sha256"]
            == EXPECTED_TRAINER_SHA256,
            "Checkpoint trainer SHA mismatch.",
        )

    if "pair_pool_sha256" in checkpoint:
        require(
            checkpoint["pair_pool_sha256"]
            == EXPECTED_PAIR_POOL_SHA256,
            "Checkpoint pair-pool SHA mismatch.",
        )

    if "schedule_sha256" in checkpoint:
        require(
            checkpoint["schedule_sha256"]
            == EXPECTED_SCHEDULE_SHA256,
            "Checkpoint schedule SHA mismatch.",
        )

    if "start_checkpoint_sha256" in checkpoint:
        require(
            checkpoint[
                "start_checkpoint_sha256"
            ]
            == EXPECTED_START_CHECKPOINT_SHA256,
            "Checkpoint start SHA mismatch.",
        )

    if "base_parameter_count" in checkpoint:
        require(
            int(
                checkpoint[
                    "base_parameter_count"
                ]
            )
            == BASE_PARAMETER_COUNT,
            "Checkpoint base parameter-count mismatch.",
        )

    if (
        "query_projection_parameter_count"
        in checkpoint
    ):
        require(
            int(
                checkpoint[
                    "query_projection_parameter_count"
                ]
            )
            == QUERY_PROJECTION_PARAMETER_COUNT,
            (
                "Checkpoint projection "
                "parameter-count mismatch."
            ),
        )

    if "parameter_count" in checkpoint:
        require(
            int(
                checkpoint[
                    "parameter_count"
                ]
            )
            == TOTAL_PARAMETER_COUNT,
            (
                "Checkpoint total "
                "parameter-count mismatch."
            ),
        )

    if "answer_weight" in checkpoint:
        require(
            math.isclose(
                float(
                    checkpoint[
                        "answer_weight"
                    ]
                ),
                ANSWER_WEIGHT,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "Checkpoint answer_weight mismatch.",
        )

    if "margin" in checkpoint:
        require(
            math.isclose(
                float(
                    checkpoint["margin"]
                ),
                SELECTOR_MARGIN,
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "Checkpoint margin mismatch.",
        )

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    model.to(
        device
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )

    print(
        f"Checkpoint step:                "
        f"{checkpoint['step']}"
    )

    print(
        f"Checkpoint SHA256:              "
        f"{final_checkpoint_sha256}"
    )

    print(
        "Strict full T8 state load:      PASS"
    )

    print(
        "Model mode:                     eval()"
    )

    print(
        "Parameter gradients:            DISABLED"
    )

    print(
        "Optimizer created:              NO"
    )

    print(
        "Backward performed:             NO"
    )

    # -------------------------------------------------------------------------
    # EXACT ALL-32K AUDIT
    # -------------------------------------------------------------------------

    heading(
        "RUNNING EXACT ALL-32,000 FINAL RETENTION AUDIT"
    )

    overall_accumulator = (
        MetricAccumulator()
    )

    slot0_accumulator = (
        MetricAccumulator()
    )

    slot1_accumulator = (
        MetricAccumulator()
    )

    audited_event_count = 0

    audit_batch_size = (
        args.batch_size
    )

    with torch.inference_mode():

        for start_index in range(
            0,
            EXPECTED_EVENT_COUNT,
            audit_batch_size,
        ):
            end_index = min(
                start_index
                + audit_batch_size,
                EXPECTED_EVENT_COUNT,
            )

            batch_events = (
                frozen_events[
                    start_index:end_index
                ]
            )

            current_batch_size = len(
                batch_events
            )

            documents = torch.tensor(
                [
                    event[
                        "full_document_token_ids"
                    ]
                    for event
                    in batch_events
                ],
                dtype=torch.long,
                device=device,
            )

            require(
                documents.shape
                == (
                    current_batch_size,
                    MODEL_VISIBLE_DOCUMENT_LENGTH,
                ),
                (
                    "Unexpected document batch "
                    f"shape {tuple(documents.shape)}."
                ),
            )

            inputs = (
                documents[:, :-1]
            )

            labels = (
                documents[:, 1:]
            )

            query_positions = torch.tensor(
                [
                    event[
                        "query_difference_position"
                    ]
                    for event
                    in batch_events
                ],
                dtype=torch.long,
                device=device,
            )

            answer_positions = torch.tensor(
                [
                    event[
                        "answer_causal_position"
                    ]
                    for event
                    in batch_events
                ],
                dtype=torch.long,
                device=device,
            )

            target_ids = torch.tensor(
                [
                    event[
                        "target_token_id"
                    ]
                    for event
                    in batch_events
                ],
                dtype=torch.long,
                device=device,
            )

            distractor_ids = torch.tensor(
                [
                    event[
                        "distractor_token_id"
                    ]
                    for event
                    in batch_events
                ],
                dtype=torch.long,
                device=device,
            )

            slots = torch.tensor(
                [
                    event[
                        "query_slot"
                    ]
                    for event
                    in batch_events
                ],
                dtype=torch.long,
                device=device,
            )

            rows = torch.arange(
                current_batch_size,
                device=device,
            )

            observed_answer_labels = (
                labels[
                    rows,
                    answer_positions,
                ]
            )

            require(
                torch.equal(
                    observed_answer_labels,
                    target_ids,
                ),
                (
                    "Runtime causal-label "
                    "verification failed."
                ),
            )

            base_logits = model(
                inputs
            )

            require(
                base_logits.shape
                == (
                    current_batch_size,
                    CAUSAL_POSITIONS_PER_DOCUMENT,
                    VOCAB_SIZE,
                ),
                (
                    "Unexpected native Baby "
                    f"logit shape "
                    f"{tuple(base_logits.shape)}."
                ),
            )

            adjusted = (
                model.get_adjusted_answer_logits(
                    base_logits,
                    query_positions,
                    answer_positions,
                )
            )

            answer_logits = (
                adjusted[
                    "adjusted_answer_logits"
                ]
            )

            query_bias = (
                adjusted[
                    "query_bias"
                ]
            )

            query_hidden = (
                adjusted[
                    "query_hidden"
                ]
            )

            # -------------------------------------------------------------
            # OVERALL
            # -------------------------------------------------------------

            overall_accumulator.update(
                answer_logits,
                target_ids,
                distractor_ids,
                query_bias,
                query_hidden,
            )

            # -------------------------------------------------------------
            # SLOT 0
            # -------------------------------------------------------------

            slot0_mask = (
                slots == 0
            )

            if bool(
                slot0_mask.any().item()
            ):
                slot0_accumulator.update(
                    answer_logits[
                        slot0_mask
                    ],
                    target_ids[
                        slot0_mask
                    ],
                    distractor_ids[
                        slot0_mask
                    ],
                    query_bias[
                        slot0_mask
                    ],
                    query_hidden[
                        slot0_mask
                    ],
                )

            # -------------------------------------------------------------
            # SLOT 1
            # -------------------------------------------------------------

            slot1_mask = (
                slots == 1
            )

            if bool(
                slot1_mask.any().item()
            ):
                slot1_accumulator.update(
                    answer_logits[
                        slot1_mask
                    ],
                    target_ids[
                        slot1_mask
                    ],
                    distractor_ids[
                        slot1_mask
                    ],
                    query_bias[
                        slot1_mask
                    ],
                    query_hidden[
                        slot1_mask
                    ],
                )

            audited_event_count += (
                current_batch_size
            )

            if (
                audited_event_count
                % 3200
                == 0
                or audited_event_count
                == EXPECTED_EVENT_COUNT
            ):
                print(
                    f"Audited "
                    f"{audited_event_count:5d}"
                    f"/"
                    f"{EXPECTED_EVENT_COUNT} "
                    f"events"
                )

    # -------------------------------------------------------------------------
    # FINAL EVENT COUNTS
    # -------------------------------------------------------------------------

    require(
        audited_event_count
        == EXPECTED_EVENT_COUNT,
        (
            f"Audited {audited_event_count} "
            f"events instead of "
            f"{EXPECTED_EVENT_COUNT}."
        ),
    )

    require(
        overall_accumulator.events
        == EXPECTED_EVENT_COUNT,
        "Overall metric event count mismatch.",
    )

    require(
        slot0_accumulator.events
        == EXPECTED_SLOT0_COUNT,
        "Slot0 metric event count mismatch.",
    )

    require(
        slot1_accumulator.events
        == EXPECTED_SLOT1_COUNT,
        "Slot1 metric event count mismatch.",
    )

    overall_metrics = (
        overall_accumulator.summary()
    )

    slot0_metrics = (
        slot0_accumulator.summary()
    )

    slot1_metrics = (
        slot1_accumulator.summary()
    )

    # -------------------------------------------------------------------------
    # REPORT METRICS
    # -------------------------------------------------------------------------

    print_metric_summary(
        "FINAL RESULTS — ALL 32,000 EVENTS",
        overall_metrics,
    )

    print_metric_summary(
        "FINAL RESULTS — SLOT 0",
        slot0_metrics,
    )

    print_metric_summary(
        "FINAL RESULTS — SLOT 1",
        slot1_metrics,
    )

    # -------------------------------------------------------------------------
    # PREREGISTERED GATE
    # -------------------------------------------------------------------------

    heading(
        "PREREGISTERED FINAL RETENTION GATE"
    )

    observed_correct = (
        overall_metrics[
            "correct_over_distractor"
        ]
    )

    observed_margin = (
        overall_metrics[
            "fraction_margin_ge_0_5"
        ]
    )

    correct_gate_pass = (
        observed_correct
        >= CORRECT_GATE
    )

    margin_gate_pass = (
        observed_margin
        >= MARGIN_GATE
    )

    overall_gate_pass = (
        correct_gate_pass
        and margin_gate_pass
    )

    print(
        "Correct > distractor:"
    )

    print(
        f"  observed: {observed_correct:.6f}"
    )

    print(
        f"  required: {CORRECT_GATE:.6f}"
    )

    print(
        "  result:   "
        + (
            "PASS"
            if correct_gate_pass
            else "FAIL"
        )
    )

    print()

    print(
        "Fraction raw logit delta >= 0.5:"
    )

    print(
        f"  observed: {observed_margin:.6f}"
    )

    print(
        f"  required: {MARGIN_GATE:.6f}"
    )

    print(
        "  result:   "
        + (
            "PASS"
            if margin_gate_pass
            else "FAIL"
        )
    )

    if overall_gate_pass:
        classification = (
            "STRONG_FINAL_TRAINING_RETENTION"
        )

    else:
        classification = (
            "FINAL_TRAINING_RETENTION_BELOW_GATE"
        )

    # -------------------------------------------------------------------------
    # SAVE AUDIT RESULT
    # -------------------------------------------------------------------------

    audit_result = {
        "treatment": 8,

        "treatment_name": (
            "contextualized_query_logit_pathway"
        ),

        "audit_type": (
            "exact_all_32000_final_training_retention"
        ),

        "classification": (
            classification
        ),

        "frozen_artifacts": {
            "pair_pool_path": str(
                PAIR_POOL_PATH
            ),

            "pair_pool_sha256": (
                pair_pool_sha256
            ),

            "schedule_path": str(
                SCHEDULE_PATH
            ),

            "schedule_sha256": (
                schedule_sha256
            ),

            "start_checkpoint_path": str(
                START_CHECKPOINT_PATH
            ),

            "start_checkpoint_sha256": (
                start_checkpoint_sha256
            ),

            "final_checkpoint_path": str(
                FINAL_CHECKPOINT_PATH
            ),

            "final_checkpoint_sha256": (
                final_checkpoint_sha256
            ),

            "trainer_sha256": (
                EXPECTED_TRAINER_SHA256
            ),
        },

        "frozen_contract": {
            "seed": SEED,

            "schedule_seed": (
                SCHEDULE_SEED
            ),

            "steps": MAX_STEPS,

            "training_batch_size": (
                TRAIN_BATCH_SIZE
            ),

            "complete_pairs_per_training_batch": (
                PAIRS_PER_TRAIN_BATCH
            ),

            "unique_pair_count": (
                EXPECTED_PAIR_COUNT
            ),

            "scheduled_events": (
                EXPECTED_EVENT_COUNT
            ),

            "slot0_events": (
                EXPECTED_SLOT0_COUNT
            ),

            "slot1_events": (
                EXPECTED_SLOT1_COUNT
            ),

            "pair_exposure_min": (
                normalized_schedule[
                    "pair_exposure_min"
                ]
            ),

            "pair_exposure_max": (
                normalized_schedule[
                    "pair_exposure_max"
                ]
            ),

            "model_visible_document_length": (
                MODEL_VISIBLE_DOCUMENT_LENGTH
            ),

            "causal_positions_per_document": (
                CAUSAL_POSITIONS_PER_DOCUMENT
            ),

            "total_supervised_positions": (
                EXPECTED_TOTAL_SUPERVISED_POSITIONS
            ),

            "answer_substitutions": (
                EXPECTED_ANSWER_SUBSTITUTIONS
            ),

            "ordinary_nonanswer_positions": (
                EXPECTED_NONANSWER_SUPERVISED_POSITIONS
            ),

            "answer_weight_lambda": (
                ANSWER_WEIGHT
            ),

            "selector_margin": (
                SELECTOR_MARGIN
            ),
        },

        "architecture": {
            "base_builder": (
                'original_run.build_model("untied")'
            ),

            "base_parameter_count": (
                BASE_PARAMETER_COUNT
            ),

            "query_projection_parameter_count": (
                QUERY_PROJECTION_PARAMETER_COUNT
            ),

            "total_parameter_count": (
                TOTAL_PARAMETER_COUNT
            ),

            "context_source": (
                "base_model.final_norm output"
            ),

            "query_representation": (
                "contextual_hidden[row, "
                "query_difference_position, :]"
            ),

            "query_projection": (
                "Linear(320 -> 1024, bias=True)"
            ),

            "answer_logit_adjustment": (
                "base_answer_logits + query_bias"
            ),
        },

        "audit_execution": {
            "device": str(
                device
            ),

            "inference_batch_size": (
                args.batch_size
            ),

            "inference_batch_size_changes_frozen_schedule": (
                False
            ),
        },

        "overall": (
            overall_metrics
        ),

        "slot0": (
            slot0_metrics
        ),

        "slot1": (
            slot1_metrics
        ),

        "gate": {
            "correct_over_distractor_required": (
                CORRECT_GATE
            ),

            "correct_over_distractor_observed": (
                observed_correct
            ),

            "correct_over_distractor_pass": (
                correct_gate_pass
            ),

            "fraction_margin_ge_0_5_required": (
                MARGIN_GATE
            ),

            "fraction_margin_ge_0_5_observed": (
                observed_margin
            ),

            "fraction_margin_ge_0_5_pass": (
                margin_gate_pass
            ),

            "overall_pass": (
                overall_gate_pass
            ),
        },

        "safety": {
            "optimizer_created": False,
            "backward_performed": False,
            "gradients_enabled": False,
            "training_performed": False,
            "positive_controls_evaluated": False,
            "sealed_evaluation_opened": False,
        },
    }

    save_json(
        AUDIT_RESULT_PATH,
        audit_result,
    )

    # -------------------------------------------------------------------------
    # FINAL CLASSIFICATION
    # -------------------------------------------------------------------------

    heading(
        "TREATMENT #8 FINAL RETENTION CLASSIFICATION"
    )

    print(
        classification
    )

    print()

    if overall_gate_pass:

        print(
            "T8 PASSED the preregistered exact "
            "training-retention gate."
        )

        print()

        print(
            "NEXT STEP:"
        )

        print(
            "Positive controls may be evaluated "
            "under the unchanged T8 contract."
        )

        print(
            "Sealed evaluation remains CLOSED "
            "unless positive controls pass."
        )

    else:

        print(
            "T8 FAILED the preregistered exact "
            "training-retention gate."
        )

        print()

        print(
            "CLASSIFICATION:"
        )

        print(
            "TRAINING-TASK FAILURE"
        )

        print()

        print(
            "DO NOT run positive controls."
        )

        print(
            "DO NOT open sealed evaluation."
        )

        print(
            "DO NOT tune T8."
        )

        print(
            "DO NOT add steps."
        )

        print()

        print(
            "Next frozen-workflow step:"
        )

        print(
            "Literature research + Grandpa "
            "review before Treatment #9."
        )

    print()

    print(
        "Audit result:"
    )

    print(
        AUDIT_RESULT_PATH
    )

    print()

    print(
        "Optimizer created: NO"
    )

    print(
        "Backward performed: NO"
    )

    print(
        "Gradients enabled: NO"
    )

    print(
        "Training performed: NO"
    )

    print(
        "Positive controls evaluated: NO"
    )

    print(
        "Sealed evaluation opened: NO"
    )


if __name__ == "__main__":
    main()