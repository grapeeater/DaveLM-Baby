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
# DAVELM v0.9 — TREATMENT #9 FINAL RETENTION AUDIT
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
# Evaluate the final Treatment #9 checkpoint on the EXACT frozen
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

T9_ROOT = (
    CADAVER_ROOT
    / "treatment9_answer_position_retrieval_seed8380"
)

PREFLIGHT_RESULT_PATH = (
    T9_ROOT
    / "treatment9_preflight_result.json"
)

TRAINING_RESULT_PATH = (
    T9_ROOT
    / "treatment9_training_result.json"
)

FINAL_CHECKPOINT_PATH = (
    T9_ROOT
    / "checkpoints"
    / "answer_position_retrieval"
    / "seed_8380"
    / "latest.pt"
)

AUDIT_RESULT_PATH = (
    T9_ROOT
    / "treatment9_final_retention_audit.json"
)

TRAINER_PATH = (
    CADAVER_ROOT
    / "treatment9_train.py"
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
    "5d319d6ff5600e8594b1ff3bae931addf1ef54bc112c415158458016078a90cc"
)

EXPECTED_TRAINER_SHA256 = (
    "f4d02805911de4d0369dec055ec9a56e7587e24b55dfe8d56d03b5945e9dca04"
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
RETRIEVAL_PARAMETER_COUNT = 410_880
TOTAL_PARAMETER_COUNT = 11_005_824

ANSWER_WEIGHT = 191.0
SELECTOR_MARGIN = 0.5

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 2.0

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
# AUTHORITATIVE T9 WRAPPER
# Imported from the frozen treatment9_train.py, the read-only
# source of truth for the Treatment9RetrievalModel implementation.
# =============================================================================

def _import_treatment9_train_module():
    root_string = str(CADAVER_ROOT)

    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    import treatment9_train as authoritative_treatment9_train

    return authoritative_treatment9_train


authoritative_treatment9_train = _import_treatment9_train_module()
Treatment9RetrievalModel = authoritative_treatment9_train.Treatment9RetrievalModel


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
                < answer_position,
                (
                    f"{pair_id}: query is not "
                    "strictly before answer position."
                ),
            )

            require(
                1
                <= answer_position
                < CAUSAL_POSITIONS_PER_DOCUMENT,
                (
                    f"{pair_id}: invalid "
                    "answer position."
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

        self.retrieved_context_l2_sum = 0.0
        self.answer_hidden_l2_sum = 0.0
        self.augmented_answer_hidden_l2_sum = 0.0
        self.attention_entropy_sum = 0.0
        self.max_attention_weight_sum = 0.0

    def update(
        self,
        logits,
        target_ids,
        distractor_ids,
        retrieved_context,
        answer_hidden,
        augmented_answer_hidden,
        attention_weights,
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

        require(
            attention_weights is not None,
            "Expected attention weights for logging.",
        )

        require(
            attention_weights.shape
            == (
                batch_size,
                1,
                CAUSAL_POSITIONS_PER_DOCUMENT,
            ),
            (
                "Unexpected attention-weight "
                f"shape {tuple(attention_weights.shape)}."
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

        ranks = (
            logits
            > target_logits.unsqueeze(-1)
        ).sum(
            dim=-1
        ) + 1

        attention_vector = (
            attention_weights[:, 0, :]
        )

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

        self.retrieved_context_l2_sum += float(
            torch.linalg.vector_norm(
                retrieved_context,
                ord=2,
                dim=-1,
            ).sum().item()
        )

        self.answer_hidden_l2_sum += float(
            torch.linalg.vector_norm(
                answer_hidden,
                ord=2,
                dim=-1,
            ).sum().item()
        )

        self.augmented_answer_hidden_l2_sum += float(
            torch.linalg.vector_norm(
                augmented_answer_hidden,
                ord=2,
                dim=-1,
            ).sum().item()
        )

        self.attention_entropy_sum += float(
            attention_entropy.sum().item()
        )

        self.max_attention_weight_sum += float(
            max_attention_weight.sum().item()
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

            "mean_retrieved_context_l2": (
                self.retrieved_context_l2_sum
                / self.events
            ),

            "mean_answer_hidden_l2": (
                self.answer_hidden_l2_sum
                / self.events
            ),

            "mean_augmented_answer_hidden_l2": (
                self.augmented_answer_hidden_l2_sum
                / self.events
            ),

            "mean_attention_entropy": (
                self.attention_entropy_sum
                / self.events
            ),

            "mean_max_attention_weight": (
                self.max_attention_weight_sum
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
        f"Mean retrieved-context L2:      "
        f"{metrics['mean_retrieved_context_l2']:.6f}"
    )

    print(
        f"Mean answer-hidden L2:          "
        f"{metrics['mean_answer_hidden_l2']:.6f}"
    )

    print(
        f"Mean augmented-hidden L2:       "
        f"{metrics['mean_augmented_answer_hidden_l2']:.6f}"
    )

    print(
        f"Mean attention entropy:         "
        f"{metrics['mean_attention_entropy']:.6f}"
    )

    print(
        f"Mean max attention weight:      "
        f"{metrics['mean_max_attention_weight']:.6f}"
    )


# =============================================================================
# PREFLIGHT / TRAINING-RESULT IDENTITY
# =============================================================================

def verify_preflight_contract(preflight):
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
        "train_seed": SEED,
        "schedule_seed": SCHEDULE_SEED,
        "steps": MAX_STEPS,
        "batch_size": TRAIN_BATCH_SIZE,
        "pairs_per_batch": PAIRS_PER_TRAIN_BATCH,
        "scheduled_examples": EXPECTED_EVENT_COUNT,
        "slot0_presentations": EXPECTED_SLOT0_COUNT,
        "slot1_presentations": EXPECTED_SLOT1_COUNT,
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

    gate = preflight[
        "final_retention_gate"
    ]

    require(
        math.isclose(
            float(
                gate[
                    "correct_over_distractor_min"
                ]
            ),
            CORRECT_GATE,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Preflight correct gate mismatch.",
    )

    require(
        math.isclose(
            float(
                gate[
                    "fraction_raw_delta_ge_0_5_min"
                ]
            ),
            MARGIN_GATE,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Preflight margin gate mismatch.",
    )

    require(
        gate["pass_classification"]
        == "STRONG_FINAL_TRAINING_RETENTION",
        "Preflight pass classification mismatch.",
    )

    require(
        gate["fail_classification"]
        == "FINAL_TRAINING_RETENTION_BELOW_GATE",
        "Preflight fail classification mismatch.",
    )

    require(
        int(gate["exact_scheduled_events"])
        == EXPECTED_EVENT_COUNT,
        "Preflight exact scheduled events mismatch.",
    )


def verify_training_result(training_result):
    require(
        isinstance(
            training_result,
            dict,
        ),
        "Training result JSON is not a dictionary.",
    )

    require(
        int(training_result["treatment"]) == 9,
        "Training-result treatment number mismatch.",
    )

    require(
        training_result["treatment_name"]
        == "answer_position_content_addressable_retrieval",
        "Training-result treatment name mismatch.",
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
            "Unexpected T9 training-result "
            f"status: {training_status!r}"
        ),
    )

    require(
        training_result[
            "required_next_step"
        ]
        == "EXACT_ALL_32000_FINAL_RETENTION_AUDIT",
        "Training-result required_next_step mismatch.",
    )

    require(
        training_result[
            "trainer_sha256"
        ]
        == EXPECTED_TRAINER_SHA256,
        "Training-result trainer SHA mismatch.",
    )

    require(
        training_result[
            "pair_pool_sha256"
        ]
        == EXPECTED_PAIR_POOL_SHA256,
        "Training-result pair-pool SHA mismatch.",
    )

    require(
        training_result[
            "schedule_sha256"
        ]
        == EXPECTED_SCHEDULE_SHA256,
        "Training-result schedule SHA mismatch.",
    )

    require(
        training_result[
            "start_checkpoint_sha256"
        ]
        == EXPECTED_START_CHECKPOINT_SHA256,
        "Training-result start SHA mismatch.",
    )

    require(
        training_result[
            "final_checkpoint_sha256"
        ]
        == EXPECTED_FINAL_CHECKPOINT_SHA256,
        "Training-result final checkpoint SHA mismatch.",
    )

    architecture = training_result[
        "architecture"
    ]

    require(
        architecture[
            "t8_query_logit_projection_retained"
        ]
        is False,
        (
            "Training result unexpectedly retains "
            "the T8 query-logit branch."
        ),
    )

    require(
        int(
            architecture[
                "base_parameter_count"
            ]
        )
        == BASE_PARAMETER_COUNT,
        "Training-result base parameter-count mismatch.",
    )

    require(
        int(
            architecture[
                "retrieval_parameter_count"
            ]
        )
        == RETRIEVAL_PARAMETER_COUNT,
        (
            "Training-result retrieval "
            "parameter-count mismatch."
        ),
    )

    require(
        int(
            architecture[
                "total_parameter_count"
            ]
        )
        == TOTAL_PARAMETER_COUNT,
        "Training-result total parameter-count mismatch.",
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

    heading(
        "DAVELM v0.9 — TREATMENT #9 FINAL RETENTION AUDIT"
    )

    print(
        "ANSWER-POSITION CONTENT-ADDRESSABLE RETRIEVAL"
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

    heading(
        "VERIFYING REQUIRED ARTIFACTS"
    )

    required_paths = (
        PAIR_POOL_PATH,
        SCHEDULE_PATH,
        START_CHECKPOINT_PATH,
        FINAL_CHECKPOINT_PATH,
        PREFLIGHT_RESULT_PATH,
        TRAINING_RESULT_PATH,
        TRAINER_PATH,
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

    trainer_sha256 = sha256_file(
        TRAINER_PATH
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
        "Final T9 checkpoint SHA256:"
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
        "Final T9 checkpoint SHA256 mismatch.",
    )

    print("PASS")
    print()

    print(
        "Trainer SHA256:"
    )

    print(
        "observed:",
        trainer_sha256,
    )

    print(
        "expected:",
        EXPECTED_TRAINER_SHA256,
    )

    require(
        trainer_sha256
        == EXPECTED_TRAINER_SHA256,
        "Trainer SHA256 mismatch.",
    )

    print("PASS")

    heading(
        "VERIFYING T9 PREFLIGHT CONTRACT"
    )

    preflight = load_json(
        PREFLIGHT_RESULT_PATH
    )

    verify_preflight_contract(
        preflight
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

    print(
        "Frozen retention gate identity:  PASS"
    )

    heading(
        "VERIFYING T9 TRAINING RESULT"
    )

    training_result = load_json(
        TRAINING_RESULT_PATH
    )

    verify_training_result(
        training_result
    )

    print("PASS")

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

    heading(
        "BUILDING EXACT T9 MODEL"
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
        Treatment9RetrievalModel(
            base_model
        )
    )

    observed_retrieval_parameter_count = (
        parameter_count(
            model.retrieval
        )
    )

    observed_total_parameter_count = (
        parameter_count(
            model
        )
    )

    require(
        observed_retrieval_parameter_count
        == RETRIEVAL_PARAMETER_COUNT,
        "Retrieval parameter-count mismatch.",
    )

    require(
        observed_total_parameter_count
        == TOTAL_PARAMETER_COUNT,
        "Total T9 parameter-count mismatch.",
    )

    print(
        f"Base parameters:                "
        f"{observed_base_parameter_count:,}"
    )

    print(
        f"Retrieval parameters:           "
        f"{observed_retrieval_parameter_count:,}"
    )

    print(
        f"Total T9 parameters:            "
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
        "Retrieval query:"
    )

    print(
        "  hidden[row, answer_causal_position, :]"
    )

    print(
        "Retrieval memory:"
    )

    print(
        "  hidden[row, position, :] for "
        "position < answer_causal_position"
    )

    print(
        "Retrieval module:"
    )

    print(
        "  nn.MultiheadAttention("
        "embed_dim=320, num_heads=1, "
        "batch_first=True, bias=True)"
    )

    print(
        "Residual:"
    )

    print(
        "  answer_hidden + retrieved_context"
    )

    print(
        "Answer logits:"
    )

    print(
        "  base_model.language_head("
        "augmented_answer_hidden)"
    )

    print(
        "T8 query-logit branch: ABSENT"
    )

    heading(
        "LOADING FINAL T9 CHECKPOINT"
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
        "Final T9 checkpoint is not a dictionary.",
    )

    require(
        "model_state"
        in checkpoint,
        "Final checkpoint missing model_state.",
    )

    require(
        int(
            checkpoint["treatment"]
        )
        == 9,
        "Checkpoint treatment number mismatch.",
    )

    require(
        checkpoint["treatment_name"]
        == "answer_position_content_addressable_retrieval",
        "Checkpoint treatment name mismatch.",
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

    require(
        checkpoint["trainer_sha256"]
        == EXPECTED_TRAINER_SHA256,
        "Checkpoint trainer SHA mismatch.",
    )

    require(
        checkpoint["pair_pool_sha256"]
        == EXPECTED_PAIR_POOL_SHA256,
        "Checkpoint pair-pool SHA mismatch.",
    )

    require(
        checkpoint["schedule_sha256"]
        == EXPECTED_SCHEDULE_SHA256,
        "Checkpoint schedule SHA mismatch.",
    )

    require(
        checkpoint[
            "start_checkpoint_sha256"
        ]
        == EXPECTED_START_CHECKPOINT_SHA256,
        "Checkpoint start SHA mismatch.",
    )

    require(
        int(
            checkpoint[
                "base_parameter_count"
            ]
        )
        == BASE_PARAMETER_COUNT,
        "Checkpoint base parameter-count mismatch.",
    )

    require(
        int(
            checkpoint[
                "retrieval_parameter_count"
            ]
        )
        == RETRIEVAL_PARAMETER_COUNT,
        (
            "Checkpoint retrieval "
            "parameter-count mismatch."
        ),
    )

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

    require(
        math.isclose(
            float(checkpoint["margin"]),
            SELECTOR_MARGIN,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Checkpoint margin mismatch.",
    )

    architecture = checkpoint[
        "architecture"
    ]

    require(
        architecture[
            "t8_query_logit_projection_retained"
        ]
        is False,
        (
            "Checkpoint unexpectedly retains "
            "the T8 query-logit branch."
        ),
    )

    model_state = checkpoint[
        "model_state"
    ]

    require(
        isinstance(model_state, dict),
        "Checkpoint model_state is not a dict.",
    )

    model_state_keys = list(
        model_state.keys()
    )

    require(
        any(
            key.startswith("base_model.")
            for key in model_state_keys
        ),
        "Checkpoint model_state missing base_model keys.",
    )

    require(
        any(
            key.startswith("retrieval.")
            for key in model_state_keys
        ),
        "Checkpoint model_state missing retrieval keys.",
    )

    require(
        not any(
            "query_projection" in key
            for key in model_state_keys
        ),
        (
            "Checkpoint unexpectedly contains "
            "T8 query_projection keys."
        ),
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
        "Strict full T9 state load:      PASS"
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

            retrieval_output = (
                model.retrieve_for_answers(
                    answer_positions,
                    need_weights=True,
                )
            )

            answer_logits = (
                retrieval_output[
                    "adjusted_answer_logits"
                ]
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

            overall_accumulator.update(
                answer_logits,
                target_ids,
                distractor_ids,
                retrieved_context,
                answer_hidden,
                augmented_hidden,
                attention_weights,
            )

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
                    retrieved_context[
                        slot0_mask
                    ],
                    answer_hidden[
                        slot0_mask
                    ],
                    augmented_hidden[
                        slot0_mask
                    ],
                    attention_weights[
                        slot0_mask
                    ],
                )

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
                    retrieved_context[
                        slot1_mask
                    ],
                    answer_hidden[
                        slot1_mask
                    ],
                    augmented_hidden[
                        slot1_mask
                    ],
                    attention_weights[
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

        gate_label = "PASS"

    else:
        classification = (
            "FINAL_TRAINING_RETENTION_BELOW_GATE"
        )

        gate_label = "FAIL"

    audit_result = {
        "treatment": 9,

        "treatment_name": (
            "answer_position_content_addressable_retrieval"
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

            "trainer_path": str(
                TRAINER_PATH
            ),

            "trainer_sha256": (
                trainer_sha256
            ),

            "preflight_result_path": str(
                PREFLIGHT_RESULT_PATH
            ),

            "training_result_path": str(
                TRAINING_RESULT_PATH
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

            "retrieval_parameter_count": (
                RETRIEVAL_PARAMETER_COUNT
            ),

            "total_parameter_count": (
                TOTAL_PARAMETER_COUNT
            ),

            "context_source": (
                "base_model.final_norm output"
            ),

            "query_source": (
                "hidden[row, answer_causal_position, :]"
            ),

            "memory_source": (
                "hidden[row, position, :] for "
                "position < answer_causal_position"
            ),

            "retrieval_module": (
                "nn.MultiheadAttention("
                "embed_dim=320,num_heads=1,"
                "batch_first=True,bias=True)"
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

    heading(
        "TREATMENT #9 FINAL RETENTION CLASSIFICATION"
    )

    print(
        f"Gate: {gate_label}"
    )

    print()

    print(
        classification
    )

    print()

    if overall_gate_pass:

        print(
            "T9 PASSED the preregistered exact "
            "training-retention gate."
        )

        print()

        print(
            "POSITIVE CONTROLS MAY BE CONSIDERED NEXT. "
            "SEALED EVALUATION REMAINS CLOSED UNTIL "
            "POSITIVE CONTROLS PASS."
        )

    else:

        print(
            "T9 FAILED the preregistered exact "
            "training-retention gate."
        )

        print()

        print(
            "NO POSITIVE CONTROLS. "
            "SEALED EVALUATION REMAINS CLOSED. "
            "NO EXTRA STEPS OR TUNING."
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
