from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


# =============================================================================
# DaveLM v0.9 — Treatment #6
# Exact Final Training-Retention Audit
#
# PURPOSE:
#
# Evaluate the FINAL Treatment #6 checkpoint on the exact 32,000 historical
# frozen training presentations.
#
# THIS SCRIPT IS STRICTLY READ-ONLY WITH RESPECT TO THE MODEL.
#
# It performs:
#   - NO optimizer construction
#   - NO backward()
#   - NO gradient computation
#   - NO training
#   - NO checkpoint modification
#   - NO positive-control evaluation
#   - NO sealed evaluation
#
# Pre-registered retention gate:
#
#   correct > distractor fraction >= 0.95
#
# AND
#
#   fraction satisfying:
#
#       correct_logit - distractor_logit >= 0.5
#
#   must be >= 0.90
#
# If either gate fails:
#
#   FINAL_TRAINING_RETENTION_BELOW_GATE
#
#   STOP.
#
# Positive controls remain unopened.
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

SOURCE_ROOT = Path(
    r"C:\DaveLM-v0.9"
)

PAIR_POOL_PATH = Path(
    r"C:\DaveLM-CADAVER"
    r"\treatment5_counterfactual_pairs_seed8382"
    r"\treatment5_full_document_pair_pool.json"
)

SCHEDULE_PATH = Path(
    r"C:\DaveLM-CADAVER"
    r"\treatment5_counterfactual_pairs_seed8382"
    r"\treatment5_full_document_frozen_schedule.json"
)

FINAL_CHECKPOINT = Path(
    r"C:\DaveLM-CADAVER"
    r"\treatment6_answer_signal_amplification_seed8380"
    r"\checkpoints"
    r"\answer_signal_amplification"
    r"\seed_8380"
    r"\latest.pt"
)

OUTPUT_ROOT = Path(
    r"C:\DaveLM-CADAVER"
    r"\treatment6_answer_signal_amplification_seed8380"
)

AUDIT_RESULT_PATH = (
    OUTPUT_ROOT
    / "treatment6_final_retention_audit.json"
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

EXPECTED_FINAL_CHECKPOINT_SHA256 = (
    "6c2498fef822897abb672d2ef25a4a88a"
    "63f66d57d441814372e59f91c995535"
)


# =============================================================================
# FROZEN EXPERIMENT CONSTANTS
# =============================================================================

EXPECTED_PARAMETER_COUNT = 10_594_944

EXPECTED_PAIR_COUNT = 1_536

EXPECTED_EVENTS = 32_000

EXPECTED_SLOT0 = 16_000

EXPECTED_SLOT1 = 16_000

EXPECTED_TOTAL_SUPERVISED = 6_144_000

EXPECTED_ANSWER_REPLACEMENTS = 32_000

EXPECTED_NONANSWER_SUPERVISED = 6_112_000

EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH = 193

EXPECTED_CAUSAL_SEQUENCE_LENGTH = 192

EXPECTED_NONANSWER_PER_EXAMPLE = 191

EXPECTED_PAIR_EXPOSURE_MIN = 10

EXPECTED_PAIR_EXPOSURE_MAX = 11

MARGIN = 0.5

ANSWER_WEIGHT = 191.0


# =============================================================================
# PRE-REGISTERED FINAL RETENTION GATE
# =============================================================================

CORRECT_GT_DISTRACTOR_GATE = 0.95

MARGIN_SATISFIED_GATE = 0.90


# =============================================================================
# HELPERS
# =============================================================================

def header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
    print()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

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
        ) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# IMPORT EXACT NATIVE MODEL BUILDER
# =============================================================================

def import_native_module():
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(SOURCE_ROOT),
        )

    from experiments.two_mapping_contextual_binding import run as original_run

    return original_run


# =============================================================================
# CHECKPOINT STATE EXTRACTION
# =============================================================================

def extract_model_state(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Final checkpoint is not a dictionary."
        )

    for key in (
        "model_state",
        "model_state_dict",
        "model",
    ):
        candidate = checkpoint.get(key)

        if isinstance(candidate, dict):
            return candidate

    if checkpoint and all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        return checkpoint

    raise RuntimeError(
        "Could not identify model state "
        "inside final checkpoint."
    )


# =============================================================================
# VERIFY FROZEN ARTIFACT HASHES
# =============================================================================

def verify_hashes() -> dict[str, str]:
    header(
        "VERIFYING FROZEN TREATMENT #6 AUDIT ARTIFACTS"
    )

    required = (
        PAIR_POOL_PATH,
        SCHEDULE_PATH,
        FINAL_CHECKPOINT,
    )

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required artifact not found:\n{path}"
            )

    pair_sha = sha256_file(
        PAIR_POOL_PATH
    )

    schedule_sha = sha256_file(
        SCHEDULE_PATH
    )

    checkpoint_sha = sha256_file(
        FINAL_CHECKPOINT
    )

    print("Pair-pool SHA256:")
    print(f"  observed: {pair_sha}")
    print(f"  expected: {EXPECTED_PAIR_POOL_SHA256}")

    if pair_sha != EXPECTED_PAIR_POOL_SHA256:
        raise RuntimeError(
            "PAIR-POOL SHA MISMATCH."
        )

    print("  PASS")
    print()

    print("Frozen schedule SHA256:")
    print(f"  observed: {schedule_sha}")
    print(f"  expected: {EXPECTED_SCHEDULE_SHA256}")

    if schedule_sha != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError(
            "SCHEDULE SHA MISMATCH."
        )

    print("  PASS")
    print()

    print("Final T6 checkpoint SHA256:")
    print(f"  observed: {checkpoint_sha}")
    print(
        f"  expected: "
        f"{EXPECTED_FINAL_CHECKPOINT_SHA256}"
    )

    if checkpoint_sha != (
        EXPECTED_FINAL_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "FINAL T6 CHECKPOINT SHA MISMATCH."
        )

    print("  PASS")

    return {
        "pair_pool_sha256": pair_sha,
        "schedule_sha256": schedule_sha,
        "final_checkpoint_sha256": checkpoint_sha,
    }


# =============================================================================
# LOAD FROZEN DATA
# =============================================================================

def load_artifacts():
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

    return pair_pool, schedule


# =============================================================================
# STRUCTURAL AUDIT
# =============================================================================

def audit_structure(
    pair_pool: dict[str, Any],
    schedule: dict[str, Any],
):
    header(
        "AUDITING EXACT 32,000-EVENT TRAINING HISTORY"
    )

    pairs = pair_pool.get("pairs")

    if not isinstance(pairs, list):
        raise RuntimeError(
            "Pair pool does not contain a valid 'pairs' list."
        )

    if len(pairs) != EXPECTED_PAIR_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIR_COUNT} pairs; "
            f"found {len(pairs)}."
        )

    pair_lookup: dict[
        str,
        dict[str, Any],
    ] = {}

    for pair in pairs:
        pair_id = str(
            pair["pair_id"]
        )

        if pair_id in pair_lookup:
            raise RuntimeError(
                f"Duplicate pair ID: {pair_id}"
            )

        twin_a = pair["twin_a"]
        twin_b = pair["twin_b"]

        if int(
            twin_a["query_slot"]
        ) != 0:
            raise RuntimeError(
                f"{pair_id}: Twin A slot != 0."
            )

        if int(
            twin_b["query_slot"]
        ) != 1:
            raise RuntimeError(
                f"{pair_id}: Twin B slot != 1."
            )

        a_target = int(
            twin_a["target_token_id"]
        )

        a_distractor = int(
            twin_a["distractor_token_id"]
        )

        b_target = int(
            twin_b["target_token_id"]
        )

        b_distractor = int(
            twin_b["distractor_token_id"]
        )

        if a_target == a_distractor:
            raise RuntimeError(
                f"{pair_id}: "
                f"Twin A target == distractor."
            )

        if b_target == b_distractor:
            raise RuntimeError(
                f"{pair_id}: "
                f"Twin B target == distractor."
            )

        if a_target != b_distractor:
            raise RuntimeError(
                f"{pair_id}: "
                f"A target != B distractor."
            )

        if b_target != a_distractor:
            raise RuntimeError(
                f"{pair_id}: "
                f"B target != A distractor."
            )

        query_position = int(
            pair["query_difference_position"]
        )

        answer_index = int(
            pair["answer_token_index"]
        )

        answer_position = int(
            pair["answer_causal_position"]
        )

        if answer_position != (
            answer_index - 1
        ):
            raise RuntimeError(
                f"{pair_id}: "
                f"answer causal-position mismatch."
            )

        doc_a = [
            int(x)
            for x in twin_a[
                "full_document_token_ids"
            ]
        ]

        doc_b = [
            int(x)
            for x in twin_b[
                "full_document_token_ids"
            ]
        ]

        if len(doc_a) != (
            EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH
        ):
            raise RuntimeError(
                f"{pair_id}: "
                f"Twin A length mismatch."
            )

        if len(doc_b) != (
            EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH
        ):
            raise RuntimeError(
                f"{pair_id}: "
                f"Twin B length mismatch."
            )

        differences = [
            index
            for index, (a, b)
            in enumerate(
                zip(
                    doc_a,
                    doc_b,
                )
            )
            if a != b
        ]

        expected_differences = sorted(
            [
                query_position,
                answer_index,
            ]
        )

        if differences != expected_differences:
            raise RuntimeError(
                f"{pair_id}: "
                f"twin document differences "
                f"{differences}; expected "
                f"{expected_differences}."
            )

        if doc_a[
            answer_index
        ] != a_target:
            raise RuntimeError(
                f"{pair_id}: "
                f"Twin A answer != target."
            )

        if doc_b[
            answer_index
        ] != b_target:
            raise RuntimeError(
                f"{pair_id}: "
                f"Twin B answer != target."
            )

        pair_lookup[
            pair_id
        ] = pair

    steps_data = schedule.get(
        "steps_data"
    )

    if not isinstance(
        steps_data,
        list,
    ):
        raise RuntimeError(
            "Schedule steps_data is not a list."
        )

    if len(steps_data) != 1000:
        raise RuntimeError(
            f"Expected 1000 schedule steps; "
            f"found {len(steps_data)}."
        )

    events: list[
        dict[str, Any]
    ] = []

    slot0 = 0
    slot1 = 0

    total_supervised = 0
    total_nonanswer = 0

    pair_exposures: dict[
        str,
        int,
    ] = {}

    for expected_step, step_data in enumerate(
        steps_data,
        start=1,
    ):
        if int(
            step_data["step"]
        ) != expected_step:
            raise RuntimeError(
                f"Schedule step mismatch at "
                f"{expected_step}."
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

        if len(examples) != 32:
            raise RuntimeError(
                f"Step {expected_step}: "
                f"batch size != 32."
            )

        for offset in range(
            0,
            32,
            2,
        ):
            a = examples[offset]
            b = examples[offset + 1]

            if str(
                a["pair_id"]
            ) != str(
                b["pair_id"]
            ):
                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"pair adjacency failure."
                )

            if str(
                a["twin"]
            ) != "A":
                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"first pair member != A."
                )

            if str(
                b["twin"]
            ) != "B":
                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"second pair member != B."
                )

            pair_id = str(
                a["pair_id"]
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
                example["pair_id"]
            )

            if pair_id not in pair_lookup:
                raise RuntimeError(
                    f"Schedule references "
                    f"unknown pair {pair_id}."
                )

            twin = str(
                example["twin"]
            )

            if twin == "A":
                frozen_twin = (
                    pair_lookup[
                        pair_id
                    ][
                        "twin_a"
                    ]
                )

            elif twin == "B":
                frozen_twin = (
                    pair_lookup[
                        pair_id
                    ][
                        "twin_b"
                    ]
                )

            else:
                raise RuntimeError(
                    f"Invalid twin label: {twin}"
                )

            schedule_doc = [
                int(x)
                for x in example[
                    "full_document_token_ids"
                ]
            ]

            frozen_doc = [
                int(x)
                for x in frozen_twin[
                    "full_document_token_ids"
                ]
            ]

            if schedule_doc != frozen_doc:
                raise RuntimeError(
                    f"{pair_id} {twin}: "
                    f"schedule/pool document mismatch."
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
                answer_index - 1
            ):
                raise RuntimeError(
                    f"{pair_id} {twin}: "
                    f"answer position mismatch."
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
                    f"{pair_id} {twin}: "
                    f"target == distractor."
                )

            if schedule_doc[
                answer_index
            ] != target:
                raise RuntimeError(
                    f"{pair_id} {twin}: "
                    f"document answer != target."
                )

            if target != int(
                frozen_twin[
                    "target_token_id"
                ]
            ):
                raise RuntimeError(
                    f"{pair_id} {twin}: "
                    f"target mismatch."
                )

            if distractor != int(
                frozen_twin[
                    "distractor_token_id"
                ]
            ):
                raise RuntimeError(
                    f"{pair_id} {twin}: "
                    f"distractor mismatch."
                )

            supervised = int(
                example[
                    "supervised_token_count"
                ]
            )

            nonanswer = int(
                example[
                    "nonanswer_supervised_token_count"
                ]
            )

            if supervised != (
                EXPECTED_CAUSAL_SEQUENCE_LENGTH
            ):
                raise RuntimeError(
                    f"{pair_id} {twin}: "
                    f"supervised count mismatch."
                )

            if nonanswer != (
                EXPECTED_NONANSWER_PER_EXAMPLE
            ):
                raise RuntimeError(
                    f"{pair_id} {twin}: "
                    f"non-answer count mismatch."
                )

            query_slot = int(
                example[
                    "query_slot"
                ]
            )

            if query_slot == 0:
                slot0 += 1

            elif query_slot == 1:
                slot1 += 1

            else:
                raise RuntimeError(
                    f"{pair_id} {twin}: "
                    f"invalid query slot."
                )

            total_supervised += supervised
            total_nonanswer += nonanswer

            events.append(
                example
            )

    if len(events) != EXPECTED_EVENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_EVENTS} events; "
            f"found {len(events)}."
        )

    if slot0 != EXPECTED_SLOT0:
        raise RuntimeError(
            f"Slot-0 count mismatch: {slot0}"
        )

    if slot1 != EXPECTED_SLOT1:
        raise RuntimeError(
            f"Slot-1 count mismatch: {slot1}"
        )

    if total_supervised != (
        EXPECTED_TOTAL_SUPERVISED
    ):
        raise RuntimeError(
            "Total supervised-position count mismatch."
        )

    if total_nonanswer != (
        EXPECTED_NONANSWER_SUPERVISED
    ):
        raise RuntimeError(
            "Total non-answer-position count mismatch."
        )

    exposure_values = list(
        pair_exposures.values()
    )

    if len(pair_exposures) != (
        EXPECTED_PAIR_COUNT
    ):
        raise RuntimeError(
            "Not all frozen pairs appear "
            "in historical schedule."
        )

    if min(exposure_values) != (
        EXPECTED_PAIR_EXPOSURE_MIN
    ):
        raise RuntimeError(
            "Minimum pair exposure mismatch."
        )

    if max(exposure_values) != (
        EXPECTED_PAIR_EXPOSURE_MAX
    ):
        raise RuntimeError(
            "Maximum pair exposure mismatch."
        )

    answer_indices = sorted(
        {
            int(
                event[
                    "answer_token_index"
                ]
            )
            for event in events
        }
    )

    answer_positions = sorted(
        {
            int(
                event[
                    "answer_causal_position"
                ]
            )
            for event in events
        }
    )

    query_positions = sorted(
        {
            int(
                event[
                    "query_difference_position"
                ]
            )
            for event in events
        }
    )

    print(
        f"Pairs:                         "
        f"{len(pair_lookup)}"
    )

    print(
        f"Historical training events:    "
        f"{len(events)}"
    )

    print(
        f"Slot 0 events:                 "
        f"{slot0}"
    )

    print(
        f"Slot 1 events:                 "
        f"{slot1}"
    )

    print(
        f"Total supervised positions:    "
        f"{total_supervised}"
    )

    print(
        f"Answer replacements:           "
        f"{len(events)}"
    )

    print(
        f"Ordinary non-answer CE:        "
        f"{total_nonanswer}"
    )

    print(
        f"Pair exposure range:           "
        f"{min(exposure_values)}-"
        f"{max(exposure_values)}"
    )

    print(
        f"Query-token positions:         "
        f"{query_positions}"
    )

    print(
        f"Answer-token indices:          "
        f"{answer_indices}"
    )

    print(
        f"Answer causal positions:       "
        f"{answer_positions}"
    )

    print()

    print(
        "Pair-pool structural audit: PASS"
    )

    print(
        "Schedule/pair exact document cross-check: PASS"
    )

    print(
        "Complete historical schedule integrity: PASS"
    )

    return events


# =============================================================================
# BUILD EXACT MODEL + LOAD FINAL T6 CHECKPOINT
# =============================================================================

def build_final_model(
    device: torch.device,
):
    header(
        "LOADING FINAL TREATMENT #6 MODEL"
    )

    original_run = import_native_module()

    build_model_fn = getattr(
        original_run,
        "build_model",
        None,
    )

    if not callable(
        build_model_fn
    ):
        raise RuntimeError(
            "Native run module does not expose "
            "callable build_model()."
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

    state = extract_model_state(
        checkpoint
    )

    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "\nFINAL T6 CHECKPOINT "
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

    model.eval()

    print(
        f"Parameters:      "
        f"{parameter_count:,}"
    )

    print(
        f"Device:          "
        f"{device}"
    )

    if device.type == "cuda":
        print(
            f"GPU:             "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        "Model builder:   "
        'original_run.build_model("untied")'
    )

    print(
        "Checkpoint load: PASS"
    )

    print(
        "model.eval():    YES"
    )

    print(
        "Optimizer:       NONE"
    )

    print(
        "Backward:        NONE"
    )

    print(
        "Training:        NONE"
    )

    return model, parameter_count


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
        "from model output."
    )


# =============================================================================
# METRIC ACCUMULATOR
# =============================================================================

class MetricAccumulator:
    def __init__(self):
        self.events = 0

        self.top1_correct = 0
        self.top5_correct = 0

        self.correct_gt = 0
        self.margin_satisfied = 0

        self.candidate_mass_sum = 0.0

        self.membership_loss_sum = 0.0
        self.selector_loss_sum = 0.0
        self.replacement_loss_sum = 0.0

        self.target_probability_sum = 0.0
        self.distractor_probability_sum = 0.0

        self.logit_delta_sum = 0.0

        self.rank_sum = 0.0
        self.ranks: list[float] = []

    def update(
        self,
        *,
        top1: torch.Tensor,
        top5: torch.Tensor,
        correct_gt: torch.Tensor,
        margin_satisfied: torch.Tensor,
        candidate_mass: torch.Tensor,
        membership_loss: torch.Tensor,
        selector_loss: torch.Tensor,
        replacement_loss: torch.Tensor,
        target_probability: torch.Tensor,
        distractor_probability: torch.Tensor,
        logit_delta: torch.Tensor,
        ranks: torch.Tensor,
    ):
        batch_events = int(
            top1.numel()
        )

        self.events += (
            batch_events
        )

        self.top1_correct += int(
            top1.sum().item()
        )

        self.top5_correct += int(
            top5.sum().item()
        )

        self.correct_gt += int(
            correct_gt.sum().item()
        )

        self.margin_satisfied += int(
            margin_satisfied.sum().item()
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

        self.replacement_loss_sum += float(
            replacement_loss.sum().item()
        )

        self.target_probability_sum += float(
            target_probability.sum().item()
        )

        self.distractor_probability_sum += float(
            distractor_probability.sum().item()
        )

        self.logit_delta_sum += float(
            logit_delta.sum().item()
        )

        self.rank_sum += float(
            ranks.sum().item()
        )

        self.ranks.extend(
            float(x)
            for x in ranks.detach().cpu().tolist()
        )

    def result(self) -> dict[str, Any]:
        if self.events == 0:
            raise RuntimeError(
                "Metric accumulator contains zero events."
            )

        sorted_ranks = sorted(
            self.ranks
        )

        count = len(
            sorted_ranks
        )

        if count % 2 == 1:
            median_rank = (
                sorted_ranks[
                    count // 2
                ]
            )
        else:
            left = sorted_ranks[
                count // 2 - 1
            ]

            right = sorted_ranks[
                count // 2
            ]

            median_rank = (
                left + right
            ) / 2.0

        return {
            "events": int(
                self.events
            ),
            "full_vocab_top1_fraction": (
                self.top1_correct
                / self.events
            ),
            "full_vocab_top5_fraction": (
                self.top5_correct
                / self.events
            ),
            "correct_gt_distractor_fraction": (
                self.correct_gt
                / self.events
            ),
            "margin_satisfied_fraction": (
                self.margin_satisfied
                / self.events
            ),
            "mean_candidate_mass": (
                self.candidate_mass_sum
                / self.events
            ),
            "mean_membership_loss": (
                self.membership_loss_sum
                / self.events
            ),
            "mean_selector_loss": (
                self.selector_loss_sum
                / self.events
            ),
            "mean_raw_answer_replacement_loss": (
                self.replacement_loss_sum
                / self.events
            ),
            "mean_target_probability": (
                self.target_probability_sum
                / self.events
            ),
            "mean_distractor_probability": (
                self.distractor_probability_sum
                / self.events
            ),
            "mean_logit_delta": (
                self.logit_delta_sum
                / self.events
            ),
            "mean_target_rank": (
                self.rank_sum
                / self.events
            ),
            "median_target_rank": float(
                median_rank
            ),
        }


# =============================================================================
# EVALUATE ONE BATCH
# =============================================================================

@torch.inference_mode()
def evaluate_batch(
    model,
    events: list[dict[str, Any]],
    device: torch.device,
):
    batch_size = len(
        events
    )

    documents = [
        [
            int(x)
            for x in event[
                "full_document_token_ids"
            ]
        ]
        for event in events
    ]

    lengths = {
        len(document)
        for document in documents
    }

    if lengths != {
        EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH
    }:
        raise RuntimeError(
            f"Unexpected document lengths: "
            f"{sorted(lengths)}"
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

    answer_positions = torch.tensor(
        [
            int(
                event[
                    "answer_causal_position"
                ]
            )
            for event in events
        ],
        dtype=torch.long,
        device=device,
    )

    correct_ids = torch.tensor(
        [
            int(
                event[
                    "target_token_id"
                ]
            )
            for event in events
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
            for event in events
        ],
        dtype=torch.long,
        device=device,
    )

    rows = torch.arange(
        batch_size,
        dtype=torch.long,
        device=device,
    )

    observed_targets = targets[
        rows,
        answer_positions,
    ]

    if not torch.equal(
        observed_targets,
        correct_ids,
    ):
        raise RuntimeError(
            "Causal answer target does not match "
            "frozen target ID."
        )

    output = model(
        inputs
    )

    logits = extract_logits(
        output
    )

    if logits.ndim != 3:
        raise RuntimeError(
            f"Expected rank-3 logits; "
            f"observed shape "
            f"{tuple(logits.shape)}."
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

    logit_delta = (
        correct_logits
        - distractor_logits
    )

    all_vocab_partition = torch.logsumexp(
        answer_logits,
        dim=-1,
    )

    candidate_partition = torch.logsumexp(
        torch.stack(
            (
                correct_logits,
                distractor_logits,
            ),
            dim=-1,
        ),
        dim=-1,
    )

    membership_loss = (
        all_vocab_partition
        - candidate_partition
    )

    selector_loss = F.relu(
        MARGIN
        - logit_delta
    )

    raw_replacement_loss = (
        membership_loss
        + selector_loss
    )

    probabilities = torch.softmax(
        answer_logits,
        dim=-1,
    )

    target_probability = probabilities[
        rows,
        correct_ids,
    ]

    distractor_probability = probabilities[
        rows,
        distractor_ids,
    ]

    candidate_mass = (
        target_probability
        + distractor_probability
    )

    top1 = (
        answer_logits.argmax(
            dim=-1
        )
        == correct_ids
    )

    top5_indices = answer_logits.topk(
        k=5,
        dim=-1,
    ).indices

    top5 = (
        top5_indices
        == correct_ids.unsqueeze(-1)
    ).any(
        dim=-1
    )

    correct_gt = (
        logit_delta > 0.0
    )

    margin_satisfied = (
        logit_delta >= MARGIN
    )

    ranks = (
        (
            answer_logits
            > correct_logits.unsqueeze(-1)
        )
        .sum(
            dim=-1
        )
        + 1
    ).float()

    return {
        "top1": top1,
        "top5": top5,
        "correct_gt": correct_gt,
        "margin_satisfied": margin_satisfied,
        "candidate_mass": candidate_mass,
        "membership_loss": membership_loss,
        "selector_loss": selector_loss,
        "replacement_loss": raw_replacement_loss,
        "target_probability": target_probability,
        "distractor_probability": distractor_probability,
        "logit_delta": logit_delta,
        "ranks": ranks,
    }


# =============================================================================
# COMPLETE 32K AUDIT
# =============================================================================

def run_retention_audit(
    model,
    events: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
):
    header(
        "RUNNING EXACT FINAL 32,000-EVENT RETENTION AUDIT"
    )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    overall = MetricAccumulator()

    slot0 = MetricAccumulator()

    slot1 = MetricAccumulator()

    total_events = len(
        events
    )

    start_time = (
        time.perf_counter()
    )

    processed = 0

    for start in range(
        0,
        total_events,
        batch_size,
    ):
        batch_events = events[
            start:
            start + batch_size
        ]

        batch_metrics = evaluate_batch(
            model,
            batch_events,
            device,
        )

        overall.update(
            **batch_metrics
        )

        # ---------------------------------------------------------------------
        # Slot-specific slices from exact same inference results.
        # ---------------------------------------------------------------------

        for slot, accumulator in (
            (0, slot0),
            (1, slot1),
        ):
            indices = [
                index
                for index, event
                in enumerate(
                    batch_events
                )
                if int(
                    event[
                        "query_slot"
                    ]
                ) == slot
            ]

            if not indices:
                continue

            index_tensor = torch.tensor(
                indices,
                dtype=torch.long,
                device=device,
            )

            accumulator.update(
                top1=batch_metrics[
                    "top1"
                ][
                    index_tensor
                ],
                top5=batch_metrics[
                    "top5"
                ][
                    index_tensor
                ],
                correct_gt=batch_metrics[
                    "correct_gt"
                ][
                    index_tensor
                ],
                margin_satisfied=batch_metrics[
                    "margin_satisfied"
                ][
                    index_tensor
                ],
                candidate_mass=batch_metrics[
                    "candidate_mass"
                ][
                    index_tensor
                ],
                membership_loss=batch_metrics[
                    "membership_loss"
                ][
                    index_tensor
                ],
                selector_loss=batch_metrics[
                    "selector_loss"
                ][
                    index_tensor
                ],
                replacement_loss=batch_metrics[
                    "replacement_loss"
                ][
                    index_tensor
                ],
                target_probability=batch_metrics[
                    "target_probability"
                ][
                    index_tensor
                ],
                distractor_probability=batch_metrics[
                    "distractor_probability"
                ][
                    index_tensor
                ],
                logit_delta=batch_metrics[
                    "logit_delta"
                ][
                    index_tensor
                ],
                ranks=batch_metrics[
                    "ranks"
                ][
                    index_tensor
                ],
            )

        processed += len(
            batch_events
        )

        if (
            processed % 3200 == 0
            or processed == total_events
        ):
            elapsed = (
                time.perf_counter()
                - start_time
            )

            print(
                f"Processed "
                f"{processed:>6d}/"
                f"{total_events} events "
                f"({processed / total_events:>6.1%}) "
                f"— {elapsed:.1f}s"
            )

    runtime = (
        time.perf_counter()
        - start_time
    )

    overall_result = (
        overall.result()
    )

    slot0_result = (
        slot0.result()
    )

    slot1_result = (
        slot1.result()
    )

    if overall_result[
        "events"
    ] != EXPECTED_EVENTS:
        raise RuntimeError(
            "Overall event-count mismatch."
        )

    if slot0_result[
        "events"
    ] != EXPECTED_SLOT0:
        raise RuntimeError(
            "Slot-0 event-count mismatch."
        )

    if slot1_result[
        "events"
    ] != EXPECTED_SLOT1:
        raise RuntimeError(
            "Slot-1 event-count mismatch."
        )

    return (
        overall_result,
        slot0_result,
        slot1_result,
        runtime,
    )


# =============================================================================
# REPORT METRICS
# =============================================================================

def print_metrics(
    title: str,
    metrics: dict[str, Any],
):
    print()
    print(title)
    print("-" * len(title))

    print(
        f"Events:                       "
        f"{metrics['events']}"
    )

    print(
        f"Full-vocab top1:              "
        f"{metrics['full_vocab_top1_fraction']:.6f}"
    )

    print(
        f"Full-vocab top5:              "
        f"{metrics['full_vocab_top5_fraction']:.6f}"
    )

    print(
        f"Correct > distractor:         "
        f"{metrics['correct_gt_distractor_fraction']:.6f}"
    )

    print(
        f"Margin >= {MARGIN}:             "
        f"{metrics['margin_satisfied_fraction']:.6f}"
    )

    print(
        f"Candidate mass:               "
        f"{metrics['mean_candidate_mass']:.6f}"
    )

    print(
        f"Membership loss:              "
        f"{metrics['mean_membership_loss']:.6f}"
    )

    print(
        f"Selector loss:                "
        f"{metrics['mean_selector_loss']:.6f}"
    )

    print(
        f"Raw answer replacement loss:  "
        f"{metrics['mean_raw_answer_replacement_loss']:.6f}"
    )

    print(
        f"Target probability:           "
        f"{metrics['mean_target_probability']:.6f}"
    )

    print(
        f"Distractor probability:       "
        f"{metrics['mean_distractor_probability']:.6f}"
    )

    print(
        f"Mean logit delta:             "
        f"{metrics['mean_logit_delta']:+.6f}"
    )

    print(
        f"Target rank mean:             "
        f"{metrics['mean_target_rank']:.6f}"
    )

    print(
        f"Target rank median:           "
        f"{metrics['median_target_rank']:.6f}"
    )


# =============================================================================
# MAIN AUDIT
# =============================================================================

def audit(
    device_name: str,
    batch_size: int,
):
    full_start = (
        time.perf_counter()
    )

    hashes = verify_hashes()

    pair_pool, schedule = (
        load_artifacts()
    )

    events = audit_structure(
        pair_pool,
        schedule,
    )

    header(
        "AUDIT EXECUTION CONTRACT"
    )

    print(
        "Final checkpoint only: YES"
    )

    print(
        "Exact 32,000 historical events: YES"
    )

    print(
        "model.eval(): YES"
    )

    print(
        "torch.inference_mode(): YES"
    )

    print(
        "Optimizer constructed: NO"
    )

    print(
        "Backward called: NO"
    )

    print(
        "Gradients used: NO"
    )

    print(
        "Training performed: NO"
    )

    print(
        "Positive controls evaluated: NO"
    )

    print(
        "Sealed evaluation loaded: NO"
    )

    print()

    print(
        "Pre-registered retention gate:"
    )

    print(
        f"  correct > distractor "
        f">= {CORRECT_GT_DISTRACTOR_GATE:.2f}"
    )

    print(
        f"  fraction with logit margin "
        f">= {MARGIN:.1f} "
        f">= {MARGIN_SATISFIED_GATE:.2f}"
    )

    # =========================================================================
    # DEVICE
    # =========================================================================

    if (
        device_name == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA/ROCm requested, but "
            "torch.cuda.is_available() is False."
        )

    device = torch.device(
        device_name
    )

    model, parameter_count = (
        build_final_model(
            device
        )
    )

    # =========================================================================
    # RUN EXACT RETENTION AUDIT
    # =========================================================================

    (
        overall,
        slot0,
        slot1,
        evaluation_runtime,
    ) = run_retention_audit(
        model,
        events,
        device,
        batch_size,
    )

    # =========================================================================
    # DECISION GATE
    # =========================================================================

    correct_gate_pass = (
        overall[
            "correct_gt_distractor_fraction"
        ]
        >= CORRECT_GT_DISTRACTOR_GATE
    )

    margin_gate_pass = (
        overall[
            "margin_satisfied_fraction"
        ]
        >= MARGIN_SATISFIED_GATE
    )

    retention_pass = (
        correct_gate_pass
        and margin_gate_pass
    )

    if retention_pass:
        classification = (
            "STRONG_FINAL_TRAINING_RETENTION"
        )

        next_action = (
            "Retention gate passed. "
            "The same unchanged positive controls "
            "used for T4 are now authorized. "
            "Sealed novel/shortcut evaluation remains "
            "closed until positive controls pass."
        )

    else:
        classification = (
            "FINAL_TRAINING_RETENTION_BELOW_GATE"
        )

        next_action = (
            "STOP before positive controls. "
            "Treatment 6 did not demonstrate strong "
            "final retention of the paired training task. "
            "Positive controls remain unopened. "
            "Sealed evaluation remains closed."
        )

    total_runtime = (
        time.perf_counter()
        - full_start
    )

    # =========================================================================
    # SAVE RESULT
    # =========================================================================

    result = {
        "experiment": (
            "DaveLM v0.9 Treatment #6"
        ),
        "audit": (
            "exact final historical "
            "training-retention audit"
        ),
        "treatment": (
            "answer-signal amplification"
        ),
        "classification": (
            classification
        ),
        "hashes": hashes,
        "checkpoint": {
            "path": str(
                FINAL_CHECKPOINT
            ),
            "sha256": (
                EXPECTED_FINAL_CHECKPOINT_SHA256
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
        "answer_weight": (
            ANSWER_WEIGHT
        ),
        "margin": (
            MARGIN
        ),
        "historical_event_count": (
            EXPECTED_EVENTS
        ),
        "slot_0_events": (
            EXPECTED_SLOT0
        ),
        "slot_1_events": (
            EXPECTED_SLOT1
        ),
        "total_supervised_positions": (
            EXPECTED_TOTAL_SUPERVISED
        ),
        "answer_replacements": (
            EXPECTED_ANSWER_REPLACEMENTS
        ),
        "ordinary_nonanswer_ce_positions": (
            EXPECTED_NONANSWER_SUPERVISED
        ),
        "retention_gate": {
            "correct_gt_distractor_fraction_min": (
                CORRECT_GT_DISTRACTOR_GATE
            ),
            "required_logit_margin": (
                MARGIN
            ),
            "margin_satisfied_fraction_min": (
                MARGIN_SATISFIED_GATE
            ),
            "correct_gate_pass": bool(
                correct_gate_pass
            ),
            "margin_gate_pass": bool(
                margin_gate_pass
            ),
            "overall_pass": bool(
                retention_pass
            ),
        },
        "overall": (
            overall
        ),
        "slot_0": (
            slot0
        ),
        "slot_1": (
            slot1
        ),
        "execution": {
            "device": str(
                device
            ),
            "batch_size": int(
                batch_size
            ),
            "model_eval": True,
            "inference_mode": True,
            "optimizer_created": False,
            "backward_called": False,
            "training_performed": False,
            "positive_controls_evaluated": False,
            "sealed_evaluation_loaded": False,
            "evaluation_runtime_seconds": float(
                evaluation_runtime
            ),
            "total_runtime_seconds": float(
                total_runtime
            ),
        },
        "next_action": (
            next_action
        ),
    }

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        AUDIT_RESULT_PATH,
        result,
    )

    # =========================================================================
    # FINAL HUMAN-READABLE REPORT
    # =========================================================================

    header(
        "TREATMENT #6 FINAL TRAINING-RETENTION AUDIT"
    )

    print_metrics(
        "OVERALL — ALL 32,000 HISTORICAL EVENTS",
        overall,
    )

    print_metrics(
        "SLOT 0 — 16,000 EVENTS",
        slot0,
    )

    print_metrics(
        "SLOT 1 — 16,000 EVENTS",
        slot1,
    )

    print()

    print(
        "=" * 100
    )

    print(
        "PRE-REGISTERED DECISION"
    )

    print(
        "=" * 100
    )

    print()

    print(
        f"Correct > distractor gate:"
    )

    print(
        f"  observed: "
        f"{overall['correct_gt_distractor_fraction']:.6f}"
    )

    print(
        f"  required: "
        f"{CORRECT_GT_DISTRACTOR_GATE:.6f}"
    )

    print(
        f"  result:   "
        f"{'PASS' if correct_gate_pass else 'FAIL'}"
    )

    print()

    print(
        f"Margin >= {MARGIN} fraction gate:"
    )

    print(
        f"  observed: "
        f"{overall['margin_satisfied_fraction']:.6f}"
    )

    print(
        f"  required: "
        f"{MARGIN_SATISFIED_GATE:.6f}"
    )

    print(
        f"  result:   "
        f"{'PASS' if margin_gate_pass else 'FAIL'}"
    )

    print()

    print(
        f"CLASSIFICATION:"
    )

    print(
        f"  {classification}"
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
        "NEXT ACTION:"
    )

    print(
        f"  {next_action}"
    )

    print()

    print(
        "AUDIT RESULT:"
    )

    print(
        f"  {AUDIT_RESULT_PATH}"
    )

    print()

    print(
        f"Evaluation runtime: "
        f"{evaluation_runtime:.2f} seconds"
    )

    print(
        f"Total runtime: "
        f"{total_runtime:.2f} seconds"
    )


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "DaveLM v0.9 Treatment #6 "
            "exact final 32,000-event "
            "training-retention audit."
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
            "Evaluation device. "
            "ROCm PyTorch uses 'cuda' "
            "for AMD GPUs."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help=(
            "Read-only evaluation batch size. "
            "Default: 256."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    audit(
        device_name=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()