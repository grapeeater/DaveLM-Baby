from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer


# =============================================================================
# DaveLM v0.9 — Treatment #4
# NATIVE POSITIVE-CONTROL EVALUATION ONLY
#
# Authorized axes:
#
#   1. trained_anchors
#   2. supported_relation_withheld_geometry
#
# Frozen gates:
#
#   trained anchors >= 95%
#   supported withheld geometry >= 90%
#
# CRITICAL SAFETY RULE:
#
# This script DOES NOT:
# - train
# - create an optimizer
# - call backward()
# - modify the checkpoint
# - call _load_corpus()
# - call _evaluation_records()
# - construct novel_pool_a / novel_pool_b
# - construct shortcut-control axes
# - evaluate sealed data
#
# It reconstructs ONLY the exact two positive-control families directly from
# the original v0.9 source logic and evaluates them using the original native
# _evaluate() implementation unchanged.
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

SOURCE_ROOT = Path(
    r"C:\DaveLM-v0.9"
)

TREATMENT_ROOT = Path(
    r"C:\DaveLM-CADAVER\two_mapping_membership_margin_manual_seed8380"
)

CANDIDATE_CHECKPOINT = (
    TREATMENT_ROOT
    / "checkpoints"
    / "membership_margin"
    / "seed_8380"
    / "latest.pt"
)

TRAINING_RESULT = (
    TREATMENT_ROOT
    / "treatment4_training_result.json"
)

OUTPUT_PATH = (
    TREATMENT_ROOT
    / "treatment4_positive_control_evaluation.json"
)


# =============================================================================
# FROZEN IDENTIFIERS
# =============================================================================

EXPECTED_CANDIDATE_SHA256 = (
    "2a3f7a24b793016d952a05ec32d84b57"
    "b0f6fd84af45d730a61c4b0a294b1476"
)

EXPECTED_START_SHA256 = (
    "345984c52a06db5f988aaf4cd47963ee"
    "a0e9d77489cee94dbb10816af2f5443e"
)

EXPECTED_SCHEDULE_SHA256 = (
    "a34bb91d258f11c6f7cb6b6cd89c79d"
    "8e6fa12d64af15dd14b028fb7ff675cff"
)


# =============================================================================
# FROZEN POSITIVE-CONTROL CONTRACT
# =============================================================================

ANCHOR_AXIS = "trained_anchors"

SUPPORTED_AXIS = (
    "supported_relation_withheld_geometry"
)

EXPECTED_ANCHORS = 256

EXPECTED_SUPPORTED = 1536

ANCHOR_GATE = 0.95

SUPPORTED_GATE = 0.90


# =============================================================================
# IMPORT ORIGINAL v0.9 IMPLEMENTATION
# =============================================================================

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SOURCE_ROOT),
    )

import experiments.two_mapping_contextual_binding.run as original_run
from experiments.two_mapping_contextual_binding import config


# =============================================================================
# BASIC HELPERS
# =============================================================================

def header(title: str) -> None:

    print()

    print(
        "=" * 96
    )

    print(
        title
    )

    print(
        "=" * 96
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
    payload: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def require_callable(
    module,
    name: str,
):

    value = getattr(
        module,
        name,
        None,
    )

    if (
        value is None
        or not callable(value)
    ):

        raise RuntimeError(
            f"Required original function {name!r} "
            "is unavailable."
        )

    return value


# =============================================================================
# SAFETY + PROVENANCE
# =============================================================================

def verify_candidate() -> dict:

    header(
        "TREATMENT #4 — CHECKPOINT AND PROVENANCE VERIFICATION"
    )

    if OUTPUT_PATH.exists():

        raise RuntimeError(
            "\nREFUSING TO OVERWRITE EXISTING POSITIVE-CONTROL RESULT:\n"
            f"{OUTPUT_PATH}\n"
        )

    if not CANDIDATE_CHECKPOINT.is_file():

        raise FileNotFoundError(
            CANDIDATE_CHECKPOINT
        )

    if not TRAINING_RESULT.is_file():

        raise FileNotFoundError(
            TRAINING_RESULT
        )

    observed_candidate_sha = sha256_file(
        CANDIDATE_CHECKPOINT
    )

    if (
        observed_candidate_sha
        != EXPECTED_CANDIDATE_SHA256
    ):

        raise RuntimeError(
            "\nCANDIDATE CHECKPOINT SHA MISMATCH.\n"
            f"Expected: {EXPECTED_CANDIDATE_SHA256}\n"
            f"Observed: {observed_candidate_sha}\n"
        )

    training = json.loads(
        TRAINING_RESULT.read_text(
            encoding="utf-8"
        )
    )

    recorded_candidate_sha = (
        training[
            "candidate_checkpoint"
        ][
            "sha256"
        ]
    )

    if (
        recorded_candidate_sha
        != EXPECTED_CANDIDATE_SHA256
    ):

        raise RuntimeError(
            "Training-result artifact candidate SHA mismatch."
        )

    recorded_start_sha = (
        training[
            "start_checkpoint"
        ][
            "sha256"
        ]
    )

    if (
        recorded_start_sha
        != EXPECTED_START_SHA256
    ):

        raise RuntimeError(
            "Training-result starting-checkpoint SHA mismatch."
        )

    recorded_schedule_sha = (
        training[
            "schedule_sha256"
        ]
    )

    if (
        recorded_schedule_sha
        != EXPECTED_SCHEDULE_SHA256
    ):

        raise RuntimeError(
            "Training-result schedule SHA mismatch."
        )

    if int(
        training[
            "steps"
        ]
    ) != 1000:

        raise RuntimeError(
            "Training artifact does not contain exactly 1000 steps."
        )

    if int(
        training[
            "supervised_tokens"
        ]
    ) != 6_176_000:

        raise RuntimeError(
            "Training artifact supervised-token count mismatch."
        )

    if int(
        training[
            "answer_substitutions"
        ]
    ) != 32_000:

        raise RuntimeError(
            "Training artifact answer-substitution count mismatch."
        )

    if training.get(
        "sealed_axes_loaded"
    ) is not False:

        raise RuntimeError(
            "Training artifact unexpectedly reports sealed axes loaded."
        )

    if training.get(
        "sealed_evaluation"
    ) is not False:

        raise RuntimeError(
            "Training artifact unexpectedly reports sealed evaluation."
        )

    print(
        "Candidate checkpoint SHA: PASS"
    )

    print(
        f"SHA256: {observed_candidate_sha}"
    )

    print()

    print(
        "Starting checkpoint provenance: PASS"
    )

    print(
        "Historical schedule provenance: PASS"
    )

    print(
        "1000-step training contract: PASS"
    )

    print(
        "6,176,000 supervised-token contract: PASS"
    )

    print(
        "32,000 answer-substitution contract: PASS"
    )

    print()

    print(
        "Training artifact says sealed axes loaded: NO"
    )

    print(
        "Training artifact says sealed evaluated:   NO"
    )

    return training


# =============================================================================
# LOAD MODEL
# =============================================================================

def extract_model_state(
    checkpoint,
):

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Candidate checkpoint is not a dictionary."
        )

    if "model_state" in checkpoint:

        return checkpoint[
            "model_state"
        ]

    if "model_state_dict" in checkpoint:

        return checkpoint[
            "model_state_dict"
        ]

    if (
        "model" in checkpoint
        and isinstance(
            checkpoint[
                "model"
            ],
            dict,
        )
    ):

        return checkpoint[
            "model"
        ]

    if (
        checkpoint
        and all(
            torch.is_tensor(value)
            for value in checkpoint.values()
        )
    ):

        return checkpoint

    raise RuntimeError(
        "Could not locate model state in candidate checkpoint."
    )


def load_candidate_model(
    device,
):

    header(
        "LOADING TREATMENT #4 CANDIDATE READ-ONLY"
    )

    build_model = require_callable(
        original_run,
        "build_model",
    )

    model = build_model(
        "untied"
    ).to(
        device
    )

    checkpoint = torch.load(
        CANDIDATE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state = extract_model_state(
        checkpoint
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Device:     {device}"
    )

    print(
        f"Parameters: {parameter_count:,}"
    )

    print(
        "State dict load: PASS"
    )

    print(
        "Optimizer created: NO"
    )

    print(
        "Training enabled: NO"
    )

    return (
        model,
        parameter_count,
    )


# =============================================================================
# RECONSTRUCT ONLY AUTHORIZED POSITIVE CONTROLS
# =============================================================================

def build_positive_controls():

    header(
        "RECONSTRUCTING ONLY THE TWO AUTHORIZED POSITIVE-CONTROL AXES"
    )

    tokenizer = Tokenizer.from_file(
        str(
            config.TOKENIZER_PATH
        )
    )

    identity_pool_fn = require_callable(
        original_run,
        "_identity_pool",
    )

    layout_tables_fn = require_callable(
        original_run,
        "_layout_tables",
    )

    eval_geometry_fn = require_callable(
        original_run,
        "_eval_geometry",
    )

    safe_distractor_fn = require_callable(
        original_run,
        "_safe_eval_distractor",
    )

    record_fn = require_callable(
        original_run,
        "_record",
    )

    groups = identity_pool_fn(
        tokenizer
    )

    identities = groups[
        "identity"
    ]

    filler = groups[
        "filler"
    ]

    tables = layout_tables_fn(
        tokenizer,
        groups,
    )

    # =========================================================================
    # POSITIVE CONTROL #1
    #
    # Exact source logic from original run.py lines 395–403:
    #
    # for pair in ANCHOR_GEOMETRY_PAIRS:
    #     for slot in (0,1):
    #         geometry = _eval_geometry(...)
    #         for source in range(64):
    #             target = (source + SUPPORTED_RELATION_OFFSET) % 64
    #             ...
    # =========================================================================

    anchors = []

    for pair in config.ANCHOR_GEOMETRY_PAIRS:

        for slot in (
            0,
            1,
        ):

            geometry = eval_geometry_fn(
                tables,
                slot,
                pair,
            )

            for source in range(
                64
            ):

                target = (
                    source
                    + config.SUPPORTED_RELATION_OFFSET
                ) % 64

                (
                    distractor_source,
                    distractor_target,
                    _,
                ) = safe_distractor_fn(
                    source,
                    target,
                    (
                        source
                        + 2
                    ) % 64,
                    f"anchor:{pair}:{slot}:{source}",
                )

                mappings = [
                    None,
                    None,
                ]

                mappings[
                    slot
                ] = (
                    source,
                    target,
                )

                mappings[
                    1 - slot
                ] = (
                    distractor_source,
                    distractor_target,
                )

                anchors.append(
                    record_fn(
                        tokenizer,
                        ANCHOR_AXIS,
                        len(
                            anchors
                        ),
                        identities,
                        filler,
                        mappings,
                        slot,
                        geometry,
                        training=False,
                    )
                )

    # =========================================================================
    # POSITIVE CONTROL #2
    #
    # Exact source logic from original run.py lines 405–413:
    #
    # for pair in sorted(WITHHELD_GEOMETRY_PAIRS):
    #     for slot in (0,1):
    #         geometry = _eval_geometry(...)
    #         for source in range(64):
    #             target = (source + 1) % 64
    #             ...
    # =========================================================================

    supported = []

    for pair in sorted(
        config.WITHHELD_GEOMETRY_PAIRS
    ):

        for slot in (
            0,
            1,
        ):

            geometry = eval_geometry_fn(
                tables,
                slot,
                pair,
            )

            for source in range(
                64
            ):

                target = (
                    source
                    + 1
                ) % 64

                (
                    distractor_source,
                    distractor_target,
                    _,
                ) = safe_distractor_fn(
                    source,
                    target,
                    (
                        source
                        + 2
                    ) % 64,
                    f"supported:{pair}:{slot}:{source}",
                )

                mappings = [
                    None,
                    None,
                ]

                mappings[
                    slot
                ] = (
                    source,
                    target,
                )

                mappings[
                    1 - slot
                ] = (
                    distractor_source,
                    distractor_target,
                )

                supported.append(
                    record_fn(
                        tokenizer,
                        SUPPORTED_AXIS,
                        len(
                            supported
                        ),
                        identities,
                        filler,
                        mappings,
                        slot,
                        geometry,
                        training=False,
                    )
                )

    # =========================================================================
    # HARD COUNTS
    # =========================================================================

    if len(
        anchors
    ) != EXPECTED_ANCHORS:

        raise RuntimeError(
            f"Expected {EXPECTED_ANCHORS} anchors; "
            f"constructed {len(anchors)}."
        )

    if len(
        supported
    ) != EXPECTED_SUPPORTED:

        raise RuntimeError(
            f"Expected {EXPECTED_SUPPORTED} supported examples; "
            f"constructed {len(supported)}."
        )

    # =========================================================================
    # NATIVE PREFIX AUDIT
    # =========================================================================

    bos_id = original_run.required_token_id(
        tokenizer,
        "<bos>",
    )

    for axis_name, records in (
        (
            ANCHOR_AXIS,
            anchors,
        ),
        (
            SUPPORTED_AXIS,
            supported,
        ),
    ):

        slot_counts = defaultdict(
            int
        )

        for index, row in enumerate(
            records
        ):

            native_prefix = (
                [
                    bos_id
                ]
                + tokenizer.encode(
                    str(
                        row[
                            "prompt"
                        ]
                    )
                ).ids
            )

            audited_prefix = [
                int(x)
                for x in row[
                    "layout"
                ][
                    "prefix_ids"
                ]
            ]

            if (
                native_prefix
                != audited_prefix
            ):

                raise RuntimeError(
                    f"{axis_name} record {index}: "
                    "native prefix mismatch."
                )

            if (
                int(
                    row[
                        "target_token_id"
                    ]
                )
                ==
                int(
                    row[
                        "distractor_token_id"
                    ]
                )
            ):

                raise RuntimeError(
                    f"{axis_name} record {index}: "
                    "target equals distractor."
                )

            slot_counts[
                int(
                    row[
                        "query_slot"
                    ]
                )
            ] += 1

        print(
            f"{axis_name}: {len(records)} records"
        )

        print(
            f"  slot 0: {slot_counts[0]}"
        )

        print(
            f"  slot 1: {slot_counts[1]}"
        )

        print(
            "  native prefix identity: PASS"
        )

    print()

    print(
        "Novel pools constructed:       NO"
    )

    print(
        "Shortcut controls constructed: NO"
    )

    print(
        "_evaluation_records() called:  NO"
    )

    print(
        "_load_corpus() called:         NO"
    )

    return (
        tokenizer,
        anchors,
        supported,
    )


# =============================================================================
# EXTRA READ-ONLY SUMMARY FROM NATIVE EVALUATOR DETAIL
# =============================================================================

def summarize_native_result(
    result: dict,
) -> dict:

    details = result.get(
        "examples_detail",
        [],
    )

    errors = [
        row
        for row in details
        if not bool(
            row[
                "exact"
            ]
        )
    ]

    distractor_errors = sum(
        bool(
            row[
                "distractor_capture"
            ]
        )
        for row in errors
    )

    if errors:

        distractor_share_among_errors = (
            distractor_errors
            / len(
                errors
            )
        )

    else:

        distractor_share_among_errors = 0.0

    target_probabilities = [
        float(
            row[
                "target_probability"
            ]
        )
        for row in details
    ]

    target_ranks = [
        int(
            row[
                "target_rank"
            ]
        )
        for row in details
    ]

    slot_groups = defaultdict(
        list
    )

    for row in details:

        slot_groups[
            int(
                row[
                    "query_slot"
                ]
            )
        ].append(
            row
        )

    by_slot = {}

    for slot in sorted(
        slot_groups
    ):

        rows = slot_groups[
            slot
        ]

        exact = sum(
            bool(
                row[
                    "exact"
                ]
            )
            for row in rows
        )

        slot_errors = [
            row
            for row in rows
            if not bool(
                row[
                    "exact"
                ]
            )
        ]

        slot_distractor_errors = sum(
            bool(
                row[
                    "distractor_capture"
                ]
            )
            for row in slot_errors
        )

        by_slot[
            str(
                slot
            )
        ] = {
            "exact":
                exact,

            "examples":
                len(
                    rows
                ),

            "exact_accuracy":
                (
                    exact
                    / len(
                        rows
                    )
                ),

            "errors":
                len(
                    slot_errors
                ),

            "distractor_errors":
                slot_distractor_errors,

            "distractor_share_among_errors":
                (
                    slot_distractor_errors
                    / len(
                        slot_errors
                    )
                    if slot_errors
                    else 0.0
                ),
        }

    return {
        "exact":
            int(
                result[
                    "exact"
                ]
            ),

        "examples":
            int(
                result[
                    "examples"
                ]
            ),

        "exact_accuracy":
            float(
                result[
                    "exact_accuracy"
                ]
            ),

        "distractor_capture":
            int(
                result[
                    "distractor_capture"
                ]
            ),

        "distractor_capture_rate":
            float(
                result[
                    "distractor_capture_rate"
                ]
            ),

        "errors":
            len(
                errors
            ),

        "distractor_errors":
            distractor_errors,

        "distractor_share_among_errors":
            distractor_share_among_errors,

        "target_probability_mean":
            (
                statistics.fmean(
                    target_probabilities
                )
                if target_probabilities
                else 0.0
            ),

        "target_rank_mean":
            (
                statistics.fmean(
                    target_ranks
                )
                if target_ranks
                else 0.0
            ),

        "target_rank_median":
            (
                statistics.median(
                    target_ranks
                )
                if target_ranks
                else 0.0
            ),

        "by_query_slot":
            by_slot,

        # Preserve native per-cell data if present.
        "by_cell":
            result.get(
                "by_cell",
                {},
            ),
    }


# =============================================================================
# EVALUATE
# =============================================================================

@torch.no_grad()
def evaluate_positive_controls(
    model,
    tokenizer,
    anchors,
    supported,
):

    header(
        "NATIVE POSITIVE-CONTROL EVALUATION"
    )

    native_evaluate = require_callable(
        original_run,
        "_evaluate",
    )

    # -------------------------------------------------------------------------
    # First gate: trained anchors
    # -------------------------------------------------------------------------

    print(
        "Evaluating trained anchors..."
    )

    anchor_native = native_evaluate(
        model,
        tokenizer,
        anchors,
    )

    anchor_summary = summarize_native_result(
        anchor_native
    )

    # -------------------------------------------------------------------------
    # Second gate: supported relation at withheld geometry
    # -------------------------------------------------------------------------

    print(
        "Evaluating supported relation at withheld geometry..."
    )

    supported_native = native_evaluate(
        model,
        tokenizer,
        supported,
    )

    supported_summary = summarize_native_result(
        supported_native
    )

    anchor_pass = (
        anchor_summary[
            "exact_accuracy"
        ]
        >= ANCHOR_GATE
    )

    supported_pass = (
        supported_summary[
            "exact_accuracy"
        ]
        >= SUPPORTED_GATE
    )

    both_pass = (
        anchor_pass
        and supported_pass
    )

    # =========================================================================
    # DISPLAY
    # =========================================================================

    header(
        "TREATMENT #4 — FROZEN POSITIVE-CONTROL RESULTS"
    )

    print(
        "TRAINED ANCHORS"
    )

    print(
        f"  Exact: "
        f"{anchor_summary['exact']}/"
        f"{anchor_summary['examples']} "
        f"= {anchor_summary['exact_accuracy']:.3%}"
    )

    print(
        f"  Gate:  {ANCHOR_GATE:.1%}"
    )

    print(
        f"  PASS:  {anchor_pass}"
    )

    print(
        f"  Distractor captures: "
        f"{anchor_summary['distractor_capture']}"
    )

    print(
        f"  Distractor share among errors: "
        f"{anchor_summary['distractor_share_among_errors']:.3%}"
    )

    print(
        f"  Target probability mean: "
        f"{anchor_summary['target_probability_mean']:.6f}"
    )

    print(
        f"  Target rank mean: "
        f"{anchor_summary['target_rank_mean']:.3f}"
    )

    print(
        f"  Target rank median: "
        f"{anchor_summary['target_rank_median']}"
    )

    for slot, metric in sorted(
        anchor_summary[
            "by_query_slot"
        ].items()
    ):

        print(
            f"  Slot {slot}: "
            f"{metric['exact']}/"
            f"{metric['examples']} "
            f"= {metric['exact_accuracy']:.3%}"
        )

    print()

    print(
        "SUPPORTED RELATION / WITHHELD GEOMETRY"
    )

    print(
        f"  Exact: "
        f"{supported_summary['exact']}/"
        f"{supported_summary['examples']} "
        f"= {supported_summary['exact_accuracy']:.3%}"
    )

    print(
        f"  Gate:  {SUPPORTED_GATE:.1%}"
    )

    print(
        f"  PASS:  {supported_pass}"
    )

    print(
        f"  Distractor captures: "
        f"{supported_summary['distractor_capture']}"
    )

    print(
        f"  Distractor share among errors: "
        f"{supported_summary['distractor_share_among_errors']:.3%}"
    )

    print(
        f"  Target probability mean: "
        f"{supported_summary['target_probability_mean']:.6f}"
    )

    print(
        f"  Target rank mean: "
        f"{supported_summary['target_rank_mean']:.3f}"
    )

    print(
        f"  Target rank median: "
        f"{supported_summary['target_rank_median']}"
    )

    for slot, metric in sorted(
        supported_summary[
            "by_query_slot"
        ].items()
    ):

        print(
            f"  Slot {slot}: "
            f"{metric['exact']}/"
            f"{metric['examples']} "
            f"= {metric['exact_accuracy']:.3%}"
        )

    print()

    print(
        "SEALED AXES EVALUATED: NO"
    )

    if both_pass:

        classification = (
            "PASS — BOTH POSITIVE-CONTROL GATES CLEARED"
        )

        rationale = (
            "Treatment #4 cleared both preregistered positive-control "
            "gates. Sealed evaluation remains unopened and requires "
            "separate explicit authorization."
        )

    else:

        classification = (
            "INCONCLUSIVE / POSITIVE-CONTROL FAILURE"
        )

        rationale = (
            "At least one frozen positive-control gate failed. "
            "Novel and shortcut-control pools remain sealed."
        )

    print()

    print(
        "CLASSIFICATION:"
    )

    print(
        classification
    )

    print()

    print(
        rationale
    )

    return {
        "classification":
            classification,

        "rationale":
            rationale,

        "positive_control_valid":
            both_pass,

        "gates": {
            "trained_anchors_required":
                ANCHOR_GATE,

            "supported_withheld_geometry_required":
                SUPPORTED_GATE,

            "trained_anchors_pass":
                anchor_pass,

            "supported_withheld_geometry_pass":
                supported_pass,
        },

        "evaluation": {
            ANCHOR_AXIS:
                anchor_summary,

            SUPPORTED_AXIS:
                supported_summary,
        },

        "sealed_evaluation_opened":
            False,

        "sealed_axes_constructed":
            False,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    header(
        "DaveLM v0.9 — TREATMENT #4 POSITIVE-CONTROL EXAM"
    )

    print(
        "This is a READ-ONLY native evaluation."
    )

    print()

    print(
        "Authorized:"
    )

    print(
        f"  {ANCHOR_AXIS}"
    )

    print(
        f"  {SUPPORTED_AXIS}"
    )

    print()

    print(
        "Forbidden in this run:"
    )

    print(
        "  novel pools"
    )

    print(
        "  order-swap controls"
    )

    print(
        "  query-switch controls"
    )

    print(
        "  distractor-swap controls"
    )

    print(
        "  any training or checkpoint mutation"
    )

    training_artifact = verify_candidate()

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

    else:

        # Evaluation is read-only and small enough to permit CPU fallback.
        device = torch.device(
            "cpu"
        )

    (
        model,
        parameter_count,
    ) = load_candidate_model(
        device
    )

    (
        tokenizer,
        anchors,
        supported,
    ) = build_positive_controls()

    result = evaluate_positive_controls(
        model,
        tokenizer,
        anchors,
        supported,
    )

    payload = {
        "experiment":
            "DaveLM v0.9 Treatment #4",

        "evaluation_stage":
            "frozen_native_positive_controls",

        "candidate_checkpoint": {
            "path":
                str(
                    CANDIDATE_CHECKPOINT
                ),

            "sha256":
                EXPECTED_CANDIDATE_SHA256,
        },

        "parameter_count":
            parameter_count,

        "training_provenance": {
            "start_checkpoint_sha256":
                EXPECTED_START_SHA256,

            "schedule_sha256":
                EXPECTED_SCHEDULE_SHA256,

            "steps":
                int(
                    training_artifact[
                        "steps"
                    ]
                ),

            "supervised_tokens":
                int(
                    training_artifact[
                        "supervised_tokens"
                    ]
                ),

            "answer_substitutions":
                int(
                    training_artifact[
                        "answer_substitutions"
                    ]
                ),
        },

        "authorized_axes": [
            ANCHOR_AXIS,
            SUPPORTED_AXIS,
        ],

        "authorized_record_counts": {
            ANCHOR_AXIS:
                len(
                    anchors
                ),

            SUPPORTED_AXIS:
                len(
                    supported
                ),
        },

        **result,
    }

    write_json(
        OUTPUT_PATH,
        payload,
    )

    header(
        "RESULT ARTIFACT WRITTEN"
    )

    print(
        OUTPUT_PATH
    )

    print()

    print(
        "Checkpoint modified: NO"
    )

    print(
        "Training performed:  NO"
    )

    print(
        "Sealed data opened:   NO"
    )

    print()

    print(
        "FINAL:"
    )

    print(
        result[
            "classification"
        ]
    )


if __name__ == "__main__":
    main()