from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer


# =============================================================================
# DaveLM v0.9 — Treatment #2
# READ-ONLY POSITIVE-CONTROL EVALUATOR
#
# This script exists because Treatment #2 training completed successfully,
# but the homemade evaluator crashed after checkpoint save.
#
# THIS SCRIPT:
# - performs ZERO training
# - creates ZERO optimizer steps
# - loads the exact already-saved Treatment #2 checkpoint
# - evaluates ONLY:
#       trained_anchors
#       supported_relation_withheld_geometry
# - uses the ORIGINAL v0.9 native _evaluate() function
# - computes distractor-target share AMONG ERRORS
# - applies the frozen 95% / 90% positive-control gates
# - NEVER opens sealed evaluation axes
#
# The Treatment #2 training result itself is NOT rerun.
# =============================================================================


# =============================================================================
# PATHS + FROZEN CONSTANTS
# =============================================================================

SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")
CADAVER_ROOT = Path(r"C:\DaveLM-CADAVER")

OUTPUT_ROOT = Path(
    r"C:\DaveLM-CADAVER\two_mapping_query_swap_manual_seed8380"
)

CANDIDATE_CHECKPOINT = (
    OUTPUT_ROOT
    / "checkpoints"
    / "paired_query_swap"
    / "seed_8380"
    / "latest.pt"
)

RESULTS_PATH = (
    OUTPUT_ROOT
    / "positive_control_results.json"
)

EVALUATION_LOG_PATH = (
    OUTPUT_ROOT
    / "positive_control_evaluation_readonly.json"
)


EXPECTED_CANDIDATE_SHA256 = (
    "bd8b2ab93f4a2eff7bc32f83c4230d5"
    "c142b8adfbea5be9a68787a102d473693"
)

EXPECTED_START_CHECKPOINT_SHA256 = (
    "345984c52a06db5f988aaf4cd47963ee"
    "a0e9d77489cee94dbb10816af2f5443e"
)

EXPECTED_PAIRED_SCHEDULE_SHA256 = (
    "4f26e9a72302be779a0b9edf25c8307b"
    "4ec9071712f49ee75763701be4c7b5e6"
)

EXPECTED_PARAMETER_COUNT = 10_594_944

EXPECTED_ANCHORS = 256
EXPECTED_SUPPORTED = 1536

ANCHOR_GATE = 0.95
SUPPORTED_GATE = 0.90


# =============================================================================
# IMPORT PREFLIGHT HELPERS + ORIGINAL v0.9 NATIVE EVALUATOR
# =============================================================================

if str(CADAVER_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(CADAVER_ROOT),
    )

import query_swap_preflight as pf


if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SOURCE_ROOT),
    )

import experiments.two_mapping_contextual_binding.run as original_run


# =============================================================================
# HELPERS
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

    if value is None or not callable(value):
        raise RuntimeError(
            f"Required function {name!r} "
            f"is unavailable in original v0.9 run.py."
        )

    return value


# =============================================================================
# SAFETY
# =============================================================================

def safety_check():

    header(
        "READ-ONLY EVALUATION SAFETY CHECK"
    )

    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(
            SOURCE_ROOT
        )

    if not CANDIDATE_CHECKPOINT.is_file():
        raise FileNotFoundError(
            CANDIDATE_CHECKPOINT
        )

    source_resolved = (
        SOURCE_ROOT.resolve()
    )

    output_resolved = (
        OUTPUT_ROOT.resolve()
    )

    try:
        output_resolved.relative_to(
            source_resolved
        )

        raise RuntimeError(
            "REFUSING TO RUN: output directory "
            "is inside protected DaveLM-v0.9."
        )

    except ValueError:
        pass

    actual_checkpoint_sha = (
        sha256_file(
            CANDIDATE_CHECKPOINT
        )
    )

    print(
        f"Protected source: {SOURCE_ROOT}"
    )

    print(
        f"Candidate checkpoint:\n"
        f"{CANDIDATE_CHECKPOINT}"
    )

    print()

    print(
        "Mode: READ-ONLY POSITIVE-CONTROL EVALUATION"
    )

    print(
        "Training: NO"
    )

    print(
        "Optimizer creation: NO"
    )

    print(
        "Backward passes: ZERO"
    )

    print(
        "Optimizer steps: ZERO"
    )

    print(
        "Sealed evaluation: DISABLED"
    )

    print()

    print(
        "Candidate checkpoint SHA256:"
    )

    print(
        actual_checkpoint_sha
    )

    if (
        actual_checkpoint_sha
        != EXPECTED_CANDIDATE_SHA256
    ):
        raise RuntimeError(
            "Candidate checkpoint SHA does not match "
            "the exact Treatment #2 checkpoint saved "
            "after the completed training run."
        )

    print()
    print(
        "Candidate checkpoint SHA: PASS"
    )


# =============================================================================
# LOAD ONLY THE TWO ALLOWED POSITIVE-CONTROL AXES
#
# IMPORTANT:
#
# We deliberately use query_swap_preflight.load_named_json_value()
# rather than json.load() on the whole corpus.
#
# Therefore sealed axes are not deserialized.
# =============================================================================

def load_positive_controls():

    header(
        "LOADING ALLOWED POSITIVE CONTROLS ONLY"
    )

    corpus_path = (
        pf.config.CORPUS_PATH
    )

    if not corpus_path.is_file():
        raise FileNotFoundError(
            corpus_path
        )

    anchors = (
        pf.load_named_json_value(
            corpus_path,
            "trained_anchors",
        )
    )

    supported = (
        pf.load_named_json_value(
            corpus_path,
            "supported_relation_withheld_geometry",
        )
    )

    if not isinstance(
        anchors,
        list,
    ):
        raise RuntimeError(
            "trained_anchors did not decode as a list."
        )

    if not isinstance(
        supported,
        list,
    ):
        raise RuntimeError(
            "supported_relation_withheld_geometry "
            "did not decode as a list."
        )

    if len(anchors) != EXPECTED_ANCHORS:
        raise RuntimeError(
            f"Expected {EXPECTED_ANCHORS} anchors; "
            f"got {len(anchors)}."
        )

    if len(supported) != EXPECTED_SUPPORTED:
        raise RuntimeError(
            f"Expected {EXPECTED_SUPPORTED} supported examples; "
            f"got {len(supported)}."
        )

    print(
        f"trained_anchors: "
        f"{len(anchors)} examples"
    )

    print(
        f"supported_relation_withheld_geometry: "
        f"{len(supported)} examples"
    )

    print()

    print(
        "Sealed axes deserialized: NONE"
    )

    return (
        anchors,
        supported,
    )


# =============================================================================
# MODEL LOAD
# =============================================================================

def load_candidate_model():

    header(
        "LOADING TREATMENT #2 CANDIDATE MODEL"
    )

    build_model = require_callable(
        original_run,
        "build_model",
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm/PyTorch GPU is unavailable."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"Device:  {device}"
    )

    print(
        f"GPU:     {torch.cuda.get_device_name(0)}"
    )

    print()

    checkpoint = torch.load(
        CANDIDATE_CHECKPOINT,
        map_location="cpu",
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Candidate checkpoint payload is not a dict."
        )

    if "model_state" not in checkpoint:
        raise RuntimeError(
            "Candidate checkpoint contains no model_state."
        )

    checkpoint_start_sha = (
        checkpoint.get(
            "starting_checkpoint_sha256"
        )
    )

    if (
        checkpoint_start_sha
        != EXPECTED_START_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "Candidate checkpoint does not record "
            "the frozen one-map starting checkpoint."
        )

    checkpoint_schedule_sha = (
        checkpoint.get(
            "paired_schedule_sha256"
        )
    )

    if (
        checkpoint_schedule_sha
        != EXPECTED_PAIRED_SCHEDULE_SHA256
    ):
        raise RuntimeError(
            "Candidate checkpoint does not record "
            "the frozen paired schedule SHA."
        )

    step = int(
        checkpoint.get(
            "step",
            -1,
        )
    )

    if step != 1000:
        raise RuntimeError(
            f"Candidate checkpoint step is {step}, not 1000."
        )

    model = build_model(
        "untied"
    )

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        f"Parameter count: {parameter_count:,}"
    )

    if (
        parameter_count
        != EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "DaveLM architecture parameter count changed."
        )

    model.load_state_dict(
        checkpoint["model_state"],
        strict=True,
    )

    model = model.to(
        device
    )

    model.eval()

    print(
        "Candidate model_state load: PASS"
    )

    print(
        "Recorded training step:     1000"
    )

    print(
        "Starting checkpoint SHA:    PASS"
    )

    print(
        "Paired schedule SHA:        PASS"
    )

    return (
        model,
        checkpoint,
        device,
    )


# =============================================================================
# DISTRACTOR SHARE AMONG ERRORS
#
# Native _evaluate() gives:
#   exact
#   distractor_capture
#
# Its stock distractor_capture_rate is over ALL examples.
#
# Our diagnostic of interest is:
#
#   among WRONG predictions,
#   what fraction equal the displayed distractor target?
# =============================================================================

def error_analysis(
    native_summary: dict,
) -> dict:

    details = native_summary.get(
        "examples_detail",
        [],
    )

    if not isinstance(
        details,
        list,
    ):
        raise RuntimeError(
            "Native evaluator examples_detail is not a list."
        )

    errors = [
        row
        for row in details
        if not bool(
            row["exact"]
        )
    ]

    distractor_errors = [
        row
        for row in errors
        if bool(
            row[
                "distractor_capture"
            ]
        )
    ]

    incorrect = len(
        errors
    )

    distractor_count = len(
        distractor_errors
    )

    share = (
        distractor_count
        / incorrect
        if incorrect
        else 0.0
    )

    return {
        "incorrect":
            incorrect,

        "distractor_target_errors":
            distractor_count,

        "distractor_target_share_among_errors":
            share,
    }


# =============================================================================
# DISPLAY ONE AXIS
# =============================================================================

def report_axis(
    name: str,
    summary: dict,
    gate: float,
):

    header(
        f"POSITIVE CONTROL: {name}"
    )

    exact = int(
        summary["exact"]
    )

    examples = int(
        summary["examples"]
    )

    accuracy = float(
        summary["exact_accuracy"]
    )

    errors = (
        error_analysis(
            summary
        )
    )

    passed = (
        accuracy >= gate
    )

    print(
        f"Exact:"
    )

    print(
        f"{exact}/{examples} "
        f"= {accuracy:.3%}"
    )

    print()

    print(
        f"Frozen gate:"
    )

    print(
        f"{gate:.0%}"
    )

    print()

    print(
        f"Gate result:"
    )

    print(
        "PASS"
        if passed
        else "FAIL"
    )

    print()

    print(
        "Distractor-target share among errors:"
    )

    if errors["incorrect"]:

        print(
            f"{errors['distractor_target_errors']}/"
            f"{errors['incorrect']} "
            f"= "
            f"{errors['distractor_target_share_among_errors']:.3%}"
        )

    else:

        print(
            "0/0 — no errors"
        )

    print()

    print(
        "Native target probability mean:"
    )

    print(
        f"{float(summary['target_probability_mean']):.6f}"
    )

    print()

    print(
        "Native target rank mean / median:"
    )

    print(
        f"{float(summary['target_rank_mean']):.3f} / "
        f"{float(summary['target_rank_median']):.3f}"
    )

    if (
        "by_query_slot"
        in summary
    ):

        print()
        print(
            "By query slot:"
        )

        for slot, slot_summary in (
            summary[
                "by_query_slot"
            ].items()
        ):

            print(
                f"  slot {slot}: "
                f"{slot_summary['exact']}/"
                f"{slot_summary['examples']} "
                f"= "
                f"{float(slot_summary['exact_accuracy']):.3%}"
            )

    return {
        "passed":
            passed,

        "error_analysis":
            errors,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    safety_check()

    native_evaluate = (
        require_callable(
            original_run,
            "_evaluate",
        )
    )

    tokenizer = (
        Tokenizer.from_file(
            str(
                pf.config.TOKENIZER_PATH
            )
        )
    )

    (
        anchors,
        supported,
    ) = load_positive_controls()

    (
        model,
        checkpoint,
        device,
    ) = load_candidate_model()

    # =========================================================================
    # NATIVE POSITIVE-CONTROL EVALUATION ONLY
    # =========================================================================

    header(
        "RUNNING ORIGINAL v0.9 NATIVE EVALUATOR"
    )

    print(
        "Evaluating trained_anchors..."
    )

    anchors_summary = (
        native_evaluate(
            model,
            tokenizer,
            anchors,
        )
    )

    print(
        "Evaluating supported_relation_withheld_geometry..."
    )

    supported_summary = (
        native_evaluate(
            model,
            tokenizer,
            supported,
        )
    )

    print()

    print(
        "Native positive-control evaluation complete."
    )

    print(
        "Sealed evaluation opened: NO"
    )

    anchors_report = (
        report_axis(
            "trained_anchors",
            anchors_summary,
            ANCHOR_GATE,
        )
    )

    supported_report = (
        report_axis(
            "supported_relation_withheld_geometry",
            supported_summary,
            SUPPORTED_GATE,
        )
    )

    overall_pass = (
        anchors_report["passed"]
        and supported_report["passed"]
    )

    if overall_pass:

        classification = (
            "POSITIVE-CONTROL PASS / "
            "ELIGIBLE FOR SEPARATE SEALED EVALUATION"
        )

    else:

        classification = (
            "INCONCLUSIVE / POSITIVE-CONTROL FAILURE"
        )

    # =========================================================================
    # WRITE READ-ONLY EVALUATION ARTIFACT
    # =========================================================================

    evaluation_payload = {
        "experiment":
            "DaveLM v0.9 Treatment #2 paired query-swap",

        "evaluation":
            "read-only native positive-control evaluation",

        "candidate_checkpoint":
            str(
                CANDIDATE_CHECKPOINT
            ),

        "candidate_checkpoint_sha256":
            EXPECTED_CANDIDATE_SHA256,

        "starting_checkpoint_sha256":
            EXPECTED_START_CHECKPOINT_SHA256,

        "paired_schedule_sha256":
            EXPECTED_PAIRED_SCHEDULE_SHA256,

        "classification":
            classification,

        "positive_controls": {
            "trained_anchors": {
                "native_summary":
                    anchors_summary,

                "error_analysis":
                    anchors_report[
                        "error_analysis"
                    ],

                "gate":
                    ANCHOR_GATE,

                "gate_pass":
                    anchors_report[
                        "passed"
                    ],
            },

            "supported_relation_withheld_geometry": {
                "native_summary":
                    supported_summary,

                "error_analysis":
                    supported_report[
                        "error_analysis"
                    ],

                "gate":
                    SUPPORTED_GATE,

                "gate_pass":
                    supported_report[
                        "passed"
                    ],
            },

            "overall_gate_pass":
                overall_pass,
        },

        "training_rerun":
            False,

        "optimizer_created":
            False,

        "backward_passes":
            0,

        "optimizer_steps":
            0,

        "sealed_evaluation_opened":
            False,

        "sealed_evaluation_performed":
            False,
    }

    write_json(
        EVALUATION_LOG_PATH,
        evaluation_payload,
    )

    # This is the canonical missing result artifact that the crashed
    # trainer was supposed to write after evaluation.
    if RESULTS_PATH.exists():

        print()
        print(
            "WARNING: positive_control_results.json already exists."
        )

        print(
            "Leaving existing file untouched."
        )

    else:

        compact_results = {
            "experiment":
                "DaveLM v0.9 Treatment #2 paired query-swap",

            "classification":
                classification,

            "candidate_checkpoint":
                str(
                    CANDIDATE_CHECKPOINT
                ),

            "candidate_checkpoint_sha256":
                EXPECTED_CANDIDATE_SHA256,

            "starting_checkpoint_sha256":
                EXPECTED_START_CHECKPOINT_SHA256,

            "paired_schedule_sha256":
                EXPECTED_PAIRED_SCHEDULE_SHA256,

            "positive_controls": {
                "trained_anchors": {
                    "exact":
                        int(
                            anchors_summary[
                                "exact"
                            ]
                        ),

                    "examples":
                        int(
                            anchors_summary[
                                "examples"
                            ]
                        ),

                    "exact_accuracy":
                        float(
                            anchors_summary[
                                "exact_accuracy"
                            ]
                        ),

                    "incorrect":
                        anchors_report[
                            "error_analysis"
                        ][
                            "incorrect"
                        ],

                    "distractor_target_errors":
                        anchors_report[
                            "error_analysis"
                        ][
                            "distractor_target_errors"
                        ],

                    "distractor_target_share_among_errors":
                        anchors_report[
                            "error_analysis"
                        ][
                            "distractor_target_share_among_errors"
                        ],

                    "gate":
                        ANCHOR_GATE,

                    "gate_pass":
                        anchors_report[
                            "passed"
                        ],
                },

                "supported_relation_withheld_geometry": {
                    "exact":
                        int(
                            supported_summary[
                                "exact"
                            ]
                        ),

                    "examples":
                        int(
                            supported_summary[
                                "examples"
                            ]
                        ),

                    "exact_accuracy":
                        float(
                            supported_summary[
                                "exact_accuracy"
                            ]
                        ),

                    "incorrect":
                        supported_report[
                            "error_analysis"
                        ][
                            "incorrect"
                        ],

                    "distractor_target_errors":
                        supported_report[
                            "error_analysis"
                        ][
                            "distractor_target_errors"
                        ],

                    "distractor_target_share_among_errors":
                        supported_report[
                            "error_analysis"
                        ][
                            "distractor_target_share_among_errors"
                        ],

                    "gate":
                        SUPPORTED_GATE,

                    "gate_pass":
                        supported_report[
                            "passed"
                        ],
                },

                "overall_gate_pass":
                    overall_pass,
            },

            "training_rerun":
                False,

            "sealed_evaluation_opened":
                False,

            "sealed_evaluation_performed":
                False,
        }

        write_json(
            RESULTS_PATH,
            compact_results,
        )

    # =========================================================================
    # FINAL VERDICT
    # =========================================================================

    header(
        "FINAL TREATMENT #2 POSITIVE-CONTROL VERDICT"
    )

    print(
        "Trained anchors:"
    )

    print(
        f"{anchors_summary['exact']}/"
        f"{anchors_summary['examples']} "
        f"= "
        f"{float(anchors_summary['exact_accuracy']):.3%}"
    )

    print(
        f"Gate: {ANCHOR_GATE:.0%} "
        f"-> "
        f"{'PASS' if anchors_report['passed'] else 'FAIL'}"
    )

    print()

    print(
        "Supported relation / withheld geometry:"
    )

    print(
        f"{supported_summary['exact']}/"
        f"{supported_summary['examples']} "
        f"= "
        f"{float(supported_summary['exact_accuracy']):.3%}"
    )

    print(
        f"Gate: {SUPPORTED_GATE:.0%} "
        f"-> "
        f"{'PASS' if supported_report['passed'] else 'FAIL'}"
    )

    print()

    anchor_errors = (
        anchors_report[
            "error_analysis"
        ]
    )

    supported_errors = (
        supported_report[
            "error_analysis"
        ]
    )

    print(
        "Anchor distractor-target share among errors:"
    )

    if anchor_errors["incorrect"]:

        print(
            f"{anchor_errors['distractor_target_errors']}/"
            f"{anchor_errors['incorrect']} "
            f"= "
            f"{anchor_errors['distractor_target_share_among_errors']:.3%}"
        )

    else:

        print(
            "0/0 — no errors"
        )

    print()

    print(
        "Supported distractor-target share among errors:"
    )

    if supported_errors["incorrect"]:

        print(
            f"{supported_errors['distractor_target_errors']}/"
            f"{supported_errors['incorrect']} "
            f"= "
            f"{supported_errors['distractor_target_share_among_errors']:.3%}"
        )

    else:

        print(
            "0/0 — no errors"
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
        "TRAINING RERUN: NO"
    )

    print(
        "OPTIMIZER CREATED: NO"
    )

    print(
        "BACKWARD PASSES: ZERO"
    )

    print(
        "OPTIMIZER STEPS: ZERO"
    )

    print(
        "SEALED EVALUATION OPENED: NO"
    )

    print(
        "SEALED EVALUATION PERFORMED: NO"
    )

    print()

    print(
        "Read-only evaluation artifact:"
    )

    print(
        EVALUATION_LOG_PATH
    )

    print()

    print(
        "Positive-control results:"
    )

    print(
        RESULTS_PATH
    )

    if overall_pass:

        print()
        print(
            "BOTH FROZEN POSITIVE-CONTROL GATES PASSED."
        )

        print(
            "STOP HERE."
        )

        print(
            "The candidate is eligible for a separately "
            "authorized sealed evaluation."
        )

        print(
            "This script does NOT perform that evaluation."
        )

    else:

        print()
        print(
            "AT LEAST ONE FROZEN POSITIVE-CONTROL GATE FAILED."
        )

        print(
            "STOP HERE."
        )

        print(
            "Do not extend training, rescue the checkpoint, "
            "or open sealed evaluation."
        )


if __name__ == "__main__":
    main()