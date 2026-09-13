from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from collections import Counter

import torch
from tokenizers import Tokenizer


# =============================================================================
# DaveLM v0.9 — Treatment #3
# READ-ONLY CAUSAL PREFIX IDENTITY AUDIT
#
# PURPOSE
#
# Treatment #3 trained a pairwise correct-vs-distractor margin at the causal
# position whose sampled TARGET token was proven to be the audited answer.
#
# This audit answers the remaining integrity question:
#
#   Was the exact causal INPUT prefix producing that margin logit token-identical
#   to the native v0.9 evaluator's audited layout.prefix_ids?
#
# IMPORTANT OFF-BY-ONE:
#
# If:
#
#   target[p] = answer
#
# then:
#
#   logits[p]
#
# predicts that answer after seeing inputs THROUGH position p inclusive.
#
# Therefore the actual causal prefix visible to the model is:
#
#   inputs[b, :p + 1]
#
# NOT:
#
#   inputs[b, :p]
#
# THIS SCRIPT DOES NOT:
# - build a model
# - load a model checkpoint into a model
# - create an optimizer
# - calculate gradients
# - call backward()
# - perform optimizer.step()
# - train
# - evaluate positive controls
# - deserialize sealed axes
# - perform sealed evaluation
#
# It reconstructs only:
# - the original 1,536 training records
# - the original frozen 1,000-step / 32-document schedule
#
# Expected historical schedule SHA:
#
# a34bb91d258f11c6f7cb6b6cd89c79d8e6fa12d64af15dd14b028fb7ff675cff
# =============================================================================


# =============================================================================
# PATHS / FROZEN CONSTANTS
# =============================================================================

SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")

OUTPUT_ROOT = Path(
    r"C:\DaveLM-CADAVER\two_mapping_selection_margin_manual_seed8380"
)

AUDIT_PATH = (
    OUTPUT_ROOT
    / "selection_margin_prefix_identity_audit.json"
)

EXPECTED_SCHEDULE_SHA256 = (
    "a34bb91d258f11c6f7cb6b6cd89c79d"
    "8e6fa12d64af15dd14b028fb7ff675cff"
)

SCHEDULE_SEED = 8381

MAX_STEPS = 1000
BATCH_SIZE = 32
CONTEXT_SIZE = 256

EXPECTED_TRAINING_RECORDS = 1536
EXPECTED_SAMPLED_EVENTS = 32_000


# =============================================================================
# IMPORT EXACT v0.9 MACHINERY
# =============================================================================

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SOURCE_ROOT),
    )

import experiments.two_mapping_contextual_binding.run as original_run

from experiments.two_mapping_contextual_binding import config

from v0_8.document_sampling import (
    DocumentTokenStore,
    sample_document_batch,
)

from v0_8_3.protection import (
    strict_json_dumps,
)


# =============================================================================
# HELPERS
# =============================================================================

def header(title: str) -> None:

    print()
    print("=" * 96)
    print(title)
    print("=" * 96)
    print()


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
            f"Required v0.9 function {name!r} is unavailable."
        )

    return value


# =============================================================================
# SAFETY
# =============================================================================

def safety_check() -> None:

    header(
        "TREATMENT #3 PREFIX AUDIT SAFETY"
    )

    if not SOURCE_ROOT.is_dir():

        raise FileNotFoundError(
            SOURCE_ROOT
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
            "is inside protected C:\\DaveLM-v0.9."
        )

    except ValueError:
        pass

    print(
        f"Protected source: {SOURCE_ROOT}"
    )

    print(
        f"Audit output:     {OUTPUT_ROOT}"
    )

    print()

    print(
        "Mode: READ-ONLY PREFIX AUDIT"
    )

    print(
        "Model construction: NO"
    )

    print(
        "Optimizer construction: NO"
    )

    print(
        "Gradient calculation: NO"
    )

    print(
        "Backward passes: ZERO"
    )

    print(
        "Optimizer steps: ZERO"
    )

    print(
        "Positive-control evaluation: NO"
    )

    print(
        "Sealed axes loaded/evaluated: NO"
    )


# =============================================================================
# RECONSTRUCT ORIGINAL TRAINING RECORDS
# =============================================================================

def reconstruct_training_records():

    header(
        "RECONSTRUCTING ORIGINAL BALANCED TWO-MAP TRAINING CORPUS"
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

    training_records_fn = require_callable(
        original_run,
        "_training_records",
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

    records = training_records_fn(
        tokenizer,
        identities,
        filler,
        tables,
    )

    if (
        len(records)
        != EXPECTED_TRAINING_RECORDS
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_TRAINING_RECORDS} records; "
            f"got {len(records)}."
        )

    print(
        f"Training records: {len(records)}"
    )

    print(
        f"Identity pool:    {len(identities)}"
    )

    print(
        f"Filler pool:      {len(filler)}"
    )

    return (
        tokenizer,
        records,
    )


# =============================================================================
# VERIFY NATIVE AUDITED PREFIXES
#
# This independently rechecks:
#
# layout.prefix_ids
#
# equals:
#
# [bos] + tokenizer.encode(prompt).ids
# =============================================================================

def verify_native_prefixes(
    tokenizer,
    records,
):

    header(
        "VERIFYING NATIVE v0.9 AUDITED PREFIXES"
    )

    bos_id = original_run.required_token_id(
        tokenizer,
        "<bos>",
    )

    prefix_lengths = Counter()

    for index, row in enumerate(
        records
    ):

        if "prompt" not in row:

            raise RuntimeError(
                f"Record {index}: prompt missing."
            )

        if "layout" not in row:

            raise RuntimeError(
                f"Record {index}: layout missing."
            )

        if "prefix_ids" not in row["layout"]:

            raise RuntimeError(
                f"Record {index}: layout.prefix_ids missing."
            )

        audited_prefix = [
            int(x)
            for x in row[
                "layout"
            ][
                "prefix_ids"
            ]
        ]

        reconstructed_prefix = (
            [bos_id]
            + tokenizer.encode(
                str(
                    row[
                        "prompt"
                    ]
                )
            ).ids
        )

        if (
            audited_prefix
            != reconstructed_prefix
        ):

            raise RuntimeError(
                "\n"
                "NATIVE PREFIX AUDIT FAILED.\n"
                f"Record index: {index}\n"
                f"Audited prefix length: {len(audited_prefix)}\n"
                f"Reconstructed prefix length: {len(reconstructed_prefix)}\n"
            )

        prefix_lengths[
            len(
                audited_prefix
            )
        ] += 1

    print(
        f"Records checked: {len(records)}"
    )

    print(
        "layout.prefix_ids == BOS + tokenized prompt: PASS"
    )

    print()

    print(
        "Native prefix length distribution:"
    )

    for length, count in sorted(
        prefix_lengths.items()
    ):

        print(
            f"  {length}: {count}"
        )

    return prefix_lengths


# =============================================================================
# BUILD ORIGINAL DOCUMENT STORE
# =============================================================================

def build_store(
    tokenizer,
    records,
):

    header(
        "BUILDING ORIGINAL TRAINING DOCUMENT STORE"
    )

    store = (
        DocumentTokenStore
        .from_documents(
            tokenizer,
            records,
        )
    )

    print(
        f"Training documents: {len(records)}"
    )

    print(
        "Document store: PASS"
    )

    return store


# =============================================================================
# FULL PREFIX-IDENTITY AUDIT
#
# For each frozen sampled event:
#
#   answer_global_index = len(layout.prefix_ids)
#
#   causal_position =
#       answer_global_index
#       - within_document_start
#       - 1
#
# Because target[p] is the answer token, logits[p] is the relevant output.
#
# The actual causal input visible to logits[p] is:
#
#   inputs[b, :p + 1]
#
# We perform THREE checks:
#
# A) TARGET POSITION CHECK
#
#    targets[b, p] == audited target_token_id
#
# B) WINDOW PREFIX CHECK
#
#    inputs[b, :p + 1]
#
#    must equal the corresponding audited prefix slice:
#
#    layout.prefix_ids[start : answer_global_index]
#
#    This proves our indexing into the sampled window is exact.
#
# C) FULL NATIVE CONTEXT CHECK
#
#    For the sampled causal context to be token-identical to native evaluation:
#
#       start == 0
#
#    AND:
#
#       inputs[b, :p + 1] == layout.prefix_ids
#
# We count this explicitly instead of assuming within_document_start == 0.
# =============================================================================

def audit_frozen_schedule_prefix_identity(
    store,
    records,
):

    header(
        "AUDITING FROZEN SCHEDULE CAUSAL PREFIX IDENTITY"
    )

    generator = (
        torch.Generator()
        .manual_seed(
            SCHEDULE_SEED
        )
    )

    digest = hashlib.sha256()

    total_events = 0

    target_position_matches = 0
    target_position_mismatches = 0

    window_prefix_matches = 0
    window_prefix_mismatches = 0

    full_native_prefix_matches = 0
    full_native_prefix_mismatches = 0

    start_counts = Counter()

    query_slot_counts = Counter()

    full_match_by_query_slot = Counter()
    mismatch_by_query_slot = Counter()

    causal_position_counts = Counter()

    mismatch_examples = []

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        (
            inputs,
            targets,
            _mask,
            meta,
        ) = sample_document_batch(
            store,
            BATCH_SIZE,
            CONTEXT_SIZE,
            generator,
            torch.device(
                "cpu"
            ),
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

        if (
            len(document_indices)
            != BATCH_SIZE
        ):

            raise RuntimeError(
                f"Step {step}: expected 32 document indices."
            )

        if (
            len(starts)
            != BATCH_SIZE
        ):

            raise RuntimeError(
                f"Step {step}: expected 32 starts."
            )

        digest.update(
            strict_json_dumps(
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

        for batch_index in range(
            BATCH_SIZE
        ):

            total_events += 1

            document_index = (
                document_indices[
                    batch_index
                ]
            )

            start = (
                starts[
                    batch_index
                ]
            )

            if not (
                0
                <= document_index
                < len(records)
            ):

                raise RuntimeError(
                    f"Step {step}, batch {batch_index}: "
                    f"invalid document index {document_index}."
                )

            row = records[
                document_index
            ]

            query_slot = int(
                row[
                    "query_slot"
                ]
            )

            query_slot_counts[
                query_slot
            ] += 1

            start_counts[
                start
            ] += 1

            audited_prefix = [
                int(x)
                for x in row[
                    "layout"
                ][
                    "prefix_ids"
                ]
            ]

            answer_global_index = len(
                audited_prefix
            )

            causal_position = (
                answer_global_index
                - start
                - 1
            )

            causal_position_counts[
                causal_position
            ] += 1

            if not (
                0
                <= causal_position
                < int(
                    targets.shape[1]
                )
            ):

                raise RuntimeError(
                    "\n"
                    "EXPECTED ANSWER POSITION IS OUTSIDE SAMPLED TARGET WINDOW.\n"
                    f"Step: {step}\n"
                    f"Batch index: {batch_index}\n"
                    f"Document index: {document_index}\n"
                    f"within_document_start: {start}\n"
                    f"answer_global_index: {answer_global_index}\n"
                    f"causal_position: {causal_position}\n"
                )

            # -----------------------------------------------------------------
            # A) Confirm the target at p is still exactly the audited answer.
            # -----------------------------------------------------------------

            expected_target_id = int(
                row[
                    "target_token_id"
                ]
            )

            actual_target_id = int(
                targets[
                    batch_index,
                    causal_position,
                ].item()
            )

            if (
                actual_target_id
                == expected_target_id
            ):

                target_position_matches += 1

            else:

                target_position_mismatches += 1

                if len(
                    mismatch_examples
                ) < 20:

                    mismatch_examples.append(
                        {
                            "type":
                                "target_position_mismatch",

                            "step":
                                step,

                            "batch_index":
                                batch_index,

                            "document_index":
                                document_index,

                            "within_document_start":
                                start,

                            "causal_position":
                                causal_position,

                            "expected_target_id":
                                expected_target_id,

                            "actual_target_id":
                                actual_target_id,
                        }
                    )

            # -----------------------------------------------------------------
            # B) Compare actual sampled causal prefix to corresponding slice
            #    of the audited native prefix.
            #
            # IMPORTANT:
            #
            # logits[p] has seen inputs through p inclusive.
            #
            # Therefore:
            #
            # inputs[b, :p + 1]
            #
            # is the actual sampled causal prefix.
            # -----------------------------------------------------------------

            sampled_prefix = [
                int(x)
                for x in inputs[
                    batch_index,
                    :causal_position + 1,
                ].tolist()
            ]

            expected_window_prefix = (
                audited_prefix[
                    start:
                    answer_global_index
                ]
            )

            window_match = (
                sampled_prefix
                == expected_window_prefix
            )

            if window_match:

                window_prefix_matches += 1

            else:

                window_prefix_mismatches += 1

                if len(
                    mismatch_examples
                ) < 20:

                    mismatch_examples.append(
                        {
                            "type":
                                "window_prefix_mismatch",

                            "step":
                                step,

                            "batch_index":
                                batch_index,

                            "document_index":
                                document_index,

                            "within_document_start":
                                start,

                            "causal_position":
                                causal_position,

                            "sampled_prefix_length":
                                len(
                                    sampled_prefix
                                ),

                            "expected_window_prefix_length":
                                len(
                                    expected_window_prefix
                                ),

                            "sampled_prefix_head":
                                sampled_prefix[
                                    :20
                                ],

                            "expected_prefix_head":
                                expected_window_prefix[
                                    :20
                                ],
                        }
                    )

            # -----------------------------------------------------------------
            # C) Does the margin logit see EXACTLY the same complete prefix
            #    used by native v0.9 evaluation?
            #
            # Full identity requires:
            #
            # start == 0
            #
            # AND:
            #
            # sampled_prefix == layout.prefix_ids
            # -----------------------------------------------------------------

            full_native_match = (
                start == 0
                and sampled_prefix
                == audited_prefix
            )

            if full_native_match:

                full_native_prefix_matches += 1

                full_match_by_query_slot[
                    query_slot
                ] += 1

            else:

                full_native_prefix_mismatches += 1

                mismatch_by_query_slot[
                    query_slot
                ] += 1

                if len(
                    mismatch_examples
                ) < 20:

                    mismatch_examples.append(
                        {
                            "type":
                                "full_native_prefix_mismatch",

                            "step":
                                step,

                            "batch_index":
                                batch_index,

                            "document_index":
                                document_index,

                            "query_slot":
                                query_slot,

                            "within_document_start":
                                start,

                            "causal_position":
                                causal_position,

                            "sampled_prefix_length":
                                len(
                                    sampled_prefix
                                ),

                            "audited_native_prefix_length":
                                len(
                                    audited_prefix
                                ),

                            "sampled_prefix_head":
                                sampled_prefix[
                                    :20
                                ],

                            "audited_prefix_head":
                                audited_prefix[
                                    :20
                                ],
                        }
                    )

    observed_sha = (
        digest.hexdigest()
    )

    if (
        total_events
        != EXPECTED_SAMPLED_EVENTS
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLED_EVENTS} events; "
            f"got {total_events}."
        )

    print(
        "Observed schedule SHA256:"
    )

    print(
        observed_sha
    )

    print()

    print(
        "Expected schedule SHA256:"
    )

    print(
        EXPECTED_SCHEDULE_SHA256
    )

    print()

    if (
        observed_sha
        != EXPECTED_SCHEDULE_SHA256
    ):

        raise RuntimeError(
            "Historical schedule SHA mismatch. "
            "STOP — this audit is not examining the Treatment #3 schedule."
        )

    print(
        "Historical schedule SHA: PASS"
    )

    print()

    print(
        f"Total sampled events: {total_events}"
    )

    print()

    print(
        "within_document_start distribution:"
    )

    for start, count in sorted(
        start_counts.items()
    ):

        print(
            f"  {start}: {count}"
        )

    print()

    print(
        "Target-position identity:"
    )

    print(
        f"  Matches:    {target_position_matches}"
    )

    print(
        f"  Mismatches: {target_position_mismatches}"
    )

    print()

    print(
        "Sampled-window prefix identity:"
    )

    print(
        f"  Matches:    {window_prefix_matches}"
    )

    print(
        f"  Mismatches: {window_prefix_mismatches}"
    )

    print()

    print(
        "FULL NATIVE EVALUATOR PREFIX IDENTITY:"
    )

    print(
        f"  Exact matches: {full_native_prefix_matches}"
    )

    print(
        f"  Mismatches:    {full_native_prefix_mismatches}"
    )

    print()

    print(
        "Full native matches by query slot:"
    )

    print(
        f"  slot 0: {full_match_by_query_slot[0]}"
    )

    print(
        f"  slot 1: {full_match_by_query_slot[1]}"
    )

    print()

    print(
        "Full native mismatches by query slot:"
    )

    print(
        f"  slot 0: {mismatch_by_query_slot[0]}"
    )

    print(
        f"  slot 1: {mismatch_by_query_slot[1]}"
    )

    print()

    print(
        "Causal-position range:"
    )

    print(
        f"  min: {min(causal_position_counts)}"
    )

    print(
        f"  max: {max(causal_position_counts)}"
    )

    print()

    # -------------------------------------------------------------------------
    # Hard interpretation.
    #
    # Target-position and window-prefix mismatches indicate an indexing/harness
    # error and should hard-fail the audit.
    #
    # Full native prefix mismatches do NOT get silently ignored. We report them
    # and classify the Treatment #3 train/eval context identity as failed.
    # -------------------------------------------------------------------------

    if (
        target_position_mismatches
        != 0
    ):

        raise RuntimeError(
            "TARGET-POSITION IDENTITY FAILED. "
            "Treatment #3 harness integrity is invalid."
        )

    if (
        window_prefix_mismatches
        != 0
    ):

        raise RuntimeError(
            "SAMPLED-WINDOW PREFIX IDENTITY FAILED. "
            "Treatment #3 harness integrity is invalid."
        )

    full_identity_pass = (
        full_native_prefix_matches
        == EXPECTED_SAMPLED_EVENTS
        and full_native_prefix_mismatches
        == 0
    )

    if full_identity_pass:

        classification = (
            "PASS — ALL 32,000 MARGIN EVENTS USED "
            "TOKEN-IDENTICAL NATIVE EVALUATOR PREFIXES"
        )

    else:

        classification = (
            "FAIL — TREATMENT #3 MARGIN CONTEXT WAS NOT "
            "TOKEN-IDENTICAL TO NATIVE EVALUATION FOR ALL EVENTS"
        )

    return {
        "schedule_sha256":
            observed_sha,

        "total_events":
            total_events,

        "within_document_start_counts": {
            str(start):
                count

            for start, count
            in sorted(
                start_counts.items()
            )
        },

        "target_position": {
            "matches":
                target_position_matches,

            "mismatches":
                target_position_mismatches,
        },

        "sampled_window_prefix": {
            "matches":
                window_prefix_matches,

            "mismatches":
                window_prefix_mismatches,
        },

        "full_native_prefix_identity": {
            "matches":
                full_native_prefix_matches,

            "mismatches":
                full_native_prefix_mismatches,

            "pass":
                full_identity_pass,

            "matches_by_query_slot": {
                "0":
                    full_match_by_query_slot[
                        0
                    ],

                "1":
                    full_match_by_query_slot[
                        1
                    ],
            },

            "mismatches_by_query_slot": {
                "0":
                    mismatch_by_query_slot[
                        0
                    ],

                "1":
                    mismatch_by_query_slot[
                        1
                    ],
            },
        },

        "causal_position_counts": {
            str(position):
                count

            for position, count
            in sorted(
                causal_position_counts.items()
            )
        },

        "mismatch_examples":
            mismatch_examples,

        "classification":
            classification,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    safety_check()

    (
        tokenizer,
        records,
    ) = reconstruct_training_records()

    verify_native_prefixes(
        tokenizer,
        records,
    )

    store = build_store(
        tokenizer,
        records,
    )

    result = (
        audit_frozen_schedule_prefix_identity(
            store,
            records,
        )
    )

    header(
        "TREATMENT #3 PREFIX-IDENTITY AUDIT VERDICT"
    )

    print(
        "Historical schedule SHA: PASS"
    )

    print()

    print(
        "Target-position matches:"
    )

    print(
        f"{result['target_position']['matches']}/"
        f"{result['total_events']}"
    )

    print()

    print(
        "Sampled-window prefix matches:"
    )

    print(
        f"{result['sampled_window_prefix']['matches']}/"
        f"{result['total_events']}"
    )

    print()

    print(
        "Full native evaluator prefix matches:"
    )

    print(
        f"{result['full_native_prefix_identity']['matches']}/"
        f"{result['total_events']}"
    )

    print()

    print(
        "Full native evaluator prefix mismatches:"
    )

    print(
        result[
            "full_native_prefix_identity"
        ][
            "mismatches"
        ]
    )

    print()

    print(
        "CLASSIFICATION:"
    )

    print(
        result[
            "classification"
        ]
    )

    print()

    print(
        "MODEL BUILT: NO"
    )

    print(
        "OPTIMIZER BUILT: NO"
    )

    print(
        "GRADIENTS CALCULATED: NO"
    )

    print(
        "BACKWARD PASSES: ZERO"
    )

    print(
        "OPTIMIZER STEPS: ZERO"
    )

    print(
        "POSITIVE-CONTROL EVALUATION: NO"
    )

    print(
        "SEALED AXES LOADED/EVALUATED: NO"
    )

    payload = {
        "experiment":
            "DaveLM v0.9 Treatment #3 causal prefix identity audit",

        "read_only":
            True,

        "expected_schedule_sha256":
            EXPECTED_SCHEDULE_SHA256,

        "result":
            result,

        "model_built":
            False,

        "optimizer_built":
            False,

        "gradients_calculated":
            False,

        "backward_passes":
            0,

        "optimizer_steps":
            0,

        "positive_control_evaluation":
            False,

        "sealed_axes_loaded":
            False,

        "sealed_evaluation":
            False,
    }

    write_json(
        AUDIT_PATH,
        payload,
    )

    print()

    print(
        "AUDIT ARTIFACT:"
    )

    print(
        AUDIT_PATH
    )

    print()

    print(
        "NO TRAINING OCCURRED."
    )


if __name__ == "__main__":
    main()