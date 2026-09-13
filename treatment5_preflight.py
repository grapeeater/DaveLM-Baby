from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


# =============================================================================
# DaveLM v0.9 — Treatment #5
# FINAL FULL-DOCUMENT PREFLIGHT
#
# Paired Counterfactual Query Binding
# =============================================================================
#
# PURPOSE
# -------
#
# Construct and freeze the COMPLETE Treatment #5 training schedule before
# training.
#
# Every native two-mapping training record becomes one base situation.
#
# From that SAME base situation we synthesize:
#
#   Twin A:
#       Query = mapping slot 0 source
#       Answer = mapping slot 0 target
#
#   Twin B:
#       Query = mapping slot 1 source
#       Answer = mapping slot 1 target
#
# The twins must preserve:
#
#   - same mappings
#   - same mapping order
#   - same filler/context
#   - same geometry/layout
#   - same candidate answer pair
#   - same suffix
#
# The ONLY model-visible differences allowed between twins are:
#
#   1. query identity token
#   2. answer identity token
#
#
# TRAINING CONTRACT
# -----------------
#
# Treatment #5 keeps the Treatment #4 objective:
#
#   Every supervised NON-ANSWER position:
#       ordinary full-vocabulary causal CE
#
#   Eligible ANSWER position only:
#
#       membership_loss =
#           logsumexp(full_vocab_logits)
#           - logsumexp([correct_logit, distractor_logit])
#
#       selector_loss =
#           relu(
#               0.5
#               - (correct_logit - distractor_logit)
#           )
#
#       replacement_loss =
#           membership_loss + selector_loss
#
#   Original answer CE is NOT retained.
#
#
# IMPORTANT
# ---------
#
# We intentionally DO NOT force Treatment #5 to reproduce Treatment #4's
# exact total number of non-answer CE tokens.
#
# Why?
#
# Treatment #5 changes the sampling distribution to complete matched twin
# pairs. Native document lengths vary. Therefore the correct experiment is:
#
#   - same number of steps
#   - same batch size
#   - same number of answer substitutions
#   - same loss semantics
#   - same starting checkpoint
#
# while allowing the paired schedule's natural non-answer-token total to be
# reported rather than manipulated post-hoc.
#
#
# SAFETY
# ------
#
# This script:
#
#   - DOES NOT construct Baby
#   - DOES NOT load a checkpoint into a model
#   - DOES NOT create an optimizer
#   - DOES NOT call backward()
#   - DOES NOT train
#   - DOES NOT evaluate positive controls
#   - DOES NOT open sealed evaluation pools
#   - DOES NOT modify C:\DaveLM-v0.9
#
# =============================================================================


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

SOURCE_ROOT = Path(
    r"C:\DaveLM-v0.9"
)

OUTPUT_ROOT = Path(
    r"C:\DaveLM-CADAVER\treatment5_counterfactual_pairs_seed8382"
)

PAIR_POOL_PATH = (
    OUTPUT_ROOT
    / "treatment5_full_document_pair_pool.json"
)

SCHEDULE_PATH = (
    OUTPUT_ROOT
    / "treatment5_full_document_frozen_schedule.json"
)

RESULT_PATH = (
    OUTPUT_ROOT
    / "treatment5_full_document_preflight_result.json"
)


# -----------------------------------------------------------------------------
# Experiment constants
# -----------------------------------------------------------------------------

TRAIN_SEED = 8380

TREATMENT5_SCHEDULE_SEED = 8382

MAX_STEPS = 1000

BATCH_SIZE = 32

PAIRS_PER_BATCH = 16

CONTEXT_SIZE = 256


# -----------------------------------------------------------------------------
# Expected native corpus facts
# -----------------------------------------------------------------------------

EXPECTED_NATIVE_RECORDS = 1536

EXPECTED_NATIVE_SLOT_COUNT = 768

EXPECTED_PAIR_POOL_SIZE = 1536


# -----------------------------------------------------------------------------
# Frozen Treatment #5 schedule facts
# -----------------------------------------------------------------------------

EXPECTED_PAIR_PRESENTATIONS = (
    MAX_STEPS
    * PAIRS_PER_BATCH
)

EXPECTED_SCHEDULED_EXAMPLES = (
    MAX_STEPS
    * BATCH_SIZE
)

EXPECTED_ANSWER_SUBSTITUTIONS = (
    EXPECTED_SCHEDULED_EXAMPLES
)

EXPECTED_SLOT_PRESENTATIONS = (
    EXPECTED_SCHEDULED_EXAMPLES
    // 2
)


# -----------------------------------------------------------------------------
# Historical Treatment #4 count — REFERENCE ONLY
#
# Do NOT fail Treatment #5 merely because paired sampling produces a slightly
# different non-answer token count.
# -----------------------------------------------------------------------------

T4_REFERENCE_SUPERVISED_TOKENS = 6_176_000

T4_REFERENCE_ANSWER_SUBSTITUTIONS = 32_000

T4_REFERENCE_NONANSWER_TOKENS = 6_144_000


# -----------------------------------------------------------------------------
# Frozen starting checkpoint
# -----------------------------------------------------------------------------

START_CHECKPOINT = Path(
    r"C:\DaveLM-v0.9"
    r"\experiments"
    r"\minimal_contextual_binding"
    r"\checkpoints"
    r"\treatment_one_mapping"
    r"\seed_8380"
    r"\latest.pt"
)

EXPECTED_START_CHECKPOINT_SHA256 = (
    "345984c52a06db5f988aaf4cd47963ee"
    "a0e9d77489cee94dbb10816af2f5443e"
)


# =============================================================================
# READ-ONLY IMPORT OF NATIVE SOURCE HELPERS
# =============================================================================

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(SOURCE_ROOT),
    )


import experiments.two_mapping_contextual_binding.run as original_run


# =============================================================================
# BASIC HELPERS
# =============================================================================

def header(
    title: str,
) -> None:

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


def canonical_bytes(
    obj: Any,
) -> bytes:

    return json.dumps(
        obj,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=True,
        allow_nan=False,
    ).encode(
        "utf-8"
    )


def sha256_bytes(
    data: bytes,
) -> str:

    return hashlib.sha256(
        data
    ).hexdigest()


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


# =============================================================================
# TOKENIZATION HELPERS
# =============================================================================

def raw_text_token_ids(
    tokenizer: Tokenizer,
    text: str,
) -> list[int]:

    encoding = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    return [
        int(token_id)
        for token_id
        in encoding.ids
    ]


def model_visible_token_ids(
    tokenizer: Tokenizer,
    text: str,
    bos_token_id: int,
) -> list[int]:

    return [
        int(
            bos_token_id
        )
    ] + raw_text_token_ids(
        tokenizer,
        text,
    )


# =============================================================================
# STRUCTURED QUERY / ANSWER LINE HELPERS
# =============================================================================

def find_exactly_one_line(
    text: str,
    prefix: str,
) -> tuple[int, str]:

    lines = text.splitlines(
        keepends=True
    )

    matches: list[
        tuple[
            int,
            str,
        ]
    ] = []


    for line_index, line in enumerate(
        lines
    ):

        stripped = line.rstrip(
            "\r\n"
        )

        if stripped.startswith(
            prefix
        ):

            value = stripped[
                len(prefix):
            ].strip()

            matches.append(
                (
                    line_index,
                    value,
                )
            )


    if len(matches) != 1:

        raise RuntimeError(
            f"Expected exactly one line "
            f"starting with {prefix!r}; "
            f"found {len(matches)}."
        )


    return matches[
        0
    ]


def replace_exactly_one_line_value(
    text: str,
    prefix: str,
    new_value: str,
) -> str:

    lines = text.splitlines(
        keepends=True
    )

    line_index, _ = find_exactly_one_line(
        text,
        prefix,
    )

    original_line = lines[
        line_index
    ]


    if original_line.endswith(
        "\r\n"
    ):

        newline = "\r\n"

    elif original_line.endswith(
        "\n"
    ):

        newline = "\n"

    elif original_line.endswith(
        "\r"
    ):

        newline = "\r"

    else:

        newline = ""


    lines[
        line_index
    ] = (
        f"{prefix} "
        f"{new_value}"
        f"{newline}"
    )


    return "".join(
        lines
    )


def normalize_query_and_answer(
    text: str,
) -> str:

    normalized = (
        replace_exactly_one_line_value(
            text,
            "Query:",
            "<QUERY>",
        )
    )

    normalized = (
        replace_exactly_one_line_value(
            normalized,
            "Answer:",
            "<ANSWER>",
        )
    )

    return normalized


# =============================================================================
# NATIVE CORPUS RECONSTRUCTION
# =============================================================================

def reconstruct_native_corpus() -> tuple[
    Tokenizer,
    list[dict[str, Any]],
    int,
]:

    header(
        "RECONSTRUCTING NATIVE TWO-MAPPING CORPUS"
    )


    tokenizer = Tokenizer.from_file(
        str(
            original_run.config.TOKENIZER_PATH
        )
    )


    groups = original_run._identity_pool(
        tokenizer
    )


    identities = groups[
        "identity"
    ]

    filler = groups[
        "filler"
    ]


    tables = original_run._layout_tables(
        tokenizer,
        groups,
    )


    records = original_run._training_records(
        tokenizer,
        identities,
        filler,
        tables,
    )


    if len(
        records
    ) != EXPECTED_NATIVE_RECORDS:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_NATIVE_RECORDS} "
            f"native records; got "
            f"{len(records)}."
        )


    bos_token_id = int(
        original_run.required_token_id(
            tokenizer,
            "<bos>",
        )
    )


    print(
        f"Native records: "
        f"{len(records)}"
    )

    print(
        f"Native BOS token ID: "
        f"{bos_token_id}"
    )


    return (
        tokenizer,
        records,
        bos_token_id,
    )


# =============================================================================
# MAPPING HELPERS
# =============================================================================

def mappings_by_slot(
    row: dict[str, Any],
) -> dict[int, dict[str, Any]]:

    mappings = row.get(
        "mappings"
    )


    if not isinstance(
        mappings,
        list,
    ):

        raise RuntimeError(
            "Record mappings field "
            "is not a list."
        )


    if len(
        mappings
    ) != 2:

        raise RuntimeError(
            f"Expected exactly two mappings; "
            f"found {len(mappings)}."
        )


    result: dict[
        int,
        dict[str, Any],
    ] = {}


    for mapping in mappings:

        if not isinstance(
            mapping,
            dict,
        ):

            raise RuntimeError(
                "Mapping is not a dictionary."
            )


        slot = int(
            mapping[
                "slot"
            ]
        )


        if slot in result:

            raise RuntimeError(
                f"Duplicate mapping slot "
                f"{slot}."
            )


        result[
            slot
        ] = mapping


    if set(
        result.keys()
    ) != {
        0,
        1,
    }:

        raise RuntimeError(
            f"Mapping slots were "
            f"{sorted(result.keys())}; "
            f"expected [0, 1]."
        )


    return result


def recover_slot_target_token_ids(
    row: dict[str, Any],
) -> tuple[int, int]:

    native_query_slot = int(
        row[
            "query_slot"
        ]
    )

    native_target = int(
        row[
            "target_token_id"
        ]
    )

    native_distractor = int(
        row[
            "distractor_token_id"
        ]
    )


    if native_target == native_distractor:

        raise RuntimeError(
            "Native target token equals "
            "native distractor token."
        )


    if native_query_slot == 0:

        slot0_target = (
            native_target
        )

        slot1_target = (
            native_distractor
        )


    elif native_query_slot == 1:

        slot0_target = (
            native_distractor
        )

        slot1_target = (
            native_target
        )


    else:

        raise RuntimeError(
            f"Invalid native query slot "
            f"{native_query_slot}."
        )


    if slot0_target == slot1_target:

        raise RuntimeError(
            "Recovered slot targets "
            "are identical."
        )


    return (
        int(
            slot0_target
        ),
        int(
            slot1_target
        ),
    )


# =============================================================================
# REQUIRED NATIVE FIELDS
# =============================================================================

REQUIRED_NATIVE_FIELDS = {
    "id",
    "text",
    "prompt",
    "query_slot",
    "mappings",
    "source_identity",
    "source_identity_index",
    "source_token_id",
    "target_identity",
    "target_identity_index",
    "target_token_id",
    "distractor_token_id",
    "answer_token_ids",
    "expected_answer",
    "document_token_count",
    "layout",
}


# =============================================================================
# NATIVE FULL-DOCUMENT AUDIT
# =============================================================================

def audit_native_documents(
    tokenizer: Tokenizer,
    records: list[dict[str, Any]],
    bos_token_id: int,
) -> dict[str, Any]:

    header(
        "AUDITING NATIVE COMPLETE DOCUMENTS"
    )


    slot_counts = Counter()

    prefix_lengths = Counter()

    raw_document_lengths = Counter()

    model_visible_lengths = Counter()

    native_prefix_matches = 0

    native_document_count_matches = 0

    native_answer_metadata_matches = 0


    for record_index, row in enumerate(
        records
    ):

        missing_fields = (
            REQUIRED_NATIVE_FIELDS
            - set(
                row.keys()
            )
        )


        if missing_fields:

            raise RuntimeError(
                f"Record {record_index}: "
                f"missing required fields "
                f"{sorted(missing_fields)}."
            )


        mappings = mappings_by_slot(
            row
        )


        query_slot = int(
            row[
                "query_slot"
            ]
        )


        if query_slot not in (
            0,
            1,
        ):

            raise RuntimeError(
                f"Record {record_index}: "
                f"invalid query_slot="
                f"{query_slot}."
            )


        slot_counts[
            query_slot
        ] += 1


        queried_mapping = mappings[
            query_slot
        ]


        # ---------------------------------------------------------------------
        # Native prompt Query: line
        # ---------------------------------------------------------------------

        _, prompt_query_identity = (
            find_exactly_one_line(
                str(
                    row[
                        "prompt"
                    ]
                ),
                "Query:",
            )
        )


        if prompt_query_identity != str(
            queried_mapping[
                "source"
            ]
        ):

            raise RuntimeError(
                f"Record {record_index}: "
                f"prompt Query identity "
                f"does not match native "
                f"queried mapping."
            )


        # ---------------------------------------------------------------------
        # Native complete-document Query: line
        # ---------------------------------------------------------------------

        _, full_query_identity = (
            find_exactly_one_line(
                str(
                    row[
                        "text"
                    ]
                ),
                "Query:",
            )
        )


        if full_query_identity != (
            prompt_query_identity
        ):

            raise RuntimeError(
                f"Record {record_index}: "
                f"full-document Query identity "
                f"differs from prompt Query."
            )


        # ---------------------------------------------------------------------
        # Native complete-document Answer: line
        # ---------------------------------------------------------------------

        _, full_answer_identity = (
            find_exactly_one_line(
                str(
                    row[
                        "text"
                    ]
                ),
                "Answer:",
            )
        )


        if full_answer_identity != str(
            row[
                "expected_answer"
            ]
        ):

            raise RuntimeError(
                f"Record {record_index}: "
                f"full-document Answer identity "
                f"does not match expected_answer."
            )


        if full_answer_identity != str(
            queried_mapping[
                "target"
            ]
        ):

            raise RuntimeError(
                f"Record {record_index}: "
                f"Answer identity does not "
                f"match queried mapping target."
            )


        # ---------------------------------------------------------------------
        # Native metadata
        # ---------------------------------------------------------------------

        if str(
            row[
                "source_identity"
            ]
        ) != str(
            queried_mapping[
                "source"
            ]
        ):

            raise RuntimeError(
                f"Record {record_index}: "
                f"source_identity mismatch."
            )


        if int(
            row[
                "source_identity_index"
            ]
        ) != int(
            queried_mapping[
                "source_index"
            ]
        ):

            raise RuntimeError(
                f"Record {record_index}: "
                f"source_identity_index mismatch."
            )


        if str(
            row[
                "target_identity"
            ]
        ) != str(
            queried_mapping[
                "target"
            ]
        ):

            raise RuntimeError(
                f"Record {record_index}: "
                f"target_identity mismatch."
            )


        if int(
            row[
                "target_identity_index"
            ]
        ) != int(
            queried_mapping[
                "target_index"
            ]
        ):

            raise RuntimeError(
                f"Record {record_index}: "
                f"target_identity_index mismatch."
            )


        native_target_token_id = int(
            row[
                "target_token_id"
            ]
        )

        native_distractor_token_id = int(
            row[
                "distractor_token_id"
            ]
        )


        if (
            native_target_token_id
            == native_distractor_token_id
        ):

            raise RuntimeError(
                f"Record {record_index}: "
                f"target == distractor."
            )


        answer_token_ids = [
            int(token_id)
            for token_id
            in row[
                "answer_token_ids"
            ]
        ]


        if answer_token_ids != [
            native_target_token_id
        ]:

            raise RuntimeError(
                f"Record {record_index}: "
                f"answer_token_ids does not "
                f"equal [target_token_id]."
            )


        native_answer_metadata_matches += 1


        # ---------------------------------------------------------------------
        # Reconstruct model-visible prompt prefix.
        # ---------------------------------------------------------------------

        reconstructed_prefix_ids = (
            model_visible_token_ids(
                tokenizer,
                str(
                    row[
                        "prompt"
                    ]
                ),
                bos_token_id,
            )
        )


        native_layout_prefix_ids = [
            int(token_id)
            for token_id
            in row[
                "layout"
            ][
                "prefix_ids"
            ]
        ]


        if (
            reconstructed_prefix_ids
            != native_layout_prefix_ids
        ):

            raise RuntimeError(
                f"Record {record_index}: "
                f"reconstructed prompt prefix "
                f"does not equal "
                f"layout['prefix_ids']."
            )


        native_prefix_matches += 1


        prefix_lengths[
            len(
                reconstructed_prefix_ids
            )
        ] += 1


        # ---------------------------------------------------------------------
        # Complete native document.
        #
        # IMPORTANT:
        #
        # native row["document_token_count"] excludes BOS.
        #
        # Therefore:
        #
        #     raw document tokens
        #       = document_token_count
        #
        #     model-visible sequence
        #       = BOS + raw document tokens
        #
        #     causal supervised positions
        #       = len(model-visible) - 1
        #       = document_token_count
        #
        # ---------------------------------------------------------------------

        raw_document_ids = (
            raw_text_token_ids(
                tokenizer,
                str(
                    row[
                        "text"
                    ]
                ),
            )
        )


        model_visible_document_ids = [
            int(
                bos_token_id
            )
        ] + raw_document_ids


        declared_document_token_count = int(
            row[
                "document_token_count"
            ]
        )


        actual_raw_document_token_count = len(
            raw_document_ids
        )


        actual_supervised_token_count = (
            len(
                model_visible_document_ids
            )
            - 1
        )


        if (
            declared_document_token_count
            != actual_raw_document_token_count
        ):

            raise RuntimeError(
                f"Record {record_index}: "
                f"document_token_count="
                f"{declared_document_token_count}, "
                f"but tokenizer produced "
                f"{actual_raw_document_token_count} "
                f"raw document tokens "
                f"excluding BOS."
            )


        if (
            declared_document_token_count
            != actual_supervised_token_count
        ):

            raise RuntimeError(
                f"Record {record_index}: "
                f"document_token_count="
                f"{declared_document_token_count}, "
                f"but causal supervised count="
                f"{actual_supervised_token_count}."
            )


        native_document_count_matches += 1


        raw_document_lengths[
            actual_raw_document_token_count
        ] += 1


        model_visible_lengths[
            len(
                model_visible_document_ids
            )
        ] += 1


        if len(
            model_visible_document_ids
        ) > (
            CONTEXT_SIZE + 1
        ):

            raise RuntimeError(
                f"Record {record_index}: "
                f"model-visible document length "
                f"{len(model_visible_document_ids)} "
                f"exceeds allowed "
                f"{CONTEXT_SIZE + 1} "
                f"(BOS + {CONTEXT_SIZE} "
                f"document tokens)."
            )


        # ---------------------------------------------------------------------
        # Recover both candidate targets from native authoritative IDs.
        # ---------------------------------------------------------------------

        slot0_target_id, slot1_target_id = (
            recover_slot_target_token_ids(
                row
            )
        )


        if (
            slot0_target_id
            == slot1_target_id
        ):

            raise RuntimeError(
                f"Record {record_index}: "
                f"slot targets identical."
            )


    # -------------------------------------------------------------------------
    # Corpus-level invariants
    # -------------------------------------------------------------------------

    if (
        slot_counts[
            0
        ]
        != EXPECTED_NATIVE_SLOT_COUNT
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_NATIVE_SLOT_COUNT} "
            f"native slot-0 records; "
            f"got {slot_counts[0]}."
        )


    if (
        slot_counts[
            1
        ]
        != EXPECTED_NATIVE_SLOT_COUNT
    ):

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_NATIVE_SLOT_COUNT} "
            f"native slot-1 records; "
            f"got {slot_counts[1]}."
        )


    if (
        native_prefix_matches
        != EXPECTED_NATIVE_RECORDS
    ):

        raise RuntimeError(
            "Native prefix audit did not "
            "cover every record."
        )


    if (
        native_document_count_matches
        != EXPECTED_NATIVE_RECORDS
    ):

        raise RuntimeError(
            "Native document-count audit "
            "did not cover every record."
        )


    if (
        native_answer_metadata_matches
        != EXPECTED_NATIVE_RECORDS
    ):

        raise RuntimeError(
            "Native answer metadata audit "
            "did not cover every record."
        )


    print(
        f"Native slot-0 records: "
        f"{slot_counts[0]}"
    )

    print(
        f"Native slot-1 records: "
        f"{slot_counts[1]}"
    )

    print(
        f"Exact native prefix matches: "
        f"{native_prefix_matches}/"
        f"{EXPECTED_NATIVE_RECORDS}"
    )

    print(
        f"Exact native document-count matches: "
        f"{native_document_count_matches}/"
        f"{EXPECTED_NATIVE_RECORDS}"
    )

    print(
        f"Exact native answer metadata matches: "
        f"{native_answer_metadata_matches}/"
        f"{EXPECTED_NATIVE_RECORDS}"
    )

    print(
        f"Distinct prompt-prefix lengths: "
        f"{len(prefix_lengths)}"
    )

    print(
        f"Distinct raw document lengths: "
        f"{len(raw_document_lengths)}"
    )

    print(
        f"Raw document length distribution: "
        f"{dict(sorted(raw_document_lengths.items()))}"
    )


    return {
        "slot_0_records": int(
            slot_counts[
                0
            ]
        ),

        "slot_1_records": int(
            slot_counts[
                1
            ]
        ),

        "native_prefix_matches": int(
            native_prefix_matches
        ),

        "native_document_count_matches": int(
            native_document_count_matches
        ),

        "native_answer_metadata_matches": int(
            native_answer_metadata_matches
        ),

        "distinct_prefix_lengths": int(
            len(
                prefix_lengths
            )
        ),

        "raw_document_length_distribution": {
            str(length): int(
                count
            )
            for length, count
            in sorted(
                raw_document_lengths.items()
            )
        },

        "model_visible_length_distribution": {
            str(length): int(
                count
            )
            for length, count
            in sorted(
                model_visible_lengths.items()
            )
        },
    }


# =============================================================================
# BUILD ONE SAME-RECORD COMPLETE-DOCUMENT TWIN PAIR
# =============================================================================

def build_counterfactual_pair(
    tokenizer: Tokenizer,
    row: dict[str, Any],
    record_index: int,
    bos_token_id: int,
) -> dict[str, Any]:

    mappings = mappings_by_slot(
        row
    )


    mapping0 = mappings[
        0
    ]

    mapping1 = mappings[
        1
    ]


    source0 = str(
        mapping0[
            "source"
        ]
    )

    source1 = str(
        mapping1[
            "source"
        ]
    )


    target_text0 = str(
        mapping0[
            "target"
        ]
    )

    target_text1 = str(
        mapping1[
            "target"
        ]
    )


    if source0 == source1:

        raise RuntimeError(
            f"Record {record_index}: "
            f"slot sources identical."
        )


    if target_text0 == target_text1:

        raise RuntimeError(
            f"Record {record_index}: "
            f"slot target identities identical."
        )


    (
        slot0_target_token_id,
        slot1_target_token_id,
    ) = recover_slot_target_token_ids(
        row
    )


    # -------------------------------------------------------------------------
    # Native base strings
    # -------------------------------------------------------------------------

    base_prompt = str(
        row[
            "prompt"
        ]
    )

    base_text = str(
        row[
            "text"
        ]
    )


    # -------------------------------------------------------------------------
    # Build Twin A prefix / complete document.
    # -------------------------------------------------------------------------

    twin_a_prompt = (
        replace_exactly_one_line_value(
            base_prompt,
            "Query:",
            source0,
        )
    )


    twin_a_text = (
        replace_exactly_one_line_value(
            base_text,
            "Query:",
            source0,
        )
    )


    twin_a_text = (
        replace_exactly_one_line_value(
            twin_a_text,
            "Answer:",
            target_text0,
        )
    )


    # -------------------------------------------------------------------------
    # Build Twin B prefix / complete document.
    # -------------------------------------------------------------------------

    twin_b_prompt = (
        replace_exactly_one_line_value(
            base_prompt,
            "Query:",
            source1,
        )
    )


    twin_b_text = (
        replace_exactly_one_line_value(
            base_text,
            "Query:",
            source1,
        )
    )


    twin_b_text = (
        replace_exactly_one_line_value(
            twin_b_text,
            "Answer:",
            target_text1,
        )
    )


    # -------------------------------------------------------------------------
    # Text-level twin contract.
    # -------------------------------------------------------------------------

    normalized_a = (
        normalize_query_and_answer(
            twin_a_text
        )
    )

    normalized_b = (
        normalize_query_and_answer(
            twin_b_text
        )
    )


    if normalized_a != normalized_b:

        raise RuntimeError(
            f"Record {record_index}: "
            f"counterfactual full documents "
            f"differ outside the Query and "
            f"Answer lines."
        )


    # -------------------------------------------------------------------------
    # Model-visible prefixes.
    # -------------------------------------------------------------------------

    prefix_a = model_visible_token_ids(
        tokenizer,
        twin_a_prompt,
        bos_token_id,
    )

    prefix_b = model_visible_token_ids(
        tokenizer,
        twin_b_prompt,
        bos_token_id,
    )


    if len(
        prefix_a
    ) != len(
        prefix_b
    ):

        raise RuntimeError(
            f"Record {record_index}: "
            f"Twin prefix lengths differ: "
            f"{len(prefix_a)} vs "
            f"{len(prefix_b)}."
        )


    prefix_difference_positions = [
        position
        for position, (
            token_a,
            token_b,
        )
        in enumerate(
            zip(
                prefix_a,
                prefix_b,
            )
        )
        if token_a != token_b
    ]


    if len(
        prefix_difference_positions
    ) != 1:

        raise RuntimeError(
            f"Record {record_index}: "
            f"expected exactly ONE "
            f"model-visible query token "
            f"difference between prefixes; "
            f"found "
            f"{len(prefix_difference_positions)} "
            f"at positions "
            f"{prefix_difference_positions}."
        )


    query_difference_position = int(
        prefix_difference_positions[
            0
        ]
    )


    # -------------------------------------------------------------------------
    # Complete model-visible documents.
    # -------------------------------------------------------------------------

    document_a = model_visible_token_ids(
        tokenizer,
        twin_a_text,
        bos_token_id,
    )

    document_b = model_visible_token_ids(
        tokenizer,
        twin_b_text,
        bos_token_id,
    )


    if len(
        document_a
    ) != len(
        document_b
    ):

        raise RuntimeError(
            f"Record {record_index}: "
            f"Twin complete-document lengths "
            f"differ: "
            f"{len(document_a)} vs "
            f"{len(document_b)}."
        )


    # BOS + <= CONTEXT_SIZE raw document tokens.
    if len(
        document_a
    ) > (
        CONTEXT_SIZE + 1
    ):

        raise RuntimeError(
            f"Record {record_index}: "
            f"Twin document contains "
            f"{len(document_a)} "
            f"model-visible IDs, exceeding "
            f"BOS + context limit "
            f"{CONTEXT_SIZE + 1}."
        )


    # -------------------------------------------------------------------------
    # CRITICAL PREFIX-CONTIGUITY AUDIT
    #
    # We explicitly prove that tokenizing the full synthetic document begins
    # with exactly the separately audited synthetic prompt prefix.
    #
    # This avoids silently assuming tokenization cannot change at the
    # prompt/answer boundary.
    # -------------------------------------------------------------------------

    if (
        document_a[
            :len(prefix_a)
        ]
        != prefix_a
    ):

        raise RuntimeError(
            f"Record {record_index}: "
            f"Twin A full document does not "
            f"begin with exact Twin A prefix."
        )


    if (
        document_b[
            :len(prefix_b)
        ]
        != prefix_b
    ):

        raise RuntimeError(
            f"Record {record_index}: "
            f"Twin B full document does not "
            f"begin with exact Twin B prefix."
        )


    # -------------------------------------------------------------------------
    # Answer position.
    #
    # The answer token is the FIRST token after the prompt prefix.
    # -------------------------------------------------------------------------

    answer_token_index = len(
        prefix_a
    )


    if (
        answer_token_index
        >= len(document_a)
    ):

        raise RuntimeError(
            f"Record {record_index}: "
            f"Twin A answer token index "
            f"is outside document."
        )


    if (
        answer_token_index
        >= len(document_b)
    ):

        raise RuntimeError(
            f"Record {record_index}: "
            f"Twin B answer token index "
            f"is outside document."
        )


    observed_answer_a = int(
        document_a[
            answer_token_index
        ]
    )

    observed_answer_b = int(
        document_b[
            answer_token_index
        ]
    )


    if (
        observed_answer_a
        != slot0_target_token_id
    ):

        raise RuntimeError(
            f"Record {record_index}: "
            f"Twin A answer token was "
            f"{observed_answer_a}, expected "
            f"slot-0 target "
            f"{slot0_target_token_id}."
        )


    if (
        observed_answer_b
        != slot1_target_token_id
    ):

        raise RuntimeError(
            f"Record {record_index}: "
            f"Twin B answer token was "
            f"{observed_answer_b}, expected "
            f"slot-1 target "
            f"{slot1_target_token_id}."
        )


    # -------------------------------------------------------------------------
    # Causal logit position.
    #
    # Token document[N] is predicted from logits[N - 1].
    # -------------------------------------------------------------------------

    answer_causal_position = (
        answer_token_index
        - 1
    )


    if answer_causal_position < 0:

        raise RuntimeError(
            f"Record {record_index}: "
            f"invalid answer causal position."
        )


    # -------------------------------------------------------------------------
    # Complete-document token difference audit.
    #
    # Exactly:
    #
    #   query token
    #   answer token
    #
    # may differ.
    # -------------------------------------------------------------------------

    document_difference_positions = [
        position
        for position, (
            token_a,
            token_b,
        )
        in enumerate(
            zip(
                document_a,
                document_b,
            )
        )
        if token_a != token_b
    ]


    expected_difference_positions = sorted(
        [
            query_difference_position,
            answer_token_index,
        ]
    )


    if (
        document_difference_positions
        != expected_difference_positions
    ):

        raise RuntimeError(
            f"Record {record_index}: "
            f"unexpected full-document token "
            f"differences. Expected exactly "
            f"{expected_difference_positions}; "
            f"got "
            f"{document_difference_positions}."
        )


    # -------------------------------------------------------------------------
    # Native record must equal exactly one synthesized twin.
    # -------------------------------------------------------------------------

    native_document = (
        model_visible_token_ids(
            tokenizer,
            base_text,
            bos_token_id,
        )
    )


    native_prefix = [
        int(token_id)
        for token_id
        in row[
            "layout"
        ][
            "prefix_ids"
        ]
    ]


    native_query_slot = int(
        row[
            "query_slot"
        ]
    )


    if native_query_slot == 0:

        if native_prefix != prefix_a:

            raise RuntimeError(
                f"Record {record_index}: "
                f"native slot-0 prefix "
                f"does not equal Twin A prefix."
            )

        if native_document != document_a:

            raise RuntimeError(
                f"Record {record_index}: "
                f"native slot-0 document "
                f"does not equal Twin A document."
            )


    elif native_query_slot == 1:

        if native_prefix != prefix_b:

            raise RuntimeError(
                f"Record {record_index}: "
                f"native slot-1 prefix "
                f"does not equal Twin B prefix."
            )

        if native_document != document_b:

            raise RuntimeError(
                f"Record {record_index}: "
                f"native slot-1 document "
                f"does not equal Twin B document."
            )


    else:

        raise RuntimeError(
            f"Record {record_index}: "
            f"invalid native query slot."
        )


    # -------------------------------------------------------------------------
    # Supervision counts.
    #
    # model-visible:
    #
    #     BOS + raw document tokens
    #
    # causal supervised positions:
    #
    #     len(model_visible) - 1
    #
    # which should equal native document_token_count.
    # -------------------------------------------------------------------------

    supervised_token_count = (
        len(
            document_a
        )
        - 1
    )


    native_document_token_count = int(
        row[
            "document_token_count"
        ]
    )


    if (
        supervised_token_count
        != native_document_token_count
    ):

        raise RuntimeError(
            f"Record {record_index}: "
            f"synthetic supervised-token count "
            f"{supervised_token_count} "
            f"does not equal native "
            f"document_token_count "
            f"{native_document_token_count}."
        )


    nonanswer_supervised_token_count = (
        supervised_token_count
        - 1
    )


    if (
        nonanswer_supervised_token_count
        < 0
    ):

        raise RuntimeError(
            f"Record {record_index}: "
            f"negative non-answer "
            f"supervision count."
        )


    # -------------------------------------------------------------------------
    # Build frozen pair payload.
    # -------------------------------------------------------------------------

    pair_id = (
        f"base_{record_index:06d}"
    )


    candidate_pair = sorted(
        [
            int(
                slot0_target_token_id
            ),
            int(
                slot1_target_token_id
            ),
        ]
    )


    return {
        "pair_id": (
            pair_id
        ),

        "base_record_index": int(
            record_index
        ),

        "base_record_id": str(
            row[
                "id"
            ]
        ),

        "native_query_slot": int(
            native_query_slot
        ),

        "base_axis": row.get(
            "axis"
        ),

        "base_cell": row.get(
            "cell"
        ),

        "base_actual_cell": row.get(
            "actual_cell"
        ),

        "base_relation_round": row.get(
            "relation_round"
        ),

        "base_layout": row.get(
            "layout"
        ),

        "mapping_0": {
            "slot": 0,

            "source": (
                source0
            ),

            "source_index": int(
                mapping0[
                    "source_index"
                ]
            ),

            "target": (
                target_text0
            ),

            "target_index": int(
                mapping0[
                    "target_index"
                ]
            ),

            "target_token_id": int(
                slot0_target_token_id
            ),
        },

        "mapping_1": {
            "slot": 1,

            "source": (
                source1
            ),

            "source_index": int(
                mapping1[
                    "source_index"
                ]
            ),

            "target": (
                target_text1
            ),

            "target_index": int(
                mapping1[
                    "target_index"
                ]
            ),

            "target_token_id": int(
                slot1_target_token_id
            ),
        },

        "candidate_pair": (
            candidate_pair
        ),

        "query_difference_position": int(
            query_difference_position
        ),

        "answer_token_index": int(
            answer_token_index
        ),

        "answer_causal_position": int(
            answer_causal_position
        ),

        "model_visible_document_length": int(
            len(
                document_a
            )
        ),

        "raw_document_token_count": int(
            len(
                document_a
            )
            - 1
        ),

        "supervised_token_count": int(
            supervised_token_count
        ),

        "nonanswer_supervised_token_count": int(
            nonanswer_supervised_token_count
        ),

        "twin_a": {
            "twin": (
                "A"
            ),

            "query_slot": 0,

            "query_identity": (
                source0
            ),

            "target_token_id": int(
                slot0_target_token_id
            ),

            "distractor_token_id": int(
                slot1_target_token_id
            ),

            "prompt": (
                twin_a_prompt
            ),

            "full_text": (
                twin_a_text
            ),

            "prefix_token_ids": (
                prefix_a
            ),

            "full_document_token_ids": (
                document_a
            ),
        },

        "twin_b": {
            "twin": (
                "B"
            ),

            "query_slot": 1,

            "query_identity": (
                source1
            ),

            "target_token_id": int(
                slot1_target_token_id
            ),

            "distractor_token_id": int(
                slot0_target_token_id
            ),

            "prompt": (
                twin_b_prompt
            ),

            "full_text": (
                twin_b_text
            ),

            "prefix_token_ids": (
                prefix_b
            ),

            "full_document_token_ids": (
                document_b
            ),
        },
    }


# =============================================================================
# BUILD / AUDIT COMPLETE PAIR POOL
# =============================================================================

def build_pair_pool(
    tokenizer: Tokenizer,
    records: list[dict[str, Any]],
    bos_token_id: int,
) -> list[dict[str, Any]]:

    header(
        "BUILDING SAME-RECORD FULL-DOCUMENT COUNTERFACTUAL PAIR POOL"
    )


    pair_pool: list[
        dict[str, Any]
    ] = []


    for record_index, row in enumerate(
        records
    ):

        pair = build_counterfactual_pair(
            tokenizer,
            row,
            record_index,
            bos_token_id,
        )

        pair_pool.append(
            pair
        )


    if len(
        pair_pool
    ) != EXPECTED_PAIR_POOL_SIZE:

        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_PAIR_POOL_SIZE} "
            f"counterfactual pairs; got "
            f"{len(pair_pool)}."
        )


    pair_ids = [
        str(
            pair[
                "pair_id"
            ]
        )
        for pair
        in pair_pool
    ]


    if len(
        set(
            pair_ids
        )
    ) != len(
        pair_ids
    ):

        raise RuntimeError(
            "Counterfactual pair IDs "
            "are not unique."
        )


    base_indices = [
        int(
            pair[
                "base_record_index"
            ]
        )
        for pair
        in pair_pool
    ]


    if sorted(
        base_indices
    ) != list(
        range(
            EXPECTED_NATIVE_RECORDS
        )
    ):

        raise RuntimeError(
            "Pair pool does not cover "
            "every native record exactly once."
        )


    # -------------------------------------------------------------------------
    # Pool-level diagnostics.
    # -------------------------------------------------------------------------

    candidate_pairs = Counter()

    prefix_lengths = Counter()

    document_lengths = Counter()

    query_positions = Counter()

    answer_positions = Counter()

    native_slot_counts = Counter()


    for pair in pair_pool:

        a = pair[
            "twin_a"
        ]

        b = pair[
            "twin_b"
        ]


        if int(
            a[
                "query_slot"
            ]
        ) != 0:

            raise RuntimeError(
                f"{pair['pair_id']}: "
                f"Twin A is not slot 0."
            )


        if int(
            b[
                "query_slot"
            ]
        ) != 1:

            raise RuntimeError(
                f"{pair['pair_id']}: "
                f"Twin B is not slot 1."
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
                f"{pair['pair_id']}: "
                f"A target != B distractor."
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
                f"{pair['pair_id']}: "
                f"B target != A distractor."
            )


        candidate_a = sorted(
            [
                int(
                    a[
                        "target_token_id"
                    ]
                ),
                int(
                    a[
                        "distractor_token_id"
                    ]
                ),
            ]
        )


        candidate_b = sorted(
            [
                int(
                    b[
                        "target_token_id"
                    ]
                ),
                int(
                    b[
                        "distractor_token_id"
                    ]
                ),
            ]
        )


        stored_candidate_pair = [
            int(token_id)
            for token_id
            in pair[
                "candidate_pair"
            ]
        ]


        if (
            candidate_a
            != candidate_b
        ):

            raise RuntimeError(
                f"{pair['pair_id']}: "
                f"twins have different "
                f"candidate pairs."
            )


        if (
            candidate_a
            != stored_candidate_pair
        ):

            raise RuntimeError(
                f"{pair['pair_id']}: "
                f"stored candidate pair "
                f"is incorrect."
            )


        ids_a = [
            int(token_id)
            for token_id
            in a[
                "full_document_token_ids"
            ]
        ]

        ids_b = [
            int(token_id)
            for token_id
            in b[
                "full_document_token_ids"
            ]
        ]


        if len(
            ids_a
        ) != len(
            ids_b
        ):

            raise RuntimeError(
                f"{pair['pair_id']}: "
                f"full document lengths differ."
            )


        actual_differences = [
            position
            for position, (
                token_a,
                token_b,
            )
            in enumerate(
                zip(
                    ids_a,
                    ids_b,
                )
            )
            if token_a != token_b
        ]


        expected_differences = sorted(
            [
                int(
                    pair[
                        "query_difference_position"
                    ]
                ),
                int(
                    pair[
                        "answer_token_index"
                    ]
                ),
            ]
        )


        if (
            actual_differences
            != expected_differences
        ):

            raise RuntimeError(
                f"{pair['pair_id']}: "
                f"full-document difference "
                f"contract changed."
            )


        candidate_pairs[
            tuple(
                stored_candidate_pair
            )
        ] += 1


        prefix_lengths[
            len(
                a[
                    "prefix_token_ids"
                ]
            )
        ] += 1


        document_lengths[
            len(
                ids_a
            )
        ] += 1


        query_positions[
            int(
                pair[
                    "query_difference_position"
                ]
            )
        ] += 1


        answer_positions[
            int(
                pair[
                    "answer_token_index"
                ]
            )
        ] += 1


        native_slot_counts[
            int(
                pair[
                    "native_query_slot"
                ]
            )
        ] += 1


    print(
        f"Verified same-record pairs: "
        f"{len(pair_pool)}"
    )

    print(
        f"Unique complete synthetic documents: "
        f"{len(pair_pool) * 2}"
    )

    print(
        f"Distinct candidate pairs: "
        f"{len(candidate_pairs)}"
    )

    print(
        f"Distinct prefix lengths: "
        f"{len(prefix_lengths)}"
    )

    print(
        f"Distinct full-document lengths: "
        f"{len(document_lengths)}"
    )

    print(
        f"Distinct query-token positions: "
        f"{len(query_positions)}"
    )

    print(
        f"Distinct answer-token positions: "
        f"{len(answer_positions)}"
    )

    print(
        f"Native slot provenance: "
        f"slot0={native_slot_counts[0]}, "
        f"slot1={native_slot_counts[1]}"
    )


    return pair_pool


# =============================================================================
# WRITE COMPLETE PAIR POOL
# =============================================================================

def write_pair_pool(
    pair_pool: list[dict[str, Any]],
) -> str:

    payload = {
        "experiment": (
            "DaveLM v0.9 Treatment #5"
        ),

        "artifact_type": (
            "full_document_counterfactual_pair_pool"
        ),

        "construction": (
            "same_native_record_to_slot0_and_slot1_twins"
        ),

        "pair_count": int(
            len(
                pair_pool
            )
        ),

        "unique_synthetic_document_count": int(
            len(
                pair_pool
            )
            * 2
        ),

        "pairs": (
            pair_pool
        ),
    }


    serialized = canonical_bytes(
        payload
    )


    digest = sha256_bytes(
        serialized
    )


    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    PAIR_POOL_PATH.write_bytes(
        serialized
    )


    reread = PAIR_POOL_PATH.read_bytes()


    if sha256_bytes(
        reread
    ) != digest:

        raise RuntimeError(
            "Pair-pool artifact "
            "write/read SHA mismatch."
        )


    return digest


# =============================================================================
# BUILD COMPLETE FROZEN 1000-STEP SCHEDULE
# =============================================================================

def build_frozen_schedule(
    pair_pool: list[dict[str, Any]],
) -> dict[str, Any]:

    header(
        "BUILDING COMPLETE 1,000-STEP FULL-DOCUMENT FROZEN SCHEDULE"
    )


    pair_by_id = {
        str(
            pair[
                "pair_id"
            ]
        ): pair
        for pair
        in pair_pool
    }


    pair_ids = sorted(
        pair_by_id.keys()
    )


    if len(
        pair_ids
    ) != EXPECTED_PAIR_POOL_SIZE:

        raise RuntimeError(
            "Pair lookup size mismatch."
        )


    rng = random.Random(
        TREATMENT5_SCHEDULE_SEED
    )


    # -------------------------------------------------------------------------
    # Build deterministic shuffled-epoch pair stream.
    #
    # 1536 unique pairs.
    #
    # 16000 total pair presentations.
    #
    # Therefore each pair naturally appears either 10 or 11 times.
    # -------------------------------------------------------------------------

    pair_stream: list[
        dict[str, Any]
    ] = []


    epoch_number = 0


    while len(
        pair_stream
    ) < EXPECTED_PAIR_PRESENTATIONS:

        epoch_pair_ids = list(
            pair_ids
        )


        rng.shuffle(
            epoch_pair_ids
        )


        presentations_remaining = (
            EXPECTED_PAIR_PRESENTATIONS
            - len(
                pair_stream
            )
        )


        take_count = min(
            presentations_remaining,
            len(
                epoch_pair_ids
            ),
        )


        for pair_id in epoch_pair_ids[
            :take_count
        ]:

            pair_stream.append(
                {
                    "pair_id": (
                        pair_id
                    ),

                    "epoch": int(
                        epoch_number
                    ),
                }
            )


        epoch_number += 1


    if len(
        pair_stream
    ) != EXPECTED_PAIR_PRESENTATIONS:

        raise RuntimeError(
            f"Pair stream length mismatch: "
            f"{len(pair_stream)}."
        )


    # -------------------------------------------------------------------------
    # Freeze steps.
    # -------------------------------------------------------------------------

    steps_data: list[
        dict[str, Any]
    ] = []


    pair_exposure = Counter()

    slot_presentations = Counter()


    total_supervised_tokens = 0

    total_answer_substitutions = 0

    total_nonanswer_supervised_tokens = 0


    cursor = 0


    for step_number in range(
        1,
        MAX_STEPS + 1,
    ):

        selected_presentations = (
            pair_stream[
                cursor:
                cursor
                + PAIRS_PER_BATCH
            ]
        )


        cursor += PAIRS_PER_BATCH


        if len(
            selected_presentations
        ) != PAIRS_PER_BATCH:

            raise RuntimeError(
                f"Step {step_number}: "
                f"expected "
                f"{PAIRS_PER_BATCH} "
                f"complete pair presentations; "
                f"got "
                f"{len(selected_presentations)}."
            )


        examples: list[
            dict[str, Any]
        ] = []


        step_pair_ids: list[
            str
        ] = []


        for presentation in (
            selected_presentations
        ):

            pair_id = str(
                presentation[
                    "pair_id"
                ]
            )


            pair = pair_by_id[
                pair_id
            ]


            pair_exposure[
                pair_id
            ] += 1


            step_pair_ids.append(
                pair_id
            )


            # -------------------------------------------------------------
            # COMPLETE PAIR STAYS ADJACENT:
            #
            #     Twin A
            #     Twin B
            #
            # -------------------------------------------------------------

            for twin_name, twin_key in (
                (
                    "A",
                    "twin_a",
                ),
                (
                    "B",
                    "twin_b",
                ),
            ):

                twin = pair[
                    twin_key
                ]


                query_slot = int(
                    twin[
                        "query_slot"
                    ]
                )


                target_token_id = int(
                    twin[
                        "target_token_id"
                    ]
                )


                distractor_token_id = int(
                    twin[
                        "distractor_token_id"
                    ]
                )


                full_document_token_ids = [
                    int(token_id)
                    for token_id
                    in twin[
                        "full_document_token_ids"
                    ]
                ]


                supervised_token_count = int(
                    pair[
                        "supervised_token_count"
                    ]
                )


                nonanswer_supervised_count = int(
                    pair[
                        "nonanswer_supervised_token_count"
                    ]
                )


                # One answer substitution per example.
                if (
                    supervised_token_count
                    != (
                        nonanswer_supervised_count
                        + 1
                    )
                ):

                    raise RuntimeError(
                        f"Step {step_number}, "
                        f"{pair_id} Twin "
                        f"{twin_name}: "
                        f"supervision accounting "
                        f"does not equal "
                        f"nonanswer + one answer."
                    )


                example = {
                    "pair_id": (
                        pair_id
                    ),

                    "base_record_index": int(
                        pair[
                            "base_record_index"
                        ]
                    ),

                    "base_record_id": str(
                        pair[
                            "base_record_id"
                        ]
                    ),

                    "epoch": int(
                        presentation[
                            "epoch"
                        ]
                    ),

                    "twin": (
                        twin_name
                    ),

                    "query_slot": (
                        query_slot
                    ),

                    "target_token_id": (
                        target_token_id
                    ),

                    "distractor_token_id": (
                        distractor_token_id
                    ),

                    "candidate_pair": [
                        int(token_id)
                        for token_id
                        in pair[
                            "candidate_pair"
                        ]
                    ],

                    "query_difference_position": int(
                        pair[
                            "query_difference_position"
                        ]
                    ),

                    "answer_token_index": int(
                        pair[
                            "answer_token_index"
                        ]
                    ),

                    "answer_causal_position": int(
                        pair[
                            "answer_causal_position"
                        ]
                    ),

                    "model_visible_document_length": int(
                        pair[
                            "model_visible_document_length"
                        ]
                    ),

                    "supervised_token_count": (
                        supervised_token_count
                    ),

                    "nonanswer_supervised_token_count": (
                        nonanswer_supervised_count
                    ),

                    # -----------------------------------------------------
                    # CRITICAL:
                    #
                    # The FINAL schedule commits to the exact complete
                    # model-consumed sequence.
                    # -----------------------------------------------------

                    "full_document_token_ids": (
                        full_document_token_ids
                    ),
                }


                examples.append(
                    example
                )


                slot_presentations[
                    query_slot
                ] += 1


                total_supervised_tokens += (
                    supervised_token_count
                )


                total_nonanswer_supervised_tokens += (
                    nonanswer_supervised_count
                )


                total_answer_substitutions += 1


        if len(
            examples
        ) != BATCH_SIZE:

            raise RuntimeError(
                f"Step {step_number}: "
                f"expected batch size "
                f"{BATCH_SIZE}; "
                f"got {len(examples)}."
            )


        # ---------------------------------------------------------------------
        # Pair adjacency / reciprocity audit inside the frozen batch.
        # ---------------------------------------------------------------------

        for batch_offset in range(
            0,
            BATCH_SIZE,
            2,
        ):

            a = examples[
                batch_offset
            ]

            b = examples[
                batch_offset + 1
            ]


            if (
                a[
                    "pair_id"
                ]
                != b[
                    "pair_id"
                ]
            ):

                raise RuntimeError(
                    f"Step {step_number}: "
                    f"complete twin pair "
                    f"was separated."
                )


            if int(
                a[
                    "query_slot"
                ]
            ) != 0:

                raise RuntimeError(
                    f"Step {step_number}: "
                    f"first member of pair "
                    f"is not slot 0."
                )


            if int(
                b[
                    "query_slot"
                ]
            ) != 1:

                raise RuntimeError(
                    f"Step {step_number}: "
                    f"second member of pair "
                    f"is not slot 1."
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
                    f"Step {step_number}: "
                    f"Twin A target != "
                    f"Twin B distractor."
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
                    f"Step {step_number}: "
                    f"Twin B target != "
                    f"Twin A distractor."
                )


            if (
                a[
                    "candidate_pair"
                ]
                != b[
                    "candidate_pair"
                ]
            ):

                raise RuntimeError(
                    f"Step {step_number}: "
                    f"twins have different "
                    f"candidate pairs."
                )


        steps_data.append(
            {
                "step": int(
                    step_number
                ),

                "pair_count": (
                    PAIRS_PER_BATCH
                ),

                "example_count": (
                    BATCH_SIZE
                ),

                "pair_ids": (
                    step_pair_ids
                ),

                "examples": (
                    examples
                ),
            }
        )


    # -------------------------------------------------------------------------
    # Final schedule accounting.
    # -------------------------------------------------------------------------

    if cursor != EXPECTED_PAIR_PRESENTATIONS:

        raise RuntimeError(
            f"Consumed {cursor} pair "
            f"presentations; expected "
            f"{EXPECTED_PAIR_PRESENTATIONS}."
        )


    if len(
        steps_data
    ) != MAX_STEPS:

        raise RuntimeError(
            f"Frozen schedule has "
            f"{len(steps_data)} steps; "
            f"expected {MAX_STEPS}."
        )


    if (
        total_answer_substitutions
        != EXPECTED_ANSWER_SUBSTITUTIONS
    ):

        raise RuntimeError(
            f"Answer substitution count "
            f"{total_answer_substitutions} "
            f"does not equal expected "
            f"{EXPECTED_ANSWER_SUBSTITUTIONS}."
        )


    if (
        slot_presentations[
            0
        ]
        != EXPECTED_SLOT_PRESENTATIONS
    ):

        raise RuntimeError(
            f"Slot-0 schedule count "
            f"{slot_presentations[0]} "
            f"does not equal "
            f"{EXPECTED_SLOT_PRESENTATIONS}."
        )


    if (
        slot_presentations[
            1
        ]
        != EXPECTED_SLOT_PRESENTATIONS
    ):

        raise RuntimeError(
            f"Slot-1 schedule count "
            f"{slot_presentations[1]} "
            f"does not equal "
            f"{EXPECTED_SLOT_PRESENTATIONS}."
        )


    # The loss accounting identity must ALWAYS hold:
    #
    #     all supervised tokens
    #       =
    #     non-answer ordinary CE tokens
    #       +
    #     answer replacement positions

    if (
        total_supervised_tokens
        != (
            total_nonanswer_supervised_tokens
            + total_answer_substitutions
        )
    ):

        raise RuntimeError(
            "Global supervision accounting "
            "identity failed."
        )


    exposure_values = [
        int(
            pair_exposure[
                pair_id
            ]
        )
        for pair_id
        in pair_ids
    ]


    if any(
        exposure <= 0
        for exposure
        in exposure_values
    ):

        raise RuntimeError(
            "At least one pair received "
            "zero schedule exposure."
        )


    min_exposure = min(
        exposure_values
    )

    max_exposure = max(
        exposure_values
    )


    if (
        max_exposure
        - min_exposure
    ) > 1:

        raise RuntimeError(
            f"Pair exposure imbalance "
            f"exceeds one presentation: "
            f"min={min_exposure}, "
            f"max={max_exposure}."
        )


    if min_exposure != 10:

        raise RuntimeError(
            f"Expected minimum pair exposure "
            f"10; got {min_exposure}."
        )


    if max_exposure != 11:

        raise RuntimeError(
            f"Expected maximum pair exposure "
            f"11; got {max_exposure}."
        )


    # -------------------------------------------------------------------------
    # Historical T4 comparison — informational only.
    # -------------------------------------------------------------------------

    supervised_token_delta_vs_t4 = (
        total_supervised_tokens
        - T4_REFERENCE_SUPERVISED_TOKENS
    )


    nonanswer_token_delta_vs_t4 = (
        total_nonanswer_supervised_tokens
        - T4_REFERENCE_NONANSWER_TOKENS
    )


    print(
        f"Steps: "
        f"{MAX_STEPS}"
    )

    print(
        f"Pairs per batch: "
        f"{PAIRS_PER_BATCH}"
    )

    print(
        f"Examples per batch: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Unique pairs: "
        f"{EXPECTED_PAIR_POOL_SIZE}"
    )

    print(
        f"Pair presentations: "
        f"{EXPECTED_PAIR_PRESENTATIONS}"
    )

    print(
        f"Scheduled examples: "
        f"{EXPECTED_SCHEDULED_EXAMPLES}"
    )

    print(
        f"Slot-0 presentations: "
        f"{slot_presentations[0]}"
    )

    print(
        f"Slot-1 presentations: "
        f"{slot_presentations[1]}"
    )

    print(
        f"Pair exposure range: "
        f"{min_exposure}-{max_exposure}"
    )

    print(
        f"Shuffle epochs used: "
        f"{epoch_number}"
    )

    print()

    print(
        f"Answer substitutions: "
        f"{total_answer_substitutions}"
    )

    print(
        f"Total supervised tokens: "
        f"{total_supervised_tokens}"
    )

    print(
        f"Non-answer ordinary-CE tokens: "
        f"{total_nonanswer_supervised_tokens}"
    )

    print()

    print(
        f"T4 reference supervised tokens: "
        f"{T4_REFERENCE_SUPERVISED_TOKENS}"
    )

    print(
        f"T5 - T4 supervised-token delta: "
        f"{supervised_token_delta_vs_t4:+d}"
    )

    print(
        f"T4 reference non-answer tokens: "
        f"{T4_REFERENCE_NONANSWER_TOKENS}"
    )

    print(
        f"T5 - T4 non-answer-token delta: "
        f"{nonanswer_token_delta_vs_t4:+d}"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "  The T4 token totals above are "
        "REFERENCE ONLY."
    )

    print(
        "  T5 is NOT failed or modified "
        "to force those totals."
    )


    return {
        "experiment": (
            "DaveLM v0.9 Treatment #5"
        ),

        "artifact_type": (
            "full_document_frozen_training_schedule"
        ),

        "construction": (
            "same_record_paired_counterfactual_query_binding"
        ),

        "train_seed": int(
            TRAIN_SEED
        ),

        "schedule_seed": int(
            TREATMENT5_SCHEDULE_SEED
        ),

        "steps": int(
            MAX_STEPS
        ),

        "batch_size": int(
            BATCH_SIZE
        ),

        "pairs_per_batch": int(
            PAIRS_PER_BATCH
        ),

        "unique_pair_count": int(
            EXPECTED_PAIR_POOL_SIZE
        ),

        "pair_presentations": int(
            EXPECTED_PAIR_PRESENTATIONS
        ),

        "scheduled_examples": int(
            EXPECTED_SCHEDULED_EXAMPLES
        ),

        "slot_0_presentations": int(
            slot_presentations[
                0
            ]
        ),

        "slot_1_presentations": int(
            slot_presentations[
                1
            ]
        ),

        "pair_exposure_min": int(
            min_exposure
        ),

        "pair_exposure_max": int(
            max_exposure
        ),

        "shuffle_epochs_used": int(
            epoch_number
        ),

        "answer_substitutions": int(
            total_answer_substitutions
        ),

        "total_supervised_tokens": int(
            total_supervised_tokens
        ),

        "nonanswer_supervised_tokens": int(
            total_nonanswer_supervised_tokens
        ),

        "t4_reference": {
            "supervised_tokens": int(
                T4_REFERENCE_SUPERVISED_TOKENS
            ),

            "answer_substitutions": int(
                T4_REFERENCE_ANSWER_SUBSTITUTIONS
            ),

            "nonanswer_tokens": int(
                T4_REFERENCE_NONANSWER_TOKENS
            ),

            "t5_supervised_token_delta": int(
                supervised_token_delta_vs_t4
            ),

            "t5_nonanswer_token_delta": int(
                nonanswer_token_delta_vs_t4
            ),

            "reference_only": True,
        },

        "steps_data": (
            steps_data
        ),
    }


# =============================================================================
# WRITE FROZEN SCHEDULE
# =============================================================================

def write_frozen_schedule(
    schedule: dict[str, Any],
) -> str:

    serialized = canonical_bytes(
        schedule
    )


    digest = sha256_bytes(
        serialized
    )


    SCHEDULE_PATH.write_bytes(
        serialized
    )


    reread = SCHEDULE_PATH.read_bytes()


    if sha256_bytes(
        reread
    ) != digest:

        raise RuntimeError(
            "Frozen schedule artifact "
            "write/read SHA mismatch."
        )


    return digest


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    header(
        "DAVELM v0.9 — TREATMENT #5 "
        "FINAL FULL-DOCUMENT PREFLIGHT"
    )


    print(
        "Construction:"
    )

    print(
        "  SAME native record -> "
        "slot-0 twin + slot-1 twin"
    )

    print()

    print(
        "Training performed: NO"
    )

    print(
        "Model constructed: NO"
    )

    print(
        "Checkpoint loaded into model: NO"
    )

    print(
        "Optimizer created: NO"
    )

    print(
        "Backward calls: 0"
    )

    print(
        "Positive-control evaluation: NO"
    )

    print(
        "Sealed evaluation opened: NO"
    )

    print(
        "Protected source modification: NO"
    )


    # =========================================================================
    # STARTING CHECKPOINT FILE INTEGRITY
    #
    # Reading bytes for SHA is NOT loading the checkpoint into Baby.
    # =========================================================================

    header(
        "VERIFYING FROZEN STARTING CHECKPOINT"
    )


    if not START_CHECKPOINT.is_file():

        raise FileNotFoundError(
            f"Frozen starting checkpoint "
            f"does not exist:\n"
            f"{START_CHECKPOINT}"
        )


    observed_start_checkpoint_sha256 = (
        sha256_file(
            START_CHECKPOINT
        )
    )


    print(
        f"Checkpoint:"
    )

    print(
        f"  {START_CHECKPOINT}"
    )

    print()

    print(
        f"Expected SHA256:"
    )

    print(
        f"  "
        f"{EXPECTED_START_CHECKPOINT_SHA256}"
    )

    print()

    print(
        f"Observed SHA256:"
    )

    print(
        f"  "
        f"{observed_start_checkpoint_sha256}"
    )


    if (
        observed_start_checkpoint_sha256
        != EXPECTED_START_CHECKPOINT_SHA256
    ):

        raise RuntimeError(
            "Frozen starting checkpoint "
            "SHA256 mismatch."
        )


    print()

    print(
        "Starting checkpoint integrity: PASS"
    )


    # =========================================================================
    # RECONSTRUCT + AUDIT NATIVE CORPUS
    # =========================================================================

    (
        tokenizer,
        records,
        bos_token_id,
    ) = reconstruct_native_corpus()


    native_audit = audit_native_documents(
        tokenizer,
        records,
        bos_token_id,
    )


    # =========================================================================
    # CONSTRUCT COMPLETE SAME-RECORD COUNTERFACTUAL PAIRS
    # =========================================================================

    pair_pool = build_pair_pool(
        tokenizer,
        records,
        bos_token_id,
    )


    # =========================================================================
    # WRITE / HASH PAIR POOL
    # =========================================================================

    header(
        "FREEZING FULL-DOCUMENT COUNTERFACTUAL PAIR POOL"
    )


    pair_pool_sha256 = write_pair_pool(
        pair_pool
    )


    print(
        "Pair-pool artifact:"
    )

    print(
        f"  {PAIR_POOL_PATH}"
    )

    print()

    print(
        "Pair-pool SHA256:"
    )

    print(
        f"  {pair_pool_sha256}"
    )


    # =========================================================================
    # BUILD COMPLETE 1000-STEP SCHEDULE
    # =========================================================================

    schedule = build_frozen_schedule(
        pair_pool
    )


    # =========================================================================
    # WRITE / HASH FINAL TRAINING SCHEDULE
    # =========================================================================

    header(
        "FREEZING COMPLETE TRAINING SCHEDULE"
    )


    schedule_sha256 = (
        write_frozen_schedule(
            schedule
        )
    )


    print(
        "Frozen schedule artifact:"
    )

    print(
        f"  {SCHEDULE_PATH}"
    )

    print()

    print(
        "Frozen schedule SHA256:"
    )

    print(
        f"  {schedule_sha256}"
    )


    # =========================================================================
    # WRITE FINAL PREFLIGHT RESULT
    # =========================================================================

    result = {
        "experiment": (
            "DaveLM v0.9 Treatment #5"
        ),

        "classification": (
            "FINAL_FULL_DOCUMENT_PREFLIGHT_PASS"
        ),

        "hypothesis": (
            "paired counterfactual query "
            "binding forces the model to "
            "condition answer selection on "
            "the query because matched twins "
            "share context and candidate pair "
            "while reciprocal query/answer "
            "identity changes."
        ),

        "construction": (
            "same_native_record_to_complete_"
            "slot0_and_slot1_counterfactual_twins"
        ),

        "bos_token_id": int(
            bos_token_id
        ),

        "native_training_records": int(
            len(
                records
            )
        ),

        "unique_counterfactual_pairs": int(
            len(
                pair_pool
            )
        ),

        "unique_complete_synthetic_documents": int(
            len(
                pair_pool
            )
            * 2
        ),

        "train_seed": int(
            TRAIN_SEED
        ),

        "schedule_seed": int(
            TREATMENT5_SCHEDULE_SEED
        ),

        "steps": int(
            MAX_STEPS
        ),

        "batch_size": int(
            BATCH_SIZE
        ),

        "pairs_per_batch": int(
            PAIRS_PER_BATCH
        ),

        "pair_presentations": int(
            schedule[
                "pair_presentations"
            ]
        ),

        "scheduled_examples": int(
            schedule[
                "scheduled_examples"
            ]
        ),

        "slot_0_presentations": int(
            schedule[
                "slot_0_presentations"
            ]
        ),

        "slot_1_presentations": int(
            schedule[
                "slot_1_presentations"
            ]
        ),

        "pair_exposure_min": int(
            schedule[
                "pair_exposure_min"
            ]
        ),

        "pair_exposure_max": int(
            schedule[
                "pair_exposure_max"
            ]
        ),

        "answer_substitutions": int(
            schedule[
                "answer_substitutions"
            ]
        ),

        "total_supervised_tokens": int(
            schedule[
                "total_supervised_tokens"
            ]
        ),

        "nonanswer_supervised_tokens": int(
            schedule[
                "nonanswer_supervised_tokens"
            ]
        ),

        "t4_reference": (
            schedule[
                "t4_reference"
            ]
        ),

        "native_audit": (
            native_audit
        ),

        "starting_checkpoint": {
            "path": str(
                START_CHECKPOINT
            ),

            "expected_sha256": (
                EXPECTED_START_CHECKPOINT_SHA256
            ),

            "observed_sha256": (
                observed_start_checkpoint_sha256
            ),

            "pass": True,
        },

        "pair_pool": {
            "path": str(
                PAIR_POOL_PATH
            ),

            "sha256": (
                pair_pool_sha256
            ),
        },

        "frozen_schedule": {
            "path": str(
                SCHEDULE_PATH
            ),

            "sha256": (
                schedule_sha256
            ),
        },

        "frozen_objective": {
            "margin": 0.5,

            "membership_loss": (
                "logsumexp(full_vocab_logits)"
                " - logsumexp("
                "[correct_logit,"
                " distractor_logit])"
            ),

            "selector_loss": (
                "relu("
                "0.5 - "
                "(correct_logit - "
                "distractor_logit)"
                ")"
            ),

            "answer_replacement_loss": (
                "membership_loss + "
                "selector_loss"
            ),

            "original_answer_ce_retained": (
                False
            ),

            "nonanswer_supervision": (
                "ordinary full-vocabulary "
                "causal cross entropy"
            ),
        },

        "safety": {
            "model_constructed": False,

            "checkpoint_loaded_into_model": False,

            "optimizer_created": False,

            "backward_calls": 0,

            "training_performed": False,

            "positive_controls_evaluated": False,

            "sealed_evaluation_opened": False,

            "protected_source_modified": False,
        },

        "what_pass_means": (
            "The complete same-record "
            "counterfactual Treatment #5 "
            "training data and exact "
            "1000-step schedule were "
            "constructed, audited, hashed, "
            "and frozen before training. "
            "A PASS establishes the training "
            "artifact contract only; it does "
            "not establish that Baby will "
            "learn or generalize query binding."
        ),
    }


    RESULT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


    # =========================================================================
    # HUMAN-READABLE FINAL REPORT
    # =========================================================================

    header(
        "TREATMENT #5 FINAL FULL-DOCUMENT PREFLIGHT REPORT"
    )


    print(
        "RESULT: PASS"
    )

    print()

    print(
        f"Native BOS token ID: "
        f"{bos_token_id}"
    )

    print()

    print(
        f"Native records: "
        f"{len(records)}"
    )

    print(
        f"Verified complete counterfactual pairs: "
        f"{len(pair_pool)}"
    )

    print(
        f"Unique complete synthetic documents: "
        f"{len(pair_pool) * 2}"
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
        f"Pair presentations: "
        f"{schedule['pair_presentations']}"
    )

    print(
        f"Scheduled examples: "
        f"{schedule['scheduled_examples']}"
    )

    print()

    print(
        f"Slot-0 presentations: "
        f"{schedule['slot_0_presentations']}"
    )

    print(
        f"Slot-1 presentations: "
        f"{schedule['slot_1_presentations']}"
    )

    print(
        f"Pair exposure range: "
        f"{schedule['pair_exposure_min']}-"
        f"{schedule['pair_exposure_max']}"
    )

    print()

    print(
        "SUPERVISION ACCOUNTING"
    )

    print(
        f"  Answer substitutions: "
        f"{schedule['answer_substitutions']}"
    )

    print(
        f"  Total supervised tokens: "
        f"{schedule['total_supervised_tokens']}"
    )

    print(
        f"  Non-answer ordinary-CE tokens: "
        f"{schedule['nonanswer_supervised_tokens']}"
    )

    print()

    print(
        "T4 REFERENCE ONLY"
    )

    print(
        f"  T4 supervised tokens: "
        f"{T4_REFERENCE_SUPERVISED_TOKENS}"
    )

    print(
        f"  T5 delta: "
        f"{schedule['t4_reference']['t5_supervised_token_delta']:+d}"
    )

    print(
        f"  T4 non-answer tokens: "
        f"{T4_REFERENCE_NONANSWER_TOKENS}"
    )

    print(
        f"  T5 delta: "
        f"{schedule['t4_reference']['t5_nonanswer_token_delta']:+d}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "  A non-zero T5/T4 non-answer-token "
        "delta is NOT automatically a failure."
    )

    print(
        "  T5 paired sampling is allowed to "
        "produce its natural document-length "
        "distribution."
    )

    print()

    print(
        "START CHECKPOINT SHA256:"
    )

    print(
        f"  "
        f"{observed_start_checkpoint_sha256}"
    )

    print()

    print(
        "FULL-DOCUMENT PAIR POOL SHA256:"
    )

    print(
        f"  "
        f"{pair_pool_sha256}"
    )

    print()

    print(
        "FINAL FROZEN SCHEDULE SHA256:"
    )

    print(
        f"  "
        f"{schedule_sha256}"
    )

    print()

    print(
        "PAIR POOL:"
    )

    print(
        f"  "
        f"{PAIR_POOL_PATH}"
    )

    print()

    print(
        "FROZEN SCHEDULE:"
    )

    print(
        f"  "
        f"{SCHEDULE_PATH}"
    )

    print()

    print(
        "PREFLIGHT RESULT:"
    )

    print(
        f"  "
        f"{RESULT_PATH}"
    )

    print()

    print(
        "SAFETY:"
    )

    print(
        "  Baby constructed: NO"
    )

    print(
        "  Checkpoint loaded into model: NO"
    )

    print(
        "  Optimizer created: NO"
    )

    print(
        "  Backward calls: 0"
    )

    print(
        "  Training performed: NO"
    )

    print(
        "  Positive controls evaluated: NO"
    )

    print(
        "  Sealed evaluation opened: NO"
    )

    print(
        "  Protected source modified: NO"
    )

    print()

    print(
        "PASS — COMPLETE TREATMENT #5 "
        "TRAINING ARTIFACTS FROZEN"
    )

    print()

    print(
        "NEXT STEP:"
    )

    print(
        "  Build treatment5_train.py against "
        "THIS exact final schedule SHA."
    )


if __name__ == "__main__":

    main()