from __future__ import annotations

import sys
import json
import hashlib
from pathlib import Path
from collections import Counter

import torch
from tokenizers import Tokenizer


# =============================================================================
# DaveLM v0.9 — Treatment #2
# PAIRED QUERY-SWAP TRAINING
#
# THIS FILE IS PREFLIGHT ONLY.
#
# IT DOES NOT:
# - build a model
# - create an optimizer
# - call backward()
# - perform optimizer.step()
# - train Baby
# - evaluate sealed axes
#
# PURPOSE:
# 1. Reconstruct the exact original balanced two-map training corpus.
# 2. Build exactly one query-swapped counterpart for every training record.
# 3. Prove the counterpart changes ONLY the intended query/target selection.
# 4. Audit ACTUAL counterpart relation × geometry against the frozen
#    supported-withheld positive-control material.
# 5. Require ZERO positive-control leakage.
# 6. Construct the NEW deterministic paired 1000-step schedule:
#       16 independently sampled base docs
#       + their exact 16 counterparts
# 7. Build that schedule twice and require identical SHA256.
#
# NO OPTIMIZER STEPS ARE POSSIBLE IN THIS SCRIPT.
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")

OUTPUT_ROOT = Path(
    r"C:\DaveLM-CADAVER\two_mapping_query_swap_manual_seed8380"
)

PREFLIGHT_PATH = OUTPUT_ROOT / "query_swap_preflight.json"

START_CHECKPOINT = Path(
    r"C:\DaveLM-v0.9\experiments"
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

SEED = 8380
MINIBATCH_SEED = 8381

TOTAL_STEPS = 1000
BASES_PER_BATCH = 16
PAIRS_PER_BATCH = 16
TOTAL_BATCH_SIZE = 32
CONTEXT_SIZE = 256

EXPECTED_RECORDS = 1536
EXPECTED_DOCUMENT_TOKENS = 192


# =============================================================================
# IMPORT EXACT v0.9 CORPUS/SAMPLING MACHINERY
# =============================================================================

sys.path.insert(0, str(SOURCE_ROOT))

from experiments.two_mapping_contextual_binding import config

from experiments.two_mapping_contextual_binding.run import (
    _identity_pool,
    _layout_tables,
    _training_records,
    _geometry_schedule,
    _ordered_training_keys,
    _record,
)

from v0_8.document_sampling import (
    DocumentTokenStore,
    sample_document_batch,
)

from v0_2.train import file_sha256

from v0_8_3.protection import (
    strict_json_dumps,
)


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)
    print()


def save_json(path: Path, payload: dict) -> None:
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


def mapping_pairs(row: dict) -> list[tuple[int, int]]:
    return [
        (
            int(item["source_index"]),
            int(item["target_index"]),
        )
        for item in row["mappings"]
    ]


def relation_set(row: dict) -> frozenset[tuple[int, int]]:
    return frozenset(mapping_pairs(row))


def answer_relation(row: dict) -> tuple[int, int]:
    return (
        int(row["source_identity_index"]),
        int(row["target_identity_index"]),
    )


def actual_answer_geometry(row: dict) -> tuple[int, int]:
    """
    Geometry that matters for the supported-withheld leakage audit.

    Deliberately does NOT use query_slot, so a slot-label difference
    cannot hide exposure to the same relation at the same physical
    query/distance geometry.
    """
    return (
        int(row["layout"]["query_index"]),
        int(row["layout"]["target_to_query_distance"]),
    )


def exposure_tuple(row: dict) -> tuple[int, int, int, int]:
    source, target = answer_relation(row)
    query_index, distance = actual_answer_geometry(row)

    return (
        source,
        target,
        query_index,
        distance,
    )


def schedule_payload(
    document_indices: list[int],
    within_document_starts: list[int],
) -> bytes:

    return strict_json_dumps(
        {
            "base_document_indices":
                document_indices,

            "within_document_starts":
                within_document_starts,

            "counterpart_document_indices":
                document_indices,

            "pairing_rule":
                "counterpart_index_equals_base_index",

            "base_count":
                BASES_PER_BATCH,

            "counterpart_count":
                PAIRS_PER_BATCH,
        }
    ).encode("ascii")


# =============================================================================
# READ EXACTLY ONE NAMED JSON VALUE
#
# The frozen corpus JSON also contains sealed axes.
#
# We need supported_relation_withheld_geometry because it is an ALLOWED
# positive-control axis.
#
# json.load() would deserialize the entire corpus, including sealed axes.
# We deliberately do NOT do that here.
#
# This helper reads the file as text, locates one exact top-level key,
# and asks JSONDecoder to deserialize ONLY that key's value.
#
# No sealed axis is deserialized or evaluated.
# =============================================================================

def load_named_json_value(
    path: Path,
    key: str,
):
    text = path.read_text(
        encoding="utf-8",
    )

    marker = json.dumps(key)

    position = text.find(marker)

    if position < 0:
        raise KeyError(
            f"Could not find JSON key: {key}"
        )

    colon = text.find(
        ":",
        position + len(marker),
    )

    if colon < 0:
        raise RuntimeError(
            f"Malformed JSON near key: {key}"
        )

    value_start = colon + 1

    while (
        value_start < len(text)
        and text[value_start].isspace()
    ):
        value_start += 1

    decoder = json.JSONDecoder()

    value, _end = decoder.raw_decode(
        text,
        value_start,
    )

    return value


# =============================================================================
# SAFETY
# =============================================================================

def safety_check() -> None:

    print_header("SAFETY")

    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(SOURCE_ROOT)

    if not START_CHECKPOINT.is_file():
        raise FileNotFoundError(
            START_CHECKPOINT
        )

    source = SOURCE_ROOT.resolve()
    output = OUTPUT_ROOT.resolve()

    try:
        output.relative_to(source)

        raise RuntimeError(
            "REFUSING TO RUN: output directory "
            "is inside protected DaveLM-v0.9."
        )

    except ValueError:
        pass

    actual_checkpoint_sha = file_sha256(
        START_CHECKPOINT
    )

    print(
        f"Protected source: {SOURCE_ROOT}"
    )

    print(
        f"Disposable output: {OUTPUT_ROOT}"
    )

    print()

    print(
        "Mode: PREFLIGHT ONLY"
    )

    print(
        "Model construction: NO"
    )

    print(
        "Optimizer construction: NO"
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

    print("Starting checkpoint SHA256:")
    print(actual_checkpoint_sha)

    if (
        actual_checkpoint_sha
        != EXPECTED_START_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "Successful one-map checkpoint SHA mismatch."
        )

    print()
    print("Starting checkpoint SHA: PASS")


# =============================================================================
# RECONSTRUCT ORIGINAL TRAINING CORPUS
# =============================================================================

def reconstruct_original():

    print_header(
        "RECONSTRUCTING ORIGINAL BALANCED TWO-MAP TRAINING CORPUS"
    )

    tokenizer = Tokenizer.from_file(
        str(config.TOKENIZER_PATH)
    )

    groups = _identity_pool(
        tokenizer
    )

    identities = groups["identity"]
    filler = groups["filler"]

    if len(identities) != 64:
        raise RuntimeError(
            "Identity pool changed."
        )

    if len(filler) != 64:
        raise RuntimeError(
            "Filler pool changed."
        )

    tables = _layout_tables(
        tokenizer,
        groups,
    )

    geometry_schedule = _geometry_schedule(
        tables
    )

    keys = _ordered_training_keys()

    original = _training_records(
        tokenizer,
        identities,
        filler,
        tables,
    )

    if len(original) != EXPECTED_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORDS} training records; "
            f"got {len(original)}."
        )

    if len(geometry_schedule) != EXPECTED_RECORDS:
        raise RuntimeError(
            "Geometry schedule count changed."
        )

    if len(keys) != EXPECTED_RECORDS:
        raise RuntimeError(
            "Ordered training key count changed."
        )

    slot_counts = Counter(
        int(row["query_slot"])
        for row in original
    )

    if slot_counts != Counter(
        {
            0: 768,
            1: 768,
        }
    ):
        raise RuntimeError(
            f"Original query-slot balance changed: {slot_counts}"
        )

    document_lengths = Counter(
        int(row["document_token_count"])
        for row in original
    )

    if document_lengths != Counter(
        {
            EXPECTED_DOCUMENT_TOKENS:
                EXPECTED_RECORDS
        }
    ):
        raise RuntimeError(
            f"Unexpected original document lengths: "
            f"{document_lengths}"
        )

    print("Identity pool:       64")
    print("Filler pool:         64")
    print("Training records:    1536")
    print("Original query slot: 768 / 768")
    print("Document length:     192 × 1536")

    return (
        tokenizer,
        identities,
        filler,
        geometry_schedule,
        original,
    )


# =============================================================================
# BUILD EXACT QUERY-SWAPPED COUNTERPARTS
# =============================================================================

def build_counterparts(
    tokenizer,
    identities,
    filler,
    geometry_schedule,
    original,
):

    print_header(
        "BUILDING 1,536 QUERY-SWAPPED COUNTERPARTS"
    )

    counterparts = []

    slot_flips = Counter()
    distance_deltas = Counter()

    for index, base in enumerate(original):

        geometry = geometry_schedule[index]

        base_slot = int(
            base["query_slot"]
        )

        counterpart_slot = (
            1 - base_slot
        )

        base_mappings = mapping_pairs(
            base
        )

        if len(base_mappings) != 2:
            raise RuntimeError(
                f"Record {index}: expected exactly two mappings."
            )

        # ---------------------------------------------------------
        # SAME PHYSICAL DOCUMENT GEOMETRY.
        #
        # We keep prefix_count and between_count exactly frozen.
        #
        # We deliberately disable the OLD target-distance enforcement,
        # because the correct target moves to the other displayed
        # mapping when query_slot flips.
        #
        # _record() will compute the ACTUAL resulting distance.
        # ---------------------------------------------------------

        counterpart_geometry = {
            "query_slot":
                counterpart_slot,

            "query_index":
                int(
                    geometry["query_index"]
                ),

            "target_to_query_distance":
                int(
                    geometry[
                        "target_to_query_distance"
                    ]
                ),

            "prefix_count":
                int(
                    geometry["prefix_count"]
                ),

            "between_count":
                int(
                    geometry["between_count"]
                ),

            "enforce":
                False,
        }

        counterpart = _record(
            tokenizer,
            "train",
            index,
            identities,
            filler,
            base_mappings,
            counterpart_slot,
            counterpart_geometry,
            training=True,
        )

        # =========================================================
        # HARD INVARIANTS
        # =========================================================

        if relation_set(base) != relation_set(counterpart):
            raise RuntimeError(
                f"Record {index}: relation set changed."
            )

        if mapping_pairs(base) != mapping_pairs(counterpart):
            raise RuntimeError(
                f"Record {index}: mapping display order changed."
            )

        if (
            int(base["query_slot"])
            ==
            int(counterpart["query_slot"])
        ):
            raise RuntimeError(
                f"Record {index}: query_slot did not flip."
            )

        if (
            int(counterpart["query_slot"])
            != 1 - int(base["query_slot"])
        ):
            raise RuntimeError(
                f"Record {index}: invalid query_slot flip."
            )

        if (
            int(
                base["layout"]["query_index"]
            )
            !=
            int(
                counterpart[
                    "layout"
                ]["query_index"]
            )
        ):
            raise RuntimeError(
                f"Record {index}: physical query_index moved."
            )

        if (
            int(base["document_token_count"])
            !=
            int(counterpart["document_token_count"])
        ):
            raise RuntimeError(
                f"Record {index}: document length changed."
            )

        if (
            int(counterpart["document_token_count"])
            != EXPECTED_DOCUMENT_TOKENS
        ):
            raise RuntimeError(
                f"Record {index}: counterpart is not 192 tokens."
            )

        # ---------------------------------------------------------
        # The counterpart's answer must become EXACTLY the other
        # displayed mapping.
        # ---------------------------------------------------------

        expected_other = base_mappings[
            counterpart_slot
        ]

        counterpart_answer = answer_relation(
            counterpart
        )

        if counterpart_answer != expected_other:
            raise RuntimeError(
                f"Record {index}: counterpart did not select "
                f"the other mapping. "
                f"Expected {expected_other}; "
                f"got {counterpart_answer}."
            )

        if (
            answer_relation(base)
            ==
            answer_relation(counterpart)
        ):
            raise RuntimeError(
                f"Record {index}: answer relation did not change."
            )

        # ---------------------------------------------------------
        # The original answer should now be the counterpart's
        # distractor.
        # ---------------------------------------------------------

        counterpart_distractor = (
            int(
                counterpart[
                    "distractor_source_index"
                ]
            ),
            int(
                counterpart[
                    "distractor_target_index"
                ]
            ),
        )

        if (
            counterpart_distractor
            != answer_relation(base)
        ):
            raise RuntimeError(
                f"Record {index}: original answer did not "
                f"become counterpart distractor."
            )

        # ---------------------------------------------------------
        # Frozen low-level geometry controls.
        # ---------------------------------------------------------

        if (
            int(geometry["prefix_count"])
            !=
            int(counterpart_geometry["prefix_count"])
        ):
            raise RuntimeError(
                f"Record {index}: prefix_count changed."
            )

        if (
            int(geometry["between_count"])
            !=
            int(counterpart_geometry["between_count"])
        ):
            raise RuntimeError(
                f"Record {index}: between_count changed."
            )

        # ---------------------------------------------------------
        # Record ACTUAL resulting target-distance difference.
        # ---------------------------------------------------------

        base_distance = int(
            base["layout"][
                "target_to_query_distance"
            ]
        )

        counterpart_distance = int(
            counterpart["layout"][
                "target_to_query_distance"
            ]
        )

        distance_deltas[
            counterpart_distance
            - base_distance
        ] += 1

        slot_flips[
            (
                base_slot,
                counterpart_slot,
            )
        ] += 1

        counterparts.append(
            counterpart
        )

    if len(counterparts) != EXPECTED_RECORDS:
        raise RuntimeError(
            "Did not construct exactly 1,536 counterparts."
        )

    expected_flips = Counter(
        {
            (0, 1): 768,
            (1, 0): 768,
        }
    )

    if slot_flips != expected_flips:
        raise RuntimeError(
            f"Unexpected slot flips: {slot_flips}"
        )

    print("Counterparts constructed: 1536")
    print("0 → 1 query-slot flips:   768")
    print("1 → 0 query-slot flips:   768")
    print("Physical query moves:     0")
    print("Relation-set changes:     0")
    print("Mapping-order changes:    0")
    print("Document-length changes:  0")

    print()
    print("Observed target-distance deltas:")

    for delta, count in sorted(
        distance_deltas.items()
    ):
        print(
            f"  {delta:+d}: {count}"
        )

    print()
    print("COUNTERPART CONSTRUCTION: PASS")

    return counterparts, distance_deltas


# =============================================================================
# LOAD ALLOWED POSITIVE-CONTROL AXIS ONLY
# =============================================================================

def load_supported_positive_control():

    print_header(
        "LOADING SUPPORTED-WITHHELD POSITIVE CONTROL ONLY"
    )

    if not config.CORPUS_PATH.is_file():
        raise FileNotFoundError(
            config.CORPUS_PATH
        )

    supported = load_named_json_value(
        config.CORPUS_PATH,
        "supported_relation_withheld_geometry",
    )

    if not isinstance(
        supported,
        list,
    ):
        raise RuntimeError(
            "Supported positive-control material "
            "did not decode as a list."
        )

    print(
        "Loaded axis: "
        "supported_relation_withheld_geometry"
    )

    print(
        f"Examples: {len(supported)}"
    )

    print(
        "Sealed axes deserialized: NONE"
    )

    return supported


# =============================================================================
# LEAKAGE AUDIT
# =============================================================================

def audit_supported_leakage(
    counterparts,
    supported,
):

    print_header(
        "AUDITING ACTUAL COUNTERPART GEOMETRY FOR POSITIVE-CONTROL LEAKAGE"
    )

    counterpart_exposures = {
        exposure_tuple(row)
        for row in counterparts
    }

    supported_exposures = {
        exposure_tuple(row)
        for row in supported
    }

    overlap = sorted(
        counterpart_exposures
        & supported_exposures
    )

    print(
        "Counterpart answer-bearing "
        f"relation×geometry tuples: "
        f"{len(counterpart_exposures)}"
    )

    print(
        "Supported-withheld "
        f"relation×geometry tuples: "
        f"{len(supported_exposures)}"
    )

    print()

    print(
        "CONSERVATIVE INTERSECTION COUNT:"
    )

    print(
        len(overlap)
    )

    if overlap:
        print()
        print(
            "LEAKAGE DETECTED."
        )

        print()
        print(
            "First overlapping tuples:"
        )

        for item in overlap[:20]:
            print(
                "  "
                f"source={item[0]} "
                f"target={item[1]} "
                f"query_index={item[2]} "
                f"target_distance={item[3]}"
            )

        raise RuntimeError(
            "STOP: paired query-swap treatment "
            "would expose supported-withheld "
            "positive-control geometry. "
            "NO TRAINING AUTHORIZED."
        )

    print()
    print(
        "SUPPORTED-WITHHELD OVERLAP: ZERO"
    )

    print()
    print(
        "LEAKAGE AUDIT: PASS"
    )

    return {
        "counterpart_exposure_count":
            len(counterpart_exposures),

        "supported_exposure_count":
            len(supported_exposures),

        "intersection_count":
            0,
    }


# =============================================================================
# BUILD DOCUMENT STORES
# =============================================================================

def build_stores(
    tokenizer,
    original,
    counterparts,
):

    print_header(
        "BUILDING BASE + COUNTERPART DOCUMENT STORES"
    )

    base_store = (
        DocumentTokenStore
        .from_documents(
            tokenizer,
            original,
        )
    )

    counterpart_store = (
        DocumentTokenStore
        .from_documents(
            tokenizer,
            counterparts,
        )
    )

    print(
        "Base store:        1536 documents"
    )

    print(
        "Counterpart store: 1536 documents"
    )

    print(
        "Every document:    192 tokens"
    )

    return (
        base_store,
        counterpart_store,
    )


# =============================================================================
# NEW PAIRED MINIBATCH SCHEDULE
#
# IMPORTANT:
#
# This is NOT the historical a34bb... schedule.
#
# Treatment #2 deliberately changes the minibatch construction:
#
#   sample 16 base documents
#   pair each with counterpart of SAME document index
#
# The resulting batch has 32 documents.
#
# We freeze this NEW schedule before training.
# =============================================================================

def build_paired_schedule_digest(
    base_store,
):

    generator = (
        torch.Generator()
        .manual_seed(
            MINIBATCH_SEED
        )
    )

    digest = hashlib.sha256()

    total_base_supervised_tokens = 0

    for _step in range(
        1,
        TOTAL_STEPS + 1,
    ):

        (
            _inputs,
            _targets,
            _mask,
            meta,
        ) = sample_document_batch(
            base_store,
            BASES_PER_BATCH,
            CONTEXT_SIZE,
            generator,
            torch.device("cpu"),
        )

        document_indices = [
            int(value)
            for value
            in meta["document_indices"]
        ]

        within_document_starts = [
            int(value)
            for value
            in meta[
                "within_document_starts"
            ]
        ]

        if (
            len(document_indices)
            != BASES_PER_BATCH
        ):
            raise RuntimeError(
                "Base batch did not contain 16 documents."
            )

        if (
            len(within_document_starts)
            != BASES_PER_BATCH
        ):
            raise RuntimeError(
                "Base batch did not contain 16 starts."
            )

        digest.update(
            schedule_payload(
                document_indices,
                within_document_starts,
            )
        )

        total_base_supervised_tokens += int(
            meta["supervised_tokens"]
        )

    # Counterparts have the same document length and use the
    # same starts. Therefore their supervised causal-token count
    # is exactly equal to the base half.
    total_paired_supervised_tokens = (
        total_base_supervised_tokens * 2
    )

    return {
        "sha256":
            digest.hexdigest(),

        "base_supervised_tokens":
            total_base_supervised_tokens,

        "counterpart_supervised_tokens":
            total_base_supervised_tokens,

        "total_supervised_tokens":
            total_paired_supervised_tokens,
    }


def verify_new_schedule(
    base_store,
):

    print_header(
        "FREEZING NEW 1,000-STEP PAIRED MINIBATCH SCHEDULE"
    )

    first = build_paired_schedule_digest(
        base_store
    )

    second = build_paired_schedule_digest(
        base_store
    )

    print(
        "First construction SHA256:"
    )

    print(
        first["sha256"]
    )

    print()

    print(
        "Second construction SHA256:"
    )

    print(
        second["sha256"]
    )

    print()

    if first["sha256"] != second["sha256"]:
        raise RuntimeError(
            "Paired schedule is not deterministic."
        )

    if (
        first["total_supervised_tokens"]
        != second["total_supervised_tokens"]
    ):
        raise RuntimeError(
            "Supervised-token accounting "
            "was not deterministic."
        )

    print(
        "Schedule deterministic: PASS"
    )

    print(
        "Generator restart:      NO"
    )

    print(
        "Steps:                  1000"
    )

    print(
        "Base docs / step:       16"
    )

    print(
        "Counterparts / step:    16"
    )

    print(
        "Total docs / step:      32"
    )

    print()

    print(
        "Base supervised causal tokens:"
    )

    print(
        first[
            "base_supervised_tokens"
        ]
    )

    print()

    print(
        "Counterpart supervised causal tokens:"
    )

    print(
        first[
            "counterpart_supervised_tokens"
        ]
    )

    print()

    print(
        "Total paired supervised causal tokens:"
    )

    print(
        first[
            "total_supervised_tokens"
        ]
    )

    print()

    print(
        "NEW PAIRED SCHEDULE SHA256:"
    )

    print(
        first["sha256"]
    )

    print()

    print(
        "PAIRED SCHEDULE: PASS"
    )

    return first


# =============================================================================
# MAIN
# =============================================================================

def main():

    safety_check()

    (
        tokenizer,
        identities,
        filler,
        geometry_schedule,
        original,
    ) = reconstruct_original()

    (
        counterparts,
        distance_deltas,
    ) = build_counterparts(
        tokenizer,
        identities,
        filler,
        geometry_schedule,
        original,
    )

    supported = (
        load_supported_positive_control()
    )

    leakage = audit_supported_leakage(
        counterparts,
        supported,
    )

    (
        base_store,
        counterpart_store,
    ) = build_stores(
        tokenizer,
        original,
        counterparts,
    )

    # counterpart_store is intentionally constructed here as
    # an integrity check. Actual training will use it later.
    _ = counterpart_store

    schedule = verify_new_schedule(
        base_store
    )

    print_header(
        "TREATMENT #2 PREFLIGHT SUMMARY"
    )

    print(
        "Starting one-map checkpoint:     PASS"
    )

    print(
        "Original training corpus:        PASS"
    )

    print(
        "Counterparts constructed:        1536 / 1536"
    )

    print(
        "Query-slot flips:                1536 / 1536"
    )

    print(
        "Physical query-index changes:    0"
    )

    print(
        "Mapping-order changes:           0"
    )

    print(
        "Relation-set changes:            0"
    )

    print(
        "Document-length changes:         0"
    )

    print(
        "Supported-withheld overlap:      0"
    )

    print(
        "New paired schedule deterministic: PASS"
    )

    print()

    print(
        "NEW FROZEN SCHEDULE SHA256:"
    )

    print(
        schedule["sha256"]
    )

    print()

    print(
        "SEALED EVALUATION OPENED: NO"
    )

    print(
        "MODEL BUILT: NO"
    )

    print(
        "OPTIMIZER BUILT: NO"
    )

    print(
        "BACKWARD PASSES: ZERO"
    )

    print(
        "OPTIMIZER STEPS: ZERO"
    )

    payload = {
        "passed":
            True,

        "experiment":
            "DaveLM v0.9 Treatment #2 paired query-swap preflight",

        "starting_checkpoint":
            str(START_CHECKPOINT),

        "starting_checkpoint_sha256":
            EXPECTED_START_CHECKPOINT_SHA256,

        "training_records":
            EXPECTED_RECORDS,

        "counterparts":
            len(counterparts),

        "counterpart_invariants": {
            "query_slot_flips":
                EXPECTED_RECORDS,

            "physical_query_index_changes":
                0,

            "mapping_order_changes":
                0,

            "relation_set_changes":
                0,

            "document_length_changes":
                0,

            "target_distance_deltas":
                {
                    str(delta): count
                    for delta, count
                    in sorted(
                        distance_deltas.items()
                    )
                },
        },

        "positive_control_leakage_audit":
            leakage,

        "paired_schedule": {
            "seed":
                MINIBATCH_SEED,

            "steps":
                TOTAL_STEPS,

            "base_documents_per_step":
                BASES_PER_BATCH,

            "counterparts_per_step":
                PAIRS_PER_BATCH,

            "total_documents_per_step":
                TOTAL_BATCH_SIZE,

            "sha256":
                schedule["sha256"],

            "base_supervised_tokens":
                schedule[
                    "base_supervised_tokens"
                ],

            "counterpart_supervised_tokens":
                schedule[
                    "counterpart_supervised_tokens"
                ],

            "total_supervised_tokens":
                schedule[
                    "total_supervised_tokens"
                ],
        },

        "sealed_evaluation_opened":
            False,

        "model_built":
            False,

        "optimizer_built":
            False,

        "backward_passes":
            0,

        "optimizer_steps":
            0,
    }

    save_json(
        PREFLIGHT_PATH,
        payload,
    )

    print()
    print(
        "PREFLIGHT RESULT: PASS"
    )

    print()

    print(
        "NO TRAINING OCCURRED."
    )

    print()

    print(
        "Paste this output back into ChatGPT "
        "before we build the actual Treatment #2 trainer."
    )


if __name__ == "__main__":
    main()