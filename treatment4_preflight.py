from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer


# =============================================================================
# DaveLM v0.9 — Treatment #4
# READ-ONLY PREFLIGHT
#
# TREATMENT:
#
# At eligible answer positions ONLY, replace the historical full-vocabulary
# correct-token CE loss with:
#
#   L_answer =
#       -log(p_correct + p_distractor)
#       + ReLU(0.5 - (z_correct - z_distractor))
#
# The candidate-membership term is to be implemented stably as:
#
#   logsumexp(all_logits)
#       - logsumexp([z_correct, z_distractor])
#
# Every other supervised causal token retains ordinary full-vocabulary CE.
#
# CRITICAL:
#
# This is PER-TOKEN SUBSTITUTION.
#
# It is NOT:
#   historical document CE + answer auxiliary loss
#
# It is NOT:
#   mean(non-answer CE) + mean(answer loss)
#
# It is NOT:
#   original answer CE + replacement answer loss
#
# Historical supervised-token reduction/count must remain unchanged.
#
# THIS SCRIPT DOES NOT:
# - build a model
# - load a checkpoint into a model
# - create an optimizer
# - calculate gradients
# - call backward()
# - train
# - evaluate positive controls
# - deserialize sealed axes
# - perform sealed evaluation
# =============================================================================


# =============================================================================
# FROZEN CONTRACT
# =============================================================================

SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")

OUTPUT_ROOT = Path(
    r"C:\DaveLM-CADAVER\two_mapping_membership_margin_manual_seed8380"
)

PREFLIGHT_PATH = (
    OUTPUT_ROOT
    / "treatment4_preflight.json"
)

EXPECTED_START_CHECKPOINT = Path(
    r"C:\DaveLM-v0.9\experiments"
    r"\minimal_contextual_binding"
    r"\checkpoints"
    r"\treatment_one_mapping"
    r"\seed_8380"
    r"\latest.pt"
)

EXPECTED_START_SHA256 = (
    "345984c52a06db5f988aaf4cd47963ee"
    "a0e9d77489cee94dbb10816af2f5443e"
)

EXPECTED_SCHEDULE_SHA256 = (
    "a34bb91d258f11c6f7cb6b6cd89c79d"
    "8e6fa12d64af15dd14b028fb7ff675cff"
)

SEED = 8380
SCHEDULE_SEED = 8381

MAX_STEPS = 1000
BATCH_SIZE = 32
CONTEXT_SIZE = 256

MARGIN = 0.5

EXPECTED_TRAINING_RECORDS = 1536
EXPECTED_RELATIONS = 1536

EXPECTED_EVENTS = 32_000

EXPECTED_SUPERVISED_TOKENS = 6_176_000
EXPECTED_ANSWER_SUBSTITUTIONS = 32_000
EXPECTED_NONANSWER_TOKENS = (
    EXPECTED_SUPERVISED_TOKENS
    - EXPECTED_ANSWER_SUBSTITUTIONS
)


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


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

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

    if value is None or not callable(
        value
    ):

        raise RuntimeError(
            f"Required v0.9 function {name!r} is unavailable."
        )

    return value


# =============================================================================
# SAFETY
# =============================================================================

def safety_check():

    header(
        "TREATMENT #4 PREFLIGHT SAFETY"
    )

    if not SOURCE_ROOT.is_dir():

        raise FileNotFoundError(
            SOURCE_ROOT
        )

    if not EXPECTED_START_CHECKPOINT.is_file():

        raise FileNotFoundError(
            EXPECTED_START_CHECKPOINT
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
            "REFUSING TO RUN: treatment output "
            "is inside protected C:\\DaveLM-v0.9."
        )

    except ValueError:
        pass

    observed_checkpoint_sha = (
        sha256_file(
            EXPECTED_START_CHECKPOINT
        )
    )

    if (
        observed_checkpoint_sha
        != EXPECTED_START_SHA256
    ):

        raise RuntimeError(
            "\n"
            "START CHECKPOINT SHA MISMATCH.\n"
            f"Expected: {EXPECTED_START_SHA256}\n"
            f"Observed: {observed_checkpoint_sha}\n"
        )

    print(
        f"Protected source: {SOURCE_ROOT}"
    )

    print(
        f"Treatment output: {OUTPUT_ROOT}"
    )

    print()

    print(
        "Starting checkpoint SHA: PASS"
    )

    print(
        f"Frozen answer margin: {MARGIN}"
    )

    print()

    print(
        "Mode: PREFLIGHT ONLY"
    )

    print(
        "Model construction: NO"
    )

    print(
        "Checkpoint model_state load: NO"
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

    return observed_checkpoint_sha


# =============================================================================
# RECONSTRUCT ORIGINAL BALANCED TWO-MAP TRAINING CORPUS
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

    if len(records) != EXPECTED_TRAINING_RECORDS:

        raise RuntimeError(
            f"Expected {EXPECTED_TRAINING_RECORDS} "
            f"training records; got {len(records)}."
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
# RECORD-LEVEL CONTRACT
# =============================================================================

def validate_records(
    tokenizer,
    records,
):

    header(
        "VALIDATING TREATMENT #4 ANSWER METADATA"
    )

    bos_id = original_run.required_token_id(
        tokenizer,
        "<bos>",
    )

    vocab_size = tokenizer.get_vocab_size()

    queried_relations = set()

    slot_counts = Counter()
    prefix_lengths = Counter()

    for index, row in enumerate(
        records
    ):

        required = (
            "prompt",
            "target_token_id",
            "distractor_token_id",
            "query_slot",
            "layout",
        )

        for key in required:

            if key not in row:

                raise RuntimeError(
                    f"Record {index}: missing {key!r}."
                )

        target = int(
            row[
                "target_token_id"
            ]
        )

        distractor = int(
            row[
                "distractor_token_id"
            ]
        )

        if target == distractor:

            raise RuntimeError(
                f"Record {index}: target equals distractor."
            )

        if not (
            0 <= target < vocab_size
        ):

            raise RuntimeError(
                f"Record {index}: invalid target token {target}."
            )

        if not (
            0 <= distractor < vocab_size
        ):

            raise RuntimeError(
                f"Record {index}: invalid distractor token {distractor}."
            )

        layout = row[
            "layout"
        ]

        if "prefix_ids" not in layout:

            raise RuntimeError(
                f"Record {index}: layout.prefix_ids missing."
            )

        audited_prefix = [
            int(x)
            for x in layout[
                "prefix_ids"
            ]
        ]

        native_prefix = (
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
            != native_prefix
        ):

            raise RuntimeError(
                f"Record {index}: native evaluator prefix mismatch."
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
                f"Record {index}: invalid query slot {query_slot}."
            )

        slot_counts[
            query_slot
        ] += 1

        prefix_lengths[
            len(
                audited_prefix
            )
        ] += 1

        # The historical corpus uses one queried relation per training record.
        # Record IDs are retained as the conservative unique-relation identity
        # if no explicit relation_id field exists.
        if "relation_id" in row:

            relation_key = str(
                row[
                    "relation_id"
                ]
            )

        elif "id" in row:

            relation_key = str(
                row[
                    "id"
                ]
            )

        else:

            relation_key = (
                f"{target}:"
                f"{distractor}:"
                f"{query_slot}:"
                f"{index}"
            )

        queried_relations.add(
            relation_key
        )

    if (
        len(queried_relations)
        != EXPECTED_RELATIONS
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_RELATIONS} unique queried relations; "
            f"got {len(queried_relations)}."
        )

    print(
        f"Tokenizer vocab size:     {vocab_size}"
    )

    print(
        f"Records validated:        {len(records)}"
    )

    print(
        f"Unique queried relations: {len(queried_relations)}"
    )

    print(
        f"Query slot 0:             {slot_counts[0]}"
    )

    print(
        f"Query slot 1:             {slot_counts[1]}"
    )

    print(
        "Correct/distractor IDs:  PASS"
    )

    print(
        "Native prompt prefixes:  PASS"
    )

    return {
        "vocab_size":
            vocab_size,

        "queried_relations":
            len(
                queried_relations
            ),

        "slot_counts": {
            "0":
                slot_counts[
                    0
                ],

            "1":
                slot_counts[
                    1
                ],
        },

        "prefix_lengths": {
            str(length):
                count

            for length, count
            in sorted(
                prefix_lengths.items()
            )
        },
    }


# =============================================================================
# BUILD ORIGINAL STORE
# =============================================================================

def build_store(
    tokenizer,
    records,
):

    header(
        "BUILDING ORIGINAL HISTORICAL DOCUMENT STORE"
    )

    store = (
        DocumentTokenStore
        .from_documents(
            tokenizer,
            records,
        )
    )

    print(
        "Training store: PASS"
    )

    return store


# =============================================================================
# PREFLIGHT THE EXACT HISTORICAL SCHEDULE
#
# For every sampled event:
#
#   answer_global_index = len(layout.prefix_ids)
#
#   p = answer_global_index - start - 1
#
# target[p] must equal target_token_id.
#
# logits[p] sees:
#
#   inputs[:p + 1]
#
# This must equal layout.prefix_ids for exact native-evaluator context identity.
#
# We also count the historical supervised mask directly so Treatment #4 can
# prove that exactly one existing supervised answer-token loss is substituted
# per event while the denominator remains unchanged.
# =============================================================================

def audit_schedule(
    store,
    records,
):

    header(
        "AUDITING HISTORICAL SCHEDULE AND PER-TOKEN SUBSTITUTION MAP"
    )

    generator = (
        torch.Generator()
        .manual_seed(
            SCHEDULE_SEED
        )
    )

    digest = hashlib.sha256()

    total_events = 0
    total_supervised_tokens = 0

    answer_substitutions = 0
    answer_target_matches = 0
    answer_target_mismatches = 0

    prefix_matches = 0
    prefix_mismatches = 0

    duplicate_substitutions = 0
    missing_substitutions = 0

    start_counts = Counter()
    slot_counts = Counter()
    relation_counts = Counter()

    causal_position_counts = Counter()

    mismatch_examples = []

    # One key per sampled event. The key includes step and batch position so
    # repeated sampling of the same training relation remains distinct.
    substitution_keys = set()

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        (
            inputs,
            targets,
            mask,
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

        if len(document_indices) != BATCH_SIZE:

            raise RuntimeError(
                f"Step {step}: document index count != {BATCH_SIZE}."
            )

        if len(starts) != BATCH_SIZE:

            raise RuntimeError(
                f"Step {step}: start count != {BATCH_SIZE}."
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

        # Historical supervised token denominator.
        #
        # We intentionally derive this from the same mask returned by the
        # historical sampler instead of inventing a new denominator.
        if mask is None:

            # Some historical sampler implementations encode ignored targets
            # directly rather than returning a separate mask. In that case
            # derive the supervised count using the same convention expected
            # by document_causal_loss: targets >= 0 are supervised.
            supervised_this_batch = int(
                (
                    targets >= 0
                ).sum().item()
            )

        else:

            supervised_this_batch = int(
                mask.to(
                    dtype=torch.bool
                ).sum().item()
            )

        total_supervised_tokens += (
            supervised_this_batch
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

            row = records[
                document_index
            ]

            query_slot = int(
                row[
                    "query_slot"
                ]
            )

            slot_counts[
                query_slot
            ] += 1

            start_counts[
                start
            ] += 1

            relation_counts[
                document_index
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

                missing_substitutions += 1

                if len(
                    mismatch_examples
                ) < 20:

                    mismatch_examples.append(
                        {
                            "type":
                                "answer_outside_sample",

                            "step":
                                step,

                            "batch_index":
                                batch_index,

                            "document_index":
                                document_index,

                            "start":
                                start,

                            "causal_position":
                                causal_position,
                        }
                    )

                continue

            # -------------------------------------------------------------
            # Confirm this answer slot is supervised under the historical
            # mask / target convention.
            # -------------------------------------------------------------

            if mask is None:

                answer_is_supervised = (
                    int(
                        targets[
                            batch_index,
                            causal_position,
                        ].item()
                    )
                    >= 0
                )

            else:

                answer_is_supervised = bool(
                    mask[
                        batch_index,
                        causal_position,
                    ].item()
                )

            if not answer_is_supervised:

                missing_substitutions += 1

                if len(
                    mismatch_examples
                ) < 20:

                    mismatch_examples.append(
                        {
                            "type":
                                "answer_not_supervised",

                            "step":
                                step,

                            "batch_index":
                                batch_index,

                            "document_index":
                                document_index,

                            "causal_position":
                                causal_position,
                        }
                    )

                continue

            expected_target = int(
                row[
                    "target_token_id"
                ]
            )

            actual_target = int(
                targets[
                    batch_index,
                    causal_position,
                ].item()
            )

            if (
                actual_target
                == expected_target
            ):

                answer_target_matches += 1

            else:

                answer_target_mismatches += 1

                if len(
                    mismatch_examples
                ) < 20:

                    mismatch_examples.append(
                        {
                            "type":
                                "target_mismatch",

                            "step":
                                step,

                            "batch_index":
                                batch_index,

                            "document_index":
                                document_index,

                            "expected_target":
                                expected_target,

                            "actual_target":
                                actual_target,
                        }
                    )

            sampled_prefix = [
                int(x)
                for x in inputs[
                    batch_index,
                    :causal_position + 1,
                ].tolist()
            ]

            if (
                start == 0
                and sampled_prefix
                == audited_prefix
            ):

                prefix_matches += 1

            else:

                prefix_mismatches += 1

                if len(
                    mismatch_examples
                ) < 20:

                    mismatch_examples.append(
                        {
                            "type":
                                "native_prefix_mismatch",

                            "step":
                                step,

                            "batch_index":
                                batch_index,

                            "document_index":
                                document_index,

                            "start":
                                start,

                            "causal_position":
                                causal_position,

                            "sampled_prefix_length":
                                len(
                                    sampled_prefix
                                ),

                            "audited_prefix_length":
                                len(
                                    audited_prefix
                                ),
                        }
                    )

            substitution_key = (
                step,
                batch_index,
            )

            if (
                substitution_key
                in substitution_keys
            ):

                duplicate_substitutions += 1

            else:

                substitution_keys.add(
                    substitution_key
                )

                answer_substitutions += 1

    observed_sha = (
        digest.hexdigest()
    )

    nonanswer_supervised_tokens = (
        total_supervised_tokens
        - answer_substitutions
    )

    print(
        "Expected historical schedule SHA256:"
    )

    print(
        EXPECTED_SCHEDULE_SHA256
    )

    print()

    print(
        "Observed historical schedule SHA256:"
    )

    print(
        observed_sha
    )

    print()

    schedule_pass = (
        observed_sha
        == EXPECTED_SCHEDULE_SHA256
    )

    print(
        "Historical schedule SHA:",
        "PASS" if schedule_pass else "FAIL",
    )

    print()

    print(
        f"Sampled events:                {total_events}"
    )

    print(
        f"Eligible answer substitutions: {answer_substitutions}"
    )

    print(
        f"Missing substitutions:         {missing_substitutions}"
    )

    print(
        f"Duplicate substitutions:       {duplicate_substitutions}"
    )

    print()

    print(
        f"Supervised tokens historical:  {total_supervised_tokens}"
    )

    print(
        f"Expected supervised tokens:    {EXPECTED_SUPERVISED_TOKENS}"
    )

    print(
        f"Answer-token slots replaced:   {answer_substitutions}"
    )

    print(
        f"Non-answer supervised tokens:  {nonanswer_supervised_tokens}"
    )

    print(
        f"Expected non-answer tokens:    {EXPECTED_NONANSWER_TOKENS}"
    )

    print()

    print(
        f"Answer target matches:          {answer_target_matches}"
    )

    print(
        f"Answer target mismatches:       {answer_target_mismatches}"
    )

    print()

    print(
        f"Native prefix exact matches:    {prefix_matches}"
    )

    print(
        f"Native prefix mismatches:       {prefix_mismatches}"
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
        "Sampled query slots:"
    )

    print(
        f"  slot 0: {slot_counts[0]}"
    )

    print(
        f"  slot 1: {slot_counts[1]}"
    )

    print()

    print(
        f"Unique sampled relations:       {len(relation_counts)}"
    )

    if relation_counts:

        values = list(
            relation_counts.values()
        )

        print(
            f"Relation event count min/max:   "
            f"{min(values)} / {max(values)}"
        )

        print(
            f"Relation event count mean:      "
            f"{sum(values) / len(values):.3f}"
        )

    print()

    print(
        "Causal answer position range:"
    )

    print(
        f"  min: {min(causal_position_counts)}"
    )

    print(
        f"  max: {max(causal_position_counts)}"
    )

    # ---------------------------------------------------------------------
    # Frozen hard gates.
    # ---------------------------------------------------------------------

    checks = {
        "schedule_sha_exact":
            schedule_pass,

        "event_count_exact":
            total_events
            == EXPECTED_EVENTS,

        "answer_substitution_count_exact":
            answer_substitutions
            == EXPECTED_ANSWER_SUBSTITUTIONS,

        "missing_substitutions_zero":
            missing_substitutions
            == 0,

        "duplicate_substitutions_zero":
            duplicate_substitutions
            == 0,

        "answer_targets_exact":
            answer_target_matches
            == EXPECTED_EVENTS
            and answer_target_mismatches
            == 0,

        "native_prefix_identity_exact":
            prefix_matches
            == EXPECTED_EVENTS
            and prefix_mismatches
            == 0,

        "supervised_token_count_preserved":
            total_supervised_tokens
            == EXPECTED_SUPERVISED_TOKENS,

        "nonanswer_token_count_exact":
            nonanswer_supervised_tokens
            == EXPECTED_NONANSWER_TOKENS,

        "all_relations_sampled":
            len(
                relation_counts
            )
            == EXPECTED_RELATIONS,
    }

    all_pass = all(
        checks.values()
    )

    return {
        "observed_schedule_sha256":
            observed_sha,

        "checks":
            checks,

        "all_pass":
            all_pass,

        "sampled_events":
            total_events,

        "answer_substitutions":
            answer_substitutions,

        "missing_substitutions":
            missing_substitutions,

        "duplicate_substitutions":
            duplicate_substitutions,

        "total_supervised_tokens":
            total_supervised_tokens,

        "answer_supervised_tokens":
            answer_substitutions,

        "nonanswer_supervised_tokens":
            nonanswer_supervised_tokens,

        "answer_target_matches":
            answer_target_matches,

        "answer_target_mismatches":
            answer_target_mismatches,

        "native_prefix_matches":
            prefix_matches,

        "native_prefix_mismatches":
            prefix_mismatches,

        "within_document_start_counts": {
            str(k):
                v

            for k, v
            in sorted(
                start_counts.items()
            )
        },

        "query_slot_counts": {
            "0":
                slot_counts[
                    0
                ],

            "1":
                slot_counts[
                    1
                ],
        },

        "unique_sampled_relations":
            len(
                relation_counts
            ),

        "relation_event_count_min":
            min(
                relation_counts.values()
            ),

        "relation_event_count_max":
            max(
                relation_counts.values()
            ),

        "relation_event_count_mean":
            (
                sum(
                    relation_counts.values()
                )
                / len(
                    relation_counts
                )
            ),

        "causal_position_min":
            min(
                causal_position_counts
            ),

        "causal_position_max":
            max(
                causal_position_counts
            ),

        "mismatch_examples":
            mismatch_examples,
    }


# =============================================================================
# STATIC OBJECTIVE CONTRACT
#
# Since no model/logits exist in preflight, we cannot numerically execute the
# new objective without violating the "no model" boundary.
#
# Instead we freeze its exact intended algebra for the trainer:
#
# membership =
#     logsumexp(full_vocab_logits)
#     - logsumexp([correct_logit, distractor_logit])
#
# selector =
#     relu(0.5 - (correct_logit - distractor_logit))
#
# replacement =
#     membership + selector
#
# The trainer must construct the historical per-token CE vector, replace ONLY
# audited answer entries with replacement, then apply the historical reduction.
# =============================================================================

def objective_contract():

    header(
        "FREEZING TREATMENT #4 OBJECTIVE CONTRACT"
    )

    contract = {
        "answer_position_action":
            "replace_original_ce",

        "membership_loss":
            (
                "logsumexp(full_vocab_logits) - "
                "logsumexp([correct_logit,distractor_logit])"
            ),

        "selector_loss":
            (
                "relu(0.5 - "
                "(correct_logit - distractor_logit))"
            ),

        "answer_loss":
            "membership_loss + selector_loss",

        "margin":
            MARGIN,

        "selector_coefficient":
            1.0,

        "membership_coefficient":
            1.0,

        "original_answer_ce_retained":
            False,

        "separate_answer_mean":
            False,

        "separate_auxiliary_addition":
            False,

        "answer_group_reweighting":
            False,

        "nonanswer_loss":
            "ordinary_full_vocabulary_ce",

        "reduction":
            "exact_historical_supervised_token_reduction",
    }

    print(
        "Answer positions:"
    )

    print(
        "  REMOVE original correct-token CE."
    )

    print(
        "  SUBSTITUTE:"
    )

    print(
        "    logsumexp(all 1024 logits)"
    )

    print(
        "    - logsumexp(correct,distractor)"
    )

    print(
        "    + ReLU(0.5 - (correct - distractor))"
    )

    print()

    print(
        "Non-answer positions:"
    )

    print(
        "  ordinary full-vocabulary CE unchanged"
    )

    print()

    print(
        "Reduction:"
    )

    print(
        "  exact historical supervised-token reduction"
    )

    print()

    print(
        "Separate answer mean: NO"
    )

    print(
        "Auxiliary addition to document CE: NO"
    )

    print(
        "Original answer CE retained: NO"
    )

    print(
        "Extra answer weighting: NO"
    )

    return contract


# =============================================================================
# MAIN
# =============================================================================

def main():

    start_sha = (
        safety_check()
    )

    (
        tokenizer,
        records,
    ) = reconstruct_training_records()

    record_audit = (
        validate_records(
            tokenizer,
            records,
        )
    )

    store = build_store(
        tokenizer,
        records,
    )

    schedule_audit = (
        audit_schedule(
            store,
            records,
        )
    )

    contract = (
        objective_contract()
    )

    header(
        "TREATMENT #4 PREFLIGHT VERDICT"
    )

    for name, passed in (
        schedule_audit[
            "checks"
        ].items()
    ):

        print(
            f"{name}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

    print()

    if not schedule_audit[
        "all_pass"
    ]:

        classification = (
            "FAIL — DO NOT TRAIN"
        )

    else:

        classification = (
            "PASS — PREFLIGHT ONLY; "
            "TRAINING NOT YET AUTHORIZED"
        )

    print(
        "CLASSIFICATION:"
    )

    print(
        classification
    )

    print()

    print(
        "Starting checkpoint SHA: PASS"
    )

    print(
        "Model built: NO"
    )

    print(
        "Checkpoint model_state loaded: NO"
    )

    print(
        "Optimizer built: NO"
    )

    print(
        "Gradients calculated: NO"
    )

    print(
        "Backward passes: ZERO"
    )

    print(
        "Optimizer steps: ZERO"
    )

    print(
        "Positive controls evaluated: NO"
    )

    print(
        "Sealed axes loaded/evaluated: NO"
    )

    payload = {
        "experiment":
            "DaveLM v0.9 Treatment #4 preflight",

        "classification":
            classification,

        "read_only":
            True,

        "protected_source":
            str(
                SOURCE_ROOT
            ),

        "output_root":
            str(
                OUTPUT_ROOT
            ),

        "starting_checkpoint": {
            "path":
                str(
                    EXPECTED_START_CHECKPOINT
                ),

            "expected_sha256":
                EXPECTED_START_SHA256,

            "observed_sha256":
                start_sha,

            "pass":
                start_sha
                == EXPECTED_START_SHA256,
        },

        "historical_schedule": {
            "expected_sha256":
                EXPECTED_SCHEDULE_SHA256,

            "observed_sha256":
                schedule_audit[
                    "observed_schedule_sha256"
                ],
        },

        "frozen_objective":
            contract,

        "record_audit":
            record_audit,

        "schedule_and_substitution_audit":
            schedule_audit,

        "expected_counts": {
            "training_records":
                EXPECTED_TRAINING_RECORDS,

            "sampled_events":
                EXPECTED_EVENTS,

            "supervised_tokens":
                EXPECTED_SUPERVISED_TOKENS,

            "answer_substitutions":
                EXPECTED_ANSWER_SUBSTITUTIONS,

            "nonanswer_supervised_tokens":
                EXPECTED_NONANSWER_TOKENS,
        },

        "model_built":
            False,

        "checkpoint_model_state_loaded":
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
        PREFLIGHT_PATH,
        payload,
    )

    print()

    print(
        "PREFLIGHT ARTIFACT:"
    )

    print(
        PREFLIGHT_PATH
    )

    print()

    print(
        "NO TRAINING OCCURRED."
    )


if __name__ == "__main__":
    main()