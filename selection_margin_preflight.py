from __future__ import annotations

import json
import hashlib
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


# =============================================================================
# DaveLM v0.9 — Treatment #3
# FIXED PAIRWISE QUERY-TARGET SELECTION MARGIN OBJECTIVE
#
# FROZEN TREATMENT CONTRACT
#
# CONTROL:
#   successful one-map checkpoint
#   original balanced two-map corpus
#   original independent 32-document schedule
#   ordinary document causal CE
#
# TREATMENT:
#   EXACTLY the same as control, plus:
#
#       L_select =
#           mean(
#               relu(
#                   0.5
#                   - (
#                       z_correct
#                       - z_distractor
#                   )
#               )
#           )
#
#       L_total =
#           L_CE
#           + 1.0 * L_select
#
# Preflight established:
#   - historical schedule SHA reproduced exactly
#   - 32,000 / 32,000 sampled document events are margin-eligible
#   - all 1,536 relations receive margin exposure
#   - answer-position mapping was verified against actual sampled targets
#
# THIS TRAINER:
#   - DOES train for exactly 1000 steps
#   - DOES NOT modify protected C:\DaveLM-v0.9
#   - DOES NOT load/evaluate sealed axes
#   - DOES NOT automatically perform sealed evaluation even on PASS
#
# Positive-control gates:
#   trained_anchors >= 95%
#   supported_relation_withheld_geometry >= 90%
#
# If either fails:
#   INCONCLUSIVE / POSITIVE-CONTROL FAILURE
#
# No rescue.
# No continuation.
# No extra steps.
# No second seed.
# =============================================================================


# =============================================================================
# PATHS / FROZEN CONSTANTS
# =============================================================================

SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")

OUTPUT_ROOT = Path(
    r"C:\DaveLM-CADAVER\two_mapping_selection_margin_manual_seed8380"
)

PREFLIGHT_PATH = (
    OUTPUT_ROOT
    / "selection_margin_preflight.json"
)

CHECKPOINT_DIR = (
    OUTPUT_ROOT
    / "checkpoints"
    / "selection_margin"
    / "seed_8380"
)

FINAL_CHECKPOINT = (
    CHECKPOINT_DIR
    / "latest.pt"
)

TRAINING_LOG_PATH = (
    OUTPUT_ROOT
    / "training_log.json"
)

RESULTS_PATH = (
    OUTPUT_ROOT
    / "positive_control_results.json"
)

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

EXPECTED_SCHEDULE_SHA256 = (
    "a34bb91d258f11c6f7cb6b6cd89c79d"
    "8e6fa12d64af15dd14b028fb7ff675cff"
)


SEED = 8380
SCHEDULE_SEED = 8381

MAX_STEPS = 1000
BATCH_SIZE = 32
CONTEXT_SIZE = 256

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.05
GRAD_CLIP = 2.0

MARGIN = 0.5
LAMBDA_SELECT = 1.0

EXPECTED_PARAMETER_COUNT = 10_594_944
EXPECTED_TRAINING_RECORDS = 1536
EXPECTED_SAMPLED_EVENTS = 32_000
EXPECTED_ELIGIBLE_EVENTS = 32_000
EXPECTED_SUPPORTED = 1536
EXPECTED_ANCHORS = 256

EXPECTED_TOTAL_SUPERVISED_TOKENS = 6_176_000

ANCHOR_GATE = 0.95
SUPPORTED_GATE = 0.90


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
# BASIC HELPERS
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
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
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

    if value is None or not callable(value):

        raise RuntimeError(
            f"Required original v0.9 function "
            f"{name!r} is unavailable."
        )

    return value


def answer_relation(
    row: dict,
) -> tuple[int, int]:

    return (
        int(
            row[
                "source_identity_index"
            ]
        ),
        int(
            row[
                "target_identity_index"
            ]
        ),
    )


# =============================================================================
# LOAD ONE NAMED JSON VALUE ONLY
#
# The corpus contains sealed axes.
#
# We do NOT json.loads() the entire corpus.
# We deserialize only explicitly allowed named positive-control values.
# =============================================================================

def load_named_json_value(
    path: Path,
    key: str,
):

    text = path.read_text(
        encoding="utf-8",
    )

    marker = json.dumps(
        key
    )

    position = text.find(
        marker
    )

    if position < 0:

        raise KeyError(
            f"Could not locate JSON key: {key}"
        )

    colon = text.find(
        ":",
        position + len(marker),
    )

    if colon < 0:

        raise RuntimeError(
            f"Malformed JSON near key: {key}"
        )

    value_start = (
        colon + 1
    )

    while (
        value_start < len(text)
        and text[
            value_start
        ].isspace()
    ):

        value_start += 1

    decoder = json.JSONDecoder()

    value, _end = (
        decoder.raw_decode(
            text,
            value_start,
        )
    )

    return value


# =============================================================================
# SAFETY + PREFLIGHT VERIFICATION
# =============================================================================

def safety_check():

    header(
        "TREATMENT #3 TRAINING SAFETY CHECK"
    )

    if not SOURCE_ROOT.is_dir():

        raise FileNotFoundError(
            SOURCE_ROOT
        )

    if not START_CHECKPOINT.is_file():

        raise FileNotFoundError(
            START_CHECKPOINT
        )

    if not PREFLIGHT_PATH.is_file():

        raise FileNotFoundError(
            "Treatment #3 PASS preflight artifact is missing:\n"
            f"{PREFLIGHT_PATH}"
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
            "REFUSING TO TRAIN: output directory "
            "is inside protected C:\\DaveLM-v0.9."
        )

    except ValueError:
        pass

    if FINAL_CHECKPOINT.exists():

        raise RuntimeError(
            "REFUSING TO OVERWRITE EXISTING "
            "TREATMENT #3 CHECKPOINT:\n"
            f"{FINAL_CHECKPOINT}"
        )

    if RESULTS_PATH.exists():

        raise RuntimeError(
            "REFUSING TO OVERWRITE EXISTING "
            "TREATMENT #3 RESULTS:\n"
            f"{RESULTS_PATH}"
        )

    actual_start_sha = (
        sha256_file(
            START_CHECKPOINT
        )
    )

    if (
        actual_start_sha
        != EXPECTED_START_CHECKPOINT_SHA256
    ):

        raise RuntimeError(
            "Starting one-map checkpoint SHA mismatch."
        )

    preflight = json.loads(
        PREFLIGHT_PATH.read_text(
            encoding="utf-8",
        )
    )

    if not preflight.get(
        "passed",
        False,
    ):

        raise RuntimeError(
            "Treatment #3 preflight does not report PASS."
        )

    if preflight.get(
        "model_built",
        True,
    ):

        raise RuntimeError(
            "Preflight artifact claims a model was built."
        )

    if preflight.get(
        "optimizer_built",
        True,
    ):

        raise RuntimeError(
            "Preflight artifact claims an optimizer was built."
        )

    if preflight.get(
        "gradients_calculated",
        True,
    ):

        raise RuntimeError(
            "Preflight artifact claims gradients were calculated."
        )

    if int(
        preflight.get(
            "optimizer_steps",
            -1,
        )
    ) != 0:

        raise RuntimeError(
            "Preflight artifact claims optimizer steps occurred."
        )

    if preflight.get(
        "sealed_axes_loaded",
        True,
    ):

        raise RuntimeError(
            "Preflight artifact claims sealed axes were loaded."
        )

    if preflight.get(
        "sealed_evaluation_performed",
        True,
    ):

        raise RuntimeError(
            "Preflight artifact claims sealed evaluation occurred."
        )

    objective = preflight.get(
        "objective",
        {},
    )

    if float(
        objective.get(
            "margin",
            -1,
        )
    ) != MARGIN:

        raise RuntimeError(
            "Preflight margin does not match frozen Treatment #3 margin."
        )

    if float(
        objective.get(
            "lambda",
            -1,
        )
    ) != LAMBDA_SELECT:

        raise RuntimeError(
            "Preflight lambda does not match frozen Treatment #3 lambda."
        )

    if objective.get(
        "adaptive_weighting",
        True,
    ):

        raise RuntimeError(
            "Preflight unexpectedly reports adaptive weighting."
        )

    schedule_info = preflight.get(
        "schedule",
        {},
    )

    if (
        schedule_info.get(
            "schedule_sha256"
        )
        != EXPECTED_SCHEDULE_SHA256
    ):

        raise RuntimeError(
            "Preflight schedule SHA mismatch."
        )

    if int(
        schedule_info.get(
            "total_events",
            -1,
        )
    ) != EXPECTED_SAMPLED_EVENTS:

        raise RuntimeError(
            "Preflight sampled-event count mismatch."
        )

    if int(
        schedule_info.get(
            "eligible_events",
            -1,
        )
    ) != EXPECTED_ELIGIBLE_EVENTS:

        raise RuntimeError(
            "Preflight eligible-event count mismatch."
        )

    if int(
        schedule_info.get(
            "ineligible_events",
            -1,
        )
    ) != 0:

        raise RuntimeError(
            "Preflight unexpectedly contains ineligible events."
        )

    if int(
        schedule_info.get(
            "unique_eligible_relations",
            -1,
        )
    ) != EXPECTED_TRAINING_RECORDS:

        raise RuntimeError(
            "Preflight does not show 100% relation coverage."
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
        "Treatment #3 preflight:   PASS"
    )

    print(
        "Historical schedule SHA:  PASS"
    )

    print(
        "Margin eligibility:       32000 / 32000"
    )

    print(
        "Eligible relations:       1536 / 1536"
    )

    print(
        "Frozen margin:             0.5"
    )

    print(
        "Frozen lambda:             1.0"
    )

    print(
        "Sealed axes previously loaded: NO"
    )

    print(
        "Existing candidate checkpoint: NONE"
    )

    return preflight


# =============================================================================
# RECONSTRUCT ORIGINAL CORPUS
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
# VALIDATE RECORD TARGET METADATA AGAIN
# =============================================================================

def validate_training_metadata(
    tokenizer,
    records,
):

    header(
        "REVALIDATING MARGIN TARGET METADATA"
    )

    vocab_size = (
        tokenizer.get_vocab_size()
    )

    bos_id = original_run.required_token_id(
        tokenizer,
        "<bos>",
    )

    relations = set()

    for index, row in enumerate(
        records
    ):

        for key in (
            "prompt",
            "layout",
            "query_slot",
            "source_identity_index",
            "target_identity_index",
            "target_token_id",
            "distractor_token_id",
        ):

            if key not in row:

                raise RuntimeError(
                    f"Record {index}: missing key {key!r}."
                )

        prefix_ids = [
            int(x)
            for x in row[
                "layout"
            ][
                "prefix_ids"
            ]
        ]

        audited_prefix = (
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
            prefix_ids
            != audited_prefix
        ):

            raise RuntimeError(
                f"Record {index}: audited prefix mismatch."
            )

        target_id = int(
            row[
                "target_token_id"
            ]
        )

        distractor_id = int(
            row[
                "distractor_token_id"
            ]
        )

        if (
            target_id
            == distractor_id
        ):

            raise RuntimeError(
                f"Record {index}: target equals distractor."
            )

        if not (
            0
            <= target_id
            < vocab_size
        ):

            raise RuntimeError(
                f"Record {index}: invalid target token ID."
            )

        if not (
            0
            <= distractor_id
            < vocab_size
        ):

            raise RuntimeError(
                f"Record {index}: invalid distractor token ID."
            )

        relations.add(
            answer_relation(
                row
            )
        )

    if (
        len(relations)
        != EXPECTED_TRAINING_RECORDS
    ):

        raise RuntimeError(
            "Expected 1536 distinct queried relations."
        )

    print(
        f"Tokenizer vocab size: {vocab_size}"
    )

    print(
        "Records validated:     1536"
    )

    print(
        "Queried relations:     1536"
    )

    print(
        "Correct token IDs:     PASS"
    )

    print(
        "Distractor token IDs:  PASS"
    )

    print(
        "Audited prompt layout: PASS"
    )


# =============================================================================
# VERIFY FROZEN HISTORICAL SCHEDULE BEFORE MODEL CREATION
# =============================================================================

def verify_schedule(
    store,
):

    header(
        "VERIFYING ORIGINAL HISTORICAL SCHEDULE BEFORE MODEL CREATION"
    )

    generator = (
        torch.Generator()
        .manual_seed(
            SCHEDULE_SEED
        )
    )

    digest = hashlib.sha256()

    supervised_tokens = 0

    for _step in range(
        MAX_STEPS
    ):

        (
            _inputs,
            _targets,
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

        supervised_tokens += int(
            meta[
                "supervised_tokens"
            ]
        )

    observed_sha = (
        digest.hexdigest()
    )

    print(
        "Expected SHA256:"
    )

    print(
        EXPECTED_SCHEDULE_SHA256
    )

    print()

    print(
        "Observed SHA256:"
    )

    print(
        observed_sha
    )

    print()

    if (
        observed_sha
        != EXPECTED_SCHEDULE_SHA256
    ):

        raise RuntimeError(
            "Historical schedule SHA mismatch. "
            "STOP BEFORE MODEL CREATION."
        )

    if (
        supervised_tokens
        != EXPECTED_TOTAL_SUPERVISED_TOKENS
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_SUPERVISED_TOKENS} "
            f"supervised causal tokens; got {supervised_tokens}."
        )

    print(
        "Historical schedule SHA: PASS"
    )

    print(
        "Supervised causal tokens: 6,176,000"
    )

    print(
        "Schedule integrity: PASS"
    )

    return {
        "sha256":
            observed_sha,

        "supervised_tokens":
            supervised_tokens,
    }


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

def build_initialized_model(
    device,
):

    header(
        "INITIALIZING FROM VERIFIED ONE-MAP CHECKPOINT"
    )

    build_model = require_callable(
        original_run,
        "build_model",
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            SEED
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

    checkpoint = torch.load(
        START_CHECKPOINT,
        map_location="cpu",
    )

    if (
        not isinstance(
            checkpoint,
            dict,
        )
        or "model_state"
        not in checkpoint
    ):

        raise RuntimeError(
            "Starting checkpoint does not contain model_state."
        )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ],
        strict=True,
    )

    model = model.to(
        device
    )

    print(
        "One-map model_state load: PASS"
    )

    print(
        "Fresh optimizer has NOT been created yet."
    )

    return model


# =============================================================================
# BUILD PAIRWISE SELECTION LOSS
#
# Every event is known from preflight to contain the audited answer position.
#
# For batch element b:
#
#   answer_global_index =
#       len(record["layout"]["prefix_ids"])
#
#   causal_position =
#       answer_global_index
#       - within_document_start
#       - 1
#
# Then:
#
#   z_correct =
#       logits[b, causal_position, correct_token_id]
#
#   z_distractor =
#       logits[b, causal_position, distractor_token_id]
#
# L_select =
#   mean(
#       relu(
#           MARGIN
#           - (
#               z_correct
#               - z_distractor
#           )
#       )
#   )
# =============================================================================

def compute_selection_loss(
    *,
    logits,
    targets,
    records,
    meta,
):

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
            "Margin loss received wrong document-index count."
        )

    if (
        len(starts)
        != BATCH_SIZE
    ):

        raise RuntimeError(
            "Margin loss received wrong start count."
        )

    margin_terms = []

    correct_wins = 0
    margin_satisfied = 0

    for batch_index in range(
        BATCH_SIZE
    ):

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

        answer_global_index = len(
            row[
                "layout"
            ][
                "prefix_ids"
            ]
        )

        causal_position = (
            answer_global_index
            - start
            - 1
        )

        sequence_length = int(
            targets.shape[1]
        )

        if not (
            0
            <= causal_position
            < sequence_length
        ):

            raise RuntimeError(
                "Treatment #3 preflight established 100% "
                "eligibility, but a live training event is "
                "ineligible. STOP."
            )

        correct_token_id = int(
            row[
                "target_token_id"
            ]
        )

        distractor_token_id = int(
            row[
                "distractor_token_id"
            ]
        )

        sampled_target_id = int(
            targets[
                batch_index,
                causal_position,
            ].item()
        )

        if (
            sampled_target_id
            != correct_token_id
        ):

            raise RuntimeError(
                "\n"
                "LIVE ANSWER-POSITION VERIFICATION FAILED.\n"
                f"Batch index: {batch_index}\n"
                f"Document index: {document_index}\n"
                f"within_document_start: {start}\n"
                f"causal_position: {causal_position}\n"
                f"expected correct target: {correct_token_id}\n"
                f"actual sampled target: {sampled_target_id}\n"
            )

        z_correct = (
            logits[
                batch_index,
                causal_position,
                correct_token_id,
            ]
        )

        z_distractor = (
            logits[
                batch_index,
                causal_position,
                distractor_token_id,
            ]
        )

        difference = (
            z_correct
            - z_distractor
        )

        term = F.relu(
            MARGIN
            - difference
        )

        margin_terms.append(
            term
        )

        if float(
            difference.detach().item()
        ) > 0.0:

            correct_wins += 1

        if float(
            difference.detach().item()
        ) >= MARGIN:

            margin_satisfied += 1

    if not margin_terms:

        # Contractually this should never happen because preflight
        # proved 100% eligibility. This branch exists only as a safe
        # mathematical fallback.
        selection_loss = (
            logits.sum()
            * 0.0
        )

        eligible = 0

    else:

        selection_loss = torch.stack(
            margin_terms
        ).mean()

        eligible = len(
            margin_terms
        )

    return {
        "loss":
            selection_loss,

        "eligible":
            eligible,

        "correct_wins":
            correct_wins,

        "margin_satisfied":
            margin_satisfied,
    }


# =============================================================================
# TRAINING
# =============================================================================

def train(
    *,
    model,
    device,
    store,
    records,
):

    header(
        "BEGINNING TREATMENT #3 TRAINING"
    )

    document_causal_loss = require_callable(
        original_run,
        "document_causal_loss",
    )

    clip_gradients = require_callable(
        original_run,
        "clip_gradients",
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    generator = (
        torch.Generator()
        .manual_seed(
            SCHEDULE_SEED
        )
    )

    live_digest = hashlib.sha256()

    total_supervised_tokens = 0
    total_eligible_events = 0

    training_log = []

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
            device,
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

        live_digest.update(
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

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            inputs
        )

        ce_loss = document_causal_loss(
            logits,
            targets,
        )

        if not torch.isfinite(
            ce_loss
        ):

            raise RuntimeError(
                f"Step {step}: CE loss is non-finite."
            )

        selection = compute_selection_loss(
            logits=logits,
            targets=targets,
            records=records,
            meta=meta,
        )

        selection_loss = selection[
            "loss"
        ]

        if not torch.isfinite(
            selection_loss
        ):

            raise RuntimeError(
                f"Step {step}: selection loss is non-finite."
            )

        total_loss = (
            ce_loss
            + (
                LAMBDA_SELECT
                * selection_loss
            )
        )

        if not torch.isfinite(
            total_loss
        ):

            raise RuntimeError(
                f"Step {step}: total loss is non-finite."
            )

        total_loss.backward()

        for name, parameter in (
            model.named_parameters()
        ):

            if parameter.grad is None:
                continue

            if not torch.isfinite(
                parameter.grad
            ).all():

                raise RuntimeError(
                    f"Step {step}: non-finite gradient "
                    f"in {name}."
                )

        clip_gradients(
            model,
            GRAD_CLIP,
            step,
        )

        optimizer.step()

        for name, parameter in (
            model.named_parameters()
        ):

            if not torch.isfinite(
                parameter
            ).all():

                raise RuntimeError(
                    f"Step {step}: non-finite parameter "
                    f"after optimizer step: {name}."
                )

        supervised_tokens = int(
            meta[
                "supervised_tokens"
            ]
        )

        total_supervised_tokens += (
            supervised_tokens
        )

        eligible = int(
            selection[
                "eligible"
            ]
        )

        total_eligible_events += (
            eligible
        )

        if eligible != BATCH_SIZE:

            raise RuntimeError(
                f"Step {step}: expected 32 eligible margin events; "
                f"got {eligible}."
            )

        ce_value = float(
            ce_loss.detach().cpu().item()
        )

        selection_value = float(
            selection_loss.detach().cpu().item()
        )

        total_value = float(
            total_loss.detach().cpu().item()
        )

        correct_win_rate = (
            selection[
                "correct_wins"
            ]
            / eligible
        )

        margin_satisfied_rate = (
            selection[
                "margin_satisfied"
            ]
            / eligible
        )

        training_log.append(
            {
                "step":
                    step,

                "ce_loss":
                    ce_value,

                "selection_loss":
                    selection_value,

                "total_loss":
                    total_value,

                "eligible_margin_events":
                    eligible,

                "correct_target_outscores_distractor_rate":
                    correct_win_rate,

                "margin_satisfied_rate":
                    margin_satisfied_rate,

                "supervised_tokens":
                    supervised_tokens,

                "cumulative_supervised_tokens":
                    total_supervised_tokens,
            }
        )

        if (
            step == 1
            or step == 10
            or step % 100 == 0
        ):

            print(
                f"step {step:4d}/{MAX_STEPS} "
                f"ce={ce_value:.6f} "
                f"select={selection_value:.6f} "
                f"total={total_value:.6f} "
                f"correct>distractor={correct_win_rate:.3%} "
                f"margin>=0.5={margin_satisfied_rate:.3%} "
                f"supervised_tokens={total_supervised_tokens}"
            )

    live_schedule_sha = (
        live_digest.hexdigest()
    )

    print()
    print(
        "LIVE TRAINING SCHEDULE SHA256:"
    )

    print(
        live_schedule_sha
    )

    if (
        live_schedule_sha
        != EXPECTED_SCHEDULE_SHA256
    ):

        raise RuntimeError(
            "Live Treatment #3 schedule SHA does not match "
            "the frozen historical schedule. "
            "REFUSING TO SAVE CANDIDATE."
        )

    if (
        total_supervised_tokens
        != EXPECTED_TOTAL_SUPERVISED_TOKENS
    ):

        raise RuntimeError(
            "Supervised causal-token total changed. "
            "REFUSING TO SAVE CANDIDATE."
        )

    if (
        total_eligible_events
        != EXPECTED_ELIGIBLE_EVENTS
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_ELIGIBLE_EVENTS} live "
            f"margin events; got {total_eligible_events}. "
            "REFUSING TO SAVE CANDIDATE."
        )

    print()
    print(
        "LIVE SCHEDULE SHA: PASS"
    )

    print(
        "SUPERVISED TOKEN TOTAL: PASS"
    )

    print(
        "MARGIN EVENT TOTAL: 32000 / 32000 PASS"
    )

    return {
        "optimizer":
            optimizer,

        "training_log":
            training_log,

        "total_supervised_tokens":
            total_supervised_tokens,

        "total_eligible_events":
            total_eligible_events,

        "schedule_sha256":
            live_schedule_sha,
    }


# =============================================================================
# SAVE CANDIDATE BEFORE EVALUATION
#
# If the evaluator itself ever has an implementation problem, the completed
# Treatment #3 training result remains safely preserved and need not be rerun.
# =============================================================================

def save_candidate_checkpoint(
    *,
    model,
    optimizer,
    training_summary,
):

    header(
        "SAVING ISOLATED TREATMENT #3 CANDIDATE"
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "experiment":
            "DaveLM v0.9 Treatment #3 fixed selection margin",

        "seed":
            SEED,

        "step":
            MAX_STEPS,

        "starting_checkpoint":
            str(
                START_CHECKPOINT
            ),

        "starting_checkpoint_sha256":
            EXPECTED_START_CHECKPOINT_SHA256,

        "schedule_sha256":
            training_summary[
                "schedule_sha256"
            ],

        "margin":
            MARGIN,

        "lambda_select":
            LAMBDA_SELECT,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "gradient_clip":
            GRAD_CLIP,

        "supervised_tokens":
            training_summary[
                "total_supervised_tokens"
            ],

        "eligible_margin_events":
            training_summary[
                "total_eligible_events"
            ],

        "architecture":
            "untied",

        "model_state":
            model.state_dict(),

        "optimizer_state":
            optimizer.state_dict(),

        "sealed_evaluation_opened":
            False,
    }

    torch.save(
        payload,
        FINAL_CHECKPOINT,
    )

    checkpoint_sha = (
        sha256_file(
            FINAL_CHECKPOINT
        )
    )

    print(
        "Candidate checkpoint:"
    )

    print(
        FINAL_CHECKPOINT
    )

    print()

    print(
        "Candidate checkpoint SHA256:"
    )

    print(
        checkpoint_sha
    )

    return checkpoint_sha


# =============================================================================
# LOAD ALLOWED POSITIVE CONTROLS ONLY
# =============================================================================

def load_positive_controls():

    header(
        "LOADING ALLOWED POSITIVE CONTROLS ONLY"
    )

    if not config.CORPUS_PATH.is_file():

        raise FileNotFoundError(
            config.CORPUS_PATH
        )

    anchors = load_named_json_value(
        config.CORPUS_PATH,
        "trained_anchors",
    )

    supported = load_named_json_value(
        config.CORPUS_PATH,
        "supported_relation_withheld_geometry",
    )

    if not isinstance(
        anchors,
        list,
    ):

        raise RuntimeError(
            "trained_anchors is not a list."
        )

    if not isinstance(
        supported,
        list,
    ):

        raise RuntimeError(
            "supported_relation_withheld_geometry is not a list."
        )

    if (
        len(anchors)
        != EXPECTED_ANCHORS
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_ANCHORS} anchors; "
            f"got {len(anchors)}."
        )

    if (
        len(supported)
        != EXPECTED_SUPPORTED
    ):

        raise RuntimeError(
            f"Expected {EXPECTED_SUPPORTED} supported examples; "
            f"got {len(supported)}."
        )

    print(
        f"trained_anchors: {len(anchors)}"
    )

    print(
        "supported_relation_withheld_geometry: "
        f"{len(supported)}"
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
# DISTRACTOR SHARE AMONG ERRORS
# =============================================================================

def error_analysis(
    summary: dict,
):

    details = summary.get(
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
        item
        for item in details
        if not bool(
            item[
                "exact"
            ]
        )
    ]

    distractor_errors = [
        item
        for item in errors
        if bool(
            item[
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
# REPORT POSITIVE CONTROL
# =============================================================================

def report_axis(
    *,
    name: str,
    summary: dict,
    gate: float,
):

    header(
        f"POSITIVE CONTROL: {name}"
    )

    exact = int(
        summary[
            "exact"
        ]
    )

    examples = int(
        summary[
            "examples"
        ]
    )

    accuracy = float(
        summary[
            "exact_accuracy"
        ]
    )

    errors = error_analysis(
        summary
    )

    passed = (
        accuracy
        >= gate
    )

    print(
        "Exact:"
    )

    print(
        f"{exact}/{examples} "
        f"= {accuracy:.3%}"
    )

    print()

    print(
        f"Frozen gate: {gate:.0%}"
    )

    print(
        f"Gate result: "
        f"{'PASS' if passed else 'FAIL'}"
    )

    print()

    print(
        "Distractor-target share among errors:"
    )

    if errors[
        "incorrect"
    ]:

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

    if "by_query_slot" in summary:

        print()
        print(
            "By query slot:"
        )

        for (
            slot,
            slot_summary
        ) in summary[
            "by_query_slot"
        ].items():

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

        "errors":
            errors,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    safety_check()

    if not torch.cuda.is_available():

        raise RuntimeError(
            "ROCm/PyTorch GPU is unavailable. "
            "REFUSING TO ACCIDENTALLY RUN Treatment #3 on CPU."
        )

    device = torch.device(
        "cuda"
    )

    header(
        "DEVICE"
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

    (
        tokenizer,
        records,
    ) = reconstruct_training_records()

    validate_training_metadata(
        tokenizer,
        records,
    )

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
        "Training store: PASS"
    )

    verify_schedule(
        store
    )

    model = build_initialized_model(
        device
    )

    training_summary = train(
        model=model,
        device=device,
        store=store,
        records=records,
    )

    checkpoint_sha = save_candidate_checkpoint(
        model=model,
        optimizer=training_summary[
            "optimizer"
        ],
        training_summary=training_summary,
    )

    write_json(
        TRAINING_LOG_PATH,
        {
            "experiment":
                "DaveLM v0.9 Treatment #3 fixed selection margin",

            "starting_checkpoint_sha256":
                EXPECTED_START_CHECKPOINT_SHA256,

            "schedule_sha256":
                training_summary[
                    "schedule_sha256"
                ],

            "margin":
                MARGIN,

            "lambda_select":
                LAMBDA_SELECT,

            "steps":
                MAX_STEPS,

            "total_supervised_tokens":
                training_summary[
                    "total_supervised_tokens"
                ],

            "total_eligible_margin_events":
                training_summary[
                    "total_eligible_events"
                ],

            "log":
                training_summary[
                    "training_log"
                ],
        },
    )

    # =========================================================================
    # POSITIVE CONTROLS ONLY
    #
    # Use the exact native v0.9 evaluator that successfully graded Treatment #2.
    # =========================================================================

    (
        anchors,
        supported,
    ) = load_positive_controls()

    native_evaluate = require_callable(
        original_run,
        "_evaluate",
    )

    header(
        "RUNNING ORIGINAL v0.9 NATIVE POSITIVE-CONTROL EVALUATOR"
    )

    print(
        "Evaluating trained_anchors..."
    )

    anchors_summary = native_evaluate(
        model,
        tokenizer,
        anchors,
    )

    print(
        "Evaluating supported_relation_withheld_geometry..."
    )

    supported_summary = native_evaluate(
        model,
        tokenizer,
        supported,
    )

    print()

    print(
        "Native positive-control evaluation complete."
    )

    print(
        "Sealed evaluation opened: NO"
    )

    anchors_report = report_axis(
        name="trained_anchors",
        summary=anchors_summary,
        gate=ANCHOR_GATE,
    )

    supported_report = report_axis(
        name="supported_relation_withheld_geometry",
        summary=supported_summary,
        gate=SUPPORTED_GATE,
    )

    overall_pass = (
        anchors_report[
            "passed"
        ]
        and supported_report[
            "passed"
        ]
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

    results = {
        "experiment":
            "DaveLM v0.9 Treatment #3 fixed selection margin",

        "classification":
            classification,

        "starting_checkpoint_sha256":
            EXPECTED_START_CHECKPOINT_SHA256,

        "candidate_checkpoint":
            str(
                FINAL_CHECKPOINT
            ),

        "candidate_checkpoint_sha256":
            checkpoint_sha,

        "schedule_sha256":
            training_summary[
                "schedule_sha256"
            ],

        "objective": {
            "ordinary_document_ce":
                True,

            "auxiliary":
                "pairwise correct-vs-distractor margin",

            "margin":
                MARGIN,

            "lambda":
                LAMBDA_SELECT,
        },

        "training": {
            "seed":
                SEED,

            "steps":
                MAX_STEPS,

            "batch_size":
                BATCH_SIZE,

            "context_size":
                CONTEXT_SIZE,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "gradient_clip":
                GRAD_CLIP,

            "total_supervised_causal_tokens":
                training_summary[
                    "total_supervised_tokens"
                ],

            "eligible_margin_events":
                training_summary[
                    "total_eligible_events"
                ],
        },

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
                        "errors"
                    ][
                        "incorrect"
                    ],

                "distractor_target_errors":
                    anchors_report[
                        "errors"
                    ][
                        "distractor_target_errors"
                    ],

                "distractor_target_share_among_errors":
                    anchors_report[
                        "errors"
                    ][
                        "distractor_target_share_among_errors"
                    ],

                "target_probability_mean":
                    float(
                        anchors_summary[
                            "target_probability_mean"
                        ]
                    ),

                "target_rank_mean":
                    float(
                        anchors_summary[
                            "target_rank_mean"
                        ]
                    ),

                "target_rank_median":
                    float(
                        anchors_summary[
                            "target_rank_median"
                        ]
                    ),

                "by_query_slot":
                    anchors_summary.get(
                        "by_query_slot",
                        {},
                    ),

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
                        "errors"
                    ][
                        "incorrect"
                    ],

                "distractor_target_errors":
                    supported_report[
                        "errors"
                    ][
                        "distractor_target_errors"
                    ],

                "distractor_target_share_among_errors":
                    supported_report[
                        "errors"
                    ][
                        "distractor_target_share_among_errors"
                    ],

                "target_probability_mean":
                    float(
                        supported_summary[
                            "target_probability_mean"
                        ]
                    ),

                "target_rank_mean":
                    float(
                        supported_summary[
                            "target_rank_mean"
                        ]
                    ),

                "target_rank_median":
                    float(
                        supported_summary[
                            "target_rank_median"
                        ]
                    ),

                "by_query_slot":
                    supported_summary.get(
                        "by_query_slot",
                        {},
                    ),

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

        "sealed_evaluation_opened":
            False,

        "sealed_evaluation_performed":
            False,
    }

    write_json(
        RESULTS_PATH,
        results,
    )

    header(
        "FINAL TREATMENT #3 VERDICT"
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
            "errors"
        ]
    )

    supported_errors = (
        supported_report[
            "errors"
        ]
    )

    print(
        "Anchor distractor-target share among errors:"
    )

    if anchor_errors[
        "incorrect"
    ]:

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

    if supported_errors[
        "incorrect"
    ]:

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
        "SEALED EVALUATION OPENED: NO"
    )

    print(
        "SEALED EVALUATION PERFORMED: NO"
    )

    print()

    print(
        "Candidate checkpoint SHA256:"
    )

    print(
        checkpoint_sha
    )

    print()

    print(
        "Results file:"
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
            "Candidate is eligible for a separately "
            "authorized sealed evaluation."
        )

        print(
            "This trainer does NOT perform that evaluation."
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
            "run a second seed, or open sealed evaluation."
        )


if __name__ == "__main__":
    main()