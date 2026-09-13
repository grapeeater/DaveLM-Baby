from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


# =============================================================================
# DaveLM v0.9 — Treatment #4
# CANDIDATE-SET MEMBERSHIP + STRICT QUERY-TARGET SELECTION
#
# FROZEN ANSWER-POSITION OBJECTIVE
#
# At each eligible answer position:
#
#   L_answer =
#       -log(p_correct + p_distractor)
#       + ReLU(0.5 - (z_correct - z_distractor))
#
# Implemented stably as:
#
#   membership =
#       logsumexp(all_vocab_logits)
#       - logsumexp([z_correct, z_distractor])
#
#   selector =
#       ReLU(0.5 - (z_correct - z_distractor))
#
#   replacement =
#       membership + selector
#
# CRITICAL:
# This is exact PER-TOKEN SUBSTITUTION.
#
# Historical ordinary CE is computed for every token.
# At audited answer-token slots ONLY, the original CE value is removed and
# replaced with the expression above.
#
# The final loss is the mean over the exact same supervised token set:
#
#   targets != -100
#
# This reproduces the original:
#
#   F.cross_entropy(..., ignore_index=-100)
#
# reduction exactly.
#
# NO:
# - auxiliary answer mean
# - answer-group weighting
# - schedule changes
# - architecture changes
# - early stopping
# - post-hoc hyperparameter changes
# - positive-control evaluation in this file
# - sealed evaluation
#
# This script trains exactly 1000 steps and SAVES THE CANDIDATE BEFORE ANY
# POSITIVE-CONTROL EVALUATION.
# =============================================================================


# =============================================================================
# FROZEN PATHS
# =============================================================================

SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")

PREFLIGHT_ROOT = Path(
    r"C:\DaveLM-CADAVER\two_mapping_membership_margin_manual_seed8380"
)

PREFLIGHT_PATH = (
    PREFLIGHT_ROOT
    / "treatment4_preflight.json"
)

OUTPUT_ROOT = (
    PREFLIGHT_ROOT
    / "checkpoints"
    / "membership_margin"
    / "seed_8380"
)

FINAL_CHECKPOINT = (
    OUTPUT_ROOT
    / "latest.pt"
)

RESULT_PATH = (
    PREFLIGHT_ROOT
    / "treatment4_training_result.json"
)

START_CHECKPOINT = Path(
    r"C:\DaveLM-v0.9\experiments"
    r"\minimal_contextual_binding"
    r"\checkpoints"
    r"\treatment_one_mapping"
    r"\seed_8380"
    r"\latest.pt"
)


# =============================================================================
# FROZEN HASHES
# =============================================================================

EXPECTED_START_SHA256 = (
    "345984c52a06db5f988aaf4cd47963ee"
    "a0e9d77489cee94dbb10816af2f5443e"
)

EXPECTED_SCHEDULE_SHA256 = (
    "a34bb91d258f11c6f7cb6b6cd89c79d"
    "8e6fa12d64af15dd14b028fb7ff675cff"
)


# =============================================================================
# FROZEN TRAINING CONTRACT
# =============================================================================

SEED = 8380
SCHEDULE_SEED = 8381

MAX_STEPS = 1000
BATCH_SIZE = 32
CONTEXT_SIZE = 256

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.05
GRAD_CLIP = 2.0

MARGIN = 0.5

EXPECTED_RECORDS = 1536
EXPECTED_EVENTS = 32_000

EXPECTED_SUPERVISED_TOKENS = 6_176_000
EXPECTED_ANSWER_SUBSTITUTIONS = 32_000
EXPECTED_NONANSWER_TOKENS = 6_144_000


# =============================================================================
# IMPORT EXACT v0.9 CODE
# =============================================================================

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import experiments.two_mapping_contextual_binding.run as original_run
from experiments.two_mapping_contextual_binding import config

from v0_8.document_sampling import (
    DocumentTokenStore,
    sample_document_batch,
)

from v0_8_3.protection import strict_json_dumps


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
            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
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


def require_callable(module, name: str):
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
# HARD SAFETY / CONTRACT CHECKS
# =============================================================================

def safety_and_preflight_check() -> dict:

    header(
        "TREATMENT #4 — SAFETY AND PREFLIGHT VERIFICATION"
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU/ROCm is unavailable. REFUSING TO TRAIN."
        )

    if not START_CHECKPOINT.is_file():
        raise FileNotFoundError(
            START_CHECKPOINT
        )

    if not PREFLIGHT_PATH.is_file():
        raise FileNotFoundError(
            f"Required Treatment #4 preflight artifact not found:\n"
            f"{PREFLIGHT_PATH}"
        )

    if FINAL_CHECKPOINT.exists():
        raise RuntimeError(
            "\nREFUSING TO OVERWRITE EXISTING TREATMENT #4 CHECKPOINT:\n"
            f"{FINAL_CHECKPOINT}\n"
        )

    if RESULT_PATH.exists():
        raise RuntimeError(
            "\nREFUSING TO OVERWRITE EXISTING TREATMENT #4 RESULT:\n"
            f"{RESULT_PATH}\n"
        )

    source_resolved = SOURCE_ROOT.resolve()
    output_resolved = OUTPUT_ROOT.resolve()

    try:
        output_resolved.relative_to(
            source_resolved
        )

        raise RuntimeError(
            "REFUSING TO TRAIN: output path lies inside protected "
            "C:\\DaveLM-v0.9."
        )

    except ValueError:
        pass

    observed_start_sha = sha256_file(
        START_CHECKPOINT
    )

    if observed_start_sha != EXPECTED_START_SHA256:
        raise RuntimeError(
            "\nSTART CHECKPOINT SHA MISMATCH.\n"
            f"Expected: {EXPECTED_START_SHA256}\n"
            f"Observed: {observed_start_sha}\n"
        )

    preflight = json.loads(
        PREFLIGHT_PATH.read_text(
            encoding="utf-8"
        )
    )

    expected_classification = (
        "PASS — PREFLIGHT ONLY; "
        "TRAINING NOT YET AUTHORIZED"
    )

    if (
        preflight.get("classification")
        != expected_classification
    ):
        raise RuntimeError(
            "Treatment #4 preflight did not record a clean PASS."
        )

    schedule_audit = preflight[
        "schedule_and_substitution_audit"
    ]

    if not schedule_audit.get(
        "all_pass",
        False,
    ):
        raise RuntimeError(
            "Treatment #4 preflight hard gates were not all PASS."
        )

    observed_preflight_schedule = (
        preflight[
            "historical_schedule"
        ][
            "observed_sha256"
        ]
    )

    if (
        observed_preflight_schedule
        != EXPECTED_SCHEDULE_SHA256
    ):
        raise RuntimeError(
            "Preflight schedule SHA does not match frozen contract."
        )

    if (
        int(
            schedule_audit[
                "answer_substitutions"
            ]
        )
        != EXPECTED_ANSWER_SUBSTITUTIONS
    ):
        raise RuntimeError(
            "Preflight answer-substitution count mismatch."
        )

    if (
        int(
            schedule_audit[
                "total_supervised_tokens"
            ]
        )
        != EXPECTED_SUPERVISED_TOKENS
    ):
        raise RuntimeError(
            "Preflight supervised-token count mismatch."
        )

    if (
        int(
            schedule_audit[
                "nonanswer_supervised_tokens"
            ]
        )
        != EXPECTED_NONANSWER_TOKENS
    ):
        raise RuntimeError(
            "Preflight non-answer token count mismatch."
        )

    objective = preflight[
        "frozen_objective"
    ]

    if objective.get(
        "answer_position_action"
    ) != "replace_original_ce":
        raise RuntimeError(
            "Preflight objective is not exact CE substitution."
        )

    if objective.get(
        "original_answer_ce_retained"
    ) is not False:
        raise RuntimeError(
            "Preflight says original answer CE is retained."
        )

    if objective.get(
        "separate_answer_mean"
    ) is not False:
        raise RuntimeError(
            "Preflight permits separate answer averaging."
        )

    if objective.get(
        "separate_auxiliary_addition"
    ) is not False:
        raise RuntimeError(
            "Preflight permits auxiliary answer addition."
        )

    if float(
        objective.get(
            "margin"
        )
    ) != MARGIN:
        raise RuntimeError(
            "Preflight margin does not equal frozen 0.5."
        )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        "Starting checkpoint SHA: PASS"
    )

    print(
        "Treatment #4 preflight artifact: PASS"
    )

    print(
        "Frozen schedule contract: PASS"
    )

    print(
        "Frozen substitution contract: PASS"
    )

    print(
        "Protected v0.9 output isolation: PASS"
    )

    print(
        "Existing candidate overwrite protection: PASS"
    )

    return preflight


# =============================================================================
# RECONSTRUCT ORIGINAL TRAINING CORPUS
# =============================================================================

def reconstruct_training():

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

    if len(records) != EXPECTED_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORDS} records; got {len(records)}."
        )

    bos_id = original_run.required_token_id(
        tokenizer,
        "<bos>",
    )

    for index, row in enumerate(records):

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
                f"Record {index}: target == distractor."
            )

        audited_prefix = [
            int(x)
            for x in row[
                "layout"
            ][
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

        if audited_prefix != native_prefix:
            raise RuntimeError(
                f"Record {index}: native prefix mismatch."
            )

    store = DocumentTokenStore.from_documents(
        tokenizer,
        records,
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

    print(
        "Native prefix mapping: PASS"
    )

    return (
        tokenizer,
        records,
        store,
    )


# =============================================================================
# VERIFY EXACT HISTORICAL SCHEDULE BEFORE TRAINING
# =============================================================================

def preverify_schedule(
    store,
    records,
):

    header(
        "RECONSTRUCTING FROZEN 1000-STEP HISTORICAL SCHEDULE"
    )

    generator = (
        torch.Generator()
        .manual_seed(
            SCHEDULE_SEED
        )
    )

    digest = hashlib.sha256()

    events = 0
    supervised_tokens = 0
    answer_events = 0

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

        # Exact historical document_causal_loss supervision rule.
        supervised_tokens += int(
            (
                targets != -100
            ).sum().item()
        )

        for batch_index in range(
            BATCH_SIZE
        ):

            events += 1

            row = records[
                document_indices[
                    batch_index
                ]
            ]

            start = starts[
                batch_index
            ]

            prefix_ids = [
                int(x)
                for x in row[
                    "layout"
                ][
                    "prefix_ids"
                ]
            ]

            answer_global_index = len(
                prefix_ids
            )

            causal_position = (
                answer_global_index
                - start
                - 1
            )

            if not (
                0
                <= causal_position
                < targets.shape[1]
            ):
                raise RuntimeError(
                    f"Schedule precheck step {step}: "
                    "answer is outside sampled window."
                )

            if int(
                targets[
                    batch_index,
                    causal_position,
                ].item()
            ) == -100:
                raise RuntimeError(
                    f"Schedule precheck step {step}: "
                    "answer position is not supervised."
                )

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

            if actual_target != expected_target:
                raise RuntimeError(
                    f"Schedule precheck step {step}: "
                    "answer target mismatch."
                )

            sampled_prefix = [
                int(x)
                for x in inputs[
                    batch_index,
                    :causal_position + 1,
                ].tolist()
            ]

            if sampled_prefix != prefix_ids:
                raise RuntimeError(
                    f"Schedule precheck step {step}: "
                    "native causal prefix mismatch."
                )

            answer_events += 1

    observed_sha = digest.hexdigest()

    if observed_sha != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError(
            "\nHISTORICAL SCHEDULE SHA MISMATCH.\n"
            f"Expected: {EXPECTED_SCHEDULE_SHA256}\n"
            f"Observed: {observed_sha}\n"
        )

    if events != EXPECTED_EVENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_EVENTS} events; got {events}."
        )

    if answer_events != EXPECTED_ANSWER_SUBSTITUTIONS:
        raise RuntimeError(
            "Answer-event count does not match frozen contract."
        )

    if supervised_tokens != EXPECTED_SUPERVISED_TOKENS:
        raise RuntimeError(
            "\nSUPERVISED TOKEN COUNT MISMATCH.\n"
            f"Expected: {EXPECTED_SUPERVISED_TOKENS}\n"
            f"Observed: {supervised_tokens}\n"
        )

    nonanswer = (
        supervised_tokens
        - answer_events
    )

    if nonanswer != EXPECTED_NONANSWER_TOKENS:
        raise RuntimeError(
            "Non-answer supervised-token count mismatch."
        )

    print(
        f"Schedule SHA:          {observed_sha}"
    )

    print(
        "Historical schedule:   PASS"
    )

    print(
        f"Sampled events:        {events}"
    )

    print(
        f"Answer substitutions:  {answer_events}"
    )

    print(
        f"Supervised tokens:     {supervised_tokens}"
    )

    print(
        f"Non-answer tokens:     {nonanswer}"
    )

    return observed_sha


# =============================================================================
# CHECKPOINT MODEL-STATE EXTRACTION
# =============================================================================

def extract_model_state(
    checkpoint,
):

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Starting checkpoint is not a dictionary."
        )

    if "model_state" in checkpoint:
        return checkpoint[
            "model_state"
        ]

    if "model_state_dict" in checkpoint:
        return checkpoint[
            "model_state_dict"
        ]

    if "model" in checkpoint:
        candidate = checkpoint[
            "model"
        ]

        if isinstance(candidate, dict):
            return candidate

    # Defensive fallback:
    # If every top-level value is a tensor, checkpoint itself may be state_dict.
    if checkpoint and all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        return checkpoint

    raise RuntimeError(
        "Could not identify model_state in starting checkpoint."
    )


# =============================================================================
# EXACT TREATMENT #4 LOSS
# =============================================================================

def treatment4_loss(
    logits,
    targets,
    answer_batch_indices,
    answer_positions,
    correct_token_ids,
    distractor_token_ids,
):

    if logits.shape[:2] != targets.shape:
        raise ValueError(
            "Logit and target batch/time dimensions must match."
        )

    batch_size, sequence_length, vocab_size = (
        logits.shape
    )

    # -------------------------------------------------------------------------
    # Historical per-token full-vocabulary CE.
    #
    # Original:
    #
    # F.cross_entropy(
    #     logits.reshape(-1, vocab),
    #     targets.reshape(-1),
    #     ignore_index=-100,
    # )
    #
    # We first construct the exact same tokenwise CE values.
    # -------------------------------------------------------------------------

    token_ce = F.cross_entropy(
        logits.reshape(
            -1,
            vocab_size,
        ),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(
        batch_size,
        sequence_length,
    )

    supervised_mask = (
        targets != -100
    )

    supervised_count = int(
        supervised_mask.sum().item()
    )

    # Clone so original answer CE is literally replaced.
    substituted_losses = token_ce.clone()

    answer_batch_indices = torch.as_tensor(
        answer_batch_indices,
        dtype=torch.long,
        device=logits.device,
    )

    answer_positions = torch.as_tensor(
        answer_positions,
        dtype=torch.long,
        device=logits.device,
    )

    correct_token_ids = torch.as_tensor(
        correct_token_ids,
        dtype=torch.long,
        device=logits.device,
    )

    distractor_token_ids = torch.as_tensor(
        distractor_token_ids,
        dtype=torch.long,
        device=logits.device,
    )

    answer_logits = logits[
        answer_batch_indices,
        answer_positions,
        :,
    ]

    row_indices = torch.arange(
        answer_logits.shape[0],
        device=logits.device,
    )

    correct_logits = answer_logits[
        row_indices,
        correct_token_ids,
    ]

    distractor_logits = answer_logits[
        row_indices,
        distractor_token_ids,
    ]

    # -------------------------------------------------------------------------
    # Candidate-set membership:
    #
    # -log(p_c + p_d)
    #
    # = logsumexp(all logits) - logsumexp([z_c, z_d])
    # -------------------------------------------------------------------------

    all_lse = torch.logsumexp(
        answer_logits,
        dim=-1,
    )

    candidate_pair = torch.stack(
        (
            correct_logits,
            distractor_logits,
        ),
        dim=-1,
    )

    candidate_lse = torch.logsumexp(
        candidate_pair,
        dim=-1,
    )

    membership_loss = (
        all_lse
        - candidate_lse
    )

    # -------------------------------------------------------------------------
    # Strict query-target selector:
    #
    # ReLU(0.5 - (z_c - z_d))
    # -------------------------------------------------------------------------

    logit_gap = (
        correct_logits
        - distractor_logits
    )

    selector_loss = F.relu(
        MARGIN
        - logit_gap
    )

    answer_replacement_loss = (
        membership_loss
        + selector_loss
    )

    # -------------------------------------------------------------------------
    # Exact per-token substitution.
    # -------------------------------------------------------------------------

    substituted_losses[
        answer_batch_indices,
        answer_positions,
    ] = answer_replacement_loss

    # -------------------------------------------------------------------------
    # Historical reduction:
    #
    # mean over exactly targets != -100.
    #
    # This matches original cross_entropy(... ignore_index=-100).
    # -------------------------------------------------------------------------

    final_loss = substituted_losses[
        supervised_mask
    ].mean()

    # -------------------------------------------------------------------------
    # Descriptive-only metrics.
    # No training decisions may depend on these.
    # -------------------------------------------------------------------------

    with torch.no_grad():

        log_candidate_mass = (
            candidate_lse
            - all_lse
        )

        candidate_mass = torch.exp(
            log_candidate_mass
        )

        mean_candidate_mass = float(
            candidate_mass.mean().item()
        )

        correct_gt_distractor = float(
            (
                logit_gap > 0
            )
            .float()
            .mean()
            .item()
        )

        margin_satisfied = float(
            (
                logit_gap >= MARGIN
            )
            .float()
            .mean()
            .item()
        )

        mean_membership_loss = float(
            membership_loss.mean().item()
        )

        mean_selector_loss = float(
            selector_loss.mean().item()
        )

        mean_answer_replacement = float(
            answer_replacement_loss.mean().item()
        )

        mean_original_answer_ce = float(
            token_ce[
                answer_batch_indices,
                answer_positions,
            ].mean().item()
        )

    metrics = {
        "supervised_count":
            supervised_count,

        "answer_count":
            int(
                answer_positions.numel()
            ),

        "mean_candidate_mass":
            mean_candidate_mass,

        "correct_gt_distractor_fraction":
            correct_gt_distractor,

        "margin_satisfied_fraction":
            margin_satisfied,

        "mean_membership_loss":
            mean_membership_loss,

        "mean_selector_loss":
            mean_selector_loss,

        "mean_answer_replacement_loss":
            mean_answer_replacement,

        "mean_original_answer_ce_removed":
            mean_original_answer_ce,
    }

    return (
        final_loss,
        metrics,
    )


# =============================================================================
# PRE-STEP-1 IMPLEMENTATION SELF-AUDIT
# =============================================================================

def implementation_self_audit(
    model,
    store,
    records,
    device,
):

    header(
        "PRE-STEP-1 TREATMENT #4 IMPLEMENTATION SELF-AUDIT"
    )

    # Fresh independent generator so this audit does NOT consume the training
    # schedule generator.
    audit_generator = (
        torch.Generator()
        .manual_seed(
            SCHEDULE_SEED
        )
    )

    (
        inputs,
        targets,
        _,
        meta,
    ) = sample_document_batch(
        store,
        BATCH_SIZE,
        CONTEXT_SIZE,
        audit_generator,
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

    answer_batch_indices = []
    answer_positions = []
    correct_ids = []
    distractor_ids = []

    for batch_index in range(
        BATCH_SIZE
    ):

        row = records[
            document_indices[
                batch_index
            ]
        ]

        start = starts[
            batch_index
        ]

        prefix_ids = [
            int(x)
            for x in row[
                "layout"
            ][
                "prefix_ids"
            ]
        ]

        position = (
            len(prefix_ids)
            - start
            - 1
        )

        if int(
            targets[
                batch_index,
                position,
            ].item()
        ) != int(
            row[
                "target_token_id"
            ]
        ):
            raise RuntimeError(
                "Self-audit target mapping failed."
            )

        if int(
            targets[
                batch_index,
                position,
            ].item()
        ) == -100:
            raise RuntimeError(
                "Self-audit answer position is ignored."
            )

        sampled_prefix = [
            int(x)
            for x in inputs[
                batch_index,
                :position + 1,
            ].tolist()
        ]

        if sampled_prefix != prefix_ids:
            raise RuntimeError(
                "Self-audit native prefix identity failed."
            )

        answer_batch_indices.append(
            batch_index
        )

        answer_positions.append(
            position
        )

        correct_ids.append(
            int(
                row[
                    "target_token_id"
                ]
            )
        )

        distractor_ids.append(
            int(
                row[
                    "distractor_token_id"
                ]
            )
        )

    with torch.no_grad():

        logits = model(
            inputs
        )

        new_loss, metrics = treatment4_loss(
            logits,
            targets,
            answer_batch_indices,
            answer_positions,
            correct_ids,
            distractor_ids,
        )

        # Historical reference loss on exactly same logits/targets.
        historical_loss = F.cross_entropy(
            logits.reshape(
                -1,
                logits.shape[-1],
            ),
            targets.reshape(-1),
            ignore_index=-100,
        )

        # Rebuild manually to prove historical reduction identity.
        token_ce = F.cross_entropy(
            logits.reshape(
                -1,
                logits.shape[-1],
            ),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(
            targets
        )

        mask = (
            targets != -100
        )

        reconstructed_historical = (
            token_ce[
                mask
            ].mean()
        )

        if not torch.allclose(
            historical_loss,
            reconstructed_historical,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise RuntimeError(
                "Historical reduction reconstruction FAILED."
            )

    if metrics[
        "answer_count"
    ] != BATCH_SIZE:
        raise RuntimeError(
            "Self-audit did not substitute exactly one answer per document."
        )

    if metrics[
        "supervised_count"
    ] <= metrics[
        "answer_count"
    ]:
        raise RuntimeError(
            "Invalid supervised-token accounting."
        )

    if not torch.isfinite(
        new_loss
    ):
        raise RuntimeError(
            "Treatment #4 self-audit produced non-finite loss."
        )

    print(
        "Historical CE reconstruction: PASS"
    )

    print(
        f"Supervised tokens this batch: {metrics['supervised_count']}"
    )

    print(
        f"Answer substitutions:         {metrics['answer_count']}"
    )

    print(
        f"Non-answer CE positions:      "
        f"{metrics['supervised_count'] - metrics['answer_count']}"
    )

    print(
        "Original answer CE retained: NO"
    )

    print(
        "Separate answer mean:        NO"
    )

    print(
        "Auxiliary answer addition:   NO"
    )

    print(
        "Per-token substitution:      PASS"
    )

    print(
        "Native prefix identity:      PASS"
    )

    print(
        "Target-position identity:    PASS"
    )

    print()

    print(
        f"Reference historical CE:      "
        f"{float(historical_loss.item()):.6f}"
    )

    print(
        f"Treatment #4 substituted loss:"
        f" {float(new_loss.item()):.6f}"
    )

    print()

    print(
        "SELF-AUDIT CLASSIFICATION: PASS"
    )


# =============================================================================
# MODEL BUILD + STARTING CHECKPOINT LOAD
# =============================================================================

def build_model_and_optimizer(
    device,
):

    header(
        "BUILDING BABY AND LOADING VERIFIED ONE-MAP CHECKPOINT"
    )

    torch.manual_seed(
        SEED
    )

    torch.cuda.manual_seed_all(
        SEED
    )

    build_model_fn = require_callable(
        original_run,
        "build_model",
    )

    model = build_model_fn(
        "untied"
    ).to(
        device
    )

    checkpoint = torch.load(
        START_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model_state = extract_model_state(
        checkpoint
    )

    missing, unexpected = model.load_state_dict(
        model_state,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "\nSTART CHECKPOINT STATE-DICT MISMATCH.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}\n"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Parameters: {parameter_count:,}"
    )

    print(
        f"Device:     {device}"
    )

    print(
        "One-map model_state: PASS"
    )

    print(
        "Optimizer: fresh AdamW"
    )

    print(
        f"LR:         {LEARNING_RATE}"
    )

    print(
        f"WD:         {WEIGHT_DECAY}"
    )

    print(
        f"Clip:       {GRAD_CLIP}"
    )

    return (
        model,
        optimizer,
        parameter_count,
    )


# =============================================================================
# TRAIN
# =============================================================================

def train(
    model,
    optimizer,
    store,
    records,
    device,
):

    header(
        "TREATMENT #4 TRAINING — 1000 FROZEN STEPS"
    )

    model.train()

    generator = (
        torch.Generator()
        .manual_seed(
            SCHEDULE_SEED
        )
    )

    schedule_digest = hashlib.sha256()

    cumulative_supervised = 0
    cumulative_answer_substitutions = 0

    training_log = []

    log_steps = {
        1,
        10,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
    }

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        (
            inputs,
            targets,
            _,
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

        schedule_digest.update(
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

        answer_batch_indices = []
        answer_positions = []
        correct_ids = []
        distractor_ids = []

        for batch_index in range(
            BATCH_SIZE
        ):

            row = records[
                document_indices[
                    batch_index
                ]
            ]

            start = starts[
                batch_index
            ]

            prefix_ids = [
                int(x)
                for x in row[
                    "layout"
                ][
                    "prefix_ids"
                ]
            ]

            position = (
                len(prefix_ids)
                - start
                - 1
            )

            if not (
                0
                <= position
                < targets.shape[1]
            ):
                raise RuntimeError(
                    f"Step {step}: answer outside sampled window."
                )

            expected_target = int(
                row[
                    "target_token_id"
                ]
            )

            actual_target = int(
                targets[
                    batch_index,
                    position,
                ].item()
            )

            if actual_target != expected_target:
                raise RuntimeError(
                    f"Step {step}: sampled target mismatch."
                )

            if actual_target == -100:
                raise RuntimeError(
                    f"Step {step}: answer is not supervised."
                )

            sampled_prefix = [
                int(x)
                for x in inputs[
                    batch_index,
                    :position + 1,
                ].tolist()
            ]

            if sampled_prefix != prefix_ids:
                raise RuntimeError(
                    f"Step {step}: native prefix identity mismatch."
                )

            answer_batch_indices.append(
                batch_index
            )

            answer_positions.append(
                position
            )

            correct_ids.append(
                expected_target
            )

            distractor_ids.append(
                int(
                    row[
                        "distractor_token_id"
                    ]
                )
            )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            inputs
        )

        loss, metrics = treatment4_loss(
            logits,
            targets,
            answer_batch_indices,
            answer_positions,
            correct_ids,
            distractor_ids,
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"Step {step}: non-finite loss."
            )

        loss.backward()

        # Historical clip function if available; otherwise exact standard
        # global norm clipping with the frozen threshold.
        clip_fn = getattr(
            original_run,
            "clip_gradients",
            None,
        )

        if callable(
            clip_fn
        ):
            clip_fn(
                model,
                GRAD_CLIP,
                step,
            )

        else:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

        for name, parameter in model.named_parameters():

            if (
                parameter.grad is not None
                and not torch.isfinite(
                    parameter.grad
                ).all()
            ):
                raise RuntimeError(
                    f"Step {step}: non-finite gradient in {name}."
                )

        optimizer.step()

        cumulative_supervised += int(
            metrics[
                "supervised_count"
            ]
        )

        cumulative_answer_substitutions += int(
            metrics[
                "answer_count"
            ]
        )

        if step in log_steps:

            row = {
                "step":
                    step,

                "loss":
                    float(
                        loss.detach().item()
                    ),

                "mean_candidate_mass":
                    metrics[
                        "mean_candidate_mass"
                    ],

                "correct_gt_distractor_fraction":
                    metrics[
                        "correct_gt_distractor_fraction"
                    ],

                "margin_satisfied_fraction":
                    metrics[
                        "margin_satisfied_fraction"
                    ],

                "mean_membership_loss":
                    metrics[
                        "mean_membership_loss"
                    ],

                "mean_selector_loss":
                    metrics[
                        "mean_selector_loss"
                    ],

                "mean_answer_replacement_loss":
                    metrics[
                        "mean_answer_replacement_loss"
                    ],

                "mean_original_answer_ce_removed":
                    metrics[
                        "mean_original_answer_ce_removed"
                    ],

                "cumulative_supervised_tokens":
                    cumulative_supervised,

                "cumulative_answer_substitutions":
                    cumulative_answer_substitutions,
            }

            training_log.append(
                row
            )

            print(
                f"step {step:4d} | "
                f"loss {row['loss']:.6f} | "
                f"candidate mass "
                f"{100.0 * row['mean_candidate_mass']:.3f}% | "
                f"correct>distractor "
                f"{100.0 * row['correct_gt_distractor_fraction']:.3f}% | "
                f"margin>=0.5 "
                f"{100.0 * row['margin_satisfied_fraction']:.3f}% | "
                f"membership "
                f"{row['mean_membership_loss']:.6f} | "
                f"selector "
                f"{row['mean_selector_loss']:.6f} | "
                f"supervised "
                f"{cumulative_supervised}"
            )

    observed_schedule_sha = (
        schedule_digest.hexdigest()
    )

    if (
        observed_schedule_sha
        != EXPECTED_SCHEDULE_SHA256
    ):
        raise RuntimeError(
            "\nLIVE TRAINING SCHEDULE SHA MISMATCH.\n"
            f"Expected: {EXPECTED_SCHEDULE_SHA256}\n"
            f"Observed: {observed_schedule_sha}\n"
        )

    if (
        cumulative_supervised
        != EXPECTED_SUPERVISED_TOKENS
    ):
        raise RuntimeError(
            "\nFINAL SUPERVISED TOKEN COUNT MISMATCH.\n"
            f"Expected: {EXPECTED_SUPERVISED_TOKENS}\n"
            f"Observed: {cumulative_supervised}\n"
        )

    if (
        cumulative_answer_substitutions
        != EXPECTED_ANSWER_SUBSTITUTIONS
    ):
        raise RuntimeError(
            "\nFINAL ANSWER SUBSTITUTION COUNT MISMATCH.\n"
            f"Expected: {EXPECTED_ANSWER_SUBSTITUTIONS}\n"
            f"Observed: {cumulative_answer_substitutions}\n"
        )

    print()

    print(
        "LIVE HISTORICAL SCHEDULE SHA: PASS"
    )

    print(
        f"Observed: {observed_schedule_sha}"
    )

    print(
        f"Supervised tokens: "
        f"{cumulative_supervised} — PASS"
    )

    print(
        f"Answer substitutions: "
        f"{cumulative_answer_substitutions} — PASS"
    )

    return {
        "schedule_sha256":
            observed_schedule_sha,

        "supervised_tokens":
            cumulative_supervised,

        "answer_substitutions":
            cumulative_answer_substitutions,

        "training_log":
            training_log,
    }


# =============================================================================
# SAVE CANDIDATE BEFORE EVALUATION
# =============================================================================

def save_candidate(
    model,
    optimizer,
    parameter_count,
    train_result,
):

    header(
        "SAVING TREATMENT #4 CANDIDATE — BEFORE POSITIVE-CONTROL EVALUATION"
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "model_state":
            model.state_dict(),

        "optimizer_state":
            optimizer.state_dict(),

        "step":
            MAX_STEPS,

        "seed":
            SEED,

        "schedule_seed":
            SCHEDULE_SEED,

        "treatment":
            "membership_margin_per_token_substitution",

        "answer_objective": {
            "membership":
                (
                    "logsumexp(all_logits) - "
                    "logsumexp([correct_logit,distractor_logit])"
                ),

            "selector":
                (
                    "relu(0.5 - "
                    "(correct_logit - distractor_logit))"
                ),

            "margin":
                MARGIN,

            "membership_weight":
                1.0,

            "selector_weight":
                1.0,

            "per_token_substitution":
                True,

            "original_answer_ce_retained":
                False,

            "historical_reduction":
                (
                    "mean over targets != -100, "
                    "matching document_causal_loss"
                ),
        },

        "parameter_count":
            parameter_count,

        "start_checkpoint_sha256":
            EXPECTED_START_SHA256,

        "schedule_sha256":
            train_result[
                "schedule_sha256"
            ],

        "supervised_tokens":
            train_result[
                "supervised_tokens"
            ],

        "answer_substitutions":
            train_result[
                "answer_substitutions"
            ],
    }

    torch.save(
        payload,
        FINAL_CHECKPOINT,
    )

    candidate_sha = sha256_file(
        FINAL_CHECKPOINT
    )

    print(
        "Candidate saved:"
    )

    print(
        FINAL_CHECKPOINT
    )

    print()

    print(
        "Candidate SHA256:"
    )

    print(
        candidate_sha
    )

    print()

    print(
        "Positive controls evaluated: NO"
    )

    print(
        "Sealed evaluation performed: NO"
    )

    return candidate_sha


# =============================================================================
# MAIN
# =============================================================================

def main():

    header(
        "DaveLM v0.9 — TREATMENT #4"
    )

    print(
        "Candidate-set membership + strict query-target margin"
    )

    print()

    print(
        "THIS RUN IS FROZEN:"
    )

    print(
        "  1000 steps"
    )

    print(
        "  original balanced two-map corpus"
    )

    print(
        "  original historical schedule"
    )

    print(
        "  one-map initialization"
    )

    print(
        "  fresh AdamW"
    )

    print(
        "  exact per-token answer-loss substitution"
    )

    print(
        "  margin = 0.5"
    )

    print(
        "  no early stopping"
    )

    print(
        "  no positive-control evaluation in this script"
    )

    print(
        "  sealed remains closed"
    )

    preflight = safety_and_preflight_check()

    (
        tokenizer,
        records,
        store,
    ) = reconstruct_training()

    pretraining_schedule_sha = (
        preverify_schedule(
            store,
            records,
        )
    )

    if (
        pretraining_schedule_sha
        != EXPECTED_SCHEDULE_SHA256
    ):
        raise RuntimeError(
            "Unexpected schedule verification state."
        )

    device = torch.device(
        "cuda"
    )

    (
        model,
        optimizer,
        parameter_count,
    ) = build_model_and_optimizer(
        device
    )

    implementation_self_audit(
        model,
        store,
        records,
        device,
    )

    header(
        "TRAINING AUTHORIZED — BEGINNING FROZEN 1000-STEP RUN"
    )

    train_result = train(
        model,
        optimizer,
        store,
        records,
        device,
    )

    candidate_sha = save_candidate(
        model,
        optimizer,
        parameter_count,
        train_result,
    )

    result = {
        "experiment":
            "DaveLM v0.9 Treatment #4",

        "classification":
            (
                "TRAINING COMPLETE — "
                "POSITIVE CONTROLS NOT YET EVALUATED"
            ),

        "objective":
            (
                "candidate-set membership + "
                "0.5 correct-vs-distractor margin "
                "via exact per-token substitution"
            ),

        "start_checkpoint": {
            "path":
                str(
                    START_CHECKPOINT
                ),

            "sha256":
                EXPECTED_START_SHA256,
        },

        "preflight_artifact":
            str(
                PREFLIGHT_PATH
            ),

        "schedule_sha256":
            train_result[
                "schedule_sha256"
            ],

        "steps":
            MAX_STEPS,

        "batch_size":
            BATCH_SIZE,

        "supervised_tokens":
            train_result[
                "supervised_tokens"
            ],

        "answer_substitutions":
            train_result[
                "answer_substitutions"
            ],

        "nonanswer_supervised_tokens":
            (
                train_result[
                    "supervised_tokens"
                ]
                - train_result[
                    "answer_substitutions"
                ]
            ),

        "candidate_checkpoint": {
            "path":
                str(
                    FINAL_CHECKPOINT
                ),

            "sha256":
                candidate_sha,
        },

        "training_log":
            train_result[
                "training_log"
            ],

        "positive_controls_evaluated":
            False,

        "sealed_axes_loaded":
            False,

        "sealed_evaluation":
            False,
    }

    write_json(
        RESULT_PATH,
        result,
    )

    header(
        "TREATMENT #4 TRAINING COMPLETE"
    )

    print(
        "Candidate checkpoint:"
    )

    print(
        FINAL_CHECKPOINT
    )

    print()

    print(
        "SHA256:"
    )

    print(
        candidate_sha
    )

    print()

    print(
        f"Historical schedule SHA: "
        f"{train_result['schedule_sha256']}"
    )

    print(
        f"Supervised tokens: "
        f"{train_result['supervised_tokens']}"
    )

    print(
        f"Answer substitutions: "
        f"{train_result['answer_substitutions']}"
    )

    print()

    print(
        "CLASSIFICATION:"
    )

    print(
        "TRAINING COMPLETE — POSITIVE CONTROLS NOT YET EVALUATED"
    )

    print()

    print(
        "SEALED AXES REMAIN CLOSED."
    )

    print()

    print(
        "DO NOT interpret capability until native positive controls run."
    )


if __name__ == "__main__":
    main()