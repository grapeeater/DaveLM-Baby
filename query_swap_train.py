from __future__ import annotations

import json
import math
import hashlib
import sys
from pathlib import Path

import torch


# =============================================================================
# DaveLM v0.9 — Treatment #2
# PAIRED QUERY-SWAP AUGMENTATION + FORCED PAIRING
#
# APPROVED FROZEN CONTRACT
#
# Initialization:
#   successful one-map checkpoint only
#
# Training:
#   1000 optimizer steps
#   batch = 16 original base documents + their 16 exact query-swapped twins
#   ordinary document causal CE
#   fresh AdamW
#   lr = 0.0003
#   weight decay = 0.05
#   grad clip = 2.0
#
# Pair invariant:
#   SAME mappings/order/filler/prefix/between/document length/query position
#   FLIP query slot/source and therefore correct target
#
# Positive controls:
#   trained_anchors >= 95%
#   supported_relation_withheld_geometry >= 90%
#
# SEALED EVALUATION IS NEVER AUTOMATICALLY OPENED.
# Even on a positive-control PASS, this trainer stops.
# =============================================================================


# =============================================================================
# IMPORT THE ALREADY-PASSED PREFLIGHT
#
# This prevents us from maintaining a second subtly different implementation
# of counterpart construction / leakage auditing / schedule hashing.
# =============================================================================

CADAVER_ROOT = Path(r"C:\DaveLM-CADAVER")

if str(CADAVER_ROOT) not in sys.path:
    sys.path.insert(0, str(CADAVER_ROOT))

import query_swap_preflight as pf


# =============================================================================
# FROZEN TREATMENT CONSTANTS
# =============================================================================

SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")

OUTPUT_ROOT = Path(
    r"C:\DaveLM-CADAVER\two_mapping_query_swap_manual_seed8380"
)

CHECKPOINT_DIR = (
    OUTPUT_ROOT
    / "checkpoints"
    / "paired_query_swap"
    / "seed_8380"
)

FINAL_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"

RESULTS_PATH = (
    OUTPUT_ROOT
    / "positive_control_results.json"
)

TRAINING_LOG_PATH = (
    OUTPUT_ROOT
    / "training_log.json"
)

PREFLIGHT_PATH = (
    OUTPUT_ROOT
    / "query_swap_preflight.json"
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

EXPECTED_PAIRED_SCHEDULE_SHA256 = (
    "4f26e9a72302be779a0b9edf25c8307b"
    "4ec9071712f49ee75763701be4c7b5e6"
)


SEED = 8380
MINIBATCH_SEED = 8381

MAX_STEPS = 1000
BASES_PER_STEP = 16
COUNTERPARTS_PER_STEP = 16
TOTAL_BATCH_SIZE = 32

CONTEXT_SIZE = 256

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.05
GRAD_CLIP = 2.0

EXPECTED_PARAMETER_COUNT = 10_594_944

ANCHOR_GATE = 0.95
SUPPORTED_GATE = 0.90

EXPECTED_ANCHORS = 256
EXPECTED_SUPPORTED = 1536

EXPECTED_TOTAL_SUPERVISED_TOKENS = 6_176_000


# =============================================================================
# IMPORT THE ORIGINAL v0.9 TRAINING IMPLEMENTATION
#
# We intentionally reuse the exact model/loss/clipping machinery already
# used by the balanced two-map experiment instead of rewriting those pieces.
# =============================================================================

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import experiments.two_mapping_contextual_binding.run as original_run


# =============================================================================
# BASIC HELPERS
# =============================================================================

def header(title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)
    print()


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def require_callable(module, name: str):
    value = getattr(
        module,
        name,
        None,
    )

    if value is None or not callable(value):
        raise RuntimeError(
            f"Required original training function "
            f"{name!r} is unavailable."
        )

    return value


# =============================================================================
# SOURCE API VALIDATION
# =============================================================================

def validate_original_training_api():

    header(
        "VALIDATING ORIGINAL v0.9 TRAINING API"
    )

    build_model = require_callable(
        original_run,
        "build_model",
    )

    document_causal_loss = require_callable(
        original_run,
        "document_causal_loss",
    )

    clip_gradients = require_callable(
        original_run,
        "clip_gradients",
    )

    sample_document_batch = require_callable(
        original_run,
        "sample_document_batch",
    )

    DocumentTokenStore = getattr(
        original_run,
        "DocumentTokenStore",
        None,
    )

    if DocumentTokenStore is None:
        raise RuntimeError(
            "Original run module does not expose "
            "DocumentTokenStore."
        )

    print("build_model:           PASS")
    print("document_causal_loss: PASS")
    print("clip_gradients:       PASS")
    print("sample_document_batch:PASS")
    print("DocumentTokenStore:   PASS")

    return (
        build_model,
        document_causal_loss,
        clip_gradients,
        sample_document_batch,
        DocumentTokenStore,
    )


# =============================================================================
# SAFETY / OUTPUT PROTECTION
# =============================================================================

def safety_check():

    header("TRAINING SAFETY CHECK")

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
            "Passed Treatment #2 preflight artifact is missing: "
            f"{PREFLIGHT_PATH}"
        )

    source_resolved = SOURCE_ROOT.resolve()
    output_resolved = OUTPUT_ROOT.resolve()

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
            f"TREATMENT CHECKPOINT:\n{FINAL_CHECKPOINT}"
        )

    if RESULTS_PATH.exists():
        raise RuntimeError(
            "REFUSING TO OVERWRITE EXISTING "
            f"RESULTS FILE:\n{RESULTS_PATH}"
        )

    actual_start_sha = sha256_file(
        START_CHECKPOINT
    )

    if (
        actual_start_sha
        != EXPECTED_START_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "Starting successful one-map checkpoint SHA "
            "does not match the frozen treatment contract."
        )

    preflight_payload = json.loads(
        PREFLIGHT_PATH.read_text(
            encoding="utf-8",
        )
    )

    if not preflight_payload.get(
        "passed",
        False,
    ):
        raise RuntimeError(
            "Treatment #2 preflight artifact does not "
            "report PASS."
        )

    preflight_schedule_sha = (
        preflight_payload
        .get(
            "paired_schedule",
            {},
        )
        .get(
            "sha256"
        )
    )

    if (
        preflight_schedule_sha
        != EXPECTED_PAIRED_SCHEDULE_SHA256
    ):
        raise RuntimeError(
            "Preflight schedule SHA does not match "
            "the now-frozen Treatment #2 SHA."
        )

    overlap = (
        preflight_payload
        .get(
            "positive_control_leakage_audit",
            {},
        )
        .get(
            "intersection_count"
        )
    )

    if overlap != 0:
        raise RuntimeError(
            "Preflight artifact does not prove zero "
            "supported-withheld overlap."
        )

    if preflight_payload.get(
        "sealed_evaluation_opened",
        True,
    ):
        raise RuntimeError(
            "Preflight artifact claims sealed evaluation "
            "was opened."
        )

    print(
        f"Protected source:   {SOURCE_ROOT}"
    )

    print(
        f"Treatment output:   {OUTPUT_ROOT}"
    )

    print()

    print(
        "Starting checkpoint SHA: PASS"
    )

    print(
        "Preflight artifact:       PASS"
    )

    print(
        "Preflight leakage count:  0"
    )

    print(
        "Frozen paired schedule:   PASS"
    )

    print(
        "Existing treatment result: NONE"
    )

    print(
        "Existing treatment checkpoint: NONE"
    )

    return preflight_payload


# =============================================================================
# REBUILD + REVERIFY TREATMENT BEFORE MODEL CONSTRUCTION
# =============================================================================

def rebuild_treatment_material():

    header(
        "REVERIFYING QUERY-SWAP TREATMENT BEFORE TRAINING"
    )

    (
        tokenizer,
        identities,
        filler,
        geometry_schedule,
        original_records,
    ) = pf.reconstruct_original()

    (
        counterparts,
        distance_deltas,
    ) = pf.build_counterparts(
        tokenizer,
        identities,
        filler,
        geometry_schedule,
        original_records,
    )

    supported_records = (
        pf.load_supported_positive_control()
    )

    leakage = pf.audit_supported_leakage(
        counterparts,
        supported_records,
    )

    if leakage["intersection_count"] != 0:
        raise RuntimeError(
            "Treatment-integrity leakage audit failed."
        )

    print()
    print(
        "Treatment reconstruction immediately "
        "before training: PASS"
    )

    return (
        tokenizer,
        identities,
        filler,
        original_records,
        counterparts,
        supported_records,
        distance_deltas,
    )


# =============================================================================
# LOAD TRAINED ANCHORS — ALLOWED POSITIVE CONTROL ONLY
#
# Like the preflight, this deserializes only the explicitly named
# positive-control value from the corpus JSON.
# =============================================================================

def load_trained_anchors():

    anchors = pf.load_named_json_value(
        pf.config.CORPUS_PATH,
        "trained_anchors",
    )

    if not isinstance(
        anchors,
        list,
    ):
        raise RuntimeError(
            "trained_anchors did not decode as a list."
        )

    if len(anchors) != EXPECTED_ANCHORS:
        raise RuntimeError(
            f"Expected {EXPECTED_ANCHORS} trained anchors; "
            f"got {len(anchors)}."
        )

    print(
        f"Loaded allowed positive-control "
        f"trained_anchors: {len(anchors)}"
    )

    return anchors


# =============================================================================
# DOCUMENT TEXT DISCOVERY
#
# _record() rows are consumed successfully by DocumentTokenStore.
# We avoid assuming a single historical key name here.
# =============================================================================

def get_document_text(
    tokenizer,
    row: dict,
) -> str:

    preferred_keys = (
        "text",
        "document",
        "document_text",
        "constant_text",
    )

    expected_length = int(
        row.get(
            "document_token_count",
            -1,
        )
    )

    for key in preferred_keys:

        value = row.get(key)

        if not isinstance(
            value,
            str,
        ):
            continue

        encoded = tokenizer.encode(
            value
        ).ids

        if (
            expected_length < 0
            or len(encoded) == expected_length
        ):
            return value

    # Conservative fallback:
    # find exactly one string field whose tokenization equals
    # document_token_count.
    candidates = []

    for key, value in row.items():

        if not isinstance(
            value,
            str,
        ):
            continue

        try:
            encoded = tokenizer.encode(
                value
            ).ids

        except Exception:
            continue

        if len(encoded) == expected_length:
            candidates.append(
                (
                    key,
                    value,
                )
            )

    if len(candidates) != 1:
        raise RuntimeError(
            "Could not uniquely identify full document text. "
            f"Matching fields: {[x[0] for x in candidates]}"
        )

    return candidates[0][1]


# =============================================================================
# IDENTITY TOKEN HELPERS
# =============================================================================

def identity_token_id(
    tokenizer,
    identities,
    identity_index: int,
) -> int:

    identity_text = identities[
        int(identity_index)
    ]

    ids = tokenizer.encode(
        identity_text
    ).ids

    if len(ids) != 1:
        raise RuntimeError(
            f"Identity index {identity_index} "
            f"is not exactly one token: {ids}"
        )

    return int(ids[0])


# =============================================================================
# POSITIVE-CONTROL EXACT EVALUATION
#
# We evaluate the next-token answer directly.
#
# The correct answer identity appears after the physical query.
# Padding/filler identities belong to a separate filler pool.
#
# We therefore:
#   1. tokenize the frozen document
#   2. locate the correct identity token occurrence AFTER query_index
#   3. require exactly one such occurrence
#   4. feed all tokens before it
#   5. score the next token
#
# This avoids touching any sealed evaluation helper.
# =============================================================================

@torch.no_grad()
def evaluate_axis(
    *,
    name: str,
    model,
    device,
    tokenizer,
    identities,
    records,
):

    header(
        f"EVALUATING POSITIVE CONTROL: {name}"
    )

    model.eval()

    correct = 0
    wrong = 0

    distractor_errors = 0

    predictions = []

    for index, row in enumerate(
        records
    ):

        text = get_document_text(
            tokenizer,
            row,
        )

        token_ids = tokenizer.encode(
            text
        ).ids

        query_index = int(
            row["layout"]["query_index"]
        )

        target_index = int(
            row["target_identity_index"]
        )

        distractor_target_index = int(
            row["distractor_target_index"]
        )

        target_token = identity_token_id(
            tokenizer,
            identities,
            target_index,
        )

        distractor_token = identity_token_id(
            tokenizer,
            identities,
            distractor_target_index,
        )

        target_positions_after_query = [
            position
            for position, token_id
            in enumerate(token_ids)
            if (
                position > query_index
                and int(token_id) == target_token
            )
        ]

        if (
            len(target_positions_after_query)
            != 1
        ):
            raise RuntimeError(
                f"{name} row {index}: expected exactly one "
                f"correct target occurrence after query; "
                f"found {target_positions_after_query}"
            )

        answer_position = (
            target_positions_after_query[0]
        )

        if answer_position <= 0:
            raise RuntimeError(
                f"{name} row {index}: invalid answer position."
            )

        prefix = token_ids[
            :answer_position
        ]

        if len(prefix) > CONTEXT_SIZE:
            prefix = prefix[
                -CONTEXT_SIZE:
            ]

        inputs = torch.tensor(
            [prefix],
            dtype=torch.long,
            device=device,
        )

        logits = model(
            inputs
        )

        prediction = int(
            torch.argmax(
                logits[
                    0,
                    -1,
                    :
                ]
            ).item()
        )

        is_correct = (
            prediction == target_token
        )

        if is_correct:
            correct += 1

        else:
            wrong += 1

            if prediction == distractor_token:
                distractor_errors += 1

        predictions.append(
            {
                "row_index":
                    index,

                "source_identity_index":
                    int(
                        row[
                            "source_identity_index"
                        ]
                    ),

                "target_identity_index":
                    target_index,

                "distractor_target_index":
                    distractor_target_index,

                "query_slot":
                    int(
                        row["query_slot"]
                    ),

                "query_index":
                    query_index,

                "target_to_query_distance":
                    int(
                        row["layout"][
                            "target_to_query_distance"
                        ]
                    ),

                "prediction_token_id":
                    prediction,

                "target_token_id":
                    target_token,

                "distractor_target_token_id":
                    distractor_token,

                "correct":
                    is_correct,

                "predicted_distractor":
                    (
                        prediction
                        == distractor_token
                    ),
            }
        )

    total = len(records)

    accuracy = (
        correct / total
        if total
        else 0.0
    )

    distractor_share = (
        distractor_errors / wrong
        if wrong
        else 0.0
    )

    print(
        f"Correct: {correct}/{total} "
        f"= {accuracy:.3%}"
    )

    print(
        f"Incorrect: {wrong}"
    )

    print(
        f"Distractor-target errors: "
        f"{distractor_errors}/{wrong} "
        f"= {distractor_share:.3%}"
        if wrong
        else
        "Distractor-target errors: 0/0"
    )

    return {
        "name":
            name,

        "correct":
            correct,

        "total":
            total,

        "accuracy":
            accuracy,

        "incorrect":
            wrong,

        "distractor_target_errors":
            distractor_errors,

        "distractor_target_share_among_errors":
            distractor_share,

        "predictions":
            predictions,
    }


# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

def build_initialized_model(
    build_model,
    device,
):

    header(
        "INITIALIZING FROM VERIFIED ONE-MAP CHECKPOINT"
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
        checkpoint["model_state"],
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
# FROZEN SCHEDULE PRE-CHECK
# =============================================================================

def verify_frozen_schedule(
    base_store,
):

    header(
        "VERIFYING FROZEN PAIRED SCHEDULE BEFORE OPTIMIZER CREATION"
    )

    schedule = (
        pf.build_paired_schedule_digest(
            base_store
        )
    )

    observed = schedule[
        "sha256"
    ]

    print(
        "Expected:"
    )

    print(
        EXPECTED_PAIRED_SCHEDULE_SHA256
    )

    print()

    print(
        "Observed:"
    )

    print(
        observed
    )

    if (
        observed
        != EXPECTED_PAIRED_SCHEDULE_SHA256
    ):
        raise RuntimeError(
            "Frozen paired schedule SHA mismatch."
        )

    if (
        int(
            schedule[
                "total_supervised_tokens"
            ]
        )
        !=
        EXPECTED_TOTAL_SUPERVISED_TOKENS
    ):
        raise RuntimeError(
            "Frozen paired schedule supervised-token "
            "count changed."
        )

    print()
    print(
        "Frozen paired schedule: PASS"
    )

    print(
        "Expected supervised causal tokens: "
        f"{EXPECTED_TOTAL_SUPERVISED_TOKENS}"
    )

    return schedule


# =============================================================================
# TRAINING
# =============================================================================

def train(
    *,
    model,
    device,
    base_store,
    counterpart_store,
    document_causal_loss,
    clip_gradients,
    sample_document_batch,
):

    header(
        "BEGINNING TREATMENT #2 TRAINING"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # The BASE generator defines the single frozen sampling schedule.
    #
    # At each step we clone its PRE-SAMPLE state into a temporary mirror
    # generator. The base sampler advances the canonical generator;
    # the counterpart sampler consumes the mirror state.
    #
    # Therefore both halves receive identical document indices and
    # identical within-document starts, while only the canonical base
    # generator determines the next step's schedule.
    generator = (
        torch.Generator()
        .manual_seed(
            MINIBATCH_SEED
        )
    )

    live_schedule_digest = (
        hashlib.sha256()
    )

    training_log = []

    total_supervised_tokens = 0

    model.train()

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        # ---------------------------------------------------------
        # Clone pre-sample generator state for exact twin sampling.
        # ---------------------------------------------------------

        mirror = torch.Generator()

        mirror.set_state(
            generator.get_state()
        )

        # ---------------------------------------------------------
        # Sample 16 ORIGINAL documents.
        # ---------------------------------------------------------

        (
            base_inputs,
            base_targets,
            _base_mask,
            base_meta,
        ) = sample_document_batch(
            base_store,
            BASES_PER_STEP,
            CONTEXT_SIZE,
            generator,
            device,
        )

        # ---------------------------------------------------------
        # Sample SAME 16 indices/starts from counterpart store.
        # ---------------------------------------------------------

        (
            counterpart_inputs,
            counterpart_targets,
            _counterpart_mask,
            counterpart_meta,
        ) = sample_document_batch(
            counterpart_store,
            COUNTERPARTS_PER_STEP,
            CONTEXT_SIZE,
            mirror,
            device,
        )

        base_indices = [
            int(x)
            for x
            in base_meta[
                "document_indices"
            ]
        ]

        counterpart_indices = [
            int(x)
            for x
            in counterpart_meta[
                "document_indices"
            ]
        ]

        base_starts = [
            int(x)
            for x
            in base_meta[
                "within_document_starts"
            ]
        ]

        counterpart_starts = [
            int(x)
            for x
            in counterpart_meta[
                "within_document_starts"
            ]
        ]

        if (
            base_indices
            != counterpart_indices
        ):
            raise RuntimeError(
                f"Step {step}: base/counterpart "
                "document indices diverged."
            )

        if (
            base_starts
            != counterpart_starts
        ):
            raise RuntimeError(
                f"Step {step}: base/counterpart "
                "within-document starts diverged."
            )

        live_schedule_digest.update(
            pf.schedule_payload(
                base_indices,
                base_starts,
            )
        )

        # ---------------------------------------------------------
        # Form the actual 32-document forced-pair batch.
        # ---------------------------------------------------------

        inputs = torch.cat(
            [
                base_inputs,
                counterpart_inputs,
            ],
            dim=0,
        )

        targets = torch.cat(
            [
                base_targets,
                counterpart_targets,
            ],
            dim=0,
        )

        if inputs.shape[0] != TOTAL_BATCH_SIZE:
            raise RuntimeError(
                f"Step {step}: batch does not contain 32 documents."
            )

        # ---------------------------------------------------------
        # Ordinary causal CE.
        # ---------------------------------------------------------

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            inputs
        )

        loss = document_causal_loss(
            logits,
            targets,
        )

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"Step {step}: non-finite loss."
            )

        loss.backward()

        clip_gradients(
            model,
            GRAD_CLIP,
            step,
        )

        # ---------------------------------------------------------
        # Gradient finite check BEFORE update.
        # ---------------------------------------------------------

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
                    f"in parameter {name}."
                )

        optimizer.step()

        # ---------------------------------------------------------
        # Parameter finite check AFTER update.
        # ---------------------------------------------------------

        for name, parameter in (
            model.named_parameters()
        ):

            if not torch.isfinite(
                parameter
            ).all():

                raise RuntimeError(
                    f"Step {step}: non-finite parameter "
                    f"after update: {name}."
                )

        base_supervised = int(
            base_meta[
                "supervised_tokens"
            ]
        )

        counterpart_supervised = int(
            counterpart_meta[
                "supervised_tokens"
            ]
        )

        if (
            base_supervised
            != counterpart_supervised
        ):
            raise RuntimeError(
                f"Step {step}: base/counterpart supervised "
                "token counts differ."
            )

        step_supervised = (
            base_supervised
            + counterpart_supervised
        )

        total_supervised_tokens += (
            step_supervised
        )

        loss_value = float(
            loss.detach().cpu().item()
        )

        training_log.append(
            {
                "step":
                    step,

                "loss":
                    loss_value,

                "base_supervised_tokens":
                    base_supervised,

                "counterpart_supervised_tokens":
                    counterpart_supervised,

                "total_supervised_tokens":
                    step_supervised,
            }
        )

        if (
            step == 1
            or step == 10
            or step % 100 == 0
        ):
            print(
                f"step {step:4d}/{MAX_STEPS} "
                f"loss={loss_value:.6f} "
                f"supervised_tokens={total_supervised_tokens}"
            )

    observed_live_schedule_sha = (
        live_schedule_digest.hexdigest()
    )

    print()
    print(
        "LIVE TRAINING SCHEDULE SHA256:"
    )

    print(
        observed_live_schedule_sha
    )

    if (
        observed_live_schedule_sha
        != EXPECTED_PAIRED_SCHEDULE_SHA256
    ):
        raise RuntimeError(
            "Training completed but live schedule SHA "
            "does NOT match frozen Treatment #2 schedule. "
            "REFUSING TO SAVE A CANDIDATE CHECKPOINT."
        )

    if (
        total_supervised_tokens
        != EXPECTED_TOTAL_SUPERVISED_TOKENS
    ):
        raise RuntimeError(
            "Training supervised-token total changed. "
            "REFUSING TO SAVE."
        )

    print()
    print(
        "LIVE SCHEDULE SHA: PASS"
    )

    print(
        "SUPERVISED TOKEN TOTAL: PASS"
    )

    return (
        optimizer,
        training_log,
        total_supervised_tokens,
        observed_live_schedule_sha,
    )


# =============================================================================
# CHECKPOINT SAVE
# =============================================================================

def save_candidate_checkpoint(
    *,
    model,
    optimizer,
    total_supervised_tokens,
    live_schedule_sha,
):

    header(
        "SAVING ISOLATED TREATMENT #2 CANDIDATE"
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "experiment":
            "DaveLM v0.9 Treatment #2 paired query-swap",

        "seed":
            SEED,

        "step":
            MAX_STEPS,

        "starting_checkpoint":
            str(START_CHECKPOINT),

        "starting_checkpoint_sha256":
            EXPECTED_START_CHECKPOINT_SHA256,

        "paired_schedule_sha256":
            live_schedule_sha,

        "supervised_tokens":
            total_supervised_tokens,

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

    checkpoint_sha = sha256_file(
        FINAL_CHECKPOINT
    )

    print(
        f"Candidate checkpoint:\n{FINAL_CHECKPOINT}"
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
# MAIN
# =============================================================================

def main():

    safety_check()

    (
        build_model,
        document_causal_loss,
        clip_gradients,
        sample_document_batch,
        DocumentTokenStore,
    ) = validate_original_training_api()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm/PyTorch GPU device is unavailable. "
            "REFUSING TO ACCIDENTALLY RUN Treatment #2 on CPU."
        )

    device = torch.device(
        "cuda"
    )

    header("DEVICE")

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
        identities,
        filler,
        original_records,
        counterparts,
        supported_records,
        distance_deltas,
    ) = rebuild_treatment_material()

    anchors = load_trained_anchors()

    if len(supported_records) != EXPECTED_SUPPORTED:
        raise RuntimeError(
            f"Expected {EXPECTED_SUPPORTED} supported "
            f"positive-control records; got {len(supported_records)}."
        )

    header(
        "BUILDING TRAINING DOCUMENT STORES"
    )

    base_store = (
        DocumentTokenStore
        .from_documents(
            tokenizer,
            original_records,
        )
    )

    counterpart_store = (
        DocumentTokenStore
        .from_documents(
            tokenizer,
            counterparts,
        )
    )

    verify_frozen_schedule(
        base_store
    )

    model = build_initialized_model(
        build_model,
        device,
    )

    (
        optimizer,
        training_log,
        total_supervised_tokens,
        live_schedule_sha,
    ) = train(
        model=model,
        device=device,
        base_store=base_store,
        counterpart_store=counterpart_store,
        document_causal_loss=document_causal_loss,
        clip_gradients=clip_gradients,
        sample_document_batch=sample_document_batch,
    )

    checkpoint_sha = (
        save_candidate_checkpoint(
            model=model,
            optimizer=optimizer,
            total_supervised_tokens=total_supervised_tokens,
            live_schedule_sha=live_schedule_sha,
        )
    )

    write_json(
        TRAINING_LOG_PATH,
        {
            "experiment":
                "DaveLM v0.9 Treatment #2 paired query-swap",

            "steps":
                MAX_STEPS,

            "schedule_sha256":
                live_schedule_sha,

            "total_supervised_tokens":
                total_supervised_tokens,

            "log":
                training_log,
        },
    )

    # =============================================================
    # POSITIVE CONTROLS ONLY.
    #
    # No novel/sealed axis is loaded anywhere in this trainer.
    # =============================================================

    anchors_result = evaluate_axis(
        name="trained_anchors",
        model=model,
        device=device,
        tokenizer=tokenizer,
        identities=identities,
        records=anchors,
    )

    supported_result = evaluate_axis(
        name="supported_relation_withheld_geometry",
        model=model,
        device=device,
        tokenizer=tokenizer,
        identities=identities,
        records=supported_records,
    )

    anchors_pass = (
        anchors_result["accuracy"]
        >= ANCHOR_GATE
    )

    supported_pass = (
        supported_result["accuracy"]
        >= SUPPORTED_GATE
    )

    positive_control_pass = (
        anchors_pass
        and supported_pass
    )

    if positive_control_pass:

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
            "DaveLM v0.9 Treatment #2 paired query-swap",

        "classification":
            classification,

        "starting_checkpoint_sha256":
            EXPECTED_START_CHECKPOINT_SHA256,

        "candidate_checkpoint":
            str(FINAL_CHECKPOINT),

        "candidate_checkpoint_sha256":
            checkpoint_sha,

        "paired_schedule_sha256":
            live_schedule_sha,

        "training": {
            "seed":
                SEED,

            "steps":
                MAX_STEPS,

            "batch_size":
                TOTAL_BATCH_SIZE,

            "base_documents_per_step":
                BASES_PER_STEP,

            "counterpart_documents_per_step":
                COUNTERPARTS_PER_STEP,

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "gradient_clip":
                GRAD_CLIP,

            "total_supervised_causal_tokens":
                total_supervised_tokens,
        },

        "counterpart_distance_deltas":
            {
                str(delta): count
                for delta, count
                in sorted(
                    distance_deltas.items()
                )
            },

        "positive_controls": {
            "trained_anchors": {
                key: value
                for key, value
                in anchors_result.items()
                if key != "predictions"
            },

            "supported_relation_withheld_geometry": {
                key: value
                for key, value
                in supported_result.items()
                if key != "predictions"
            },

            "anchor_gate":
                ANCHOR_GATE,

            "supported_gate":
                SUPPORTED_GATE,

            "anchor_gate_pass":
                anchors_pass,

            "supported_gate_pass":
                supported_pass,

            "overall_gate_pass":
                positive_control_pass,
        },

        "detailed_predictions": {
            "trained_anchors":
                anchors_result[
                    "predictions"
                ],

            "supported_relation_withheld_geometry":
                supported_result[
                    "predictions"
                ],
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
        "FINAL TREATMENT #2 RESULT"
    )

    print(
        "Trained anchors:"
    )

    print(
        f"{anchors_result['correct']}/"
        f"{anchors_result['total']} "
        f"= {anchors_result['accuracy']:.3%}"
    )

    print(
        f"Gate: {ANCHOR_GATE:.0%} "
        f"-> {'PASS' if anchors_pass else 'FAIL'}"
    )

    print()

    print(
        "Supported relation / withheld geometry:"
    )

    print(
        f"{supported_result['correct']}/"
        f"{supported_result['total']} "
        f"= {supported_result['accuracy']:.3%}"
    )

    print(
        f"Gate: {SUPPORTED_GATE:.0%} "
        f"-> {'PASS' if supported_pass else 'FAIL'}"
    )

    print()

    print(
        "Anchor distractor-target share among errors:"
    )

    print(
        f"{anchors_result['distractor_target_errors']}/"
        f"{anchors_result['incorrect']} "
        f"= "
        f"{anchors_result['distractor_target_share_among_errors']:.3%}"
        if anchors_result["incorrect"]
        else
        "0/0"
    )

    print()

    print(
        "Supported distractor-target share among errors:"
    )

    print(
        f"{supported_result['distractor_target_errors']}/"
        f"{supported_result['incorrect']} "
        f"= "
        f"{supported_result['distractor_target_share_among_errors']:.3%}"
        if supported_result["incorrect"]
        else
        "0/0"
    )

    print()

    print(
        f"CLASSIFICATION:"
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

    if positive_control_pass:

        print()
        print(
            "Both frozen positive-control gates passed."
        )

        print(
            "STOP HERE. This checkpoint is only ELIGIBLE "
            "for a separately authorized sealed evaluation."
        )

    else:

        print()
        print(
            "At least one frozen positive-control gate failed."
        )

        print(
            "STOP HERE. Do not continue, rescue, extend, "
            "or open sealed evaluation."
        )


if __name__ == "__main__":
    main()