from __future__ import annotations

import sys
import json
import time
import hashlib
from pathlib import Path
from collections import Counter

import torch
from tokenizers import Tokenizer


# =============================================================================
# BABY CURRICULUM BRIDGE
#
# One-map successful checkpoint
#          ↓
# Steps 1-500:
#   two-map task with queried/correct mapping always displayed FIRST
#          ↓
# Steps 501-1000:
#   exact original balanced two-map task
#
# IMPORTANT:
#   First run with MODE = "PREFLIGHT".
#
#   PREFLIGHT performs ZERO optimizer steps.
#   It verifies corpus construction, checkpoint identity,
#   and the complete frozen 1,000-step minibatch schedule.
#
#   It NEVER opens sealed evaluation axes.
# =============================================================================


MODE = "TRAIN"

# After PREFLIGHT passes, WE will change this to:
#
# MODE = "TRAIN"
#
# Do not change it yet.


# =============================================================================
# PATHS
# =============================================================================

SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")

OUTPUT_ROOT = Path(
    r"C:\DaveLM-CADAVER\two_mapping_curriculum_bridge_manual_seed8380"
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
    "a34bb91d258f11c6f7cb6b6cd89c79"
    "d8e6fa12d64af15dd14b028fb7ff675cff"
)

CHECKPOINT_PATH = (
    OUTPUT_ROOT
    / "checkpoints"
    / "curriculum"
    / "seed_8380"
    / "latest.pt"
)

RESULT_PATH = (
    OUTPUT_ROOT
    / "positive_control_results.json"
)

PREFLIGHT_PATH = (
    OUTPUT_ROOT
    / "preflight.json"
)


# =============================================================================
# FROZEN TRAINING SETTINGS
# =============================================================================

SEED = 8380
MINIBATCH_SEED = 8381

TOTAL_STEPS = 1000
CURRICULUM_STEPS = 500

BATCH_SIZE = 32
CONTEXT_SIZE = 256

LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.05
GRADIENT_CLIP = 2.0

EXPECTED_PARAMETER_COUNT = 10_594_944

ANCHOR_GATE = 0.95
SUPPORTED_GATE = 0.90


# =============================================================================
# IMPORT EXACT v0.9 MACHINERY
# =============================================================================

sys.path.insert(
    0,
    str(SOURCE_ROOT),
)

from experiments.two_mapping_contextual_binding import config

from experiments.two_mapping_contextual_binding.run import (
    _identity_pool,
    _layout_tables,
    _training_records,
    _geometry_schedule,
    _ordered_training_keys,
    _training_pair,
    _distractor_pair,
    _record,
    _evaluate,
    _small_anchor,
    _model_digest,
)

from v0_8.document_sampling import (
    DocumentTokenStore,
    document_causal_loss,
    sample_document_batch,
)

from v0_8_2.model import (
    build_model,
)

from v0_7.model import (
    trainable_parameter_count,
)

from v0_2_1.stability import (
    clip_gradients,
    require_finite_model,
    require_finite_optimizer,
    require_finite_tensor,
)

from v0_8_3.protection import (
    strict_json_dumps,
)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def file_sha256(path: Path) -> str:
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


def mapping_pairs(row):
    return [
        (
            int(item["source_index"]),
            int(item["target_index"]),
        )
        for item in row["mappings"]
    ]


def relation_set(row):
    return frozenset(
        mapping_pairs(row)
    )


def schedule_payload(meta):
    return strict_json_dumps(
        {
            "document_indices":
                meta["document_indices"],

            "within_document_starts":
                meta["within_document_starts"],
        }
    ).encode(
        "ascii"
    )


def save_json(
    path: Path,
    payload,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def print_header(title):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)
    print()


# =============================================================================
# SAFETY
# =============================================================================

def safety_check():

    if MODE not in {
        "PREFLIGHT",
        "TRAIN",
    }:
        raise RuntimeError(
            "MODE must be PREFLIGHT or TRAIN."
        )

    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(
            SOURCE_ROOT
        )

    if not START_CHECKPOINT.is_file():
        raise FileNotFoundError(
            START_CHECKPOINT
        )

    # Absolutely refuse to place our output
    # anywhere inside the real v0.9 tree.

    source_resolved = SOURCE_ROOT.resolve()
    output_resolved = OUTPUT_ROOT.resolve()

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

    print_header(
        "SAFETY"
    )

    print(
        f"MODE: {MODE}"
    )

    print(
        f"Protected source: {SOURCE_ROOT}"
    )

    print(
        f"Disposable output: {OUTPUT_ROOT}"
    )

    print(
        "Sealed evaluation: DISABLED"
    )

    if MODE == "PREFLIGHT":
        print(
            "Optimizer steps: ZERO"
        )

    else:
        print(
            "Optimizer steps: 1000"
        )

    print()


# =============================================================================
# LOAD TOKENIZER + RECONSTRUCT ORIGINAL CORPUS
# =============================================================================

def reconstruct():

    print_header(
        "RECONSTRUCTING ORIGINAL TWO-MAP EXPERIMENT"
    )

    tokenizer = Tokenizer.from_file(
        str(
            config.TOKENIZER_PATH
        )
    )

    groups = _identity_pool(
        tokenizer
    )

    identities = groups[
        "identity"
    ]

    filler = groups[
        "filler"
    ]

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

    geometry_schedule = (
        _geometry_schedule(
            tables
        )
    )

    keys = (
        _ordered_training_keys()
    )

    original = (
        _training_records(
            tokenizer,
            identities,
            filler,
            tables,
        )
    )

    if (
        len(geometry_schedule)
        != 1536
    ):
        raise RuntimeError(
            "Expected 1536 geometry rows."
        )

    if len(keys) != 1536:
        raise RuntimeError(
            "Expected 1536 training keys."
        )

    if len(original) != 1536:
        raise RuntimeError(
            "Expected 1536 training records."
        )

    print(
        "Identity pool: 64"
    )

    print(
        "Filler pool:   64"
    )

    print(
        "Geometry:      1536"
    )

    print(
        "Training docs: 1536"
    )

    return (
        tokenizer,
        groups,
        identities,
        filler,
        tables,
        geometry_schedule,
        keys,
        original,
    )


# =============================================================================
# BUILD PHASE 1
# =============================================================================

def build_phase1(
    tokenizer,
    identities,
    filler,
    geometry_schedule,
    keys,
    original,
):

    print_header(
        "BUILDING PHASE 1 CURRICULUM"
    )

    phase1 = []

    unchanged = 0
    swapped = 0

    distance_deltas = Counter()

    for index, (
        round_index,
        source_index,
    ) in enumerate(
        keys
    ):

        old = original[index]

        geometry = (
            geometry_schedule[index]
        )

        relevant = (
            _training_pair(
                round_index,
                source_index,
            )
        )

        (
            distractor_source,
            distractor_target,
            distractor_round,
        ) = _distractor_pair(
            round_index,
            source_index,
        )

        old_slot = int(
            old["query_slot"]
        )

        # -----------------------------------------------------
        # Preserve the original low-level filler geometry.
        #
        # The OLD target_to_query_distance is NOT enforced,
        # because moving the queried mapping from slot 1 to
        # slot 0 necessarily changes that distance.
        # -----------------------------------------------------

        phase_geometry = {
            "query_slot": 0,

            "query_index":
                int(
                    geometry[
                        "query_index"
                    ]
                ),

            "target_to_query_distance":
                int(
                    geometry[
                        "target_to_query_distance"
                    ]
                ),

            "prefix_count":
                int(
                    geometry[
                        "prefix_count"
                    ]
                ),

            "between_count":
                int(
                    geometry[
                        "between_count"
                    ]
                ),

            "enforce": False,
        }

        mappings = [
            relevant,
            (
                distractor_source,
                distractor_target,
            ),
        ]

        new = _record(
            tokenizer,
            "train",
            index,
            identities,
            filler,
            mappings,
            0,
            phase_geometry,
            training=True,
        )

        new[
            "relation_round"
        ] = round_index

        new[
            "distractor_relation_round"
        ] = distractor_round

        # -----------------------------------------------------
        # UNIVERSAL INVARIANTS
        # -----------------------------------------------------

        if (
            relation_set(old)
            != relation_set(new)
        ):
            raise RuntimeError(
                f"Relation set changed "
                f"at record {index}."
            )

        if (
            int(
                old[
                    "source_identity_index"
                ]
            )
            != int(
                new[
                    "source_identity_index"
                ]
            )
        ):
            raise RuntimeError(
                f"Queried source changed "
                f"at record {index}."
            )

        if (
            int(
                old[
                    "target_identity_index"
                ]
            )
            != int(
                new[
                    "target_identity_index"
                ]
            )
        ):
            raise RuntimeError(
                f"Correct target changed "
                f"at record {index}."
            )

        if (
            int(
                old[
                    "distractor_source_index"
                ]
            )
            != int(
                new[
                    "distractor_source_index"
                ]
            )
        ):
            raise RuntimeError(
                f"Distractor source changed "
                f"at record {index}."
            )

        if (
            int(
                old[
                    "distractor_target_index"
                ]
            )
            != int(
                new[
                    "distractor_target_index"
                ]
            )
        ):
            raise RuntimeError(
                f"Distractor target changed "
                f"at record {index}."
            )

        if (
            int(
                old[
                    "document_token_count"
                ]
            )
            != 192
        ):
            raise RuntimeError(
                "Original document length "
                "changed."
            )

        if (
            int(
                new[
                    "document_token_count"
                ]
            )
            != 192
        ):
            raise RuntimeError(
                "Phase-1 document length "
                "changed."
            )

        if (
            int(
                old["layout"][
                    "query_index"
                ]
            )
            != int(
                new["layout"][
                    "query_index"
                ]
            )
        ):
            raise RuntimeError(
                f"Query position changed "
                f"at record {index}."
            )

        # -----------------------------------------------------
        # SLOT-SPECIFIC INVARIANTS
        # -----------------------------------------------------

        if old_slot == 0:

            unchanged += 1

            if (
                old["text"]
                != new["text"]
            ):
                raise RuntimeError(
                    f"Already-first record "
                    f"{index} changed."
                )

        elif old_slot == 1:

            swapped += 1

            old_pairs = (
                mapping_pairs(old)
            )

            new_pairs = (
                mapping_pairs(new)
            )

            if new_pairs != [
                old_pairs[1],
                old_pairs[0],
            ]:
                raise RuntimeError(
                    f"Mapping swap failed "
                    f"at record {index}."
                )

        else:
            raise RuntimeError(
                f"Unexpected slot "
                f"{old_slot}."
            )

        old_distance = int(
            old["layout"][
                "target_to_query_distance"
            ]
        )

        new_distance = int(
            new["layout"][
                "target_to_query_distance"
            ]
        )

        delta = (
            new_distance
            - old_distance
        )

        distance_deltas[
            (
                old_slot,
                delta,
            )
        ] += 1

        phase1.append(
            new
        )

    if unchanged != 768:
        raise RuntimeError(
            f"Expected 768 unchanged; "
            f"got {unchanged}."
        )

    if swapped != 768:
        raise RuntimeError(
            f"Expected 768 swapped; "
            f"got {swapped}."
        )

    if distance_deltas != Counter(
        {
            (0, 0): 768,
            (1, 12): 768,
        }
    ):
        raise RuntimeError(
            "Unexpected distance-delta "
            f"distribution: "
            f"{distance_deltas}"
        )

    if any(
        int(row["query_slot"])
        != 0
        for row in phase1
    ):
        raise RuntimeError(
            "Phase 1 was not fully "
            "canonicalized."
        )

    print(
        "Already-first unchanged: 768"
    )

    print(
        "Originally-second swapped: 768"
    )

    print(
        "Phase-1 query slot 0: 1536"
    )

    print(
        "Distance delta slot 0: "
        "0 × 768"
    )

    print(
        "Distance delta slot 1: "
        "+12 × 768"
    )

    print(
        "Document lengths: "
        "192 × 1536"
    )

    print()
    print(
        "PHASE 1 CORPUS: PASS"
    )

    return phase1


# =============================================================================
# BUILD BOTH DOCUMENT STORES
# =============================================================================

def build_stores(
    tokenizer,
    original,
    phase1,
):

    print_header(
        "BUILDING DOCUMENT STORES"
    )

    phase1_store = (
        DocumentTokenStore
        .from_documents(
            tokenizer,
            phase1,
        )
    )

    phase2_store = (
        DocumentTokenStore
        .from_documents(
            tokenizer,
            original,
        )
    )

    print(
        "Phase 1 store: 1536 "
        "canonicalized documents"
    )

    print(
        "Phase 2 store: 1536 "
        "original documents"
    )

    return (
        phase1_store,
        phase2_store,
    )


# =============================================================================
# CHECK STARTING CHECKPOINT
# =============================================================================

def inspect_start_checkpoint():

    print_header(
        "CHECKING ONE-MAP STARTING BRAIN"
    )

    actual_sha = file_sha256(
        START_CHECKPOINT
    )

    print(
        "Checkpoint:"
    )

    print(
        START_CHECKPOINT
    )

    print()

    print(
        "SHA256:"
    )

    print(
        actual_sha
    )

    if (
        actual_sha
        != EXPECTED_START_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "Starting checkpoint SHA "
            "does not match the frozen "
            "successful one-map checkpoint."
        )

    payload = torch.load(
        START_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if (
        int(
            payload[
                "parameter_count"
            ]
        )
        != EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "Starting checkpoint "
            "architecture changed."
        )

    if "model_state" not in payload:
        raise RuntimeError(
            "Starting checkpoint has "
            "no model_state."
        )

    print()
    print(
        "Checkpoint SHA: PASS"
    )

    print(
        "Parameter count: PASS"
    )

    print(
        "Starting brain: SUCCESSFUL "
        "ONE-MAP CHECKPOINT"
    )

    return payload


# =============================================================================
# VERIFY THE COMPLETE 1000-STEP SCHEDULE
# =============================================================================

def verify_schedule(
    phase1_store,
    phase2_store,
):

    print_header(
        "VERIFYING FROZEN 1000-STEP MINIBATCH SCHEDULE"
    )

    generator = (
        torch.Generator()
        .manual_seed(
            MINIBATCH_SEED
        )
    )

    digest = hashlib.sha256()

    phase1_supervised = 0
    phase2_supervised = 0

    for step in range(
        1,
        TOTAL_STEPS + 1,
    ):

        store = (
            phase1_store
            if step <= CURRICULUM_STEPS
            else phase2_store
        )

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
            torch.device("cpu"),
        )

        digest.update(
            schedule_payload(
                meta
            )
        )

        if step <= CURRICULUM_STEPS:
            phase1_supervised += int(
                meta[
                    "supervised_tokens"
                ]
            )
        else:
            phase2_supervised += int(
                meta[
                    "supervised_tokens"
                ]
            )

    actual = (
        digest.hexdigest()
    )

    print(
        "Expected schedule:"
    )

    print(
        EXPECTED_SCHEDULE_SHA256
    )

    print()

    print(
        "Observed schedule:"
    )

    print(
        actual
    )

    print()

    if (
        actual
        != EXPECTED_SCHEDULE_SHA256
    ):
        raise RuntimeError(
            "MINIBATCH SCHEDULE MISMATCH."
        )

    print(
        "SCHEDULE: EXACT MATCH"
    )

    print()

    print(
        "Steps 1-500: "
        "canonicalized store"
    )

    print(
        "Steps 501-1000: "
        "original store"
    )

    print(
        "Generator restart at 500: NO"
    )

    return {
        "schedule_sha256":
            actual,

        "phase1_supervised_tokens":
            phase1_supervised,

        "phase2_supervised_tokens":
            phase2_supervised,
    }


# =============================================================================
# POSITIVE-AXIS CORPUS
#
# IMPORTANT:
# We need the ORIGINAL evaluation corpus,
# but we deliberately retain ONLY the two allowed positive axes.
#
# SEALED AXES ARE NEVER PASSED TO THE EVALUATOR.
# =============================================================================

def load_positive_evaluation_corpus():

    print_header(
        "LOADING POSITIVE-CONTROL AXES ONLY"
    )

    # The experiment's corpus file already exists
    # in protected v0.9 and is read-only here.

    if not config.CORPUS_PATH.is_file():
        raise FileNotFoundError(
            config.CORPUS_PATH
        )

    with config.CORPUS_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        full = json.load(
            handle
        )

    positive = {
        "trained_anchors":
            full["trained_anchors"],

        "supported_relation_withheld_geometry":
            full[
                "supported_relation_withheld_geometry"
            ],
    }

    print(
        "Loaded: trained_anchors"
    )

    print(
        "Loaded: "
        "supported_relation_withheld_geometry"
    )

    print(
        "Loaded sealed axes: NONE"
    )

    return positive


# =============================================================================
# DISTRACTOR CAPTURE
# =============================================================================

def distractor_capture(
    evaluation,
):

    details = evaluation.get(
        "examples_detail",
        [],
    )

    incorrect = [
        row
        for row in details
        if not bool(
            row.get(
                "exact",
                False,
            )
        )
    ]

    if not incorrect:
        return {
            "incorrect": 0,
            "distractor_target_errors": 0,
            "distractor_target_share_of_errors": 0.0,
        }

    distractor_errors = 0

    for row in incorrect:

        generated = int(
            row[
                "generated_token_id"
            ]
        )

        distractor = int(
            row[
                "distractor_token_id"
            ]
        )

        if generated == distractor:
            distractor_errors += 1

    return {
        "incorrect":
            len(incorrect),

        "distractor_target_errors":
            distractor_errors,

        "distractor_target_share_of_errors":
            (
                distractor_errors
                / len(incorrect)
            ),
    }


# =============================================================================
# TRAINING
# =============================================================================

def train(
    tokenizer,
    phase1_store,
    phase2_store,
    start_payload,
    positive_corpus,
):

    print_header(
        "BABY GOES TO CLASS"
    )

    if CHECKPOINT_PATH.exists():
        raise RuntimeError(
            "Refusing overwrite/resume: "
            f"{CHECKPOINT_PATH}"
        )

    if RESULT_PATH.exists():
        raise RuntimeError(
            "Refusing overwrite: "
            f"{RESULT_PATH}"
        )

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.manual_seed(
        SEED
    )

    torch.cuda.manual_seed_all(
        SEED
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm GPU is not available."
        )

    device = torch.device(
        "cuda"
    )

    print(
        "GPU:"
    )

    print(
        torch.cuda.get_device_name(
            device
        )
    )

    print()

    # ---------------------------------------------------------
    # SAME ARCHITECTURE
    # ---------------------------------------------------------

    model = build_model(
        "untied"
    )

    if (
        trainable_parameter_count(model)
        != EXPECTED_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "Architecture changed."
        )

    # ---------------------------------------------------------
    # LOAD ONLY MODEL STATE FROM SUCCESSFUL ONE-MAP BRAIN
    #
    # DO NOT load old optimizer state.
    # ---------------------------------------------------------

    model.load_state_dict(
        start_payload[
            "model_state"
        ],
        strict=True,
    )

    initialization_digest = (
        _model_digest(
            model
        )
    )

    model = model.to(
        device
    )

    # ---------------------------------------------------------
    # FRESH OPTIMIZER
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    generator = (
        torch.Generator()
        .manual_seed(
            MINIBATCH_SEED
        )
    )

    schedule_digest = (
        hashlib.sha256()
    )

    history = []

    supervised_tokens = 0

    torch.cuda.reset_peak_memory_stats(
        device
    )

    started = (
        time.perf_counter()
    )

    eval_steps = {
        1,
        10,
        100,
        200,
        300,
        400,
        500,
        501,
        600,
        700,
        800,
        900,
        1000,
    }

    print(
        "Initialization: one-map "
        "successful checkpoint"
    )

    print(
        "Optimizer: FRESH AdamW"
    )

    print(
        f"LR: {LEARNING_RATE}"
    )

    print(
        f"Weight decay: {WEIGHT_DECAY}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Context: {CONTEXT_SIZE}"
    )

    print(
        f"Gradient clip: {GRADIENT_CLIP}"
    )

    print()
    print(
        "PHASE 1: steps 1-500"
    )

    print(
        "Correct/query mapping always first."
    )

    print()

    print(
        "PHASE 2: steps 501-1000"
    )

    print(
        "Original balanced two-map corpus."
    )

    print()

    for step in range(
        1,
        TOTAL_STEPS + 1,
    ):

        if step == 501:
            print_header(
                "TRAINING WHEELS OFF — ENTERING PHASE 2"
            )

        store = (
            phase1_store
            if step <= CURRICULUM_STEPS
            else phase2_store
        )

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

        schedule_digest.update(
            schedule_payload(
                meta
            )
        )

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            inputs
        )

        loss = (
            document_causal_loss(
                logits,
                targets,
            )
        )

        require_finite_tensor(
            loss,
            "manual curriculum bridge loss",
            "manual_curriculum",
            step,
        )

        loss.backward()

        (
            gradient_norm,
            clipped,
        ) = clip_gradients(
            model,
            GRADIENT_CLIP,
            step,
        )

        optimizer.step()

        require_finite_optimizer(
            optimizer,
            step,
        )

        require_finite_model(
            model,
            "manual_curriculum",
            step,
        )

        supervised_tokens += int(
            meta[
                "supervised_tokens"
            ]
        )

        if step in eval_steps:

            # Small anchor is allowed positive-control material.
            anchor = _small_anchor(
                model,
                tokenizer,
                positive_corpus,
            )

            record = {
                "step":
                    step,

                "phase":
                    (
                        1
                        if step <= 500
                        else 2
                    ),

                "supervised_tokens":
                    supervised_tokens,

                "train_batch_loss":
                    float(
                        loss
                        .detach()
                        .item()
                    ),

                "gradient_norm_before_clip":
                    float(
                        gradient_norm
                    ),

                "gradient_clipped":
                    bool(
                        clipped
                    ),

                "anchor_sample":
                    anchor,
            }

            history.append(
                record
            )

            print(
                f"step {step:4d} | "
                f"phase {record['phase']} | "
                f"loss {loss.item():.5f} | "
                f"anchor "
                f"{anchor['exact_accuracy']:.3%}",
                flush=True,
            )

    # ---------------------------------------------------------
    # SCHEDULE MUST STILL MATCH AFTER ACTUAL TRAINING
    # ---------------------------------------------------------

    actual_schedule_sha = (
        schedule_digest.hexdigest()
    )

    if (
        actual_schedule_sha
        != EXPECTED_SCHEDULE_SHA256
    ):
        raise RuntimeError(
            "Training completed with "
            "unexpected minibatch schedule."
        )

    print()
    print(
        "Training schedule SHA: PASS"
    )

    # ---------------------------------------------------------
    # POSITIVE CONTROLS ONLY
    # ---------------------------------------------------------

    print_header(
        "POSITIVE-CONTROL EXAM"
    )

    anchors = _evaluate(
        model,
        tokenizer,
        positive_corpus[
            "trained_anchors"
        ],
    )

    supported = _evaluate(
        model,
        tokenizer,
        positive_corpus[
            "supported_relation_withheld_geometry"
        ],
    )

    anchor_capture = (
        distractor_capture(
            anchors
        )
    )

    supported_capture = (
        distractor_capture(
            supported
        )
    )

    anchor_accuracy = float(
        anchors[
            "exact_accuracy"
        ]
    )

    supported_accuracy = float(
        supported[
            "exact_accuracy"
        ]
    )

    positive_valid = (
        anchor_accuracy
        >= ANCHOR_GATE
        and
        supported_accuracy
        >= SUPPORTED_GATE
    )

    runtime = (
        time.perf_counter()
        - started
    )

    print(
        "Trained anchors:"
    )

    print(
        f"  {anchors['exact']}/"
        f"{anchors['examples']} "
        f"= {anchor_accuracy:.3%}"
    )

    print()

    print(
        "Supported relation / "
        "withheld geometry:"
    )

    print(
        f"  {supported['exact']}/"
        f"{supported['examples']} "
        f"= {supported_accuracy:.3%}"
    )

    print()

    print(
        "Anchor distractor-target "
        "share among errors:"
    )

    print(
        f"  "
        f"{anchor_capture['distractor_target_errors']}/"
        f"{anchor_capture['incorrect']} "
        f"= "
        f"{anchor_capture['distractor_target_share_of_errors']:.3%}"
        if anchor_capture["incorrect"]
        else
        "  no anchor errors"
    )

    print()

    print(
        "Supported distractor-target "
        "share among errors:"
    )

    print(
        f"  "
        f"{supported_capture['distractor_target_errors']}/"
        f"{supported_capture['incorrect']} "
        f"= "
        f"{supported_capture['distractor_target_share_of_errors']:.3%}"
        if supported_capture["incorrect"]
        else
        "  no supported errors"
    )

    print()

    print(
        "Positive-control gate:"
    )

    print(
        "  PASS"
        if positive_valid
        else "  FAIL"
    )

    print()

    print(
        "SEALED EVALUATION OPENED: NO"
    )

    # ---------------------------------------------------------
    # SAVE CANDIDATE CHECKPOINT
    #
    # This is CADAVER output only.
    # ---------------------------------------------------------

    checkpoint_payload = {
        "version":
            "manual_curriculum_bridge_candidate",

        "kind":
            "DaveLM_v0.9_manual_candidate_not_promoted",

        "seed":
            SEED,

        "step":
            TOTAL_STEPS,

        "source_checkpoint":
            str(
                START_CHECKPOINT
            ),

        "source_checkpoint_sha256":
            EXPECTED_START_CHECKPOINT_SHA256,

        "initialization_digest":
            initialization_digest,

        "final_model_digest":
            _model_digest(
                model
            ),

        "parameter_count":
            EXPECTED_PARAMETER_COUNT,

        "curriculum": {
            "phase_1_steps":
                [1, 500],

            "phase_1":
                "queried mapping canonicalized to displayed slot 0",

            "phase_2_steps":
                [501, 1000],

            "phase_2":
                "exact original balanced two-mapping corpus",

            "accepted_phase_1_geometry_consequence":
                "target_to_query_distance +12 on the 768 originally-slot-1 documents",
        },

        "training_settings": {
            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "gradient_clip":
                GRADIENT_CLIP,

            "batch_size":
                BATCH_SIZE,

            "context_size":
                CONTEXT_SIZE,

            "maximum_steps":
                TOTAL_STEPS,

            "scheduler":
                None,

            "ordinary_causal_cross_entropy":
                True,

            "answer_weighting":
                False,

            "auxiliary_loss":
                False,

            "fresh_optimizer":
                True,
        },

        "minibatch_schedule_sha256":
            actual_schedule_sha,

        "supervised_tokens":
            supervised_tokens,

        "history":
            history,

        "positive_control": {
            "trained_anchors":
                anchors,

            "supported_relation_withheld_geometry":
                supported,

            "anchor_distractor_capture":
                anchor_capture,

            "supported_distractor_capture":
                supported_capture,

            "passed":
                positive_valid,

            "anchor_gate":
                ANCHOR_GATE,

            "supported_gate":
                SUPPORTED_GATE,
        },

        "sealed_evaluation_opened":
            False,

        "model_state": {
            name:
                tensor
                .detach()
                .cpu()
                .clone()

            for name, tensor
            in model.state_dict().items()
        },

        "optimizer_state":
            optimizer.state_dict(),

        "minibatch_generator_state":
            generator.get_state(),

        "torch_rng_state":
            torch.get_rng_state(),

        "cuda_rng_state_all":
            torch.cuda.get_rng_state_all(),

        "runtime_seconds":
            runtime,

        "peak_gpu_allocated_bytes":
            int(
                torch.cuda
                .max_memory_allocated(
                    device
                )
            ),

        "peak_gpu_reserved_bytes":
            int(
                torch.cuda
                .max_memory_reserved(
                    device
                )
            ),
    }

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint_payload,
        CHECKPOINT_PATH,
    )

    checkpoint_sha = file_sha256(
        CHECKPOINT_PATH
    )

    result_payload = {
        "classification":
            (
                "POSITIVE_CONTROL_PASS"
                if positive_valid
                else
                "INCONCLUSIVE_POSITIVE_CONTROL_FAILURE"
            ),

        "trained_anchors": {
            "exact":
                anchors["exact"],

            "examples":
                anchors["examples"],

            "exact_accuracy":
                anchor_accuracy,
        },

        "supported_relation_withheld_geometry": {
            "exact":
                supported["exact"],

            "examples":
                supported["examples"],

            "exact_accuracy":
                supported_accuracy,
        },

        "anchor_distractor_capture":
            anchor_capture,

        "supported_distractor_capture":
            supported_capture,

        "positive_control_valid":
            positive_valid,

        "sealed_evaluation_opened":
            False,

        "checkpoint":
            str(
                CHECKPOINT_PATH
            ),

        "checkpoint_sha256":
            checkpoint_sha,

        "schedule_sha256":
            actual_schedule_sha,

        "runtime_seconds":
            runtime,
    }

    save_json(
        RESULT_PATH,
        result_payload,
    )

    print_header(
        "CLASS DISMISSED"
    )

    print(
        "Candidate checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print()

    print(
        "Checkpoint SHA256:"
    )

    print(
        checkpoint_sha
    )

    print()

    print(
        "Result:"
    )

    if positive_valid:
        print(
            "POSITIVE-CONTROL PASS"
        )

        print()
        print(
            "IMPORTANT: sealed exam "
            "was NOT opened."
        )

        print(
            "STOP HERE and inspect "
            "the result before doing so."
        )

    else:
        print(
            "INCONCLUSIVE / "
            "POSITIVE-CONTROL FAILURE"
        )

        print()
        print(
            "Sealed exam remains sealed."
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    safety_check()

    (
        tokenizer,
        groups,
        identities,
        filler,
        tables,
        geometry_schedule,
        keys,
        original,
    ) = reconstruct()

    phase1 = build_phase1(
        tokenizer,
        identities,
        filler,
        geometry_schedule,
        keys,
        original,
    )

    (
        phase1_store,
        phase2_store,
    ) = build_stores(
        tokenizer,
        original,
        phase1,
    )

    start_payload = (
        inspect_start_checkpoint()
    )

    schedule_info = (
        verify_schedule(
            phase1_store,
            phase2_store,
        )
    )

    print_header(
        "FULL PREFLIGHT SUMMARY"
    )

    print(
        "Starting checkpoint SHA: PASS"
    )

    print(
        "Architecture:             PASS"
    )

    print(
        "Original corpus:          PASS"
    )

    print(
        "Phase-1 transformation:   PASS"
    )

    print(
        "Phase-2 unchanged:        PASS"
    )

    print(
        "1000-step schedule:       PASS"
    )

    print(
        "Schedule SHA:"
    )

    print(
        schedule_info[
            "schedule_sha256"
        ]
    )

    print()

    if MODE == "PREFLIGHT":

        payload = {
            "passed":
                True,

            "mode":
                "PREFLIGHT",

            "optimizer_steps":
                0,

            "starting_checkpoint":
                str(
                    START_CHECKPOINT
                ),

            "starting_checkpoint_sha256":
                EXPECTED_START_CHECKPOINT_SHA256,

            "schedule_sha256":
                schedule_info[
                    "schedule_sha256"
                ],

            "phase1_supervised_tokens":
                schedule_info[
                    "phase1_supervised_tokens"
                ],

            "phase2_supervised_tokens":
                schedule_info[
                    "phase2_supervised_tokens"
                ],

            "sealed_evaluation_opened":
                False,
        }

        OUTPUT_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_json(
            PREFLIGHT_PATH,
            payload,
        )

        print(
            "PREFLIGHT PASSED."
        )

        print()

        print(
            "ZERO OPTIMIZER STEPS "
            "WERE PERFORMED."
        )

        print()

        print(
            "DO NOT TRAIN YET."
        )

        print(
            "Paste this output back "
            "into ChatGPT first."
        )

        return

    # ---------------------------------------------------------
    # TRAIN MODE ONLY
    # ---------------------------------------------------------

    positive_corpus = (
        load_positive_evaluation_corpus()
    )

    train(
        tokenizer,
        phase1_store,
        phase2_store,
        start_payload,
        positive_corpus,
    )


if __name__ == "__main__":
    main()