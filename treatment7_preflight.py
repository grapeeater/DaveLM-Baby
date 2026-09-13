from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# =============================================================================
# DaveLM v0.9 — Treatment #7
# Direct Query-to-Logit Pathway
#
# PREFLIGHT ONLY
#
# SCIENTIFIC QUESTION:
#
# Does the paired counterfactual query-binding task fail because Baby lacks a
# sufficiently direct trainable pathway from query identity to answer logits?
#
# ONE SUBSTANTIVE T7 CHANGE:
#
# Add one trainable affine projection:
#
#     Linear(hidden_size=320, vocab_size=1024, bias=True)
#
# The projection receives the EXISTING native token embedding of the query token.
#
# At the answer position only:
#
#     adjusted_answer_logits
#         = base_answer_logits
#         + query_projection(query_token_embedding)
#
# No candidate IDs, target IDs, distractor IDs, slot labels, or mapping IDs are
# supplied to this projection.
#
# THIS SCRIPT DOES NOT TRAIN.
#
# It performs:
#   - NO optimizer construction
#   - NO backward()
#   - NO gradient updates
#   - NO treatment training
#   - NO positive controls
#   - NO sealed evaluation
#   - NO protected-source modification
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

SOURCE_ROOT = Path(
    r"C:\DaveLM-v0.9"
)

PAIR_POOL_PATH = Path(
    r"C:\DaveLM-CADAVER"
    r"\treatment5_counterfactual_pairs_seed8382"
    r"\treatment5_full_document_pair_pool.json"
)

SCHEDULE_PATH = Path(
    r"C:\DaveLM-CADAVER"
    r"\treatment5_counterfactual_pairs_seed8382"
    r"\treatment5_full_document_frozen_schedule.json"
)

START_CHECKPOINT = Path(
    r"C:\DaveLM-v0.9"
    r"\experiments"
    r"\minimal_contextual_binding"
    r"\checkpoints"
    r"\treatment_one_mapping"
    r"\seed_8380"
    r"\latest.pt"
)

OUTPUT_ROOT = Path(
    r"C:\DaveLM-CADAVER"
    r"\treatment7_direct_query_logit_pathway_seed8380"
)

PREFLIGHT_RESULT_PATH = (
    OUTPUT_ROOT
    / "treatment7_preflight_result.json"
)


# =============================================================================
# FROZEN HASHES
# =============================================================================

EXPECTED_PAIR_POOL_SHA256 = (
    "0b31e39361f7a45dc4182c49ab4b4967"
    "bffe539cd2ff12e7b196eabfb3dac307"
)

EXPECTED_SCHEDULE_SHA256 = (
    "a4e64fd9d1b934aa440a5d036a3bec0a"
    "c7c9ac97d58c72fb45df6d8576bb7c12"
)

EXPECTED_START_CHECKPOINT_SHA256 = (
    "345984c52a06db5f988aaf4cd47963ee"
    "a0e9d77489cee94dbb10816af2f5443e"
)


# =============================================================================
# FROZEN MODEL / TRAINING CONTRACT
# =============================================================================

EXPECTED_BASE_PARAMETER_COUNT = 10_594_944

EXPECTED_HIDDEN_SIZE = 320

EXPECTED_VOCAB_SIZE = 1024

EXPECTED_QUERY_PROJECTION_PARAMETERS = (
    EXPECTED_HIDDEN_SIZE * EXPECTED_VOCAB_SIZE
    + EXPECTED_VOCAB_SIZE
)

EXPECTED_TOTAL_T7_PARAMETER_COUNT = (
    EXPECTED_BASE_PARAMETER_COUNT
    + EXPECTED_QUERY_PROJECTION_PARAMETERS
)

EXPECTED_PAIR_COUNT = 1536

EXPECTED_STEPS = 1000

EXPECTED_BATCH_SIZE = 32

EXPECTED_PAIRS_PER_BATCH = 16

EXPECTED_EVENTS = 32_000

EXPECTED_SLOT0 = 16_000

EXPECTED_SLOT1 = 16_000

EXPECTED_TOTAL_SUPERVISED = 6_144_000

EXPECTED_ANSWER_REPLACEMENTS = 32_000

EXPECTED_NONANSWER_SUPERVISED = 6_112_000

EXPECTED_MODEL_VISIBLE_LENGTH = 193

EXPECTED_CAUSAL_LENGTH = 192

EXPECTED_NONANSWER_PER_EXAMPLE = 191

EXPECTED_PAIR_EXPOSURE_MIN = 10

EXPECTED_PAIR_EXPOSURE_MAX = 11

MARGIN = 0.5

ANSWER_WEIGHT = 191.0


# =============================================================================
# HELPERS
# =============================================================================

def header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)
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


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# IMPORT EXACT NATIVE BUILDER
# =============================================================================

def import_native_module():
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(
            0,
            str(SOURCE_ROOT),
        )

    from experiments.two_mapping_contextual_binding import run as original_run

    return original_run


# =============================================================================
# CHECKPOINT STATE EXTRACTION
# =============================================================================

def extract_model_state(
    checkpoint: Any,
) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Starting checkpoint is not a dictionary."
        )

    for key in (
        "model_state",
        "model_state_dict",
        "model",
    ):
        candidate = checkpoint.get(key)

        if isinstance(candidate, dict):
            return candidate

    if checkpoint and all(
        torch.is_tensor(value)
        for value in checkpoint.values()
    ):
        return checkpoint

    raise RuntimeError(
        "Could not identify model state "
        "inside starting checkpoint."
    )


# =============================================================================
# T7 WRAPPER
# =============================================================================

class Treatment7QueryLogitModel(nn.Module):
    """
    Wraps the exact native DaveLM base model.

    The ONLY new learned component is query_projection:

        Linear(320 -> 1024, bias=True)

    IMPORTANT:

    This wrapper does NOT change the base model's ordinary forward behavior.

    The query-conditioned bias is computed explicitly and is intended to be
    added ONLY to answer-position logits by the future T7 trainer.
    """

    def __init__(
        self,
        base_model: nn.Module,
    ):
        super().__init__()

        self.base_model = base_model

        token_embedding = getattr(
            base_model,
            "token_embedding",
            None,
        )

        if not isinstance(
            token_embedding,
            nn.Embedding,
        ):
            raise RuntimeError(
                "Base model does not expose expected "
                "nn.Embedding as token_embedding."
            )

        hidden_size = int(
            token_embedding.embedding_dim
        )

        vocab_size = int(
            token_embedding.num_embeddings
        )

        if hidden_size != EXPECTED_HIDDEN_SIZE:
            raise RuntimeError(
                f"Hidden-size mismatch. "
                f"Expected {EXPECTED_HIDDEN_SIZE}; "
                f"observed {hidden_size}."
            )

        if vocab_size != EXPECTED_VOCAB_SIZE:
            raise RuntimeError(
                f"Vocab-size mismatch. "
                f"Expected {EXPECTED_VOCAB_SIZE}; "
                f"observed {vocab_size}."
            )

        self.query_projection = nn.Linear(
            hidden_size,
            vocab_size,
            bias=True,
        )

    def get_query_bias(
        self,
        query_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        query_embeddings = (
            self.base_model.token_embedding(
                query_token_ids
            )
        )

        return self.query_projection(
            query_embeddings
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ):
        return self.base_model(
            input_ids
        )


# =============================================================================
# VERIFY FROZEN FILES
# =============================================================================

def verify_artifacts() -> dict[str, str]:
    header(
        "VERIFYING FROZEN TREATMENT #7 INPUT ARTIFACTS"
    )

    for path in (
        SOURCE_ROOT,
        PAIR_POOL_PATH,
        SCHEDULE_PATH,
        START_CHECKPOINT,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required artifact missing:\n{path}"
            )

    pair_sha = sha256_file(
        PAIR_POOL_PATH
    )

    schedule_sha = sha256_file(
        SCHEDULE_PATH
    )

    checkpoint_sha = sha256_file(
        START_CHECKPOINT
    )

    print("Pair-pool SHA256:")
    print(f"  observed: {pair_sha}")
    print(f"  expected: {EXPECTED_PAIR_POOL_SHA256}")

    if pair_sha != EXPECTED_PAIR_POOL_SHA256:
        raise RuntimeError(
            "PAIR-POOL SHA MISMATCH."
        )

    print("  PASS")
    print()

    print("Frozen schedule SHA256:")
    print(f"  observed: {schedule_sha}")
    print(f"  expected: {EXPECTED_SCHEDULE_SHA256}")

    if schedule_sha != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError(
            "SCHEDULE SHA MISMATCH."
        )

    print("  PASS")
    print()

    print("Starting checkpoint SHA256:")
    print(f"  observed: {checkpoint_sha}")
    print(f"  expected: {EXPECTED_START_CHECKPOINT_SHA256}")

    if checkpoint_sha != EXPECTED_START_CHECKPOINT_SHA256:
        raise RuntimeError(
            "START CHECKPOINT SHA MISMATCH."
        )

    print("  PASS")

    return {
        "pair_pool_sha256": pair_sha,
        "schedule_sha256": schedule_sha,
        "start_checkpoint_sha256": checkpoint_sha,
    }


# =============================================================================
# LOAD + VERIFY PAIR/SCHEDULE CONTRACT
# =============================================================================

def audit_frozen_training_contract():
    header(
        "AUDITING FROZEN T5/T6 TRAINING CONTRACT FOR T7"
    )

    pair_pool = json.loads(
        PAIR_POOL_PATH.read_text(
            encoding="utf-8"
        )
    )

    schedule = json.loads(
        SCHEDULE_PATH.read_text(
            encoding="utf-8"
        )
    )

    pairs = pair_pool.get(
        "pairs"
    )

    if not isinstance(
        pairs,
        list,
    ):
        raise RuntimeError(
            "Pair pool missing valid 'pairs' list."
        )

    if len(pairs) != EXPECTED_PAIR_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIR_COUNT} pairs; "
            f"found {len(pairs)}."
        )

    pair_lookup = {}

    query_positions = set()
    answer_indices = set()
    answer_positions = set()

    for pair in pairs:
        pair_id = str(
            pair["pair_id"]
        )

        if pair_id in pair_lookup:
            raise RuntimeError(
                f"Duplicate pair ID: {pair_id}"
            )

        twin_a = pair[
            "twin_a"
        ]

        twin_b = pair[
            "twin_b"
        ]

        if int(
            twin_a["query_slot"]
        ) != 0:
            raise RuntimeError(
                f"{pair_id}: Twin A query_slot != 0."
            )

        if int(
            twin_b["query_slot"]
        ) != 1:
            raise RuntimeError(
                f"{pair_id}: Twin B query_slot != 1."
            )

        a_target = int(
            twin_a["target_token_id"]
        )

        a_distractor = int(
            twin_a["distractor_token_id"]
        )

        b_target = int(
            twin_b["target_token_id"]
        )

        b_distractor = int(
            twin_b["distractor_token_id"]
        )

        if a_target == a_distractor:
            raise RuntimeError(
                f"{pair_id}: Twin A target == distractor."
            )

        if b_target == b_distractor:
            raise RuntimeError(
                f"{pair_id}: Twin B target == distractor."
            )

        if a_target != b_distractor:
            raise RuntimeError(
                f"{pair_id}: A target != B distractor."
            )

        if b_target != a_distractor:
            raise RuntimeError(
                f"{pair_id}: B target != A distractor."
            )

        query_position = int(
            pair["query_difference_position"]
        )

        answer_index = int(
            pair["answer_token_index"]
        )

        answer_position = int(
            pair["answer_causal_position"]
        )

        if answer_position != answer_index - 1:
            raise RuntimeError(
                f"{pair_id}: answer causal-position mismatch."
            )

        doc_a = [
            int(x)
            for x in twin_a[
                "full_document_token_ids"
            ]
        ]

        doc_b = [
            int(x)
            for x in twin_b[
                "full_document_token_ids"
            ]
        ]

        if len(doc_a) != EXPECTED_MODEL_VISIBLE_LENGTH:
            raise RuntimeError(
                f"{pair_id}: Twin A document length mismatch."
            )

        if len(doc_b) != EXPECTED_MODEL_VISIBLE_LENGTH:
            raise RuntimeError(
                f"{pair_id}: Twin B document length mismatch."
            )

        differences = [
            index
            for index, (a, b)
            in enumerate(
                zip(
                    doc_a,
                    doc_b,
                )
            )
            if a != b
        ]

        expected_differences = sorted(
            [
                query_position,
                answer_index,
            ]
        )

        if differences != expected_differences:
            raise RuntimeError(
                f"{pair_id}: twin differences "
                f"{differences}; expected "
                f"{expected_differences}."
            )

        if doc_a[
            answer_index
        ] != a_target:
            raise RuntimeError(
                f"{pair_id}: Twin A answer != target."
            )

        if doc_b[
            answer_index
        ] != b_target:
            raise RuntimeError(
                f"{pair_id}: Twin B answer != target."
            )

        query_positions.add(
            query_position
        )

        answer_indices.add(
            answer_index
        )

        answer_positions.add(
            answer_position
        )

        pair_lookup[
            pair_id
        ] = pair

    steps_data = schedule.get(
        "steps_data"
    )

    if not isinstance(
        steps_data,
        list,
    ):
        raise RuntimeError(
            "Schedule steps_data is not a list."
        )

    if len(steps_data) != EXPECTED_STEPS:
        raise RuntimeError(
            f"Expected {EXPECTED_STEPS} steps; "
            f"found {len(steps_data)}."
        )

    event_count = 0
    slot0_count = 0
    slot1_count = 0

    total_supervised = 0
    total_nonanswer = 0

    pair_exposures: dict[str, int] = {}

    first_example = None

    for expected_step, step_data in enumerate(
        steps_data,
        start=1,
    ):
        if int(
            step_data["step"]
        ) != expected_step:
            raise RuntimeError(
                f"Schedule step mismatch at {expected_step}."
            )

        examples = step_data.get(
            "examples"
        )

        if not isinstance(
            examples,
            list,
        ):
            raise RuntimeError(
                f"Step {expected_step}: examples not a list."
            )

        if len(examples) != EXPECTED_BATCH_SIZE:
            raise RuntimeError(
                f"Step {expected_step}: batch size mismatch."
            )

        for offset in range(
            0,
            EXPECTED_BATCH_SIZE,
            2,
        ):
            twin_a = examples[
                offset
            ]

            twin_b = examples[
                offset + 1
            ]

            if str(
                twin_a["pair_id"]
            ) != str(
                twin_b["pair_id"]
            ):
                raise RuntimeError(
                    f"Step {expected_step}: "
                    "adjacent examples are not same pair."
                )

            if str(
                twin_a["twin"]
            ) != "A":
                raise RuntimeError(
                    f"Step {expected_step}: first twin != A."
                )

            if str(
                twin_b["twin"]
            ) != "B":
                raise RuntimeError(
                    f"Step {expected_step}: second twin != B."
                )

            pair_id = str(
                twin_a["pair_id"]
            )

            pair_exposures[
                pair_id
            ] = (
                pair_exposures.get(
                    pair_id,
                    0,
                )
                + 1
            )

        for example in examples:
            if first_example is None:
                first_example = example

            pair_id = str(
                example["pair_id"]
            )

            if pair_id not in pair_lookup:
                raise RuntimeError(
                    f"Schedule references unknown pair {pair_id}."
                )

            twin_label = str(
                example["twin"]
            )

            if twin_label == "A":
                frozen_twin = pair_lookup[
                    pair_id
                ][
                    "twin_a"
                ]

            elif twin_label == "B":
                frozen_twin = pair_lookup[
                    pair_id
                ][
                    "twin_b"
                ]

            else:
                raise RuntimeError(
                    f"Invalid twin label: {twin_label}"
                )

            schedule_doc = [
                int(x)
                for x in example[
                    "full_document_token_ids"
                ]
            ]

            frozen_doc = [
                int(x)
                for x in frozen_twin[
                    "full_document_token_ids"
                ]
            ]

            if schedule_doc != frozen_doc:
                raise RuntimeError(
                    f"{pair_id} {twin_label}: "
                    "schedule/pair-pool document mismatch."
                )

            answer_index = int(
                example[
                    "answer_token_index"
                ]
            )

            answer_position = int(
                example[
                    "answer_causal_position"
                ]
            )

            query_position = int(
                example[
                    "query_difference_position"
                ]
            )

            if answer_position != answer_index - 1:
                raise RuntimeError(
                    f"{pair_id}: answer position mismatch."
                )

            target = int(
                example[
                    "target_token_id"
                ]
            )

            distractor = int(
                example[
                    "distractor_token_id"
                ]
            )

            if target == distractor:
                raise RuntimeError(
                    f"{pair_id}: target == distractor."
                )

            if schedule_doc[
                answer_index
            ] != target:
                raise RuntimeError(
                    f"{pair_id}: document answer != target."
                )

            if int(
                frozen_twin[
                    "target_token_id"
                ]
            ) != target:
                raise RuntimeError(
                    f"{pair_id}: frozen target mismatch."
                )

            if int(
                frozen_twin[
                    "distractor_token_id"
                ]
            ) != distractor:
                raise RuntimeError(
                    f"{pair_id}: frozen distractor mismatch."
                )

            if int(
                example[
                    "model_visible_document_length"
                ]
            ) != EXPECTED_MODEL_VISIBLE_LENGTH:
                raise RuntimeError(
                    f"{pair_id}: model-visible length mismatch."
                )

            if int(
                example[
                    "supervised_token_count"
                ]
            ) != EXPECTED_CAUSAL_LENGTH:
                raise RuntimeError(
                    f"{pair_id}: supervised count mismatch."
                )

            if int(
                example[
                    "nonanswer_supervised_token_count"
                ]
            ) != EXPECTED_NONANSWER_PER_EXAMPLE:
                raise RuntimeError(
                    f"{pair_id}: non-answer count mismatch."
                )

            query_slot = int(
                example[
                    "query_slot"
                ]
            )

            if query_slot == 0:
                slot0_count += 1

            elif query_slot == 1:
                slot1_count += 1

            else:
                raise RuntimeError(
                    f"{pair_id}: invalid query slot."
                )

            if query_position >= answer_index:
                raise RuntimeError(
                    f"{pair_id}: query is not before answer."
                )

            query_positions.add(
                query_position
            )

            answer_indices.add(
                answer_index
            )

            answer_positions.add(
                answer_position
            )

            event_count += 1

            total_supervised += int(
                example[
                    "supervised_token_count"
                ]
            )

            total_nonanswer += int(
                example[
                    "nonanswer_supervised_token_count"
                ]
            )

    if event_count != EXPECTED_EVENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_EVENTS} events; "
            f"found {event_count}."
        )

    if slot0_count != EXPECTED_SLOT0:
        raise RuntimeError(
            f"Slot-0 count mismatch: {slot0_count}"
        )

    if slot1_count != EXPECTED_SLOT1:
        raise RuntimeError(
            f"Slot-1 count mismatch: {slot1_count}"
        )

    if total_supervised != EXPECTED_TOTAL_SUPERVISED:
        raise RuntimeError(
            "Total supervised-position count mismatch."
        )

    if total_nonanswer != EXPECTED_NONANSWER_SUPERVISED:
        raise RuntimeError(
            "Total non-answer-position count mismatch."
        )

    if len(pair_exposures) != EXPECTED_PAIR_COUNT:
        raise RuntimeError(
            "Not all frozen pairs appear in schedule."
        )

    exposure_values = list(
        pair_exposures.values()
    )

    if min(exposure_values) != EXPECTED_PAIR_EXPOSURE_MIN:
        raise RuntimeError(
            "Minimum pair exposure mismatch."
        )

    if max(exposure_values) != EXPECTED_PAIR_EXPOSURE_MAX:
        raise RuntimeError(
            "Maximum pair exposure mismatch."
        )

    print(
        f"Pairs:                         {len(pair_lookup)}"
    )

    print(
        f"Steps:                         {len(steps_data)}"
    )

    print(
        f"Batch size:                    {EXPECTED_BATCH_SIZE}"
    )

    print(
        f"Complete twin pairs / batch:   {EXPECTED_PAIRS_PER_BATCH}"
    )

    print(
        f"Historical events:             {event_count}"
    )

    print(
        f"Slot 0 events:                 {slot0_count}"
    )

    print(
        f"Slot 1 events:                 {slot1_count}"
    )

    print(
        f"Total supervised positions:    {total_supervised}"
    )

    print(
        f"Answer replacements:           {EXPECTED_ANSWER_REPLACEMENTS}"
    )

    print(
        f"Ordinary non-answer CE:        {total_nonanswer}"
    )

    print(
        f"Pair exposure range:           "
        f"{min(exposure_values)}-{max(exposure_values)}"
    )

    print(
        f"Query-token positions:         {sorted(query_positions)}"
    )

    print(
        f"Answer-token indices:          {sorted(answer_indices)}"
    )

    print(
        f"Answer causal positions:       {sorted(answer_positions)}"
    )

    print()

    print(
        "Pair-pool structural audit: PASS"
    )

    print(
        "Schedule/pair exact document cross-check: PASS"
    )

    print(
        "Frozen training contract: PASS"
    )

    return first_example


# =============================================================================
# BUILD EXACT BASE MODEL + VERIFY NEW T7 PATH
# =============================================================================

def build_and_verify_t7(
    device: torch.device,
    first_example: dict[str, Any],
):
    header(
        "BUILDING T7 PREFLIGHT MODEL"
    )

    original_run = import_native_module()

    build_model_fn = getattr(
        original_run,
        "build_model",
        None,
    )

    if not callable(
        build_model_fn
    ):
        raise RuntimeError(
            "Native run module does not expose callable build_model()."
        )

    base_model = build_model_fn(
        "untied"
    )

    checkpoint = torch.load(
        START_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state = extract_model_state(
        checkpoint
    )

    missing, unexpected = (
        base_model.load_state_dict(
            state,
            strict=False,
        )
    )

    if missing or unexpected:
        raise RuntimeError(
            "\nBASE CHECKPOINT STATE-DICT MISMATCH.\n"
            f"Missing: {missing}\n"
            f"Unexpected: {unexpected}\n"
        )

    base_parameter_count = sum(
        parameter.numel()
        for parameter in base_model.parameters()
    )

    if base_parameter_count != EXPECTED_BASE_PARAMETER_COUNT:
        raise RuntimeError(
            f"Base parameter-count mismatch. "
            f"Expected {EXPECTED_BASE_PARAMETER_COUNT:,}; "
            f"observed {base_parameter_count:,}."
        )

    model = Treatment7QueryLogitModel(
        base_model
    ).to(
        device
    )

    projection_parameter_count = sum(
        parameter.numel()
        for parameter
        in model.query_projection.parameters()
    )

    if projection_parameter_count != (
        EXPECTED_QUERY_PROJECTION_PARAMETERS
    ):
        raise RuntimeError(
            f"Query-projection parameter-count mismatch. "
            f"Expected {EXPECTED_QUERY_PROJECTION_PARAMETERS:,}; "
            f"observed {projection_parameter_count:,}."
        )

    total_parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    if total_parameter_count != (
        EXPECTED_TOTAL_T7_PARAMETER_COUNT
    ):
        raise RuntimeError(
            f"T7 total parameter-count mismatch. "
            f"Expected {EXPECTED_TOTAL_T7_PARAMETER_COUNT:,}; "
            f"observed {total_parameter_count:,}."
        )

    # -------------------------------------------------------------------------
    # Verify exactly one new named parameter family exists beyond base model.
    # -------------------------------------------------------------------------

    projection_names = [
        name
        for name, _
        in model.named_parameters()
        if name.startswith(
            "query_projection."
        )
    ]

    expected_projection_names = {
        "query_projection.weight",
        "query_projection.bias",
    }

    if set(projection_names) != expected_projection_names:
        raise RuntimeError(
            "Unexpected Treatment #7 projection parameter names:\n"
            f"{projection_names}"
        )

    # -------------------------------------------------------------------------
    # Verify projection input/output geometry.
    # -------------------------------------------------------------------------

    if tuple(
        model.query_projection.weight.shape
    ) != (
        EXPECTED_VOCAB_SIZE,
        EXPECTED_HIDDEN_SIZE,
    ):
        raise RuntimeError(
            "query_projection.weight shape mismatch."
        )

    if tuple(
        model.query_projection.bias.shape
    ) != (
        EXPECTED_VOCAB_SIZE,
    ):
        raise RuntimeError(
            "query_projection.bias shape mismatch."
        )

    # -------------------------------------------------------------------------
    # One read-only forward sanity check.
    #
    # This does NOT evaluate task performance.
    # It only verifies tensor routing.
    # -------------------------------------------------------------------------

    document = torch.tensor(
        [
            int(x)
            for x in first_example[
                "full_document_token_ids"
            ]
        ],
        dtype=torch.long,
        device=device,
    )

    input_ids = (
        document[
            :-1
        ]
        .unsqueeze(0)
        .contiguous()
    )

    query_position = int(
        first_example[
            "query_difference_position"
        ]
    )

    answer_position = int(
        first_example[
            "answer_causal_position"
        ]
    )

    query_token_id = input_ids[
        0,
        query_position,
    ].reshape(1)

    model.eval()

    with torch.inference_mode():
        base_output = model(
            input_ids
        )

        if torch.is_tensor(
            base_output
        ):
            base_logits = base_output

        elif isinstance(
            base_output,
            dict,
        ) and torch.is_tensor(
            base_output.get(
                "logits"
            )
        ):
            base_logits = base_output[
                "logits"
            ]

        elif isinstance(
            base_output,
            (
                tuple,
                list,
            ),
        ):
            base_logits = None

            for candidate in base_output:
                if (
                    torch.is_tensor(
                        candidate
                    )
                    and candidate.ndim == 3
                ):
                    base_logits = candidate
                    break

            if base_logits is None:
                raise RuntimeError(
                    "Could not extract logits from tuple/list output."
                )

        else:
            candidate = getattr(
                base_output,
                "logits",
                None,
            )

            if not torch.is_tensor(
                candidate
            ):
                raise RuntimeError(
                    "Could not extract logits from base model output."
                )

            base_logits = candidate

        query_bias = model.get_query_bias(
            query_token_id
        )

        if tuple(
            query_bias.shape
        ) != (
            1,
            EXPECTED_VOCAB_SIZE,
        ):
            raise RuntimeError(
                f"Query-bias shape mismatch: "
                f"{tuple(query_bias.shape)}"
            )

        base_answer_logits = base_logits[
            0,
            answer_position,
            :
        ]

        adjusted_answer_logits = (
            base_answer_logits
            + query_bias[
                0
            ]
        )

        if tuple(
            adjusted_answer_logits.shape
        ) != (
            EXPECTED_VOCAB_SIZE,
        ):
            raise RuntimeError(
                "Adjusted answer-logit shape mismatch."
            )

        # Explicitly verify we have NOT altered ordinary base logits.
        base_output_again = model(
            input_ids
        )

        if torch.is_tensor(
            base_output_again
        ):
            base_logits_again = base_output_again

        elif isinstance(
            base_output_again,
            dict,
        ):
            base_logits_again = base_output_again[
                "logits"
            ]

        elif isinstance(
            base_output_again,
            (
                tuple,
                list,
            ),
        ):
            base_logits_again = None

            for candidate in base_output_again:
                if (
                    torch.is_tensor(
                        candidate
                    )
                    and candidate.ndim == 3
                ):
                    base_logits_again = candidate
                    break

            if base_logits_again is None:
                raise RuntimeError(
                    "Could not re-extract logits."
                )

        else:
            base_logits_again = getattr(
                base_output_again,
                "logits",
            )

        if not torch.equal(
            base_logits,
            base_logits_again,
        ):
            raise RuntimeError(
                "Base forward is not deterministic in eval mode."
            )

    print(
        f"Base parameters:               {base_parameter_count:,}"
    )

    print(
        f"New query-projection params:   {projection_parameter_count:,}"
    )

    print(
        f"Expected new params:           "
        f"{EXPECTED_QUERY_PROJECTION_PARAMETERS:,}"
    )

    print(
        f"Total T7 parameters:           {total_parameter_count:,}"
    )

    print(
        f"Expected total params:         "
        f"{EXPECTED_TOTAL_T7_PARAMETER_COUNT:,}"
    )

    print()

    print(
        "Base builder:                  "
        'original_run.build_model("untied")'
    )

    print(
        "Starting checkpoint load:      PASS"
    )

    print(
        "Query source:                  native token embedding"
    )

    print(
        "Query projection:              Linear(320 -> 1024, bias=True)"
    )

    print(
        "Projection input target IDs:   NO"
    )

    print(
        "Projection distractor IDs:     NO"
    )

    print(
        "Projection candidate IDs:      NO"
    )

    print(
        "Projection query-slot label:   NO"
    )

    print(
        "Projection mapping metadata:   NO"
    )

    print(
        "Answer-only adjusted-logit tensor construction: PASS"
    )

    print(
        "Ordinary base forward unchanged: PASS"
    )

    return {
        "base_parameter_count": base_parameter_count,
        "query_projection_parameter_count": projection_parameter_count,
        "total_parameter_count": total_parameter_count,
        "projection_weight_shape": list(
            model.query_projection.weight.shape
        ),
        "projection_bias_shape": list(
            model.query_projection.bias.shape
        ),
    }


# =============================================================================
# MAIN PREFLIGHT
# =============================================================================

def main():
    header(
        "DAVELM v0.9 — TREATMENT #7 PREFLIGHT"
    )

    print(
        "Treatment:"
    )

    print(
        "  Direct Query-to-Logit Pathway"
    )

    print()

    print(
        "Scientific hypothesis:"
    )

    print(
        "  A direct trainable route from query identity to answer logits "
        "may be sufficient to break the T5/T6 reciprocal-selection failure."
    )

    print()

    print(
        "ONE substantive change:"
    )

    print(
        "  Add Linear(320 -> 1024, bias=True) from the native query-token "
        "embedding to answer-position logits."
    )

    print()

    print(
        "Frozen from T6:"
    )

    print(
        "  Base architecture/checkpoint: YES"
    )

    print(
        "  Pair pool: YES"
    )

    print(
        "  Frozen schedule: YES"
    )

    print(
        "  1000 steps / batch 32 / 16 complete pairs: YES"
    )

    print(
        "  Raw membership + selector objective: YES"
    )

    print(
        f"  Answer weight lambda = {ANSWER_WEIGHT:.1f}: YES"
    )

    print(
        f"  Selector margin = {MARGIN:.1f}: YES"
    )

    print(
        "  Optimizer/LR/WD/grad clip for future trainer: unchanged"
    )

    print()

    print(
        "PREFLIGHT SAFETY:"
    )

    print(
        "  Training performed: NO"
    )

    print(
        "  Optimizer created: NO"
    )

    print(
        "  Backward called: NO"
    )

    print(
        "  Positive controls: NO"
    )

    print(
        "  Sealed evaluation: NO"
    )

    print(
        "  Protected source modified: NO"
    )

    hashes = verify_artifacts()

    first_example = audit_frozen_training_contract()

    if first_example is None:
        raise RuntimeError(
            "No schedule example available for routing sanity check."
        )

    if torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )
    else:
        device = torch.device(
            "cpu"
        )

    model_contract = build_and_verify_t7(
        device,
        first_example,
    )

    result = {
        "experiment": (
            "DaveLM v0.9 Treatment #7"
        ),
        "treatment": (
            "Direct Query-to-Logit Pathway"
        ),
        "status": (
            "PREFLIGHT_PASS"
        ),
        "scientific_hypothesis": (
            "A direct trainable route from the native query-token embedding "
            "to answer-position vocabulary logits may be sufficient to break "
            "the paired reciprocal-selection failure seen in Treatments 5 and 6."
        ),
        "single_substantive_change": {
            "module": (
                "Linear(320 -> 1024, bias=True)"
            ),
            "input": (
                "existing native query-token embedding"
            ),
            "output": (
                "full-vocabulary logit bias"
            ),
            "application": (
                "answer position only"
            ),
            "explicit_candidate_information_supplied": False,
            "explicit_target_information_supplied": False,
            "explicit_distractor_information_supplied": False,
            "explicit_query_slot_label_supplied": False,
        },
        "frozen_contract": {
            "pair_pool_sha256": (
                EXPECTED_PAIR_POOL_SHA256
            ),
            "schedule_sha256": (
                EXPECTED_SCHEDULE_SHA256
            ),
            "start_checkpoint_sha256": (
                EXPECTED_START_CHECKPOINT_SHA256
            ),
            "steps": (
                EXPECTED_STEPS
            ),
            "batch_size": (
                EXPECTED_BATCH_SIZE
            ),
            "pairs_per_batch": (
                EXPECTED_PAIRS_PER_BATCH
            ),
            "events": (
                EXPECTED_EVENTS
            ),
            "slot0_events": (
                EXPECTED_SLOT0
            ),
            "slot1_events": (
                EXPECTED_SLOT1
            ),
            "total_supervised_positions": (
                EXPECTED_TOTAL_SUPERVISED
            ),
            "answer_replacements": (
                EXPECTED_ANSWER_REPLACEMENTS
            ),
            "ordinary_nonanswer_ce_positions": (
                EXPECTED_NONANSWER_SUPERVISED
            ),
            "answer_weight": (
                ANSWER_WEIGHT
            ),
            "margin": (
                MARGIN
            ),
        },
        "hashes_observed": (
            hashes
        ),
        "model_contract": (
            model_contract
        ),
        "future_retention_gate": {
            "correct_gt_distractor_fraction_min": (
                0.95
            ),
            "required_logit_margin": (
                0.5
            ),
            "margin_satisfied_fraction_min": (
                0.90
            ),
        },
        "safety": {
            "training_performed": False,
            "optimizer_created": False,
            "backward_called": False,
            "positive_controls_evaluated": False,
            "sealed_evaluation_opened": False,
            "protected_source_modified": False,
        },
    }

    write_json(
        PREFLIGHT_RESULT_PATH,
        result,
    )

    header(
        "TREATMENT #7 PREFLIGHT RESULT"
    )

    print(
        "RESULT: PASS"
    )

    print()

    print(
        "Base model/checkpoint: PASS"
    )

    print(
        "Frozen pair pool/schedule: PASS"
    )

    print(
        "Counterfactual-pair structure: PASS"
    )

    print(
        "New parameter family only: query_projection.weight/bias"
    )

    print(
        "Direct query-token embedding -> vocab-logit pathway: PASS"
    )

    print(
        "Answer-only bias construction: PASS"
    )

    print(
        "No candidate/target/distractor metadata fed to new pathway: PASS"
    )

    print()

    print(
        f"Expected T7 total params: "
        f"{EXPECTED_TOTAL_T7_PARAMETER_COUNT:,}"
    )

    print()

    print(
        "NO TRAINING PERFORMED."
    )

    print(
        "NO POSITIVE CONTROLS OPENED."
    )

    print(
        "NO SEALED EVALUATION OPENED."
    )

    print()

    print(
        "NEXT STEP:"
    )

    print(
        "  Build treatment7_train.py against this exact frozen contract."
    )

    print()

    print(
        "Preflight result:"
    )

    print(
        f"  {PREFLIGHT_RESULT_PATH}"
    )


if __name__ == "__main__":
    main()