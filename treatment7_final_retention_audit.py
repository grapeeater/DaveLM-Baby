from __future__ import annotations

"""
DaveLM v0.9 — Treatment #7 Final Training-Retention Audit

READ-ONLY.

This audit:
- verifies the frozen pair-pool SHA
- verifies the frozen schedule SHA
- verifies the exact final T7 checkpoint SHA
- reconstructs all 32,000 exact historical T7 training presentations
- rebuilds the exact T7 model:
      original_run.build_model("untied")
      + Linear(320 -> 1024, bias=True)
- restores BOTH base Baby + query projection
- reconstructs T7 adjusted answer logits:
      base_answer_logits + query_projection(query_token_embedding)
- aggregates exact final retention across all 32,000 events

This script DOES NOT:
- create an optimizer
- call backward()
- train Baby
- change parameters
- run positive controls
- open sealed novel/shortcut evaluation pools

Run:

    py -3.12 .\treatment7_final_retention_audit.py --device cuda --batch-size 256
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# PATHS
# =============================================================================

CADAVER_ROOT = Path(
    r"C:\DaveLM-CADAVER"
)

SOURCE_ROOT = Path(
    r"C:\DaveLM-v0.9"
)

PAIR_POOL_PATH = (
    CADAVER_ROOT
    / "treatment5_counterfactual_pairs_seed8382"
    / "treatment5_full_document_pair_pool.json"
)

SCHEDULE_PATH = (
    CADAVER_ROOT
    / "treatment5_counterfactual_pairs_seed8382"
    / "treatment5_full_document_frozen_schedule.json"
)

FINAL_CHECKPOINT_PATH = (
    CADAVER_ROOT
    / "treatment7_direct_query_logit_pathway_seed8380"
    / "checkpoints"
    / "direct_query_logit_pathway"
    / "seed_8380"
    / "latest.pt"
)

OUTPUT_PATH = (
    CADAVER_ROOT
    / "treatment7_direct_query_logit_pathway_seed8380"
    / "treatment7_final_retention_audit.json"
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

EXPECTED_FINAL_CHECKPOINT_SHA256 = (
    "01a973b87693e09d10a38cb92cd8ea6b"
    "061494d4badc87a0c83c8cafe4e6b1ae"
)

EXPECTED_TRAINER_SHA256 = (
    "5d7ae5e45c5150b4443e1e923d4280bd"
    "468ce3fb951a19fc7850e7f908cdfc97"
)


# =============================================================================
# FROZEN EXPERIMENT CONTRACT
# =============================================================================

EXPECTED_PAIR_COUNT = 1_536

EXPECTED_STEPS = 1_000

EXPECTED_BATCH_SIZE = 32

EXPECTED_PAIRS_PER_BATCH = 16

EXPECTED_EXAMPLES = 32_000

EXPECTED_SLOT0 = 16_000

EXPECTED_SLOT1 = 16_000

EXPECTED_PAIR_EXPOSURE_MIN = 10

EXPECTED_PAIR_EXPOSURE_MAX = 11

EXPECTED_TOTAL_SUPERVISED_POSITIONS = 6_144_000

EXPECTED_ANSWER_REPLACEMENTS = 32_000

EXPECTED_NONANSWER_CE_POSITIONS = 6_112_000

EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH = 193

EXPECTED_CAUSAL_SEQUENCE_LENGTH = 192

EXPECTED_NONANSWER_PER_EXAMPLE = 191

EXPECTED_BASE_PARAMETER_COUNT = 10_594_944

EXPECTED_HIDDEN_SIZE = 320

EXPECTED_VOCAB_SIZE = 1024

EXPECTED_QUERY_PROJECTION_PARAMETER_COUNT = 328_704

EXPECTED_TOTAL_PARAMETER_COUNT = 10_923_648

MARGIN = 0.5


# =============================================================================
# PRE-REGISTERED RETENTION GATE
# =============================================================================

RETENTION_CORRECT_GT_GATE = 0.95

RETENTION_MARGIN_VALUE = 0.5

RETENTION_MARGIN_FRACTION_GATE = 0.90


# =============================================================================
# HELPERS
# =============================================================================

def banner(
    text: str,
) -> None:
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)
    print()


def require(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise RuntimeError(
            message
        )


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


def verify_sha(
    path: Path,
    expected: str,
    label: str,
) -> str:
    require(
        path.is_file(),
        f"Missing {label}:\n{path}",
    )

    actual = sha256_file(
        path
    )

    print(
        f"{label} SHA256:"
    )

    print(
        f"  {actual}"
    )

    print(
        "Expected:"
    )

    print(
        f"  {expected}"
    )

    require(
        actual == expected,
        f"{label} SHA256 mismatch.",
    )

    print(
        f"{label} integrity: PASS"
    )

    print()

    return actual


def load_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
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
# IMPORT EXACT NATIVE BABY BUILDER
# =============================================================================

def import_native_module():
    if str(
        SOURCE_ROOT
    ) not in sys.path:
        sys.path.insert(
            0,
            str(SOURCE_ROOT),
        )

    from experiments.two_mapping_contextual_binding import run as original_run

    return original_run


# =============================================================================
# EXACT PAIR-POOL AUDIT
# =============================================================================

def audit_pair_pool(
    pair_pool: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    banner(
        "AUDITING FROZEN COUNTERFACTUAL PAIR POOL"
    )

    require(
        isinstance(
            pair_pool,
            dict,
        ),
        "Pair-pool root is not a dictionary.",
    )

    pairs = pair_pool.get(
        "pairs"
    )

    require(
        isinstance(
            pairs,
            list,
        ),
        "Pair-pool 'pairs' is not a list.",
    )

    require(
        len(pairs) == EXPECTED_PAIR_COUNT,
        f"Expected {EXPECTED_PAIR_COUNT} pairs; "
        f"found {len(pairs)}.",
    )

    pair_lookup: dict[
        str,
        dict[str, Any],
    ] = {}

    query_positions: set[int] = set()

    answer_indices: set[int] = set()

    answer_causal_positions: set[int] = set()

    for pair in pairs:
        pair_id = str(
            pair[
                "pair_id"
            ]
        )

        require(
            pair_id not in pair_lookup,
            f"Duplicate pair_id: {pair_id}",
        )

        twin_a = pair[
            "twin_a"
        ]

        twin_b = pair[
            "twin_b"
        ]

        require(
            str(
                twin_a[
                    "twin"
                ]
            ) == "A",
            f"{pair_id}: twin_a is not A.",
        )

        require(
            str(
                twin_b[
                    "twin"
                ]
            ) == "B",
            f"{pair_id}: twin_b is not B.",
        )

        require(
            int(
                twin_a[
                    "query_slot"
                ]
            ) == 0,
            f"{pair_id}: Twin A query slot != 0.",
        )

        require(
            int(
                twin_b[
                    "query_slot"
                ]
            ) == 1,
            f"{pair_id}: Twin B query slot != 1.",
        )

        a_target = int(
            twin_a[
                "target_token_id"
            ]
        )

        a_distractor = int(
            twin_a[
                "distractor_token_id"
            ]
        )

        b_target = int(
            twin_b[
                "target_token_id"
            ]
        )

        b_distractor = int(
            twin_b[
                "distractor_token_id"
            ]
        )

        require(
            a_target != a_distractor,
            f"{pair_id}: Twin A target == distractor.",
        )

        require(
            b_target != b_distractor,
            f"{pair_id}: Twin B target == distractor.",
        )

        require(
            a_target == b_distractor,
            f"{pair_id}: A target != B distractor.",
        )

        require(
            b_target == a_distractor,
            f"{pair_id}: B target != A distractor.",
        )

        query_position = int(
            pair[
                "query_difference_position"
            ]
        )

        answer_index = int(
            pair[
                "answer_token_index"
            ]
        )

        answer_causal_position = int(
            pair[
                "answer_causal_position"
            ]
        )

        require(
            answer_causal_position == (
                answer_index - 1
            ),
            f"{pair_id}: answer causal position mismatch.",
        )

        require(
            query_position < answer_index,
            f"{pair_id}: query position is not before answer.",
        )

        full_a = [
            int(x)
            for x in twin_a[
                "full_document_token_ids"
            ]
        ]

        full_b = [
            int(x)
            for x in twin_b[
                "full_document_token_ids"
            ]
        ]

        require(
            len(full_a) == (
                EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH
            ),
            f"{pair_id}: Twin A document length mismatch.",
        )

        require(
            len(full_b) == (
                EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH
            ),
            f"{pair_id}: Twin B document length mismatch.",
        )

        require(
            full_a[
                answer_index
            ] == a_target,
            f"{pair_id}: Twin A answer token != target.",
        )

        require(
            full_b[
                answer_index
            ] == b_target,
            f"{pair_id}: Twin B answer token != target.",
        )

        prefix_a = [
            int(x)
            for x in twin_a[
                "prefix_token_ids"
            ]
        ]

        prefix_b = [
            int(x)
            for x in twin_b[
                "prefix_token_ids"
            ]
        ]

        require(
            len(prefix_a) == len(prefix_b),
            f"{pair_id}: prefix lengths differ.",
        )

        prefix_differences = [
            index
            for index, (
                a,
                b,
            ) in enumerate(
                zip(
                    prefix_a,
                    prefix_b,
                )
            )
            if a != b
        ]

        require(
            prefix_differences == [
                query_position
            ],
            f"{pair_id}: prefixes should differ only at query "
            f"position {query_position}; "
            f"got {prefix_differences}.",
        )

        full_differences = [
            index
            for index, (
                a,
                b,
            ) in enumerate(
                zip(
                    full_a,
                    full_b,
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

        require(
            full_differences == expected_differences,
            f"{pair_id}: full twins should differ exactly at "
            f"query + answer positions "
            f"{expected_differences}; "
            f"got {full_differences}.",
        )

        require(
            int(
                pair[
                    "supervised_token_count"
                ]
            ) == (
                EXPECTED_CAUSAL_SEQUENCE_LENGTH
            ),
            f"{pair_id}: supervised-token count mismatch.",
        )

        require(
            int(
                pair[
                    "nonanswer_supervised_token_count"
                ]
            ) == (
                EXPECTED_NONANSWER_PER_EXAMPLE
            ),
            f"{pair_id}: non-answer-token count mismatch.",
        )

        query_positions.add(
            query_position
        )

        answer_indices.add(
            answer_index
        )

        answer_causal_positions.add(
            answer_causal_position
        )

        pair_lookup[
            pair_id
        ] = pair

    print(
        f"Pairs: {len(pair_lookup)}"
    )

    print(
        f"Query-token positions: "
        f"{sorted(query_positions)}"
    )

    print(
        f"Answer-token indices: "
        f"{sorted(answer_indices)}"
    )

    print(
        f"Answer causal positions: "
        f"{sorted(answer_causal_positions)}"
    )

    print()

    print(
        "Twin reciprocity: PASS"
    )

    print(
        "Prefix query-only difference: PASS"
    )

    print(
        "Full-document query+answer difference: PASS"
    )

    print(
        "Exact answer-position structure: PASS"
    )

    print(
        "Frozen pair-pool structural audit: PASS"
    )

    return pair_lookup


# =============================================================================
# EXACT FROZEN SCHEDULE AUDIT
# =============================================================================

def audit_schedule(
    schedule: dict[str, Any],
    pair_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    banner(
        "AUDITING EXACT 32,000-EVENT FROZEN SCHEDULE"
    )

    expected_values = {
        "schedule_seed": 8382,
        "steps": EXPECTED_STEPS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "pairs_per_batch": EXPECTED_PAIRS_PER_BATCH,
        "pair_presentations": 16_000,
        "scheduled_examples": EXPECTED_EXAMPLES,
        "slot_0_presentations": EXPECTED_SLOT0,
        "slot_1_presentations": EXPECTED_SLOT1,
        "pair_exposure_min": EXPECTED_PAIR_EXPOSURE_MIN,
        "pair_exposure_max": EXPECTED_PAIR_EXPOSURE_MAX,
        "answer_substitutions": EXPECTED_ANSWER_REPLACEMENTS,
        "total_supervised_tokens": EXPECTED_TOTAL_SUPERVISED_POSITIONS,
        "nonanswer_supervised_tokens": EXPECTED_NONANSWER_CE_POSITIONS,
    }

    for key, expected in expected_values.items():
        require(
            key in schedule,
            f"Frozen schedule missing field {key!r}.",
        )

        require(
            int(
                schedule[
                    key
                ]
            ) == int(
                expected
            ),
            f"Frozen schedule field {key!r} mismatch: "
            f"expected {expected}, "
            f"observed {schedule[key]}.",
        )

    steps_data = schedule.get(
        "steps_data"
    )

    require(
        isinstance(
            steps_data,
            list,
        ),
        "schedule['steps_data'] is not a list.",
    )

    require(
        len(steps_data) == EXPECTED_STEPS,
        f"Expected {EXPECTED_STEPS} steps; "
        f"found {len(steps_data)}.",
    )

    events: list[
        dict[str, Any]
    ] = []

    slot_counts = Counter()

    pair_exposure = Counter()

    observed_supervised = 0

    observed_nonanswer = 0

    observed_answers = 0

    for expected_step, step_data in enumerate(
        steps_data,
        start=1,
    ):
        require(
            int(
                step_data[
                    "step"
                ]
            ) == expected_step,
            f"Step ordering mismatch at {expected_step}.",
        )

        examples = step_data.get(
            "examples"
        )

        require(
            isinstance(
                examples,
                list,
            ),
            f"Step {expected_step}: examples is not a list.",
        )

        require(
            len(examples) == EXPECTED_BATCH_SIZE,
            f"Step {expected_step}: "
            f"expected 32 examples; "
            f"got {len(examples)}.",
        )

        for offset in range(
            0,
            EXPECTED_BATCH_SIZE,
            2,
        ):
            a = examples[
                offset
            ]

            b = examples[
                offset + 1
            ]

            require(
                str(
                    a[
                        "pair_id"
                    ]
                ) == str(
                    b[
                        "pair_id"
                    ]
                ),
                f"Step {expected_step}: pair adjacency failure.",
            )

            require(
                str(
                    a[
                        "twin"
                    ]
                ) == "A",
                f"Step {expected_step}: expected Twin A.",
            )

            require(
                str(
                    b[
                        "twin"
                    ]
                ) == "B",
                f"Step {expected_step}: expected Twin B.",
            )

            require(
                int(
                    a[
                        "query_slot"
                    ]
                ) == 0,
                f"Step {expected_step}: Twin A slot != 0.",
            )

            require(
                int(
                    b[
                        "query_slot"
                    ]
                ) == 1,
                f"Step {expected_step}: Twin B slot != 1.",
            )

            require(
                int(
                    a[
                        "target_token_id"
                    ]
                ) == int(
                    b[
                        "distractor_token_id"
                    ]
                ),
                f"Step {expected_step}: "
                f"A target != B distractor.",
            )

            require(
                int(
                    b[
                        "target_token_id"
                    ]
                ) == int(
                    a[
                        "distractor_token_id"
                    ]
                ),
                f"Step {expected_step}: "
                f"B target != A distractor.",
            )

            pair_id = str(
                a[
                    "pair_id"
                ]
            )

            pair_exposure[
                pair_id
            ] += 1

        for example in examples:
            pair_id = str(
                example[
                    "pair_id"
                ]
            )

            require(
                pair_id in pair_lookup,
                f"Step {expected_step}: "
                f"unknown pair_id {pair_id}.",
            )

            twin_name = str(
                example[
                    "twin"
                ]
            )

            if twin_name == "A":
                frozen_twin = pair_lookup[
                    pair_id
                ][
                    "twin_a"
                ]

            elif twin_name == "B":
                frozen_twin = pair_lookup[
                    pair_id
                ][
                    "twin_b"
                ]

            else:
                raise RuntimeError(
                    f"Step {expected_step}: "
                    f"invalid twin {twin_name!r}."
                )

            full_ids = [
                int(x)
                for x in example[
                    "full_document_token_ids"
                ]
            ]

            frozen_ids = [
                int(x)
                for x in frozen_twin[
                    "full_document_token_ids"
                ]
            ]

            require(
                full_ids == frozen_ids,
                f"Step {expected_step}, "
                f"{pair_id} Twin {twin_name}: "
                f"schedule document differs from pair pool.",
            )

            require(
                len(full_ids) == (
                    EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH
                ),
                f"Step {expected_step}: "
                f"document length mismatch.",
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

            require(
                answer_position == (
                    answer_index - 1
                ),
                f"Step {expected_step}: "
                f"answer causal-position mismatch.",
            )

            require(
                query_position < answer_index,
                f"Step {expected_step}: "
                f"query position not before answer.",
            )

            require(
                answer_index == int(
                    pair_lookup[
                        pair_id
                    ][
                        "answer_token_index"
                    ]
                ),
                f"Step {expected_step}: "
                f"answer index differs from pair pool.",
            )

            require(
                answer_position == int(
                    pair_lookup[
                        pair_id
                    ][
                        "answer_causal_position"
                    ]
                ),
                f"Step {expected_step}: "
                f"answer causal position differs from pair pool.",
            )

            require(
                query_position == int(
                    pair_lookup[
                        pair_id
                    ][
                        "query_difference_position"
                    ]
                ),
                f"Step {expected_step}: "
                f"query position differs from pair pool.",
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

            require(
                target != distractor,
                f"Step {expected_step}: target == distractor.",
            )

            require(
                target == int(
                    frozen_twin[
                        "target_token_id"
                    ]
                ),
                f"Step {expected_step}: "
                f"target differs from pair pool.",
            )

            require(
                distractor == int(
                    frozen_twin[
                        "distractor_token_id"
                    ]
                ),
                f"Step {expected_step}: "
                f"distractor differs from pair pool.",
            )

            require(
                full_ids[
                    answer_index
                ] == target,
                f"Step {expected_step}: "
                f"full document answer token != target.",
            )

            supervised_count = int(
                example[
                    "supervised_token_count"
                ]
            )

            nonanswer_count = int(
                example[
                    "nonanswer_supervised_token_count"
                ]
            )

            require(
                supervised_count == (
                    EXPECTED_CAUSAL_SEQUENCE_LENGTH
                ),
                f"Step {expected_step}: "
                f"supervised count != 192.",
            )

            require(
                nonanswer_count == (
                    EXPECTED_NONANSWER_PER_EXAMPLE
                ),
                f"Step {expected_step}: "
                f"non-answer count != 191.",
            )

            slot = int(
                example[
                    "query_slot"
                ]
            )

            require(
                slot in (
                    0,
                    1,
                ),
                f"Step {expected_step}: "
                f"invalid query slot.",
            )

            slot_counts[
                slot
            ] += 1

            observed_supervised += (
                supervised_count
            )

            observed_nonanswer += (
                nonanswer_count
            )

            observed_answers += 1

            events.append(
                {
                    "step": (
                        expected_step
                    ),
                    "pair_id": (
                        pair_id
                    ),
                    "query_slot": (
                        slot
                    ),
                    "twin": (
                        twin_name
                    ),
                    "full_document_token_ids": (
                        full_ids
                    ),
                    "query_difference_position": (
                        query_position
                    ),
                    "answer_token_index": (
                        answer_index
                    ),
                    "answer_causal_position": (
                        answer_position
                    ),
                    "target_token_id": (
                        target
                    ),
                    "distractor_token_id": (
                        distractor
                    ),
                }
            )

    require(
        len(events) == EXPECTED_EXAMPLES,
        f"Expected {EXPECTED_EXAMPLES} events; "
        f"got {len(events)}.",
    )

    require(
        slot_counts[
            0
        ] == EXPECTED_SLOT0,
        f"Slot-0 count mismatch: "
        f"{slot_counts[0]}.",
    )

    require(
        slot_counts[
            1
        ] == EXPECTED_SLOT1,
        f"Slot-1 count mismatch: "
        f"{slot_counts[1]}.",
    )

    require(
        observed_answers == (
            EXPECTED_ANSWER_REPLACEMENTS
        ),
        "Answer replacement count mismatch.",
    )

    require(
        observed_supervised == (
            EXPECTED_TOTAL_SUPERVISED_POSITIONS
        ),
        "Total supervised-position count mismatch.",
    )

    require(
        observed_nonanswer == (
            EXPECTED_NONANSWER_CE_POSITIONS
        ),
        "Non-answer CE count mismatch.",
    )

    require(
        len(
            pair_exposure
        ) == EXPECTED_PAIR_COUNT,
        "Not every pair appeared in frozen schedule.",
    )

    require(
        min(
            pair_exposure.values()
        ) == (
            EXPECTED_PAIR_EXPOSURE_MIN
        ),
        "Minimum pair exposure mismatch.",
    )

    require(
        max(
            pair_exposure.values()
        ) == (
            EXPECTED_PAIR_EXPOSURE_MAX
        ),
        "Maximum pair exposure mismatch.",
    )

    print(
        f"Steps: "
        f"{EXPECTED_STEPS}"
    )

    print(
        f"Batch size: "
        f"{EXPECTED_BATCH_SIZE}"
    )

    print(
        f"Complete twin pairs / batch: "
        f"{EXPECTED_PAIRS_PER_BATCH}"
    )

    print(
        f"Historical events: "
        f"{len(events)}"
    )

    print(
        f"Slot 0: "
        f"{slot_counts[0]}"
    )

    print(
        f"Slot 1: "
        f"{slot_counts[1]}"
    )

    print(
        f"Total supervised positions: "
        f"{observed_supervised}"
    )

    print(
        f"Answer replacements: "
        f"{observed_answers}"
    )

    print(
        f"Ordinary non-answer CE positions: "
        f"{observed_nonanswer}"
    )

    print(
        f"Pair exposure range: "
        f"{min(pair_exposure.values())}-"
        f"{max(pair_exposure.values())}"
    )

    print()

    print(
        "Schedule/pair exact document cross-check: PASS"
    )

    print(
        "Complete twin-pair schedule structure: PASS"
    )

    print(
        "Exact 32,000-event schedule audit: PASS"
    )

    return events


# =============================================================================
# T7 MODEL
# =============================================================================

class Treatment7QueryLogitModel(
    nn.Module
):
    def __init__(
        self,
        base_model: nn.Module,
    ):
        super().__init__()

        self.base_model = (
            base_model
        )

        token_embedding = getattr(
            base_model,
            "token_embedding",
            None,
        )

        require(
            isinstance(
                token_embedding,
                nn.Embedding,
            ),
            "Base Baby does not expose expected token_embedding.",
        )

        require(
            int(
                token_embedding.embedding_dim
            ) == EXPECTED_HIDDEN_SIZE,
            "Base hidden size mismatch.",
        )

        require(
            int(
                token_embedding.num_embeddings
            ) == EXPECTED_VOCAB_SIZE,
            "Base vocabulary size mismatch.",
        )

        self.query_projection = nn.Linear(
            EXPECTED_HIDDEN_SIZE,
            EXPECTED_VOCAB_SIZE,
            bias=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ):
        return self.base_model(
            input_ids
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


# =============================================================================
# MODEL OUTPUT HELPER
# =============================================================================

def extract_logits(
    output: Any,
) -> torch.Tensor:
    if torch.is_tensor(
        output
    ):
        logits = output

    elif isinstance(
        output,
        dict,
    ) and torch.is_tensor(
        output.get(
            "logits"
        )
    ):
        logits = output[
            "logits"
        ]

    elif isinstance(
        output,
        (
            tuple,
            list,
        ),
    ):
        logits = None

        for candidate in output:
            if (
                torch.is_tensor(
                    candidate
                )
                and candidate.ndim == 3
            ):
                logits = candidate
                break

        require(
            logits is not None,
            "Could not find 3D logits tensor in model output.",
        )

    elif torch.is_tensor(
        getattr(
            output,
            "logits",
            None,
        )
    ):
        logits = output.logits

    else:
        raise RuntimeError(
            "Could not extract logits from Baby output."
        )

    require(
        logits.ndim == 3,
        f"Expected [batch,time,vocab] logits; "
        f"got {tuple(logits.shape)}.",
    )

    return logits


# =============================================================================
# LOAD EXACT T7 FINAL CHECKPOINT
# =============================================================================

def load_t7_model(
    device: torch.device,
) -> tuple[
    Treatment7QueryLogitModel,
    dict[str, Any],
]:
    banner(
        "LOADING EXACT FINAL T7 CHECKPOINT"
    )

    original_run = (
        import_native_module()
    )

    build_model_fn = getattr(
        original_run,
        "build_model",
        None,
    )

    require(
        callable(
            build_model_fn
        ),
        'Native module does not expose build_model("untied").',
    )

    base_model = build_model_fn(
        "untied"
    )

    base_parameter_count = sum(
        parameter.numel()
        for parameter
        in base_model.parameters()
    )

    require(
        base_parameter_count == (
            EXPECTED_BASE_PARAMETER_COUNT
        ),
        f"Base parameter count mismatch: "
        f"{base_parameter_count:,}.",
    )

    model = Treatment7QueryLogitModel(
        base_model
    )

    projection_parameter_count = sum(
        parameter.numel()
        for parameter
        in model.query_projection.parameters()
    )

    total_parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    require(
        projection_parameter_count == (
            EXPECTED_QUERY_PROJECTION_PARAMETER_COUNT
        ),
        f"Projection parameter count mismatch: "
        f"{projection_parameter_count:,}.",
    )

    require(
        total_parameter_count == (
            EXPECTED_TOTAL_PARAMETER_COUNT
        ),
        f"Total parameter count mismatch: "
        f"{total_parameter_count:,}.",
    )

    checkpoint = torch.load(
        FINAL_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    require(
        isinstance(
            checkpoint,
            dict,
        ),
        "Final checkpoint root is not a dictionary.",
    )

    require(
        int(
            checkpoint.get(
                "step",
                -1,
            )
        ) == 1000,
        "Final T7 checkpoint is not step 1000.",
    )

    require(
        checkpoint.get(
            "trainer_sha256"
        ) == EXPECTED_TRAINER_SHA256,
        "Checkpoint trainer SHA does not match canonical T7 trainer.",
    )

    require(
        int(
            checkpoint.get(
                "parameter_count",
                -1,
            )
        ) == (
            EXPECTED_TOTAL_PARAMETER_COUNT
        ),
        "Checkpoint total parameter-count metadata mismatch.",
    )

    require(
        int(
            checkpoint.get(
                "base_parameter_count",
                -1,
            )
        ) == (
            EXPECTED_BASE_PARAMETER_COUNT
        ),
        "Checkpoint base parameter-count metadata mismatch.",
    )

    require(
        int(
            checkpoint.get(
                "query_projection_parameter_count",
                -1,
            )
        ) == (
            EXPECTED_QUERY_PROJECTION_PARAMETER_COUNT
        ),
        "Checkpoint projection parameter-count metadata mismatch.",
    )

    state = checkpoint.get(
        "model_state"
    )

    require(
        isinstance(
            state,
            dict,
        ),
        "Checkpoint does not contain model_state dictionary.",
    )

    require(
        "query_projection.weight" in state,
        "Checkpoint missing query_projection.weight.",
    )

    require(
        "query_projection.bias" in state,
        "Checkpoint missing query_projection.bias.",
    )

    require(
        any(
            key.startswith(
                "base_model."
            )
            for key
            in state
        ),
        "Checkpoint missing base_model.* state keys.",
    )

    incompatible = model.load_state_dict(
        state,
        strict=True,
    )

    require(
        not incompatible.missing_keys,
        f"Missing checkpoint keys: "
        f"{incompatible.missing_keys}",
    )

    require(
        not incompatible.unexpected_keys,
        f"Unexpected checkpoint keys: "
        f"{incompatible.unexpected_keys}",
    )

    model.to(
        device
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )

    print(
        f"Device: "
        f"{device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Base parameters: "
        f"{base_parameter_count:,}"
    )

    print(
        f"Query-projection parameters: "
        f"{projection_parameter_count:,}"
    )

    print(
        f"Total parameters: "
        f"{total_parameter_count:,}"
    )

    print()

    print(
        'Base builder: original_run.build_model("untied")'
    )

    print(
        "T7 projection: Linear(320 -> 1024, bias=True)"
    )

    print(
        "Checkpoint wrapper state: PASS"
    )

    print(
        "model.eval(): YES"
    )

    print(
        "requires_grad: FALSE"
    )

    print(
        "Optimizer: NONE"
    )

    print(
        "Backward: NONE"
    )

    print(
        "Training: NONE"
    )

    print(
        "Positive controls: NO"
    )

    print(
        "Sealed evaluation: UNTOUCHED"
    )

    return (
        model,
        checkpoint,
    )


# =============================================================================
# METRICS
# =============================================================================

class Metrics:
    def __init__(
        self,
    ) -> None:
        self.count = 0

        self.top1 = 0

        self.top5 = 0

        self.correct_gt_distractor = 0

        self.margin_satisfied = 0

        self.sum_candidate_mass = 0.0

        self.sum_membership_loss = 0.0

        self.sum_selector_loss = 0.0

        self.sum_replacement_loss = 0.0

        self.sum_target_probability = 0.0

        self.sum_distractor_probability = 0.0

        self.sum_logit_delta = 0.0

        self.ranks: list[
            int
        ] = []

    def add(
        self,
        answer_logits: torch.Tensor,
        target_ids: torch.Tensor,
        distractor_ids: torch.Tensor,
    ) -> None:
        batch_size = int(
            answer_logits.shape[
                0
            ]
        )

        rows = torch.arange(
            batch_size,
            dtype=torch.long,
            device=answer_logits.device,
        )

        target_logits = answer_logits[
            rows,
            target_ids,
        ]

        distractor_logits = answer_logits[
            rows,
            distractor_ids,
        ]

        delta = (
            target_logits
            - distractor_logits
        )

        full_lse = torch.logsumexp(
            answer_logits,
            dim=-1,
        )

        pair_lse = torch.logsumexp(
            torch.stack(
                (
                    target_logits,
                    distractor_logits,
                ),
                dim=-1,
            ),
            dim=-1,
        )

        membership_loss = (
            full_lse
            - pair_lse
        )

        selector_loss = F.relu(
            MARGIN
            - delta
        )

        replacement_loss = (
            membership_loss
            + selector_loss
        )

        probabilities = torch.softmax(
            answer_logits,
            dim=-1,
        )

        target_probability = probabilities[
            rows,
            target_ids,
        ]

        distractor_probability = probabilities[
            rows,
            distractor_ids,
        ]

        candidate_mass = (
            target_probability
            + distractor_probability
        )

        predicted_ids = answer_logits.argmax(
            dim=-1
        )

        top5_ids = answer_logits.topk(
            k=5,
            dim=-1,
        ).indices

        top5_correct = (
            top5_ids
            == target_ids.unsqueeze(
                -1
            )
        ).any(
            dim=-1
        )

        ranks = (
            1
            + (
                answer_logits
                > target_logits.unsqueeze(
                    -1
                )
            ).sum(
                dim=-1
            )
        )

        self.count += (
            batch_size
        )

        self.top1 += int(
            (
                predicted_ids
                == target_ids
            ).sum().item()
        )

        self.top5 += int(
            top5_correct.sum().item()
        )

        self.correct_gt_distractor += int(
            (
                delta > 0.0
            ).sum().item()
        )

        self.margin_satisfied += int(
            (
                delta >= MARGIN
            ).sum().item()
        )

        self.sum_candidate_mass += float(
            candidate_mass.sum().item()
        )

        self.sum_membership_loss += float(
            membership_loss.sum().item()
        )

        self.sum_selector_loss += float(
            selector_loss.sum().item()
        )

        self.sum_replacement_loss += float(
            replacement_loss.sum().item()
        )

        self.sum_target_probability += float(
            target_probability.sum().item()
        )

        self.sum_distractor_probability += float(
            distractor_probability.sum().item()
        )

        self.sum_logit_delta += float(
            delta.sum().item()
        )

        self.ranks.extend(
            int(
                rank
            )
            for rank
            in ranks.detach().cpu().tolist()
        )

    def summary(
        self,
    ) -> dict[str, Any]:
        require(
            self.count > 0,
            "Cannot summarize zero events.",
        )

        return {
            "events": (
                self.count
            ),
            "full_vocab_top1": (
                self.top1
                / self.count
            ),
            "full_vocab_top5": (
                self.top5
                / self.count
            ),
            "correct_gt_distractor": (
                self.correct_gt_distractor
                / self.count
            ),
            "margin_ge_0_5": (
                self.margin_satisfied
                / self.count
            ),
            "candidate_mass": (
                self.sum_candidate_mass
                / self.count
            ),
            "membership_loss": (
                self.sum_membership_loss
                / self.count
            ),
            "selector_loss": (
                self.sum_selector_loss
                / self.count
            ),
            "raw_answer_replacement_loss": (
                self.sum_replacement_loss
                / self.count
            ),
            "target_probability": (
                self.sum_target_probability
                / self.count
            ),
            "distractor_probability": (
                self.sum_distractor_probability
                / self.count
            ),
            "mean_logit_delta": (
                self.sum_logit_delta
                / self.count
            ),
            "target_rank_mean": (
                statistics.fmean(
                    self.ranks
                )
            ),
            "target_rank_median": (
                statistics.median(
                    self.ranks
                )
            ),
        }


# =============================================================================
# EXACT 32K T7 RETENTION REPLAY
# =============================================================================

def run_retention_audit(
    model: Treatment7QueryLogitModel,
    events: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    banner(
        "REPLAYING ALL 32,000 EXACT HISTORICAL T7 EVENTS"
    )

    overall = Metrics()

    slot0 = Metrics()

    slot1 = Metrics()

    processed = 0

    next_progress = 3_200

    with torch.inference_mode():
        for start in range(
            0,
            len(events),
            batch_size,
        ):
            batch = events[
                start:
                start + batch_size
            ]

            full_ids = torch.tensor(
                [
                    event[
                        "full_document_token_ids"
                    ]
                    for event
                    in batch
                ],
                dtype=torch.long,
                device=device,
            )

            require(
                full_ids.ndim == 2,
                "Full-document batch is not rank-2.",
            )

            require(
                full_ids.shape[
                    1
                ] == (
                    EXPECTED_MODEL_VISIBLE_DOCUMENT_LENGTH
                ),
                "Full-document batch length mismatch.",
            )

            inputs = (
                full_ids[
                    :,
                    :-1,
                ]
                .contiguous()
            )

            require(
                inputs.shape[
                    1
                ] == (
                    EXPECTED_CAUSAL_SEQUENCE_LENGTH
                ),
                "Causal input sequence length mismatch.",
            )

            local_batch_size = int(
                inputs.shape[
                    0
                ]
            )

            rows = torch.arange(
                local_batch_size,
                dtype=torch.long,
                device=device,
            )

            answer_positions = torch.tensor(
                [
                    event[
                        "answer_causal_position"
                    ]
                    for event
                    in batch
                ],
                dtype=torch.long,
                device=device,
            )

            query_positions = torch.tensor(
                [
                    event[
                        "query_difference_position"
                    ]
                    for event
                    in batch
                ],
                dtype=torch.long,
                device=device,
            )

            target_ids = torch.tensor(
                [
                    event[
                        "target_token_id"
                    ]
                    for event
                    in batch
                ],
                dtype=torch.long,
                device=device,
            )

            distractor_ids = torch.tensor(
                [
                    event[
                        "distractor_token_id"
                    ]
                    for event
                    in batch
                ],
                dtype=torch.long,
                device=device,
            )

            slots = torch.tensor(
                [
                    event[
                        "query_slot"
                    ]
                    for event
                    in batch
                ],
                dtype=torch.long,
                device=device,
            )

            query_token_ids = inputs[
                rows,
                query_positions,
            ]

            base_output = model(
                inputs
            )

            base_logits = extract_logits(
                base_output
            )

            require(
                tuple(
                    base_logits.shape[
                        :2
                    ]
                ) == tuple(
                    inputs.shape
                ),
                "Base-logit time shape mismatch.",
            )

            require(
                base_logits.shape[
                    -1
                ] == EXPECTED_VOCAB_SIZE,
                "Base-logit vocabulary mismatch.",
            )

            base_answer_logits = base_logits[
                rows,
                answer_positions,
                :
            ]

            query_bias = (
                model.get_query_bias(
                    query_token_ids
                )
            )

            require(
                query_bias.shape == (
                    local_batch_size,
                    EXPECTED_VOCAB_SIZE,
                ),
                "Query-bias shape mismatch.",
            )

            adjusted_answer_logits = (
                base_answer_logits
                + query_bias
            )

            overall.add(
                adjusted_answer_logits,
                target_ids,
                distractor_ids,
            )

            mask0 = (
                slots == 0
            )

            if bool(
                mask0.any()
            ):
                slot0.add(
                    adjusted_answer_logits[
                        mask0
                    ],
                    target_ids[
                        mask0
                    ],
                    distractor_ids[
                        mask0
                    ],
                )

            mask1 = (
                slots == 1
            )

            if bool(
                mask1.any()
            ):
                slot1.add(
                    adjusted_answer_logits[
                        mask1
                    ],
                    target_ids[
                        mask1
                    ],
                    distractor_ids[
                        mask1
                    ],
                )

            processed += (
                local_batch_size
            )

            while (
                processed >= next_progress
                and next_progress <= (
                    EXPECTED_EXAMPLES
                )
            ):
                print(
                    f"  audited at least "
                    f"{next_progress:>5}/"
                    f"{EXPECTED_EXAMPLES} events"
                )

                next_progress += (
                    3_200
                )

    require(
        overall.count == (
            EXPECTED_EXAMPLES
        ),
        f"Overall event count mismatch: "
        f"{overall.count}",
    )

    require(
        slot0.count == (
            EXPECTED_SLOT0
        ),
        f"Slot-0 event count mismatch: "
        f"{slot0.count}",
    )

    require(
        slot1.count == (
            EXPECTED_SLOT1
        ),
        f"Slot-1 event count mismatch: "
        f"{slot1.count}",
    )

    return (
        overall.summary(),
        slot0.summary(),
        slot1.summary(),
    )


# =============================================================================
# REPORTING
# =============================================================================

def print_summary(
    title: str,
    summary: dict[str, Any],
) -> None:
    print(
        title
    )

    print(
        f"  events:                     "
        f"{summary['events']}"
    )

    print(
        f"  full-vocab top1:            "
        f"{summary['full_vocab_top1']:.6f}"
    )

    print(
        f"  full-vocab top5:            "
        f"{summary['full_vocab_top5']:.6f}"
    )

    print(
        f"  correct > distractor:       "
        f"{summary['correct_gt_distractor']:.6f}"
    )

    print(
        f"  margin >= 0.5:              "
        f"{summary['margin_ge_0_5']:.6f}"
    )

    print(
        f"  candidate mass:             "
        f"{summary['candidate_mass']:.6f}"
    )

    print(
        f"  membership loss:            "
        f"{summary['membership_loss']:.6f}"
    )

    print(
        f"  selector loss:              "
        f"{summary['selector_loss']:.6f}"
    )

    print(
        f"  raw replacement loss:       "
        f"{summary['raw_answer_replacement_loss']:.6f}"
    )

    print(
        f"  target probability:         "
        f"{summary['target_probability']:.6f}"
    )

    print(
        f"  distractor probability:     "
        f"{summary['distractor_probability']:.6f}"
    )

    print(
        f"  mean logit delta:           "
        f"{summary['mean_logit_delta']:+.6f}"
    )

    print(
        f"  target rank mean:           "
        f"{summary['target_rank_mean']:.6f}"
    )

    print(
        f"  target rank median:         "
        f"{summary['target_rank_median']:.6f}"
    )

    print()


def classify_retention(
    overall: dict[str, Any],
) -> tuple[
    str,
    bool,
    bool,
]:
    correct_gate_pass = (
        overall[
            "correct_gt_distractor"
        ]
        >= RETENTION_CORRECT_GT_GATE
    )

    margin_gate_pass = (
        overall[
            "margin_ge_0_5"
        ]
        >= RETENTION_MARGIN_FRACTION_GATE
    )

    if (
        correct_gate_pass
        and margin_gate_pass
    ):
        classification = (
            "STRONG_FINAL_TRAINING_RETENTION"
        )

    else:
        classification = (
            "FINAL_TRAINING_RETENTION_BELOW_GATE"
        )

    return (
        classification,
        correct_gate_pass,
        margin_gate_pass,
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only final 32,000-event "
            "retention audit for DaveLM Treatment #7."
        )
    )

    parser.add_argument(
        "--device",
        choices=(
            "cuda",
            "cpu",
        ),
        default="cuda",
        help=(
            "Inference device. "
            "ROCm PyTorch uses 'cuda' for AMD GPUs."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help=(
            "Read-only inference batch size. "
            "Default: 256."
        ),
    )

    args = parser.parse_args()

    require(
        args.batch_size >= 1,
        "--batch-size must be >= 1.",
    )

    run_start = (
        time.perf_counter()
    )

    banner(
        "TREATMENT #7 FINAL TRAINING-RETENTION AUDIT — READ ONLY"
    )

    banner(
        "VERIFYING FROZEN ARTIFACTS"
    )

    pair_pool_sha = verify_sha(
        PAIR_POOL_PATH,
        EXPECTED_PAIR_POOL_SHA256,
        "Pair pool",
    )

    schedule_sha = verify_sha(
        SCHEDULE_PATH,
        EXPECTED_SCHEDULE_SHA256,
        "Frozen schedule",
    )

    checkpoint_sha = verify_sha(
        FINAL_CHECKPOINT_PATH,
        EXPECTED_FINAL_CHECKPOINT_SHA256,
        "Final T7 checkpoint",
    )

    pair_pool = load_json(
        PAIR_POOL_PATH
    )

    schedule = load_json(
        SCHEDULE_PATH
    )

    pair_lookup = audit_pair_pool(
        pair_pool
    )

    events = audit_schedule(
        schedule,
        pair_lookup,
    )

    banner(
        "DEVICE"
    )

    if args.device == "cuda":
        require(
            torch.cuda.is_available(),
            "CUDA/ROCm requested but "
            "torch.cuda.is_available() returned False.",
        )

    device = torch.device(
        args.device
    )

    print(
        f"Audit device: {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    model, checkpoint = load_t7_model(
        device
    )

    eval_start = (
        time.perf_counter()
    )

    (
        overall,
        slot0,
        slot1,
    ) = run_retention_audit(
        model=model,
        events=events,
        device=device,
        batch_size=args.batch_size,
    )

    eval_runtime = (
        time.perf_counter()
        - eval_start
    )

    total_runtime = (
        time.perf_counter()
        - run_start
    )

    banner(
        "FINAL T7 TRAINING-RETENTION RESULTS"
    )

    print_summary(
        "OVERALL",
        overall,
    )

    print_summary(
        "SLOT 0",
        slot0,
    )

    print_summary(
        "SLOT 1",
        slot1,
    )

    (
        classification,
        correct_gate_pass,
        margin_gate_pass,
    ) = classify_retention(
        overall
    )

    banner(
        "PRE-REGISTERED RETENTION GATE"
    )

    print(
        f"correct > distractor >= "
        f"{RETENTION_CORRECT_GT_GATE:.2f}: "
        f"{'PASS' if correct_gate_pass else 'FAIL'} "
        f"({overall['correct_gt_distractor']:.6f})"
    )

    print(
        f"fraction(logit delta >= "
        f"{RETENTION_MARGIN_VALUE:.1f}) >= "
        f"{RETENTION_MARGIN_FRACTION_GATE:.2f}: "
        f"{'PASS' if margin_gate_pass else 'FAIL'} "
        f"({overall['margin_ge_0_5']:.6f})"
    )

    print()

    print(
        f"CLASSIFICATION:"
    )

    print(
        f"  {classification}"
    )

    print()

    if classification == (
        "STRONG_FINAL_TRAINING_RETENTION"
    ):
        print(
            "Retention gate PASSED."
        )

        print(
            "NEXT:"
        )

        print(
            "  Run the unchanged T4 positive controls "
            "with the T7 query pathway active."
        )

        print(
            "  Sealed novel/shortcut evaluation remains CLOSED "
            "until positive controls pass."
        )

    else:
        print(
            "Retention gate FAILED."
        )

        print(
            "T7 is therefore a TRAINING-TASK FAILURE "
            "under the pre-registered decision rule."
        )

        print(
            "DO NOT run positive controls."
        )

        print(
            "DO NOT open sealed evaluation."
        )

        print(
            "NEXT:"
        )

        print(
            "  Stop T7 here and consult Grandpa "
            "for Treatment #8."
        )

    result = {
        "audit": (
            "DaveLM v0.9 Treatment #7 "
            "final training-retention audit"
        ),
        "read_only": (
            True
        ),
        "model_mode": (
            "eval"
        ),
        "inference_mode": (
            True
        ),
        "optimizer_created": (
            False
        ),
        "backward_called": (
            False
        ),
        "training_performed": (
            False
        ),
        "positive_controls_evaluated": (
            False
        ),
        "sealed_evaluation_opened": (
            False
        ),
        "frozen_artifacts": {
            "pair_pool_path": (
                str(
                    PAIR_POOL_PATH
                )
            ),
            "pair_pool_sha256": (
                pair_pool_sha
            ),
            "schedule_path": (
                str(
                    SCHEDULE_PATH
                )
            ),
            "schedule_sha256": (
                schedule_sha
            ),
            "final_checkpoint_path": (
                str(
                    FINAL_CHECKPOINT_PATH
                )
            ),
            "final_checkpoint_sha256": (
                checkpoint_sha
            ),
            "trainer_sha256_from_checkpoint": (
                checkpoint.get(
                    "trainer_sha256"
                )
            ),
        },
        "contract": {
            "pair_count": (
                EXPECTED_PAIR_COUNT
            ),
            "steps": (
                EXPECTED_STEPS
            ),
            "batch_size": (
                EXPECTED_BATCH_SIZE
            ),
            "complete_pairs_per_batch": (
                EXPECTED_PAIRS_PER_BATCH
            ),
            "scheduled_examples": (
                EXPECTED_EXAMPLES
            ),
            "slot0_examples": (
                EXPECTED_SLOT0
            ),
            "slot1_examples": (
                EXPECTED_SLOT1
            ),
            "pair_exposure_min": (
                EXPECTED_PAIR_EXPOSURE_MIN
            ),
            "pair_exposure_max": (
                EXPECTED_PAIR_EXPOSURE_MAX
            ),
            "total_supervised_positions": (
                EXPECTED_TOTAL_SUPERVISED_POSITIONS
            ),
            "answer_replacements": (
                EXPECTED_ANSWER_REPLACEMENTS
            ),
            "ordinary_nonanswer_ce_positions": (
                EXPECTED_NONANSWER_CE_POSITIONS
            ),
            "base_parameter_count": (
                EXPECTED_BASE_PARAMETER_COUNT
            ),
            "query_projection_parameter_count": (
                EXPECTED_QUERY_PROJECTION_PARAMETER_COUNT
            ),
            "total_parameter_count": (
                EXPECTED_TOTAL_PARAMETER_COUNT
            ),
            "margin": (
                MARGIN
            ),
        },
        "t7_pathway": {
            "query_source": (
                "actual query token ID extracted from frozen input sequence"
            ),
            "embedding_source": (
                "base_model.token_embedding"
            ),
            "projection": (
                "Linear(320 -> 1024, bias=True)"
            ),
            "application": (
                "base answer logits + query projection bias"
            ),
            "answer_only": (
                True
            ),
        },
        "overall": (
            overall
        ),
        "slot0": (
            slot0
        ),
        "slot1": (
            slot1
        ),
        "retention_gate": {
            "correct_gt_distractor_required": (
                RETENTION_CORRECT_GT_GATE
            ),
            "correct_gt_distractor_observed": (
                overall[
                    "correct_gt_distractor"
                ]
            ),
            "correct_gate_pass": (
                correct_gate_pass
            ),
            "required_raw_margin": (
                RETENTION_MARGIN_VALUE
            ),
            "margin_fraction_required": (
                RETENTION_MARGIN_FRACTION_GATE
            ),
            "margin_fraction_observed": (
                overall[
                    "margin_ge_0_5"
                ]
            ),
            "margin_gate_pass": (
                margin_gate_pass
            ),
        },
        "classification": (
            classification
        ),
        "evaluation_runtime_seconds": (
            float(
                eval_runtime
            )
        ),
        "total_runtime_seconds": (
            float(
                total_runtime
            )
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        OUTPUT_PATH,
        result,
    )

    banner(
        "AUDIT COMPLETE"
    )

    print(
        f"Evaluation runtime: "
        f"{eval_runtime:.2f} seconds"
    )

    print(
        f"Total runtime: "
        f"{total_runtime:.2f} seconds"
    )

    print()

    print(
        "Result JSON:"
    )

    print(
        f"  {OUTPUT_PATH}"
    )

    print()

    print(
        "Optimizer created: NO"
    )

    print(
        "Backward pass: NO"
    )

    print(
        "Training performed: NO"
    )

    print(
        "Positive controls evaluated: NO"
    )

    print(
        "Sealed evaluation opened: NO"
    )


if __name__ == "__main__":
    main()