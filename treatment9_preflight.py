import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn


# =============================================================================
# DAVELM v0.9 — TREATMENT #9 PREFLIGHT
# ANSWER-POSITION CONTENT-ADDRESSABLE RETRIEVAL
# =============================================================================
#
# READ-ONLY / CONSTRUCTION-ONLY PREFLIGHT.
#
# NO optimizer.
# NO backward.
# NO training.
# NO positive controls.
# NO sealed evaluation.
#
# Purpose:
# Freeze and mechanically verify the Treatment #9 contract before training.
#
# T9 hypothesis:
#
#   Baby's remaining reciprocal-binding failure is caused by the absence of a
#   reliable content-addressable retrieval operation at the answer position.
#
# Single architectural intervention:
#
#   Add ONE learned single-head attention retrieval module after Baby's final
#   contextual hidden states.
#
#   Query:
#       final hidden at answer_causal_position
#
#   Keys / values:
#       final hidden states STRICTLY BEFORE answer_causal_position
#
#   Retrieved context:
#       added residually to answer hidden
#
#   Output:
#       Baby's EXISTING language_head(augmented_answer_hidden)
#
# IMPORTANT:
#
#   T8's contextual-query -> vocabulary-logit projection is NOT retained.
#
#   No target IDs, distractor IDs, candidate IDs, query-slot labels,
#   mapping metadata, or mapping-position hints enter the retrieval module.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

CADAVER_ROOT = Path(r"C:\DaveLM-CADAVER")
PROTECTED_ROOT = Path(r"C:\DaveLM-v0.9")

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

START_CHECKPOINT_PATH = (
    PROTECTED_ROOT
    / "experiments"
    / "minimal_contextual_binding"
    / "checkpoints"
    / "treatment_one_mapping"
    / "seed_8380"
    / "latest.pt"
)

T9_ROOT = (
    CADAVER_ROOT
    / "treatment9_answer_position_retrieval_seed8380"
)

PREFLIGHT_RESULT_PATH = (
    T9_ROOT
    / "treatment9_preflight_result.json"
)


# =============================================================================
# FROZEN HASHES
# =============================================================================

EXPECTED_PAIR_POOL_SHA256 = (
    "0b31e39361f7a45dc4182c49ab4b4967bffe539cd2ff12e7b196eabfb3dac307"
)

EXPECTED_SCHEDULE_SHA256 = (
    "a4e64fd9d1b934aa440a5d036a3bec0ac7c9ac97d58c72fb45df6d8576bb7c12"
)

EXPECTED_START_CHECKPOINT_SHA256 = (
    "345984c52a06db5f988aaf4cd47963eea0e9d77489cee94dbb10816af2f5443e"
)


# =============================================================================
# FROZEN TRAINING CONTRACT
# =============================================================================

TRAIN_SEED = 8380
SCHEDULE_SEED = 8382

MAX_STEPS = 1000
BATCH_SIZE = 32
PAIRS_PER_BATCH = 16

EXPECTED_PAIR_COUNT = 1536
EXPECTED_SCHEDULED_EXAMPLES = 32000
EXPECTED_SLOT0_PRESENTATIONS = 16000
EXPECTED_SLOT1_PRESENTATIONS = 16000

EXPECTED_PAIR_PRESENTATIONS = 16000
EXPECTED_PAIR_EXPOSURE_MIN = 10
EXPECTED_PAIR_EXPOSURE_MAX = 11

MODEL_VISIBLE_DOCUMENT_LENGTH = 193
CAUSAL_SEQUENCE_LENGTH = 192

EXPECTED_TOTAL_SUPERVISED_TOKENS = 6_144_000
EXPECTED_ANSWER_SUBSTITUTIONS = 32_000
EXPECTED_NONANSWER_SUPERVISED_TOKENS = 6_112_000

VOCAB_SIZE = 1024
HIDDEN_SIZE = 320

BASE_PARAMETER_COUNT = 10_594_944

ANSWER_WEIGHT = 191.0
SELECTOR_MARGIN = 0.5

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.05
GRAD_CLIP = 2.0

RETENTION_CORRECT_GATE = 0.95
RETENTION_MARGIN_GATE = 0.90


# =============================================================================
# T9 ARCHITECTURAL CONTRACT
# =============================================================================

RETRIEVAL_EMBED_DIM = 320
RETRIEVAL_NUM_HEADS = 1
RETRIEVAL_BATCH_FIRST = True
RETRIEVAL_BIAS = True

T9_ARCHITECTURE_NAME = "answer_position_content_addressable_retrieval"

T9_QUERY_SOURCE = (
    "base_model.final_norm output at answer_causal_position"
)

T9_KEY_VALUE_SOURCE = (
    "base_model.final_norm outputs strictly before answer_causal_position"
)

T9_RETRIEVAL_OPERATION = (
    "single learned nn.MultiheadAttention("
    "embed_dim=320, num_heads=1, batch_first=True, bias=True)"
)

T9_RESIDUAL_OPERATION = (
    "augmented_answer_hidden = base_answer_hidden + retrieved_context"
)

T9_OUTPUT_OPERATION = (
    "adjusted_answer_logits = base_model.language_head(augmented_answer_hidden)"
)


# =============================================================================
# UTILITIES
# =============================================================================

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def heading(text):
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
        )


def parameter_count(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


# =============================================================================
# IMPORT NATIVE BABY
# =============================================================================

def import_original_run():
    protected_root_string = str(PROTECTED_ROOT)

    if protected_root_string not in sys.path:
        sys.path.insert(
            0,
            protected_root_string,
        )

    import experiments.two_mapping_contextual_binding.run as original_run

    return original_run


# =============================================================================
# T9 MODEL WRAPPER
# =============================================================================

class Treatment9RetrievalModel(nn.Module):
    """
    Native Baby + one answer-position content-addressable retrieval module.

    Native Baby forward remains untouched.

    A forward hook captures base_model.final_norm output:

        hidden: [B, T, 320]

    At answer positions:

        query = hidden[row, answer_position, :]

    Retrieval memory:

        hidden[row, positions < answer_position, :]

    One learned single-head MultiheadAttention performs retrieval.

    Retrieved context is added residually:

        augmented = answer_hidden + retrieved

    Baby's EXISTING language_head produces answer logits:

        logits = language_head(augmented)

    This module receives no task metadata.
    """

    def __init__(self, base_model):
        super().__init__()

        self.base_model = base_model

        self.retrieval = nn.MultiheadAttention(
            embed_dim=RETRIEVAL_EMBED_DIM,
            num_heads=RETRIEVAL_NUM_HEADS,
            bias=RETRIEVAL_BIAS,
            batch_first=RETRIEVAL_BATCH_FIRST,
        )

        self._captured_final_hidden = None

        self._hook_handle = (
            self.base_model.final_norm.register_forward_hook(
                self._capture_final_hidden
            )
        )

    def _capture_final_hidden(
        self,
        module,
        inputs,
        output,
    ):
        self._captured_final_hidden = output

    def forward(self, input_ids):
        self._captured_final_hidden = None

        base_logits = self.base_model(input_ids)

        require(
            self._captured_final_hidden is not None,
            "final_norm hook captured no hidden states.",
        )

        return base_logits

    def answer_retrieval(
        self,
        answer_positions,
    ):
        hidden = self._captured_final_hidden

        require(
            hidden is not None,
            "No final hidden state has been captured.",
        )

        require(
            hidden.ndim == 3,
            f"Expected hidden [B,T,D], got {tuple(hidden.shape)}.",
        )

        batch_size, seq_len, hidden_size = hidden.shape

        require(
            hidden_size == HIDDEN_SIZE,
            (
                f"Expected hidden size {HIDDEN_SIZE}, "
                f"got {hidden_size}."
            ),
        )

        require(
            answer_positions.shape == (batch_size,),
            (
                f"Expected answer_positions shape {(batch_size,)}, "
                f"got {tuple(answer_positions.shape)}."
            ),
        )

        rows = torch.arange(
            batch_size,
            device=hidden.device,
        )

        answer_hidden = hidden[
            rows,
            answer_positions,
            :
        ]

        # Query shape required by batch_first MHA:
        # [B, 1, D]
        retrieval_query = answer_hidden.unsqueeze(1)

        # Keys and values are ALL final hidden positions.
        #
        # key_padding_mask then hides:
        #   position >= answer_causal_position
        #
        # Therefore each example can retrieve ONLY from strictly earlier tokens.
        memory = hidden

        positions = torch.arange(
            seq_len,
            device=hidden.device,
        ).unsqueeze(0)

        key_padding_mask = (
            positions
            >= answer_positions.unsqueeze(1)
        )

        # Safety:
        # every answer position must have at least one earlier visible token.
        visible_counts = (
            (~key_padding_mask)
            .sum(dim=1)
        )

        require(
            bool((visible_counts > 0).all().item()),
            "At least one example has no legal retrieval-memory positions.",
        )

        retrieved_context, attention_weights = self.retrieval(
            query=retrieval_query,
            key=memory,
            value=memory,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        # [B,1,D] -> [B,D]
        retrieved_context = retrieved_context[:, 0, :]

        augmented_answer_hidden = (
            answer_hidden
            + retrieved_context
        )

        adjusted_answer_logits = (
            self.base_model.language_head(
                augmented_answer_hidden
            )
        )

        return {
            "answer_hidden": answer_hidden,
            "retrieved_context": retrieved_context,
            "augmented_answer_hidden": augmented_answer_hidden,
            "adjusted_answer_logits": adjusted_answer_logits,
            "attention_weights": attention_weights,
            "key_padding_mask": key_padding_mask,
        }


# =============================================================================
# PAIR POOL VERIFICATION
# =============================================================================

def verify_pair_pool(pair_pool):
    require(
        isinstance(pair_pool, dict),
        "Pair pool must be a dict.",
    )

    require(
        "pairs" in pair_pool,
        "Pair pool missing 'pairs'.",
    )

    pairs = pair_pool["pairs"]

    require(
        isinstance(pairs, list),
        "pair_pool['pairs'] must be a list.",
    )

    require(
        len(pairs) == EXPECTED_PAIR_COUNT,
        (
            f"Expected {EXPECTED_PAIR_COUNT} pairs, "
            f"found {len(pairs)}."
        ),
    )

    if "pair_count" in pair_pool:
        require(
            int(pair_pool["pair_count"]) == EXPECTED_PAIR_COUNT,
            "pair_count mismatch.",
        )

    pair_ids = set()

    for pair_index, pair in enumerate(pairs):
        require(
            isinstance(pair, dict),
            f"Pair {pair_index} is not a dict.",
        )

        required_fields = (
            "pair_id",
            "candidate_pair",
            "query_difference_position",
            "answer_causal_position",
            "answer_token_index",
            "twin_a",
            "twin_b",
        )

        for field in required_fields:
            require(
                field in pair,
                f"Pair {pair_index} missing {field!r}.",
            )

        pair_id = pair["pair_id"]

        require(
            pair_id not in pair_ids,
            f"Duplicate pair ID: {pair_id}",
        )

        pair_ids.add(pair_id)

        twin_a = pair["twin_a"]
        twin_b = pair["twin_b"]

        require(
            int(twin_a["query_slot"]) == 0,
            f"{pair_id}: twin A slot != 0.",
        )

        require(
            int(twin_b["query_slot"]) == 1,
            f"{pair_id}: twin B slot != 1.",
        )

        require(
            int(twin_a["target_token_id"])
            == int(twin_b["distractor_token_id"]),
            f"{pair_id}: A target != B distractor.",
        )

        require(
            int(twin_b["target_token_id"])
            == int(twin_a["distractor_token_id"]),
            f"{pair_id}: B target != A distractor.",
        )

        require(
            int(pair["answer_token_index"])
            == int(pair["answer_causal_position"]) + 1,
            f"{pair_id}: answer index contract failed.",
        )

        for twin_name, twin in (
            ("A", twin_a),
            ("B", twin_b),
        ):
            document = twin["full_document_token_ids"]

            require(
                len(document) == MODEL_VISIBLE_DOCUMENT_LENGTH,
                (
                    f"{pair_id}/{twin_name}: "
                    f"document length {len(document)} "
                    f"!= {MODEL_VISIBLE_DOCUMENT_LENGTH}."
                ),
            )

            answer_position = int(
                pair["answer_causal_position"]
            )

            target_id = int(
                twin["target_token_id"]
            )

            require(
                int(document[answer_position + 1])
                == target_id,
                (
                    f"{pair_id}/{twin_name}: "
                    "causal answer label != target."
                ),
            )

    return pairs


# =============================================================================
# SCHEDULE VERIFICATION
# =============================================================================

def verify_schedule(schedule):
    require(
        isinstance(schedule, dict),
        "Frozen schedule must be a dict.",
    )

    exact_fields = {
        "steps": MAX_STEPS,
        "batch_size": BATCH_SIZE,
        "pairs_per_batch": PAIRS_PER_BATCH,
        "schedule_seed": SCHEDULE_SEED,
        "scheduled_examples": EXPECTED_SCHEDULED_EXAMPLES,
        "slot_0_presentations": EXPECTED_SLOT0_PRESENTATIONS,
        "slot_1_presentations": EXPECTED_SLOT1_PRESENTATIONS,
        "pair_presentations": EXPECTED_PAIR_PRESENTATIONS,
        "pair_exposure_min": EXPECTED_PAIR_EXPOSURE_MIN,
        "pair_exposure_max": EXPECTED_PAIR_EXPOSURE_MAX,
        "total_supervised_tokens": EXPECTED_TOTAL_SUPERVISED_TOKENS,
        "answer_substitutions": EXPECTED_ANSWER_SUBSTITUTIONS,
        "nonanswer_supervised_tokens": EXPECTED_NONANSWER_SUPERVISED_TOKENS,
    }

    for field, expected in exact_fields.items():
        require(
            field in schedule,
            f"Schedule missing {field!r}.",
        )

        observed = int(schedule[field])

        require(
            observed == expected,
            (
                f"Schedule {field}: "
                f"{observed} != {expected}."
            ),
        )

    require(
        "steps_data" in schedule,
        "Schedule missing 'steps_data'.",
    )

    steps_data = schedule["steps_data"]

    require(
        isinstance(steps_data, list),
        "steps_data must be a list.",
    )

    require(
        len(steps_data) == MAX_STEPS,
        (
            f"steps_data length {len(steps_data)} "
            f"!= {MAX_STEPS}."
        ),
    )

    event_count = 0
    slot0_count = 0
    slot1_count = 0

    observed_query_positions = set()
    observed_answer_positions = set()

    for list_index, step in enumerate(steps_data):
        expected_step = list_index + 1

        require(
            int(step["step"]) == expected_step,
            (
                f"Expected step {expected_step}, "
                f"found {step['step']}."
            ),
        )

        require(
            int(step["example_count"]) == BATCH_SIZE,
            f"Step {expected_step}: example_count mismatch.",
        )

        require(
            int(step["pair_count"]) == PAIRS_PER_BATCH,
            f"Step {expected_step}: pair_count mismatch.",
        )

        examples = step["examples"]

        require(
            len(examples) == BATCH_SIZE,
            f"Step {expected_step}: examples length mismatch.",
        )

        per_step_pairs = {}

        for event in examples:
            event_count += 1

            slot = int(event["query_slot"])

            require(
                slot in (0, 1),
                f"Invalid query slot {slot}.",
            )

            if slot == 0:
                slot0_count += 1
            else:
                slot1_count += 1

            pair_id = event["pair_id"]

            per_step_pairs.setdefault(
                pair_id,
                set(),
            )

            require(
                slot not in per_step_pairs[pair_id],
                (
                    f"Step {expected_step}: "
                    f"duplicate slot {slot} for {pair_id}."
                ),
            )

            per_step_pairs[pair_id].add(slot)

            document = event["full_document_token_ids"]

            require(
                len(document) == MODEL_VISIBLE_DOCUMENT_LENGTH,
                (
                    f"{pair_id}: document length "
                    f"{len(document)} != "
                    f"{MODEL_VISIBLE_DOCUMENT_LENGTH}."
                ),
            )

            require(
                int(event["supervised_token_count"])
                == CAUSAL_SEQUENCE_LENGTH,
                f"{pair_id}: supervised count mismatch.",
            )

            require(
                int(event["nonanswer_supervised_token_count"])
                == CAUSAL_SEQUENCE_LENGTH - 1,
                f"{pair_id}: nonanswer count mismatch.",
            )

            query_position = int(
                event["query_difference_position"]
            )

            answer_position = int(
                event["answer_causal_position"]
            )

            answer_token_index = int(
                event["answer_token_index"]
            )

            target_id = int(
                event["target_token_id"]
            )

            distractor_id = int(
                event["distractor_token_id"]
            )

            observed_query_positions.add(
                query_position
            )

            observed_answer_positions.add(
                answer_position
            )

            require(
                0 <= query_position < answer_position,
                (
                    f"{pair_id}: query position "
                    f"{query_position} is not before "
                    f"answer causal position {answer_position}."
                ),
            )

            require(
                1 <= answer_position < CAUSAL_SEQUENCE_LENGTH,
                (
                    f"{pair_id}: invalid answer position "
                    f"{answer_position}."
                ),
            )

            require(
                answer_token_index == answer_position + 1,
                f"{pair_id}: answer index mismatch.",
            )

            require(
                target_id != distractor_id,
                f"{pair_id}: target equals distractor.",
            )

            require(
                int(document[answer_position + 1])
                == target_id,
                (
                    f"{pair_id}: target does not equal "
                    "causal answer label."
                ),
            )

        require(
            len(per_step_pairs) == PAIRS_PER_BATCH,
            (
                f"Step {expected_step}: expected "
                f"{PAIRS_PER_BATCH} unique pairs, "
                f"found {len(per_step_pairs)}."
            ),
        )

        for pair_id, slots in per_step_pairs.items():
            require(
                slots == {0, 1},
                (
                    f"Step {expected_step}: "
                    f"{pair_id} slots = {slots}, expected {{0,1}}."
                ),
            )

    require(
        event_count == EXPECTED_SCHEDULED_EXAMPLES,
        (
            f"Event count {event_count} "
            f"!= {EXPECTED_SCHEDULED_EXAMPLES}."
        ),
    )

    require(
        slot0_count == EXPECTED_SLOT0_PRESENTATIONS,
        (
            f"Slot0 count {slot0_count} "
            f"!= {EXPECTED_SLOT0_PRESENTATIONS}."
        ),
    )

    require(
        slot1_count == EXPECTED_SLOT1_PRESENTATIONS,
        (
            f"Slot1 count {slot1_count} "
            f"!= {EXPECTED_SLOT1_PRESENTATIONS}."
        ),
    )

    return {
        "event_count": event_count,
        "slot0_count": slot0_count,
        "slot1_count": slot1_count,
        "query_positions": sorted(observed_query_positions),
        "answer_positions": sorted(observed_answer_positions),
    }


# =============================================================================
# START CHECKPOINT STATE EXTRACTION
# =============================================================================

def extract_model_state(checkpoint):
    require(
        isinstance(checkpoint, dict),
        "Start checkpoint is not a dictionary.",
    )

    possible_keys = (
        "model_state",
        "model_state_dict",
        "state_dict",
    )

    for key in possible_keys:
        value = checkpoint.get(key)

        if isinstance(value, dict):
            return value, key

    # Some checkpoint formats ARE the state dict directly.
    if checkpoint and all(
        isinstance(k, str)
        for k in checkpoint.keys()
    ):
        tensor_values = sum(
            1
            for value in checkpoint.values()
            if torch.is_tensor(value)
        )

        if tensor_values > 0:
            return checkpoint, "<checkpoint-is-state-dict>"

    raise RuntimeError(
        "Could not locate model state in start checkpoint."
    )


# =============================================================================
# MAIN PREFLIGHT
# =============================================================================

def main():
    heading(
        "DAVELM v0.9 — TREATMENT #9 PREFLIGHT"
    )

    print(
        "ANSWER-POSITION CONTENT-ADDRESSABLE RETRIEVAL"
    )

    print()

    print("Training performed:          NO")
    print("Optimizer created:           NO")
    print("Backward performed:          NO")
    print("Positive controls opened:    NO")
    print("Sealed evaluation opened:    NO")

    # -------------------------------------------------------------------------
    # REQUIRED ARTIFACTS
    # -------------------------------------------------------------------------

    heading(
        "VERIFYING REQUIRED FROZEN ARTIFACTS"
    )

    for path in (
        PAIR_POOL_PATH,
        SCHEDULE_PATH,
        START_CHECKPOINT_PATH,
    ):
        require(
            path.exists(),
            f"Missing required artifact: {path}",
        )

        print(f"PASS: {path}")

    # -------------------------------------------------------------------------
    # HASHES
    # -------------------------------------------------------------------------

    heading(
        "VERIFYING FROZEN HASHES"
    )

    pair_sha = sha256_file(
        PAIR_POOL_PATH
    )

    schedule_sha = sha256_file(
        SCHEDULE_PATH
    )

    start_sha = sha256_file(
        START_CHECKPOINT_PATH
    )

    print(
        f"Pair pool:\n"
        f"  observed {pair_sha}\n"
        f"  expected {EXPECTED_PAIR_POOL_SHA256}"
    )

    require(
        pair_sha == EXPECTED_PAIR_POOL_SHA256,
        "Pair-pool SHA mismatch.",
    )

    print("  PASS")
    print()

    print(
        f"Schedule:\n"
        f"  observed {schedule_sha}\n"
        f"  expected {EXPECTED_SCHEDULE_SHA256}"
    )

    require(
        schedule_sha == EXPECTED_SCHEDULE_SHA256,
        "Schedule SHA mismatch.",
    )

    print("  PASS")
    print()

    print(
        f"Start checkpoint:\n"
        f"  observed {start_sha}\n"
        f"  expected {EXPECTED_START_CHECKPOINT_SHA256}"
    )

    require(
        start_sha == EXPECTED_START_CHECKPOINT_SHA256,
        "Start-checkpoint SHA mismatch.",
    )

    print("  PASS")

    # -------------------------------------------------------------------------
    # FROZEN DATA CONTRACT
    # -------------------------------------------------------------------------

    heading(
        "VERIFYING FROZEN DATA / SCHEDULE CONTRACT"
    )

    pair_pool = load_json(
        PAIR_POOL_PATH
    )

    schedule = load_json(
        SCHEDULE_PATH
    )

    pairs = verify_pair_pool(
        pair_pool
    )

    schedule_summary = verify_schedule(
        schedule
    )

    print(
        f"Unique pairs:                  {len(pairs)}"
    )

    print(
        f"Frozen steps:                  {MAX_STEPS}"
    )

    print(
        f"Batch size:                    {BATCH_SIZE}"
    )

    print(
        f"Complete pairs/batch:          {PAIRS_PER_BATCH}"
    )

    print(
        f"Scheduled examples:            "
        f"{schedule_summary['event_count']}"
    )

    print(
        f"Slot0 presentations:           "
        f"{schedule_summary['slot0_count']}"
    )

    print(
        f"Slot1 presentations:           "
        f"{schedule_summary['slot1_count']}"
    )

    print(
        f"Query causal positions:        "
        f"{schedule_summary['query_positions']}"
    )

    print(
        f"Answer causal positions:       "
        f"{schedule_summary['answer_positions']}"
    )

    print(
        f"Pair exposure:                 "
        f"{EXPECTED_PAIR_EXPOSURE_MIN}-"
        f"{EXPECTED_PAIR_EXPOSURE_MAX}"
    )

    print(
        f"Supervised positions:          "
        f"{EXPECTED_TOTAL_SUPERVISED_TOKENS}"
    )

    print(
        f"Answer replacements:           "
        f"{EXPECTED_ANSWER_SUBSTITUTIONS}"
    )

    print(
        f"Ordinary non-answer positions: "
        f"{EXPECTED_NONANSWER_SUPERVISED_TOKENS}"
    )

    print()
    print("Frozen schedule contract: PASS")

    # -------------------------------------------------------------------------
    # BUILD / LOAD NATIVE BABY
    # -------------------------------------------------------------------------

    heading(
        "BUILDING EXACT NATIVE BABY"
    )

    original_run = import_original_run()

    torch.manual_seed(
        TRAIN_SEED
    )

    base_model = original_run.build_model(
        "untied"
    )

    observed_base_params = parameter_count(
        base_model
    )

    require(
        observed_base_params == BASE_PARAMETER_COUNT,
        (
            f"Base parameter count "
            f"{observed_base_params:,} != "
            f"{BASE_PARAMETER_COUNT:,}."
        ),
    )

    start_checkpoint = safe_torch_load(
        START_CHECKPOINT_PATH,
        map_location="cpu",
    )

    model_state, state_key = extract_model_state(
        start_checkpoint
    )

    base_model.load_state_dict(
        model_state,
        strict=True,
    )

    print(
        f"Base params:                    "
        f"{observed_base_params:,}"
    )

    print(
        f"Start checkpoint state source:  "
        f"{state_key}"
    )

    print(
        "Strict native start load:       PASS"
    )

    # -------------------------------------------------------------------------
    # CONSTRUCT T9 RETRIEVAL MODULE
    # -------------------------------------------------------------------------

    heading(
        "CONSTRUCTING FROZEN T9 ARCHITECTURE"
    )

    # Seed is intentionally left at the deterministic stream established
    # above: build native Baby -> load exact starting state -> construct T9.
    #
    # This mirrors the treatment-family pattern where the newly introduced
    # module receives deterministic standard PyTorch initialization.

    model = Treatment9RetrievalModel(
        base_model
    )

    retrieval_params = parameter_count(
        model.retrieval
    )

    total_params = parameter_count(
        model
    )

    # nn.MultiheadAttention(320, 1, bias=True)
    #
    # in_proj_weight: 3*320*320 = 307,200
    # in_proj_bias:   3*320     =     960
    # out_proj.weight: 320*320  = 102,400
    # out_proj.bias:   320      =     320
    #
    # total = 410,880
    expected_retrieval_params = (
        3 * HIDDEN_SIZE * HIDDEN_SIZE
        + 3 * HIDDEN_SIZE
        + HIDDEN_SIZE * HIDDEN_SIZE
        + HIDDEN_SIZE
    )

    expected_total_params = (
        BASE_PARAMETER_COUNT
        + expected_retrieval_params
    )

    require(
        retrieval_params == expected_retrieval_params,
        (
            f"Retrieval params {retrieval_params:,} "
            f"!= expected {expected_retrieval_params:,}."
        ),
    )

    require(
        total_params == expected_total_params,
        (
            f"T9 total params {total_params:,} "
            f"!= expected {expected_total_params:,}."
        ),
    )

    print(
        f"Base parameters:                "
        f"{BASE_PARAMETER_COUNT:,}"
    )

    print(
        f"Retrieval parameters:           "
        f"{retrieval_params:,}"
    )

    print(
        f"Total T9 parameters:            "
        f"{total_params:,}"
    )

    print()

    print(
        "Retrieval module:"
    )

    print(
        "  nn.MultiheadAttention("
        "embed_dim=320, num_heads=1, "
        "batch_first=True, bias=True)"
    )

    print()

    print(
        "Query:"
    )

    print(
        "  final contextual hidden at "
        "answer_causal_position"
    )

    print()

    print(
        "Keys / values:"
    )

    print(
        "  final contextual hiddens at "
        "positions STRICTLY BEFORE "
        "answer_causal_position"
    )

    print()

    print(
        "Residual:"
    )

    print(
        "  augmented_answer_hidden = "
        "answer_hidden + retrieved_context"
    )

    print()

    print(
        "Output:"
    )

    print(
        "  base_model.language_head("
        "augmented_answer_hidden)"
    )

    print()

    print(
        "T8 direct query->logit projection retained: NO"
    )

    # -------------------------------------------------------------------------
    # VERIFY MODULE HAS NO EXTERNAL TASK INPUTS
    # -------------------------------------------------------------------------

    heading(
        "VERIFYING NO TASK-STRUCTURE LEAKAGE"
    )

    print(
        "Retrieval forward inputs:"
    )

    print(
        "  - native final hidden states"
    )

    print(
        "  - answer causal position ONLY for causal masking/extraction"
    )

    print()

    print(
        "Explicitly NOT supplied:"
    )

    forbidden_inputs = (
        "target_token_id",
        "distractor_token_id",
        "candidate_pair",
        "query_slot",
        "pair_id",
        "mapping_0",
        "mapping_1",
        "mapping key positions",
        "mapping value positions",
        "query_difference_position",
    )

    for item in forbidden_inputs:
        print(f"  - {item}")

    print()

    print(
        "NOTE:"
    )

    print(
        "answer_causal_position is already part of the frozen "
        "supervision geometry and is required to identify the "
        "single position whose answer objective is replaced."
    )

    print(
        "It does NOT identify which contextual token is correct."
    )

    print()

    print("No target/candidate metadata leakage: PASS")

    # -------------------------------------------------------------------------
    # DRY FORWARD ON ONE REAL FROZEN BATCH
    # -------------------------------------------------------------------------

    heading(
        "DRY FORWARD — ONE REAL FROZEN BATCH"
    )

    first_step = schedule["steps_data"][0]
    examples = first_step["examples"]

    documents = torch.tensor(
        [
            event["full_document_token_ids"]
            for event in examples
        ],
        dtype=torch.long,
    )

    require(
        documents.shape
        == (
            BATCH_SIZE,
            MODEL_VISIBLE_DOCUMENT_LENGTH,
        ),
        (
            f"Unexpected document tensor shape "
            f"{tuple(documents.shape)}."
        ),
    )

    inputs = documents[:, :-1]
    labels = documents[:, 1:]

    answer_positions = torch.tensor(
        [
            int(event["answer_causal_position"])
            for event in examples
        ],
        dtype=torch.long,
    )

    target_ids = torch.tensor(
        [
            int(event["target_token_id"])
            for event in examples
        ],
        dtype=torch.long,
    )

    rows = torch.arange(
        BATCH_SIZE
    )

    require(
        torch.equal(
            labels[
                rows,
                answer_positions,
            ],
            target_ids,
        ),
        "Dry-forward target-label alignment failed.",
    )

    model.eval()

    with torch.inference_mode():
        base_logits = model(
            inputs
        )

        retrieval_output = model.answer_retrieval(
            answer_positions
        )

    require(
        base_logits.shape
        == (
            BATCH_SIZE,
            CAUSAL_SEQUENCE_LENGTH,
            VOCAB_SIZE,
        ),
        (
            f"Unexpected base-logit shape "
            f"{tuple(base_logits.shape)}."
        ),
    )

    require(
        retrieval_output["answer_hidden"].shape
        == (
            BATCH_SIZE,
            HIDDEN_SIZE,
        ),
        "answer_hidden shape mismatch.",
    )

    require(
        retrieval_output["retrieved_context"].shape
        == (
            BATCH_SIZE,
            HIDDEN_SIZE,
        ),
        "retrieved_context shape mismatch.",
    )

    require(
        retrieval_output["augmented_answer_hidden"].shape
        == (
            BATCH_SIZE,
            HIDDEN_SIZE,
        ),
        "augmented answer hidden shape mismatch.",
    )

    require(
        retrieval_output["adjusted_answer_logits"].shape
        == (
            BATCH_SIZE,
            VOCAB_SIZE,
        ),
        "adjusted answer logits shape mismatch.",
    )

    attention_weights = retrieval_output[
        "attention_weights"
    ]

    require(
        attention_weights.shape
        == (
            BATCH_SIZE,
            1,
            CAUSAL_SEQUENCE_LENGTH,
        ),
        (
            f"Unexpected attention-weight shape "
            f"{tuple(attention_weights.shape)}."
        ),
    )

    key_padding_mask = retrieval_output[
        "key_padding_mask"
    ]

    # -------------------------------------------------------------------------
    # HARD CAUSAL MASK VERIFICATION
    # -------------------------------------------------------------------------

    for row in range(BATCH_SIZE):
        answer_position = int(
            answer_positions[row].item()
        )

        # Everything at answer_position or later must be masked.
        require(
            bool(
                key_padding_mask[
                    row,
                    answer_position:
                ].all().item()
            ),
            (
                f"Row {row}: retrieval memory leaks "
                "answer/current/future positions."
            ),
        )

        # Everything before answer_position must be visible.
        require(
            bool(
                (
                    ~key_padding_mask[
                        row,
                        :answer_position
                    ]
                ).all().item()
            ),
            (
                f"Row {row}: a legal previous position "
                "was incorrectly masked."
            ),
        )

        # Masked attention probability should be numerically zero.
        masked_weight_mass = float(
            attention_weights[
                row,
                0,
                answer_position:
            ].sum().item()
        )

        require(
            abs(masked_weight_mass) < 1e-7,
            (
                f"Row {row}: masked attention mass "
                f"{masked_weight_mass} is not zero."
            ),
        )

        visible_weight_mass = float(
            attention_weights[
                row,
                0,
                :answer_position
            ].sum().item()
        )

        require(
            math.isclose(
                visible_weight_mass,
                1.0,
                rel_tol=0.0,
                abs_tol=1e-5,
            ),
            (
                f"Row {row}: visible attention mass "
                f"{visible_weight_mass} != 1."
            ),
        )

    print(
        f"Input shape:                    "
        f"{tuple(inputs.shape)}"
    )

    print(
        f"Native logit shape:             "
        f"{tuple(base_logits.shape)}"
    )

    print(
        f"Answer hidden shape:            "
        f"{tuple(retrieval_output['answer_hidden'].shape)}"
    )

    print(
        f"Retrieved context shape:        "
        f"{tuple(retrieval_output['retrieved_context'].shape)}"
    )

    print(
        f"Adjusted answer-logit shape:    "
        f"{tuple(retrieval_output['adjusted_answer_logits'].shape)}"
    )

    print()

    print(
        "Answer/current/future retrieval masking: PASS"
    )

    print(
        "Strictly-prior retrieval memory only:    PASS"
    )

    print(
        "Normal native language_head reused:      PASS"
    )

    # -------------------------------------------------------------------------
    # VERIFY NON-ANSWER PATH REMAINS NATIVE
    # -------------------------------------------------------------------------

    heading(
        "VERIFYING NON-ANSWER PATH REMAINS UNTOUCHED"
    )

    print(
        "Ordinary non-answer CE will use:"
    )

    print(
        "  base_model(input_ids) native logits"
    )

    print()

    print(
        "Retrieval-adjusted logits exist ONLY for:"
    )

    print(
        "  one answer_causal_position per example"
    )

    print()

    print(
        "Native transformer forward rewritten: NO"
    )

    print(
        "Native non-answer logits modified:     NO"
    )

    print(
        "T8 query-logit bias retained:          NO"
    )

    print()

    print(
        "Non-answer/native-output isolation: PASS"
    )

    # -------------------------------------------------------------------------
    # FREEZE CONTRACT
    # -------------------------------------------------------------------------

    heading(
        "FROZEN T9 CONTRACT"
    )

    contract = {
        "treatment": 9,
        "treatment_name": T9_ARCHITECTURE_NAME,

        "hypothesis": (
            "The remaining reciprocal-binding failure is caused "
            "by the absence of a reliable content-addressable "
            "retrieval operation at the answer position."
        ),

        "single_architectural_intervention": (
            "One learned single-head answer-position retrieval "
            "module operating over strictly prior final contextual "
            "hidden states."
        ),

        "t8_query_logit_projection_retained": False,

        "frozen_artifacts": {
            "pair_pool_path": str(PAIR_POOL_PATH),
            "pair_pool_sha256": pair_sha,

            "schedule_path": str(SCHEDULE_PATH),
            "schedule_sha256": schedule_sha,

            "start_checkpoint_path": str(
                START_CHECKPOINT_PATH
            ),
            "start_checkpoint_sha256": start_sha,
        },

        "training": {
            "train_seed": TRAIN_SEED,
            "schedule_seed": SCHEDULE_SEED,
            "steps": MAX_STEPS,
            "batch_size": BATCH_SIZE,
            "pairs_per_batch": PAIRS_PER_BATCH,
            "scheduled_examples": EXPECTED_SCHEDULED_EXAMPLES,
            "slot0_presentations": EXPECTED_SLOT0_PRESENTATIONS,
            "slot1_presentations": EXPECTED_SLOT1_PRESENTATIONS,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "answer_weight_lambda": ANSWER_WEIGHT,
            "selector_margin": SELECTOR_MARGIN,
        },

        "architecture": {
            "base_parameter_count": BASE_PARAMETER_COUNT,
            "retrieval_parameter_count": retrieval_params,
            "total_parameter_count": total_params,

            "retrieval_module": T9_RETRIEVAL_OPERATION,
            "query_source": T9_QUERY_SOURCE,
            "key_value_source": T9_KEY_VALUE_SOURCE,
            "residual_operation": T9_RESIDUAL_OPERATION,
            "output_operation": T9_OUTPUT_OPERATION,

            "retrieval_active_only_at_answer_positions": True,
            "retrieval_memory_strictly_before_answer_position": True,
            "normal_language_head_reused": True,
            "base_transformer_forward_rewritten": False,
            "native_nonanswer_logits_modified": False,
        },

        "forbidden_retrieval_inputs": [
            "target_token_id",
            "distractor_token_id",
            "candidate_pair",
            "query_slot",
            "pair_id",
            "mapping metadata",
            "mapping key positions",
            "mapping value positions",
            "query_difference_position",
        ],

        "objective": {
            "ordinary_nonanswer_objective": (
                "native full-vocabulary cross entropy"
            ),

            "answer_membership_loss": (
                "logsumexp(full_vocab_logits) - "
                "logsumexp([correct_logit, distractor_logit])"
            ),

            "answer_selector_loss": (
                "relu(0.5 - "
                "(correct_logit - distractor_logit))"
            ),

            "answer_replacement": (
                "membership_loss + selector_loss"
            ),

            "answer_weight_lambda": ANSWER_WEIGHT,

            "reduction": (
                "ordinary CE table over all 192 causal positions; "
                "replace answer-position CE with "
                "191*(membership+selector); mean over Bx192"
            ),
        },

        "final_retention_gate": {
            "exact_scheduled_events": (
                EXPECTED_SCHEDULED_EXAMPLES
            ),

            "correct_over_distractor_min": (
                RETENTION_CORRECT_GATE
            ),

            "fraction_raw_delta_ge_0_5_min": (
                RETENTION_MARGIN_GATE
            ),

            "pass_classification": (
                "STRONG_FINAL_TRAINING_RETENTION"
            ),

            "fail_classification": (
                "FINAL_TRAINING_RETENTION_BELOW_GATE"
            ),
        },

        "evaluation_sequence": {
            "first": (
                "exact all-32000 final training retention audit"
            ),

            "if_retention_fails": (
                "STOP; no positive controls; sealed evaluation closed"
            ),

            "if_retention_passes": (
                "run unchanged positive controls"
            ),

            "sealed_evaluation_condition": (
                "positive controls must pass first"
            ),
        },

        "preflight": {
            "optimizer_created": False,
            "backward_performed": False,
            "training_performed": False,
            "positive_controls_opened": False,
            "sealed_evaluation_opened": False,
            "dry_forward_only": True,
        },
    }

    T9_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        PREFLIGHT_RESULT_PATH,
        contract,
    )

    # -------------------------------------------------------------------------
    # FINAL REPORT
    # -------------------------------------------------------------------------

    print(
        f"Treatment:                      #9"
    )

    print(
        f"Name:                           "
        f"{T9_ARCHITECTURE_NAME}"
    )

    print()

    print(
        f"Start checkpoint SHA:           "
        f"{start_sha}"
    )

    print(
        f"Pair pool SHA:                  "
        f"{pair_sha}"
    )

    print(
        f"Schedule SHA:                   "
        f"{schedule_sha}"
    )

    print()

    print(
        f"Base parameters:                "
        f"{BASE_PARAMETER_COUNT:,}"
    )

    print(
        f"Retrieval parameters:           "
        f"{retrieval_params:,}"
    )

    print(
        f"Total parameters:               "
        f"{total_params:,}"
    )

    print()

    print(
        "Only substantive architectural addition:"
    )

    print(
        "  answer-position content-addressable "
        "single-head retrieval"
    )

    print()

    print(
        "T8 direct query->logit branch: REMOVED"
    )

    print()

    print(
        "Retrieval target/candidate metadata: NONE"
    )

    print(
        "Mapping-position hints:                NONE"
    )

    print(
        "Query-slot hints:                      NONE"
    )

    print()

    print(
        "Retrieval memory:"
    )

    print(
        "  ONLY final hidden states strictly before "
        "the answer causal position"
    )

    print()

    print(
        "Retrieved value destination:"
    )

    print(
        "  residual addition to answer hidden "
        "BEFORE native language_head"
    )

    print()

    print(
        "Final retention gate:"
    )

    print(
        f"  correct>distractor >= "
        f"{RETENTION_CORRECT_GATE:.2f}"
    )

    print(
        f"  delta>=0.5          >= "
        f"{RETENTION_MARGIN_GATE:.2f}"
    )

    print()

    print(
        "Positive controls during preflight: NO"
    )

    print(
        "Sealed evaluation during preflight: NO"
    )

    print()

    print(
        "Preflight result:"
    )

    print(
        PREFLIGHT_RESULT_PATH
    )

    heading(
        "TREATMENT #9 PREFLIGHT PASS"
    )

    print(
        "Right patient."
    )

    print(
        "Right frozen data."
    )

    print(
        "Right starting checkpoint."
    )

    print(
        "One retrieval mechanism."
    )

    print(
        "No answer leakage."
    )

    print(
        "No fucking participation trophy."
    )


if __name__ == "__main__":
    main()