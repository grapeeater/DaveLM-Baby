from __future__ import annotations

r"""
DaveLM v0.9 - Treatment #5 Final Training-Retention Audit (v3)

READ-ONLY. NO TRAINING.

Run from C:\DaveLM-CADAVER:

    py -3.12 .\treatment5_final_retention_audit_v3.py --device cuda

Why v3 exists:
The final full-document T5 artifacts are hash-locked, but their JSON wrapper/field
names are not assumed here. This audit verifies the exact frozen hashes first,
then discovers the schedule/pair records by their semantic contents rather than
requiring one hard-coded artifact label.

This script does NOT:
- create an optimizer
- call backward()
- enable gradients
- modify model parameters
- train Baby
- evaluate positive controls
- open sealed novel/shortcut evaluation pools
"""

import argparse
import hashlib
import importlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


# =============================================================================
# FROZEN CONTRACT
# =============================================================================

CADAVER_ROOT = Path(r"C:\DaveLM-CADAVER")
SOURCE_ROOT = Path(r"C:\DaveLM-v0.9")

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
    / "treatment5_paired_counterfactual_query_binding_seed8380"
    / "checkpoints"
    / "paired_counterfactual"
    / "seed_8380"
    / "latest.pt"
)

OUTPUT_PATH = (
    CADAVER_ROOT
    / "treatment5_paired_counterfactual_query_binding_seed8380"
    / "treatment5_final_retention_audit.json"
)

EXPECTED_PAIR_POOL_SHA256 = (
    "0b31e39361f7a45dc4182c49ab4b4967bffe539cd2ff12e7b196eabfb3dac307"
)
EXPECTED_SCHEDULE_SHA256 = (
    "a4e64fd9d1b934aa440a5d036a3bec0ac7c9ac97d58c72fb45df6d8576bb7c12"
)
EXPECTED_FINAL_CHECKPOINT_SHA256 = (
    "cfe1e99f17c02ee7fc7ab6869eb367960d4e98fc477bb9d22173a978361f077a"
)

EXPECTED_PAIR_COUNT = 1536
EXPECTED_STEPS = 1000
EXPECTED_BATCH_SIZE = 32
EXPECTED_PAIRS_PER_BATCH = 16
EXPECTED_EXAMPLES = 32000
EXPECTED_SLOT0 = 16000
EXPECTED_SLOT1 = 16000
EXPECTED_PAIR_EXPOSURE_MIN = 10
EXPECTED_PAIR_EXPOSURE_MAX = 11
EXPECTED_TOTAL_SUPERVISED_POSITIONS = 6_144_000
EXPECTED_ANSWER_REPLACEMENTS = 32_000
EXPECTED_NONANSWER_CE_POSITIONS = 6_112_000
EXPECTED_PARAMETER_COUNT = 10_594_944
EXPECTED_BOS_ID = 2

MARGIN = 0.5
STRONG_RETENTION_PAIR_WIN = 0.95
STRONG_RETENTION_MARGIN = 0.90


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def banner(text: str) -> None:
    print()
    print("=" * 100)
    print(text)
    print("=" * 100)
    print()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def verify_sha(path: Path, expected: str, label: str) -> str:
    require(path.exists(), f"Missing {label}: {path}")
    actual = sha256_file(path)

    print(f"{label} SHA256:")
    print(f"  {actual}")
    print("Expected:")
    print(f"  {expected}")

    require(actual == expected, f"{label} SHA256 mismatch.")
    print(f"{label} integrity: PASS\n")
    return actual


def first_value(record: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return default


def int_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(x, int) and not isinstance(x, bool) for x in value)
    )


def walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


TOKEN_KEYS = (
    "full_document_token_ids",
    "full_document_ids",
    "document_token_ids",
    "document_ids",
    "model_visible_token_ids",
    "model_visible_ids",
    "sequence_token_ids",
    "sequence_ids",
    "token_ids",
    "input_token_ids",
    "input_ids",
    "ids",
)

TARGET_KEYS = (
    "target_token_id",
    "answer_token_id",
    "correct_token_id",
    "target_id",
)

DISTRACTOR_KEYS = (
    "distractor_token_id",
    "wrong_token_id",
    "negative_token_id",
    "distractor_id",
)

SLOT_KEYS = (
    "query_slot",
    "slot",
    "query_index",
)

PAIR_ID_KEYS = (
    "pair_id",
    "base_record_id",
    "pair_index",
    "base_record_index",
    "native_record_index",
    "source_record_index",
    "record_index",
)


def find_token_ids(record: dict[str, Any]) -> tuple[str, list[int]] | None:
    candidates: list[tuple[int, int, str, list[int]]] = []

    for priority, key in enumerate(TOKEN_KEYS):
        value = record.get(key)
        if int_list(value):
            ids = [int(x) for x in value]
            candidates.append((len(ids), -priority, key, ids))

    # Fallback: semantic discovery if the final artifact uses an unexpected key.
    if not candidates:
        for key, value in record.items():
            if int_list(value):
                ids = [int(x) for x in value]
                if ids and ids[0] == EXPECTED_BOS_ID and len(ids) >= 16:
                    candidates.append((len(ids), -999, key, ids))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    _, _, key, ids = candidates[0]
    return key, ids


def canonical_pair_key(record: dict[str, Any], parent_pair_key: Any = None) -> str | None:
    value = first_value(record, PAIR_ID_KEYS, None)
    if value is None:
        value = parent_pair_key
    if value is None:
        return None
    return str(value)


def normalize_event(
    record: dict[str, Any],
    *,
    parent_pair_key: Any = None,
    allow_missing_tokens: bool = False,
) -> dict[str, Any] | None:
    slot = first_value(record, SLOT_KEYS, None)
    target = first_value(record, TARGET_KEYS, None)
    distractor = first_value(record, DISTRACTOR_KEYS, None)

    if slot is None or target is None or distractor is None:
        return None

    try:
        slot = int(slot)
        target = int(target)
        distractor = int(distractor)
    except (TypeError, ValueError):
        return None

    if slot not in (0, 1) or target == distractor:
        return None

    token_result = find_token_ids(record)
    if token_result is None and not allow_missing_tokens:
        return None

    token_key = None
    ids = None
    if token_result is not None:
        token_key, ids = token_result
        require(ids[0] == EXPECTED_BOS_ID,
                f"Event token sequence does not begin with BOS={EXPECTED_BOS_ID}.")

    pair_key = canonical_pair_key(record, parent_pair_key)

    candidate_pair = first_value(
        record,
        ("candidate_pair", "candidate_token_ids", "candidate_ids"),
        None,
    )
    if isinstance(candidate_pair, list) and len(candidate_pair) == 2:
        candidate_pair = [int(candidate_pair[0]), int(candidate_pair[1])]
    else:
        candidate_pair = sorted([target, distractor])

    return {
        "pair_key": pair_key,
        "query_slot": slot,
        "target_token_id": target,
        "distractor_token_id": distractor,
        "candidate_pair": candidate_pair,
        "token_key": token_key,
        "token_ids": ids,
        "query_identity": first_value(record, ("query_identity", "query", "source_identity"), None),
        "query_difference_position": first_value(
            record,
            ("query_difference_position", "query_token_position", "query_position"),
            None,
        ),
        "raw": record,
    }


# =============================================================================
# PAIR-POOL DISCOVERY
# =============================================================================

def child_event_dicts(pair_record: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    # Common immediate twin containers.
    for key in (
        "twin_a", "twin_b", "slot0", "slot1", "slot_0", "slot_1",
        "example_a", "example_b", "example0", "example1",
    ):
        value = pair_record.get(key)
        if isinstance(value, dict) and normalize_event(value, parent_pair_key=None) is not None:
            found.append(value)

    # Common list form.
    for key in ("twins", "examples", "documents", "variants"):
        value = pair_record.get(key)
        if isinstance(value, list):
            for child in value:
                if isinstance(child, dict) and normalize_event(child, parent_pair_key=None) is not None:
                    found.append(child)

    # Last-resort immediate child scan.
    if len(found) < 2:
        for value in pair_record.values():
            if isinstance(value, dict) and normalize_event(value, parent_pair_key=None) is not None:
                found.append(value)

    # Deduplicate by object identity.
    unique: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in found:
        if id(item) not in seen:
            unique.append(item)
            seen.add(id(item))
    return unique


def locate_pair_list(pair_pool_raw: Any) -> list[dict[str, Any]]:
    preferred_keys = ("pairs", "pair_pool", "counterfactual_pairs", "records", "items")

    if isinstance(pair_pool_raw, dict):
        for key in preferred_keys:
            value = pair_pool_raw.get(key)
            if (
                isinstance(value, list)
                and len(value) == EXPECTED_PAIR_COUNT
                and all(isinstance(x, dict) for x in value)
            ):
                return value

        for value in pair_pool_raw.values():
            if (
                isinstance(value, list)
                and len(value) == EXPECTED_PAIR_COUNT
                and all(isinstance(x, dict) for x in value)
            ):
                return value

    if (
        isinstance(pair_pool_raw, list)
        and len(pair_pool_raw) == EXPECTED_PAIR_COUNT
        and all(isinstance(x, dict) for x in pair_pool_raw)
    ):
        return pair_pool_raw

    raise RuntimeError(
        "Could not locate the 1,536-record pair list inside the hash-verified final pair-pool JSON."
    )


def discover_pair_pool(
    pair_pool_raw: Any,
) -> dict[str, dict[int, dict[str, Any]]]:
    pair_records = locate_pair_list(pair_pool_raw)
    lookup: dict[str, dict[int, dict[str, Any]]] = {}

    for index, pair_record in enumerate(pair_records):
        parent_key = canonical_pair_key(pair_record, index)
        require(parent_key is not None, f"Pair {index}: could not determine pair key.")

        child_dicts = child_event_dicts(pair_record)
        events: list[dict[str, Any]] = []

        for child in child_dicts:
            event = normalize_event(child, parent_pair_key=parent_key)
            if event is not None:
                events.append(event)

        by_slot = {event["query_slot"]: event for event in events}
        require(
            set(by_slot) == {0, 1},
            f"Pair {parent_key}: could not discover exactly one slot-0 and one slot-1 twin."
        )

        a = by_slot[0]
        b = by_slot[1]

        require(
            a["target_token_id"] == b["distractor_token_id"],
            f"Pair {parent_key}: slot-0 target != slot-1 distractor."
        )
        require(
            b["target_token_id"] == a["distractor_token_id"],
            f"Pair {parent_key}: slot-1 target != slot-0 distractor."
        )
        require(
            set(a["candidate_pair"]) == {a["target_token_id"], a["distractor_token_id"]},
            f"Pair {parent_key}: slot-0 candidate pair mismatch."
        )
        require(
            set(b["candidate_pair"]) == {b["target_token_id"], b["distractor_token_id"]},
            f"Pair {parent_key}: slot-1 candidate pair mismatch."
        )

        a_ids = a["token_ids"]
        b_ids = b["token_ids"]
        require(a_ids is not None and b_ids is not None,
                f"Pair {parent_key}: missing frozen token sequence.")
        require(len(a_ids) == len(b_ids),
                f"Pair {parent_key}: twin sequence lengths differ.")

        diffs = [i for i, (x, y) in enumerate(zip(a_ids, b_ids)) if x != y]

        a_is_full = a_ids[-1] == a["target_token_id"]
        b_is_full = b_ids[-1] == b["target_token_id"]

        if a_is_full and b_is_full:
            require(
                len(diffs) == 2 and (len(a_ids) - 1) in diffs,
                f"Pair {parent_key}: full-document twins should differ at query + answer only; got {diffs}."
            )
        else:
            require(
                len(diffs) == 1,
                f"Pair {parent_key}: prefix twins should differ only at query token; got {diffs}."
            )

        if parent_key in lookup:
            raise RuntimeError(f"Duplicate pair key discovered: {parent_key}")
        lookup[parent_key] = by_slot

    require(
        len(lookup) == EXPECTED_PAIR_COUNT,
        f"Expected {EXPECTED_PAIR_COUNT} discovered pairs, got {len(lookup)}."
    )
    return lookup


# =============================================================================
# SCHEDULE DISCOVERY
# =============================================================================

def locate_steps_list(schedule_raw: Any) -> list[Any]:
    preferred_keys = (
        "steps_data",
        "batches",
        "schedule",
        "frozen_schedule",
        "training_steps",
        "step_data",
    )

    if isinstance(schedule_raw, dict):
        for key in preferred_keys:
            value = schedule_raw.get(key)
            if isinstance(value, list) and len(value) == EXPECTED_STEPS:
                return value

        for value in schedule_raw.values():
            if isinstance(value, list) and len(value) == EXPECTED_STEPS:
                return value

    if isinstance(schedule_raw, list) and len(schedule_raw) == EXPECTED_STEPS:
        return schedule_raw

    raise RuntimeError(
        "Could not locate the 1,000-step schedule list inside the hash-verified final schedule JSON."
    )


def locate_examples_in_step(step: Any) -> list[Any] | None:
    if isinstance(step, list):
        if len(step) in (EXPECTED_BATCH_SIZE, EXPECTED_PAIRS_PER_BATCH):
            return step
        return None

    if not isinstance(step, dict):
        return None

    for key in ("examples", "events", "items", "batch", "presentations", "records"):
        value = step.get(key)
        if isinstance(value, list) and len(value) in (EXPECTED_BATCH_SIZE, EXPECTED_PAIRS_PER_BATCH):
            return value

    # Scan for a direct 32/16-list if the key name is unfamiliar.
    for value in step.values():
        if isinstance(value, list) and len(value) in (EXPECTED_BATCH_SIZE, EXPECTED_PAIRS_PER_BATCH):
            return value

    return None


def resolve_pair_key_from_ref(ref: Any) -> str:
    if isinstance(ref, (str, int)):
        return str(ref)

    if isinstance(ref, dict):
        key = canonical_pair_key(ref)
        if key is not None:
            return key

    raise RuntimeError(f"Could not resolve schedule pair reference: {ref!r}")


def token_sequences_compatible(schedule_ids: list[int], pool_ids: list[int], target: int) -> bool:
    if schedule_ids == pool_ids:
        return True
    if len(pool_ids) == len(schedule_ids) + 1:
        return pool_ids[:-1] == schedule_ids and pool_ids[-1] == target
    if len(schedule_ids) == len(pool_ids) + 1:
        return schedule_ids[:-1] == pool_ids and schedule_ids[-1] == target
    return False


def discover_schedule(
    schedule_raw: Any,
    pair_lookup: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    steps = locate_steps_list(schedule_raw)
    all_events: list[dict[str, Any]] = []
    pair_exposure = Counter()
    slot_counts = Counter()

    for step_number, step in enumerate(steps, start=1):
        items = locate_examples_in_step(step)
        require(items is not None,
                f"Step {step_number}: could not discover its 32 examples or 16 pair references.")

        step_events: list[dict[str, Any]] = []

        if len(items) == EXPECTED_BATCH_SIZE:
            for item in items:
                require(isinstance(item, dict),
                        f"Step {step_number}: 32-item schedule contains a non-object event.")

                event = normalize_event(item, allow_missing_tokens=True)
                require(event is not None,
                        f"Step {step_number}: could not interpret a scheduled example.")

                pair_key = event["pair_key"]
                require(pair_key is not None,
                        f"Step {step_number}: scheduled example has no resolvable pair key.")
                require(pair_key in pair_lookup,
                        f"Step {step_number}: unknown pair key {pair_key!r}.")

                pool_event = pair_lookup[pair_key][event["query_slot"]]

                require(
                    event["target_token_id"] == pool_event["target_token_id"],
                    f"Step {step_number} pair {pair_key}: target mismatch vs pair pool."
                )
                require(
                    event["distractor_token_id"] == pool_event["distractor_token_id"],
                    f"Step {step_number} pair {pair_key}: distractor mismatch vs pair pool."
                )

                if event["token_ids"] is None:
                    event["token_ids"] = pool_event["token_ids"]
                    event["token_key"] = f"pair_pool:{pool_event['token_key']}"
                else:
                    require(
                        token_sequences_compatible(
                            event["token_ids"],
                            pool_event["token_ids"],
                            event["target_token_id"],
                        ),
                        f"Step {step_number} pair {pair_key}: schedule/pair-pool token sequences disagree."
                    )

                step_events.append(event)

        else:
            # 16 complete pair references: expand A then B from the frozen pair pool.
            for ref in items:
                pair_key = resolve_pair_key_from_ref(ref)
                require(pair_key in pair_lookup,
                        f"Step {step_number}: unknown pair reference {pair_key!r}.")
                step_events.append(dict(pair_lookup[pair_key][0]))
                step_events.append(dict(pair_lookup[pair_key][1]))

        require(
            len(step_events) == EXPECTED_BATCH_SIZE,
            f"Step {step_number}: expected 32 resolved events, got {len(step_events)}."
        )

        per_pair_slots: dict[str, set[int]] = defaultdict(set)
        for event in step_events:
            pair_key = event["pair_key"]
            require(pair_key is not None, f"Step {step_number}: resolved event lacks pair key.")
            per_pair_slots[pair_key].add(event["query_slot"])
            slot_counts[event["query_slot"]] += 1

            ids = event["token_ids"]
            require(ids is not None and len(ids) >= 2,
                    f"Step {step_number} pair {pair_key}: missing token sequence.")
            require(ids[0] == EXPECTED_BOS_ID,
                    f"Step {step_number} pair {pair_key}: missing BOS token.")

            event["step"] = step_number
            all_events.append(event)

        require(
            len(per_pair_slots) == EXPECTED_PAIRS_PER_BATCH,
            f"Step {step_number}: expected 16 distinct pairs, got {len(per_pair_slots)}."
        )

        for pair_key, slots in per_pair_slots.items():
            require(
                slots == {0, 1},
                f"Step {step_number}: pair {pair_key} is not a complete slot-0/slot-1 twin."
            )
            pair_exposure[pair_key] += 1

    require(len(all_events) == EXPECTED_EXAMPLES,
            f"Expected {EXPECTED_EXAMPLES} events, got {len(all_events)}.")
    require(slot_counts[0] == EXPECTED_SLOT0,
            f"Expected {EXPECTED_SLOT0} slot-0 events, got {slot_counts[0]}.")
    require(slot_counts[1] == EXPECTED_SLOT1,
            f"Expected {EXPECTED_SLOT1} slot-1 events, got {slot_counts[1]}.")
    require(len(pair_exposure) == EXPECTED_PAIR_COUNT,
            f"Expected all {EXPECTED_PAIR_COUNT} pairs to appear, saw {len(pair_exposure)}.")
    require(min(pair_exposure.values()) == EXPECTED_PAIR_EXPOSURE_MIN,
            "Observed minimum pair exposure is not 10.")
    require(max(pair_exposure.values()) == EXPECTED_PAIR_EXPOSURE_MAX,
            "Observed maximum pair exposure is not 11.")

    return all_events


# =============================================================================
# MODEL LOADING
# =============================================================================

def import_original_run() -> Any:
    require(SOURCE_ROOT.exists(), f"Protected source root not found: {SOURCE_ROOT}")

    source_root = str(SOURCE_ROOT)
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

    return importlib.import_module("experiments.two_mapping_contextual_binding.run")


def extract_model_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    require(isinstance(checkpoint, dict), "Checkpoint root is not a dictionary.")

    for key in ("model_state", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
            return value

    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    raise RuntimeError("Could not find a model state dictionary in the final checkpoint.")


def load_model(device: torch.device) -> torch.nn.Module:
    original_run = import_original_run()
    require(hasattr(original_run, "build_model"),
            'Original module does not expose build_model("untied").')

    model = original_run.build_model("untied").to(device)

    parameter_count = sum(p.numel() for p in model.parameters())
    require(
        parameter_count == EXPECTED_PARAMETER_COUNT,
        f"Parameter count mismatch: expected {EXPECTED_PARAMETER_COUNT:,}, got {parameter_count:,}."
    )

    checkpoint = torch.load(FINAL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state = extract_model_state(checkpoint)

    incompatible = model.load_state_dict(state, strict=False)
    require(not incompatible.missing_keys,
            f"Missing checkpoint keys: {incompatible.missing_keys}")
    require(not incompatible.unexpected_keys,
            f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        logits = output
    elif isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        logits = output[0]
    elif isinstance(output, dict) and torch.is_tensor(output.get("logits")):
        logits = output["logits"]
    elif hasattr(output, "logits") and torch.is_tensor(output.logits):
        logits = output.logits
    else:
        raise RuntimeError(f"Could not extract logits from model output type {type(output).__name__}.")

    require(logits.ndim == 3,
            f"Expected logits [batch,time,vocab], got shape {tuple(logits.shape)}.")
    return logits


# =============================================================================
# ANSWER-INPUT SEMANTICS
# =============================================================================

def prepare_answer_input(event: dict[str, Any]) -> tuple[list[int], str]:
    ids = event["token_ids"]
    target = event["target_token_id"]

    require(ids is not None and len(ids) >= 2, "Missing event token sequence.")

    # Full document artifact: final token is the answer. Feed everything before it.
    if ids[-1] == target:
        return ids[:-1], "full_document_minus_answer"

    # Prefix artifact: sequence already ends at "Answer:" and predicts next token.
    return ids, "answer_prefix"


# =============================================================================
# METRIC ACCUMULATION
# =============================================================================

class Metrics:
    def __init__(self) -> None:
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
        self.ranks: list[int] = []

    def add(
        self,
        answer_logits: torch.Tensor,
        target_ids: torch.Tensor,
        distractor_ids: torch.Tensor,
    ) -> None:
        batch_size = answer_logits.shape[0]
        rows = torch.arange(batch_size, device=answer_logits.device)

        target_logits = answer_logits[rows, target_ids]
        distractor_logits = answer_logits[rows, distractor_ids]
        delta = target_logits - distractor_logits

        full_lse = torch.logsumexp(answer_logits, dim=-1)
        candidate_lse = torch.logsumexp(
            torch.stack((target_logits, distractor_logits), dim=-1),
            dim=-1,
        )

        membership_loss = full_lse - candidate_lse
        selector_loss = F.relu(MARGIN - delta)
        replacement_loss = membership_loss + selector_loss

        probabilities = torch.softmax(answer_logits, dim=-1)
        target_probability = probabilities[rows, target_ids]
        distractor_probability = probabilities[rows, distractor_ids]
        candidate_mass = target_probability + distractor_probability

        top1_ids = answer_logits.argmax(dim=-1)
        top5_ids = torch.topk(answer_logits, k=5, dim=-1).indices
        top5_hit = (top5_ids == target_ids.unsqueeze(-1)).any(dim=-1)

        ranks = 1 + (answer_logits > target_logits.unsqueeze(-1)).sum(dim=-1)

        self.count += batch_size
        self.top1 += int((top1_ids == target_ids).sum().item())
        self.top5 += int(top5_hit.sum().item())
        self.correct_gt_distractor += int((delta > 0).sum().item())
        self.margin_satisfied += int((delta >= MARGIN).sum().item())

        self.sum_candidate_mass += float(candidate_mass.sum().item())
        self.sum_membership_loss += float(membership_loss.sum().item())
        self.sum_selector_loss += float(selector_loss.sum().item())
        self.sum_replacement_loss += float(replacement_loss.sum().item())
        self.sum_target_probability += float(target_probability.sum().item())
        self.sum_distractor_probability += float(distractor_probability.sum().item())
        self.sum_logit_delta += float(delta.sum().item())
        self.ranks.extend(int(x) for x in ranks.detach().cpu().tolist())

    def summary(self) -> dict[str, Any]:
        require(self.count > 0, "Cannot summarize zero events.")

        return {
            "count": self.count,
            "full_vocab_answer_top1": self.top1 / self.count,
            "full_vocab_answer_top5": self.top5 / self.count,
            "correct_gt_distractor": self.correct_gt_distractor / self.count,
            "margin_ge_0_5": self.margin_satisfied / self.count,
            "candidate_mass": self.sum_candidate_mass / self.count,
            "membership_loss": self.sum_membership_loss / self.count,
            "selector_loss": self.sum_selector_loss / self.count,
            "replacement_loss": self.sum_replacement_loss / self.count,
            "target_probability": self.sum_target_probability / self.count,
            "distractor_probability": self.sum_distractor_probability / self.count,
            "mean_logit_delta": self.sum_logit_delta / self.count,
            "target_rank_mean": statistics.fmean(self.ranks),
            "target_rank_median": statistics.median(self.ranks),
        }


# =============================================================================
# RETENTION REPLAY
# =============================================================================

def replay(
    model: torch.nn.Module,
    events: list[dict[str, Any]],
    device: torch.device,
    inference_batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Counter]:
    overall = Metrics()
    slot0 = Metrics()
    slot1 = Metrics()
    semantics = Counter()

    buckets: dict[int, list[tuple[dict[str, Any], list[int]]]] = defaultdict(list)

    for event in events:
        model_input, mode = prepare_answer_input(event)
        semantics[mode] += 1
        buckets[len(model_input)].append((event, model_input))

    processed = 0
    next_report = 3200

    with torch.inference_mode():
        for input_length in sorted(buckets):
            bucket = buckets[input_length]

            for start in range(0, len(bucket), inference_batch_size):
                rows = bucket[start:start + inference_batch_size]
                batch_events = [row[0] for row in rows]
                batch_inputs = [row[1] for row in rows]

                inputs = torch.tensor(batch_inputs, dtype=torch.long, device=device)
                target_ids = torch.tensor(
                    [event["target_token_id"] for event in batch_events],
                    dtype=torch.long,
                    device=device,
                )
                distractor_ids = torch.tensor(
                    [event["distractor_token_id"] for event in batch_events],
                    dtype=torch.long,
                    device=device,
                )
                slots = torch.tensor(
                    [event["query_slot"] for event in batch_events],
                    dtype=torch.long,
                    device=device,
                )

                logits = extract_logits(model(inputs))
                answer_logits = logits[:, -1, :]

                overall.add(answer_logits, target_ids, distractor_ids)

                mask0 = slots == 0
                if bool(mask0.any()):
                    slot0.add(answer_logits[mask0], target_ids[mask0], distractor_ids[mask0])

                mask1 = slots == 1
                if bool(mask1.any()):
                    slot1.add(answer_logits[mask1], target_ids[mask1], distractor_ids[mask1])

                processed += len(rows)
                while processed >= next_report and next_report <= EXPECTED_EXAMPLES:
                    print(f"  audited at least {next_report:>5}/{EXPECTED_EXAMPLES} events")
                    next_report += 3200

    require(overall.count == EXPECTED_EXAMPLES, "Overall replay count mismatch.")
    require(slot0.count == EXPECTED_SLOT0, "Slot-0 replay count mismatch.")
    require(slot1.count == EXPECTED_SLOT1, "Slot-1 replay count mismatch.")

    return overall.summary(), slot0.summary(), slot1.summary(), semantics


def print_summary(title: str, summary: dict[str, Any]) -> None:
    print(title)
    print(f"  events:                    {summary['count']}")
    print(f"  full-vocab answer top1:    {summary['full_vocab_answer_top1']:.6f}")
    print(f"  full-vocab answer top5:    {summary['full_vocab_answer_top5']:.6f}")
    print(f"  correct > distractor:      {summary['correct_gt_distractor']:.6f}")
    print(f"  margin >= 0.5:             {summary['margin_ge_0_5']:.6f}")
    print(f"  candidate mass:            {summary['candidate_mass']:.6f}")
    print(f"  membership loss:           {summary['membership_loss']:.6f}")
    print(f"  selector loss:             {summary['selector_loss']:.6f}")
    print(f"  replacement loss:          {summary['replacement_loss']:.6f}")
    print(f"  target probability:        {summary['target_probability']:.6f}")
    print(f"  distractor probability:    {summary['distractor_probability']:.6f}")
    print(f"  mean logit delta:          {summary['mean_logit_delta']:+.6f}")
    print(f"  target rank mean:          {summary['target_rank_mean']:.6f}")
    print(f"  target rank median:        {summary['target_rank_median']:.6f}")
    print()


def classify(overall: dict[str, Any]) -> tuple[str, str]:
    if (
        overall["correct_gt_distractor"] >= STRONG_RETENTION_PAIR_WIN
        and overall["margin_ge_0_5"] >= STRONG_RETENTION_MARGIN
    ):
        return (
            "STRONG_FINAL_TRAINING_RETENTION",
            "Retention is strong. Proceed only to the same pre-authorized T4 positive controls. "
            "Novel/shortcut pools remain sealed unless both positive-control gates pass.",
        )

    return (
        "TRAINING_TASK_FAILURE_OR_INCOMPLETE_RETENTION",
        "Do NOT run positive controls yet. The final checkpoint did not demonstrate strong "
        "retention of the paired counterfactual training task under the frozen T5 objective.",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only final training-retention audit for DaveLM Treatment #5."
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Inference device. Default: cuda",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Inference batch size within equal-length answer-input buckets. Default: 128",
    )
    args = parser.parse_args()

    require(args.batch_size >= 1, "--batch-size must be at least 1.")

    banner("TREATMENT #5 FINAL TRAINING-RETENTION AUDIT v3 - READ ONLY")

    banner("VERIFYING EXACT FROZEN ARTIFACTS")
    schedule_sha = verify_sha(
        SCHEDULE_PATH,
        EXPECTED_SCHEDULE_SHA256,
        "Frozen schedule",
    )
    pair_pool_sha = verify_sha(
        PAIR_POOL_PATH,
        EXPECTED_PAIR_POOL_SHA256,
        "Pair-pool",
    )
    checkpoint_sha = verify_sha(
        FINAL_CHECKPOINT_PATH,
        EXPECTED_FINAL_CHECKPOINT_SHA256,
        "Final checkpoint",
    )

    banner("DISCOVERING AND VALIDATING HASH-VERIFIED T5 ARTIFACT STRUCTURE")
    pair_pool_raw = load_json(PAIR_POOL_PATH)
    schedule_raw = load_json(SCHEDULE_PATH)

    pair_lookup = discover_pair_pool(pair_pool_raw)
    events = discover_schedule(schedule_raw, pair_lookup)

    frozen_lengths = sorted({len(event["token_ids"]) for event in events})
    input_modes = Counter(
        "full_document" if event["token_ids"][-1] == event["target_token_id"] else "answer_prefix"
        for event in events
    )

    print(f"Discovered frozen pairs:     {len(pair_lookup)}")
    print(f"Resolved schedule events:    {len(events)}")
    print(f"Slot-0 events:               {sum(e['query_slot'] == 0 for e in events)}")
    print(f"Slot-1 events:               {sum(e['query_slot'] == 1 for e in events)}")
    print(f"Frozen sequence lengths:     {frozen_lengths}")
    print(f"Frozen sequence semantics:   {dict(input_modes)}")
    print(f"Answer replacements:         {EXPECTED_ANSWER_REPLACEMENTS}")
    print(f"Training supervised total:   {EXPECTED_TOTAL_SUPERVISED_POSITIONS}")
    print(f"Training non-answer CE:      {EXPECTED_NONANSWER_CE_POSITIONS}")
    print()
    print("All 1,536 reciprocal pairs validated: PASS")
    print("Every frozen batch resolves to 16 complete A/B pairs: PASS")
    print("Pair exposure range 10-11: PASS")
    print("Exact 32,000-event schedule replay contract: PASS")

    banner("LOADING FINAL BABY CHECKPOINT")
    if args.device == "cuda":
        require(torch.cuda.is_available(),
                "CUDA/ROCm requested but torch.cuda.is_available() is False.")

    device = torch.device(args.device)
    model = load_model(device)

    print(f"Device:       {device}")
    if device.type == "cuda":
        print(f"GPU:          {torch.cuda.get_device_name(device)}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
    print('Builder:      original_run.build_model("untied")')
    print("Mode:         model.eval()")
    print("Inference:    torch.inference_mode()")
    print("Optimizer:    NONE")
    print("Backward:     NONE")
    print("Training:     NONE")
    print("Positive PCs: NONE")
    print("Sealed eval:  UNTOUCHED")

    banner("REPLAYING ALL 32,000 FROZEN T5 ANSWER EVENTS")
    overall, slot0, slot1, replay_semantics = replay(
        model=model,
        events=events,
        device=device,
        inference_batch_size=args.batch_size,
    )

    banner("FINAL T5 TRAINING-RETENTION RESULTS")
    print_summary("OVERALL", overall)
    print_summary("SLOT 0", slot0)
    print_summary("SLOT 1", slot1)

    classification, next_step = classify(overall)

    print(f"CLASSIFICATION: {classification}")
    print()
    print(next_step)

    result = {
        "audit": "DaveLM v0.9 Treatment #5 final training-retention audit v3",
        "read_only": True,
        "positive_controls_evaluated": False,
        "sealed_evaluation_opened": False,
        "frozen_artifacts": {
            "schedule_path": str(SCHEDULE_PATH),
            "schedule_sha256": schedule_sha,
            "pair_pool_path": str(PAIR_POOL_PATH),
            "pair_pool_sha256": pair_pool_sha,
            "final_checkpoint_path": str(FINAL_CHECKPOINT_PATH),
            "final_checkpoint_sha256": checkpoint_sha,
        },
        "contract": {
            "pair_count": EXPECTED_PAIR_COUNT,
            "steps": EXPECTED_STEPS,
            "batch_size": EXPECTED_BATCH_SIZE,
            "complete_pairs_per_batch": EXPECTED_PAIRS_PER_BATCH,
            "scheduled_examples": EXPECTED_EXAMPLES,
            "slot0_examples": EXPECTED_SLOT0,
            "slot1_examples": EXPECTED_SLOT1,
            "pair_exposure_min": EXPECTED_PAIR_EXPOSURE_MIN,
            "pair_exposure_max": EXPECTED_PAIR_EXPOSURE_MAX,
            "total_supervised_positions": EXPECTED_TOTAL_SUPERVISED_POSITIONS,
            "answer_replacements": EXPECTED_ANSWER_REPLACEMENTS,
            "ordinary_nonanswer_ce_positions": EXPECTED_NONANSWER_CE_POSITIONS,
            "margin": MARGIN,
            "strong_retention_correct_gt_distractor_threshold": STRONG_RETENTION_PAIR_WIN,
            "strong_retention_margin_threshold": STRONG_RETENTION_MARGIN,
        },
        "artifact_discovery": {
            "frozen_sequence_lengths": frozen_lengths,
            "frozen_sequence_semantics": dict(input_modes),
            "replay_input_semantics": dict(replay_semantics),
            "schedule_cross_checked_against_pair_pool": True,
        },
        "overall": overall,
        "slot0": slot0,
        "slot1": slot1,
        "classification": classification,
        "next_step": next_step,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    banner("AUDIT COMPLETE")
    print("Result JSON:")
    print(f"  {OUTPUT_PATH}")
    print()
    print("No optimizer was created.")
    print("No backward pass was run.")
    print("No model parameters were changed.")
    print("No positive controls were evaluated.")
    print("No sealed evaluation pools were opened.")


if __name__ == "__main__":
    main()
