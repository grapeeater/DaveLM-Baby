from __future__ import annotations

"""
DaveLM v0.9 - Treatment #5 Final Training-Retention Audit

Read-only audit of the exact 32,000 frozen T5 answer events.

Run from C:\DaveLM-CADAVER with:
    py -3.12 .\treatment5_final_retention_audit.py --device cuda

This script does NOT:
- create an optimizer
- call backward()
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
from typing import Any

import torch
import torch.nn.functional as F


# ============================================================================
# FROZEN TREATMENT #5 CONTRACT
# ============================================================================

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


# ============================================================================
# HELPERS
# ============================================================================

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
    with path.open("r", encoding="utf-8") as handle:
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


# ============================================================================
# EXACT ARTIFACT VALIDATION
# ============================================================================

def validate_pair_pool(raw: Any) -> dict[str, dict[str, Any]]:
    require(isinstance(raw, dict), "Pair-pool root must be a JSON object.")
    require(raw.get("artifact") == "same_record_counterfactual_pair_pool",
            "Unexpected pair-pool artifact type.")
    require(raw.get("experiment") == "DaveLM v0.9 Treatment #5",
            "Unexpected pair-pool experiment label.")
    require(raw.get("pair_count") == EXPECTED_PAIR_COUNT,
            f"Expected pair_count={EXPECTED_PAIR_COUNT}.")

    pairs = raw.get("pairs")
    require(isinstance(pairs, list), "pair_pool['pairs'] must be a list.")
    require(len(pairs) == EXPECTED_PAIR_COUNT,
            f"Expected {EXPECTED_PAIR_COUNT} pairs, found {len(pairs)}.")

    lookup: dict[str, dict[str, Any]] = {}

    for index, pair in enumerate(pairs):
        require(isinstance(pair, dict), f"Pair {index} is not a JSON object.")

        pair_id = pair.get("pair_id")
        require(isinstance(pair_id, str), f"Pair {index} has invalid pair_id.")
        require(pair_id not in lookup, f"Duplicate pair_id: {pair_id}")

        twin_a = pair.get("twin_a")
        twin_b = pair.get("twin_b")
        require(isinstance(twin_a, dict) and isinstance(twin_b, dict),
                f"{pair_id}: missing twin_a/twin_b objects.")

        candidate_pair = pair.get("candidate_pair")
        require(isinstance(candidate_pair, list) and len(candidate_pair) == 2,
                f"{pair_id}: candidate_pair must contain exactly two token IDs.")

        require(twin_a.get("query_slot") == 0, f"{pair_id}: twin_a must be slot 0.")
        require(twin_b.get("query_slot") == 1, f"{pair_id}: twin_b must be slot 1.")

        a_target = int(twin_a["target_token_id"])
        a_dist = int(twin_a["distractor_token_id"])
        b_target = int(twin_b["target_token_id"])
        b_dist = int(twin_b["distractor_token_id"])

        require(a_target == b_dist, f"{pair_id}: A target != B distractor.")
        require(b_target == a_dist, f"{pair_id}: B target != A distractor.")
        require(a_target != a_dist, f"{pair_id}: target equals distractor.")
        require(set(candidate_pair) == {a_target, a_dist},
                f"{pair_id}: candidate_pair does not match reciprocal targets.")

        a_ids = twin_a.get("input_token_ids")
        b_ids = twin_b.get("input_token_ids")
        require(isinstance(a_ids, list) and isinstance(b_ids, list),
                f"{pair_id}: missing input_token_ids.")
        require(len(a_ids) == len(b_ids), f"{pair_id}: twin prefix lengths differ.")
        require(len(a_ids) == int(pair["prefix_length"]),
                f"{pair_id}: prefix_length does not match input_token_ids length.")
        require(a_ids[0] == EXPECTED_BOS_ID and b_ids[0] == EXPECTED_BOS_ID,
                f"{pair_id}: missing BOS token ID {EXPECTED_BOS_ID}.")

        query_pos = int(pair["query_difference_position"])
        require(0 <= query_pos < len(a_ids), f"{pair_id}: query difference position out of range.")

        differences = [i for i, (x, y) in enumerate(zip(a_ids, b_ids)) if x != y]
        require(differences == [query_pos],
                f"{pair_id}: twin prefixes should differ only at query position; got {differences}.")

        require(a_ids[query_pos] == int(pair["query_token_a"]),
                f"{pair_id}: twin A query token mismatch.")
        require(b_ids[query_pos] == int(pair["query_token_b"]),
                f"{pair_id}: twin B query token mismatch.")

        lookup[pair_id] = pair

    return lookup


def validate_schedule(
    raw: Any,
    pair_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    require(isinstance(raw, dict), "Schedule root must be a JSON object.")

    require(raw.get("experiment") == "DaveLM v0.9 Treatment #5",
            "Unexpected schedule experiment label.")
    require(raw.get("objective") == "Paired Counterfactual Query Binding",
            "Unexpected schedule objective.")
    require(raw.get("construction") == "same-record synthetic query twins",
            "Unexpected schedule construction.")
    require(raw.get("steps") == EXPECTED_STEPS, "Schedule step count metadata mismatch.")
    require(raw.get("batch_size") == EXPECTED_BATCH_SIZE, "Schedule batch size mismatch.")
    require(raw.get("pairs_per_batch") == EXPECTED_PAIRS_PER_BATCH,
            "Schedule pairs_per_batch mismatch.")
    require(raw.get("slot_0_presentations") == EXPECTED_SLOT0,
            "Schedule slot_0_presentations mismatch.")
    require(raw.get("slot_1_presentations") == EXPECTED_SLOT1,
            "Schedule slot_1_presentations mismatch.")
    require(raw.get("pair_exposure_min") == EXPECTED_PAIR_EXPOSURE_MIN,
            "Schedule pair_exposure_min mismatch.")
    require(raw.get("pair_exposure_max") == EXPECTED_PAIR_EXPOSURE_MAX,
            "Schedule pair_exposure_max mismatch.")

    steps_data = raw.get("steps_data")
    require(isinstance(steps_data, list), "schedule['steps_data'] must be a list.")
    require(len(steps_data) == EXPECTED_STEPS,
            f"Expected {EXPECTED_STEPS} steps, found {len(steps_data)}.")

    all_events: list[dict[str, Any]] = []
    slot_counts = Counter()
    pair_exposure = Counter()

    for expected_step, step in enumerate(steps_data, start=1):
        require(isinstance(step, dict), f"Step {expected_step} is not a JSON object.")
        require(step.get("step") == expected_step,
                f"Step numbering mismatch at expected step {expected_step}.")
        require(step.get("example_count") == EXPECTED_BATCH_SIZE,
                f"Step {expected_step}: example_count mismatch.")
        require(step.get("pair_count") == EXPECTED_PAIRS_PER_BATCH,
                f"Step {expected_step}: pair_count mismatch.")

        examples = step.get("examples")
        pair_ids = step.get("pair_ids")
        require(isinstance(examples, list) and len(examples) == EXPECTED_BATCH_SIZE,
                f"Step {expected_step}: expected 32 examples.")
        require(isinstance(pair_ids, list) and len(pair_ids) == EXPECTED_PAIRS_PER_BATCH,
                f"Step {expected_step}: expected 16 pair_ids.")
        require(len(set(pair_ids)) == EXPECTED_PAIRS_PER_BATCH,
                f"Step {expected_step}: duplicate pair IDs inside batch.")

        seen_slots: dict[str, set[int]] = defaultdict(set)

        for event in examples:
            require(isinstance(event, dict),
                    f"Step {expected_step}: schedule example is not a JSON object.")

            pair_id = event.get("pair_id")
            slot = event.get("query_slot")
            twin_name = event.get("twin")

            require(pair_id in pair_lookup,
                    f"Step {expected_step}: unknown pair_id {pair_id!r}.")
            require(pair_id in pair_ids,
                    f"Step {expected_step}: example pair_id missing from step pair_ids.")
            require(slot in (0, 1),
                    f"Step {expected_step} {pair_id}: invalid query_slot {slot}.")
            require(twin_name == ("A" if slot == 0 else "B"),
                    f"Step {expected_step} {pair_id}: twin/slot mismatch.")

            frozen_pair = pair_lookup[pair_id]
            frozen_twin = frozen_pair["twin_a" if slot == 0 else "twin_b"]

            # The schedule must carry the exact frozen prefix and answer metadata.
            require(event.get("input_token_ids") == frozen_twin.get("input_token_ids"),
                    f"Step {expected_step} {pair_id} slot {slot}: input_token_ids mismatch vs pair pool.")
            require(event.get("target_token_id") == frozen_twin.get("target_token_id"),
                    f"Step {expected_step} {pair_id} slot {slot}: target mismatch vs pair pool.")
            require(event.get("distractor_token_id") == frozen_twin.get("distractor_token_id"),
                    f"Step {expected_step} {pair_id} slot {slot}: distractor mismatch vs pair pool.")
            require(event.get("query_identity") == frozen_twin.get("query_identity"),
                    f"Step {expected_step} {pair_id} slot {slot}: query identity mismatch vs pair pool.")
            require(event.get("query_difference_position") == frozen_pair.get("query_difference_position"),
                    f"Step {expected_step} {pair_id} slot {slot}: query position mismatch vs pair pool.")

            ids = event["input_token_ids"]
            require(ids[0] == EXPECTED_BOS_ID,
                    f"Step {expected_step} {pair_id}: missing BOS token.")
            require(len(ids) == int(frozen_pair["prefix_length"]),
                    f"Step {expected_step} {pair_id}: prefix length mismatch.")
            require(int(event["target_token_id"]) != int(event["distractor_token_id"]),
                    f"Step {expected_step} {pair_id}: target == distractor.")

            seen_slots[pair_id].add(slot)
            slot_counts[slot] += 1

            clean_event = {
                "step": expected_step,
                "epoch": int(event["epoch"]),
                "pair_id": pair_id,
                "base_record_id": event["base_record_id"],
                "base_record_index": int(event["base_record_index"]),
                "query_slot": int(slot),
                "twin": twin_name,
                "query_identity": event["query_identity"],
                "query_difference_position": int(event["query_difference_position"]),
                "input_token_ids": [int(x) for x in ids],
                "target_token_id": int(event["target_token_id"]),
                "distractor_token_id": int(event["distractor_token_id"]),
                "candidate_pair": [int(x) for x in event["candidate_pair"]],
            }
            all_events.append(clean_event)

        require(set(seen_slots) == set(pair_ids),
                f"Step {expected_step}: pair IDs in examples do not match pair_ids metadata.")

        for pair_id in pair_ids:
            require(seen_slots[pair_id] == {0, 1},
                    f"Step {expected_step}: pair {pair_id} is not a complete A/B twin pair.")
            pair_exposure[pair_id] += 1

    require(len(all_events) == EXPECTED_EXAMPLES,
            f"Expected {EXPECTED_EXAMPLES} events, got {len(all_events)}.")
    require(slot_counts[0] == EXPECTED_SLOT0,
            f"Expected {EXPECTED_SLOT0} slot-0 events, got {slot_counts[0]}.")
    require(slot_counts[1] == EXPECTED_SLOT1,
            f"Expected {EXPECTED_SLOT1} slot-1 events, got {slot_counts[1]}.")
    require(len(pair_exposure) == EXPECTED_PAIR_COUNT,
            f"Expected all {EXPECTED_PAIR_COUNT} pairs to appear in the schedule.")
    require(min(pair_exposure.values()) == EXPECTED_PAIR_EXPOSURE_MIN,
            "Observed minimum pair exposure mismatch.")
    require(max(pair_exposure.values()) == EXPECTED_PAIR_EXPOSURE_MAX,
            "Observed maximum pair exposure mismatch.")

    return all_events


# ============================================================================
# MODEL LOADING
# ============================================================================

def import_original_run() -> Any:
    require(SOURCE_ROOT.exists(), f"Protected source root not found: {SOURCE_ROOT}")

    source_root_str = str(SOURCE_ROOT)
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)

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
            'Original experiment module does not expose build_model("untied").')

    model = original_run.build_model("untied").to(device)

    parameter_count = sum(p.numel() for p in model.parameters())
    require(parameter_count == EXPECTED_PARAMETER_COUNT,
            f"Parameter count mismatch: expected {EXPECTED_PARAMETER_COUNT:,}, got {parameter_count:,}.")

    checkpoint = torch.load(FINAL_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state = extract_model_state(checkpoint)

    incompatible = model.load_state_dict(state, strict=False)
    require(not incompatible.missing_keys,
            f"Missing checkpoint keys: {incompatible.missing_keys}")
    require(not incompatible.unexpected_keys,
            f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")

    model.eval()

    # Belt-and-suspenders read-only guard.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def extract_logits(model_output: Any) -> torch.Tensor:
    if torch.is_tensor(model_output):
        logits = model_output
    elif isinstance(model_output, (tuple, list)) and model_output and torch.is_tensor(model_output[0]):
        logits = model_output[0]
    elif isinstance(model_output, dict) and torch.is_tensor(model_output.get("logits")):
        logits = model_output["logits"]
    elif hasattr(model_output, "logits") and torch.is_tensor(model_output.logits):
        logits = model_output.logits
    else:
        raise RuntimeError(f"Could not extract logits from model output type {type(model_output).__name__}.")

    require(logits.ndim == 3,
            f"Expected logits shaped [batch, time, vocab], got {tuple(logits.shape)}.")
    return logits


# ============================================================================
# METRICS
# ============================================================================

class Metrics:
    def __init__(self) -> None:
        self.count = 0
        self.top1 = 0
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
        pair_lse = torch.logsumexp(
            torch.stack((target_logits, distractor_logits), dim=-1),
            dim=-1,
        )

        membership_loss = full_lse - pair_lse
        selector_loss = F.relu(MARGIN - delta)
        replacement_loss = membership_loss + selector_loss

        probabilities = torch.softmax(answer_logits, dim=-1)
        target_probability = probabilities[rows, target_ids]
        distractor_probability = probabilities[rows, distractor_ids]
        candidate_mass = target_probability + distractor_probability

        predicted_ids = answer_logits.argmax(dim=-1)
        ranks = 1 + (answer_logits > target_logits.unsqueeze(-1)).sum(dim=-1)

        self.count += batch_size
        self.top1 += int((predicted_ids == target_ids).sum().item())
        self.correct_gt_distractor += int((delta > 0).sum().item())
        self.margin_satisfied += int((delta >= MARGIN).sum().item())

        self.sum_candidate_mass += float(candidate_mass.sum().item())
        self.sum_membership_loss += float(membership_loss.sum().item())
        self.sum_selector_loss += float(selector_loss.sum().item())
        self.sum_replacement_loss += float(replacement_loss.sum().item())
        self.sum_target_probability += float(target_probability.sum().item())
        self.sum_distractor_probability += float(distractor_probability.sum().item())
        self.sum_logit_delta += float(delta.sum().item())

        self.ranks.extend(int(rank) for rank in ranks.detach().cpu().tolist())

    def summary(self) -> dict[str, Any]:
        require(self.count > 0, "Cannot summarize zero events.")

        return {
            "count": self.count,
            "full_vocab_answer_top1": self.top1 / self.count,
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


# ============================================================================
# RETENTION AUDIT
# ============================================================================

def run_retention_audit(
    model: torch.nn.Module,
    events: list[dict[str, Any]],
    device: torch.device,
    inference_batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    overall = Metrics()
    slot0 = Metrics()
    slot1 = Metrics()

    # Frozen prefixes have several lengths. Grouping only avoids padding.
    # It does not regenerate, alter, or resample any event.
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        buckets[len(event["input_token_ids"])].append(event)

    processed = 0
    next_progress = 3200

    with torch.inference_mode():
        for prefix_length in sorted(buckets):
            bucket = buckets[prefix_length]

            for start in range(0, len(bucket), inference_batch_size):
                batch = bucket[start:start + inference_batch_size]

                inputs = torch.tensor(
                    [event["input_token_ids"] for event in batch],
                    dtype=torch.long,
                    device=device,
                )
                target_ids = torch.tensor(
                    [event["target_token_id"] for event in batch],
                    dtype=torch.long,
                    device=device,
                )
                distractor_ids = torch.tensor(
                    [event["distractor_token_id"] for event in batch],
                    dtype=torch.long,
                    device=device,
                )
                slots = torch.tensor(
                    [event["query_slot"] for event in batch],
                    dtype=torch.long,
                    device=device,
                )

                require(inputs.shape[1] == prefix_length,
                        "Internal prefix-length bucket mismatch.")

                logits = extract_logits(model(inputs))

                # input_token_ids end at the native "Answer:" prefix.
                # The final causal logit therefore predicts the frozen target answer token.
                answer_logits = logits[:, -1, :]

                overall.add(answer_logits, target_ids, distractor_ids)

                mask0 = slots == 0
                if bool(mask0.any()):
                    slot0.add(answer_logits[mask0], target_ids[mask0], distractor_ids[mask0])

                mask1 = slots == 1
                if bool(mask1.any()):
                    slot1.add(answer_logits[mask1], target_ids[mask1], distractor_ids[mask1])

                processed += len(batch)
                while processed >= next_progress and next_progress <= EXPECTED_EXAMPLES:
                    print(f"  audited at least {next_progress:>5}/{EXPECTED_EXAMPLES} events")
                    next_progress += 3200

    require(overall.count == EXPECTED_EXAMPLES,
            f"Overall event count mismatch: {overall.count}")
    require(slot0.count == EXPECTED_SLOT0,
            f"Slot-0 event count mismatch: {slot0.count}")
    require(slot1.count == EXPECTED_SLOT1,
            f"Slot-1 event count mismatch: {slot1.count}")

    return overall.summary(), slot0.summary(), slot1.summary()


def print_summary(title: str, summary: dict[str, Any]) -> None:
    print(title)
    print(f"  events:                    {summary['count']}")
    print(f"  full-vocab answer top1:    {summary['full_vocab_answer_top1']:.6f}")
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


def classify_retention(overall: dict[str, Any]) -> tuple[str, str]:
    if (
        overall["correct_gt_distractor"] >= 0.95
        and overall["margin_ge_0_5"] >= 0.90
    ):
        return (
            "STRONG_FINAL_TRAINING_RETENTION",
            "Retention is strong. The same pre-authorized T4 positive-control gates may now "
            "be evaluated unchanged. Novel/shortcut pools remain sealed unless both gates pass.",
        )

    return (
        "TRAINING_TASK_FAILURE_OR_INCOMPLETE_RETENTION",
        "Do NOT run positive controls yet. The final checkpoint did not demonstrate strong "
        "retention of the paired counterfactual training task under the frozen T5 objective. "
        "Stop here and interpret the training result before designing any future treatment.",
    )


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
        help="Inference batch size within equal-length prefix buckets. Default: 128",
    )
    args = parser.parse_args()

    require(args.batch_size >= 1, "--batch-size must be at least 1.")

    banner("TREATMENT #5 FINAL TRAINING-RETENTION AUDIT - READ ONLY")

    banner("VERIFYING FROZEN ARTIFACTS")
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

    banner("VALIDATING EXACT FROZEN PAIR POOL AND SCHEDULE")

    pair_pool_raw = load_json(PAIR_POOL_PATH)
    schedule_raw = load_json(SCHEDULE_PATH)

    pair_lookup = validate_pair_pool(pair_pool_raw)
    events = validate_schedule(schedule_raw, pair_lookup)

    prefix_lengths = sorted({len(event["input_token_ids"]) for event in events})

    print(f"Verified pair pool:          {len(pair_lookup)} pairs")
    print(f"Frozen schedule:             {EXPECTED_STEPS} steps")
    print(f"Exact scheduled events:      {len(events)}")
    print(f"Slot-0 events:               {sum(e['query_slot'] == 0 for e in events)}")
    print(f"Slot-1 events:               {sum(e['query_slot'] == 1 for e in events)}")
    print(f"Distinct prefix lengths:     {len(prefix_lengths)}")
    print(f"Prefix length range:         {min(prefix_lengths)}-{max(prefix_lengths)}")
    print(f"Answer replacements:         {EXPECTED_ANSWER_REPLACEMENTS}")
    print(f"Total supervised positions:  {EXPECTED_TOTAL_SUPERVISED_POSITIONS}")
    print(f"Ordinary non-answer CE:      {EXPECTED_NONANSWER_CE_POSITIONS}")
    print()
    print("Every scheduled example exactly matches its frozen pair-pool twin: PASS")
    print("Every batch contains 16 complete A/B pairs: PASS")
    print("Exact frozen schedule validation: PASS")

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

    overall, slot0, slot1 = run_retention_audit(
        model=model,
        events=events,
        device=device,
        inference_batch_size=args.batch_size,
    )

    banner("FINAL T5 TRAINING-RETENTION RESULTS")
    print_summary("OVERALL", overall)
    print_summary("SLOT 0", slot0)
    print_summary("SLOT 1", slot1)

    classification, next_step = classify_retention(overall)

    print(f"CLASSIFICATION: {classification}")
    print()
    print(next_step)

    result = {
        "audit": "DaveLM v0.9 Treatment #5 final training-retention audit",
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
        },
        "audit_input_semantics": {
            "source": "exact schedule steps_data[*].examples[*].input_token_ids",
            "answer_logit_position": "final logit of each frozen Answer: prefix",
            "schedule_examples_cross_checked_against_pair_pool": True,
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
