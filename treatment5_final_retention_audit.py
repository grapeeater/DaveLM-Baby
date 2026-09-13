from __future__ import annotations

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


# =============================================================================
# FROZEN T5 CONTRACT
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

CHECKPOINT_PATH = (
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

EXPECTED_PAIR_POOL_SHA = (
    "0b31e39361f7a45dc4182c49ab4b4967bffe539cd2ff12e7b196eabfb3dac307"
)

EXPECTED_SCHEDULE_SHA = (
    "a4e64fd9d1b934aa440a5d036a3bec0ac7c9ac97d58c72fb45df6d8576bb7c12"
)

EXPECTED_CHECKPOINT_SHA = (
    "cfe1e99f17c02ee7fc7ab6869eb367960d4e98fc477bb9d22173a978361f077a"
)

EXPECTED_PAIR_COUNT = 1536
EXPECTED_STEPS = 1000
EXPECTED_BATCH_SIZE = 32
EXPECTED_PAIRS_PER_BATCH = 16

EXPECTED_EXAMPLES = 32000
EXPECTED_SLOT0 = 16000
EXPECTED_SLOT1 = 16000

EXPECTED_DOCUMENT_LENGTH = 193
EXPECTED_SUPERVISED_PER_DOCUMENT = 192
EXPECTED_NONANSWER_PER_DOCUMENT = 191

EXPECTED_TOTAL_SUPERVISED = 6_144_000
EXPECTED_ANSWER_REPLACEMENTS = 32_000
EXPECTED_NONANSWER_CE = 6_112_000

EXPECTED_PAIR_EXPOSURE_MIN = 10
EXPECTED_PAIR_EXPOSURE_MAX = 11

EXPECTED_PARAMETER_COUNT = 10_594_944
EXPECTED_BOS_ID = 2

MARGIN = 0.5

STRONG_PAIR_WIN_GATE = 0.95
STRONG_MARGIN_GATE = 0.90


# =============================================================================
# HELPERS
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
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def verify_sha(path: Path, expected: str, label: str) -> str:
    require(path.exists(), f"Missing {label}: {path}")

    actual = sha256_file(path)

    print(f"{label}:")
    print(f"  observed: {actual}")
    print(f"  expected: {expected}")

    require(actual == expected, f"{label} SHA256 mismatch.")

    print("  PASS")
    print()

    return actual


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


# =============================================================================
# PAIR POOL VALIDATION
# =============================================================================

def validate_pair_pool(raw: Any) -> dict[str, dict[str, Any]]:
    require(isinstance(raw, dict), "Pair-pool root must be an object.")

    pairs = raw.get("pairs")

    require(isinstance(pairs, list), "Pair-pool missing 'pairs'.")
    require(
        len(pairs) == EXPECTED_PAIR_COUNT,
        f"Expected {EXPECTED_PAIR_COUNT} pairs, got {len(pairs)}.",
    )

    lookup: dict[str, dict[str, Any]] = {}

    for index, pair in enumerate(pairs):
        require(isinstance(pair, dict), f"Pair {index} is not an object.")

        pair_id = pair["pair_id"]

        require(isinstance(pair_id, str), f"Pair {index}: invalid pair_id.")
        require(pair_id not in lookup, f"Duplicate pair_id: {pair_id}")

        answer_index = int(pair["answer_token_index"])
        answer_causal_position = int(pair["answer_causal_position"])
        query_position = int(pair["query_difference_position"])

        require(
            answer_causal_position == answer_index - 1,
            f"{pair_id}: answer causal position != answer index - 1.",
        )

        require(
            int(pair["model_visible_document_length"]) == EXPECTED_DOCUMENT_LENGTH,
            f"{pair_id}: model-visible length mismatch.",
        )

        require(
            int(pair["supervised_token_count"]) == EXPECTED_SUPERVISED_PER_DOCUMENT,
            f"{pair_id}: supervised-token count mismatch.",
        )

        require(
            int(pair["nonanswer_supervised_token_count"])
            == EXPECTED_NONANSWER_PER_DOCUMENT,
            f"{pair_id}: non-answer count mismatch.",
        )

        twin_a = pair["twin_a"]
        twin_b = pair["twin_b"]

        require(twin_a["query_slot"] == 0, f"{pair_id}: twin A is not slot 0.")
        require(twin_b["query_slot"] == 1, f"{pair_id}: twin B is not slot 1.")

        a_target = int(twin_a["target_token_id"])
        a_distractor = int(twin_a["distractor_token_id"])

        b_target = int(twin_b["target_token_id"])
        b_distractor = int(twin_b["distractor_token_id"])

        require(a_target != a_distractor, f"{pair_id}: A target == distractor.")
        require(b_target != b_distractor, f"{pair_id}: B target == distractor.")

        require(
            a_target == b_distractor,
            f"{pair_id}: A target != B distractor.",
        )

        require(
            b_target == a_distractor,
            f"{pair_id}: B target != A distractor.",
        )

        require(
            set(int(x) for x in pair["candidate_pair"])
            == {a_target, a_distractor},
            f"{pair_id}: candidate pair mismatch.",
        )

        a_prefix = [int(x) for x in twin_a["prefix_token_ids"]]
        b_prefix = [int(x) for x in twin_b["prefix_token_ids"]]

        a_doc = [int(x) for x in twin_a["full_document_token_ids"]]
        b_doc = [int(x) for x in twin_b["full_document_token_ids"]]

        require(
            len(a_doc) == EXPECTED_DOCUMENT_LENGTH,
            f"{pair_id}: A document length mismatch.",
        )

        require(
            len(b_doc) == EXPECTED_DOCUMENT_LENGTH,
            f"{pair_id}: B document length mismatch.",
        )

        require(a_doc[0] == EXPECTED_BOS_ID, f"{pair_id}: A missing BOS.")
        require(b_doc[0] == EXPECTED_BOS_ID, f"{pair_id}: B missing BOS.")

        require(
            len(a_prefix) == answer_index,
            f"{pair_id}: A prefix length != answer index.",
        )

        require(
            len(b_prefix) == answer_index,
            f"{pair_id}: B prefix length != answer index.",
        )

        require(
            a_doc[:answer_index] == a_prefix,
            f"{pair_id}: A prefix/full-document mismatch.",
        )

        require(
            b_doc[:answer_index] == b_prefix,
            f"{pair_id}: B prefix/full-document mismatch.",
        )

        require(
            a_doc[answer_index] == a_target,
            f"{pair_id}: A answer token != target.",
        )

        require(
            b_doc[answer_index] == b_target,
            f"{pair_id}: B answer token != target.",
        )

        prefix_diffs = [
            i
            for i, (a, b) in enumerate(zip(a_prefix, b_prefix))
            if a != b
        ]

        require(
            prefix_diffs == [query_position],
            f"{pair_id}: prefix twins should differ only at query position; "
            f"got {prefix_diffs}.",
        )

        doc_diffs = [
            i
            for i, (a, b) in enumerate(zip(a_doc, b_doc))
            if a != b
        ]

        require(
            doc_diffs == [query_position, answer_index],
            f"{pair_id}: full twins should differ at query + answer only; "
            f"got {doc_diffs}.",
        )

        lookup[pair_id] = {
            "pair_id": pair_id,
            "query_difference_position": query_position,
            "answer_token_index": answer_index,
            "answer_causal_position": answer_causal_position,
            "slot0": {
                "full_document_token_ids": a_doc,
                "target_token_id": a_target,
                "distractor_token_id": a_distractor,
            },
            "slot1": {
                "full_document_token_ids": b_doc,
                "target_token_id": b_target,
                "distractor_token_id": b_distractor,
            },
        }

    require(
        len(lookup) == EXPECTED_PAIR_COUNT,
        "Pair lookup count mismatch.",
    )

    return lookup


# =============================================================================
# SCHEDULE VALIDATION
# =============================================================================

def validate_schedule(
    raw: Any,
    pair_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    require(isinstance(raw, dict), "Schedule root must be an object.")

    steps = raw["steps_data"]

    require(isinstance(steps, list), "'steps_data' must be a list.")
    require(
        len(steps) == EXPECTED_STEPS,
        f"Expected {EXPECTED_STEPS} steps, got {len(steps)}.",
    )

    events: list[dict[str, Any]] = []

    slot_counts = Counter()
    pair_exposure = Counter()

    for step_number, step in enumerate(steps, start=1):
        require(
            int(step["step"]) == step_number,
            f"Step number mismatch at step {step_number}.",
        )

        examples = step["examples"]

        require(
            isinstance(examples, list),
            f"Step {step_number}: examples is not a list.",
        )

        require(
            len(examples) == EXPECTED_BATCH_SIZE,
            f"Step {step_number}: expected 32 examples, got {len(examples)}.",
        )

        seen_slots: dict[str, set[int]] = defaultdict(set)

        for example in examples:
            pair_id = example["pair_id"]
            slot = int(example["query_slot"])

            require(
                pair_id in pair_lookup,
                f"Step {step_number}: unknown pair_id {pair_id}.",
            )

            require(
                slot in (0, 1),
                f"Step {step_number} {pair_id}: invalid slot.",
            )

            frozen = pair_lookup[pair_id][f"slot{slot}"]

            schedule_doc = [
                int(x)
                for x in example["full_document_token_ids"]
            ]

            target = int(example["target_token_id"])
            distractor = int(example["distractor_token_id"])

            answer_index = int(example["answer_token_index"])
            answer_causal_position = int(example["answer_causal_position"])

            require(
                len(schedule_doc) == EXPECTED_DOCUMENT_LENGTH,
                f"Step {step_number} {pair_id}: document length mismatch.",
            )

            require(
                schedule_doc == frozen["full_document_token_ids"],
                f"Step {step_number} {pair_id} slot {slot}: "
                f"schedule document != pair-pool document.",
            )

            require(
                target == frozen["target_token_id"],
                f"Step {step_number} {pair_id}: target mismatch.",
            )

            require(
                distractor == frozen["distractor_token_id"],
                f"Step {step_number} {pair_id}: distractor mismatch.",
            )

            require(
                answer_index == pair_lookup[pair_id]["answer_token_index"],
                f"Step {step_number} {pair_id}: answer index mismatch.",
            )

            require(
                answer_causal_position
                == pair_lookup[pair_id]["answer_causal_position"],
                f"Step {step_number} {pair_id}: answer causal position mismatch.",
            )

            require(
                schedule_doc[answer_index] == target,
                f"Step {step_number} {pair_id}: answer token != target.",
            )

            require(
                int(example["model_visible_document_length"])
                == EXPECTED_DOCUMENT_LENGTH,
                f"Step {step_number} {pair_id}: model-visible length mismatch.",
            )

            require(
                int(example["supervised_token_count"])
                == EXPECTED_SUPERVISED_PER_DOCUMENT,
                f"Step {step_number} {pair_id}: supervised count mismatch.",
            )

            require(
                int(example["nonanswer_supervised_token_count"])
                == EXPECTED_NONANSWER_PER_DOCUMENT,
                f"Step {step_number} {pair_id}: non-answer count mismatch.",
            )

            seen_slots[pair_id].add(slot)
            slot_counts[slot] += 1

            events.append(
                {
                    "step": step_number,
                    "pair_id": pair_id,
                    "query_slot": slot,
                    "full_document_token_ids": schedule_doc,
                    "answer_token_index": answer_index,
                    "answer_causal_position": answer_causal_position,
                    "target_token_id": target,
                    "distractor_token_id": distractor,
                }
            )

        require(
            len(seen_slots) == EXPECTED_PAIRS_PER_BATCH,
            f"Step {step_number}: expected 16 distinct pairs, "
            f"got {len(seen_slots)}.",
        )

        for pair_id, slots in seen_slots.items():
            require(
                slots == {0, 1},
                f"Step {step_number}: {pair_id} is not a complete twin pair.",
            )

            pair_exposure[pair_id] += 1

    require(
        len(events) == EXPECTED_EXAMPLES,
        f"Expected 32,000 examples, got {len(events)}.",
    )

    require(
        slot_counts[0] == EXPECTED_SLOT0,
        f"Expected 16,000 slot-0 examples, got {slot_counts[0]}.",
    )

    require(
        slot_counts[1] == EXPECTED_SLOT1,
        f"Expected 16,000 slot-1 examples, got {slot_counts[1]}.",
    )

    require(
        len(pair_exposure) == EXPECTED_PAIR_COUNT,
        f"Expected all {EXPECTED_PAIR_COUNT} pairs in schedule, "
        f"got {len(pair_exposure)}.",
    )

    require(
        min(pair_exposure.values()) == EXPECTED_PAIR_EXPOSURE_MIN,
        f"Minimum pair exposure was {min(pair_exposure.values())}, expected 10.",
    )

    require(
        max(pair_exposure.values()) == EXPECTED_PAIR_EXPOSURE_MAX,
        f"Maximum pair exposure was {max(pair_exposure.values())}, expected 11.",
    )

    total_supervised = (
        len(events)
        * EXPECTED_SUPERVISED_PER_DOCUMENT
    )

    require(
        total_supervised == EXPECTED_TOTAL_SUPERVISED,
        "Total supervised-position count mismatch.",
    )

    require(
        EXPECTED_TOTAL_SUPERVISED
        - EXPECTED_ANSWER_REPLACEMENTS
        == EXPECTED_NONANSWER_CE,
        "Non-answer CE count mismatch.",
    )

    return events


# =============================================================================
# MODEL LOADING
# =============================================================================

def import_original_run() -> Any:
    require(
        SOURCE_ROOT.exists(),
        f"Protected source root missing: {SOURCE_ROOT}",
    )

    source_string = str(SOURCE_ROOT)

    if source_string not in sys.path:
        sys.path.insert(0, source_string)

    return importlib.import_module(
        "experiments.two_mapping_contextual_binding.run"
    )


def extract_model_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    require(
        isinstance(checkpoint, dict),
        "Checkpoint root is not a dictionary.",
    )

    for key in (
        "model_state",
        "model_state_dict",
        "model",
    ):
        value = checkpoint.get(key)

        if (
            isinstance(value, dict)
            and value
            and all(torch.is_tensor(v) for v in value.values())
        ):
            return value

    if (
        checkpoint
        and all(torch.is_tensor(v) for v in checkpoint.values())
    ):
        return checkpoint

    raise RuntimeError(
        "Could not locate model state in final checkpoint."
    )


def load_model(device: torch.device) -> torch.nn.Module:
    original_run = import_original_run()

    require(
        hasattr(original_run, "build_model"),
        'Original module does not expose build_model("untied").',
    )

    model = original_run.build_model("untied").to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    require(
        parameter_count == EXPECTED_PARAMETER_COUNT,
        f"Expected {EXPECTED_PARAMETER_COUNT:,} parameters, "
        f"got {parameter_count:,}.",
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state = extract_model_state(checkpoint)

    incompatible = model.load_state_dict(
        state,
        strict=False,
    )

    require(
        not incompatible.missing_keys,
        f"Missing model keys: {incompatible.missing_keys}",
    )

    require(
        not incompatible.unexpected_keys,
        f"Unexpected model keys: {incompatible.unexpected_keys}",
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def extract_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        logits = output

    elif (
        isinstance(output, (tuple, list))
        and output
        and torch.is_tensor(output[0])
    ):
        logits = output[0]

    elif (
        isinstance(output, dict)
        and torch.is_tensor(output.get("logits"))
    ):
        logits = output["logits"]

    elif (
        hasattr(output, "logits")
        and torch.is_tensor(output.logits)
    ):
        logits = output.logits

    else:
        raise RuntimeError(
            f"Could not extract logits from "
            f"{type(output).__name__}."
        )

    require(
        logits.ndim == 3,
        f"Expected logits [batch,time,vocab], "
        f"got {tuple(logits.shape)}.",
    )

    return logits


# =============================================================================
# METRICS
# =============================================================================

class Metrics:
    def __init__(self) -> None:
        self.count = 0

        self.top1 = 0
        self.top5 = 0
        self.pair_win = 0
        self.margin_pass = 0

        self.candidate_mass_sum = 0.0
        self.membership_loss_sum = 0.0
        self.selector_loss_sum = 0.0
        self.replacement_loss_sum = 0.0

        self.target_probability_sum = 0.0
        self.distractor_probability_sum = 0.0

        self.logit_delta_sum = 0.0

        self.ranks: list[int] = []

    def add(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        distractor_ids: torch.Tensor,
    ) -> None:
        batch_size = logits.shape[0]

        rows = torch.arange(
            batch_size,
            device=logits.device,
        )

        target_logits = logits[
            rows,
            target_ids,
        ]

        distractor_logits = logits[
            rows,
            distractor_ids,
        ]

        delta = (
            target_logits
            - distractor_logits
        )

        full_lse = torch.logsumexp(
            logits,
            dim=-1,
        )

        candidate_lse = torch.logsumexp(
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
            - candidate_lse
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
            logits,
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

        top1_ids = logits.argmax(
            dim=-1
        )

        top5_ids = torch.topk(
            logits,
            k=5,
            dim=-1,
        ).indices

        top5_hits = (
            top5_ids
            == target_ids.unsqueeze(-1)
        ).any(dim=-1)

        ranks = (
            1
            + (
                logits
                > target_logits.unsqueeze(-1)
            ).sum(dim=-1)
        )

        self.count += batch_size

        self.top1 += int(
            (
                top1_ids
                == target_ids
            ).sum().item()
        )

        self.top5 += int(
            top5_hits.sum().item()
        )

        self.pair_win += int(
            (
                delta
                > 0
            ).sum().item()
        )

        self.margin_pass += int(
            (
                delta
                >= MARGIN
            ).sum().item()
        )

        self.candidate_mass_sum += float(
            candidate_mass.sum().item()
        )

        self.membership_loss_sum += float(
            membership_loss.sum().item()
        )

        self.selector_loss_sum += float(
            selector_loss.sum().item()
        )

        self.replacement_loss_sum += float(
            replacement_loss.sum().item()
        )

        self.target_probability_sum += float(
            target_probability.sum().item()
        )

        self.distractor_probability_sum += float(
            distractor_probability.sum().item()
        )

        self.logit_delta_sum += float(
            delta.sum().item()
        )

        self.ranks.extend(
            int(x)
            for x in ranks.detach().cpu().tolist()
        )

    def summary(self) -> dict[str, Any]:
        require(
            self.count > 0,
            "Cannot summarize zero events.",
        )

        return {
            "count": self.count,
            "full_vocab_top1": self.top1 / self.count,
            "full_vocab_top5": self.top5 / self.count,
            "correct_gt_distractor": self.pair_win / self.count,
            "margin_ge_0_5": self.margin_pass / self.count,
            "candidate_mass": self.candidate_mass_sum / self.count,
            "membership_loss": self.membership_loss_sum / self.count,
            "selector_loss": self.selector_loss_sum / self.count,
            "replacement_loss": self.replacement_loss_sum / self.count,
            "target_probability": self.target_probability_sum / self.count,
            "distractor_probability": (
                self.distractor_probability_sum
                / self.count
            ),
            "mean_logit_delta": self.logit_delta_sum / self.count,
            "target_rank_mean": statistics.fmean(self.ranks),
            "target_rank_median": statistics.median(self.ranks),
        }


# =============================================================================
# READ-ONLY REPLAY
# =============================================================================

def run_audit(
    model: torch.nn.Module,
    events: list[dict[str, Any]],
    device: torch.device,
    audit_batch_size: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    overall = Metrics()
    slot0 = Metrics()
    slot1 = Metrics()

    processed = 0

    with torch.inference_mode():
        for start in range(
            0,
            len(events),
            audit_batch_size,
        ):
            batch = events[
                start:
                start + audit_batch_size
            ]

            documents = torch.tensor(
                [
                    event["full_document_token_ids"]
                    for event in batch
                ],
                dtype=torch.long,
                device=device,
            )

            # Exact causal training form:
            # input positions = document[:-1]
            inputs = documents[:, :-1]

            target_ids = torch.tensor(
                [
                    event["target_token_id"]
                    for event in batch
                ],
                dtype=torch.long,
                device=device,
            )

            distractor_ids = torch.tensor(
                [
                    event["distractor_token_id"]
                    for event in batch
                ],
                dtype=torch.long,
                device=device,
            )

            answer_positions = torch.tensor(
                [
                    event["answer_causal_position"]
                    for event in batch
                ],
                dtype=torch.long,
                device=device,
            )

            slots = torch.tensor(
                [
                    event["query_slot"]
                    for event in batch
                ],
                dtype=torch.long,
                device=device,
            )

            output = model(inputs)

            logits = extract_logits(output)

            require(
                logits.shape[1]
                == EXPECTED_SUPERVISED_PER_DOCUMENT,
                f"Expected {EXPECTED_SUPERVISED_PER_DOCUMENT} "
                f"causal output positions, got {logits.shape[1]}.",
            )

            rows = torch.arange(
                len(batch),
                device=device,
            )

            answer_logits = logits[
                rows,
                answer_positions,
                :,
            ]

            overall.add(
                answer_logits,
                target_ids,
                distractor_ids,
            )

            mask0 = slots == 0

            if bool(mask0.any()):
                slot0.add(
                    answer_logits[mask0],
                    target_ids[mask0],
                    distractor_ids[mask0],
                )

            mask1 = slots == 1

            if bool(mask1.any()):
                slot1.add(
                    answer_logits[mask1],
                    target_ids[mask1],
                    distractor_ids[mask1],
                )

            processed += len(batch)

            if (
                processed % 3200 == 0
                or processed == EXPECTED_EXAMPLES
            ):
                print(
                    f"audited "
                    f"{processed:>5}/"
                    f"{EXPECTED_EXAMPLES}"
                )

    require(
        overall.count == EXPECTED_EXAMPLES,
        "Overall event count mismatch.",
    )

    require(
        slot0.count == EXPECTED_SLOT0,
        "Slot-0 count mismatch.",
    )

    require(
        slot1.count == EXPECTED_SLOT1,
        "Slot-1 count mismatch.",
    )

    return (
        overall.summary(),
        slot0.summary(),
        slot1.summary(),
    )


def print_metrics(
    title: str,
    metrics: dict[str, Any],
) -> None:
    print(title)

    print(
        f"  events:                  "
        f"{metrics['count']}"
    )

    print(
        f"  full-vocab top1:         "
        f"{metrics['full_vocab_top1']:.6f}"
    )

    print(
        f"  full-vocab top5:         "
        f"{metrics['full_vocab_top5']:.6f}"
    )

    print(
        f"  correct > distractor:    "
        f"{metrics['correct_gt_distractor']:.6f}"
    )

    print(
        f"  margin >= 0.5:           "
        f"{metrics['margin_ge_0_5']:.6f}"
    )

    print(
        f"  candidate mass:          "
        f"{metrics['candidate_mass']:.6f}"
    )

    print(
        f"  membership loss:         "
        f"{metrics['membership_loss']:.6f}"
    )

    print(
        f"  selector loss:           "
        f"{metrics['selector_loss']:.6f}"
    )

    print(
        f"  replacement loss:        "
        f"{metrics['replacement_loss']:.6f}"
    )

    print(
        f"  target probability:      "
        f"{metrics['target_probability']:.6f}"
    )

    print(
        f"  distractor probability:  "
        f"{metrics['distractor_probability']:.6f}"
    )

    print(
        f"  mean logit delta:        "
        f"{metrics['mean_logit_delta']:+.6f}"
    )

    print(
        f"  target rank mean:        "
        f"{metrics['target_rank_mean']:.6f}"
    )

    print(
        f"  target rank median:      "
        f"{metrics['target_rank_median']:.6f}"
    )

    print()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    require(
        args.batch_size >= 1,
        "--batch-size must be at least 1.",
    )

    banner(
        "TREATMENT #5 FINAL TRAINING-RETENTION AUDIT"
    )

    banner(
        "1. VERIFY FROZEN HASHES"
    )

    schedule_sha = verify_sha(
        SCHEDULE_PATH,
        EXPECTED_SCHEDULE_SHA,
        "Frozen schedule",
    )

    pair_pool_sha = verify_sha(
        PAIR_POOL_PATH,
        EXPECTED_PAIR_POOL_SHA,
        "Pair pool",
    )

    checkpoint_sha = verify_sha(
        CHECKPOINT_PATH,
        EXPECTED_CHECKPOINT_SHA,
        "Final checkpoint",
    )

    banner(
        "2. VALIDATE EXACT PAIR POOL + FROZEN SCHEDULE"
    )

    pair_pool_raw = load_json(
        PAIR_POOL_PATH
    )

    schedule_raw = load_json(
        SCHEDULE_PATH
    )

    pair_lookup = validate_pair_pool(
        pair_pool_raw
    )

    events = validate_schedule(
        schedule_raw,
        pair_lookup,
    )

    answer_indices = sorted(
        {
            pair["answer_token_index"]
            for pair in pair_lookup.values()
        }
    )

    answer_causal_positions = sorted(
        {
            pair["answer_causal_position"]
            for pair in pair_lookup.values()
        }
    )

    query_positions = sorted(
        {
            pair["query_difference_position"]
            for pair in pair_lookup.values()
        }
    )

    print(
        f"Pairs:                     "
        f"{len(pair_lookup)}"
    )

    print(
        f"Scheduled events:          "
        f"{len(events)}"
    )

    print(
        f"Slot 0 events:             "
        f"{sum(e['query_slot'] == 0 for e in events)}"
    )

    print(
        f"Slot 1 events:             "
        f"{sum(e['query_slot'] == 1 for e in events)}"
    )

    print(
        f"Query token positions:     "
        f"{query_positions}"
    )

    print(
        f"Answer token indices:      "
        f"{answer_indices}"
    )

    print(
        f"Answer causal positions:   "
        f"{answer_causal_positions}"
    )

    print(
        f"Total supervised:          "
        f"{EXPECTED_TOTAL_SUPERVISED}"
    )

    print(
        f"Answer replacements:       "
        f"{EXPECTED_ANSWER_REPLACEMENTS}"
    )

    print(
        f"Ordinary non-answer CE:    "
        f"{EXPECTED_NONANSWER_CE}"
    )

    print()

    print(
        "Pair-pool structural audit: PASS"
    )

    print(
        "Schedule/pair-pool exact document cross-check: PASS"
    )

    print(
        "16 complete twin pairs per batch: PASS"
    )

    print(
        "Pair exposure 10-11: PASS"
    )

    banner(
        "3. LOAD FINAL BABY CHECKPOINT"
    )

    if args.device == "cuda":
        require(
            torch.cuda.is_available(),
            "CUDA/ROCm requested but unavailable.",
        )

    device = torch.device(
        args.device
    )

    model = load_model(
        device
    )

    print(
        f"Device:      {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU:         "
            f"{torch.cuda.get_device_name(device)}"
        )

    print(
        f"Parameters:  "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    print(
        'Builder:     '
        'original_run.build_model("untied")'
    )

    print(
        "Mode:        model.eval()"
    )

    print(
        "Inference:   torch.inference_mode()"
    )

    print(
        "Optimizer:   NONE"
    )

    print(
        "Backward:    NONE"
    )

    print(
        "Training:    NONE"
    )

    print(
        "PCs:         NOT RUN"
    )

    print(
        "Sealed eval: UNTOUCHED"
    )

    banner(
        "4. REPLAY ALL 32,000 FROZEN TRAINING EVENTS"
    )

    overall, slot0, slot1 = run_audit(
        model=model,
        events=events,
        device=device,
        audit_batch_size=args.batch_size,
    )

    banner(
        "5. FINAL TRAINING-RETENTION RESULTS"
    )

    print_metrics(
        "OVERALL",
        overall,
    )

    print_metrics(
        "SLOT 0",
        slot0,
    )

    print_metrics(
        "SLOT 1",
        slot1,
    )

    if (
        overall["correct_gt_distractor"]
        >= STRONG_PAIR_WIN_GATE
        and overall["margin_ge_0_5"]
        >= STRONG_MARGIN_GATE
    ):
        classification = (
            "STRONG_FINAL_TRAINING_RETENTION"
        )

        next_step = (
            "Retention gate PASSED. "
            "Next allowed step: run the unchanged "
            "pre-authorized positive controls. "
            "Sealed novel/shortcut evaluation stays closed "
            "unless both positive-control gates pass."
        )

    else:
        classification = (
            "FINAL_TRAINING_RETENTION_BELOW_GATE"
        )

        next_step = (
            "STOP before positive controls. "
            "Treatment 5 did not demonstrate strong final "
            "retention of the paired training task."
        )

    print(
        f"CLASSIFICATION: {classification}"
    )

    print()
    print(next_step)

    result = {
        "audit":
            "DaveLM v0.9 Treatment #5 "
            "final training-retention audit",

        "read_only":
            True,

        "positive_controls_evaluated":
            False,

        "sealed_evaluation_opened":
            False,

        "hashes": {
            "schedule":
                schedule_sha,

            "pair_pool":
                pair_pool_sha,

            "final_checkpoint":
                checkpoint_sha,
        },

        "contract": {
            "pair_count":
                EXPECTED_PAIR_COUNT,

            "steps":
                EXPECTED_STEPS,

            "batch_size":
                EXPECTED_BATCH_SIZE,

            "pairs_per_batch":
                EXPECTED_PAIRS_PER_BATCH,

            "scheduled_examples":
                EXPECTED_EXAMPLES,

            "slot0_examples":
                EXPECTED_SLOT0,

            "slot1_examples":
                EXPECTED_SLOT1,

            "document_length":
                EXPECTED_DOCUMENT_LENGTH,

            "supervised_positions_per_document":
                EXPECTED_SUPERVISED_PER_DOCUMENT,

            "total_supervised_positions":
                EXPECTED_TOTAL_SUPERVISED,

            "answer_replacements":
                EXPECTED_ANSWER_REPLACEMENTS,

            "ordinary_nonanswer_ce_positions":
                EXPECTED_NONANSWER_CE,

            "margin":
                MARGIN,

            "strong_pair_win_gate":
                STRONG_PAIR_WIN_GATE,

            "strong_margin_gate":
                STRONG_MARGIN_GATE,
        },

        "structural_audit": {
            "query_positions":
                query_positions,

            "answer_token_indices":
                answer_indices,

            "answer_causal_positions":
                answer_causal_positions,
        },

        "overall":
            overall,

        "slot0":
            slot0,

        "slot1":
            slot1,

        "classification":
            classification,

        "next_step":
            next_step,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    banner(
        "AUDIT COMPLETE"
    )

    print(
        f"Saved:\n{OUTPUT_PATH}"
    )

    print()
    print("No optimizer created.")
    print("No backward pass.")
    print("No training.")
    print("No positive controls.")
    print("No sealed evaluation.")


if __name__ == "__main__":
    main()