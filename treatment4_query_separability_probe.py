from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

import treatment4_train as t4t
import treatment4_eval as t4e


# =============================================================================
# DaveLM v0.9 — Treatment #4
# QUERY / REPRESENTATION SEPARABILITY PROBE
#
# READ-ONLY DIAGNOSTIC.
#
# BABY:
#   - final Treatment #4 checkpoint only
#   - eval mode
#   - no optimizer
#   - no backward
#   - no gradients
#   - no checkpoint mutation
#
# DATA:
#   - Treatment #4 training records
#   - trained_anchors positive control
#   - supported_relation_withheld_geometry positive control
#
# FORBIDDEN:
#   - novel pools
#   - shortcut controls
#   - Treatment #5
#   - protected source modification
#
# PROBE:
#   - closed-form ridge linear classifier
#   - NO optimizer
#   - NO neural-probe training
#   - Baby activations are detached/frozen
# =============================================================================


OUTPUT_PATH = (
    t4e.TREATMENT_ROOT
    / "treatment4_query_separability_probe.json"
)

RIDGE = 1.0e-3
TEST_FOLD = 0
NUM_FOLDS = 5
BATCH_SIZE = 64


# =============================================================================
# HELPERS
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


def deterministic_fold(key: str) -> int:
    digest = hashlib.sha256(
        key.encode("utf-8")
    ).digest()

    value = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return value % NUM_FOLDS


def record_pair_key(row: dict) -> str:
    """
    Keep the same target/distractor candidate pair in one fold.

    This makes the probe harder to game by simply memorizing one answer pair.
    """

    target = int(
        row["target_token_id"]
    )

    distractor = int(
        row["distractor_token_id"]
    )

    lo = min(
        target,
        distractor,
    )

    hi = max(
        target,
        distractor,
    )

    return f"{lo}:{hi}"


def get_query_slot(row: dict) -> int:
    slot = int(
        row["query_slot"]
    )

    if slot not in (
        0,
        1,
    ):
        raise RuntimeError(
            f"Unexpected query_slot: {slot}"
        )

    return slot


def prefix_ids(row: dict) -> list[int]:
    return [
        int(x)
        for x in row[
            "layout"
        ][
            "prefix_ids"
        ]
    ]


def safe_tensor_output(output):
    if torch.is_tensor(
        output
    ):
        return output

    if (
        isinstance(
            output,
            (tuple, list),
        )
        and output
        and torch.is_tensor(
            output[0]
        )
    ):
        return output[0]

    raise RuntimeError(
        "Hook received unsupported module output."
    )


# =============================================================================
# LOAD FROZEN BABY
# =============================================================================

def load_frozen_model(
    device: torch.device,
):
    header(
        "1. PROVENANCE + FROZEN FINAL CHECKPOINT"
    )

    observed_sha = t4e.sha256_file(
        t4e.CANDIDATE_CHECKPOINT
    )

    if (
        observed_sha
        != t4e.EXPECTED_CANDIDATE_SHA256
    ):
        raise RuntimeError(
            "\nFINAL CHECKPOINT SHA MISMATCH.\n"
            f"Expected: {t4e.EXPECTED_CANDIDATE_SHA256}\n"
            f"Observed: {observed_sha}\n"
        )

    print(
        "[PASS] Final Treatment #4 checkpoint SHA exact"
    )

    build_model = t4e.require_callable(
        t4e.original_run,
        "build_model",
    )

    model = build_model(
        "untied"
    ).to(
        device
    )

    checkpoint = torch.load(
        t4e.CANDIDATE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state = t4e.extract_model_state(
        checkpoint
    )

    incompatible = model.load_state_dict(
        state,
        strict=True,
    )

    if (
        incompatible.missing_keys
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "Unexpected state-dict mismatch."
        )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(
            False
        )

    print(
        "[PASS] State dict exact"
    )

    print(
        f"Device:     {device}"
    )

    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    print(
        "Optimizer:  NONE"
    )

    print(
        "Backward:   NONE"
    )

    print(
        "Gradients:  DISABLED"
    )

    return model


# =============================================================================
# HOOK BABY'S 8 BLOCKS + FINAL NORM
# =============================================================================

class ActivationCollector:

    def __init__(
        self,
        model,
    ):
        self.current = {}
        self.handles = []

        if len(
            model.blocks
        ) != 8:
            raise RuntimeError(
                f"Expected 8 transformer blocks; "
                f"found {len(model.blocks)}."
            )

        for index, block in enumerate(
            model.blocks
        ):
            name = f"block_{index}"

            handle = block.register_forward_hook(
                self._make_hook(
                    name
                )
            )

            self.handles.append(
                handle
            )

        self.handles.append(
            model.final_norm.register_forward_hook(
                self._make_hook(
                    "final_norm"
                )
            )
        )

    def _make_hook(
        self,
        name: str,
    ):
        def hook(
            module,
            inputs,
            output,
        ):
            tensor = safe_tensor_output(
                output
            )

            if tensor.ndim != 3:
                raise RuntimeError(
                    f"{name}: expected [B,T,C], "
                    f"got {tuple(tensor.shape)}"
                )

            # Full-prefix evaluation:
            # final prefix token is the causal position
            # whose logits predict the answer token.
            self.current[
                name
            ] = (
                tensor[
                    :,
                    -1,
                    :,
                ]
                .detach()
                .float()
                .cpu()
            )

        return hook

    def clear(
        self,
    ):
        self.current = {}

    def close(
        self,
    ):
        for handle in self.handles:
            handle.remove()

        self.handles = []


# =============================================================================
# DATASET EXTRACTION
# =============================================================================

@torch.no_grad()
def extract_dataset(
    model,
    collector: ActivationCollector,
    records: list[dict],
    device: torch.device,
    dataset_name: str,
):
    header(
        f"EXTRACTING: {dataset_name}"
    )

    if not records:
        raise RuntimeError(
            f"{dataset_name}: empty record set."
        )

    # Group by prefix length so batching requires no padding,
    # attention-mask guessing, or altered native prefixes.
    length_groups = defaultdict(
        list
    )

    for index, row in enumerate(
        records
    ):
        ids = prefix_ids(
            row
        )

        if not ids:
            raise RuntimeError(
                f"{dataset_name} record {index}: empty prefix."
            )

        if len(
            ids
        ) > t4t.CONTEXT_SIZE:
            raise RuntimeError(
                f"{dataset_name} record {index}: "
                f"prefix length {len(ids)} exceeds context "
                f"{t4t.CONTEXT_SIZE}."
            )

        length_groups[
            len(
                ids
            )
        ].append(
            (
                index,
                row,
                ids,
            )
        )

    layer_names = [
        *(f"block_{i}" for i in range(8)),
        "final_norm",
    ]

    activations = {
        layer:
            []
        for layer in layer_names
    }

    labels = []
    folds = []
    baby_correct = []
    target_minus_distractor = []
    canonical_pair_delta = []

    processed = 0

    for length in sorted(
        length_groups
    ):
        group = length_groups[
            length
        ]

        for begin in range(
            0,
            len(group),
            BATCH_SIZE,
        ):
            chunk = group[
                begin:
                begin + BATCH_SIZE
            ]

            inputs = torch.tensor(
                [
                    item[2]
                    for item in chunk
                ],
                dtype=torch.long,
                device=device,
            )

            collector.clear()

            logits = model(
                inputs
            )

            if logits.ndim != 3:
                raise RuntimeError(
                    f"Model logits expected [B,T,V], "
                    f"got {tuple(logits.shape)}"
                )

            answer_logits = (
                logits[
                    :,
                    -1,
                    :
                ]
                .detach()
                .float()
                .cpu()
            )

            for layer in layer_names:
                if layer not in collector.current:
                    raise RuntimeError(
                        f"Missing activation hook: {layer}"
                    )

                activations[
                    layer
                ].append(
                    collector.current[
                        layer
                    ]
                )

            for local_index, (
                record_index,
                row,
                ids,
            ) in enumerate(
                chunk
            ):
                slot = get_query_slot(
                    row
                )

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
                        f"{dataset_name} record "
                        f"{record_index}: target == distractor."
                    )

                row_logits = answer_logits[
                    local_index
                ]

                top1 = int(
                    torch.argmax(
                        row_logits
                    ).item()
                )

                td_margin = float(
                    (
                        row_logits[
                            target
                        ]
                        -
                        row_logits[
                            distractor
                        ]
                    ).item()
                )

                # Canonical orientation by query slot.
                #
                # slot 0:
                #   positive model behavior should push this NEGATIVE
                #
                # slot 1:
                #   positive model behavior should push this POSITIVE
                #
                # Therefore this quantity should align with a slot-1-positive
                # representation if the output readout uses query identity.
                canonical_delta = (
                    td_margin
                    if slot == 1
                    else -td_margin
                )

                labels.append(
                    slot
                )

                folds.append(
                    deterministic_fold(
                        record_pair_key(
                            row
                        )
                    )
                )

                baby_correct.append(
                    int(
                        top1
                        == target
                    )
                )

                target_minus_distractor.append(
                    td_margin
                )

                canonical_pair_delta.append(
                    canonical_delta
                )

            processed += len(
                chunk
            )

    for layer in layer_names:
        activations[
            layer
        ] = torch.cat(
            activations[
                layer
            ],
            dim=0,
        )

        if (
            activations[
                layer
            ].shape[0]
            != len(
                records
            )
        ):
            raise RuntimeError(
                f"{dataset_name} {layer}: activation count mismatch."
            )

    labels_tensor = torch.tensor(
        labels,
        dtype=torch.long,
    )

    folds_tensor = torch.tensor(
        folds,
        dtype=torch.long,
    )

    correct_tensor = torch.tensor(
        baby_correct,
        dtype=torch.float32,
    )

    td_tensor = torch.tensor(
        target_minus_distractor,
        dtype=torch.float32,
    )

    canonical_tensor = torch.tensor(
        canonical_pair_delta,
        dtype=torch.float32,
    )

    slot0 = int(
        (
            labels_tensor == 0
        ).sum().item()
    )

    slot1 = int(
        (
            labels_tensor == 1
        ).sum().item()
    )

    print(
        f"Records:       {len(records)}"
    )

    print(
        f"Slot 0:        {slot0}"
    )

    print(
        f"Slot 1:        {slot1}"
    )

    print(
        f"Baby top-1:    "
        f"{correct_tensor.mean().item():.4f}"
    )

    print(
        f"Mean T-D logit margin: "
        f"{td_tensor.mean().item():.4f}"
    )

    return {
        "name":
            dataset_name,

        "activations":
            activations,

        "labels":
            labels_tensor,

        "folds":
            folds_tensor,

        "baby_correct":
            correct_tensor,

        "target_minus_distractor":
            td_tensor,

        "canonical_pair_delta":
            canonical_tensor,
    }


# =============================================================================
# CLOSED-FORM LINEAR PROBE
# =============================================================================

def standardize_train_test(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
):
    mean = x_train.mean(
        dim=0,
        keepdim=True,
    )

    std = x_train.std(
        dim=0,
        unbiased=False,
        keepdim=True,
    )

    std = torch.where(
        std < 1.0e-8,
        torch.ones_like(
            std
        ),
        std,
    )

    return (
        (
            x_train
            - mean
        ) / std,
        (
            x_test
            - mean
        ) / std,
    )


def fit_ridge_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
):
    """
    Closed-form ridge regression used as a binary linear probe.

    Labels:
        slot 0 -> -1
        slot 1 -> +1

    No optimizer.
    No backward.
    """

    x_train = x_train.double()
    x_test = x_test.double()

    x_train, x_test = standardize_train_test(
        x_train,
        x_test,
    )

    y_signed = (
        y_train.double()
        * 2.0
        - 1.0
    )

    ones_train = torch.ones(
        (
            x_train.shape[0],
            1,
        ),
        dtype=torch.float64,
    )

    ones_test = torch.ones(
        (
            x_test.shape[0],
            1,
        ),
        dtype=torch.float64,
    )

    design_train = torch.cat(
        [
            x_train,
            ones_train,
        ],
        dim=1,
    )

    design_test = torch.cat(
        [
            x_test,
            ones_test,
        ],
        dim=1,
    )

    dim = design_train.shape[
        1
    ]

    regularizer = (
        torch.eye(
            dim,
            dtype=torch.float64,
        )
        * RIDGE
    )

    # Do not penalize bias.
    regularizer[
        -1,
        -1,
    ] = 0.0

    lhs = (
        design_train.T
        @ design_train
        + regularizer
    )

    rhs = (
        design_train.T
        @ y_signed
    )

    weights = torch.linalg.solve(
        lhs,
        rhs,
    )

    test_scores = (
        design_test
        @ weights
    )

    return (
        test_scores.float(),
        weights.float(),
    )


def pearson(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    a = (
        a.float()
        - a.float().mean()
    )

    b = (
        b.float()
        - b.float().mean()
    )

    denom = (
        torch.sqrt(
            torch.sum(
                a * a
            )
        )
        *
        torch.sqrt(
            torch.sum(
                b * b
            )
        )
    )

    if float(
        denom.item()
    ) <= 1.0e-12:
        return 0.0

    return float(
        (
            torch.sum(
                a * b
            )
            / denom
        ).item()
    )


def evaluate_linear_probe(
    dataset: dict,
):
    labels = dataset[
        "labels"
    ]

    folds = dataset[
        "folds"
    ]

    train_mask = (
        folds != TEST_FOLD
    )

    test_mask = (
        folds == TEST_FOLD
    )

    train_count = int(
        train_mask.sum().item()
    )

    test_count = int(
        test_mask.sum().item()
    )

    if train_count == 0:
        raise RuntimeError(
            f"{dataset['name']}: empty probe training split."
        )

    if test_count == 0:
        raise RuntimeError(
            f"{dataset['name']}: empty probe test split."
        )

    train_labels = labels[
        train_mask
    ]

    test_labels = labels[
        test_mask
    ]

    if len(
        torch.unique(
            train_labels
        )
    ) != 2:
        raise RuntimeError(
            f"{dataset['name']}: probe training split "
            "does not contain both query slots."
        )

    if len(
        torch.unique(
            test_labels
        )
    ) != 2:
        raise RuntimeError(
            f"{dataset['name']}: probe test split "
            "does not contain both query slots."
        )

    test_baseline = max(
        float(
            (
                test_labels == 0
            ).float().mean().item()
        ),
        float(
            (
                test_labels == 1
            ).float().mean().item()
        ),
    )

    results = {}

    for layer, x in dataset[
        "activations"
    ].items():
        scores, _ = fit_ridge_probe(
            x[
                train_mask
            ],
            train_labels,
            x[
                test_mask
            ],
        )

        predictions = (
            scores >= 0.0
        ).long()

        accuracy = float(
            (
                predictions
                == test_labels
            )
            .float()
            .mean()
            .item()
        )

        canonical_delta = dataset[
            "canonical_pair_delta"
        ][
            test_mask
        ]

        score_logit_correlation = pearson(
            scores,
            canonical_delta,
        )

        results[
            layer
        ] = {
            "held_out_accuracy":
                accuracy,

            "majority_baseline":
                test_baseline,

            "train_examples":
                train_count,

            "test_examples":
                test_count,

            "probe_score_vs_canonical_logit_delta_pearson":
                score_logit_correlation,
        }

    return {
        "split": {
            "strategy":
                "candidate_pair_grouped_sha256_5fold",

            "test_fold":
                TEST_FOLD,

            "num_folds":
                NUM_FOLDS,

            "train_examples":
                train_count,

            "test_examples":
                test_count,

            "test_majority_baseline":
                test_baseline,
        },

        "layers":
            results,
    }


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify(
    training_result: dict,
    anchor_result: dict,
    supported_result: dict,
    datasets: dict,
):
    train_final = training_result[
        "layers"
    ][
        "final_norm"
    ][
        "held_out_accuracy"
    ]

    anchor_final = anchor_result[
        "layers"
    ][
        "final_norm"
    ][
        "held_out_accuracy"
    ]

    supported_final = supported_result[
        "layers"
    ][
        "final_norm"
    ][
        "held_out_accuracy"
    ]

    anchor_baby = float(
        datasets[
            "trained_anchors"
        ][
            "baby_correct"
        ].mean().item()
    )

    supported_baby = float(
        datasets[
            "supported_relation_withheld_geometry"
        ][
            "baby_correct"
        ].mean().item()
    )

    pc_layer_names = [
        *(f"block_{i}" for i in range(8)),
        "final_norm",
    ]

    anchor_earlier_best = max(
        anchor_result[
            "layers"
        ][
            layer
        ][
            "held_out_accuracy"
        ]
        for layer in pc_layer_names[
            :-1
        ]
    )

    supported_earlier_best = max(
        supported_result[
            "layers"
        ][
            layer
        ][
            "held_out_accuracy"
        ]
        for layer in pc_layer_names[
            :-1
        ]
    )

    # -------------------------------------------------------------------------
    # Highest-priority special case:
    # representation was strong earlier but collapsed by final_norm.
    # -------------------------------------------------------------------------

    if (
        (
            anchor_earlier_best >= 0.85
            and anchor_final <= 0.65
            and (
                anchor_earlier_best
                - anchor_final
            ) >= 0.20
        )
        or
        (
            supported_earlier_best >= 0.85
            and supported_final <= 0.65
            and (
                supported_earlier_best
                - supported_final
            ) >= 0.20
        )
    ):
        return {
            "classification":
                "LATE_LAYER_COLLAPSE",

            "rationale":
                (
                    "Query identity becomes strongly linearly separable "
                    "inside the transformer stack but is substantially "
                    "less separable by final_norm on at least one "
                    "positive-control family."
                ),
        }

    # -------------------------------------------------------------------------
    # Query representation survives, but Baby still answers near chance.
    # -------------------------------------------------------------------------

    if (
        train_final >= 0.85
        and anchor_final >= 0.85
        and supported_final >= 0.85
        and (
            anchor_baby <= 0.65
            or supported_baby <= 0.65
        )
    ):
        return {
            "classification":
                "QUERY_INFO_EXISTS_READOUT_FAILS",

            "rationale":
                (
                    "Query slot is strongly linearly recoverable from "
                    "final hidden states on training and both authorized "
                    "positive controls, while Baby's positive-control "
                    "answer accuracy remains poor. The query information "
                    "exists internally but is not being translated "
                    "reliably into the correct output choice."
                ),
        }

    # -------------------------------------------------------------------------
    # Training representation strong; positive-control representation weak.
    # -------------------------------------------------------------------------

    if (
        train_final >= 0.85
        and (
            anchor_final <= 0.65
            or supported_final <= 0.65
        )
    ):
        return {
            "classification":
                "QUERY_REPRESENTATION_DOES_NOT_GENERALIZE",

            "rationale":
                (
                    "Query slot is strongly linearly separable on the "
                    "training distribution but weak on at least one "
                    "authorized positive-control family. This is "
                    "consistent with a training-specific query/binding "
                    "representation that does not transport to the "
                    "positive-control geometry."
                ),
        }

    return {
        "classification":
            "AMBIGUOUS",

        "rationale":
            (
                "The linear probe does not cleanly isolate a single "
                "representation-vs-readout failure mode under the "
                "predefined diagnostic thresholds."
            ),
    }


# =============================================================================
# DISPLAY
# =============================================================================

def print_probe_table(
    name: str,
    result: dict,
):
    print()
    print(
        name
    )

    print(
        f"{'LAYER':<14}"
        f"{'A/B ACC':>10}"
        f"{'BASE':>10}"
        f"{'SCORE↔LOGIT':>14}"
    )

    print(
        "-" * 48
    )

    for layer, metrics in result[
        "layers"
    ].items():
        print(
            f"{layer:<14}"
            f"{metrics['held_out_accuracy']:>10.4f}"
            f"{metrics['majority_baseline']:>10.4f}"
            f"{metrics['probe_score_vs_canonical_logit_delta_pearson']:>14.4f}"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    header(
        "DaveLM v0.9 — TREATMENT #4 QUERY-SEPARABILITY PROBE"
    )

    print(
        "Purpose:"
    )

    print(
        "  Determine whether Baby's hidden states encode query_slot 0 vs 1"
    )

    print(
        "  on training records versus authorized positive controls."
    )

    print()

    print(
        "Baby training:          NO"
    )

    print(
        "Probe optimizer:        NO"
    )

    print(
        "Backward calls:         NO"
    )

    print(
        "Checkpoint mutation:    NO"
    )

    print(
        "Sealed pools:           FORBIDDEN / NOT CONSTRUCTED"
    )

    if torch.cuda.is_available():
        device = torch.device(
            "cuda"
        )
    else:
        device = torch.device(
            "cpu"
        )

    model = load_frozen_model(
        device
    )

    # Exact Treatment #4 training record reconstruction.
    header(
        "2. RECONSTRUCT TRAINING RECORDS"
    )

    (
        _training_tokenizer,
        training_records,
        _training_store,
    ) = t4t.reconstruct_training()

    if len(
        training_records
    ) != t4t.EXPECTED_RECORDS:
        raise RuntimeError(
            "Treatment #4 training record count mismatch."
        )

    print(
        f"[PASS] Training records: {len(training_records)}"
    )

    # Exact authorized positive-control reconstruction.
    header(
        "3. RECONSTRUCT AUTHORIZED POSITIVE CONTROLS"
    )

    (
        _pc_tokenizer,
        anchors,
        supported,
    ) = t4e.build_positive_controls()

    print()
    print(
        "[PASS] Only authorized PC families constructed."
    )

    print(
        "Novel pools:       NO"
    )

    print(
        "Shortcut controls: NO"
    )

    collector = ActivationCollector(
        model
    )

    try:
        training_dataset = extract_dataset(
            model,
            collector,
            training_records,
            device,
            "training_records",
        )

        anchor_dataset = extract_dataset(
            model,
            collector,
            anchors,
            device,
            "trained_anchors",
        )

        supported_dataset = extract_dataset(
            model,
            collector,
            supported,
            device,
            "supported_relation_withheld_geometry",
        )

    finally:
        collector.close()

    header(
        "4. HELD-OUT LINEAR QUERY-SLOT PROBES"
    )

    training_probe = evaluate_linear_probe(
        training_dataset
    )

    anchor_probe = evaluate_linear_probe(
        anchor_dataset
    )

    supported_probe = evaluate_linear_probe(
        supported_dataset
    )

    print_probe_table(
        "TRAINING RECORDS",
        training_probe,
    )

    print_probe_table(
        "TRAINED ANCHORS",
        anchor_probe,
    )

    print_probe_table(
        "SUPPORTED RELATION / WITHHELD GEOMETRY",
        supported_probe,
    )

    datasets = {
        "training_records":
            training_dataset,

        "trained_anchors":
            anchor_dataset,

        "supported_relation_withheld_geometry":
            supported_dataset,
    }

    diagnosis = classify(
        training_probe,
        anchor_probe,
        supported_probe,
        datasets,
    )

    header(
        "5. DIAGNOSIS"
    )

    print(
        diagnosis[
            "classification"
        ]
    )

    print()

    print(
        diagnosis[
            "rationale"
        ]
    )

    payload = {
        "experiment":
            "DaveLM v0.9 Treatment #4",

        "diagnostic":
            "query_separability_probe",

        "candidate_checkpoint": {
            "path":
                str(
                    t4e.CANDIDATE_CHECKPOINT
                ),

            "sha256":
                t4e.EXPECTED_CANDIDATE_SHA256,
        },

        "device":
            str(
                device
            ),

        "safety": {
            "baby_read_only":
                True,

            "baby_optimizer_created":
                False,

            "probe_optimizer_created":
                False,

            "backward_calls":
                0,

            "gradients_enabled":
                False,

            "checkpoint_modified":
                False,

            "novel_pools_constructed":
                False,

            "shortcut_controls_constructed":
                False,

            "sealed_evaluation_opened":
                False,
        },

        "probe": {
            "type":
                "closed_form_ridge_linear",

            "ridge":
                RIDGE,

            "split_strategy":
                (
                    "candidate-pair-grouped deterministic "
                    "SHA256 5-fold; fold 0 held out"
                ),
        },

        "datasets": {
            "training_records": {
                "examples":
                    len(
                        training_records
                    ),

                "baby_top1_accuracy":
                    float(
                        training_dataset[
                            "baby_correct"
                        ].mean().item()
                    ),

                "probe":
                    training_probe,
            },

            "trained_anchors": {
                "examples":
                    len(
                        anchors
                    ),

                "baby_top1_accuracy":
                    float(
                        anchor_dataset[
                            "baby_correct"
                        ].mean().item()
                    ),

                "probe":
                    anchor_probe,
            },

            "supported_relation_withheld_geometry": {
                "examples":
                    len(
                        supported
                    ),

                "baby_top1_accuracy":
                    float(
                        supported_dataset[
                            "baby_correct"
                        ].mean().item()
                    ),

                "probe":
                    supported_probe,
            },
        },

        "diagnosis":
            diagnosis,
    }

    write_json(
        OUTPUT_PATH,
        payload,
    )

    header(
        "6. RESULT ARTIFACT"
    )

    print(
        OUTPUT_PATH
    )

    print()

    print(
        "Baby modified:         NO"
    )

    print(
        "Training performed:    NO"
    )

    print(
        "Sealed data opened:    NO"
    )

    print()

    print(
        "FINAL:"
    )

    print(
        diagnosis[
            "classification"
        ]
    )


if __name__ == "__main__":
    main()