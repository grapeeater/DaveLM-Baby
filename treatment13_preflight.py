r"""Treatment-13 localization preflight + initialization smoke (NO training).

Runs an outcome-blind structural census on the generated T13 train/retention
pools, verifies anti-cheat properties and the grammar value-offset pairing, then
builds the Treatment13Model skeleton (base from the authoritative common
checkpoint + randomly initialized localization/retrieval parameters) and runs
one forward pass to confirm shapes, parameter counts, and finiteness. No
optimizer step and no backward pass.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch

from treatment13_config import (
    BATCH_SIZE,
    DOCUMENT_LENGTH,
    GENERATOR_RESULT_PATH,
    NAME,
    OUT_ROOT,
    PREFLIGHT_REPORT_PATH,
    PREFLIGHT_RESULT_PATH,
    RETENTION_POOL_PATH,
    ROW_VALUE_OFFSET,
    START_CHECKPOINT_PATH,
    START_CHECKPOINT_SHA256,
    TRAIN_POOL_PATH,
    VOCAB_SIZE,
)
from treatment13_model import Treatment13Model

LAYOUT_KEYS = [
    "sequence_length", "q", "answer_causal", "k0", "v0", "k1", "v1",
    "q_to_k0", "q_to_v0", "q_to_k1", "q_to_v1",
    "k0_to_v0", "k1_to_v1", "row0_to_row1_spacing", "query_to_answer",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def layout_signature(doc: Dict[str, Any]) -> str:
    # row0/row1 by mapping-line order derived exactly as the T12 grammar defines.
    qs = int(doc["query_slot"])
    qc = int(doc["query_key_clause_pos"])
    tv = int(doc["target_clause_pos"])
    oc = int(doc["other_key_pos"])
    dv = int(doc["distractor_value_pos"])
    if qs == 0:
        r0s, r0v, r1s, r1v = qc, tv, oc, dv
    else:
        r0s, r0v, r1s, r1v = oc, dv, qc, tv
    q = int(doc["qdp"])
    ans = int(doc["answer_causal_position"])
    rec = {
        "sequence_length": len(doc["full_document_token_ids"]),
        "q": q, "answer_causal": ans, "k0": r0s, "v0": r0v, "k1": r1s, "v1": r1v,
        "q_to_k0": q - r0s, "q_to_v0": q - r0v, "q_to_k1": q - r1s, "q_to_v1": q - r1v,
        "k0_to_v0": r0v - r0s, "k1_to_v1": r1v - r1s,
        "row0_to_row1_spacing": r1s - r0s, "query_to_answer": ans - q,
    }
    return "|".join(repr(rec[k]) for k in LAYOUT_KEYS)


def main(argv: List[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    require(torch.cuda.is_available() if args.device == "cuda" else True, "GPU unavailable")
    device = torch.device(args.device)

    generator = read_json(GENERATOR_RESULT_PATH)
    require(generator["status"] == "TREATMENT13_GENERATOR_PASS", "generator status")
    train_pool = read_json(TRAIN_POOL_PATH)
    retention_pool = read_json(RETENTION_POOL_PATH)
    require(sha256_file(TRAIN_POOL_PATH) == generator["train_pool_sha256"], "train hash")
    require(sha256_file(RETENTION_POOL_PATH) == generator["retention_pool_sha256"], "retention hash")

    train_docs = [d for q in train_pool["quartets"] for d in q["docs"]]
    retention_docs = [d for q in retention_pool["quartets"] for d in q["docs"]]
    train_sigs = [layout_signature(d) for d in train_docs]
    ret_sigs = [layout_signature(d) for d in retention_docs]
    train_sig_set = set(train_sigs)
    ret_sig_set = set(ret_sigs)
    require(not (ret_sig_set & train_sig_set), "retention signature overlaps train")

    # value-pairing grammar offset check on every row of every doc
    offset_errors = 0
    for doc in train_docs + retention_docs:
        if int(doc["target_clause_pos"]) - int(doc["query_key_clause_pos"]) != ROW_VALUE_OFFSET:
            offset_errors += 1
        if int(doc["distractor_value_pos"]) - int(doc["other_key_pos"]) != ROW_VALUE_OFFSET:
            offset_errors += 1
    require(offset_errors == 0, "grammar row-value offset violations")

    def per_sig(docs):
        out = {}
        for doc in docs:
            sig = layout_signature(doc)
            entry = out.setdefault(sig, {"slots": set(), "orientations": set(),
                                         "correct_rows": set(), "n": 0})
            entry["slots"].add(int(doc["query_slot"]))
            entry["orientations"].add(int(doc["orientation"]))
            # correct row = row whose source token equals the query key token
            qs = int(doc["query_slot"])
            if qs == 0:
                entry["correct_rows"].add(0)  # query key sits in row0
            else:
                entry["correct_rows"].add(1)
            entry["n"] += 1
        return out

    train_by_sig = per_sig(train_docs)
    ret_by_sig = per_sig(retention_docs)
    for sig, entry in train_by_sig.items():
        require(entry["slots"] == {0, 1}, "train sig missing slot")
        require(entry["orientations"] == {1, 2}, "train sig missing orientation")
    for sig, entry in ret_by_sig.items():
        require(entry["slots"] == {0, 1}, "retention sig missing slot")
        require(entry["orientations"] == {1, 2}, "retention sig missing orientation")

    mapping_source_positions = sorted({
        int(d["query_key_clause_pos"]) for d in train_docs} | {
        int(d["other_key_pos"]) for d in train_docs})
    mapping_value_positions = sorted({
        int(d["target_clause_pos"]) for d in train_docs} | {
        int(d["distractor_value_pos"]) for d in train_docs})

    # token overlap check
    def digests(docs):
        return {hashlib.sha256(",".join(map(str, d["full_document_token_ids"])).encode()).hexdigest()
                for d in docs}

    require(not (digests(train_docs) & digests(retention_docs)), "train/retention token overlap")

    baseline_hash = sha256_file(START_CHECKPOINT_PATH)
    require(baseline_hash == START_CHECKPOINT_SHA256, "baseline checkpoint hash")

    # ---------- initialization smoke (no training) ----------
    model = Treatment13Model()
    model.to(device)
    model.eval()
    baseline = torch.load(START_CHECKPOINT_PATH, map_location=device)
    model.base_model.load_state_dict(extract_state(baseline))

    batch = train_docs[:BATCH_SIZE]
    require(len(batch) == BATCH_SIZE, "batch size")
    input_ids = torch.tensor([d["full_document_token_ids"] for d in batch],
                             dtype=torch.long, device=device)
    qpos = torch.tensor([d["qdp"] for d in batch], dtype=torch.long, device=device)
    anspos = torch.tensor([d["answer_causal_position"] for d in batch],
                          dtype=torch.long, device=device)
    with torch.no_grad():
        logits, extras = model(input_ids, qpos, anspos)
    require(tuple(logits.shape) == (BATCH_SIZE, DOCUMENT_LENGTH, VOCAB_SIZE), "logits shape")
    require(extras["localization_attention"].shape[0] == BATCH_SIZE and
            extras["localization_attention"].shape[-1] == 2, "localization shape")
    att = extras["localization_attention"]
    finite = bool(torch.isfinite(logits).all().item()) and \
        bool(torch.isfinite(att).all().item()) and bool(torch.isfinite(extras["retrieved"]).all().item())
    require(finite, "non-finite smoke tensors")

    smoke = {
        "input_ids_shape": list(input_ids.shape),
        "logits_shape": list(logits.shape),
        "localization_attention_shape": list(att.shape),
        "slot_scores_shape": list(extras["slot_scores"].shape),
        "row_weights_shape": list(extras["row_weights"].shape),
        "retrieved_shape": list(extras["retrieved"].shape),
        "valid_candidate_count_min": int(extras["valid_candidate_counts"].min().item()),
        "all_finite": finite,
        "no_optimizer_step": True,
        "no_backward": True,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    result = {
        "status": "TREATMENT13_LOCALIZATION_PREFLIGHT_PASS",
        "experiment": NAME,
        "removed_scaffold": (
            "T12 supplied exact mapping-row source-key and value positions (row0/row1 source and "
            "value). T13 does not receive any mapping-row position."),
        "still_supplied": (
            "final query-key position (qdp) and answer prediction position; both are public "
            "grammar locations."),
        "localization_mechanism": (
            "two learned row-slot attention distributions over candidate token positions before "
            "the query (learned localizer), no absolute-position rule and no row label."),
        "value_pairing": (
            "once a candidate source position is weighted, its row-local value is the hidden state "
            "at position + ROW_VALUE_OFFSET (=4), a fixed public grammar offset verified on every "
            "row of every document; it does not reveal which row matches the query."),
        "selection_after_localization": (
            "T12-style query-conditioned retrieval scores the two localized row-slot source "
            "representations against the final query and retrieves the weighted value mixture."),
        "provenance": {
            "train_pool_sha256": sha256_file(TRAIN_POOL_PATH),
            "retention_pool_sha256": sha256_file(RETENTION_POOL_PATH),
            "baseline_checkpoint_sha256": baseline_hash,
            "train_quartets": train_pool["quartet_count"],
            "train_documents": len(train_docs),
            "retention_quartets": retention_pool["quartet_count"],
            "retention_documents": len(retention_docs),
        },
        "geometry": {
            "train_layout_signatures": len(train_sig_set),
            "retention_layout_signatures": len(ret_sig_set),
            "retention_unseen_vs_train": len(ret_sig_set - train_sig_set),
            "retention_overlap_with_train": len(ret_sig_set & train_sig_set),
            "train_distinct_mapping_source_positions": len(mapping_source_positions),
            "train_distinct_mapping_value_positions": len(mapping_value_positions),
            "train_layout_combo_counts": dict(sorted(Counter(
                d["layout_combo"] for d in train_docs).items())),
            "retention_layout_combo_counts": dict(sorted(Counter(
                d["layout_combo"] for d in retention_docs).items())),
        },
        "determinism_checks": {
            "train_sigs_with_both_query_slots": sum(
                1 for e in train_by_sig.values() if e["slots"] == {0, 1}),
            "retention_sigs_with_both_query_slots": sum(
                1 for e in ret_by_sig.values() if e["slots"] == {0, 1}),
            "train_sigs_with_both_orientations": sum(
                1 for e in train_by_sig.values() if e["orientations"] == {1, 2}),
            "retention_sigs_with_both_orientations": sum(
                1 for e in ret_by_sig.values() if e["orientations"] == {1, 2}),
            "token_array_overlap_train_retention": 0,
            "grammar_row_value_offset_violations": offset_errors,
            "fixed_absolute_position_rule_possible": len(mapping_source_positions) <= 1,
        },
        "orientation_counts": {
            "train": dict(sorted(Counter(d["orientation"] for d in train_docs).items())),
            "retention": dict(sorted(Counter(d["orientation"] for d in retention_docs).items())),
        },
        "query_slot_counts": {
            "train": dict(sorted(Counter(d["query_slot"] for d in train_docs).items())),
            "retention": dict(sorted(Counter(d["query_slot"] for d in retention_docs).items())),
        },
        "smoke": smoke,
        "parameter_counts": {
            "base_model_untied": model.base_model_parameter_count(),
            "t12_retrieval_module_reference": 245760,
            "t13_total_model": model.parameter_count(),
            "t13_new_parameters": model.new_parameter_count(),
            "delta_vs_t12": model.new_parameter_count() - 245760,
        },
        "no_label_leakage": (
            "model forward receives only input_ids, query position, and answer position; no "
            "correct-row label, target token, or candidate identity is passed."),
    }
    write_json(PREFLIGHT_RESULT_PATH, result)
    write_json(PREFLIGHT_REPORT_PATH, json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    print("T13_LOCALIZATION_PREFLIGHT: PASS")
    return 0


def extract_state(checkpoint: Any) -> Dict[str, torch.Tensor]:
    for key in ("model_state_dict", "model_state", "model", "state_dict"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    raise RuntimeError("unsupported checkpoint")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"PREFLIGHT ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
