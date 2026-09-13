r"""Frozen T12 novel-geometry generalization preflight + feasibility test.

Read-only / inference-only diagnostic. Builds a small deterministic set of
strict-counterfactual quartets whose structural mapping/query/value positions
lie genuinely OUTSIDE the T12 training geometry support, then runs the frozen
T12 checkpoint on it with correct structural indices externally supplied.

No training, no checkpoint modification, no T13. Novel layouts are constructed
with the same grammar, vocabulary, task, answer objective, and model weights.

Decisions are preregistered (not tuned after outcomes):
  PASS          if target>distractor rate >= 0.90 and retrieval-row accuracy >= 0.90
  FAIL          otherwise (clean set constructed)
  INCONCLUSIVE  if a clean novel-geometry set could not be constructed
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

OUT_DIR = Path(r"C:\DaveLM-CADAVER\treatment12_frozen_geometry_generalization")

EXPECTED_TRAIN_POOL_SHA256 = "8f5d60de10fe2fdc8e772a9c1fc3e9f07861edd1583d7c413a095c2f55c6903c"
EXPECTED_FINAL_CHECKPOINT_SHA256 = (
    "e378d3d85f3a53add4046aefc7cc42d643f59d12525859af6ea79dbfca266cd5"
)

# Geometry combos strictly outside the T10/T11 training ranges (train used
# prefix_len in 8..12 and between_len in 3..6).
NOVEL_COMBOS = [
    (16, 8), (20, 8), (24, 10), (28, 10),
    (32, 12), (16, 12), (24, 8), (32, 8),
]
QUARTETS_PER_COMBO = 2


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


def digest_ids(ids) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


# ---------------- layout signature helpers (outcome-blind) ----------------
LAYOUT_KEYS = [
    "sequence_length", "q", "answer_causal", "answer_index",
    "k0", "v0", "k1", "v1",
    "q_to_k0", "q_to_k1", "q_to_v0", "q_to_v1",
    "k0_to_v0", "k1_to_v1",
    "row0_to_row1_spacing", "row0_val_to_row1_val_spacing",
    "query_to_answer",
]


def layout_signature(meta: Dict[str, Any], seq_len: int) -> str:
    record = {
        "sequence_length": seq_len,
        "q": meta["qdp"],
        "answer_causal": meta["answer_causal_pos"],
        "answer_index": meta["answer_token_index"],
        "k0": meta["row0_src_pos"],
        "v0": meta["row0_val_pos"],
        "k1": meta["row1_src_pos"],
        "v1": meta["row1_val_pos"],
        "q_to_k0": meta["qdp"] - meta["row0_src_pos"],
        "q_to_k1": meta["qdp"] - meta["row1_src_pos"],
        "q_to_v0": meta["qdp"] - meta["row0_val_pos"],
        "q_to_v1": meta["qdp"] - meta["row1_val_pos"],
        "k0_to_v0": meta["row0_val_pos"] - meta["row0_src_pos"],
        "k1_to_v1": meta["row1_val_pos"] - meta["row1_src_pos"],
        "row0_to_row1_spacing": meta["row1_src_pos"] - meta["row0_src_pos"],
        "row0_val_to_row1_val_spacing": meta["row1_val_pos"] - meta["row0_val_pos"],
        "query_to_answer": meta["answer_causal_pos"] - meta["qdp"],
    }
    return "|".join(repr(record[k]) for k in LAYOUT_KEYS)


# ---------------- reusable frozen doc builder primitives ----------------
sys.path.insert(0, r"C:\DaveLM-v0.9")
sys.path.insert(0, r"C:\DaveLM-CADAVER")
import experiments.two_mapping_contextual_binding.run as original_run  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from treatment10_common import (  # noqa: E402
    build_member_doc,
    load_tokenizer,
    verify_isolated_single_tokens,
)
from treatment12_config import RETENTION_POOL_PATH, TRAIN_POOL_PATH  # noqa: E402
from treatment12_model import Treatment12Model, doc_retrieval_meta  # noqa: E402


def load_key_value_pools(train_pool) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
    key_map: Dict[int, str] = {}
    value_map: Dict[int, str] = {}
    for quartet in train_pool["quartets"]:
        for doc in quartet["docs"]:
            key_map[int(doc["query_key_token"])] = str(doc["query_key_word"])
            value_map[int(doc["target_value_token"])] = str(doc["target_value_word"])
            value_map[int(doc["distractor_value_token"])] = str(doc["distractor_value_word"])
    keys = [{"word": word, "token_id": tok} for tok, word in sorted(key_map.items())]
    values = [{"word": word, "token_id": tok} for tok, word in sorted(value_map.items())]
    require(len(keys) == 5 and len(values) == 5, "unexpected key/value pool size")
    return keys, values


def build_novel_quartets():
    tokenizer = load_tokenizer()
    groups = original_run._identity_pool(tokenizer)
    filler = groups["filler"]
    filler_words = [str(item["value"]) for item in filler]
    verify_isolated_single_tokens(tokenizer, filler, "filler")

    train_pool = read_json(TRAIN_POOL_PATH)
    require(sha256_file(TRAIN_POOL_PATH) == EXPECTED_TRAIN_POOL_SHA256, "train pool hash")
    train_docs = [doc for q in train_pool["quartets"] for doc in q["docs"]]
    train_layout_sigs = set()
    for doc in train_docs:
        train_layout_sigs.add(layout_signature(doc_retrieval_meta(doc), len(doc["full_document_token_ids"])))
    require(len(train_layout_sigs) == 20, "unexpected train layout signature count")

    keys, values = load_key_value_pools(train_pool)
    key_pairs = list(combinations(range(5), 2))
    value_pairs = list(combinations(range(5), 2))

    bos = int(original_run.required_token_id(tokenizer, "<bos>"))
    quartets: List[Dict[str, Any]] = []
    used_signatures: set = set()
    global_index = 0

    for combo_index, (prefix_len, between_len) in enumerate(NOVEL_COMBOS):
        for local in range(QUARTETS_PER_COMBO):
            kp = key_pairs[(combo_index + local) % len(key_pairs)]
            vp = value_pairs[(combo_index * 2 + local) % len(value_pairs)]
            order_mode = (combo_index + local) % 2
            k0, k1 = keys[kp[0]], keys[kp[1]]
            v0, v1 = values[vp[0]], values[vp[1]]

            seed = 80_000_000 + combo_index * 1000 + local * 17
            rng = random.Random(seed)
            prefix_words = [filler_words[rng.randrange(len(filler_words))] for _ in range(prefix_len)]
            between_words = [filler_words[rng.randrange(len(filler_words))] for _ in range(between_len)]
            tail_words = [filler_words[rng.randrange(len(filler_words))] for _ in range(192)]

            line0_key = k0 if order_mode == 0 else k1
            line1_key = k1 if order_mode == 0 else k0

            def value_for(key_tok, orientation):
                if orientation == 1:
                    return v0 if key_tok == k0["token_id"] else v1
                return v1 if key_tok == k0["token_id"] else v0

            docs = []
            for member in ("o1_k0", "o1_k1", "o2_k0", "o2_k1"):
                orientation = 1 if member.startswith("o1") else 2
                query_key = k0 if member.endswith("k0") else k1
                lines = []
                for line_key in (line0_key, line1_key):
                    val = value_for(line_key["token_id"], orientation)
                    lines.append([line_key["word"], val["word"]])
                ans = value_for(query_key["token_id"], orientation)
                raw_ids = build_member_doc(tokenizer, lines, query_key["word"], ans["word"],
                                           prefix_words, between_words, tail_words)
                model_ids = [bos] + raw_ids
                require(len(model_ids) == 193, "novel doc length != 193")
                other_key = k1 if query_key["token_id"] == k0["token_id"] else k0
                distractor = value_for(other_key["token_id"], orientation)

                occ = {
                    "k0": [i for i, x in enumerate(model_ids) if x == k0["token_id"]],
                    "k1": [i for i, x in enumerate(model_ids) if x == k1["token_id"]],
                    "v0": [i for i, x in enumerate(model_ids) if x == v0["token_id"]],
                    "v1": [i for i, x in enumerate(model_ids) if x == v1["token_id"]],
                }
                qk_label = "k0" if query_key["token_id"] == k0["token_id"] else "k1"
                ok_label = "k1" if qk_label == "k0" else "k0"
                tgt_label = "v0" if ans["token_id"] == v0["token_id"] else "v1"
                dst_label = "v1" if tgt_label == "v0" else "v0"
                require(len(occ[qk_label]) == 2 and len(occ[tgt_label]) == 2,
                        "query/target occurrence count")
                require(len(occ[ok_label]) == 1 and len(occ[dst_label]) == 1,
                        "other-key/distractor occurrence count")
                qdp = max(occ[qk_label])
                ans_index = max(occ[tgt_label])
                q_clause = min(occ[qk_label])
                t_clause = min(occ[tgt_label])
                other_pos = occ[ok_label][0]
                dist_pos = occ[dst_label][0]
                docs.append({
                    "doc_id": f"novel:qt_{global_index:04d}:{member}",
                    "quartet_id": f"novel:qt_{global_index:04d}",
                    "member": member,
                    "axis": "novel",
                    "orientation": orientation,
                    "query_key_word": query_key["word"],
                    "query_key_token": query_key["token_id"],
                    "target_value_word": ans["word"],
                    "target_value_token": ans["token_id"],
                    "distractor_value_token": distractor["token_id"],
                    "distractor_value_word": distractor["word"],
                    "candidate_pair_sorted": sorted([v0["token_id"], v1["token_id"]]),
                    "query_slot": 0 if line0_key["token_id"] == query_key["token_id"] else 1,
                    "mapping_order": order_mode,
                    "qdp": qdp,
                    "query_key_clause_pos": q_clause,
                    "target_clause_pos": t_clause,
                    "answer_token_index": ans_index,
                    "answer_causal_position": ans_index - 1,
                    "distractor_value_pos": dist_pos,
                    "other_key_pos": other_pos,
                    "full_document_token_ids": model_ids,
                    "novel_combo": f"p{prefix_len}_b{between_len}",
                })
            quartet = {
                "quartet_id": f"novel:qt_{global_index:04d}",
                "axis": "novel",
                "novel_combo": f"p{prefix_len}_b{between_len}",
                "key_pair_tokens": sorted([k0["token_id"], k1["token_id"]]),
                "value_pair_tokens": sorted([v0["token_id"], v1["token_id"]]),
                "docs": docs,
            }
            quartets.append(quartet)
            global_index += 1

            # record geometry signature actually realized for this quartet
            for doc in docs:
                used_signatures.add(layout_signature(doc_retrieval_meta(doc), 193))

    novel_sigs = sorted(used_signatures)
    unseen = sorted(set(novel_sigs) - train_layout_sigs)
    require(unseen, "no genuinely unseen novel layout signature produced")
    require(len(novel_sigs) >= len(NOVEL_COMBOS), "fewer novel signatures than combos")
    return quartets, train_layout_sigs, novel_sigs, unseen


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    require(torch.cuda.is_available() if args.device == "cuda" else True, "GPU unavailable")
    device = torch.device(args.device)

    require(sha256_file(TRAIN_POOL_PATH) == EXPECTED_TRAIN_POOL_SHA256, "train pool hash")
    train_pool = read_json(TRAIN_POOL_PATH)
    train_docs = [doc for q in train_pool["quartets"] for doc in q["docs"]]
    train_digests = {digest_ids(d["full_document_token_ids"]) for d in train_docs}

    quartets, train_layout_sigs, novel_sigs, unseen = build_novel_quartets()
    docs = [doc for q in quartets for doc in q["docs"]]

    # outcome-blind geometry preflight checks
    balance_orient = defaultdict(int)
    balance_slot = defaultdict(int)
    sig_counts = defaultdict(int)
    digests = set()
    for doc in docs:
        sig = layout_signature(doc_retrieval_meta(doc), len(doc["full_document_token_ids"]))
        sig_counts[sig] += 1
        digests.add(digest_ids(doc["full_document_token_ids"]))
        balance_orient[doc["orientation"]] += 1
        balance_slot[doc["query_slot"]] += 1
    require(not (digests & train_digests), "accidental token-array overlap with training")
    require(balance_orient[1] == balance_orient[2], "orientation imbalance")
    require(balance_slot[0] == balance_slot[1], "slot imbalance")
    require(set(sig_counts) <= set(novel_sigs), "unexpected signature")
    require(all(s not in train_layout_sigs for s in sig_counts), "train-geometry overlap in novel set")

    pool_payload = {
        "artifact_type": "treatment12_novel_geometry_diagnostic_pool",
        "quartets": quartets,
    }
    write_json(OUT_DIR / "novel_geometry_pool.json", pool_payload)

    preflight = {
        "status": "NOVEL_GEOMETRY_PREFLIGHT_PASS",
        "train_pool_sha256": sha256_file(TRAIN_POOL_PATH),
        "train_layout_signature_count": len(train_layout_sigs),
        "novel_layout_signature_count": len(set(sig_counts)),
        "proposed_combo_count": len(NOVEL_COMBOS),
        "novel_signatures_total_realized": len(novel_sigs),
        "novel_signatures_unseen_vs_train": len(unseen),
        "novel_signatures_overlap_with_train": len(set(sig_counts) & train_layout_sigs),
        "quartets": len(quartets),
        "documents": len(docs),
        "orientation_counts": dict(sorted(balance_orient.items())),
        "query_slot_counts": dict(sorted(balance_slot.items())),
        "token_array_overlap_with_train": 0,
        "determinism_note": (
            "Each novel quartet contains both reversal orientations and both query slots under the "
            "same layout, so layout cannot deterministically predict orientation, query slot, or "
            "target row; mapping rows and values are assigned independently of layout."
        ),
    }
    write_json(OUT_DIR / "novel_geometry_preflight.json", preflight)
    write_json(OUT_DIR / "novel_geometry_preflight_report.md",
               "# T12 novel-geometry preflight\n\n" + json.dumps(preflight, indent=1))

    # ---------------- frozen inference ----------------
    require(sha256_file(Path(r"C:\DaveLM-CADAVER\treatment12_query_conditioned_mapping_retrieval_seed8380\checkpoints\query_conditioned_mapping_retrieval\seed_8380\latest.pt")) == EXPECTED_FINAL_CHECKPOINT_SHA256,
            "T12 final checkpoint hash mismatch")
    model = Treatment12Model()
    model.to(device)
    state = torch.load(
        Path(r"C:\DaveLM-CADAVER\treatment12_query_conditioned_mapping_retrieval_seed8380\checkpoints\query_conditioned_mapping_retrieval\seed_8380\latest.pt"),
        map_location=device)
    model.load_state_dict(extract_state(state))
    model.eval()

    metas = [doc_retrieval_meta(d) for d in docs]
    rows = []
    with torch.no_grad():
        for start in range(0, len(docs), 32):
            chunk_docs = docs[start:start + 32]
            chunk_metas = metas[start:start + 32]
            input_ids = torch.tensor([d["full_document_token_ids"] for d in chunk_docs],
                                     dtype=torch.long, device=device)
            positions = {
                key: torch.tensor([m[key] for m in chunk_metas], dtype=torch.long, device=device)
                for key in ("qdp", "answer_causal_pos", "row0_src_pos", "row0_val_pos",
                            "row1_src_pos", "row1_val_pos")
            }
            pos_renamed = {
                "q": positions["qdp"],
                "k0": positions["row0_src_pos"],
                "k1": positions["row1_src_pos"],
                "v0": positions["row0_val_pos"],
                "v1": positions["row1_val_pos"],
                "answer": positions["answer_causal_pos"],
            }
            logits, extras = model(input_ids, pos_renamed)
            for i, (doc, meta) in enumerate(zip(chunk_docs, chunk_metas)):
                vec = logits[i, meta["answer_causal_pos"], :]
                target = int(doc["target_value_token"])
                distractor = int(doc["distractor_value_token"])
                target_logit = float(vec[target])
                distractor_logit = float(vec[distractor])
                probs = torch.softmax(vec.float(), dim=-1)
                correct = correct_row(meta)
                weights = extras["retrieval_weights"][i]
                scores = extras["retrieval_scores"][i]
                rows.append({
                    "doc_id": doc["doc_id"],
                    "quartet_id": doc["quartet_id"],
                    "member": doc["member"],
                    "orientation": int(doc["orientation"]),
                    "query_slot": int(doc["query_slot"]),
                    "novel_combo": doc["novel_combo"],
                    "signature": layout_signature(meta, 193),
                    "target_token": target,
                    "distractor_token": distractor,
                    "pred_token": int(vec.argmax(-1).item()),
                    "prediction_equals_target": int(int(vec.argmax(-1).item()) == target),
                    "prediction_equals_distractor": int(int(vec.argmax(-1).item()) == distractor),
                    "margin": target_logit - distractor_logit,
                    "candidate_mass": float(probs[target] + probs[distractor]),
                    "retrieval_correct": int(weights.argmax(-1).item()) == correct,
                    "correct_row_weight": float(weights[correct].item()),
                    "incorrect_row_weight": float(weights[1 - correct].item()),
                    "retrieval_margin": float((scores[correct] - scores[1 - correct]).item()),
                })

    summary = summarize(rows)
    by_signature = summarize_splits(rows, "signature")
    reversal = reversal_summary(rows)
    marker = decide(summary)
    results = {
        "status": "NOVEL_GEOMETRY_GENERALIZATION_RUN",
        "provenance": {
            "train_pool_sha256": sha256_file(TRAIN_POOL_PATH),
            "final_checkpoint_sha256": sha256_file(Path(
                r"C:\DaveLM-CADAVER\treatment12_query_conditioned_mapping_retrieval_seed8380\checkpoints\query_conditioned_mapping_retrieval\seed_8380\latest.pt")),
            "novel_documents": len(docs),
            "novel_quartets": len(quartets),
        },
        "overall": summary,
        "per_signature": by_signature,
        "reversal_following": reversal,
        "per_document_rows": rows,
        "marker": marker,
        "interpretation_limits": [
            "Result does not establish self-localization.",
            "Result does not establish general binding.",
            "Result does not establish universal positional robustness.",
            "Correct structural positions were externally supplied to T12.",
        ],
    }
    write_json(OUT_DIR / "frozen_geometry_results.json", results)
    write_json(OUT_DIR / "frozen_geometry_report.md",
               "# T12 novel-geometry generalization report\n\n"
               + json.dumps({k: v for k, v in results.items() if k != "per_document_rows"}, indent=1))
    print(json.dumps({"overall": summary, "per_signature_counts": {
        k: v["n"] for k, v in by_signature.items()}, "reversal": reversal}, indent=1))
    print(f"T12_NOVEL_GEOMETRY_GENERALIZATION: {marker}")
    return 0


def extract_state(checkpoint: Any) -> Dict[str, torch.Tensor]:
    for key in ("model_state_dict", "model_state", "model", "state_dict"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    raise RuntimeError("unsupported checkpoint")


def correct_row(meta: Dict[str, Any]) -> int:
    if meta["row0_src_token"] == meta["query_key_token"]:
        return 0
    if meta["row1_src_token"] == meta["query_key_token"]:
        return 1
    raise RuntimeError("query key not a row source")


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    margins = [r["margin"] for r in rows]
    n = len(rows)
    group = defaultdict(list)
    for r in rows:
        group[r["quartet_id"]].append(r)
    quartet_ok = sum(len(g) == 4 and all(x["prediction_equals_target"] for x in g)
                     for g in group.values())
    return {
        "n": n,
        "answer_accuracy": sum(r["prediction_equals_target"] for r in rows) / n,
        "predictions_equal_target_count": sum(r["prediction_equals_target"] for r in rows),
        "predictions_equal_distractor_count": sum(r["prediction_equals_distractor"] for r in rows),
        "target_gt_distractor_rate": sum(r["margin"] > 0 for r in rows) / n,
        "mean_answer_margin": statistics.mean(margins),
        "median_answer_margin": statistics.median(margins),
        "mean_candidate_mass": statistics.mean(r["candidate_mass"] for r in rows),
        "complete_quartet_success_count": quartet_ok,
        "complete_quartet_success_rate": quartet_ok / len(group),
        "retrieval_argmax_matches_correct_row_rate": statistics.mean(
            r["retrieval_correct"] for r in rows),
        "mean_correct_row_attention_weight": statistics.mean(
            r["correct_row_weight"] for r in rows),
        "mean_incorrect_row_attention_weight": statistics.mean(
            r["incorrect_row_weight"] for r in rows),
        "mean_retrieval_score_margin": statistics.mean(r["retrieval_margin"] for r in rows),
    }


def summarize_splits(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[r[key]].append(r)
    return {k: summarize(v) for k, v in sorted(grouped.items())}


def reversal_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    group = defaultdict(list)
    for r in rows:
        group[r["quartet_id"]].append(r)
    pairs = changed = followed = 0
    for member_rows in group.values():
        by_member = {r["member"]: r for r in member_rows}
        for key in ("k0", "k1"):
            o1 = by_member.get(f"o1_{key}")
            o2 = by_member.get(f"o2_{key}")
            if o1 is None or o2 is None:
                continue
            pairs += 1
            pref1 = o1["target_token"] if o1["margin"] > 0 else o1["distractor_token"]
            pref2 = o2["target_token"] if o2["margin"] > 0 else o2["distractor_token"]
            changed_bool = pref1 != pref2
            changed += changed_bool
            followed += changed_bool and o1["margin"] > 0 and o2["margin"] > 0
    return {
        "source_orientation_pairs": pairs,
        "preferred_candidate_changed_count": changed,
        "preferred_candidate_changed_rate": changed / pairs,
        "changed_and_followed_local_mapping_count": followed,
        "changed_and_followed_local_mapping_rate": followed / pairs,
    }


def decide(summary: Dict[str, Any]) -> str:
    if summary["target_gt_distractor_rate"] >= 0.90 and \
            summary["retrieval_argmax_matches_correct_row_rate"] >= 0.90:
        return "PASS"
    return "FAIL"


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"DIAGNOSTIC ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
