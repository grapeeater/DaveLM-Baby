r"""Frozen Treatment-12 causal validation battery (bounded audit, no training).

Uses only the frozen T12 final checkpoint and the frozen 160-document retention
set. Runs the normal forward pass and four causal interventions, reports
per-document paired records, and applies preregistered interpretation logic.

Conditions:
  A normal frozen T12
  B uniform row routing (alpha = [0.5, 0.5])
  C zero retrieved-value residual (Wo(retrieved) removed)
  D force the wrong row (alpha one-hot on the non-correct row)
  E in-distribution query swap using matched same-orientation query twins
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from treatment12_config import (
    BATCH_SIZE,
    DOCUMENT_LENGTH,
    FINAL_CHECKPOINT_PATH,
    RETENTION_POOL_PATH,
    VOCAB_SIZE,
)
from treatment12_model import (
    Treatment12Model,
    correct_row_index,
    doc_retrieval_meta,
)

OUT_DIR = Path(r"C:\DaveLM-CADAVER\treatment12_frozen_causal_validation")
EXPECTED_FINAL_CHECKPOINT_SHA256 = (
    "e378d3d85f3a53add4046aefc7cc42d643f59d12525859af6ea79dbfca266cd5"
)
EXPECTED_RETENTION_SHA256 = (
    "c341d7308b145bd3c633b62d56e01391f4b05635bfcf2edfb146bcd9f2de4e69"
)

ACC_DROP_FOR_CAUSAL_IMPORTANCE = 0.20
MARGIN_RELATIVE_DROP_FOR_CAUSAL_IMPORTANCE = 0.50


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


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    for key in ("model_state_dict", "model_state", "model", "state_dict"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            return checkpoint[key]
    raise RuntimeError("unsupported checkpoint")


def retrieval_1d(model, q, k0, k1, v0, v1, alpha_override=None):
    """q/k/v are 1-D hidden vectors [H]. Returns scores[2], alpha[2], retrieved[H]."""
    q_proj = model.retrieval.wq(q)
    k0_proj = model.retrieval.wk(k0)
    k1_proj = model.retrieval.wk(k1)
    scale = model.retrieval.scale
    s0 = (q_proj * k0_proj).sum() * scale
    s1 = (q_proj * k1_proj).sum() * scale
    scores = torch.stack([s0, s1])
    if alpha_override is None:
        alpha = F.softmax(scores, dim=0)
    else:
        alpha = alpha_override
    retrieved = alpha[0] * model.retrieval.wv(v0) + alpha[1] * model.retrieval.wv(v1)
    return scores, alpha, retrieved


def per_doc_row(doc, meta, pred_token, logits_vec, target, distractor,
                scores, alpha, alpha_label: str) -> Dict[str, Any]:
    target_logit = float(logits_vec[target])
    distractor_logit = float(logits_vec[distractor])
    probs = F.softmax(logits_vec, dim=-1)
    correct = correct_row_index(meta)
    return {
        "doc_id": doc["doc_id"],
        "quartet_id": doc["quartet_id"],
        "member": doc["member"],
        "orientation": int(doc["orientation"]),
        "query_slot": int(doc["query_slot"]),
        "target_token": int(target),
        "distractor_token": int(distractor),
        "pred_token": int(pred_token),
        "prediction_equals_target": int(pred_token == int(target)),
        "prediction_equals_distractor": int(pred_token == int(distractor)),
        "margin": target_logit - distractor_logit,
        "candidate_mass": float(probs[int(target)] + probs[int(distractor)]),
        "retrieval_argmax_matches_correct": int(scores.argmax(-1).item()) == correct,
        "correct_row_weight": float(alpha[correct].item()),
        "incorrect_row_weight": float(alpha[1 - correct].item()),
        "retrieval_score_margin": float((scores[correct] - scores[1 - correct]).item()),
        "alpha_label": alpha_label,
    }


def summarize_rows(rows: List[Dict[str, Any]], condition: str) -> Dict[str, Any]:
    n = len(rows)
    margins = [r["margin"] for r in rows]
    accuracy = sum(r["prediction_equals_target"] for r in rows)
    group = defaultdict(list)
    for r in rows:
        group[r["quartet_id"]].append(r)
    quartet_success = sum(len(g) == 4 and all(x["prediction_equals_target"] for x in g)
                          for g in group.values())
    reversal = reversal_following(group)
    return {
        "condition": condition,
        "n": n,
        "answer_accuracy": accuracy / n,
        "predictions_equal_target_count": accuracy,
        "predictions_equal_distractor_count": sum(r["prediction_equals_distractor"] for r in rows),
        "target_gt_distractor_rate": sum(r["margin"] > 0 for r in rows) / n,
        "mean_answer_margin": statistics.mean(margins),
        "median_answer_margin": statistics.median(margins),
        "mean_candidate_mass": statistics.mean(r["candidate_mass"] for r in rows),
        "complete_quartet_success_count": quartet_success,
        "complete_quartet_success_rate": quartet_success / len(group),
        "retrieval_argmax_matches_correct_row_rate": statistics.mean(
            r["retrieval_argmax_matches_correct"] for r in rows),
        "mean_correct_row_attention_weight": statistics.mean(
            r["correct_row_weight"] for r in rows),
        "mean_incorrect_row_attention_weight": statistics.mean(
            r["incorrect_row_weight"] for r in rows),
        "mean_retrieval_score_margin": statistics.mean(
            r["retrieval_score_margin"] for r in rows),
        "reversal_following": reversal,
    }


def reversal_following(group: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
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
        "preferred_candidate_changed_rate": changed / pairs if pairs else 0.0,
        "changed_and_followed_local_mapping_count": followed,
        "changed_and_followed_local_mapping_rate": followed / pairs if pairs else 0.0,
    }


def render_markdown(results: Dict[str, Any], census: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Treatment-12 frozen causal validation report")
    lines.append("")
    lines.append("Bounded audit. No training. Frozen checkpoint and frozen retention set only.")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    p = results["provenance"]
    lines.append(f"- retention pool sha256: `{p['retention_pool_sha256']}`")
    lines.append(f"- final T12 checkpoint sha256: `{p['final_checkpoint_sha256']}`")
    lines.append(f"- documents: {p['retention_documents']}, quartets: {p['retention_quartets']}")
    lines.append("")
    lines.append("## Baseline reproduction (Condition A)")
    lines.append("")
    lines.append(f"passed: {results['baseline_verification']['passed']}")
    for key, value in results["baseline_verification"]["summary"].items():
        if key == "reversal_following":
            continue
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Geometry census findings")
    lines.append("")
    lines.append(f"- unique layout signatures: "
                 f"train={census['unique_layout_signatures']['train']}, "
                 f"retention={census['unique_layout_signatures']['retention']}, "
                 f"retention-not-in-train={census['unique_layout_signatures']['retention_not_in_train_count']}")
    lines.append(f"- retention has genuinely unseen layout geometry: "
                 f"{census['retention_has_genuinely_unseen_geometry']}")
    lines.append(f"- all absolute q/k/v/answer positions reused in retention: "
                 f"{census['all_absolute_positions_reused_in_retention']}")
    lines.append(f"- relative-distance combinations reused in retention: "
                 f"{census['relative_distance_combos_reused_in_retention']}")
    det = census["deterministic_relationships"]
    lines.append(f"- query_slot deterministic given layout: {det['query_slot_deterministic_given_layout']}; "
                 f"orientation deterministic given layout: {det['orientation_deterministic_given_layout']}; "
                 f"target row deterministic given layout: {det['target_row_deterministic_given_layout']}")
    lines.append("")
    lines.append("## Intervention definitions")
    lines.append("")
    for name, definition in results["intervention_definitions"].items():
        lines.append(f"- **{name}**: {definition}")
    lines.append("")
    lines.append("## Results table")
    lines.append("")
    lines.append("| condition | answer acc | tgt>dist | mean margin | quartet succ | retr row acc | corr weight |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in ("A", "B", "C", "D"):
        s = results["condition_summaries"][name]
        lines.append(f"| {name} | {s['answer_accuracy']:.4f} | {s['target_gt_distractor_rate']:.4f} | "
                     f"{s['mean_answer_margin']:.4f} | {s['complete_quartet_success_count']}/40 | "
                     f"{s['retrieval_argmax_matches_correct_row_rate']:.4f} | "
                     f"{s['mean_correct_row_attention_weight']:.4f} |")
    lines.append("")
    lines.append("## Per-paired A->D change analysis")
    lines.append("")
    paired = results["paired_a_to_d_change"]
    lines.append(f"- A->D prediction changed: {sum(x['A_to_D_pred_changed'] for x in paired)}/{len(paired)}")
    lines.append(f"- D predictions equal distractor: "
                 f"{sum(x['D_prediction_equals_distractor'] for x in paired)}/{len(paired)}")
    lines.append(f"- margin sign inverted A->D: {sum(x['margin_sign_inverted'] for x in paired)}/{len(paired)}")
    lines.append(f"- A->B prediction changed: {sum(x['B_pred_changed_from_A'] for x in paired)}/{len(paired)}")
    lines.append(f"- A->C prediction changed: {sum(x['C_pred_changed_from_A'] for x in paired)}/{len(paired)}")
    lines.append("")
    lines.append("## Query-swap (E)")
    lines.append("")
    qs = results["query_swap"]
    lines.append(f"- pairs: {qs['pair_count']}")
    lines.append(f"- routing-flip rate: {qs['routing_flip_rate']:.4f}")
    lines.append(f"- predicted-direction routing-flip rate: {qs['predicted_direction_routing_flip_rate']:.4f}")
    lines.append(f"- answer-flip rate: {qs['answer_flip_rate']:.4f}")
    lines.append(f"- answer follows patched query's local mapping rate: "
                 f"{qs['answer_follows_patched_query_mapping_rate']:.4f}")
    lines.append("")
    lines.append("## Preregistered interpretation logic")
    lines.append("")
    for key, value in results["interpretation_logic"].items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("## Causal question markers")
    lines.append("")
    for key, marker in results["markers"].items():
        lines.append(f"- {key}: {marker}")
    lines.append("")
    lines.append("## Prohibited overclaims")
    lines.append("")
    for claim in results["prohibited_overclaims"]:
        lines.append(f"- {claim}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- T12 remains structurally assisted (q/k/v/answer positions are grammar-supplied).")
    lines.append("- No threshold was tuned after inspecting intervention outcomes.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    require(torch.cuda.is_available() if args.device == "cuda" else True, "GPU unavailable")
    device = torch.device(args.device)

    require(FINAL_CHECKPOINT_PATH.is_file(), "missing T12 final checkpoint")
    checkpoint_sha = sha256_file(FINAL_CHECKPOINT_PATH)
    require(checkpoint_sha == EXPECTED_FINAL_CHECKPOINT_SHA256,
            f"T12 final checkpoint hash mismatch: {checkpoint_sha}")
    retention_sha = sha256_file(RETENTION_POOL_PATH)
    require(retention_sha == EXPECTED_RETENTION_SHA256, "retention pool hash mismatch")

    pool = read_json(RETENTION_POOL_PATH)
    quartets = pool["quartets"]
    docs = [doc for q in quartets for doc in q["docs"]]
    metas = [doc_retrieval_meta(d) for d in docs]
    n_docs = len(docs)
    require(n_docs == 160, "retention document count != 160")

    model = Treatment12Model()
    model.to(device)
    state = torch.load(FINAL_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(extract_state_dict(state))
    model.eval()

    cond_rows: Dict[str, List[Dict[str, Any]]] = {name: [] for name in ("A", "B", "C", "D")}
    e_pairs: List[Dict[str, Any]] = []

    with torch.no_grad():
        for start in range(0, n_docs, BATCH_SIZE):
            chunk_docs = docs[start:start + BATCH_SIZE]
            chunk_metas = metas[start:start + BATCH_SIZE]
            input_ids = torch.tensor(
                [d["full_document_token_ids"] for d in chunk_docs],
                dtype=torch.long, device=device)
            captured: List[torch.Tensor] = []
            handle = model.base_model.final_norm.register_forward_hook(
                lambda m, i, o: captured.append(o))
            model.base_model(input_ids)
            handle.remove()
            hidden = captured[0].float()

            for i, (doc, meta) in enumerate(zip(chunk_docs, chunk_metas)):
                q = hidden[i, meta["qdp"]]
                k0 = hidden[i, meta["row0_src_pos"]]
                k1 = hidden[i, meta["row1_src_pos"]]
                v0 = hidden[i, meta["row0_val_pos"]]
                v1 = hidden[i, meta["row1_val_pos"]]
                ans_hidden = hidden[i, meta["answer_causal_pos"]]
                correct = correct_row_index(meta)
                target = int(doc["target_value_token"])
                distractor = int(doc["distractor_value_token"])
                head = model.base_model.language_head

                # A: normal
                scores, alpha, retrieved = retrieval_1d(model, q, k0, k1, v0, v1)
                logits_a = head(ans_hidden + model.retrieval.wo(retrieved))
                pred_a = int(logits_a.argmax(-1).item())
                cond_rows["A"].append(per_doc_row(
                    doc, meta, pred_a, logits_a, target, distractor, scores, alpha, "normal"))

                # B: uniform alpha
                uniform = torch.full((2,), 0.5, device=device)
                _, alpha_b, retrieved_b = retrieval_1d(
                    model, q, k0, k1, v0, v1, alpha_override=uniform)
                logits_b = head(ans_hidden + model.retrieval.wo(retrieved_b))
                pred_b = int(logits_b.argmax(-1).item())
                cond_rows["B"].append(per_doc_row(
                    doc, meta, pred_b, logits_b, target, distractor, scores, alpha_b, "uniform"))

                # C: zero retrieved residual (scores/alpha normal)
                logits_c = head(ans_hidden)
                pred_c = int(logits_c.argmax(-1).item())
                cond_rows["C"].append(per_doc_row(
                    doc, meta, pred_c, logits_c, target, distractor, scores, alpha, "zero-residual"))

                # D: force wrong row
                forced = torch.zeros(2, device=device)
                forced[1 - correct] = 1.0
                _, alpha_d, retrieved_d = retrieval_1d(
                    model, q, k0, k1, v0, v1, alpha_override=forced)
                logits_d = head(ans_hidden + model.retrieval.wo(retrieved_d))
                pred_d = int(logits_d.argmax(-1).item())
                cond_rows["D"].append(per_doc_row(
                    doc, meta, pred_d, logits_d, target, distractor, scores, alpha_d, "forced-wrong"))

                # E: query swap from matched same-orientation twin
                orientation = int(doc["orientation"])
                key = doc["member"].split("_")[1]
                other_key = "k1" if key == "k0" else "k0"
                twin_member = f"o{orientation}_{other_key}"
                twin_index = None
                for ii, other in enumerate(chunk_docs):
                    if other["quartet_id"] == doc["quartet_id"] and other["member"] == twin_member:
                        twin_index = ii
                        break
                require(twin_index is not None, "matched query twin missing from chunk")
                twin_meta = chunk_metas[twin_index]
                q_patched = hidden[twin_index, twin_meta["qdp"]]
                scores_e, alpha_e, retrieved_e = retrieval_1d(model, q_patched, k0, k1, v0, v1)
                logits_e = head(ans_hidden + model.retrieval.wo(retrieved_e))
                pred_e = int(logits_e.argmax(-1).item())
                twin_q_token = int(chunk_docs[twin_index]["query_key_token"])
                patched_row_correct = 0 if meta["row0_src_token"] == twin_q_token else 1
                e_pairs.append({
                    "target_doc": doc["doc_id"],
                    "source_twin_doc": chunk_docs[twin_index]["doc_id"],
                    "orientation": orientation,
                    "query_slot": int(doc["query_slot"]),
                    "original_argmax_row": int(scores.argmax(-1).item()),
                    "patched_argmax_row": int(scores_e.argmax(-1).item()),
                    "routing_flip": int(scores.argmax(-1).item() != scores_e.argmax(-1).item()),
                    "predicted_direction_routing_flip": int(
                        scores_e.argmax(-1).item() == patched_row_correct),
                    "original_pred_token": pred_a,
                    "patched_pred_token": pred_e,
                    "answer_flip": int(pred_a != pred_e),
                    "answer_follows_patched_query_mapping": int(
                        pred_e == int(chunk_docs[twin_index]["target_value_token"])),
                    "original_correct_row_weight": float(alpha[correct].item()),
                    "patched_query_row_weight": float(alpha_e[patched_row_correct].item()),
                })

    summary_a = summarize_rows(cond_rows["A"], "A")
    baseline_ok = (
        summary_a["answer_accuracy"] == 1.0
        and summary_a["target_gt_distractor_rate"] == 1.0
        and summary_a["complete_quartet_success_count"] == 40
        and abs(summary_a["mean_answer_margin"] - 13.184189319610596) < 0.5
        and abs(summary_a["median_answer_margin"] - 12.976156234741211) < 0.5
        and summary_a["retrieval_argmax_matches_correct_row_rate"] == 1.0
        and summary_a["mean_correct_row_attention_weight"] > 0.99
        and summary_a["reversal_following"]["changed_and_followed_local_mapping_count"] == 80
    )

    if not baseline_ok:
        write_failure(summary_a, checkpoint_sha, retention_sha)
        print("T12_FROZEN_BASELINE: FAIL")
        print("GEOMETRY_CENSUS: COMPLETE")
        print("Baseline reproduction materially disagrees with the authoritative audit. "
              "Stopping before interventions.")
        return 2

    summaries = {"A": summary_a}
    for name in ("B", "C", "D"):
        summaries[name] = summarize_rows(cond_rows[name], name)

    a_map = {r["doc_id"]: r for r in cond_rows["A"]}
    d_map = {r["doc_id"]: r for r in cond_rows["D"]}
    b_map = {r["doc_id"]: r for r in cond_rows["B"]}
    c_map = {r["doc_id"]: r for r in cond_rows["C"]}
    paired = []
    for r in cond_rows["A"]:
        paired.append({
            "doc_id": r["doc_id"],
            "A_pred": r["pred_token"],
            "D_pred": d_map[r["doc_id"]]["pred_token"],
            "A_margin": r["margin"],
            "D_margin": d_map[r["doc_id"]]["margin"],
            "A_to_D_pred_changed": int(r["pred_token"] != d_map[r["doc_id"]]["pred_token"]),
            "D_prediction_equals_distractor": d_map[r["doc_id"]]["prediction_equals_distractor"],
            "margin_sign_inverted": int((r["margin"] > 0) != (d_map[r["doc_id"]]["margin"] > 0)),
            "B_margin": b_map[r["doc_id"]]["margin"],
            "C_margin": c_map[r["doc_id"]]["margin"],
            "B_pred_changed_from_A": int(r["pred_token"] != b_map[r["doc_id"]]["pred_token"]),
            "C_pred_changed_from_A": int(r["pred_token"] != c_map[r["doc_id"]]["pred_token"]),
        })

    markers = {}
    acc_a = summary_a["answer_accuracy"]
    margin_a = summary_a["mean_answer_margin"]
    rate_a = summary_a["target_gt_distractor_rate"]
    b = summaries["B"]
    uniform_degraded = (
        (acc_a - b["answer_accuracy"]) >= ACC_DROP_FOR_CAUSAL_IMPORTANCE
        or (rate_a - b["target_gt_distractor_rate"]) >= ACC_DROP_FOR_CAUSAL_IMPORTANCE
        or b["mean_answer_margin"] <= margin_a * (1 - MARGIN_RELATIVE_DROP_FOR_CAUSAL_IMPORTANCE))
    markers["uniform_alpha"] = "PASS" if uniform_degraded else "INCONCLUSIVE"

    c = summaries["C"]
    zero_residual_degraded = (
        (acc_a - c["answer_accuracy"]) >= ACC_DROP_FOR_CAUSAL_IMPORTANCE
        or (rate_a - c["target_gt_distractor_rate"]) >= ACC_DROP_FOR_CAUSAL_IMPORTANCE
        or c["mean_answer_margin"] <= margin_a * (1 - MARGIN_RELATIVE_DROP_FOR_CAUSAL_IMPORTANCE))
    retrieval_stayed_accurate = c["retrieval_argmax_matches_correct_row_rate"] > 0.95
    markers["zero_retrieval"] = (
        "PASS" if (zero_residual_degraded and retrieval_stayed_accurate) else "INCONCLUSIVE")

    d = summaries["D"]
    markers["force_wrong_row"] = (
        "PASS" if (d["predictions_equal_distractor_count"] >= 0.5 * n_docs
                   and d["target_gt_distractor_rate"] < 0.5)
        else "INCONCLUSIVE")

    markers["query_swap"] = (
        "PASS" if (sum(1 for p in e_pairs if p["predicted_direction_routing_flip"]) / len(e_pairs)) > 0.7
        else "INCONCLUSIVE")

    routing_flip = sum(1 for p in e_pairs if p["routing_flip"])
    predicted_dir = sum(1 for p in e_pairs if p["predicted_direction_routing_flip"])
    answer_flip = sum(1 for p in e_pairs if p["answer_flip"])
    answer_follows = sum(1 for p in e_pairs if p["answer_follows_patched_query_mapping"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    census = read_json(OUT_DIR / "geometry_census.json")
    results = {
        "provenance": {
            "retention_pool_sha256": retention_sha,
            "final_checkpoint_sha256": checkpoint_sha,
            "retention_documents": n_docs,
            "retention_quartets": len(quartets),
        },
        "baseline_verification": {"passed": baseline_ok, "summary": summary_a},
        "intervention_definitions": {
            "A": "normal frozen T12 forward pass",
            "B": "uniform row routing alpha=[0.5,0.5]; original scores reported for diagnostics",
            "C": "normal scores/alpha but zero Wo(retrieved) residual (answer_hidden only)",
            "D": "alpha one-hot forced to the WRONG row (correct row used only to build the test)",
            "E": "in-distribution q activation swap from matched same-orientation query twin",
        },
        "condition_summaries": summaries,
        "per_document_records": {name: cond_rows[name] for name in ("A", "B", "C", "D")},
        "paired_a_to_d_change": paired,
        "query_swap": {
            "pair_count": len(e_pairs),
            "routing_flip_count": routing_flip,
            "routing_flip_rate": routing_flip / len(e_pairs),
            "predicted_direction_routing_flip_count": predicted_dir,
            "predicted_direction_routing_flip_rate": predicted_dir / len(e_pairs),
            "answer_flip_count": answer_flip,
            "answer_flip_rate": answer_flip / len(e_pairs),
            "answer_follows_patched_query_mapping_count": answer_follows,
            "answer_follows_patched_query_mapping_rate": answer_follows / len(e_pairs),
            "pairs": e_pairs,
        },
        "markers": markers,
        "interpretation_logic": {
            "A": "Baseline must reproduce the authoritative 160/160 retention result (provenance check).",
            "B": "Uniform alpha causing large degradation => learned row selection is causally important.",
            "C": "Zero retrieved residual causing large degradation while retrieval stays accurate => "
                 "the value-transfer residual is causally important beyond row scoring.",
            "D": "Forced wrong row shifting answers strongly toward the distractor => row routing "
                 "controls answer identity in the predicted direction.",
            "E": "Query swap flipping row preference in the predicted direction => the query "
                 "representation governs row selection.",
            "general": ("If an intervention does not behave as expected, report it plainly and "
                        "identify which conclusion is weakened. No thresholds were tuned after "
                        "outcomes. Emphasis is on effect sizes and predicted-direction paired changes."),
        },
        "prohibited_overclaims": [
            "Transformers cannot bind without T12",
            "T11 proved native self-attention is incapable",
            "T12 solved general variable binding",
            "T12 proves Baby can reason",
            "T12's architecture is universally necessary",
            "T12 can search arbitrary documents",
            "Fixed/absolute positional embeddings caused T9/T11 failure",
            "Structural assistance is cheating",
        ],
        "strongest_supported_conclusion": None,
    }
    (OUT_DIR / "frozen_causal_results.json").write_text(
        json.dumps(results, indent=1), encoding="utf-8")
    (OUT_DIR / "frozen_causal_report.md").write_text(
        render_markdown(results, census), encoding="utf-8")

    print(json.dumps({
        "baseline_passed": baseline_ok,
        "A": summary_a,
        "B": b,
        "C": c,
        "D": d,
        "E_summary": results["query_swap"],
    }, indent=1))
    print()
    print("T12_FROZEN_BASELINE: PASS")
    print("GEOMETRY_CENSUS: COMPLETE")
    print(f"UNIFORM_ALPHA_CAUSAL_TEST: {markers['uniform_alpha']}")
    print(f"ZERO_RETRIEVAL_CAUSAL_TEST: {markers['zero_retrieval']}")
    print(f"FORCE_WRONG_ROW_CAUSAL_TEST: {markers['force_wrong_row']}")
    print(f"QUERY_SWAP_CAUSAL_TEST: {markers['query_swap']}")
    overall = all(markers[k] == "PASS" for k in markers)
    print(f"OVERALL_T12_FROZEN_CAUSAL_VALIDATION: {'PASS' if overall else 'INCONCLUSIVE'}")
    return 0


def write_failure(summary_a, checkpoint_sha, retention_sha) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "BASELINE_REPRODUCTION_FAILED",
        "retention_pool_sha256": retention_sha,
        "final_checkpoint_sha256": checkpoint_sha,
        "observed_condition_A_summary": summary_a,
        "note": "Interventions were not run because baseline reproduction materially disagreed "
                "with the authoritative frozen T12 audit.",
    }
    (OUT_DIR / "frozen_causal_results.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8")
    (OUT_DIR / "frozen_causal_report.md").write_text(
        "# T12 frozen causal validation\n\n## BASELINE REPRODUCTION FAILED\n\n```json\n"
        + json.dumps(payload, indent=1) + "\n```\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"AUDIT ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
