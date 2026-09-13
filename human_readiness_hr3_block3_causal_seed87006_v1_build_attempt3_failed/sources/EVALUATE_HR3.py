"""Frozen nonsacred evaluation for HR-3 checkpoints and the unchanged parent."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
from tokenizers import Tokenizer

from hr3_block3_runtime import (
    PARENT_SHA256,
    TOKENIZER_SHA256,
    aligned_dev_loss,
    atomic_json,
    binding_gate,
    binding_summary,
    candidate_log_probability,
    configure_runtime,
    greedy_ids,
    load_model,
    pinned_binding,
    read_json,
    sha,
    verify_integrity,
)


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def decoded_trigram_repeat(text: str) -> bool:
    words = normalized(text).split()
    grams = [tuple(words[i : i + 3]) for i in range(max(0, len(words) - 2))]
    return len(set(grams)) != len(grams)


def three_identical_token_run(ids: list[int]) -> bool:
    return any(ids[i] == ids[i + 1] == ids[i + 2] for i in range(max(0, len(ids) - 2)))


def score_controlled(model, tok: Tokenizer, rows: list[dict], device: torch.device) -> list[dict]:
    scored = []
    for row in rows:
        prompt_ids = tok.encode(row["prompt"]).ids
        candidate_ids = row["candidate_token_ids"]
        assert candidate_ids == [tok.encode(candidate).ids for candidate in row["candidates"]]
        scores_and_tokens = [candidate_log_probability(model, prompt_ids, ids, device) for ids in candidate_ids]
        scores = [value[0] for value in scores_and_tokens]
        correct = row["correct_index"]
        other = 1 - correct
        margin = scores[correct] - scores[other]
        chosen = correct if margin > 0 else other if margin < 0 else None
        sequence = greedy_ids(model, prompt_ids, device, 32)
        generated = sequence[len(prompt_ids) + 1 :]
        expected = candidate_ids[correct] + [3]
        scored.append(
            {
                "id": row["id"],
                "category": row["category"],
                "family_id": row.get("family_id"),
                "reversal_pair_id": row.get("reversal_pair_id"),
                "assignment": row.get("assignment"),
                "query": row.get("query"),
                "fact_order": row.get("fact_order"),
                "prompt": row["prompt"],
                "prompt_token_ids": prompt_ids,
                "candidates": row["candidates"],
                "candidate_token_ids": candidate_ids,
                "correct_index": correct,
                "candidate_log_scores": scores,
                "candidate_token_log_scores": [value[1] for value in scores_and_tokens],
                "correct_minus_incorrect_margin": margin,
                "chosen_index": chosen,
                "correct": chosen == correct,
                "tie": chosen is None,
                "candidate_pair_probability_mass": [math.exp(score - max(scores)) / sum(math.exp(x - max(scores)) for x in scores) for score in scores],
                "greedy_response_ids": generated,
                "greedy_response_decoded": tok.decode(generated, skip_special_tokens=True),
                "greedy_exact_correct_then_eos": generated == expected,
            }
        )
    return scored


def summarize_controlled(rows: list[dict]) -> dict:
    out = {
        "items": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "ties": sum(bool(row["tie"]) for row in rows),
        "greedy_exact_correct_then_eos": sum(bool(row["greedy_exact_correct_then_eos"]) for row in rows),
        "mean_signed_margin": sum(row["correct_minus_incorrect_margin"] for row in rows) / len(rows),
        "minimum_signed_margin": min(row["correct_minus_incorrect_margin"] for row in rows),
        "maximum_signed_margin": max(row["correct_minus_incorrect_margin"] for row in rows),
    }
    out["accuracy"] = out["correct"] / out["items"]
    if rows and rows[0]["category"] == "fact":
        reversals = defaultdict(list)
        families = defaultdict(list)
        for row in rows:
            reversals[row["reversal_pair_id"]].append(row)
            families[row["family_id"]].append(row)
        profiles = []
        for pair_id, members in sorted(reversals.items()):
            assert len(members) == 2
            profiles.append({"reversal_pair_id": pair_id, "item_ids": [m["id"] for m in members], "both_correct": all(m["correct"] for m in members), "margins": [m["correct_minus_incorrect_margin"] for m in members]})
        out.update(
            {
                "reversal_pairs": len(profiles),
                "successful_reversal_pairs": sum(x["both_correct"] for x in profiles),
                "complete_families": sum(all(x["correct"] for x in members) for members in families.values()),
                "family_count": len(families),
                "reversal_profiles": profiles,
            }
        )
    return out


def score_generations(model, tok: Tokenizer, rows: list[dict], device: torch.device) -> tuple[list[dict], dict]:
    raw = []
    for row in rows:
        prompt_ids = tok.encode(row["prompt"]).ids
        sequence = greedy_ids(model, prompt_ids, device, int(row["generation_max_new_tokens"]))
        response = sequence[len(prompt_ids) + 1 :]
        decoded = tok.decode(response, skip_special_tokens=True)
        raw.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "prompt_token_ids": prompt_ids,
                "greedy_generated_ids": sequence,
                "response_ids": response,
                "raw_decoded_response": decoded,
                "immediate_eos": response == [3],
                "length_including_eos": len(response),
                "three_identical_token_run": three_identical_token_run(response),
                "repeated_decoded_trigram": decoded_trigram_repeat(decoded),
            }
        )
    frequency = Counter(normalized(x["raw_decoded_response"]) for x in raw if not x["immediate_eos"])
    for row in raw:
        duplicate = frequency[normalized(row["raw_decoded_response"])] > 3 if not row["immediate_eos"] else False
        row["duplicate_normalized_non_eos_over_three"] = duplicate
        row["automatic_non_degenerate"] = not (row["immediate_eos"] or row["three_identical_token_run"] or row["repeated_decoded_trigram"] or duplicate)
        row["review_complete"] = None
        row["review_relevant"] = None
    summary = {
        "prompts": len(raw),
        "non_immediate_eos": sum(not x["immediate_eos"] for x in raw),
        "automatic_non_degenerate": sum(x["automatic_non_degenerate"] for x in raw),
        "human_review_status": "REQUIRED_BY_FROZEN_PROTOCOL; raw outputs preserved without post-hoc completion or relevance judgment",
    }
    return raw, summary


def binding_results(model, bundle: Path, protocol: dict, device: torch.device, pinned) -> dict:
    result = {}
    model.eval()
    with torch.no_grad():
        for name, rel in protocol["data"]["binding_dev_pools"].items():
            pool = read_json(bundle / rel)
            docs = [doc for quartet in pool["quartets"] for doc in quartet["docs"]]
            raw = pinned.binding_eval(model, docs, device)
            summary = binding_summary(raw)
            result[name] = {"summary": summary, "gate_pass": binding_gate(summary), "raw": raw}
    return result


def candidate_gates(evaluation: dict, protocol: dict) -> dict:
    controlled = evaluation["controlled_summary"]
    facts = controlled["fact"]
    instructions = controlled["instruction"]
    continuity = controlled["continuity"]
    generation = evaluation["generation_summary"]
    metrics = {
        "aligned_tinystories_loss": evaluation["aligned_tinystories"]["loss"] <= 3.25,
        "fact_correct": facts["correct"] >= 20,
        "fact_reversals": facts["successful_reversal_pairs"] >= 10,
        "fact_complete_families": facts["complete_families"] >= 4,
        "fact_greedy_exact": facts["greedy_exact_correct_then_eos"] >= 18,
        "instruction_correct": instructions["correct"] >= 10,
        "instruction_greedy_exact": instructions["greedy_exact_correct_then_eos"] >= 9,
        "continuity_correct": continuity["correct"] >= 10,
        "continuity_greedy_exact": continuity["greedy_exact_correct_then_eos"] >= 9,
        "generation_non_immediate_eos": generation["non_immediate_eos"] >= 16,
        "generation_automatic_non_degenerate": generation["automatic_non_degenerate"] >= 16,
    }
    binding = {name: value["gate_pass"] for name, value in evaluation["binding"].items()}
    return {
        "machine_gates": metrics,
        "machine_gates_all_pass": all(metrics.values()),
        "binding_gates": binding,
        "binding_gates_all_pass": all(binding.values()),
        "human_review_required": True,
        "development_candidate_status": "PENDING_HUMAN_REVIEW" if all(metrics.values()) and all(binding.values()) else "FAIL_MACHINE_OR_BINDING_GATES",
    }


def evaluate_one(name: str, checkpoint: Path, bundle: Path, protocol: dict, tok: Tokenizer, device: torch.device, pinned) -> dict:
    assert sha(checkpoint), f"unhashable checkpoint: {checkpoint}"
    model = load_model(checkpoint, device, bundle).eval()
    dev = jsonl(bundle / protocol["data"]["english_dev"])
    battery_root = Path(protocol["readiness_dev"]["path"])
    receipt_expected = protocol["readiness_dev"]["receipt_sha256"]
    assert sha(battery_root / "FREEZE_RECEIPT.json") == receipt_expected
    facts = jsonl(battery_root / "DEV_FACTS.jsonl")
    instructions = jsonl(battery_root / "DEV_INSTRUCTIONS.jsonl")
    continuity = jsonl(battery_root / "DEV_CONTINUITY.jsonl")
    generations = jsonl(battery_root / "DEV_GENERATION.jsonl")
    controlled = {"fact": score_controlled(model, tok, facts, device), "instruction": score_controlled(model, tok, instructions, device), "continuity": score_controlled(model, tok, continuity, device)}
    generation_rows, generation_summary = score_generations(model, tok, generations, device)
    result = {
        "name": name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "aligned_tinystories": aligned_dev_loss(model, dev, device),
        "controlled_rows": controlled,
        "controlled_summary": {category: summarize_controlled(rows) for category, rows in controlled.items()},
        "generations": generation_rows,
        "generation_summary": generation_summary,
        "binding": binding_results(model, bundle, protocol, device, pinned),
    }
    result["gates"] = candidate_gates(result, protocol)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    verify_integrity(args.bundle)
    protocol = read_json(args.bundle / "HR3_PROTOCOL.json")
    assert sha(Path(protocol["tokenizer_path"])) == TOKENIZER_SHA256
    configure_runtime(protocol["training"]["seed"])
    device = torch.device("cuda")
    tok = Tokenizer.from_file(protocol["tokenizer_path"])
    assert Path(protocol["readiness_dev"]["path"]).name == "human_test_readiness_v2_seed87010"
    assert "FINAL" not in str(Path(protocol["readiness_dev"]["path"])).upper()
    pinned = pinned_binding(args.bundle)
    checkpoints = [("Pilot1_parent", Path(protocol["parent_path"]))]
    for update in (100, 250, 500):
        path = args.run_dir / f"checkpoint_{update}.pt"
        if path.exists():
            checkpoints.append((f"HR3_block3_update_{update}", path))
    assert len(checkpoints) >= 2, "no completed HR-3 checkpoint available for frozen evaluation"
    output = {
        "status": "HR3_NONSACRED_DEV_EVALUATION_COMPLETE",
        "bundle": str(args.bundle),
        "bundle_receipt_sha256": sha(args.bundle / "FREEZE_RECEIPT.json"),
        "readiness_dev_receipt_sha256": sha(Path(protocol["readiness_dev"]["path"]) / "FREEZE_RECEIPT.json"),
        "final_readiness_accessed": False,
        "sacred_accessed": False,
        "checkpoints": [evaluate_one(name, path, args.bundle, protocol, tok, device, pinned) for name, path in checkpoints],
    }
    atomic_json(output, args.run_dir / "NONSACRED_DEV_EVALUATION.json")
    print(json.dumps({"status": output["status"], "checkpoints": [x["name"] for x in output["checkpoints"]]}, indent=2))


if __name__ == "__main__":
    main()
