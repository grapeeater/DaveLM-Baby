"""Outcome-blind protocol and token-index-map builder.

This script loads only tokenizer/corpus metadata. It does not import torch or
load a checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

SITE = r"C:\DaveLM\.venv\Lib\site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)
from tokenizers import Tokenizer  # type: ignore

ROOT = Path(r"C:\DaveLM-CADAVER")
OUT = ROOT / "sf1_token_position_patching_forensic_v1"
TRAIN = ROOT / "single_fact_acquisition_sf1_seed87011" / "TRAIN.json"
TOK_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
PATTERN = re.compile(r"^(?P<subject>[A-Za-z]+) (?P<predicate1>[a-z]+) the (?P<object1>[^.]+)\.\nThe person who (?P<predicate2>[a-z]+) the (?P<object2>.+) was$")


def sha(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes())
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def span_tokens(offsets, span):
    start, end = span
    return [i for i, (a, b) in enumerate(offsets) if b > start and a < end]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("TOKEN_INDEX_MAP.json", "PROTOCOL.json", "PREINFERENCE_RECEIPT.json"):
        assert not (OUT / name).exists(), f"refusing to overwrite pre-existing frozen artifact: {name}"
    tok = Tokenizer.from_file(str(TOK_PATH))
    rows = json.loads(TRAIN.read_text(encoding="utf-8"))
    assert len(rows) == 16
    maps = []
    for row in rows:
        prompt = row["prompt"]
        match = PATTERN.fullmatch(prompt)
        assert match, row["id"]
        assert match.group("subject") == row["actor"]
        assert match.group("predicate1") == match.group("predicate2") == row["predicate"]
        assert match.group("object1") == match.group("object2") == row["object"]
        enc = tok.encode(prompt)
        assert enc.ids == row["prompt_token_ids"]
        assert tok.decode(enc.ids) == prompt
        spans = {
            "factual_subject": match.span("subject"),
            "factual_object_first": match.span("object1"),
            "queried_object_repeat": match.span("object2"),
            "predicate_fact": match.span("predicate1"),
            "predicate_query": match.span("predicate2"),
        }
        token_indices = {name: span_tokens(enc.offsets, span) for name, span in spans.items()}
        assert all(token_indices.values())
        semantic_sets = [set(v) for v in token_indices.values()]
        assert all(not (semantic_sets[i] & semantic_sets[j]) for i in range(len(semantic_sets)) for j in range(i + 1, len(semantic_sets)))
        p = len(enc.ids)
        final_predictive = [p - 1]
        predicate_relation = sorted(set(token_indices["predicate_fact"] + token_indices["predicate_query"]))
        used = set(final_predictive)
        for key in ("factual_subject", "factual_object_first", "queried_object_repeat"):
            used.update(token_indices[key])
        used.update(predicate_relation)
        remaining = sorted(set(range(p)) - used)
        base_groups_token_index = {
            "final_predictive": final_predictive,
            "factual_subject": token_indices["factual_subject"],
            "factual_object_first": token_indices["factual_object_first"],
            "queried_object_repeat": token_indices["queried_object_repeat"],
            "object_all_mentions": sorted(set(token_indices["factual_object_first"] + token_indices["queried_object_repeat"])),
            "predicate_relation": predicate_relation,
            "remaining_prompt_context": remaining,
            "all_nonfinal_context": list(range(0, p - 1)),
        }
        # Stored positions are model positions after prepending BOS once.
        model_groups = {k: [x + 1 for x in v] for k, v in base_groups_token_index.items()}
        assert model_groups["final_predictive"] == [p]
        exclusive = ["final_predictive", "factual_subject", "factual_object_first", "queried_object_repeat", "predicate_relation", "remaining_prompt_context"]
        union = set()
        for key in exclusive:
            assert not union.intersection(model_groups[key]); union.update(model_groups[key])
        assert union == set(range(1, p + 1))
        candidate_lengths = [len(x) for x in row["candidate_token_ids"]]
        assert candidate_lengths == [4, 4]
        maps.append({
            "id": row["id"], "family_id": row["family_id"], "pair_id": row["pair_id"],
            "prompt": prompt, "prompt_token_ids": enc.ids, "prompt_tokens": enc.tokens,
            "prompt_offsets": [list(x) for x in enc.offsets], "prompt_token_count": p,
            "character_spans": {k: list(v) for k, v in spans.items()},
            "token_indices_without_bos": token_indices,
            "model_positions_with_bos": model_groups,
            "candidate_token_ids": row["candidate_token_ids"], "candidate_lengths": candidate_lengths,
            "dynamic_groups": {
                "answer_generation_path": f"model positions {p} through final current sequence position inclusive",
                "all_positions": "all current sequence positions including BOS",
            },
            "exclusive_prompt_partition": exclusive,
        })
    write_json(OUT / "TOKEN_INDEX_MAP.json", {"indexing": "prompt token index is zero-based; model position adds one for BOS at position 0", "records": maps})

    protocol = {
        "name": "SF1_TOKEN_POSITION_PATCHING_FORENSIC_V1",
        "status": "FROZEN_BEFORE_CHECKPOINT_LOAD",
        "authorized_inputs": {
            "factual": "exact 16 SF1 TRAIN records",
            "D3": "exact frozen 256 unrelated TinyStories positions from prior forensic",
            "language": "same aligned first 128 TinyStories DEV records used by prior SF1 forensics",
        },
        "sites": [
            {"name": "after_block_4_attention_residual", "stage": 8}, {"name": "after_block_4_mlp_residual", "stage": 9},
            {"name": "after_block_5_attention_residual", "stage": 10}, {"name": "after_block_5_mlp_residual", "stage": 11},
            {"name": "after_block_7_attention_residual", "stage": 14}, {"name": "after_block_7_mlp_residual", "stage": 15},
        ],
        "factual_position_groups": [
            "final_predictive", "factual_subject", "factual_object_first", "queried_object_repeat",
            "object_all_mentions", "predicate_relation", "remaining_prompt_context", "all_nonfinal_context",
            "answer_generation_path", "all_positions",
        ],
        "D3_language_position_groups": ["final_predictive", "all_prior_context", "all_positions"],
        "directions": {
            "S_into_P": "compute both states through site, replace selected Pilot1 recipient positions with SF1 donor positions, then run Pilot1 suffix",
            "P_into_S": "compute both states through site, replace selected SF1 recipient positions with Pilot1 donor positions, then run SF1 suffix",
        },
        "output_paths": ["Pilot1 final_norm+untied head", "SF1 final_norm+untied head"],
        "patch_semantics": "torch.where on a frozen boolean [batch,time] position mask at the named residual boundary; recipient suffix remains fixed",
        "factual_scoring": {
            "candidate": "sum four candidate-token conditional log probabilities, excluding EOS; positive correct-minus-distractor wins",
            "exact_answer_eos": "greedy from BOS+exact prompt, ordinary EOS, max32; exact iff generated ids equal correct four-token candidate then EOS",
            "pairs": "both assignment reversals must have positive sequence margin",
        },
        "D3_scoring": "first next-token logits/softmax on the exact prior 256 positions; trained-name mass and individual names",
        "language_scoring": "correctly aligned causal next-token CE on the same four batches of 32 records",
        "baseline_requirements": {
            "same_source_manual_patch_max_abs_logit": 1e-6,
            "prior_cross_run_aggregate_tolerance": 1e-5,
            "all_positions_cross_patch_aggregate_tolerance": 1e-5,
            "policy": "failure seals uninterpreted results; tolerances may not be silently changed",
        },
        "interpretive_limit": "inference-time causal sufficiency under hybrids only; no training-time-site or held-out-generalization claim",
        "forbidden": ["training", "optimizer", "autograd", "weight mutation", "locked SF1 panels", "FINAL", "sacred material", "treatment design"],
    }
    write_json(OUT / "PROTOCOL.json", protocol)
    receipt = {
        "protocol_sha256": sha(OUT / "PROTOCOL.json"),
        "token_index_map_sha256": sha(OUT / "TOKEN_INDEX_MAP.json"),
        "train_sha256": sha(TRAIN), "tokenizer_sha256": sha(TOK_PATH),
        "record_count": len(maps), "checkpoint_loaded": False, "model_behavior_accessed": False,
    }
    write_json(OUT / "PREINFERENCE_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
