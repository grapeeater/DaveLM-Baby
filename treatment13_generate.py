r"""Treatment-13 candidate universe generator (preflight only, no training).

Builds train/retention quartet pools over variable mapping-row layouts. Rows
appear at many positions; retention uses layout combinations unseen in train.
No behavioral content; purely structural.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

from treatment13_config import (
    GENERATOR_RESULT_PATH,
    NAME,
    OUT_ROOT,
    QUARTETS_PER_RETENTION_COMBO,
    QUARTETS_PER_TRAIN_COMBO,
    RETENTION_COMBOS,
    RETENTION_POOL_PATH,
    ROW_VALUE_OFFSET,
    SEED,
    SOURCE_T11_TRAIN_POOL,
    TRAIN_BETWEEN_RANGE,
    TRAIN_POOL_PATH,
    TRAIN_PREFIX_RANGE,
)

sys.path.insert(0, r"C:\DaveLM-v0.9")
sys.path.insert(0, r"C:\DaveLM-CADAVER")

import experiments.two_mapping_contextual_binding.run as original_run  # noqa: E402
from treatment10_common import build_member_doc, load_tokenizer, verify_isolated_single_tokens  # noqa: E402


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


def load_key_value_pools():
    train_pool = read_json(SOURCE_T11_TRAIN_POOL)
    key_map: Dict[int, str] = {}
    value_map: Dict[int, str] = {}
    for quartet in train_pool["quartets"]:
        for doc in quartet["docs"]:
            key_map[int(doc["query_key_token"])] = str(doc["query_key_word"])
            value_map[int(doc["target_value_token"])] = str(doc["target_value_word"])
            value_map[int(doc["distractor_value_token"])] = str(doc["distractor_value_word"])
    keys = [{"word": word, "token_id": tok} for tok, word in sorted(key_map.items())]
    values = [{"word": word, "token_id": tok} for tok, word in sorted(value_map.items())]
    require(len(keys) == 5 and len(values) == 5, "key/value pool size")
    return keys, values


def build_pool(tokenizer, keys, values, filler_words, layout_combos, quartets_per_combo, axis, bos):
    key_pairs = list(combinations(range(5), 2))
    value_pairs = list(combinations(range(5), 2))
    quartets: List[Dict[str, Any]] = []
    gid = 0
    for combo_index, (prefix_len, between_len) in enumerate(layout_combos):
        for local in range(quartets_per_combo):
            kp = key_pairs[(combo_index + local) % len(key_pairs)]
            vp = value_pairs[(combo_index * 2 + local) % len(value_pairs)]
            order_mode = (combo_index + local) % 2
            k0, k1 = keys[kp[0]], keys[kp[1]]
            v0, v1 = values[vp[0]], values[vp[1]]
            seed = 100_000_000 + combo_index * 1_000 + local * 17 + (0 if axis == "train" else 500_000_000)
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
                raw = build_member_doc(tokenizer, lines, query_key["word"], ans["word"],
                                       prefix_words, between_words, tail_words)
                model_ids = [bos] + raw
                require(len(model_ids) == 193, "doc length")
                other_key = k1 if query_key["token_id"] == k0["token_id"] else k0
                distractor = value_for(other_key["token_id"], orientation)

                occ = {
                    "k0": [i for i, x in enumerate(model_ids) if x == k0["token_id"]],
                    "k1": [i for i, x in enumerate(model_ids) if x == k1["token_id"]],
                    "v0": [i for i, x in enumerate(model_ids) if x == v0["token_id"]],
                    "v1": [i for i, x in enumerate(model_ids) if x == v1["token_id"]],
                }
                qk = "k0" if query_key["token_id"] == k0["token_id"] else "k1"
                ok = "k1" if qk == "k0" else "k0"
                tg = "v0" if ans["token_id"] == v0["token_id"] else "v1"
                dt = "v1" if tg == "v0" else "v0"
                require(len(occ[qk]) == 2 and len(occ[tg]) == 2, "query/target counts")
                require(len(occ[ok]) == 1 and len(occ[dt]) == 1, "other/distractor counts")
                qdp = max(occ[qk])
                ans_index = max(occ[tg])
                q_clause = min(occ[qk])
                t_clause = min(occ[tg])
                other_pos = occ[ok][0]
                dist_pos = occ[dt][0]
                docs.append({
                    "doc_id": f"t13_{axis}:qt_{gid:06d}:{member}",
                    "quartet_id": f"t13_{axis}:qt_{gid:06d}",
                    "member": member,
                    "axis": axis,
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
                    "layout_combo": f"p{prefix_len}_b{between_len}",
                })
            quartets.append({
                "quartet_id": f"t13_{axis}:qt_{gid:06d}",
                "axis": axis,
                "layout_combo": f"p{prefix_len}_b{between_len}",
                "key_pair_tokens": sorted([k0["token_id"], k1["token_id"]]),
                "value_pair_tokens": sorted([v0["token_id"], v1["token_id"]]),
                "docs": docs,
            })
            gid += 1
    return quartets


def main() -> int:
    tokenizer = load_tokenizer()
    groups = original_run._identity_pool(tokenizer)
    filler = groups["filler"]
    filler_words = [str(item["value"]) for item in filler]
    verify_isolated_single_tokens(tokenizer, filler, "filler")
    keys, values = load_key_value_pools()
    bos = int(original_run.required_token_id(tokenizer, "<bos>"))

    prefix_range = range(TRAIN_PREFIX_RANGE[0], TRAIN_PREFIX_RANGE[1] + 1)
    between_range = range(TRAIN_BETWEEN_RANGE[0], TRAIN_BETWEEN_RANGE[1] + 1)
    train_combos = [(p, b) for p in prefix_range for b in between_range]

    train_quartets = build_pool(tokenizer, keys, values, filler_words,
                                train_combos, QUARTETS_PER_TRAIN_COMBO, "train", bos)
    retention_quartets = build_pool(tokenizer, keys, values, filler_words,
                                    RETENTION_COMBOS, QUARTETS_PER_RETENTION_COMBO, "retention", bos)

    train_docs = [d for q in train_quartets for d in q["docs"]]
    retention_docs = [d for q in retention_quartets for d in q["docs"]]
    train_digests = {hashlib.sha256(",".join(map(str, d["full_document_token_ids"])).encode()).hexdigest()
                     for d in train_docs}
    retention_digests = {hashlib.sha256(",".join(map(str, d["full_document_token_ids"])).encode()).hexdigest()
                         for d in retention_docs}
    require(not (train_digests & retention_digests), "train/retention token overlap")

    def pool(quartets, axis):
        return {
            "artifact_type": f"treatment13_{axis}_quartet_pool",
            "experiment": NAME,
            "seed": SEED,
            "quartet_count": len(quartets),
            "document_count": len([d for q in quartets for d in q["docs"]]),
            "quartets": quartets,
        }

    write_json(TRAIN_POOL_PATH, pool(train_quartets, "training"))
    write_json(RETENTION_POOL_PATH, pool(retention_quartets, "retention"))
    result = {
        "status": "TREATMENT13_GENERATOR_PASS",
        "experiment": NAME,
        "train_quartets": len(train_quartets),
        "train_documents": len(train_docs),
        "retention_quartets": len(retention_quartets),
        "retention_documents": len(retention_docs),
        "train_layout_combos": len(train_combos),
        "retention_layout_combos": len(RETENTION_COMBOS),
        "train_pool_sha256": sha256_file(TRAIN_POOL_PATH),
        "retention_pool_sha256": sha256_file(RETENTION_POOL_PATH),
        "train_retention_token_overlap": 0,
    }
    write_json(GENERATOR_RESULT_PATH, result)
    print(result)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"GENERATOR ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
