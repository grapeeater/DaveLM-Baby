r"""Treatment-10 generator: strict counterfactual binding quartet universe.

Constructs a NEW training universe and a NEW held-out retention universe.
Nothing here loads a model or checkpoint, and nothing reads any historical
treatment outcome. Output is purely structural (token arrays + identities).

Two source identities K0/K1 and two value identities V0/V1 form a matched
quartet:
    Orientation 1 (natural):   Mapping K0->V0, K1->V1; Query K0->V0, K1->V1
    Orientation 2 (reversal):  Mapping K0->V1, K1->V0; Query K0->V1, K1->V0
Within a quartet, filler, template, mapping order, mapping positions, query
position, answer position, and document length are identical; only the two
mapping-value identities and the matching answer change between orientations.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List
from itertools import combinations

from treatment10_config import (
    GENERATOR_RESULT_PATH,
    KEY_COUNT,
    MODEL_VISIBLE_DOCUMENT_LENGTH,
    NAME,
    OUT_ROOT,
    RAW_DOCUMENT_TOKEN_COUNT,
    RETENTION_COUNT,
    RETENTION_POOL_PATH,
    RETENTION_SUBSET_STEP,
    SEED,
    TRAIN_POOL_PATH,
    VALUE_COUNT,
)
from treatment10_common import (
    Treatment10Error,
    build_member_doc,
    draw_words,
    load_identity_groups,
    load_tokenizer,
    original_run,
    require,
    sha256_file,
    verify_isolated_single_tokens,
    write_json,
)

MEMBER_ORDER = ("o1_k0", "o1_k1", "o2_k0", "o2_k1")


def _array_digest(ids: List[int]) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(",".join(str(int(t)) for t in ids).encode("ascii"))
    return digest.hexdigest()


def build_doc(
    tokenizer,
    mapping_lines,
    query_key,
    answer_value,
    prefix_words,
    between_words,
    tail_words,
):
    raw_ids = build_member_doc(
        tokenizer, mapping_lines, query_key, answer_value,
        prefix_words, between_words, tail_words,
    )
    bos = int(original_run.required_token_id(tokenizer, "<bos>"))
    return [bos] + raw_ids


def geometry_params(seed_token: int) -> Dict[str, Any]:
    prefix_len = 8 + (seed_token % 5)
    between_len = 3 + ((seed_token // 5) % 4)
    return {
        "prefix_len": prefix_len,
        "between_len": between_len,
        "prefix_seed": 900_000 + seed_token,
        "between_seed": 910_000 + seed_token,
        "tail_seed": 920_000 + seed_token,
    }


def enumerate_quartet_specs():
    """Return ordered list of (key_indices, value_indices, order_mode)."""
    specs = []
    key_pairs = list(combinations(range(KEY_COUNT), 2))
    value_pairs = list(combinations(range(VALUE_COUNT), 2))
    for kp in key_pairs:
        for vp in value_pairs:
            for order_mode in (0, 1):
                specs.append((kp, vp, order_mode))
    return specs


def main() -> int:
    tokenizer = load_tokenizer()
    identities, filler = load_identity_groups(tokenizer)
    filler_words = [str(item["value"]) for item in filler]
    verify_isolated_single_tokens(tokenizer, filler, "filler")

    key_items = identities[:KEY_COUNT]
    value_items = identities[KEY_COUNT: KEY_COUNT + VALUE_COUNT]
    keys = [{"word": str(item["value"]), "token_id": int(item["token_id"])} for item in key_items]
    values = [{"word": str(item["value"]), "token_id": int(item["token_id"])} for item in value_items]

    specs = enumerate_quartet_specs()
    specs_all = specs
    train_specs = list(specs_all)
    retention_specs = specs_all[::RETENTION_SUBSET_STEP]
    require(len(train_specs) == RETENTION_COUNT * (len(train_specs) // RETENTION_COUNT),
            "unexpected train spec count")
    require(len(retention_specs) == RETENTION_COUNT,
            f"retention spec count != RETENTION_COUNT ({len(retention_specs)})")

    def make_quartet(spec, axis, seed_offset, index):
        kp, vp, order_mode = spec
        k0, k1 = keys[kp[0]], keys[kp[1]]
        v0, v1 = values[vp[0]], values[vp[1]]
        seed_token = seed_offset + index
        geom = geometry_params(seed_token)
        prefix_words = draw_words(filler_words, geom["prefix_len"], geom["prefix_seed"])
        between_words = draw_words(filler_words, geom["between_len"], geom["between_seed"])
        tail_words = draw_words(filler_words, 160, geom["tail_seed"])

        # Mapping lines depend on orientation and on which key is first.
        line0_key = k0 if order_mode == 0 else k1
        line1_key = k1 if order_mode == 0 else k0

        def value_for_key(key_token_id, orientation):
            if orientation == 1:
                return v0 if key_token_id == k0["token_id"] else v1
            return v1 if key_token_id == k0["token_id"] else v0

        def member_doc(orientation, query_key):
            lines = []
            for line_key in (line0_key, line1_key):
                val = value_for_key(line_key["token_id"], orientation)
                lines.append([line_key["word"], val["word"]])
            ans = value_for_key(query_key["token_id"], orientation)
            model_ids = build_doc(
                tokenizer, lines, query_key["word"], ans["word"],
                prefix_words, between_words, tail_words,
            )
            require(len(model_ids) == MODEL_VISIBLE_DOCUMENT_LENGTH,
                    "model-visible length != 193.")
            return model_ids, lines, ans

        docs = []
        for member in MEMBER_ORDER:
            orientation = 1 if member.startswith("o1") else 2
            query_key = k0 if member.endswith("k0") else k1
            model_ids, lines, ans = member_doc(orientation, query_key)
            other_key = k1 if query_key["token_id"] == k0["token_id"] else k0
            distractor_value = value_for_key(other_key["token_id"], orientation)
            qtok = query_key["token_id"]
            candidate_pair = sorted([v0["token_id"], v1["token_id"]])

            # Geometry by scanning the model-visible ids for the four tokens.
            occ = {
                "k0": [i for i, x in enumerate(model_ids) if x == k0["token_id"]],
                "k1": [i for i, x in enumerate(model_ids) if x == k1["token_id"]],
                "v0": [i for i, x in enumerate(model_ids) if x == v0["token_id"]],
                "v1": [i for i, x in enumerate(model_ids) if x == v1["token_id"]],
            }
            query_key_label = "k0" if query_key["token_id"] == k0["token_id"] else "k1"
            other_key_label = "k1" if query_key_label == "k0" else "k0"
            target_label = "v0" if ans["token_id"] == v0["token_id"] else "v1"
            distractor_label = "v1" if target_label == "v0" else "v0"

            require(len(occ[query_key_label]) == 2,
                    f"{axis}/{index}/{member}: query token count != 2.")
            require(len(occ[target_label]) == 2,
                    f"{axis}/{index}/{member}: target value count != 2.")
            require(len(occ[other_key_label]) == 1,
                    f"{axis}/{index}/{member}: nonqueried key count != 1.")
            require(len(occ[distractor_label]) == 1,
                    f"{axis}/{index}/{member}: distractor value count != 1.")
            q_clause, qdp = min(occ[query_key_label]), max(occ[query_key_label])
            t_clause, answer_index = min(occ[target_label]), max(occ[target_label])
            require(q_clause < qdp, "query clause not before query line.")
            require(t_clause < answer_index, "value clause not before answer.")
            require(q_clause < t_clause, "mapping clause not before value.")
            # Mapping clause for this key precedes the answer and lies before qdp.
            distractor_value_pos = occ[distractor_label][0]
            other_key_pos = occ[other_key_label][0]

            docs.append({
                "doc_id": f"{axis}:qt_{index:06d}:{member}",
                "quartet_id": f"{axis}:qt_{index:06d}",
                "member": member,
                "axis": axis,
                "orientation": orientation,
                "query_key_word": query_key["word"],
                "query_key_token": qtok,
                "target_value_word": ans["word"],
                "target_value_token": ans["token_id"],
                "distractor_value_token": distractor_value["token_id"],
                "distractor_value_word": distractor_value["word"],
                "candidate_pair_sorted": candidate_pair,
                "query_slot": 0 if line0_key["token_id"] == qtok else 1,
                "mapping_order": order_mode,
                "qdp": qdp,
                "query_key_clause_pos": q_clause,
                "target_clause_pos": t_clause,
                "answer_token_index": answer_index,
                "answer_causal_position": answer_index - 1,
                "distractor_value_pos": distractor_value_pos,
                "other_key_pos": other_key_pos,
                "raw_document_token_count": RAW_DOCUMENT_TOKEN_COUNT,
                "model_visible_document_length": len(model_ids),
                "full_document_token_ids": model_ids,
                "document_digest": _array_digest(model_ids),
            })

        return {
            "quartet_id": f"{axis}:qt_{index:06d}",
            "axis": axis,
            "key_pair_tokens": sorted([k0["token_id"], k1["token_id"]]),
            "key_words": [k0["word"], k1["word"]],
            "value_pair_tokens": sorted([v0["token_id"], v1["token_id"]]),
            "value_words": [v0["word"], v1["word"]],
            "mapping_order": order_mode,
            "geometry": geom,
            "docs": docs,
        }

    train_quartets = []
    train_digests = set()
    for index, spec in enumerate(train_specs):
        quartet = make_quartet(spec, "train", 0, index)
        train_quartets.append(quartet)
        for doc in quartet["docs"]:
            require(doc["document_digest"] not in train_digests, "duplicate train document")
            train_digests.add(doc["document_digest"])

    retention_quartets = []
    retention_digests = set()
    for index, spec in enumerate(retention_specs):
        quartet = make_quartet(spec, "retention", 2_000_000_000, index)
        retention_quartets.append(quartet)
        for doc in quartet["docs"]:
            retention_digests.add(doc["document_digest"])

    overlap = train_digests & retention_digests
    require(not overlap, f"retention/training document overlap: {len(overlap)}")

    train_pool = {
        "artifact_type": "treatment10_training_quartet_pool",
        "experiment": NAME,
        "seed": SEED,
        "quartet_count": len(train_quartets),
        "document_count": len(train_digests),
        "quartets": train_quartets,
    }
    retention_pool = {
        "artifact_type": "treatment10_retention_quartet_pool",
        "experiment": NAME,
        "seed": SEED,
        "quartet_count": len(retention_quartets),
        "document_count": len(retention_digests),
        "quartets": retention_quartets,
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(TRAIN_POOL_PATH, train_pool)
    write_json(RETENTION_POOL_PATH, retention_pool)

    generator_result = {
        "experiment": NAME,
        "status": "GENERATOR_PASS",
        "train_pool_path": str(TRAIN_POOL_PATH),
        "retention_pool_path": str(RETENTION_POOL_PATH),
        "train_pool_sha256": sha256_file(TRAIN_POOL_PATH),
        "retention_pool_sha256": sha256_file(RETENTION_POOL_PATH),
        "train_quartets": len(train_quartets),
        "train_documents": len(train_digests),
        "retention_quartets": len(retention_quartets),
        "retention_documents": len(retention_digests),
        "retention_training_overlap": len(overlap),
        "keys": [{"word": k["word"], "token_id": k["token_id"]} for k in keys],
        "values": [{"word": v["word"], "token_id": v["token_id"]} for v in values],
        "note": "Structural generation only. No model, checkpoint, or behavioral outcome used.",
    }
    write_json(GENERATOR_RESULT_PATH, generator_result)
    print(generator_result)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Treatment10Error as exc:
        print(f"GENERATOR ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
