from __future__ import annotations

"""Stack2 R5: native BESIDE partner, size-WHO harden, mixed-relation switch."""

from .data_language_bridge import score_bind_sentence, score_has_sentence
from .data import BOS
from .data_language_bridge import encode_ids
from .selection_language_bridge import greedy_decode_until_stop


def build_beside_order_heldout() -> list[dict]:
    """Landmark first and last. Eval-only. Disjoint from the official pack wording."""
    return [
        {"id": "ord_last_dog", "facts": "A hen sits. The cat is beside the dog.", "entity": "cat", "value": "dog", "query": "Who is beside the dog?", "kind": "beside_who", "landmark": "last"},
        {"id": "ord_first_dog", "facts": "A hen sits. The dog is beside the cat.", "entity": "cat", "value": "dog", "query": "Who is beside the dog?", "kind": "beside_who", "landmark": "first"},
        {"id": "ord_last_hen", "facts": "A bear sits. The duck is beside the hen.", "entity": "duck", "value": "hen", "query": "Which animal is next to the hen?", "kind": "beside_who", "landmark": "last"},
        {"id": "ord_first_hen", "facts": "A bear sits. The hen is beside the duck.", "entity": "duck", "value": "hen", "query": "Which animal is next to the hen?", "kind": "beside_who", "landmark": "first"},
        {"id": "ord_where_last", "facts": "A frog sits. The bear is beside the cat.", "entity": "bear", "value": "cat", "query": "Where is the bear?", "kind": "beside_where", "landmark": "subject"},
        {"id": "ord_where_obj", "facts": "A frog sits. The cat is beside the bear.", "entity": "cat", "value": "bear", "query": "Where is the cat?", "kind": "beside_where", "landmark": "subject"},
        {"id": "ord_who_obj_first", "facts": "A duck sits. The frog is beside the bear.", "entity": "bear", "value": "frog", "query": "Who is beside the frog?", "kind": "beside_who", "landmark": "first"},
        {"id": "ord_who_obj_last", "facts": "A duck sits. The bear is beside the frog.", "entity": "bear", "value": "frog", "query": "Who is beside the frog?", "kind": "beside_who", "landmark": "last"},
    ]


def score_beside_items(model, tokenizer, device, items: list[dict]) -> dict:
    rows = []
    for item in items:
        prompt = f"{item['facts']} {item['query']}"
        ids = [BOS, *encode_ids(tokenizer, prompt)]
        emitted, stopped = greedy_decode_until_stop(model, ids, device, tokenizer, max_new=20)
        decoded = tokenizer.decode(emitted, skip_special_tokens=True)
        scored = score_bind_sentence(
            full_text=decoded,
            entity=item["entity"],
            value=item["value"],
            stopped=stopped,
            predicates=("beside", "next"),
        )
        rows.append({"id": item["id"], "decoded": decoded.strip(), "kind": item.get("kind"), **scored})
    n = len(rows) or 1
    return {
        "n": len(rows),
        "bare": sum(bool(r["sentence_ok"]) for r in rows) / n,
        "first_entity": sum(bool(r["first_entity"]) for r in rows) / n,
        "rows": rows,
    }


def build_mixed_relation_heldout() -> list[dict]:
    """Same context, different relation queries. Eval-only. Last fact is not always gold."""
    return [
        {
            "id": "mix2_who_after_has",
            "facts": "The dog is white. The cat has the red object.",
            "query": "Who is white?",
            "entity": "dog",
            "value": "white",
            "family": "who",
            "gold_last": False,
        },
        {
            "id": "mix2_has_after_who",
            "facts": "The dog is white. The cat has the red object.",
            "query": "What does the cat have?",
            "entity": "cat",
            "value": "red",
            "family": "has",
            "gold_last": True,
        },
        {
            "id": "mix2_who_recency_trap",
            "facts": "The hen has the green object. The duck is pink.",
            "query": "Who is pink?",
            "entity": "duck",
            "value": "pink",
            "family": "who",
            "gold_last": True,
        },
        {
            "id": "mix2_has_not_last",
            "facts": "The hen has the green object. The duck is pink.",
            "query": "What does the hen have?",
            "entity": "hen",
            "value": "green",
            "family": "has",
            "gold_last": False,
        },
        {
            "id": "mix2_bes_after_who",
            "facts": "The bear is blue. The frog is beside the bear.",
            "query": "Who is beside the bear?",
            "entity": "frog",
            "value": "bear",
            "family": "beside",
            "gold_last": True,
        },
        {
            "id": "mix2_who_not_beside",
            "facts": "The bear is blue. The frog is beside the bear.",
            "query": "Who is blue?",
            "entity": "bear",
            "value": "blue",
            "family": "who",
            "gold_last": False,
        },
        {
            "id": "mix3_who",
            "facts": "The dog is white. The cat has the red object. The bear is beside the dog.",
            "query": "Who is white?",
            "entity": "dog",
            "value": "white",
            "family": "who",
            "gold_last": False,
        },
        {
            "id": "mix3_has",
            "facts": "The dog is white. The cat has the red object. The bear is beside the dog.",
            "query": "What does the cat have?",
            "entity": "cat",
            "value": "red",
            "family": "has",
            "gold_last": False,
        },
        {
            "id": "mix3_bes",
            "facts": "The dog is white. The cat has the red object. The bear is beside the dog.",
            "query": "Who is beside the dog?",
            "entity": "bear",
            "value": "dog",
            "family": "beside",
            "gold_last": True,
        },
        {
            "id": "mix3_bes_not_last",
            "facts": "The bear is beside the dog. The dog is white. The cat has the red object.",
            "query": "Who is beside the dog?",
            "entity": "bear",
            "value": "dog",
            "family": "beside",
            "gold_last": False,
        },
        {
            "id": "mix3_has_last",
            "facts": "The bear is beside the dog. The dog is white. The cat has the red object.",
            "query": "What does the cat have?",
            "entity": "cat",
            "value": "red",
            "family": "has",
            "gold_last": True,
        },
        {
            "id": "mix3_who_middle",
            "facts": "The bear is beside the dog. The dog is white. The cat has the red object.",
            "query": "Who is white?",
            "entity": "dog",
            "value": "white",
            "family": "who",
            "gold_last": False,
        },
        {
            "id": "mix3_overlap_who",
            "facts": "The hen is tiny. The hen has the yellow object. The duck is beside the hen.",
            "query": "Who is tiny?",
            "entity": "hen",
            "value": "tiny",
            "family": "who",
            "gold_last": False,
        },
        {
            "id": "mix3_overlap_has",
            "facts": "The hen is tiny. The hen has the yellow object. The duck is beside the hen.",
            "query": "What does the hen have?",
            "entity": "hen",
            "value": "yellow",
            "family": "has",
            "gold_last": False,
        },
        {
            "id": "mix3_overlap_bes",
            "facts": "The hen is tiny. The hen has the yellow object. The duck is beside the hen.",
            "query": "Which animal is next to the hen?",
            "entity": "duck",
            "value": "hen",
            "family": "beside",
            "gold_last": True,
        },
        {
            "id": "mix3_para_who",
            "facts": "The frog is pink. The frog has the blue object. The cat is beside the frog.",
            "query": "Which animal is pink?",
            "entity": "frog",
            "value": "pink",
            "family": "who",
            "gold_last": False,
        },
    ]


def score_mixed_pack(model, tokenizer, device, items: list[dict]) -> dict:
    rows = []
    for item in items:
        prompt = f"{item['facts']} {item['query']}"
        ids = [BOS, *encode_ids(tokenizer, prompt)]
        emitted, stopped = greedy_decode_until_stop(model, ids, device, tokenizer, max_new=20)
        decoded = tokenizer.decode(emitted, skip_special_tokens=True)
        family = item["family"]
        if family == "has":
            scored = score_has_sentence(full_text=decoded, entity=item["entity"], value=item["value"], stopped=stopped)
        elif family == "beside":
            scored = score_bind_sentence(
                full_text=decoded,
                entity=item["entity"],
                value=item["value"],
                stopped=stopped,
                predicates=("beside", "next"),
            )
        else:
            scored = score_bind_sentence(
                full_text=decoded,
                entity=item["entity"],
                value=item["value"],
                stopped=stopped,
                predicates=("is", "looks"),
            )
        rows.append(
            {
                "id": item["id"],
                "family": family,
                "decoded": decoded.strip(),
                "gold_last": item.get("gold_last"),
                **scored,
            }
        )
    n = len(rows) or 1
    by = {}
    for fam in ("who", "has", "beside"):
        sub = [r for r in rows if r["family"] == fam]
        by[fam] = sum(bool(r["sentence_ok"]) for r in sub) / len(sub) if sub else None
    return {
        "n": len(rows),
        "bare": sum(bool(r["sentence_ok"]) for r in rows) / n,
        "first_entity": sum(bool(r["first_entity"]) for r in rows) / n,
        "by_family": by,
        "rows": rows,
    }


def score_mixed_multiturn(model, tokenizer, device) -> dict:
    facts = "The dog is white. The cat has the red object. The bear is beside the dog."
    turns = [
        {"query": "Who is white?", "entity": "dog", "value": "white", "family": "who"},
        {"query": "What does the cat have?", "entity": "cat", "value": "red", "family": "has"},
        {"query": "Who is beside the dog?", "entity": "bear", "value": "dog", "family": "beside"},
    ]
    context = facts
    rows = []
    for turn in turns:
        prompt = f"{context} {turn['query']}"
        ids = [BOS, *encode_ids(tokenizer, prompt)]
        emitted, stopped = greedy_decode_until_stop(model, ids, device, tokenizer, max_new=20)
        decoded = tokenizer.decode(emitted, skip_special_tokens=True).strip()
        if turn["family"] == "has":
            scored = score_has_sentence(full_text=decoded, entity=turn["entity"], value=turn["value"], stopped=stopped)
        elif turn["family"] == "beside":
            scored = score_bind_sentence(
                full_text=decoded,
                entity=turn["entity"],
                value=turn["value"],
                stopped=stopped,
                predicates=("beside", "next"),
            )
        else:
            scored = score_bind_sentence(
                full_text=decoded,
                entity=turn["entity"],
                value=turn["value"],
                stopped=stopped,
                predicates=("is", "looks"),
            )
        rows.append({"query": turn["query"], "family": turn["family"], "decoded": decoded, **scored})
        context = f"{context} {turn['query']} {decoded}"
    n = len(rows) or 1
    return {
        "n": len(rows),
        "bare": sum(bool(r["sentence_ok"]) for r in rows) / n,
        "rows": rows,
    }
