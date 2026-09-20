from __future__ import annotations

"""Stack2 R4: WHO sentences + native HAS + held-out paraphrases on frozen s5b3+r3b."""

from .data_language_bridge import score_bind_sentence, score_has_sentence
from .selection_language_bridge import greedy_decode_until_stop
from .data import BOS
from .data_language_bridge import encode_ids


# Train-side wording (not scored). Held-out packs below use disjoint wrappers.
TRAIN_PARAPHRASE_STEMS = (
    "Who looks {value}?",
    "Tell me who looks {value}?",
    "What does {entity} have?",
    "Which one has the {value} object?",
    "Who is beside the {value}?",
    "Where is {entity}?",
)


def build_who_paraphrase_heldout() -> list[dict]:
    """Unseen WHO wrappers. Eval-only."""
    return [
        {"id": "p_who_animal_white", "facts": "dog looks white. hen looks red.", "entity": "dog", "value": "white", "query": "Which animal is white?", "kind": "who", "family": "who"},
        {"id": "p_who_one_red", "facts": "dog looks white. hen looks red.", "entity": "hen", "value": "red", "query": "Which one is red?", "kind": "who", "family": "who"},
        {"id": "p_who_is_blue", "facts": "Remember: the bear is blue. Remember: the cat is yellow.", "entity": "bear", "value": "blue", "query": "Who is blue?", "kind": "who", "family": "who"},
        {"id": "p_who_animal_yellow", "facts": "Remember: the bear is blue. Remember: the cat is yellow.", "entity": "cat", "value": "yellow", "query": "Which animal is yellow?", "kind": "who", "family": "who"},
        {"id": "p_who_one_green", "facts": "That duck is green. That frog is pink.", "entity": "duck", "value": "green", "query": "Which one is green?", "kind": "who", "family": "who"},
        {"id": "p_who_is_pink", "facts": "That duck is green. That frog is pink.", "entity": "frog", "value": "pink", "query": "Who is pink?", "kind": "who", "family": "who"},
        {"id": "p_who_animal_huge", "facts": "Remember: the dog is huge in size. That hen is tiny in size.", "entity": "dog", "value": "huge", "query": "Which animal is huge?", "kind": "who", "family": "who"},
        {"id": "p_who_one_thin", "facts": "bear looks wide in size. cat looks thin in size.", "entity": "cat", "value": "thin", "query": "Which one is thin?", "kind": "who", "family": "who"},
    ]


def build_has_paraphrase_heldout() -> list[dict]:
    """Unseen HAS wrappers. Eval-only."""
    return [
        {"id": "p_has_obj_dog", "facts": "That dog has the white object. Remember: the hen has the red object.", "entity": "dog", "value": "white", "query": "What object does the dog have?", "kind": "has", "family": "has"},
        {"id": "p_has_belong_hen", "facts": "That dog has the white object. Remember: the hen has the red object.", "entity": "hen", "value": "red", "query": "Which object belongs to the hen?", "kind": "has", "family": "has"},
        {"id": "p_has_obj_bear", "facts": "That bear has the blue object. Remember: the cat has the yellow object.", "entity": "bear", "value": "blue", "query": "What object does the bear have?", "kind": "has", "family": "has"},
        {"id": "p_has_belong_cat", "facts": "That duck has the pink object. Remember: the cat has the green object.", "entity": "cat", "value": "green", "query": "Which object belongs to the cat?", "kind": "has", "family": "has"},
        {"id": "p_has_obj_frog", "facts": "That frog has the pink object. Remember: the duck has the green object.", "entity": "frog", "value": "pink", "query": "What object does the frog have?", "kind": "has", "family": "has"},
        {"id": "p_has_belong_duck", "facts": "That frog has the yellow object. Remember: the duck has the pink object.", "entity": "duck", "value": "pink", "query": "Which object belongs to the duck?", "kind": "has", "family": "has"},
    ]


def build_beside_paraphrase_heldout() -> list[dict]:
    """Unseen BESIDE wrappers. Eval-only."""
    return [
        {"id": "p_bes_next_hen", "facts": "That dog is beside the hen.", "entity": "dog", "value": "hen", "query": "Who is next to the hen?", "kind": "beside", "family": "beside"},
        {"id": "p_bes_animal_cat", "facts": "Remember: the bear is beside the cat.", "entity": "bear", "value": "cat", "query": "Which animal is next to the cat?", "kind": "beside", "family": "beside"},
        {"id": "p_bes_next_frog", "facts": "That duck is beside the frog.", "entity": "duck", "value": "frog", "query": "Who is next to the frog?", "kind": "beside", "family": "beside"},
        {"id": "p_bes_animal_dog", "facts": "Remember: the cat is beside the dog.", "entity": "cat", "value": "dog", "query": "Which animal is next to the dog?", "kind": "beside", "family": "beside"},
        {"id": "p_bes_next_duck", "facts": "That hen is beside the duck.", "entity": "hen", "value": "duck", "query": "Who is next to the duck?", "kind": "beside", "family": "beside"},
        {"id": "p_bes_animal_bear", "facts": "Remember: the frog is beside the bear.", "entity": "frog", "value": "bear", "query": "Which animal is next to the bear?", "kind": "beside", "family": "beside"},
    ]


def score_paraphrase_pack(model, tokenizer, device, items: list[dict], *, family: str) -> dict:
    rows = []
    for item in items:
        prompt = f"{item['facts']} {item['query']}"
        ids = [BOS, *encode_ids(tokenizer, prompt)]
        emitted, stopped = greedy_decode_until_stop(model, ids, device, tokenizer, max_new=20)
        decoded = tokenizer.decode(emitted, skip_special_tokens=True)
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
        rows.append({"id": item["id"], "decoded": decoded.strip(), **scored})
    n = len(rows) or 1
    return {
        "n": len(rows),
        "bare": sum(bool(r["sentence_ok"]) for r in rows) / n,
        "first_entity": sum(bool(r["first_entity"]) for r in rows) / n,
        "rows": rows,
    }
