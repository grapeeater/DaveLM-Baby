from __future__ import annotations

"""Stack2 R7: generalized inverse entity binding.

Selection identifies the bound entity from a queried property/value/relation.
Finish wording is a separate canned plan. RelAssist/WhoProp stay off.
Does not open TEST/FINAL/SACRED. Does not copy Form A.
"""

from .data import BOS
from .data_language_bridge import ENTITIES, _usable_words, encode_ids, score_has_sentence
from .selection_language_bridge import greedy_decode_until_stop


def first_bridge_entity(text: str) -> str | None:
    for word in _usable_words(text):
        if word in ENTITIES:
            return word
    return None


def score_entity_selection(decoded: str, item: dict) -> dict:
    """Pass only if the correct entity is identified. Wrong entity + right property fails."""
    text = decoded.lower()
    words = _usable_words(decoded)
    entity = str(item.get("entity") or "").strip().lower()
    value = str(item.get("value") or "").strip().lower()
    landmark = str(item.get("landmark") or "").strip().lower()
    first = first_bridge_entity(decoded)
    family = item.get("family") or "who"
    entity_ok = bool(entity) and entity in words
    first_ok = first == entity if entity else first is None
    value_ok = (not value) or value in text
    landmark_lead = bool(landmark) and first == landmark and landmark != entity
    if family in {"combine", "story_combine", "direct", "color", "size"}:
        ok = value_ok
    elif family == "about":
        needed = list(item.get("colors") or []) + list(item.get("sizes") or [])
        have = [word for word in needed if word in text]
        ok = entity_ok and first_ok and len(have) == len(needed) and bool(needed)
    elif family in {"has", "has_value", "has_entity"}:
        scored = score_has_sentence(full_text=decoded, entity=entity, value=value, stopped=True)
        ok = bool(scored["sentence_ok"]) and first_ok and not landmark_lead
    elif family == "beside":
        rel_ok = "beside" in words or "next" in words
        ok = entity_ok and first_ok and rel_ok and not landmark_lead
    elif family in {"who", "who_sent", "who_inv"}:
        ok = entity_ok and first_ok
    else:
        ok = entity_ok and first_ok
    return {
        "entity_ok": entity_ok,
        "first_ok": first_ok,
        "first_entity": first,
        "value_ok": value_ok,
        "landmark_lead": landmark_lead,
        "ok": bool(ok),
    }


def score_r7_items(model, tokenizer, device, items: list[dict]) -> dict:
    rows = []
    for item in items:
        prompt = item.get("prompt") or f"{item['facts']} {item['query']}"
        ids = [BOS, *encode_ids(tokenizer, prompt)]
        emitted, stopped = greedy_decode_until_stop(model, ids, device, tokenizer, max_new=24)
        decoded = tokenizer.decode(emitted, skip_special_tokens=True).strip()
        scored = score_entity_selection(decoded, item)
        rows.append(
            {
                "id": item["id"],
                "family": item.get("family"),
                "decoded": decoded,
                "stopped": stopped,
                "gold_entity": item.get("entity"),
                **scored,
            }
        )
    n = len(rows) or 1
    by: dict[str, float] = {}
    for fam in sorted({str(r["family"]) for r in rows}):
        sub = [r for r in rows if str(r["family"]) == fam]
        by[fam] = sum(bool(r["ok"]) for r in sub) / len(sub)
    return {
        "n": len(rows),
        "overall": sum(bool(r["ok"]) for r in rows) / n,
        "entity_lead": sum(bool(r["first_ok"]) for r in rows) / n,
        "by_family": by,
        "miss": [r for r in rows if not r["ok"]],
        "rows": rows,
    }


def score_r7_reuse(model, tokenizer, device, scripts: list[dict]) -> dict:
    rows = []
    for script in scripts:
        context = script["facts"]
        for turn in script["turns"]:
            prompt = f"{context}\nHuman: {turn['query']}\nBaby:"
            ids = [BOS, *encode_ids(tokenizer, prompt)]
            emitted, stopped = greedy_decode_until_stop(model, ids, device, tokenizer, max_new=20)
            decoded = tokenizer.decode(emitted, skip_special_tokens=True).strip()
            item = {
                "family": turn.get("family") or "who",
                "entity": turn.get("entity") or turn.get("gold"),
                "value": turn.get("value") or turn.get("gold"),
                "landmark": turn.get("landmark"),
                "colors": turn.get("colors"),
                "sizes": turn.get("sizes"),
            }
            scored = score_entity_selection(decoded, item)
            if (turn.get("family") or "qa") in {"color", "size", "direct", "combine"}:
                scored["ok"] = turn["gold"].lower() in decoded.lower()
            rows.append(
                {
                    "id": script["id"],
                    "query": turn["query"],
                    "gold": turn.get("gold"),
                    "decoded": decoded,
                    "stopped": stopped,
                    "come_back": bool(turn.get("come_back")),
                    "family": turn.get("family") or "qa",
                    **scored,
                }
            )
            context = f"{context} {turn['query']} {decoded if decoded else turn.get('gold') or ''}"
    n = len(rows) or 1
    come = [r for r in rows if r["come_back"]]
    return {
        "n": len(rows),
        "ok": sum(bool(r["ok"]) for r in rows) / n,
        "reuse": (sum(bool(r["ok"]) for r in come) / len(come)) if come else None,
        "stop": sum(bool(r["stopped"]) for r in rows) / n,
        "miss": [r for r in rows if not r["ok"]],
        "rows": rows,
    }


def build_r7_mechanism_canary() -> list[dict]:
    """Tiny independently generated mechanism probes. Not Form A. Not the frozen pack."""
    return [
        {
            "id": "m_who_left_fullvocab",
            "family": "who",
            "facts": "The cow is blue. The pig is red.",
            "query": "Who is red?",
            "entity": "pig",
            "value": "red",
        },
        {
            "id": "m_who_value_before",
            "family": "who",
            "facts": "A pink fox hid. A tiny cow sat.",
            "query": "Who is pink?",
            "entity": "fox",
            "value": "pink",
        },
        {
            "id": "m_who_middle_3e",
            "family": "who",
            "facts": "The cat is blue. The pig is red. The dog is white.",
            "query": "Which one is red?",
            "entity": "pig",
            "value": "red",
        },
        {
            "id": "m_has_value",
            "family": "has_value",
            "facts": "The pig has the red object. The cow has the white object.",
            "query": "What does the pig have?",
            "entity": "pig",
            "value": "red",
        },
        {
            "id": "m_has_entity",
            "family": "has_entity",
            "facts": "The pig has the red object. The cow has the white object.",
            "query": "Which one has the white object?",
            "entity": "cow",
            "value": "white",
        },
        {
            "id": "m_bes_partner",
            "family": "beside",
            "facts": "The fox is beside the duck. The pig is red.",
            "query": "Who is beside the duck?",
            "entity": "fox",
            "value": "duck",
            "landmark": "duck",
        },
        {
            "id": "m_bes_no_stem",
            "family": "beside",
            "facts": "The fox is beside the duck. The pig is red.",
            "query": "Who is near the duck?",
            "entity": "fox",
            "value": "duck",
            "landmark": "duck",
        },
        {
            "id": "m_bes_where",
            "family": "beside",
            "facts": "The fox is beside the duck. The pig is red.",
            "query": "Where is the fox?",
            "entity": "fox",
            "value": "duck",
            "landmark": "fox",
        },
        {
            "id": "m_direct_collision",
            "family": "direct",
            "facts": "That bird is tiny in size. fox looks white.",
            "query": "Tell me the size of the bird.",
            "entity": "bird",
            "value": "tiny",
        },
        {
            "id": "m_combine_hold",
            "family": "combine",
            "facts": "The hen is yellow. The hen is short. The duck is pink. The duck is wide.",
            "query": "Which color is the short one?",
            "entity": "hen",
            "value": "yellow",
        },
        {
            "id": "m_about_hold",
            "family": "about",
            "facts": "The bear is green. The frog is white. The bear is thin.",
            "query": "What do you know about the bear?",
            "entity": "bear",
            "value": "green",
            "colors": ["green"],
            "sizes": ["thin"],
        },
        {
            "id": "m_then_about",
            "family": "about",
            "facts": "The cat is blue. The dog is red.",
            "query": "the dog then?",
            "entity": "dog",
            "value": "red",
            "colors": ["red"],
            "sizes": [],
        },
    ]


def build_r7_integration_pack() -> list[dict]:
    """Fresh developmental mix. Frozen by construction in source. Not trained. Not Form A.

    Broader than the 22-item R6 pack: full bridge entities, both orderings, 2e/3e+,
    HAS-value vs HAS-entity, relation direction, collision-prone names, combine/story,
    about, mixed scenes, and short multi-turn items scored separately.
    """
    return [
        {"id": "r7_who_e2_late", "family": "who", "facts": "The hen is yellow. The duck is pink.", "query": "Who is pink?", "entity": "duck", "value": "pink"},
        {"id": "r7_who_e2_early", "family": "who", "facts": "The hen is yellow. The duck is pink.", "query": "Who is yellow?", "entity": "hen", "value": "yellow"},
        {"id": "r7_who_e3_mid", "family": "who", "facts": "The cat is blue. The pig is red. The dog is white.", "query": "Which one is red?", "entity": "pig", "value": "red"},
        {"id": "r7_who_e3_first", "family": "who", "facts": "The cow is green. The frog is white. The bear is thin.", "query": "Who is green?", "entity": "cow", "value": "green"},
        {"id": "r7_who_adj_fox", "family": "who", "facts": "A pink fox hid. A tiny cow sat.", "query": "Tell me who is pink.", "entity": "fox", "value": "pink"},
        {"id": "r7_who_adj_pig", "family": "who", "facts": "Long ago a huge pig ran. That pig was white.", "query": "Who looks huge?", "entity": "pig", "value": "huge"},
        {"id": "r7_who_size_bird", "family": "who", "facts": "Remember: the frog is wide in size. That bird is tiny in size.", "query": "Name the tiny one.", "entity": "bird", "value": "tiny"},
        {"id": "r7_who_sent_hold", "family": "who_sent", "facts": "Remember: the dog is huge in size. That hen is tiny in size.", "query": "Tell me who looks tiny.", "entity": "hen", "value": "tiny"},
        {"id": "r7_who_para", "family": "who", "facts": "That bear is green. That frog is white.", "query": "Which animal is green?", "entity": "bear", "value": "green"},
        {"id": "r7_who_collision_pig", "family": "who", "facts": "The duck is yellow. The pig is red.", "query": "Who is red?", "entity": "pig", "value": "red"},
        {"id": "r7_has_value_cat", "family": "has_value", "facts": "The cat has the blue object. The dog has the white object.", "query": "What does the cat have?", "entity": "cat", "value": "blue"},
        {"id": "r7_has_value_belong", "family": "has_value", "facts": "That hen has the yellow object. Remember: the duck has the pink object.", "query": "Which object belongs to the duck?", "entity": "duck", "value": "pink"},
        {"id": "r7_has_value_pig", "family": "has_value", "facts": "The pig has the red object. The cow has the green object.", "query": "What does the pig have?", "entity": "pig", "value": "red"},
        {"id": "r7_has_entity_white", "family": "has_entity", "facts": "The cat has the blue object. The dog has the white object.", "query": "Which one has the white object?", "entity": "dog", "value": "white"},
        {"id": "r7_has_entity_pig", "family": "has_entity", "facts": "The pig has the red object. The cow has the green object.", "query": "Which one has the red object?", "entity": "pig", "value": "red"},
        {"id": "r7_has_entity_mid", "family": "has_entity", "facts": "The hen has the yellow object. The fox has the pink object. The bird has the blue object.", "query": "Which one has the pink object?", "entity": "fox", "value": "pink"},
        {"id": "r7_bes_who", "family": "beside", "facts": "The frog sits. The duck is beside the hen.", "query": "Who is beside the hen?", "entity": "duck", "value": "hen", "landmark": "hen"},
        {"id": "r7_bes_rev", "family": "beside", "facts": "The frog sits. The duck is beside the hen.", "query": "Who is beside the duck?", "entity": "hen", "value": "duck", "landmark": "duck"},
        {"id": "r7_bes_where", "family": "beside", "facts": "The frog sits. The duck is beside the hen.", "query": "Where is the duck?", "entity": "duck", "value": "hen", "landmark": "duck"},
        {"id": "r7_bes_next", "family": "beside", "facts": "Remember: the cat is beside the bear.", "query": "Which animal is next to the bear?", "entity": "cat", "value": "bear", "landmark": "bear"},
        {"id": "r7_bes_near", "family": "beside", "facts": "The fox is beside the duck. The pig is red.", "query": "Who is near the duck?", "entity": "fox", "value": "duck", "landmark": "duck"},
        {"id": "r7_bes_collision", "family": "beside", "facts": "The pig is beside the cow. The hen is yellow.", "query": "Who is beside the cow?", "entity": "pig", "value": "cow", "landmark": "cow"},
        {"id": "r7_color_nonrecent", "family": "color", "facts": "The duck is pink. The hen is yellow.", "query": "Which color is the duck?", "entity": "duck", "value": "pink"},
        {"id": "r7_size_bird", "family": "size", "facts": "That bird is tiny in size. fox looks white.", "query": "Tell me the size of the bird.", "entity": "bird", "value": "tiny"},
        {"id": "r7_direct_fox", "family": "direct", "facts": "That bird is tiny in size. fox looks white.", "query": "Which color is the fox?", "entity": "fox", "value": "white"},
        {"id": "r7_direct_pig", "family": "direct", "facts": "pig looks red. The hen is yellow.", "query": "Which color is the pig?", "entity": "pig", "value": "red"},
        {"id": "r7_combine_color", "family": "combine", "facts": "The hen is yellow. The hen is short. The duck is pink. The duck is wide.", "query": "Which color is the short one?", "entity": "hen", "value": "yellow"},
        {"id": "r7_combine_size", "family": "combine", "facts": "The hen is yellow. The hen is short. The duck is pink. The duck is wide.", "query": "Which size is the pink one?", "entity": "duck", "value": "wide"},
        {"id": "r7_combine_3e", "family": "combine", "facts": "The cat is blue. The cat is tiny. The dog is red. The dog is huge. The hen is yellow. The hen is short.", "query": "Tell me the color of the huge one.", "entity": "dog", "value": "red"},
        {"id": "r7_story_combine", "family": "story_combine", "facts": "Long ago a tiny hen sat. That hen was yellow. A huge dog ran. That dog was red.", "query": "Tell me the color of the tiny one.", "entity": "hen", "value": "yellow"},
        {"id": "r7_story_adj_pig", "family": "story_combine", "facts": "A tiny pig sat. That pig was white. A huge cow ran. That cow was green.", "query": "Which color is the tiny one?", "entity": "pig", "value": "white"},
        {"id": "r7_about_two", "family": "about", "facts": "The bear is green. The frog is white. The bear is thin.", "query": "What do you know about the bear?", "entity": "bear", "value": "green", "colors": ["green"], "sizes": ["thin"]},
        {"id": "r7_about_distract", "family": "about", "facts": "The duck is pink. The hen is yellow. The duck is wide.", "query": "Talk about the hen.", "entity": "hen", "value": "yellow", "colors": ["yellow"], "sizes": []},
        {"id": "r7_about_then", "family": "about", "facts": "The cat is blue. The dog is red.", "query": "the dog then?", "entity": "dog", "value": "red", "colors": ["red"], "sizes": []},
        {"id": "r7_mix_who", "family": "who", "facts": "The dog is white. The cat has the red object. The bear is beside the dog.", "query": "Who is white?", "entity": "dog", "value": "white"},
        {"id": "r7_mix_has", "family": "has_value", "facts": "The dog is white. The cat has the red object. The bear is beside the dog.", "query": "What does the cat have?", "entity": "cat", "value": "red"},
        {"id": "r7_mix_bes", "family": "beside", "facts": "The dog is white. The cat has the red object. The bear is beside the dog.", "query": "Who is beside the dog?", "entity": "bear", "value": "dog", "landmark": "dog"},
        {"id": "r7_mix_near", "family": "beside", "facts": "The dog is white. The cat has the red object. The bear is beside the dog.", "query": "Who is near the dog?", "entity": "bear", "value": "dog", "landmark": "dog"},
        {"id": "r7_distractor_late", "family": "who", "facts": "The cow is blue. The fox is pink. The bird is green. The pig is red.", "query": "Who is pink?", "entity": "fox", "value": "pink"},
        {"id": "r7_looks_bridge", "family": "who_sent", "facts": "fox looks white. pig looks red.", "query": "Who looks white?", "entity": "fox", "value": "white"},
    ]


def build_r7_reuse_pack() -> list[dict]:
    return [
        {
            "id": "r7_ru_who",
            "facts": "The pig is red. The cow is blue.",
            "turns": [
                {"query": "Who is red?", "gold": "pig", "entity": "pig", "value": "red", "family": "who", "come_back": False},
                {"query": "Who is blue?", "gold": "cow", "entity": "cow", "value": "blue", "family": "who", "come_back": False},
                {"query": "Who is red?", "gold": "pig", "entity": "pig", "value": "red", "family": "who", "come_back": True},
            ],
        },
        {
            "id": "r7_ru_has",
            "facts": "The hen has the yellow object. The duck has the pink object.",
            "turns": [
                {"query": "What does the hen have?", "gold": "yellow", "entity": "hen", "value": "yellow", "family": "has_value", "come_back": False},
                {"query": "Which one has the pink object?", "gold": "duck", "entity": "duck", "value": "pink", "family": "has_entity", "come_back": False},
                {"query": "What does the hen have?", "gold": "yellow", "entity": "hen", "value": "yellow", "family": "has_value", "come_back": True},
            ],
        },
        {
            "id": "r7_ru_bes",
            "facts": "The fox is beside the duck. The pig is red.",
            "turns": [
                {"query": "Who is beside the duck?", "gold": "fox", "entity": "fox", "value": "duck", "landmark": "duck", "family": "beside", "come_back": False},
                {"query": "Who is red?", "gold": "pig", "entity": "pig", "value": "red", "family": "who", "come_back": False},
                {"query": "Who is near the duck?", "gold": "fox", "entity": "fox", "value": "duck", "landmark": "duck", "family": "beside", "come_back": True},
            ],
        },
        {
            "id": "r7_ru_combine",
            "facts": "The hen is yellow. The hen is short. The duck is pink. The duck is wide.",
            "turns": [
                {"query": "Which color is the short one?", "gold": "yellow", "entity": "hen", "value": "yellow", "family": "combine", "come_back": False},
                {"query": "Which size is the pink one?", "gold": "wide", "entity": "duck", "value": "wide", "family": "combine", "come_back": False},
                {"query": "What do you know about the hen?", "gold": "yellow", "entity": "hen", "value": "yellow", "colors": ["yellow"], "sizes": ["short"], "family": "about", "come_back": True},
            ],
        },
    ]
