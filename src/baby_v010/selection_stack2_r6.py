from __future__ import annotations

"""Stack2 R6 endgame: fact/story combination + conversational reuse."""

import random

from .data import BOS
from .data_language_bridge import (
    ENTITIES,
    SIZES,
    VALUES,
    encode_ids,
    make_mixed_item,
    make_story_mixed_item,
    score_bind_sentence,
    score_has_sentence,
)
from .selection_language_bridge import greedy_decode_until_stop


def score_combine_items(model, tokenizer, device, items: list[dict]) -> dict:
    rows = []
    for item in items:
        prompt = item.get("prompt_text") or f"{item['facts']} {item['query']}"
        gold = str(item.get("value_text") or item.get("value") or "")
        ids = [BOS, *encode_ids(tokenizer, prompt)]
        emitted, stopped = greedy_decode_until_stop(model, ids, device, tokenizer, max_new=16)
        decoded = tokenizer.decode(emitted, skip_special_tokens=True).strip()
        text = decoded.lower()
        gold_l = gold.lower().lstrip()
        rows.append(
            {
                "id": item.get("id") or item.get("variant"),
                "gold": gold_l,
                "decoded": decoded,
                "stopped": stopped,
                "gold_in": gold_l in text,
                "first": text.lstrip().startswith(gold_l),
                "entity_lead": any(text.lstrip().startswith(e) for e in ENTITIES),
            }
        )
    n = len(rows) or 1
    return {
        "n": len(rows),
        "gold_in": sum(bool(r["gold_in"]) for r in rows) / n,
        "first": sum(bool(r["first"]) for r in rows) / n,
        "entity_lead": sum(bool(r["entity_lead"]) for r in rows) / n,
        "rows": rows,
    }


def official_combine_pack(tokenizer, *, story: bool, n: int, seed: int, n_entities: int = 2) -> list[dict]:
    rng = random.Random(seed)
    if story:
        return [make_story_mixed_item(rng, tokenizer, surface="heldout", combine=True) for _ in range(n)]
    return [make_mixed_item(rng, tokenizer, n_entities=n_entities, surface="heldout", combine=True) for _ in range(n)]


def build_about_report_heldout() -> list[dict]:
    """Multi-attribute report. Eval-only. Not the official one-word about_2fact panel."""
    return [
        {
            "id": "ab_same_ent_size_last",
            "facts": "The hen is yellow. The hen is short.",
            "query": "What do you know about the hen?",
            "entity": "hen",
            "colors": ["yellow"],
            "sizes": ["short"],
        },
        {
            "id": "ab_same_ent_color_last",
            "facts": "The duck is wide. The duck is pink.",
            "query": "Talk about the duck.",
            "entity": "duck",
            "colors": ["pink"],
            "sizes": ["wide"],
        },
        {
            "id": "ab_distract_last_other",
            "facts": "The bear is green. The frog is white. The bear is thin.",
            "query": "Tell me about the bear.",
            "entity": "bear",
            "colors": ["green"],
            "sizes": ["thin"],
        },
        {
            "id": "ab_distract_first_other",
            "facts": "The frog is white. The bear is green. The bear is thin.",
            "query": "Describe the frog.",
            "entity": "frog",
            "colors": ["white"],
            "sizes": [],
        },
        {
            "id": "ab_nonrecent_gold",
            "facts": "The cat is blue. The dog is red. The dog is huge. The cat is tiny.",
            "query": "What do you know about the dog?",
            "entity": "dog",
            "colors": ["red"],
            "sizes": ["huge"],
        },
        {
            "id": "ab_request_first_entity",
            "facts": "The hen is yellow. The duck is pink. The hen is short. The duck is wide.",
            "query": "Talk about the hen.",
            "entity": "hen",
            "colors": ["yellow"],
            "sizes": ["short"],
        },
        {
            "id": "ab_hold_looks",
            "facts": "Remember: the frog is white. That bear is green. frog looks thin in size.",
            "query": "What do you know about the frog?",
            "entity": "frog",
            "colors": ["white"],
            "sizes": ["thin"],
        },
        {
            "id": "ab_hold_that",
            "facts": "That duck is pink. Remember: the hen is yellow. That duck is small in size.",
            "query": "Tell me about the duck.",
            "entity": "duck",
            "colors": ["pink"],
            "sizes": ["small"],
        },
    ]


def score_about_report(model, tokenizer, device, items: list[dict]) -> dict:
    rows = []
    for item in items:
        prompt = f"{item['facts']} {item['query']}"
        ids = [BOS, *encode_ids(tokenizer, prompt)]
        emitted, stopped = greedy_decode_until_stop(model, ids, device, tokenizer, max_new=24)
        decoded = tokenizer.decode(emitted, skip_special_tokens=True).strip()
        text = decoded.lower()
        needed = list(item.get("colors") or []) + list(item.get("sizes") or [])
        have = [word for word in needed if word in text]
        other = [e for e in ENTITIES if e != item["entity"] and f"{e} " in f" {text} "]
        rows.append(
            {
                "id": item["id"],
                "decoded": decoded,
                "stopped": stopped,
                "entity_ok": item["entity"] in text,
                "attrs_ok": len(have) == len(needed) and bool(needed),
                "no_other_entity": not other,
                "have": have,
                "needed": needed,
            }
        )
    n = len(rows) or 1
    return {
        "n": len(rows),
        "attrs_ok": sum(bool(r["attrs_ok"]) for r in rows) / n,
        "entity_ok": sum(bool(r["entity_ok"]) for r in rows) / n,
        "clean": sum(bool(r["attrs_ok"] and r["no_other_entity"]) for r in rows) / n,
        "rows": rows,
    }


def build_story_progression_heldout() -> dict[str, list[dict]]:
    """Increasing story-combine demand. Eval-only. Official story_combine kept separate."""
    return {
        "two_facts": [
            {"id": "st_two_color", "facts": "A tiny hen sat. That hen was yellow.", "query": "Which color is the tiny one?", "value": "yellow"},
            {"id": "st_two_size", "facts": "A wide duck swam. That duck was pink.", "query": "Which size is the pink one?", "value": "wide"},
        ],
        "one_distractor": [
            {"id": "st_dist_color", "facts": "A tiny hen sat. That hen was yellow. A huge dog ran.", "query": "Which color is the tiny one?", "value": "yellow"},
            {"id": "st_dist_size", "facts": "A wide duck swam. That duck was pink. A thin cat sat.", "query": "Which size is the pink one?", "value": "wide"},
        ],
        "separated": [
            {"id": "st_sep_color", "facts": "A tiny hen sat. A huge dog ran. That hen was yellow. That dog was red.", "query": "Which color is the tiny one?", "value": "yellow"},
            {"id": "st_sep_size", "facts": "A wide duck swam. A thin cat sat. That duck was pink. That cat was blue.", "query": "Which size is the pink one?", "value": "wide"},
        ],
        "multi_entity": [
            {"id": "st_multi_first", "facts": "Long ago a tiny hen sat. That hen was yellow. A huge dog ran. That dog was red.", "query": "Tell me the color of the tiny one.", "value": "yellow"},
            {"id": "st_multi_last", "facts": "Long ago a tiny hen sat. That hen was yellow. A huge dog ran. That dog was red.", "query": "Tell me the color of the huge one.", "value": "red"},
            {"id": "st_multi_nonrecent", "facts": "In the yard a short bear ate. The bear looks green. A small frog hid. The frog looks white.", "query": "Which color is the short one?", "value": "green"},
        ],
    }


def build_reuse_heldout() -> list[dict]:
    """Short conversations that return to an earlier fact. Eval-only."""
    return [
        {
            "id": "ru_color_return",
            "facts": "The hen is yellow. The duck is pink.",
            "turns": [
                {"query": "Which color is the hen?", "gold": "yellow", "entity": "hen", "come_back": False},
                {"query": "Which color is the duck?", "gold": "pink", "entity": "duck", "come_back": False},
                {"query": "Which color is the hen?", "gold": "yellow", "entity": "hen", "come_back": True},
            ],
        },
        {
            "id": "ru_size_return",
            "facts": "That bear is thin in size. That frog is wide in size.",
            "turns": [
                {"query": "Which size is the bear?", "gold": "thin", "entity": "bear", "come_back": False},
                {"query": "Which size is the frog?", "gold": "wide", "entity": "frog", "come_back": False},
                {"query": "Tell me the size of the bear.", "gold": "thin", "entity": "bear", "come_back": True},
            ],
        },
        {
            "id": "ru_howabout_return",
            "facts": "The cat is blue. The dog is red.",
            "turns": [
                {"query": "What do you know about the cat?", "gold": "blue", "entity": "cat", "come_back": False},
                {"query": "How about the dog?", "gold": "red", "entity": "dog", "come_back": False},
                {"query": "Talk about the cat.", "gold": "blue", "entity": "cat", "come_back": True},
            ],
        },
        {
            "id": "ru_has_return",
            "facts": "The hen has the yellow object. The duck has the pink object.",
            "turns": [
                {"query": "What does the hen have?", "gold": "yellow", "entity": "hen", "come_back": False, "family": "has"},
                {"query": "What does the duck have?", "gold": "pink", "entity": "duck", "come_back": False, "family": "has"},
                {"query": "What does the hen have?", "gold": "yellow", "entity": "hen", "come_back": True, "family": "has"},
            ],
        },
        {
            "id": "ru_beside_return",
            "facts": "The cat is beside the dog. The hen is yellow.",
            "turns": [
                {"query": "Who is beside the dog?", "gold": "cat", "entity": "cat", "come_back": False, "family": "beside"},
                {"query": "Which color is the hen?", "gold": "yellow", "entity": "hen", "come_back": False},
                {"query": "Who is beside the dog?", "gold": "cat", "entity": "cat", "come_back": True, "family": "beside"},
            ],
        },
        {
            "id": "ru_mixed_switch",
            "facts": "The dog is white. The cat has the red object. The bear is beside the dog.",
            "turns": [
                {"query": "Who is white?", "gold": "dog", "entity": "dog", "come_back": False, "family": "who"},
                {"query": "What does the cat have?", "gold": "red", "entity": "cat", "come_back": False, "family": "has"},
                {"query": "Who is beside the dog?", "gold": "bear", "entity": "bear", "come_back": False, "family": "beside"},
                {"query": "Who is white?", "gold": "dog", "entity": "dog", "come_back": True, "family": "who"},
            ],
        },
        {
            "id": "ru_nonrecent_after_combine",
            "facts": "The hen is yellow. The hen is short. The duck is pink. The duck is wide.",
            "turns": [
                {"query": "Which color is the short one?", "gold": "yellow", "entity": "hen", "come_back": False},
                {"query": "Which size is the pink one?", "gold": "wide", "entity": "duck", "come_back": False},
                {"query": "What do you know about the hen?", "gold": "yellow", "entity": "hen", "come_back": True},
            ],
        },
        {
            "id": "ru_bridge_entity",
            "facts": "That bird is tiny in size. fox looks white. pig looks red.",
            "turns": [
                {"query": "Tell me the size of the bird.", "gold": "tiny", "entity": "bird", "come_back": False},
                {"query": "Which color is the fox?", "gold": "white", "entity": "fox", "come_back": False},
                {"query": "How about the pig?", "gold": "red", "entity": "pig", "come_back": False},
                {"query": "Tell me the size of the bird.", "gold": "tiny", "entity": "bird", "come_back": True},
            ],
        },
    ]


def build_integration_heldout() -> list[dict]:
    """Fresh open-book mix of existing families. Eval-only. Not trained."""
    return [
        {"id": "int_color_nonrecent", "family": "color", "facts": "The duck is pink. The hen is yellow.", "query": "Which color is the duck?", "entity": "duck", "value": "pink"},
        {"id": "int_size_nonrecent", "family": "size", "facts": "The frog is wide in size. The bear is thin in size.", "query": "What is the size of the frog?", "entity": "frog", "value": "wide"},
        {"id": "int_who_color", "family": "who", "facts": "The hen is yellow. The duck is pink.", "query": "Who is yellow?", "entity": "hen", "value": "yellow"},
        {"id": "int_who_size", "family": "who", "facts": "The bear is thin. The frog is wide.", "query": "Who is wide?", "entity": "frog", "value": "wide"},
        {"id": "int_who_sent", "family": "who_sent", "facts": "Remember: the dog is huge in size. That hen is tiny in size.", "query": "Tell me who looks tiny.", "entity": "hen", "value": "tiny"},
        {"id": "int_has_what", "family": "has", "facts": "The cat has the blue object. The dog has the white object.", "query": "What does the cat have?", "entity": "cat", "value": "blue"},
        {"id": "int_has_who", "family": "has", "facts": "The cat has the blue object. The dog has the white object.", "query": "Which one has the white object?", "entity": "dog", "value": "white"},
        {"id": "int_bes_who", "family": "beside", "facts": "The frog sits. The duck is beside the hen.", "query": "Who is beside the hen?", "entity": "duck", "value": "hen"},
        {"id": "int_bes_where", "family": "beside", "facts": "The frog sits. The duck is beside the hen.", "query": "Where is the duck?", "entity": "duck", "value": "hen"},
        {"id": "int_para_who", "family": "who", "facts": "That bear is green. That frog is white.", "query": "Which animal is green?", "entity": "bear", "value": "green"},
        {"id": "int_para_has", "family": "has", "facts": "That hen has the yellow object. Remember: the duck has the pink object.", "query": "Which object belongs to the duck?", "entity": "duck", "value": "pink"},
        {"id": "int_para_bes", "family": "beside", "facts": "Remember: the cat is beside the bear.", "query": "Which animal is next to the bear?", "entity": "cat", "value": "bear"},
        {"id": "int_combine_color", "family": "combine", "facts": "The hen is yellow. The hen is short. The duck is pink. The duck is wide.", "query": "Which color is the short one?", "entity": "hen", "value": "yellow"},
        {"id": "int_combine_size", "family": "combine", "facts": "The hen is yellow. The hen is short. The duck is pink. The duck is wide.", "query": "Which size is the pink one?", "entity": "duck", "value": "wide"},
        {"id": "int_story_combine", "family": "story_combine", "facts": "Long ago a tiny hen sat. That hen was yellow. A huge dog ran. That dog was red.", "query": "Tell me the color of the tiny one.", "entity": "hen", "value": "yellow"},
        {"id": "int_about_two", "family": "about", "facts": "The bear is green. The frog is white. The bear is thin.", "query": "What do you know about the bear?", "entity": "bear", "value": "green", "colors": ["green"], "sizes": ["thin"]},
        {"id": "int_about_distract", "family": "about", "facts": "The duck is pink. The hen is yellow. The duck is wide.", "query": "Talk about the hen.", "entity": "hen", "value": "yellow", "colors": ["yellow"], "sizes": []},
        {"id": "int_bridge_direct", "family": "size", "facts": "That bird is tiny in size. fox looks white.", "query": "Tell me the size of the bird.", "entity": "bird", "value": "tiny"},
        {"id": "int_mix_who", "family": "who", "facts": "The dog is white. The cat has the red object. The bear is beside the dog.", "query": "Who is white?", "entity": "dog", "value": "white"},
        {"id": "int_mix_has", "family": "has", "facts": "The dog is white. The cat has the red object. The bear is beside the dog.", "query": "What does the cat have?", "entity": "cat", "value": "red"},
        {"id": "int_mix_bes", "family": "beside", "facts": "The dog is white. The cat has the red object. The bear is beside the dog.", "query": "Who is beside the dog?", "entity": "bear", "value": "dog"},
        {"id": "int_then", "family": "about", "facts": "The cat is blue. The dog is red.", "query": "the dog then?", "entity": "dog", "value": "red", "colors": ["red"], "sizes": []},
    ]


def score_integration_items(model, tokenizer, device, items: list[dict]) -> dict:
    rows = []
    for item in items:
        prompt = f"{item['facts']} {item['query']}"
        ids = [BOS, *encode_ids(tokenizer, prompt)]
        emitted, stopped = greedy_decode_until_stop(model, ids, device, tokenizer, max_new=20)
        decoded = tokenizer.decode(emitted, skip_special_tokens=True).strip()
        family = item["family"]
        if family == "has":
            scored = score_has_sentence(full_text=decoded, entity=item["entity"], value=item["value"], stopped=stopped)
            ok = bool(scored["sentence_ok"])
        elif family == "beside":
            scored = score_bind_sentence(full_text=decoded, entity=item["entity"], value=item["value"], stopped=stopped, predicates=("beside", "next"))
            ok = bool(scored["sentence_ok"])
        elif family in {"who_sent"}:
            scored = score_bind_sentence(full_text=decoded, entity=item["entity"], value=item["value"], stopped=stopped, predicates=("is", "looks"))
            ok = bool(scored["sentence_ok"])
        elif family == "about":
            needed = list(item.get("colors") or []) + list(item.get("sizes") or [])
            have = [word for word in needed if word in decoded.lower()]
            ok = item["entity"] in decoded.lower() and len(have) == len(needed) and bool(needed)
            scored = {"needed": needed, "have": have}
        elif family == "who":
            text = decoded.lower()
            ok = item["entity"].lower() in text
            scored = {"entity_ok": ok, "value_in": item["value"].lower() in text}
        else:
            ok = item["value"] in decoded.lower()
            scored = {}
        rows.append({"id": item["id"], "family": family, "decoded": decoded, "ok": ok, "stopped": stopped, **scored})
    n = len(rows) or 1
    by: dict[str, float] = {}
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam]
        by[fam] = sum(bool(r["ok"]) for r in sub) / len(sub)
    return {
        "n": len(rows),
        "overall": sum(bool(r["ok"]) for r in rows) / n,
        "by_family": by,
        "rows": rows,
    }


def score_reuse_pack(model, tokenizer, device, scripts: list[dict]) -> dict:
    rows = []
    for script in scripts:
        context = script["facts"]
        for turn in script["turns"]:
            prompt = f"{context}\nHuman: {turn['query']}\nBaby:"
            ids = [BOS, *encode_ids(tokenizer, prompt)]
            emitted, stopped = greedy_decode_until_stop(model, ids, device, tokenizer, max_new=20)
            decoded = tokenizer.decode(emitted, skip_special_tokens=True).strip()
            gold = turn["gold"]
            hit = gold in decoded.lower()
            rows.append(
                {
                    "id": script["id"],
                    "query": turn["query"],
                    "gold": gold,
                    "decoded": decoded,
                    "hit": hit,
                    "stopped": stopped,
                    "come_back": bool(turn.get("come_back")),
                    "family": turn.get("family") or "qa",
                }
            )
            context = f"{context} {turn['query']} {decoded if decoded else gold}"
    n = len(rows) or 1
    come = [r for r in rows if r["come_back"]]
    return {
        "n": len(rows),
        "hit": sum(bool(r["hit"]) for r in rows) / n,
        "reuse": (sum(bool(r["hit"]) for r in come) / len(come)) if come else None,
        "stop": sum(bool(r["stopped"]) for r in rows) / n,
        "rows": rows,
    }
