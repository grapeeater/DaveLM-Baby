"""SF9 curriculum generator. Deterministic, outcome-blind, run exactly once before freeze.

Produces NEW, clean, balanced development/training material that widens Baby's factual-selection
capability across surface forms and competing-information orders, WITHOUT touching the frozen
transfer panels (HELDOUT16/ALTERNATE48/COPY8/COMPETING64) and WITHOUT introducing new name
vocabulary (the four single-token names Alex/Owen/Mia/Nora are preserved so the SF8 margin
mechanism and D3/binding machinery remain valid).

Curriculum = original TRAIN16 (preservation) + NEW surface items + NEW competing items.
New vocabulary is disjoint from all frozen panels: objects/verbs below never appear in
HELDOUT/ALTERNATE/COPY/COMPETING (which use small drum/wooden boat/soft scarf/round plate x
found/carried only).
"""
import json, hashlib
from pathlib import Path
from tokenizers import Tokenizer

H = Path(__file__).resolve().parent
SF8 = Path(r"C:\DaveLM-CADAVER\sf8_margin_dose_comparison_v1")

NAMES = {
    "A": ["Alex", "Owen"],
    "B": ["Mia", "Nora"],
}
# canonical candidate tokenizations copied verbatim from the frozen TRAIN16 (same names)
CAND = {
    " Alex.": [314, 290, 92, 18],
    " Owen.": [536, 91, 274, 18],
    " Mia.": [925, 77, 69, 18],
    " Nora.": [512, 276, 69, 18],
}

# NEW curriculum vocabulary (disjoint from frozen panels)
GROUP_OBJ = {
    "A": ["the tin cup", "the wool hat"],
    "B": ["the green ball", "the silver bell"],
}
GROUP_VERB = {
    "A": ["painted", "dropped"],
    "B": ["bought", "held"],
}
OBJKEY = {"the tin cup": "tin_cup", "the wool hat": "wool_hat", "the green ball": "green_ball", "the silver bell": "silver_bell"}

# NEW held-out dev vocabulary (never in training curriculum)
DEV_OBJ = {"A": "the clay mug", "B": "the brass key"}
DEV_VERB = {"A": "stacked", "B": "rolled"}

SURFACES = ["cloze", "active_qa", "passive_cloze", "passive_qa"]


def surface_prompt(surface, name, verb, obj, other):
    if surface == "cloze":
        return f"{name} {verb} {obj}.\nThe person who {verb} {obj} was"
    if surface == "active_qa":
        return f"{name} {verb} {obj}.\nWho {verb} {obj}?"
    if surface == "passive_cloze":
        return f"{obj} was {verb} by {name}.\nThe person who {verb} {obj} was"
    if surface == "passive_qa":
        return f"{obj} was {verb} by {name}.\nWho {verb} {obj}?"


def competing_prompt(order, query, name, verb, obj, other):
    if order == "order0":  # fact sentence first, then card
        base = f"{name} {verb} {obj}. A card says {other}."
    else:  # card first, then fact sentence
        base = f"A card says {other}. {name} {verb} {obj}."
    if query == "fact":
        return f"{base}\nThe person who {verb} {obj} was"
    return f"{base}\nCopy the name on the card:"


def build_item(pid, group, name, verb, obj, other, prompt, correct_name):
    cand_names = NAMES[group]
    cand = [f" {cand_names[0]}.", f" {cand_names[1]}."]
    correct_index = cand_names.index(correct_name)
    pids = tok.encode(prompt).ids
    return {
        "id": pid,
        "family_id": pid.rsplit(":", 1)[0],
        "pair_id": pid.rsplit(":", 1)[0],
        "subgroup": "surface",
        "arm": "factual",
        "prompt": prompt,
        "prompt_token_ids": pids,
        "candidates": cand,
        "candidate_token_ids": [list(CAND[cand[0]]), list(CAND[cand[1]])],
        "correct_index": correct_index,
        "actor": name,
        "object": obj,
        "predicate": verb,
    }


def main():
    global tok
    tok = Tokenizer.from_file(json.loads((SF8 / "PROTOCOL.json").read_text(encoding="utf-8-sig"))["tokenizer"])
    train16 = json.loads((SF8 / "TRAIN.json").read_text(encoding="utf-8-sig"))
    for it in train16:
        assert it["candidate_token_ids"][0] == CAND[it["candidates"][0]], it["candidates"][0]
        assert it["candidate_token_ids"][1] == CAND[it["candidates"][1]], it["candidates"][1]

    items = []
    # 1) Original TRAIN16 (preservation), copied verbatim.
    for it in train16:
        items.append(dict(it))

    # 2) NEW surface items: 16 base facts x 4 surfaces.
    for group, names in NAMES.items():
        for obj in GROUP_OBJ[group]:
            for verb in GROUP_VERB[group]:
                for ni, name in enumerate(names):
                    other = names[1 - ni]
                    for surface in SURFACES:
                        prompt = surface_prompt(surface, name, verb, obj, other)
                        pid = f"SF9:surface:{group}:{verb}:{OBJKEY[obj]}:{surface}:{name}"
                        it = build_item(pid, group, name, verb, obj, other, prompt, name)
                        it["subgroup"] = surface
                        it["family_id"] = f"SF9:surface:{group}:{verb}:{surface}"
                        it["pair_id"] = f"SF9:surface:{group}:{verb}:{OBJKEY[obj]}:{surface}"
                        items.append(it)

    # 3) NEW competing items: 16 base facts x 2 orders x 2 queries.
    for group, names in NAMES.items():
        for obj in GROUP_OBJ[group]:
            for verb in GROUP_VERB[group]:
                for ni, name in enumerate(names):
                    other = names[1 - ni]
                    for order in ["order0", "order1"]:
                        for query in ["fact", "copy"]:
                            prompt = competing_prompt(order, query, name, verb, obj, other)
                            correct_name = name if query == "fact" else other
                            pid = f"SF9:competing:{group}:{verb}:{OBJKEY[obj]}:{order}:{query}:{name}"
                            it = build_item(pid, group, name, verb, obj, other, prompt, correct_name)
                            it["subgroup"] = f"{order}:{query}"
                            it["family_id"] = f"SF9:competing:{group}:{verb}:{order}:{query}"
                            it["pair_id"] = f"SF9:competing:{group}:{verb}:{OBJKEY[obj]}:{order}:{query}"
                            items.append(it)

    # Balance assertions (prospective, structural).
    from collections import Counter
    ctr = Counter(it["candidates"][it["correct_index"]] for it in items if it["id"].startswith("SF9:"))
    assert len(set(ctr.values())) == 1, ("name-answer imbalance", ctr)
    surf = Counter(it["subgroup"] for it in items if it["id"].startswith("SF9:surface:"))
    assert set(surf.values()) == {16}, surf
    ordq = Counter(it["subgroup"] for it in items if it["id"].startswith("SF9:competing:"))
    assert set(ordq.values()) == {16}, ordq

    # Disjointness from frozen panels (cheap structural check on vocabulary).
    frozen = []
    for label in ["HELDOUT", "ALTERNATE", "COPY", "COMPETING"]:
        frozen += json.loads((Path(r"C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011") / f"{label}.json").read_text(encoding="utf-8-sig"))
    frozen_prompts = " ".join(it["prompt"] for it in frozen)
    new_words = (GROUP_OBJ["A"] + GROUP_OBJ["B"] + GROUP_VERB["A"] + GROUP_VERB["B"]
                 + [DEV_OBJ["A"], DEV_OBJ["B"], DEV_VERB["A"], DEV_VERB["B"]])
    for w in new_words:
        assert w not in frozen_prompts, f"vocab overlap with frozen panels: {w}"
    for it in items:
        if it["id"].startswith("SF9:"):
            for w in ["small drum", "wooden boat", "soft scarf", "round plate"]:
                assert w not in it["prompt"], it["id"]

    # 4) NEW held-out dev panels (disjoint facts: dev vocab only, never in training).
    dev_surface, dev_order = [], []
    for group, names in NAMES.items():
        dobj = DEV_OBJ[group]
        dverb = DEV_VERB[group]
        for ni, name in enumerate(names):
            other = names[1 - ni]
            for surface in SURFACES:
                prompt = surface_prompt(surface, name, dverb, dobj, other)
                pid = f"SF9DEV:surface:{group}:{surface}:{name}"
                it = build_item(pid, group, name, dverb, dobj, other, prompt, name)
                it["subgroup"] = surface
                it["family_id"] = f"SF9DEV:surface:{group}:{surface}"
                it["pair_id"] = f"SF9DEV:surface:{group}:{surface}"
                dev_surface.append(it)
            for order in ["order0", "order1"]:
                for query in ["fact", "copy"]:
                    prompt = competing_prompt(order, query, name, dverb, dobj, other)
                    correct_name = name if query == "fact" else other
                    pid = f"SF9DEV:order:{group}:{order}:{query}:{name}"
                    it = build_item(pid, group, name, dverb, dobj, other, prompt, correct_name)
                    it["subgroup"] = f"{order}:{query}"
                    it["family_id"] = f"SF9DEV:order:{group}:{order}:{query}"
                    it["pair_id"] = f"SF9DEV:order:{group}:{order}:{query}"
                    dev_order.append(it)

    # Schedule: 200 updates = 180 english + 20 binding. English = all curriculum items once
    # (fixed order). Binding entries reused verbatim from the frozen SF8/SF1 schedule.
    sf8_sched = json.loads((SF8 / "SCHEDULE.json").read_text(encoding="utf-8-sig"))
    sf8_binding = {u["update"]: u for u in sf8_sched if u["kind"] == "binding"}
    pad = max(len(it["prompt_token_ids"]) for it in items) + 5
    schedule = []
    for u in range(1, 201):
        if u % 10 == 0:
            b = sf8_binding[u]
            schedule.append({"update": u, "kind": "binding", "pad": pad,
                             "quartets": b["quartets"], "documents": b["documents"]})
        else:
            schedule.append({"update": u, "kind": "english", "pad": pad, "ids": [it["id"] for it in items]})

    def write_json(p, v):
        p.write_text(json.dumps(v, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    write_json(H / "TRAIN.json", items)
    write_json(H / "SCHEDULE.json", schedule)
    write_json(H / "DEV_SURFACE.json", dev_surface)
    write_json(H / "DEV_ORDER.json", dev_order)

    manifest = {}
    for p in [H / "TRAIN.json", H / "SCHEDULE.json", H / "DEV_SURFACE.json", H / "DEV_ORDER.json", H / "TRAIN16_RETENTION.json"]:
        manifest[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(H / "CURRICULUM_MANIFEST.json", manifest)

    print(json.dumps({
        "curriculum_items": len(items),
        "train16": len(train16),
        "surface_items": sum(1 for it in items if it["id"].startswith("SF9:surface:")),
        "competing_items": sum(1 for it in items if it["id"].startswith("SF9:competing:")),
        "dev_surface": len(dev_surface), "dev_order": len(dev_order),
        "pad": pad, "name_answer_balance": dict(ctr),
        "schedule_updates": len(schedule),
    }, indent=2))


if __name__ == "__main__":
    main()
