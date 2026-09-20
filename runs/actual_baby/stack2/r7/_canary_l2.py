"""R7 L2: broader developmental canary. Independent of Form A and of the frozen pack."""

from pathlib import Path
import json

from src.baby_v010.data_language_bridge import ENTITIES, VALUES, SIZES, load_tokenizer, make_who_bind_item
from src.baby_v010.selection_language_bridge import load_experimental_baby
from src.baby_v010.selection_p11_u16000_runtime import resolve_device
from src.baby_v010.selection_stack2_r3 import (
    S5B3_PARENT,
    S5B3_PARENT_SHA,
    PropMatchRuntime,
    load_prop_match_head,
    score_who_with_prop_match,
    verify_parent,
)
from src.baby_v010.selection_stack2_r6 import (
    build_about_report_heldout,
    official_combine_pack,
    score_about_report,
    score_combine_items,
)
from src.baby_v010.selection_stack2_r7 import (
    build_r7_mechanism_canary,
    build_r7_reuse_pack,
    score_r7_items,
    score_r7_reuse,
)

OUT = Path("runs/actual_baby/stack2/r7")
HEAD = Path("runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt")


def all_entity_who_items() -> list[dict]:
    items = []
    colors = list(VALUES)
    sizes = list(SIZES)
    for i, ent in enumerate(ENTITIES):
        other = ENTITIES[(i + 3) % len(ENTITIES)]
        if other == ent:
            other = ENTITIES[(i + 1) % len(ENTITIES)]
        c1 = colors[i % len(colors)]
        c2 = colors[(i + 2) % len(colors)]
        if c2 == c1:
            c2 = colors[(i + 1) % len(colors)]
        s1 = sizes[i % len(sizes)]
        items.append(
            {
                "id": f"all_left_{ent}",
                "family": "who",
                "facts": f"The {other} is {c2}. The {ent} is {c1}.",
                "query": f"Who is {c1}?",
                "entity": ent,
                "value": c1,
            }
        )
        items.append(
            {
                "id": f"all_adj_{ent}",
                "family": "who",
                "facts": f"A {s1} {ent} sat. That {ent} was {c1}. A {c2} {other} hid.",
                "query": f"Who is {c1}?",
                "entity": ent,
                "value": c1,
            }
        )
        items.append(
            {
                "id": f"all_which_{ent}",
                "family": "who",
                "facts": f"The {ent} is {c1}. The {other} is {c2}. The hen is yellow." if ent not in {"hen"} else f"The {ent} is {c1}. The {other} is {c2}. The dog is white.",
                "query": f"Which one is {c1}?",
                "entity": ent,
                "value": c1,
            }
        )
    return items


def extra_family_items() -> list[dict]:
    return [
        {
            "id": "x_has_val_fox",
            "family": "has_value",
            "facts": "The fox has the pink object. The bird has the blue object. The cow has the green object.",
            "query": "What does the bird have?",
            "entity": "bird",
            "value": "blue",
        },
        {
            "id": "x_has_ent_mid",
            "family": "has_entity",
            "facts": "The fox has the pink object. The bird has the blue object. The cow has the green object.",
            "query": "Which one has the blue object?",
            "entity": "bird",
            "value": "blue",
        },
        {
            "id": "x_bes_rev_pig",
            "family": "beside",
            "facts": "The pig is beside the cow. The hen is yellow.",
            "query": "Who is beside the pig?",
            "entity": "cow",
            "value": "pig",
            "landmark": "pig",
        },
        {
            "id": "x_bes_where_cow",
            "family": "beside",
            "facts": "The pig is beside the cow. The hen is yellow.",
            "query": "Where is the cow?",
            "entity": "cow",
            "value": "pig",
            "landmark": "cow",
        },
        {
            "id": "x_bes_near_fox",
            "family": "beside",
            "facts": "The bird is beside the fox. The duck is pink.",
            "query": "Who is near the fox?",
            "entity": "bird",
            "value": "fox",
            "landmark": "fox",
        },
        {
            "id": "x_who_looks_cow",
            "family": "who_sent",
            "facts": "Remember: the cow is green. That bird is tiny in size.",
            "query": "Tell me who looks green.",
            "entity": "cow",
            "value": "green",
        },
        {
            "id": "x_mix_who_not_bes",
            "family": "who",
            "facts": "The dog is white. The cat has the red object. The bear is beside the dog.",
            "query": "Who is white?",
            "entity": "dog",
            "value": "white",
        },
        {
            "id": "x_story_pig",
            "family": "story_combine",
            "facts": "A tiny pig sat. That pig was white. A huge cow ran. That cow was green.",
            "query": "Which color is the tiny one?",
            "entity": "pig",
            "value": "white",
        },
    ]


device = resolve_device("cuda")
sha = verify_parent(S5B3_PARENT, S5B3_PARENT_SHA, label="r7l2")
tok = load_tokenizer()
model, config, ckpt = load_experimental_baby(S5B3_PARENT, device)
assert not ckpt.get("protected_material_opened")
head = load_prop_match_head(HEAD, config.d_model, device)
rt = PropMatchRuntime(model, head, tok).install()
rt.enabled = True

mech = score_r7_items(model, tok, device, build_r7_mechanism_canary())
who_all = score_r7_items(model, tok, device, all_entity_who_items())
extra = score_r7_items(model, tok, device, extra_family_items())
reuse = score_r7_reuse(model, tok, device, build_r7_reuse_pack())
about = score_about_report(model, tok, device, build_about_report_heldout())
fact = score_combine_items(model, tok, device, official_combine_pack(tok, story=False, n=16, seed=611001))
story = score_combine_items(model, tok, device, official_combine_pack(tok, story=True, n=16, seed=611001))
who2 = score_who_with_prop_match(
    model, head, tok, device, [make_who_bind_item(__import__("random").Random(324001), tok, n_entities=2, surface="heldout") for _ in range(32)]
)
rt.uninstall()

out = {
    "parent": sha,
    "relassist": False,
    "whoprop": False,
    "mechanism": {k: mech[k] for k in ("n", "overall", "by_family") if k in mech},
    "mechanism_miss": mech["miss"],
    "who_all_entities": {k: who_all[k] for k in ("n", "overall", "by_family")},
    "who_all_miss": who_all["miss"],
    "extra": {k: extra[k] for k in ("n", "overall", "by_family")},
    "extra_miss": extra["miss"],
    "reuse": {k: reuse[k] for k in ("n", "ok", "reuse", "stop")},
    "reuse_miss": reuse["miss"],
    "about": {k: about[k] for k in about if k != "rows"},
    "fact_combine": {k: fact[k] for k in fact if k != "rows"},
    "story_combine": {k: story[k] for k in story if k != "rows"},
    "who_2e": {k: who2[k] for k in who2 if k != "rows"},
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "L2_CANARY.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2, default=str)[:8000])
print("WHO_ALL MISS", len(who_all["miss"]))
for row in who_all["miss"][:12]:
    print(row["id"], row["decoded"], "gold", row["gold_entity"])
print("EXTRA MISS")
for row in extra["miss"]:
    print(row)
print("REUSE MISS")
for row in reuse["miss"]:
    print(row)
