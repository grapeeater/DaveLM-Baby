from pathlib import Path
import json
import random

from src.baby_v010.data import BOS
from src.baby_v010.data_language_bridge import (
    encode_ids,
    load_tokenizer,
    make_mixed_item,
    make_story_mixed_item,
)
from src.baby_v010.selection_language_bridge import greedy_decode_until_stop, load_experimental_baby
from src.baby_v010.selection_p11_u16000_runtime import resolve_device
from src.baby_v010.selection_stack2_r3 import (
    S5B3_PARENT,
    S5B3_PARENT_SHA,
    PropMatchRuntime,
    load_prop_match_head,
    verify_parent,
)

device = resolve_device("cuda")
sha = verify_parent(S5B3_PARENT, S5B3_PARENT_SHA, label="r6comb")
tok = load_tokenizer()
model, config, _ = load_experimental_baby(S5B3_PARENT, device)
head = load_prop_match_head(
    Path("runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt"),
    config.d_model,
    device,
)


def decode_items(items):
    rows = []
    for item in items:
        prompt = item["prompt_text"]
        ids = [BOS, *encode_ids(tok, prompt)]
        emitted, stopped = greedy_decode_until_stop(model, ids, device, tok, max_new=16)
        decoded = tok.decode(emitted, skip_special_tokens=True).strip()
        gold = str(item["value_text"])
        rows.append(
            {
                "prompt": prompt,
                "query": prompt.split("?")[-2].split(".")[-1].strip() + "?" if "?" in prompt else prompt[-40:],
                "gold": gold,
                "entity": item.get("entity"),
                "attr": item.get("attr"),
                "decoded": decoded,
                "gold_in": gold in decoded.lower(),
                "first_is_gold": decoded.lower().lstrip().startswith(gold),
            }
        )
    return rows


rt = PropMatchRuntime(model, head, tok).install()
rt.enabled = True
rng = random.Random(611001)
facts = [make_mixed_item(rng, tok, n_entities=2, surface="heldout", combine=True) for _ in range(16)]
stories = [make_story_mixed_item(rng, tok, surface="heldout", combine=True) for _ in range(12)]
about = [
    {
        "prompt_text": "The cat is blue. The cat is tiny. What do you know about the cat?",
        "value_text": "blue",
        "entity": "cat",
        "attr": "about",
    },
    {
        "prompt_text": "The dog is red. The cat is blue. The dog is huge. Tell me about the dog.",
        "value_text": "red",
        "entity": "dog",
        "attr": "about",
    },
]
fact_rows = decode_items(facts)
story_rows = decode_items(stories)
about_rows = decode_items(about)
rt.uninstall()
out = {
    "parent": sha,
    "fact_combine": {
        "n": len(fact_rows),
        "gold_in": sum(r["gold_in"] for r in fact_rows) / len(fact_rows),
        "first": sum(r["first_is_gold"] for r in fact_rows) / len(fact_rows),
        "rows": fact_rows,
    },
    "story_combine": {
        "n": len(story_rows),
        "gold_in": sum(r["gold_in"] for r in story_rows) / len(story_rows),
        "first": sum(r["first_is_gold"] for r in story_rows) / len(story_rows),
        "rows": story_rows,
    },
    "about": about_rows,
}
Path("runs/actual_baby/stack2/r6/DIAG_COMBINE.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: {kk: out[k][kk] for kk in out[k] if kk != "rows"} if isinstance(out[k], dict) else out[k] for k in out}, indent=2))
print("FACT")
for r in fact_rows:
    print(r["attr"], r["gold"], "=>", r["decoded"], "|", r["prompt"][-60:])
print("STORY")
for r in story_rows:
    print(r["attr"], r["gold"], "=>", r["decoded"])
print("ABOUT")
for r in about_rows:
    print(r["decoded"], "|", r["prompt"])
