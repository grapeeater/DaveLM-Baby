from pathlib import Path
import json
import random

from src.baby_v010.data_language_bridge import load_tokenizer, make_who_bind_item
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
from src.baby_v010.selection_stack2_r5 import (
    build_beside_order_heldout,
    build_mixed_relation_heldout,
    score_beside_items,
    score_mixed_multiturn,
    score_mixed_pack,
)
from src.baby_v010.selection_stack2_r6 import (
    build_about_report_heldout,
    build_story_progression_heldout,
    official_combine_pack,
    score_about_report,
    score_combine_items,
)
from src.baby_v010.selection_stack2_s5 import CHEAP_OPS, run_beside_decode, run_has_decode, run_who_sentence_decode, summarize_bind

device = resolve_device("cuda")
sha = verify_parent(S5B3_PARENT, S5B3_PARENT_SHA, label="r6v12")
tok = load_tokenizer()
model, config, _ = load_experimental_baby(S5B3_PARENT, device)
head = load_prop_match_head(
    Path("runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt"),
    config.d_model,
    device,
)
rt = PropMatchRuntime(model, head, tok).install()
rt.enabled = True

fact = {
    "e13": score_combine_items(model, tok, device, official_combine_pack(tok, story=False, n=32, seed=311301)),
    "s611888": score_combine_items(model, tok, device, official_combine_pack(tok, story=False, n=32, seed=611888)),
    "s612001": score_combine_items(model, tok, device, official_combine_pack(tok, story=False, n=32, seed=612001)),
    "e3": score_combine_items(model, tok, device, official_combine_pack(tok, story=False, n=16, seed=612101, n_entities=3)),
}
story = {
    "e13": score_combine_items(model, tok, device, official_combine_pack(tok, story=True, n=32, seed=311301)),
    "s611888": score_combine_items(model, tok, device, official_combine_pack(tok, story=True, n=32, seed=611888)),
}
prog = {name: score_combine_items(model, tok, device, items) for name, items in build_story_progression_heldout().items()}
about = score_about_report(model, tok, device, build_about_report_heldout())
who_rng = random.Random(324001)
who2 = score_who_with_prop_match(model, head, tok, device, [make_who_bind_item(who_rng, tok, n_entities=2, surface="heldout") for _ in range(32)])
who_sent = summarize_bind(run_who_sentence_decode(model, tok, device, include=CHEAP_OPS))
has = summarize_bind(run_has_decode(model, tok, device, include=CHEAP_OPS))
bes = summarize_bind(run_beside_decode(model, tok, device, include=CHEAP_OPS))
bes_ord = score_beside_items(model, tok, device, build_beside_order_heldout())
mixed = score_mixed_pack(model, tok, device, build_mixed_relation_heldout())
multi = score_mixed_multiturn(model, tok, device)
rt.uninstall()

def slim(scored):
    return {k: scored[k] for k in scored if k != "rows"}

out = {
    "parent": sha,
    "fact_combine": {k: slim(v) for k, v in fact.items()},
    "story_combine": {k: slim(v) for k, v in story.items()},
    "story_progress": {k: slim(v) for k, v in prog.items()},
    "about_report": slim(about),
    "retain": {
        "who_2e": who2.get("acc", who2),
        "who_sent": who_sent,
        "has": has,
        "beside": bes,
        "beside_order": slim(bes_ord),
        "mixed": slim(mixed),
        "multiturn": slim(multi),
    },
    "fact_miss": [r for r in fact["e13"]["rows"] + fact["s611888"]["rows"] + fact["e3"]["rows"] if not r["gold_in"]],
    "story_miss": [r for r in story["e13"]["rows"] + story["s611888"]["rows"] if not r["gold_in"]],
    "prog_miss": [r for items in prog.values() for r in items["rows"] if not r["gold_in"]],
}
Path("runs/actual_baby/stack2/r6/L1L2_VERIFY.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
print(json.dumps({k: out[k] for k in out if "miss" not in k}, indent=2, default=str))
print("FACT MISS", len(out["fact_miss"]), out["fact_miss"][:6])
print("STORY MISS", len(out["story_miss"]), out["story_miss"][:8])
print("PROG MISS", out["prog_miss"])
