from pathlib import Path
import json
import random

from src.baby_v010.data_language_bridge import build_e13_panels, load_tokenizer, make_who_bind_item
from src.baby_v010.selection_language_bridge import eval_panels, load_experimental_baby, run_usable_chat, slim_panels
from src.baby_v010.selection_p11_u16000_runtime import resolve_device
from src.baby_v010.selection_stack2 import run_d3_log, slim_usable
from src.baby_v010.selection_stack2_r2 import score_pack
from src.baby_v010.selection_stack2_r3 import (
    S5B3_PARENT,
    S5B3_PARENT_SHA,
    PropMatchRuntime,
    load_prop_match_head,
    verify_parent,
)
from src.baby_v010.selection_stack2_r6 import (
    build_integration_heldout,
    build_reuse_heldout,
    official_combine_pack,
    score_combine_items,
    score_integration_items,
    score_reuse_pack,
)

device = resolve_device("cuda")
sha = verify_parent(S5B3_PARENT, S5B3_PARENT_SHA, label="r6retain")
tok = load_tokenizer()
model, config, _ = load_experimental_baby(S5B3_PARENT, device)
head = load_prop_match_head(
    Path("runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt"),
    config.d_model,
    device,
)
who_items = [make_who_bind_item(random.Random(324001), tok, n_entities=2, surface="heldout") for _ in range(32)]
a_who = score_pack(model, tok, device, who_items)
rt = PropMatchRuntime(model, head, tok).install()
rt.enabled = True
usable = slim_usable(run_usable_chat(model, tok, device))
integ = score_integration_items(model, tok, device, build_integration_heldout())
reuse = score_reuse_pack(model, tok, device, build_reuse_heldout())
fact = score_combine_items(model, tok, device, official_combine_pack(tok, story=False, n=32, seed=311301))
story = score_combine_items(model, tok, device, official_combine_pack(tok, story=True, n=32, seed=311301))
panels = slim_panels(eval_panels(model, build_e13_panels(tok, n=32), device, overwrite=None, arms=("native",))["native"])
rt.uninstall()
d3 = run_d3_log(model, device, "r6_retain", english_ok=True)
out = {
    "parent": sha,
    "A": {"who_2e": a_who["acc"], "fact_combine_panels_historical": 0.65625, "story_combine_panels_historical": 0.4375},
    "B": {
        "usable4": usable["autoregressive_4turn"]["usable_turn"],
        "stop": usable["autoregressive_4turn"]["period_stop"],
        "reuse": usable["autoregressive_4turn"]["fact_reuse"],
        "integration": integ["overall"],
        "integration_by": integ["by_family"],
        "reuse_pack": reuse["reuse"],
        "fact_combine_decode": fact["gold_in"],
        "story_combine_decode": story["gold_in"],
        "panels": {
            "color": panels.get("qa_2fact_heldout"),
            "size_stop": panels.get("size_stop_heldout"),
            "mixed_2e": panels.get("mixed_2e_heldout"),
            "fact_combine": panels.get("fact_combine_heldout"),
            "story_combine": panels.get("story_combine_heldout"),
        },
    },
    "d3": {
        "n": d3["long_gap"]["n"],
        "free_exact": d3["long_gap"]["free_exact"],
        "first_correct": d3["long_gap"]["first_correct"],
        "induction": float(d3["primitive_induction"]["first_top1"]),
    },
}
Path("runs/actual_baby/stack2/r6/R6_RETAIN.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
print(json.dumps(out, indent=2, default=str))
