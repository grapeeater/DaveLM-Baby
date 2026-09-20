"""R7 retention: combination, chat, D3, English panels. RelAssist/WhoProp off."""

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
    score_who_with_prop_match,
    verify_parent,
)
from src.baby_v010.selection_stack2_r6 import (
    build_about_report_heldout,
    build_integration_heldout,
    build_reuse_heldout,
    official_combine_pack,
    score_about_report,
    score_combine_items,
    score_integration_items,
    score_reuse_pack,
)

OUT = Path("runs/actual_baby/stack2/r7")
HEAD = Path("runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt")

device = resolve_device("cuda")
sha = verify_parent(S5B3_PARENT, S5B3_PARENT_SHA, label="r7retain")
tok = load_tokenizer()
model, config, ckpt = load_experimental_baby(S5B3_PARENT, device)
assert not ckpt.get("protected_material_opened")
head = load_prop_match_head(HEAD, config.d_model, device)
who_items = [make_who_bind_item(random.Random(324001), tok, n_entities=2, surface="heldout") for _ in range(32)]
a_who = score_pack(model, tok, device, who_items)
rt = PropMatchRuntime(model, head, tok).install()
rt.enabled = True
usable = slim_usable(run_usable_chat(model, tok, device))
integ = score_integration_items(model, tok, device, build_integration_heldout())
reuse = score_reuse_pack(model, tok, device, build_reuse_heldout())
fact = score_combine_items(model, tok, device, official_combine_pack(tok, story=False, n=32, seed=311301))
story = score_combine_items(model, tok, device, official_combine_pack(tok, story=True, n=32, seed=311301))
about = score_about_report(model, tok, device, build_about_report_heldout())
who2 = score_who_with_prop_match(model, head, tok, device, who_items)
panels = slim_panels(eval_panels(model, build_e13_panels(tok, n=32), device, overwrite=None, arms=("native",))["native"])
rt.uninstall()
d3 = run_d3_log(model, device, "r7_retain", english_ok=True)
out = {
    "parent": sha,
    "relassist": False,
    "whoprop": False,
    "protected_opened": False,
    "form_a_used": False,
    "A": {
        "who_2e": a_who["acc"],
        "fact_combine_panels_historical": 0.65625,
        "story_combine_panels_historical": 0.4375,
    },
    "B": {
        "usable4": usable["autoregressive_4turn"]["usable_turn"],
        "stop": usable["autoregressive_4turn"]["period_stop"],
        "reuse": usable["autoregressive_4turn"]["fact_reuse"],
        "r6_integration": integ["overall"],
        "r6_integration_by": integ["by_family"],
        "r6_integration_miss": [r for r in integ["rows"] if not r["ok"]],
        "r6_reuse_pack": reuse["reuse"],
        "fact_combine_decode": fact["gold_in"],
        "story_combine_decode": story["gold_in"],
        "about": about["clean"],
        "who_2e_native": who2["acc"],
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
        "english_color": (d3.get("english") or {}).get("color") if isinstance(d3.get("english"), dict) else d3.get("english"),
    },
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "R7_RETAIN.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
print(json.dumps(out, indent=2, default=str))
