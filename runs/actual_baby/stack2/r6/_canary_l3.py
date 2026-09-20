from pathlib import Path
import json

from src.baby_v010.data_language_bridge import load_tokenizer
from src.baby_v010.selection_language_bridge import load_experimental_baby, run_usable_chat
from src.baby_v010.selection_p11_u16000_runtime import resolve_device
from src.baby_v010.selection_stack2 import slim_usable
from src.baby_v010.selection_stack2_r3 import (
    S5B3_PARENT,
    S5B3_PARENT_SHA,
    PropMatchRuntime,
    load_prop_match_head,
    verify_parent,
)
from src.baby_v010.selection_stack2_r6 import (
    build_about_report_heldout,
    build_reuse_heldout,
    official_combine_pack,
    score_about_report,
    score_combine_items,
    score_reuse_pack,
)

device = resolve_device("cuda")
sha = verify_parent(S5B3_PARENT, S5B3_PARENT_SHA, label="r6l3")
tok = load_tokenizer()
model, config, _ = load_experimental_baby(S5B3_PARENT, device)
head = load_prop_match_head(
    Path("runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt"),
    config.d_model,
    device,
)
rt = PropMatchRuntime(model, head, tok).install()
rt.enabled = True
reuse = score_reuse_pack(model, tok, device, build_reuse_heldout())
usable = slim_usable(run_usable_chat(model, tok, device))
fact = score_combine_items(model, tok, device, official_combine_pack(tok, story=False, n=16, seed=611001))
story = score_combine_items(model, tok, device, official_combine_pack(tok, story=True, n=16, seed=311301))
about = score_about_report(model, tok, device, build_about_report_heldout())
rt.uninstall()
out = {
    "parent": sha,
    "reuse": {k: reuse[k] for k in reuse if k != "rows"},
    "usable4": usable["autoregressive_4turn"]["usable_turn"],
    "stop": usable["autoregressive_4turn"]["period_stop"],
    "usable_reuse": usable["autoregressive_4turn"]["fact_reuse"],
    "fact_hold": {k: fact[k] for k in fact if k != "rows"},
    "story_hold": {k: story[k] for k in story if k != "rows"},
    "about": {k: about[k] for k in about if k != "rows"},
    "reuse_miss": [r for r in reuse["rows"] if not r["hit"]],
}
Path("runs/actual_baby/stack2/r6/L3_CANARY.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: out[k] for k in out if k != "reuse_miss"}, indent=2))
print("REUSE MISS")
for row in out["reuse_miss"]:
    print(row)
