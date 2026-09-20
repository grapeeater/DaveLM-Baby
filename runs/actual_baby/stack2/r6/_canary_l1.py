from pathlib import Path
import json

from src.baby_v010.data_language_bridge import load_tokenizer
from src.baby_v010.selection_language_bridge import load_experimental_baby
from src.baby_v010.selection_p11_u16000_runtime import resolve_device
from src.baby_v010.selection_stack2_r3 import (
    S5B3_PARENT,
    S5B3_PARENT_SHA,
    PropMatchRuntime,
    load_prop_match_head,
    verify_parent,
)
from src.baby_v010.selection_stack2_r6 import (
    build_about_report_heldout,
    official_combine_pack,
    score_about_report,
    score_combine_items,
)

device = resolve_device("cuda")
sha = verify_parent(S5B3_PARENT, S5B3_PARENT_SHA, label="r6l1")
tok = load_tokenizer()
model, config, _ = load_experimental_baby(S5B3_PARENT, device)
head = load_prop_match_head(
    Path("runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt"),
    config.d_model,
    device,
)
rt = PropMatchRuntime(model, head, tok).install()
rt.enabled = True
pack_a = official_combine_pack(tok, story=False, n=16, seed=611001)
pack_b = official_combine_pack(tok, story=False, n=16, seed=611777)
about = score_about_report(model, tok, device, build_about_report_heldout())
fact_a = score_combine_items(model, tok, device, pack_a)
fact_b = score_combine_items(model, tok, device, pack_b)
rt.uninstall()
out = {
    "parent": sha,
    "mechanism": "cross-attribute bind + about report finish",
    "fact_combine_611001": {k: fact_a[k] for k in fact_a if k != "rows"},
    "fact_combine_611777": {k: fact_b[k] for k in fact_b if k != "rows"},
    "about_report": {k: about[k] for k in about if k != "rows"},
    "fact_a_rows": fact_a["rows"],
    "fact_b_rows": fact_b["rows"],
    "about_rows": about["rows"],
}
Path("runs/actual_baby/stack2/r6/L1_CANARY.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: out[k] for k in out if "rows" not in k}, indent=2))
print("MISS A")
for row in fact_a["rows"]:
    if not row["gold_in"]:
        print(row)
print("MISS ABOUT")
for row in about["rows"]:
    if not row["attrs_ok"]:
        print(row)
