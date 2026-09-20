from pathlib import Path
import json
import random

from src.baby_v010.data_language_bridge import load_tokenizer, make_who_bind_item
from src.baby_v010.selection_language_bridge import load_experimental_baby, run_usable_chat
from src.baby_v010.selection_p11_u16000_runtime import resolve_device
from src.baby_v010.selection_stack2 import slim_usable
from src.baby_v010.selection_stack2_r3 import (
    S5B3_PARENT,
    S5B3_PARENT_SHA,
    PropMatchRuntime,
    load_prop_match_head,
    score_who_with_prop_match,
    verify_parent,
)
from src.baby_v010.selection_stack2_r4 import (
    build_beside_paraphrase_heldout,
    build_has_paraphrase_heldout,
    build_who_paraphrase_heldout,
    score_paraphrase_pack,
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
    build_integration_heldout,
    build_reuse_heldout,
    official_combine_pack,
    score_about_report,
    score_combine_items,
    score_integration_items,
    score_reuse_pack,
)
from src.baby_v010.selection_stack2_s5 import CHEAP_OPS, run_beside_decode, run_has_decode, run_who_sentence_decode, summarize_bind

device = resolve_device("cuda")
sha = verify_parent(S5B3_PARENT, S5B3_PARENT_SHA, label="r6l4")
tok = load_tokenizer()
model, config, _ = load_experimental_baby(S5B3_PARENT, device)
head = load_prop_match_head(
    Path("runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt"),
    config.d_model,
    device,
)
rt = PropMatchRuntime(model, head, tok).install()
rt.enabled = True
integ = score_integration_items(model, tok, device, build_integration_heldout())
reuse = score_reuse_pack(model, tok, device, build_reuse_heldout())
usable = slim_usable(run_usable_chat(model, tok, device))
who_rng = random.Random(324001)
who2 = score_who_with_prop_match(model, head, tok, device, [make_who_bind_item(who_rng, tok, n_entities=2, surface="heldout") for _ in range(32)])
who_sent = summarize_bind(run_who_sentence_decode(model, tok, device, include=CHEAP_OPS))
has = summarize_bind(run_has_decode(model, tok, device, include=CHEAP_OPS))
bes = summarize_bind(run_beside_decode(model, tok, device, include=CHEAP_OPS))
para = {
    "who": score_paraphrase_pack(model, tok, device, build_who_paraphrase_heldout(), family="who")["bare"],
    "has": score_paraphrase_pack(model, tok, device, build_has_paraphrase_heldout(), family="has")["bare"],
    "beside": score_paraphrase_pack(model, tok, device, build_beside_paraphrase_heldout(), family="beside")["bare"],
}
mixed = score_mixed_pack(model, tok, device, build_mixed_relation_heldout())
multi = score_mixed_multiturn(model, tok, device)
fact = score_combine_items(model, tok, device, official_combine_pack(tok, story=False, n=32, seed=311301))
story = score_combine_items(model, tok, device, official_combine_pack(tok, story=True, n=32, seed=311301))
about = score_about_report(model, tok, device, build_about_report_heldout())
bes_ord = score_beside_items(model, tok, device, build_beside_order_heldout())
rt.uninstall()
out = {
    "parent": sha,
    "integration": {k: integ[k] for k in integ if k != "rows"},
    "integration_miss": [r for r in integ["rows"] if not r["ok"]],
    "reuse": {k: reuse[k] for k in reuse if k != "rows"},
    "usable4": usable["autoregressive_4turn"]["usable_turn"],
    "stop": usable["autoregressive_4turn"]["period_stop"],
    "usable_reuse": usable["autoregressive_4turn"]["fact_reuse"],
    "who_2e": who2["acc"],
    "who_sent": who_sent["bare"],
    "has": has["bare"],
    "beside": bes["bare"],
    "para": para,
    "mixed": mixed["bare"],
    "multiturn": multi["bare"],
    "fact_combine": fact["gold_in"],
    "story_combine": story["gold_in"],
    "about": about["clean"],
    "beside_order": bes_ord["bare"],
}
Path("runs/actual_baby/stack2/r6/L4_CANARY.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
