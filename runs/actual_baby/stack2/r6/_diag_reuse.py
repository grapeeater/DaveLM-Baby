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

device = resolve_device("cuda")
sha = verify_parent(S5B3_PARENT, S5B3_PARENT_SHA, label="r6reuse")
tok = load_tokenizer()
model, config, _ = load_experimental_baby(S5B3_PARENT, device)
head = load_prop_match_head(
    Path("runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt"),
    config.d_model,
    device,
)
rt = PropMatchRuntime(model, head, tok).install()
rt.enabled = True
full = run_usable_chat(model, tok, device)
rt.uninstall()
usable = slim_usable(full)
misses = []
for chat in full["chats_auto"]:
    for turn in chat["turns"]:
        if turn.get("fact_reuse") is False or (turn.get("usable") is False):
            misses.append(
                {
                    "chat": chat["id"],
                    "human": turn["human"],
                    "gold": turn["gold"],
                    "decoded": turn["decoded"],
                    "skills": turn["skills"],
                    "fact_hit": turn.get("fact_hit"),
                    "fact_reuse": turn.get("fact_reuse"),
                    "usable": turn.get("usable"),
                    "period_stop": turn.get("period_stop"),
                    "rambling": turn.get("rambling"),
                    "on_topic": turn.get("on_topic"),
                    "distractors": turn.get("distractor_hits"),
                }
            )
out = {
    "parent": sha,
    "usable4": usable["autoregressive_4turn"]["usable_turn"],
    "stop": usable["autoregressive_4turn"]["period_stop"],
    "reuse": usable["autoregressive_4turn"]["fact_reuse"],
    "usable5": usable["autoregressive_5turn"]["usable_turn"],
    "reuse5": usable["autoregressive_5turn"]["fact_reuse"],
    "misses": misses,
}
Path("runs/actual_baby/stack2/r6/DIAG_REUSE.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: out[k] for k in out if k != "misses"}, indent=2))
print("MISSES", len(misses))
for row in misses:
    print(row["chat"], row["skills"], row["gold"], "=>", row["decoded"], row)
