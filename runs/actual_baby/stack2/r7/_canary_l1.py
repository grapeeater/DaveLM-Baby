"""R7 L1: tiny mechanism canary. Independently generated. Not Form A."""

from pathlib import Path
import json

import torch

from src.baby_v010.data import BOS
from src.baby_v010.data_language_bridge import encode_ids, load_tokenizer
from src.baby_v010.selection_language_bridge import greedy_decode_until_stop, load_experimental_baby
from src.baby_v010.selection_p11_u16000_runtime import resolve_device
from src.baby_v010.selection_stack2_r3 import (
    S5B3_PARENT,
    S5B3_PARENT_SHA,
    PropMatchRuntime,
    load_prop_match_head,
    verify_parent,
)
from src.baby_v010.selection_stack2_r7 import build_r7_mechanism_canary, score_r7_items

OUT = Path("runs/actual_baby/stack2/r7")
HEAD = Path("runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt")


def inspect(head, tokenizer, prompt: str, device, model):
    ids = [BOS, *encode_ids(tokenizer, prompt)]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        hidden = model.forward_hidden(x)[0]
    suffix = list(ids)
    bound = head._query_bound(suffix)
    qspan = head._question_span(suffix, bound) if bound is not None else []
    kind = head._query_kind(qspan) if qspan else "who"
    speak = head._speech_kind(kind, qspan, suffix, bound)
    plan = head._speech_plan(speak, qspan, suffix, bound)
    src = head._answer_src(speak, qspan, suffix, bound, hidden)
    return {
        "bound": bound,
        "kind": kind,
        "speak": speak,
        "qspan": tokenizer.decode(qspan, skip_special_tokens=True) if qspan else "",
        "plan": tokenizer.decode(plan, skip_special_tokens=True) if plan else "",
        "src": int(src) if src is not None else None,
        "src_tok": tokenizer.decode([src]) if src is not None else None,
        "hop_used": False,
    }


device = resolve_device("cuda")
sha = verify_parent(S5B3_PARENT, S5B3_PARENT_SHA, label="r7l1")
tok = load_tokenizer()
model, config, ckpt = load_experimental_baby(S5B3_PARENT, device)
assert not ckpt.get("protected_material_opened")
head = load_prop_match_head(HEAD, config.d_model, device)
rt = PropMatchRuntime(model, head, tok).install()
rt.enabled = True
items = build_r7_mechanism_canary()
internals = []
for item in items:
    prompt = f"{item['facts']} {item['query']}"
    internals.append({"id": item["id"], **inspect(head, tok, prompt, device, model)})
scored = score_r7_items(model, tok, device, items)
rt.uninstall()
out = {
    "parent": sha,
    "relassist": False,
    "whoprop": False,
    "hop_primary": False,
    "n": scored["n"],
    "overall": scored["overall"],
    "entity_lead": scored["entity_lead"],
    "by_family": scored["by_family"],
    "miss": scored["miss"],
    "internals": internals,
    "rows": scored["rows"],
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "L1_CANARY.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: out[k] for k in ("n", "overall", "entity_lead", "by_family", "miss")}, indent=2))
print("INTERNALS")
for row in internals:
    print(row["id"], row["kind"], "->", row["speak"], "|", row["plan"])
