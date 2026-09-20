"""Deterministic graduate canary: r3 inference, 5 mechanism probes (not Form A/B)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baby_v010.data_language_bridge import load_tokenizer
from baby_v010.selection_language_bridge import load_experimental_baby
from baby_v010.selection_p11_u16000_runtime import resolve_device
from baby_v010.selection_stack2_r3 import PropMatchRuntime, load_prop_match_head
from baby_v010.selection_stack2_r6 import official_combine_pack, score_combine_items
from baby_v010.selection_stack2_r7 import score_r7_items
from baby_v010.stack2_chat import decode_user_turn

CHECKPOINT = ROOT / "runs/actual_baby/stack2/s5b3_compose_lock_326191/checkpoint_00025.pt"
HEAD = ROOT / "runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt"

CANARY_ITEMS = [
    {
        "id": "color_direct_cat",
        "family": "color",
        "facts": "",
        "query": "The cat is blue. What color is the cat?",
        "entity": "cat",
        "value": "blue",
    },
    {
        "id": "r7_who_adj_fox",
        "family": "who",
        "facts": "A pink fox hid. A tiny cow sat.",
        "query": "Tell me who is pink.",
        "entity": "fox",
        "value": "pink",
    },
    {
        "id": "r7_has_entity_white",
        "family": "has_entity",
        "facts": "The cat has the blue object. The dog has the white object.",
        "query": "Which one has the white object?",
        "entity": "dog",
        "value": "white",
    },
    {
        "id": "r7_who_e2_late",
        "family": "who",
        "facts": "The hen is yellow. The duck is pink.",
        "query": "Who is pink?",
        "entity": "duck",
        "value": "pink",
    },
]


def main() -> int:
    device = resolve_device("cuda")
    tok = load_tokenizer()
    model, config, ckpt = load_experimental_baby(CHECKPOINT, device)
    head = load_prop_match_head(HEAD, config.d_model, device)
    rt = PropMatchRuntime(model, head, tok).install()
    rt.enabled = True

    color_rows = []
    for item in CANARY_ITEMS:
        if item["family"] == "color":
            _transcript, decoded = decode_user_turn(model, tok, device, "", item["query"])
            color_rows.append(
                {
                    "id": item["id"],
                    "input": item["query"],
                    "decoded": decoded,
                    "ok": item["value"] in decoded.lower().split()[0].strip(".,!?") or item["value"] in decoded.lower(),
                }
            )

    mech = score_r7_items(model, tok, device, [i for i in CANARY_ITEMS if i["family"] != "color"])
    combine = score_combine_items(
        model, tok, device, official_combine_pack(tok, story=False, n=1, seed=611001)
    )
    combine_row = (combine.get("rows") or [{}])[0]

    rt.uninstall()

    out = {
        "root": str(ROOT),
        "runtime": "r3",
        "relassist": False,
        "whoprop": False,
        "checkpoint_update": ckpt.get("update"),
        "color_direct": color_rows,
        "mechanism": {k: mech[k] for k in ("n", "overall", "by_family", "rows") if k in mech},
        "fact_combine_one": {
            "n": combine.get("n"),
            "overall": combine.get("overall"),
            "row": combine_row,
        },
        "pass": (
            all(r["ok"] for r in color_rows)
            and mech["overall"] >= 1.0
            and bool(combine_row.get("gold_in"))
            and bool(combine_row.get("first"))
        ),
    }
    out_path = ROOT / "VERIFY_CANARY.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
