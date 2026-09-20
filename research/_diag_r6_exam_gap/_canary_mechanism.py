"""READ-ONLY R6 mechanism canaries. Diagnostic only.

Does not open TEST/FINAL/SACRED. Does not copy Final Exam Form A.
Does not train, write weights, or change runtime/config.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.baby_v010.data import BOS
from src.baby_v010.data_language_bridge import ENTITIES, VALUES, SIZES, WHO_ENTITIES, encode_ids, load_tokenizer
from src.baby_v010.selection_language_bridge import greedy_decode_until_stop, load_experimental_baby
from src.baby_v010.selection_p11_u16000_runtime import resolve_device
from src.baby_v010.selection_stack2_r3 import (
    S5B3_PARENT,
    S5B3_PARENT_SHA,
    PropMatchRuntime,
    load_prop_match_head,
    verify_parent,
)

OUT = Path(__file__).resolve().parent
HEAD = Path("runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt")


def tok_pieces(tokenizer, word: str) -> dict:
    spaced = encode_ids(tokenizer, f" {word}")
    bare = encode_ids(tokenizer, word)
    return {
        "word": word,
        "spaced_ids": spaced,
        "spaced": [tokenizer.decode([i]) for i in spaced],
        "bare_ids": bare,
        "bare": [tokenizer.decode([i]) for i in bare],
        "in_who": word in WHO_ENTITIES,
        "in_bridge": word in ENTITIES,
    }


def inspect(head, tokenizer, ids: list[int], hidden):
    suffix = list(ids)
    bound = head._query_bound(suffix)
    qspan = head._question_span(suffix, bound) if bound is not None else []
    kind = head._query_kind(qspan) if qspan else "who"
    det_src = head._answer_src(kind, qspan, suffix, bound, hidden)
    hop = head.entity_index(hidden)
    hop_src = None
    if hop is not None:
        from src.baby_v010.selection_stack2_r2 import canonicalize_entity_src

        hop_src = canonicalize_entity_src(suffix, hop, head.entity_spells, head.spaced_ent)
        hop_src = int(head.piece_to_spaced.get(hop_src, hop_src))
    q_ent = head._query_entity_src(qspan)
    q_val = head._query_value_src(qspan)
    subj = head._subject_of_value(q_val, suffix, bound)
    partner = None
    if q_ent is not None:
        partner = head._partner_src(q_ent, suffix, bound)
    gate = float(torch.sigmoid(head.gate_logit(hidden)).detach())
    return {
        "bound": bound,
        "kind": kind,
        "qspan": tokenizer.decode(qspan, skip_special_tokens=True) if qspan else "",
        "det_src": int(det_src) if det_src is not None else None,
        "det_tok": tokenizer.decode([det_src]) if det_src is not None else None,
        "hop": hop,
        "hop_tok": tokenizer.decode([suffix[hop]]) if hop is not None and 0 <= hop < len(suffix) else None,
        "hop_src_tok": tokenizer.decode([hop_src]) if hop_src is not None else None,
        "q_ent_tok": tokenizer.decode([q_ent]) if q_ent is not None else None,
        "q_val_tok": tokenizer.decode([q_val]) if q_val is not None else None,
        "subj_tok": tokenizer.decode([subj]) if subj is not None else None,
        "partner_tok": tokenizer.decode([partner]) if partner is not None else None,
        "cue_conf": head.cue_confidence(hidden),
        "gate": gate,
        "selection": "deterministic" if det_src is not None else ("hop_fallback" if hop is not None else "none"),
    }


# Mechanism probes. Not Form A items. Novel wrappers / order / lexicon / direction.
CANARIES = [
    {
        "id": "who_canon_leftward",
        "family": "who_inverse",
        "facts": "The hen is yellow. The duck is pink.",
        "query": "Who is yellow?",
        "gold": "hen",
        "note": "developmental inverse WHO template; entity before value",
    },
    {
        "id": "who_adj_before",
        "family": "who_inverse",
        "facts": "A yellow hen sat. A pink duck swam.",
        "query": "Who is yellow?",
        "gold": "hen",
        "note": "value before entity; leftward subject bind + leftward hop should miss",
    },
    {
        "id": "who_para_point",
        "family": "who_inverse",
        "facts": "The hen is yellow. The duck is pink.",
        "query": "Point out the yellow one.",
        "gold": "hen",
        "note": "no Who/Which/looks stem; still has value word",
    },
    {
        "id": "who_3e_nonrecent",
        "family": "who_inverse",
        "facts": "The hen is yellow. The duck is pink. The bear is green.",
        "query": "Who is yellow?",
        "gold": "hen",
        "note": "3 entities, gold is first fact",
    },
    {
        "id": "who_bridge_pig",
        "family": "who_inverse",
        "facts": "The pig is red. The fox is white.",
        "query": "Who is red?",
        "gold": "pig",
        "note": "pig/fox are bridge ENTITIES, not WHO_ENTITIES",
    },
    {
        "id": "has_what_canon",
        "family": "has_value",
        "facts": "The cat has the blue object. The dog has the white object.",
        "query": "What does the cat have?",
        "gold": "blue",
        "note": "HAS-value; entity in query; deterministic copy",
    },
    {
        "id": "has_who_canon",
        "family": "has_entity",
        "facts": "The cat has the blue object. The dog has the white object.",
        "query": "Which one has the white object?",
        "gold": "dog",
        "note": "HAS-entity developmental stem; no query entity",
    },
    {
        "id": "has_who_no_object",
        "family": "has_entity",
        "facts": "The cat has the blue object. The dog has the white object.",
        "query": "Who has white?",
        "gold": "dog",
        "note": "HAS-entity without 'object' and without full value spelling in some tokenizations",
    },
    {
        "id": "has_who_owner",
        "family": "has_entity",
        "facts": "The cat has the blue object. The dog has the white object.",
        "query": "Name the owner of the white object.",
        "gold": "dog",
        "note": "no has/have/belong; should not classify as HAS",
    },
    {
        "id": "bes_who_canon",
        "family": "beside",
        "facts": "The frog sits. The duck is beside the hen.",
        "query": "Who is beside the hen?",
        "gold": "duck",
        "note": "developmental BESIDE-who",
    },
    {
        "id": "bes_fact_next_to",
        "family": "beside",
        "facts": "The frog sits. The duck is next to the hen.",
        "query": "Who is beside the hen?",
        "gold": "duck",
        "note": "fact uses next-to; query uses beside",
    },
    {
        "id": "bes_query_near",
        "family": "beside",
        "facts": "The frog sits. The duck is beside the hen.",
        "query": "Who is near the hen?",
        "gold": "duck",
        "note": "query lacks beside/next/where stems",
    },
    {
        "id": "bes_landmark_first",
        "family": "beside",
        "facts": "The frog sits. The hen is beside the duck.",
        "query": "Who is beside the hen?",
        "gold": "duck",
        "note": "inverse direction: landmark is subject",
    },
    {
        "id": "direct_color_canon",
        "family": "direct",
        "facts": "The duck is pink. The hen is yellow.",
        "query": "Which color is the duck?",
        "gold": "pink",
        "note": "direct attribute; should stay strong",
    },
    {
        "id": "combine_canon",
        "family": "combine",
        "facts": "The hen is yellow. The hen is short. The duck is pink. The duck is wide.",
        "query": "Which color is the short one?",
        "gold": "yellow",
        "note": "fact combination control",
    },
    {
        "id": "reuse_after_right",
        "family": "reuse",
        "facts": "The hen is yellow. The duck is pink.",
        "query": "Who is yellow?",
        "prefix_answer": "hen is yellow.",
        "gold": "hen",
        "note": "second-turn inverse WHO after a correct first selection",
    },
    {
        "id": "reuse_after_wrong",
        "family": "reuse",
        "facts": "The hen is yellow. The duck is pink.",
        "query": "Who is yellow?",
        "prefix_answer": "duck is yellow.",
        "gold": "hen",
        "note": "second-turn inverse WHO after an incorrect first selection in context",
    },
    {
        "id": "who_sent_looks",
        "family": "who_sent",
        "facts": "Remember: the dog is huge in size. That hen is tiny in size.",
        "query": "Tell me who looks tiny.",
        "gold": "hen",
        "note": "developmental WHO-sent stem",
    },
]


def main() -> None:
    device = resolve_device("cuda")
    sha = verify_parent(S5B3_PARENT, S5B3_PARENT_SHA, label="r6_exam_gap_diag")
    tokenizer = load_tokenizer()
    model, config, _ = load_experimental_baby(S5B3_PARENT, device)
    head = load_prop_match_head(HEAD, config.d_model, device)
    rt = PropMatchRuntime(model, head, tokenizer).install()
    rt.enabled = True

    lexicon = {
        "who_entities": list(WHO_ENTITIES),
        "bridge_entities": list(ENTITIES),
        "values": list(VALUES),
        "sizes": list(SIZES),
        "pieces": [tok_pieces(tokenizer, w) for w in ("hen", "duck", "pig", "fox", "bird", "cow", "pink", "blue", "tiny", "white", "beside", "near", "owner")],
    }

    rows = []
    with torch.no_grad():
        for item in CANARIES:
            facts = item["facts"]
            if item.get("prefix_answer"):
                prompt = f"{facts} Who is yellow? {item['prefix_answer']} {item['query']}"
            else:
                prompt = f"{facts} {item['query']}"
            ids = [BOS, *encode_ids(tokenizer, prompt)]
            x = torch.tensor([ids], dtype=torch.long, device=device)
            hidden = model.forward_hidden(x)[0]
            stages = inspect(head, tokenizer, ids, hidden)
            emitted, stopped = greedy_decode_until_stop(model, ids, device, tokenizer, max_new=20)
            decoded = tokenizer.decode(emitted, skip_special_tokens=True).strip()
            gold = item["gold"]
            ok = gold.lower() in decoded.lower()
            rows.append(
                {
                    "id": item["id"],
                    "family": item["family"],
                    "note": item["note"],
                    "query": item["query"],
                    "gold": gold,
                    "decoded": decoded,
                    "ok": ok,
                    "stopped": stopped,
                    **stages,
                }
            )

    rt.uninstall()
    by = {}
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam]
        by[fam] = sum(bool(r["ok"]) for r in sub) / len(sub)
    report = {
        "diagnostic_only": True,
        "not_form_a": True,
        "parent_sha256": sha,
        "n": len(rows),
        "overall": sum(bool(r["ok"]) for r in rows) / len(rows),
        "by_family": by,
        "misses": [r["id"] for r in rows if not r["ok"]],
        "lexicon": lexicon,
        "rows": rows,
    }
    out_path = OUT / "MECHANISM_CANARY.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k not in {"rows", "lexicon"}}, indent=2))
    for r in rows:
        print(f"{r['id']:22} kind={r['kind']:14} sel={r['selection']:14} det={r['det_tok']!s:8} hop={r['hop_src_tok']!s:8} ok={r['ok']} decoded={r['decoded']!r}")


if __name__ == "__main__":
    main()
