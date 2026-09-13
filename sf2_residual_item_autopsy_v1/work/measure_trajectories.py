"""SF2 residual-item autopsy: item-level trajectories across authorized checkpoints.

READ-ONLY inference. Loads four existing checkpoints (Pilot1 parent, SF1 u100,
SF2 u100, SF2 u200) and measures per-item first-token and sequence quantities on
the 16 authorized TRAIN records. No training, no weight mutation, no new
checkpoints, no locked panels.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BUNDLE, CHECKPOINTS, TOKENIZER_PATH, load_base, load_train, load_raw)

import torch.nn.functional as F
from tokenizers import Tokenizer

OUT = Path(r"C:\DaveLM-CADAVER\sf2_residual_item_autopsy_v1")
device = torch.device("cuda")
tok = Tokenizer.from_file(str(TOKENIZER_PATH))
train = load_train()


def measure(model, r):
    """Per-item measures on base_model only (no T13 path)."""
    prefix = [2] + r["prompt_token_ids"]
    ca = r["candidate_token_ids"]  # [ [alex 4], [owen 4] ] for g0 ; [mia],[nora] g1
    k0 = len(prefix) - 1  # logits index predicting first candidate token
    # first-token distribution (context identical for both candidates)
    x = torch.tensor([prefix], dtype=torch.long, device=device)
    with torch.no_grad():
        lg = model(x)[0]  # [T,V]
    first_logp = lg[k0].log_softmax(-1)
    first_p = first_logp.exp()
    c0 = ca[0][0]
    c1 = ca[1][0]
    out = {
        "first_token_0_id": int(c0), "first_token_1_id": int(c1),
        "first_token_0_logit": float(lg[k0, c0]), "first_token_1_logit": float(lg[k0, c1]),
        "first_token_0_prob": float(first_p[c0]), "first_token_1_prob": float(first_p[c1]),
        "first_token_0_logprob": float(first_logp[c0]), "first_token_1_logprob": float(first_logp[c1]),
        "first_margin_logit_0minus1": float(lg[k0, c0] - lg[k0, c1]),
    }
    # candidate sequence log-likelihoods (teacher-forced; 4 tokens then EOS)
    lls = {}
    for idx, cand in enumerate(ca):
        seq = prefix + cand + [3]
        x = torch.tensor([seq], dtype=torch.long, device=device)
        with torch.no_grad():
            lg2 = model(x)[0]
        lp = lg2.log_softmax(-1)
        ll4 = sum(float(lp[k0 + j, cand[j]]) for j in range(4))
        ll5 = ll4 + float(lp[k0 + 4, 3])
        lls[idx] = {"ll4": ll4, "ll5": ll5}
    # greedy generation
    gen = []
    seq = list(prefix)
    with torch.no_grad():
        for _ in range(32):
            x = torch.tensor([seq[-256:]], dtype=torch.long, device=device)
            t = int(model(x)[0, -1].argmax().item())
            seq.append(t)
            gen.append(t)
            if t == 3:
                break
    ci = r["correct_index"]
    out["cand0_ll4"] = lls[0]["ll4"]
    out["cand1_ll4"] = lls[1]["ll4"]
    out["cand0_ll5"] = lls[0]["ll5"]
    out["cand1_ll5"] = lls[1]["ll5"]
    out["seq_margin_ll4_correct_minus_other"] = lls[ci]["ll4"] - lls[1 - ci]["ll4"]
    out["seq_margin_ll4_0minus1"] = lls[0]["ll4"] - lls[1]["ll4"]
    out["seq_margin_ll5_0minus1"] = lls[0]["ll5"] - lls[1]["ll5"]
    target = ca[ci] + [3]
    out["greedy_ids"] = gen
    out["greedy_text"] = tok.decode(gen, skip_special_tokens=True)
    out["exact"] = gen == target
    return out


rows = []
for ck_name, ck_path in CHECKPOINTS.items():
    model = load_base(ck_path, device)
    for r in train:
        m = measure(model, r)
        rec = {
            "id": r["id"], "checkpoint": ck_name, "actor": r["actor"], "object": r["object"],
            "predicate": r["predicate"], "correct_index": r["correct_index"],
            "correct_name": r["candidates"][r["correct_index"]].strip().rstrip("."),
        }
        rec.update(m)
        rows.append(rec)
    del model
    torch.cuda.empty_cache()

with (OUT / "ITEM_TRAJECTORIES.jsonl").open("w", encoding="utf-8", newline="\n") as h:
    for rec in rows:
        h.write(json.dumps(rec, ensure_ascii=False) + "\n")

# print AO table
print("id checkpoint firstmargin0m1 (alex-owen logit) seq_ll4 margin0m1 exact")
for rec in rows:
    if "Alex" in rec["id"] or "Owen" in rec["id"] or rec["actor"] in ("Alex", "Owen"):
        if rec["actor"] in ("Alex", "Owen"):
            print(f"{rec['id'][:42]:42s} {rec['checkpoint']:12s} fm={rec['first_margin_logit_0minus1']:8.3f} "
                  f"ll4m={rec['seq_margin_ll4_0minus1']:8.3f} exact={int(rec['exact'])} gen={rec['greedy_text'][:12]!r}")
print("written", len(rows), "rows")
