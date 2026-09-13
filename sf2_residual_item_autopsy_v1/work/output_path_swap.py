"""SF2 residual-item autopsy: output-path (upstream/norm/head) 2x2x2 swap.

READ-ONLY. Reuses the established component-swap methodology (component_swap_
forensic_v1) restricted to the residual failed Alex item, its matched Owen item,
and the other three Alex-correct items, on the first-answer-token decision and
candidate sequence margin. Donors: Pilot1 parent (P) and SF2 update-200 (S).
No training, no checkpoint mutation, no locked panels.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CHECKPOINTS, load_base, load_train

OUT = Path(r"C:\DaveLM-CADAVER\sf2_residual_item_autopsy_v1")
device = torch.device("cuda")

items_of_interest = [
    "TRAIN:g0:carried:wooden boat:a0",  # the residual failed Alex item
    "TRAIN:g0:carried:wooden boat:a1",  # its matched Owen counterfactual
    "TRAIN:g0:found:small drum:a0",
    "TRAIN:g0:found:wooden boat:a0",
    "TRAIN:g0:carried:small drum:a0",
]

train = {r["id"]: r for r in load_train()}


def donors():
    P = load_base(CHECKPOINTS["Pilot1_parent"], device)
    S = load_base(CHECKPOINTS["SF2_u200"], device)
    return P, S


def merged(target, P, S, up, norm, head):
    """Build combined base model state: up/norm/head in {P,S}."""
    for name, p in target.named_parameters():
        if name.startswith(("token_embedding", "position_embedding", "blocks")):
            src = P if up == "P" else S
        elif name.startswith("final_norm"):
            src = P if norm == "P" else S
        elif name.startswith("language_head"):
            src = P if head == "P" else S
        else:
            raise KeyError(name)
        sd = src.state_dict()
        p.data.copy_(sd[name].to(p.data.device))
    return target


def first_and_seq(model, r):
    prefix = [2] + r["prompt_token_ids"]
    ca = r["candidate_token_ids"]
    k0 = len(prefix) - 1
    with torch.no_grad():
        lg = model(torch.tensor([prefix], device=device))[0]
    c0, c1 = ca[0][0], ca[1][0]
    fm = float(lg[k0, c0] - lg[k0, c1])
    first_choice = "Alex" if lg[k0, c0] > lg[k0, c1] else "Owen"
    lls = []
    for cand in ca:
        seq = prefix + cand + [3]
        with torch.no_grad():
            lg2 = model(torch.tensor([seq], device=device))[0]
        lp = lg2.log_softmax(-1)
        lls.append(sum(float(lp[k0 + j, cand[j]]) for j in range(4)))
    ll4_margin_0m1 = lls[0] - lls[1]
    return {"first_margin_alex_owen": fm, "first_choice": first_choice,
            "ll4_alex": lls[0], "ll4_owen": lls[1], "ll4_margin_alex_owen": ll4_margin_0m1}


def main():
    from v0_8_2.model import build_model
    P, S = donors()
    target = build_model("untied").to(device)
    target.eval()
    results = []
    combos = [(u, n, h) for u in ("P", "S") for n in ("P", "S") for h in ("P", "S")]
    for item_id in items_of_interest:
        r = train[item_id]
        for (u, n, h) in combos:
            merged(target, P, S, u, n, h)
            m = first_and_seq(target, r)
            results.append({"id": item_id, "actor": r["actor"], "object": r["object"],
                            "predicate": r["predicate"], "upstream": u, "norm": n, "head": h, **m})
    (OUT / "OUTPUT_PATH_ANALYSIS.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    # Print a readable subset: native S, native P, S-up/P-head, P-up/S-head, and head/norm single swaps.
    print("id config first_margin first_choice ll4_margin")
    for rec in results:
        print(f"{rec['id'][:42]:42s} {rec['upstream']}{rec['norm']}{rec['head']} "
              f"fm={rec['first_margin_alex_owen']:8.3f} {rec['first_choice']:5s} "
              f"ll4m={rec['ll4_margin_alex_owen']:8.3f}")


if __name__ == "__main__":
    main()
