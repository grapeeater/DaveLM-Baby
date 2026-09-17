"""Decode-time gen-index greedy / query-follow / binding for P11.

Protocol: `design/V010_SELECTION_REPAIR_P11_DECODE.md`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .residual_overwrite import LocalSlotOverwrite, attach_overwrite
from .selection_p1 import bootstrap_delta
from .selection_p3 import DEBUG_LOG, SESSION
from .selection_p11 import GATE_BIAS, OUT as P11_OUT, TRAIN_SEED
from .selection_s1 import PARENT, PARENT_SHA, digest, model_load, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/selection_p11_decode"
PROTOCOL = ROOT / "design/V010_SELECTION_REPAIR_P11_DECODE.md"
S2_DIAGNOSTIC = ROOT / "runs/selection_s2/DIAGNOSTIC.json"
TREATMENT_CKPT = P11_OUT / f"treatment_{TRAIN_SEED}" / "checkpoint_16800.pt"
BOOTSTRAP_SEED = 250300
MIN_GAP = 13
SUCCESS_DELTA = 0.10


def posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    payload = {
        "sessionId": SESSION,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with DEBUG_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    # #endregion


def long_items(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        gap = len(row["input"]) - 1 - int(row["query_position"])
        if gap >= MIN_GAP:
            out.append(row)
    return out


def load_arm(arm: str):
    model, config = model_load()
    model.set_attention_backend("sdpa")
    model.eval()
    overwrite = LocalSlotOverwrite(config.d_model, gate_bias=GATE_BIAS, gen_only=True).to(
        next(model.parameters()).device
    )
    if arm == "treatment":
        ckpt = torch.load(TREATMENT_CKPT, map_location="cpu", weights_only=False)
        if ckpt.get("parent_checkpoint_sha256") != PARENT_SHA:
            raise RuntimeError("treatment checkpoint parent mismatch")
        overwrite.load_state_dict(ckpt["overwrite_state_dict"])
    overwrite.eval()
    handle, _ = attach_overwrite(model, overwrite)
    return model, overwrite, handle


def greedy_span(model, overwrite, item: dict, device, n_tokens: int) -> list[int]:
    generated = [int(t) for t in item["input"]]
    emitted: list[int] = []
    with torch.no_grad():
        for _ in range(n_tokens):
            overwrite.gen_index = torch.tensor([len(generated) - 1], device=device, dtype=torch.long)
            x = torch.tensor([generated[-256:]], dtype=torch.long, device=device)
            logits = model(x)
            token = int(logits[0, -1].argmax().item())
            emitted.append(token)
            generated.append(token)
    return emitted


def score_arm(arm: str, items: list[dict]) -> dict:
    model, overwrite, handle = load_arm(arm)
    device = next(model.parameters()).device
    rows = []
    try:
        for item in items:
            target = [int(t) for t in item["target"]]
            emitted = greedy_span(model, overwrite, item, device, len(target))
            pred_head = emitted[0] if emitted else None
            gold_head = target[0]
            rows.append(
                {
                    "body_id": item["body_id"],
                    "query_index": item["query_index"],
                    "K": item["pair_count"],
                    "gold_head": gold_head,
                    "pred_head": pred_head,
                    "free_exact": emitted == target,
                    "first_correct": pred_head == gold_head,
                    "inventory_heads": list(item["candidate_heads"]),
                    "pred_in_inventory": pred_head in set(item["candidate_heads"]),
                    "query_key": item.get("query_key"),
                }
            )
    finally:
        handle.remove()
        overwrite.gen_index = None
    n = len(rows)
    first = sum(int(r["first_correct"]) for r in rows)
    free = sum(int(r["free_exact"]) for r in rows)
    inv = sum(int(r["pred_in_inventory"]) for r in rows)
    debug_log(
        "H1",
        "selection_p11_decode.py:score_arm",
        "arm_summary",
        {"arm": arm, "n": n, "first_correct": first, "free_exact": free, "pred_in_inventory": inv},
    )
    return {
        "arm": arm,
        "n": n,
        "first_correct": first,
        "first_accuracy": first / n,
        "free_exact": free,
        "free_accuracy": free / n,
        "pred_in_inventory": inv,
        "rows": rows,
    }


def query_follow(rows: list[dict]) -> dict:
    from collections import defaultdict

    bodies = defaultdict(list)
    for row in rows:
        bodies[row["body_id"]].append(row)
    multi = 0
    vary = 0
    follow = 0
    n = 0
    pair_n = 0
    pair_bind = 0
    for group in bodies.values():
        if len(group) < 2:
            continue
        multi += 1
        preds = {r["pred_head"] for r in group}
        if len(preds) > 1:
            vary += 1
        for row in group:
            n += 1
            follow += int(row["pred_head"] == row["gold_head"])
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if a["gold_head"] == b["gold_head"]:
                    continue
                pair_n += 1
                pair_bind += int(a["pred_head"] == a["gold_head"] and b["pred_head"] == b["gold_head"])
    return {
        "multi_query_bodies": multi,
        "bodies_pred_varies": vary,
        "follow_n": n,
        "follow_hits": follow,
        "follow_rate": (follow / n) if n else 0.0,
        "pair_n": pair_n,
        "pair_bind": pair_bind,
        "pair_bind_rate": (pair_bind / pair_n) if pair_n else 0.0,
    }


def decide_h1(treat: dict, init: dict, ci: tuple[float, float]) -> str:
    delta = treat["free_accuracy"] - init["free_accuracy"]
    if delta >= SUCCESS_DELTA and ci[0] > 0:
        return "H1_PASS"
    if delta <= 0 and ci[1] < SUCCESS_DELTA:
        return "H1_FAIL"
    return "H1_MIXED"


def run() -> dict:
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    if not TREATMENT_CKPT.exists():
        raise RuntimeError("missing P11 treatment checkpoint")
    items = long_items(json.loads(S2_DIAGNOSTIC.read_text(encoding="utf-8")))
    OUT.mkdir(parents=True, exist_ok=True)
    treat = score_arm("treatment", items)
    init = score_arm("init", items)
    treat_flags = [{"inventory_correct": r["free_exact"]} for r in treat["rows"]]
    init_flags = [{"inventory_correct": r["free_exact"]} for r in init["rows"]]
    ci = bootstrap_delta(treat_flags, init_flags, seed=BOOTSTRAP_SEED)
    h1 = decide_h1(treat, init, ci)
    h2 = {"treatment": query_follow(treat["rows"]), "init": query_follow(init["rows"])}
    decision = {
        "protocol": "V010_SELECTION_REPAIR_P11_DECODE",
        "licensed_by": "V010_SELECTION_REPAIR_P11_GEN_ONLY",
        "parent_sha256": PARENT_SHA,
        "treatment_checkpoint": posix(TREATMENT_CKPT),
        "treatment_checkpoint_sha256": digest(TREATMENT_CKPT),
        "n_long": len(items),
        "h1_verdict": h1,
        "primary": {
            "treatment_free_exact": treat["free_exact"],
            "init_free_exact": init["free_exact"],
            "treatment_first_correct": treat["first_correct"],
            "init_first_correct": init["first_correct"],
            "free_delta": treat["free_accuracy"] - init["free_accuracy"],
            "bootstrap_ci95": list(ci),
        },
        "h2_query_follow": h2,
        "h3_pair_bind_delta": h2["treatment"]["pair_bind_rate"] - h2["init"]["pair_bind_rate"],
        "protected_material_opened": False,
        "authoritative_parent_unchanged": True,
        "p11_gates_reopened": False,
    }
    write(OUT / "ADJUDICATION.json", decision)
    write(OUT / "TREATMENT.json", {k: v for k, v in treat.items() if k != "rows"})
    write(OUT / "INIT.json", {k: v for k, v in init.items() if k != "rows"})
    print(json.dumps({"h1": h1, "primary": decision["primary"], "h2": h2}), flush=True)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run"])
    args = parser.parse_args()
    if args.action == "run":
        run()


if __name__ == "__main__":
    main()
