"""D3b: L0 splice control decomposition. Does not rewrite D3 receipts."""
from __future__ import annotations

import json
import time
from pathlib import Path

from .query_splice_d3 import (
    DEBUG_LOG,
    LONG_GAP,
    OUT as D3_OUT,
    PARENT,
    PARENT_SHA,
    S2_DIAGNOSTIC,
    SESSION,
    capture_blocks,
    digest,
    eligible_item,
    load_ckpt,
    splice_block,
    splice_sources,
    write,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/query_splice_d3b"
PROTOCOL = ROOT / "design/V010_QUERY_SPLICE_D3B.md"
D3_ADJUDICATION = D3_OUT / "ADJUDICATION.json"
LIFT = 0.10
NEG = 0.03
AS_IS_HITS = 74
Q0R_HITS = 127


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


def pair_index_at_start(item: dict, start: int) -> int | None:
    from .isolation_transforms import recover_rendered_pairs

    for row in recover_rendered_pairs(item):
        if int(row["start"]) == start:
            return int(row["source_index"])
    return None


def filler_position(item: dict, sources: dict) -> int | None:
    from .isolation_transforms import recover_rendered_pairs

    forbidden = {
        int(sources["query"]),
        int(sources["match_key"]),
        int(sources["competitor_key"]),
        len(item["input"]) - 1,
    }
    forbidden.update(int(row["start"]) for row in recover_rendered_pairs(item))
    gen = len(item["input"]) - 1
    for index in range(gen - 1, -1, -1):
        if index not in forbidden:
            return index
    return None


def pred_index(logits_row, item: dict) -> int:
    import torch

    heads = [int(h) for h in item["candidate_heads"]]
    return int(torch.argmax(logits_row[heads]).item())


def decide(
    *,
    full_as_hits: int,
    full_q0r_hits: int,
    n_shared: int,
    g_as: float,
    g_q: float,
    g_c: float,
    g_r: float,
    s_as: float,
    s_c: float,
) -> dict:
    if full_as_hits != AS_IS_HITS or full_q0r_hits != Q0R_HITS or n_shared < 50:
        return {
            "verdict": "INVALID",
            "dg_q0r": None,
            "dg_c0r": None,
            "dg_r0r": None,
            "ds_c0r": None,
        }
    dg_q = g_q - g_as
    dg_c = g_c - g_as
    dg_r = g_r - g_as
    ds_c = s_c - s_as
    if dg_r >= LIFT:
        verdict = "OVERWRITE"
    elif dg_q >= LIFT and ds_c >= LIFT and dg_c <= NEG and dg_r <= NEG:
        verdict = "SPLICED_IDENTITY"
    elif dg_q >= LIFT and dg_c <= NEG and dg_r <= NEG:
        verdict = "QUERY_IDENTITY"
    elif dg_q >= LIFT and dg_c >= LIFT and dg_r <= NEG and ds_c < LIFT:
        verdict = "KEY_SUBSPACE"
    elif dg_q < LIFT:
        verdict = "NONE"
    else:
        verdict = "MIXED"
    return {
        "verdict": verdict,
        "dg_q0r": dg_q,
        "dg_c0r": dg_c,
        "dg_r0r": dg_r,
        "ds_c0r": ds_c,
    }


def licenses_next(verdict: str) -> str:
    if verdict in {"QUERY_IDENTITY", "SPLICED_IDENTITY", "KEY_SUBSPACE"}:
        return "early_query_pointer_layers_0_2_not_p2_not_p1_all_layers"
    if verdict == "OVERWRITE":
        return "do_not_replace_whole_residual_next_change_attention_q"
    return "no_training_license"


def write_manifest() -> dict:
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    d3 = json.loads(D3_ADJUDICATION.read_text(encoding="utf-8"))
    if d3.get("protocol") != "V010_QUERY_SPLICE_D3" or d3.get("verdict") != "MIXED":
        raise RuntimeError("D3b is licensed only by D3 MIXED")
    files = [
        PROTOCOL,
        Path(__file__),
        D3_ADJUDICATION,
        S2_DIAGNOSTIC,
        PARENT,
        ROOT / "src/baby_v010/query_splice_d3.py",
        ROOT / "design/V010_QUERY_SPLICE_D3.md",
    ]
    payload = {
        "protocol": "V010_QUERY_SPLICE_D3B",
        "parent_sha256": PARENT_SHA,
        "licensed_by": "V010_QUERY_SPLICE_D3",
        "d3_verdict": "MIXED",
        "p2_launched": False,
        "protected_material_opened": False,
        "files": {posix(path): digest(path) for path in files},
    }
    write(OUT / "MANIFEST.json", payload)
    return payload


def rate(hits: int, n: int) -> float:
    return hits / n if n else float("nan")


def run(device: str = "cuda") -> dict:
    import torch

    if (OUT / "QUERY_SPLICE.json").exists():
        raise RuntimeError("refusing to overwrite D3b results")
    write_manifest()
    items = [row for row in json.loads(S2_DIAGNOSTIC.read_text(encoding="utf-8")) if eligible_item(row)]
    long_items = [
        item
        for item in items
        if (len(item["input"]) - 1 - int(item["query_position"])) >= LONG_GAP
    ]
    model, blob = load_ckpt(PARENT, device)
    if int(blob.get("update", -1)) != 16000:
        raise RuntimeError("parent update is not 16000")
    model.set_attention_backend("sdpa")
    debug_log("H_QUERY", "query_splice_d3b.py:run", "d3b_start", {"n_long": len(long_items)})

    full_as = 0
    full_q0r = 0
    shared = []
    with torch.no_grad():
        for index, item in enumerate(long_items):
            sources = splice_sources(item)
            gen = len(item["input"]) - 1
            tokens = torch.tensor([item["input"]], dtype=torch.long, device=device)
            logits, blocks = capture_blocks(model, tokens)
            as_pred = pred_index(logits[0, gen], item)
            as_gold = as_pred == int(item["query_index"])
            q_out = splice_block(model, tokens, gen, 0, blocks[0][0, int(sources["query"])], False)
            q_pred = pred_index(q_out[0, gen], item)
            q_gold = q_pred == int(item["query_index"])
            full_as += int(as_gold)
            full_q0r += int(q_gold)

            fill = filler_position(item, sources)
            comp_pair = pair_index_at_start(item, int(sources["competitor_key"]))
            if fill is None or comp_pair is None:
                continue
            c_out = splice_block(
                model, tokens, gen, 0, blocks[0][0, int(sources["competitor_key"])], False
            )
            r_out = splice_block(model, tokens, gen, 0, blocks[0][0, fill], False)
            c_pred = pred_index(c_out[0, gen], item)
            r_pred = pred_index(r_out[0, gen], item)
            gold = int(item["query_index"])
            shared.append(
                {
                    "as_gold": as_gold,
                    "q_gold": q_gold,
                    "c_gold": c_pred == gold,
                    "r_gold": r_pred == gold,
                    "as_spliced": as_pred == comp_pair,
                    "c_spliced": c_pred == comp_pair,
                    "K": int(item["pair_count"]),
                }
            )
            if index % 40 == 0:
                print(json.dumps({"item": index, "n": len(long_items)}), flush=True)

    n = len(shared)
    g_as = rate(sum(int(r["as_gold"]) for r in shared), n)
    g_q = rate(sum(int(r["q_gold"]) for r in shared), n)
    g_c = rate(sum(int(r["c_gold"]) for r in shared), n)
    g_r = rate(sum(int(r["r_gold"]) for r in shared), n)
    s_as = rate(sum(int(r["as_spliced"]) for r in shared), n)
    s_c = rate(sum(int(r["c_spliced"]) for r in shared), n)
    decision = decide(
        full_as_hits=full_as,
        full_q0r_hits=full_q0r,
        n_shared=n,
        g_as=g_as,
        g_q=g_q,
        g_c=g_c,
        g_r=g_r,
        s_as=s_as,
        s_c=s_c,
    )
    decision["licenses_next"] = licenses_next(decision["verdict"])
    debug_log(
        "H_QUERY",
        "query_splice_d3b.py:run",
        "d3b_result",
        {"full_as": full_as, "full_q0r": full_q0r, "n_shared": n, **decision},
    )
    report = {
        "protocol": "V010_QUERY_SPLICE_D3B",
        "parent_sha256": PARENT_SHA,
        "full_long_n": len(long_items),
        "full_as_hits": full_as,
        "full_q0r_hits": full_q0r,
        "n_shared": n,
        "shared": {
            "as_is_gold": g_as,
            "Q0R_gold": g_q,
            "C0R_gold": g_c,
            "R0R_gold": g_r,
            "as_is_competitor_spliced": s_as,
            "C0R_spliced": s_c,
        },
        "decision": decision,
        "trained": False,
        "p2_launched": False,
        "protected_material_opened": False,
        "gates_changed": False,
        "d3_verdict_unchanged": "MIXED",
        "authoritative_parent_unchanged": True,
    }
    write(OUT / "QUERY_SPLICE.json", report)
    write(OUT / "ADJUDICATION.json", {
        "protocol": "V010_QUERY_SPLICE_D3B",
        **decision,
        "parent_sha256": PARENT_SHA,
        "full_as_hits": full_as,
        "full_q0r_hits": full_q0r,
        "n_shared": n,
        "shared": report["shared"],
        "trained": False,
        "p2_launched": False,
        "protected_material_opened": False,
        "gates_changed": False,
        "authoritative_parent_unchanged": True,
    })
    print(json.dumps({"verdict": decision["verdict"], "licenses_next": decision["licenses_next"]}), flush=True)
    return report
