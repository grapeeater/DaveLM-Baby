"""D4: L0 attention one-hot patch. Read-only. No optimizer."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch.nn.functional as F

from .query_splice_d3 import (
    DEBUG_LOG,
    LONG_GAP,
    PARENT,
    PARENT_SHA,
    S2_DIAGNOSTIC,
    SESSION,
    capture_blocks,
    digest,
    eligible_item,
    load_ckpt,
    logit_steer,
    splice_block,
    splice_sources,
    write,
)
from .query_splice_d3b import pair_index_at_start, pred_index

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/query_attn_patch_d4"
PROTOCOL = ROOT / "design/V010_QUERY_ATTN_PATCH_D4.md"
P4_ADJUDICATION = ROOT / "runs/selection_p4/ADJUDICATION_180001_400.json"
AS_IS_HITS = 74
Q0R_HITS = 127
LIFT = 0.10
NEG = 0.03
STEER_MIN = 0.95
COSINE_ABS = 0.20
COSINE_GAIN = 0.10


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


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def cosine_at(h0, gen: int, query: int) -> float:
    return float(F.cosine_similarity(h0[0, gen], h0[0, query], dim=0).item())


def patch_l0_attention(model, tokens, gen_pos: int, source_pos: int):
    import torch

    attn = model.blocks[0].attention
    original_backend = attn.backend
    original_ref = attn._reference_attention

    def patched(q, k, v):
        time = q.shape[-2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * (attn.head_dim ** -0.5)
        causal = torch.ones((time, time), dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        weights = F.softmax(scores, dim=-1).clone()
        weights[:, :, gen_pos, :] = 0.0
        weights[:, :, gen_pos, source_pos] = 1.0
        return torch.matmul(weights, v)

    attn.backend = "reference"
    attn._reference_attention = patched
    try:
        logits, blocks = capture_blocks(model, tokens)
    finally:
        attn._reference_attention = original_ref
        attn.backend = original_backend
    return logits, blocks


def cosine_writes(a0q_cosine: float, as_cosine: float) -> bool:
    return a0q_cosine >= COSINE_ABS or (a0q_cosine - as_cosine) >= COSINE_GAIN


def decide(
    *,
    full_as_hits: int,
    full_q0r_hits: int,
    steer_on_miss: float,
    n: int,
    g_as: float,
    g_q: float,
    g_c: float,
    g_p: float,
    s_as: float,
    s_c: float,
    c_as: float,
    c_q: float,
) -> str:
    if full_as_hits != AS_IS_HITS or full_q0r_hits != Q0R_HITS or n < 50 or steer_on_miss < STEER_MIN:
        return "INVALID"
    dg_q = g_q - g_as
    dg_c = g_c - g_as
    dg_p = g_p - g_as
    ds_c = s_c - s_as
    writes = cosine_writes(c_q, c_as)
    if dg_p >= LIFT:
        return "PREV_TOKEN"
    if dg_q >= LIFT and writes and ds_c >= LIFT and dg_c <= NEG and dg_p <= NEG:
        return "POINTER_SUFFICIENT"
    if dg_q >= LIFT and (not writes) and dg_p <= NEG:
        return "POINTER_SELECTS"
    if writes and dg_q < LIFT and dg_p <= NEG:
        return "POINTER_WRITES_ONLY"
    if (not writes) and dg_q < LIFT and dg_p <= NEG:
        return "OV_DEAD"
    return "MIXED"


def licenses_next(verdict: str) -> str:
    if verdict in {"POINTER_SUFFICIENT", "POINTER_SELECTS"}:
        return "p5_l0_only_pointer_not_p1_all_layers_not_p4_lambda"
    if verdict == "POINTER_WRITES_ONLY":
        return "d5_l0_mlp_wipe_or_pre_mlp_splice"
    if verdict == "OV_DEAD":
        return "architectural_l0_write_not_matching_loss"
    if verdict == "PREV_TOKEN":
        return "do_not_treat_onehot_attn_as_query_specific"
    return "no_training_license"


def assert_p4_licenses_d4() -> dict:
    adj = json.loads(P4_ADJUDICATION.read_text(encoding="utf-8"))
    if adj.get("protocol") != "V010_SELECTION_REPAIR_P4_EARLY_COPY":
        raise RuntimeError("D4 requires P4 adjudication")
    if adj.get("verdict") != "REGRESSION":
        raise RuntimeError("D4 is licensed only by P4 REGRESSION")
    return adj


def write_manifest() -> dict:
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    p4 = assert_p4_licenses_d4()
    files = [
        PROTOCOL,
        Path(__file__),
        P4_ADJUDICATION,
        S2_DIAGNOSTIC,
        PARENT,
        ROOT / "src/baby_v010/query_splice_d3.py",
        ROOT / "src/baby_v010/query_splice_d3b.py",
        ROOT / "src/baby_v010/model.py",
    ]
    payload = {
        "protocol": "V010_QUERY_ATTN_PATCH_D4",
        "parent_sha256": PARENT_SHA,
        "licensed_by": "V010_SELECTION_REPAIR_P4_EARLY_COPY",
        "p4_verdict": p4["verdict"],
        "p2_launched": False,
        "protected_material_opened": False,
        "files": {posix(path): digest(path) for path in files},
    }
    write(OUT / "MANIFEST.json", payload)
    return payload


def rate(hits: int, n: int) -> float:
    return hits / n if n else float("nan")


def grad_census(model, item, device: str) -> dict:
    import torch

    from .selection_p4 import copy_specs, forward_with_block0, residual_copy_loss
    from .selection_s1 import pack

    model.zero_grad(set_to_none=True)
    model.train()
    model.set_attention_backend("sdpa")
    x, y, mask, _ = pack([item], device)
    hidden, h0 = forward_with_block0(model, x)
    logits = model.language_head(hidden)
    ce = F.cross_entropy(logits[mask], y[mask])
    copy = residual_copy_loss(h0, copy_specs([item]))
    params = [p for p in model.parameters() if p.requires_grad]
    ce_grads = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
    copy_grads = torch.autograd.grad(copy, params, allow_unused=True)

    def nrm(grads) -> float:
        total = 0.0
        for grad in grads:
            if grad is not None:
                total += float(grad.detach().pow(2).sum().item())
        return total ** 0.5

    model.eval()
    return {
        "ce": float(ce.detach().item()),
        "copy": float(copy.detach().item()),
        "ce_grad_l2": nrm(ce_grads),
        "copy_grad_l2": nrm(copy_grads),
        "copy_over_ce_grad": nrm(copy_grads) / nrm(ce_grads) if nrm(ce_grads) else None,
        "n_copy_specs": len(copy_specs([item])),
    }


def run(device: str = "cuda") -> dict:
    import torch

    if (OUT / "QUERY_ATTN.json").exists():
        raise RuntimeError("refusing to overwrite D4 results")
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
    debug_log("D4", "query_attn_patch_d4.py:run", "d4_start", {"n_long": len(long_items)})

    as_hits = 0
    q0r_hits = 0
    a0q_hits = 0
    a0c_hits = 0
    a0c_spliced = 0
    as_spliced = 0
    a0p_hits = 0
    steer_hits = 0
    steer_n = 0
    cos_as: list[float] = []
    cos_q: list[float] = []
    cos_c: list[float] = []
    cos_p: list[float] = []
    with torch.no_grad():
        for index, item in enumerate(long_items):
            sources = splice_sources(item)
            gen = len(item["input"]) - 1
            query = int(sources["query"])
            competitor = int(sources["competitor_key"])
            gold = int(item["query_index"])
            gold_token = int(item["candidate_heads"][gold])
            tokens = torch.tensor([item["input"]], dtype=torch.long, device=device)
            model.set_attention_backend("sdpa")
            logits, blocks = capture_blocks(model, tokens)
            as_pred = pred_index(logits[0, gen], item)
            as_gold = as_pred == gold
            as_hits += int(as_gold)
            as_pair = pair_index_at_start(item, competitor)
            as_spliced += int(as_pair is not None and as_pred == as_pair)
            cos_as.append(cosine_at(blocks[0], gen, query))

            q0r = splice_block(model, tokens, gen, 0, blocks[0][0, query], mix=False)
            q0r_hits += int(pred_index(q0r[0, gen], item) == gold)

            logits_q, blocks_q = patch_l0_attention(model, tokens, gen, query)
            a0q_pred = pred_index(logits_q[0, gen], item)
            a0q_hits += int(a0q_pred == gold)
            cos_q.append(cosine_at(blocks_q[0], gen, query))

            logits_c, blocks_c = patch_l0_attention(model, tokens, gen, competitor)
            a0c_pred = pred_index(logits_c[0, gen], item)
            a0c_hits += int(a0c_pred == gold)
            a0c_spliced += int(as_pair is not None and a0c_pred == as_pair)
            cos_c.append(cosine_at(blocks_c[0], gen, query))

            logits_p, blocks_p = patch_l0_attention(model, tokens, gen, gen - 1)
            a0p_pred = pred_index(logits_p[0, gen], item)
            a0p_hits += int(a0p_pred == gold)
            cos_p.append(cosine_at(blocks_p[0], gen, query))

            if not as_gold:
                steered = logit_steer(model, tokens, gen, gold_token)
                steer_hits += int(pred_index(steered[0, gen], item) == gold)
                steer_n += 1

            if index < 3 or index % 40 == 0:
                debug_log(
                    "D4",
                    "query_attn_patch_d4.py:run",
                    "row",
                    {
                        "index": index,
                        "as_gold": as_gold,
                        "a0q": a0q_pred == gold,
                        "cos_as": cos_as[-1],
                        "cos_q": cos_q[-1],
                    },
                )
            if (index + 1) % 20 == 0:
                print(
                    json.dumps(
                        {
                            "index": index + 1,
                            "as": as_hits,
                            "q0r": q0r_hits,
                            "a0q": a0q_hits,
                            "a0c": a0c_hits,
                            "a0p": a0p_hits,
                        }
                    ),
                    flush=True,
                )

    n = len(long_items)
    steer_rate = rate(steer_hits, steer_n)
    verdict = decide(
        full_as_hits=as_hits,
        full_q0r_hits=q0r_hits,
        steer_on_miss=steer_rate,
        n=n,
        g_as=rate(as_hits, n),
        g_q=rate(a0q_hits, n),
        g_c=rate(a0c_hits, n),
        g_p=rate(a0p_hits, n),
        s_as=rate(as_spliced, n),
        s_c=rate(a0c_spliced, n),
        c_as=median(cos_as),
        c_q=median(cos_q),
    )
    census = grad_census(model, long_items[0], device)
    debug_log("D4", "query_attn_patch_d4.py:run", "verdict", {"verdict": verdict, "census": census})
    result = {
        "protocol": "V010_QUERY_ATTN_PATCH_D4",
        "n": n,
        "as_is_hits": as_hits,
        "q0r_hits": q0r_hits,
        "a0q_hits": a0q_hits,
        "a0c_hits": a0c_hits,
        "a0p_hits": a0p_hits,
        "a0c_spliced": a0c_spliced,
        "as_spliced": as_spliced,
        "steer_on_miss": steer_rate,
        "steer_n": steer_n,
        "median_cosine": {
            "as_is": median(cos_as),
            "a0q": median(cos_q),
            "a0c": median(cos_c),
            "a0p": median(cos_p),
        },
        "delta_gold": {
            "a0q": rate(a0q_hits, n) - rate(as_hits, n),
            "a0c": rate(a0c_hits, n) - rate(as_hits, n),
            "a0p": rate(a0p_hits, n) - rate(as_hits, n),
        },
        "delta_spliced_a0c": rate(a0c_spliced, n) - rate(as_spliced, n),
        "grad_census": census,
        "verdict": verdict,
        "licenses": licenses_next(verdict),
        "p2_launched": False,
        "protected_material_opened": False,
        "authoritative_parent_unchanged": True,
    }
    write(OUT / "QUERY_ATTN.json", result)
    write(OUT / "ADJUDICATION.json", result)
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run"])
    args = parser.parse_args()
    if args.action == "run":
        run()


if __name__ == "__main__":
    main()
