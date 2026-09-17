"""Read-only torch tracing and residual patching for D1.

Does not construct an optimizer. Does not write checkpoints. Default model
``forward`` math is unchanged; this module registers temporary hooks.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

import torch

from .config import BabyVNextConfig
from .model import BabyVNextLM
from .query_presence import (
    N_LAYERS,
    attention_index_masses,
    body_key_positions,
    cosine_l2,
    directed_pairs,
    flip_toward_donor,
    gap_bucket,
    gap_of,
    loc_class,
    mean,
    median,
    skip_identity,
)


def load_parent(path, device: str = "cuda") -> tuple[BabyVNextLM, dict]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("protected_material_opened"):
        raise RuntimeError("refusing a checkpoint that recorded protected access")
    config = BabyVNextConfig.from_dict(blob["config"])
    model = BabyVNextLM(config)
    model.load_state_dict(blob["model_state_dict"])
    model.to(device)
    model.eval()
    model.set_attention_backend("reference")
    return model, blob


def reference_weights(attn, normed: torch.Tensor) -> torch.Tensor:
    q, k, _v = attn._split(normed)
    time = q.shape[-2]
    scores = torch.matmul(q, k.transpose(-2, -1)) * (attn.head_dim ** -0.5)
    causal = torch.ones((time, time), dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    return torch.softmax(scores, dim=-1)


def _pack(rows: list[list[int]], device: str) -> tuple[torch.Tensor, list[int]]:
    lengths = [len(row) for row in rows]
    width = max(lengths)
    tokens = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        tokens[i, : len(row)] = torch.tensor(row, device=device)
    return tokens, lengths


@torch.no_grad()
def sdpa_reference_max_abs(model: BabyVNextLM, rows: list[list[int]], device: str) -> float:
    tokens, lengths = _pack(rows, device)
    model.set_attention_backend("sdpa")
    sdpa = model(tokens)
    model.set_attention_backend("reference")
    ref = model(tokens)
    diffs = []
    for i, length in enumerate(lengths):
        diffs.append((sdpa[i, length - 1] - ref[i, length - 1]).abs().max().item())
    return max(diffs)


@torch.no_grad()
def unembed_rank(model: BabyVNextLM) -> dict:
    weight = model.language_head.weight.detach().float()
    singular = torch.linalg.svdvals(weight.cpu())
    return {
        "shape": list(weight.shape),
        "min": float(singular.min()),
        "median": float(singular.median()),
        "max": float(singular.max()),
        "rank_gt_1e4": int((singular > 1e-4).sum()),
        "d_model": int(weight.shape[1]),
        "injective": int((singular > 1e-4).sum()) == int(weight.shape[1]),
    }


@torch.no_grad()
def trace_items(model: BabyVNextLM, items: list[dict], device: str, batch: int = 4) -> list[dict]:
    model.eval()
    model.set_attention_backend("reference")
    out: list[dict] = []
    for start in range(0, len(items), batch):
        chunk = items[start : start + batch]
        tokens, lengths = _pack([row["input"] for row in chunk], device)
        n_layers = len(model.blocks)
        attn_out = [None] * n_layers
        block_out = [None] * n_layers
        weights = [None] * n_layers
        handles = []

        for layer, block in enumerate(model.blocks):
            def attn_hook(module, inputs, output, layer=layer):
                weights[layer] = reference_weights(module, inputs[0]).detach()
                attn_out[layer] = output.detach()
                return output

            def block_hook(_module, _inputs, output, layer=layer):
                block_out[layer] = output.detach()
                return output

            handles.append(block.attention.register_forward_hook(attn_hook))
            handles.append(block.register_forward_hook(block_hook))
        try:
            logits = model(tokens)
        finally:
            for handle in handles:
                handle.remove()

        hidden = model.forward_hidden(tokens)
        embed = model.token_embedding.weight.detach()
        for i, item in enumerate(chunk):
            gen = lengths[i] - 1
            query_pos = int(item["query_position"])
            body_keys, queried_body = body_key_positions(item)
            heads = [int(h) for h in item["candidate_heads"]]
            z = logits[i, gen].detach()
            final = hidden[i, gen].detach()
            candidate = z[heads]
            argmax = int(torch.argmax(candidate).item())
            layer_attn = []
            layer_block = []
            attn_mass = []
            for layer in range(n_layers):
                layer_attn.append(attn_out[layer][i, gen].detach().cpu())
                layer_block.append(block_out[layer][i, gen].detach().cpu())
                gen_row = weights[layer][i, :, gen, : gen + 1]
                head_rows = []
                for head in range(gen_row.shape[0]):
                    masses = attention_index_masses(
                        gen_row[head].tolist(),
                        gen_pos=gen,
                        query_pos=query_pos,
                        body_keys=body_keys,
                        queried_body=queried_body,
                    )
                    head_rows.append(masses)
                max_query = max(row["query_slot"] for row in head_rows)
                max_prev = max(row["prev"] for row in head_rows)
                attn_mass.append(
                    {
                        "max_query": max_query,
                        "max_prev": max_prev,
                        "any_tracks_query": any(row["tracks_query"] for row in head_rows),
                        "any_tracks_prev": any(row["tracks_prev"] for row in head_rows),
                        "uniform": head_rows[0]["uniform"],
                        "heads": head_rows,
                    }
                )
            qk = int(item["query_key"])
            last = int(item["input"][-1])
            gold_token = heads[int(item["query_index"])]
            out.append(
                {
                    "item": item,
                    "gen_pos": gen,
                    "logits": z.cpu(),
                    "candidate": candidate.cpu(),
                    "candidate_argmax": argmax,
                    "final": final.cpu(),
                    "attn_out": layer_attn,
                    "block_out": layer_block,
                    "attn_mass": attn_mass,
                    "any_query_tracking_head": any(row["any_tracks_query"] for row in attn_mass),
                    "any_prev_tracking_head": any(row["any_tracks_prev"] for row in attn_mass),
                    "embed_cos_query": _cos(final, embed[qk]),
                    "embed_cos_last": _cos(final, embed[last]),
                    "embed_cos_gold_head": _cos(final, embed[gold_token]),
                    "loc": loc_class(item),
                    "gap": gap_of(item),
                    "bucket": gap_bucket(gap_of(item)),
                }
            )
    return out


def _cos(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.flatten().float()
    b = right.flatten().float().to(a.device)
    denom = float(a.norm() * b.norm())
    if denom == 0.0:
        return 0.0
    return float(torch.dot(a, b) / denom)


def _replace_pos(output: torch.Tensor, pos: int, vec: torch.Tensor) -> torch.Tensor:
    cloned = output.clone()
    cloned[:, pos] = vec.to(cloned.dtype)
    return cloned


@torch.no_grad()
def patched_logits(
    model: BabyVNextLM,
    tokens: torch.Tensor,
    gen_pos: int,
    site: str,
    layer: int | None,
    vec: torch.Tensor,
) -> torch.Tensor:
    if site == "final":
        hidden = model.forward_hidden(tokens)
        hidden = hidden.clone()
        hidden[:, gen_pos] = vec.to(hidden.dtype)
        return model.language_head(hidden)

    handles = []
    if site == "attn_out":
        def hook(_module, _inputs, output):
            return _replace_pos(output, gen_pos, vec)

        handles.append(model.blocks[layer].attention.register_forward_hook(hook))
    elif site == "block_out":
        def hook(_module, _inputs, output):
            return _replace_pos(output, gen_pos, vec)

        handles.append(model.blocks[layer].register_forward_hook(hook))
    else:
        raise ValueError(site)
    try:
        return model(tokens)
    finally:
        for handle in handles:
            handle.remove()


def _site_vector(trace: dict, site: str, layer: int | None) -> torch.Tensor:
    if site == "final":
        return trace["final"]
    if site == "attn_out":
        return trace["attn_out"][layer]
    if site == "block_out":
        return trace["block_out"][layer]
    raise ValueError(site)


def _candidate_argmax(logits_row: torch.Tensor, heads: list[int]) -> int:
    values = logits_row[heads]
    return int(torch.argmax(values).item())


@torch.no_grad()
def patch_directed_pairs(
    model: BabyVNextLM,
    traces: list[dict],
    device: str,
    progress: Callable[[str], None] | None = None,
) -> list[dict]:
    model.eval()
    model.set_attention_backend("reference")
    by_id = {(row["item"]["body_id"], int(row["item"]["query_index"])): row for row in traces}
    bodies: dict[str, list[dict]] = defaultdict(list)
    for row in traces:
        bodies[row["item"]["body_id"]].append(row["item"])

    sites: list[tuple[str, int | None]] = [("final", None)]
    for layer in range(N_LAYERS):
        sites.append(("block_out", layer))
        sites.append(("attn_out", layer))

    results = []
    body_ids = sorted(bodies)
    for index, body_id in enumerate(body_ids):
        group = bodies[body_id]
        if progress and index % 8 == 0:
            progress(f"patch body {index}/{len(body_ids)}")
        for donor_item, recipient_item in directed_pairs(group):
            donor = by_id[(body_id, int(donor_item["query_index"]))]
            recipient = by_id[(body_id, int(recipient_item["query_index"]))]
            heads = [int(h) for h in recipient_item["candidate_heads"]]
            donor_gold = int(donor_item["query_index"])
            before = int(recipient["candidate_argmax"])
            gen_pos = int(recipient["gen_pos"])
            tokens = torch.tensor([recipient_item["input"]], dtype=torch.long, device=device)
            pair_sites = []
            for site, layer in sites:
                vec = _site_vector(donor, site, layer)
                rec = _site_vector(recipient, site, layer)
                ident = cosine_l2(vec.tolist(), rec.tolist())
                record = {
                    "site": site,
                    "layer": layer,
                    "residual": ident,
                    "skip_identity": skip_identity(ident["max_abs"]),
                    "before": before,
                    "donor_gold": donor_gold,
                }
                if record["skip_identity"]:
                    record.update(
                        after=before,
                        changed=False,
                        flip_toward_donor=False,
                        match_donor_after=before == donor_gold,
                    )
                else:
                    logits = patched_logits(
                        model,
                        tokens,
                        gen_pos,
                        site,
                        layer,
                        vec.to(device),
                    )
                    after = _candidate_argmax(logits[0, gen_pos], heads)
                    record.update(
                        after=after,
                        changed=after != before,
                        flip_toward_donor=flip_toward_donor(before, after, donor_gold),
                        match_donor_after=after == donor_gold,
                    )
                pair_sites.append(record)
            results.append(
                {
                    "body_id": body_id,
                    "donor": int(donor_item["query_index"]),
                    "recipient": int(recipient_item["query_index"]),
                    "gap": recipient["gap"],
                    "bucket": recipient["bucket"],
                    "loc": recipient["loc"],
                    "K": int(recipient_item["pair_count"]),
                    "sites": pair_sites,
                }
            )
    return results


def twin_identity_rows(traces: list[dict]) -> list[dict]:
    bodies: dict[str, list[dict]] = defaultdict(list)
    for row in traces:
        bodies[row["item"]["body_id"]].append(row)
    rows = []
    for body_id, group in bodies.items():
        items = [row["item"] for row in group]
        lookup = {int(row["item"]["query_index"]): row for row in group}
        for donor_item, recipient_item in directed_pairs(items):
            donor = lookup[int(donor_item["query_index"])]
            recipient = lookup[int(recipient_item["query_index"])]
            full = cosine_l2(donor["logits"].tolist(), recipient["logits"].tolist())
            cand = cosine_l2(donor["candidate"].tolist(), recipient["candidate"].tolist())
            final = cosine_l2(donor["final"].tolist(), recipient["final"].tolist())
            block = [
                cosine_l2(a.tolist(), b.tolist())
                for a, b in zip(donor["block_out"], recipient["block_out"])
            ]
            attn = [
                cosine_l2(a.tolist(), b.tolist())
                for a, b in zip(donor["attn_out"], recipient["attn_out"])
            ]
            rows.append(
                {
                    "body_id": body_id,
                    "donor": int(donor_item["query_index"]),
                    "recipient": int(recipient_item["query_index"]),
                    "gap": recipient["gap"],
                    "bucket": recipient["bucket"],
                    "loc": recipient["loc"],
                    "K": int(recipient_item["pair_count"]),
                    "full_logits": full,
                    "candidate_logits": cand,
                    "final_residual": final,
                    "block_out": block,
                    "attn_out": attn,
                    "same_argmax": donor["candidate_argmax"] == recipient["candidate_argmax"],
                }
            )
    return rows


def _subset(rows: list[dict], pred) -> list[dict]:
    return [row for row in rows if pred(row)]


def _flip_rate(pairs: list[dict], site: str, layer: int | None) -> float:
    hits = 0
    n = 0
    for pair in pairs:
        for record in pair["sites"]:
            if record["site"] == site and record["layer"] == layer:
                hits += int(record["flip_toward_donor"])
                n += 1
    return hits / n if n else float("nan")


def summarize(
    traces: list[dict],
    identity: list[dict],
    patches: list[dict],
) -> dict:
    def id_pred(bucket=None, loc=None, long=False, short=False):
        def pred(row):
            if bucket is not None and row["bucket"] != bucket:
                return False
            if loc is not None and row["loc"] != loc:
                return False
            if long and row["gap"] < 13:
                return False
            if short and row["gap"] > 1:
                return False
            return True

        return pred

    def pack_identity(subset: list[dict]) -> dict:
        return {
            "n_pairs": len(subset),
            "median_full_logit_cosine": median([r["full_logits"]["cosine"] for r in subset]),
            "median_full_logit_l2": median([r["full_logits"]["l2"] for r in subset]),
            "median_full_logit_max_abs": median([r["full_logits"]["max_abs"] for r in subset]),
            "median_candidate_cosine": median([r["candidate_logits"]["cosine"] for r in subset]),
            "median_final_residual_cosine": median([r["final_residual"]["cosine"] for r in subset]),
            "median_final_residual_max_abs": median([r["final_residual"]["max_abs"] for r in subset]),
            "same_argmax_rate": mean([float(r["same_argmax"]) for r in subset]),
            "median_block_cosine": [
                median([r["block_out"][layer]["cosine"] for r in subset]) for layer in range(N_LAYERS)
            ]
            if subset
            else [],
        }

    short_id = _subset(identity, id_pred(short=True))
    long_id = _subset(identity, id_pred(long=True))
    short_p = _subset(patches, id_pred(short=True))
    long_p = _subset(patches, id_pred(long=True))

    def best_block_flip(pairs: list[dict]) -> tuple[int, float]:
        rates = [_flip_rate(pairs, "block_out", layer) for layer in range(N_LAYERS)]
        best = max(range(N_LAYERS), key=lambda layer: rates[layer])
        return best, rates[best]

    best_short_layer, best_short_rate = best_block_flip(short_p) if short_p else (0, float("nan"))
    long_block = [_flip_rate(long_p, "block_out", layer) for layer in range(N_LAYERS)]
    long_attn = [_flip_rate(long_p, "attn_out", layer) for layer in range(N_LAYERS)]

    def attn_summary(pred) -> dict:
        rows = [row for row in traces if pred(row)]
        if not rows:
            return {"n": 0}
        return {
            "n": len(rows),
            "fraction_any_query_tracking_head": mean(
                [float(row["any_query_tracking_head"]) for row in rows]
            ),
            "fraction_any_prev_tracking_head": mean(
                [float(row["any_prev_tracking_head"]) for row in rows]
            ),
            "median_max_query_mass_over_layers": median(
                [max(layer["max_query"] for layer in row["attn_mass"]) for row in rows]
            ),
            "median_max_prev_mass_over_layers": median(
                [max(layer["max_prev"] for layer in row["attn_mass"]) for row in rows]
            ),
            "median_embed_cos_query": median([row["embed_cos_query"] for row in rows]),
            "median_embed_cos_last": median([row["embed_cos_last"] for row in rows]),
        }

    return {
        "n_items": len(traces),
        "n_identity_pairs": len(identity),
        "n_patch_pairs": len(patches),
        "identity": {
            "all": pack_identity(identity),
            "g0_1": pack_identity(_subset(identity, id_pred(short=True))),
            "g31p": pack_identity(_subset(identity, lambda r: r["bucket"] == "g31p")),
            "long": pack_identity(long_id),
            "last_is_query": pack_identity(_subset(identity, id_pred(loc="last_is_query"))),
            "prev_is_query": pack_identity(_subset(identity, id_pred(loc="prev_is_query"))),
            "query_deeper": pack_identity(_subset(identity, id_pred(loc="query_deeper"))),
        },
        "attention": {
            "g0_1": attn_summary(lambda r: r["gap"] <= 1),
            "prev_is_query": attn_summary(lambda r: r["loc"] == "prev_is_query"),
            "query_deeper": attn_summary(lambda r: r["loc"] == "query_deeper"),
            "long": attn_summary(lambda r: r["gap"] >= 13),
            "g31p": attn_summary(lambda r: r["bucket"] == "g31p"),
        },
        "positive_control": {
            "n_pairs": len(short_p),
            "median_full_logit_cosine": pack_identity(short_id)["median_full_logit_cosine"],
            "flip_toward_donor_final": _flip_rate(short_p, "final", None),
            "flip_toward_donor_best_block": best_short_rate,
            "best_block_layer": best_short_layer,
            "flip_toward_donor_block": [
                _flip_rate(short_p, "block_out", layer) for layer in range(N_LAYERS)
            ],
            "flip_toward_donor_attn": [
                _flip_rate(short_p, "attn_out", layer) for layer in range(N_LAYERS)
            ],
            "instrument_prev_or_query_track_rate": attn_summary(lambda r: r["gap"] <= 1)[
                "fraction_any_prev_tracking_head"
            ]
            if traces
            else float("nan"),
        },
        "long_gap": {
            "n_pairs": len(long_p),
            "median_full_logit_cosine": pack_identity(long_id)["median_full_logit_cosine"],
            "median_final_residual_cosine": pack_identity(long_id)["median_final_residual_cosine"],
            "flip_toward_donor_final": _flip_rate(long_p, "final", None),
            "flip_toward_donor_block": long_block,
            "flip_toward_donor_attn": long_attn,
            "fraction_rows_any_query_tracking_head": attn_summary(lambda r: r["gap"] >= 13)[
                "fraction_any_query_tracking_head"
            ],
        },
    }
