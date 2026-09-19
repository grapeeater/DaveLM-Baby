from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def _set_gen_index(overwrite, index, device: torch.device) -> None:
    """Drive gen-only overwrite. None leaves the write inert (identity)."""
    if overwrite is None:
        return
    if index is None:
        overwrite.gen_index = None
        return
    overwrite.gen_index = torch.as_tensor(index, device=device, dtype=torch.long)


@torch.no_grad()
def score_items(
    model,
    items: list[dict],
    device: torch.device,
    overwrite=None,
    first_answer_only: bool = False,
) -> list[dict]:
    """Score a batch of items with batched teacher forcing and generation.

    ``overwrite`` / ``first_answer_only`` are optional. Defaults preserve the
    historical scorer (no gen-index). When ``first_answer_only`` is true, the
    gen-only overwrite is applied at the first answer-token position and then
    left off for later greedy steps.
    """
    if not items:
        return []
    if first_answer_only and overwrite is None:
        raise ValueError("first_answer_only requires an overwrite module")
    contexts = [[int(x) for x in item["input"]] for item in items]
    targets = [[int(x) for x in item["target"]] for item in items]
    sequences = [context + target[:-1] for context, target in zip(contexts, targets)]
    max_len = max(len(seq) for seq in sequences)
    x = torch.zeros((len(items), max_len), dtype=torch.long, device=device)
    for i, seq in enumerate(sequences):
        x[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    if first_answer_only:
        _set_gen_index(overwrite, [len(context) - 1 for context in contexts], device)
    else:
        _set_gen_index(overwrite, None, device)
    batch_logits = model(x)
    result: list[dict] = []
    first_step_rows = []
    for i, (item, context, target) in enumerate(zip(items, contexts, targets)):
        answer_start = len(context) - 1
        answer_logits = batch_logits[i, answer_start : answer_start + len(target)]
        target_tensor = torch.tensor(target, dtype=torch.long, device=device)
        log_probs = F.log_softmax(answer_logits, dim=-1)
        rows_idx = torch.arange(len(target), device=device)
        target_lp = log_probs[rows_idx, target_tensor]
        ranks = (answer_logits > answer_logits.gather(1, target_tensor[:, None])).sum(dim=1) + 1
        top = answer_logits.argmax(dim=-1)
        competitor = answer_logits.clone()
        competitor[rows_idx, target_tensor] = -float("inf")
        margin = (answer_logits[rows_idx, target_tensor] - competitor.max(dim=-1).values).tolist()
        first_step_rows.append((i, target, target_tensor, answer_logits, ranks, margin))
        result.append({
            "kind": item.get("kind"),
            "surface": item.get("surface"),
            "variant": item.get("variant"),
            "broken": item.get("broken"),
            "low_prior": bool(item.get("low_prior", False)),
            "target_len": len(target),
            "target_prob": float(target_lp[0].exp().item()),
            "target_rank": int(ranks[0].item()),
            "target_logit": float(answer_logits[0, target_tensor[0]].item()),
            "first_margin": float(margin[0]),
            "tf_exact": bool(torch.equal(top, target_tensor)),
            "free_exact": False,
            "first_error": len(target),
            "all_ranks": [int(x) for x in ranks.tolist()],
            "emitted": [],
        })
    # Greedy generation is also batched across items. Right padding cannot
    # affect the causal state at each row's last valid position.
    generated = [context[:] for context in contexts]
    for step_idx in range(max(len(target) for target in targets)):
        active = [i for i, target in enumerate(targets) if step_idx < len(target)]
        if not active:
            break
        max_context = max(min(256, len(generated[i])) for i in active)
        gx = torch.zeros((len(active), max_context), dtype=torch.long, device=device)
        lengths: list[int] = []
        for row, i in enumerate(active):
            current = generated[i][-256:]
            lengths.append(len(current))
            gx[row, : len(current)] = torch.tensor(current, dtype=torch.long, device=device)
        if first_answer_only and step_idx == 0:
            _set_gen_index(overwrite, [length - 1 for length in lengths], device)
        else:
            _set_gen_index(overwrite, None, device)
        z = model(gx)
        for row, i in enumerate(active):
            token = int(z[row, lengths[row] - 1].argmax().item())
            result[i]["emitted"].append(token)
            generated[i].append(token)
    for i, target in enumerate(targets):
        emitted = result[i]["emitted"]
        result[i]["free_exact"] = emitted == target
        result[i]["first_error"] = next((j for j, (a, b) in enumerate(zip(emitted, target)) if a != b), len(target))
    return result


@torch.no_grad()
def score_item(
    model,
    item: dict,
    device: torch.device,
    overwrite=None,
    first_answer_only: bool = False,
) -> dict:
    return score_items(
        model,
        [item],
        device,
        overwrite=overwrite,
        first_answer_only=first_answer_only,
    )[0]


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)
    return {
        "n": len(rows),
        "first_top1": sum(row["target_rank"] == 1 for row in rows) / len(rows),
        "tf_exact": sum(row["tf_exact"] for row in rows) / len(rows),
        "free_exact": sum(row["free_exact"] for row in rows) / len(rows),
        "mean_target_prob": mean("target_prob"),
        "median_target_rank": sorted(row["target_rank"] for row in rows)[len(rows) // 2],
        "mean_target_logit": mean("target_logit"),
        "mean_first_margin": mean("first_margin"),
        "mean_first_error": mean("first_error"),
        "distinct_emitted_tokens": len({token for row in rows for token in row["emitted"]}),
    }


@torch.no_grad()
def language_ce(model, stream: torch.Tensor, starts: list[int], device: torch.device, limit: int = 64) -> float:
    model.eval()
    starts = starts[:limit]
    offsets = torch.arange(256, dtype=torch.long)
    total = 0.0
    count = 0
    for start in starts:
        x = stream[start + offsets].to(device).unsqueeze(0)
        y = stream[start + offsets + 1].to(device).unsqueeze(0)
        z = model(x)
        total += float(F.cross_entropy(z.reshape(-1, z.shape[-1]), y.reshape(-1), reduction="sum").cpu())
        count += y.numel()
    return total / max(1, count)


def evaluate_panels(
    model,
    panels: dict,
    device: torch.device,
    limit: int | None = None,
    overwrite=None,
    first_answer_only: bool = False,
) -> dict:
    model.eval()
    summaries: dict[str, dict] = {}
    rows: dict[str, list[dict]] = {}
    for name, items in panels.items():
        if name == "all_intact":
            continue
        selected = items if limit is None else items[:limit]
        scored = []
        for start in range(0, len(selected), 16):
            scored.extend(
                score_items(
                    model,
                    selected[start : start + 16],
                    device,
                    overwrite=overwrite,
                    first_answer_only=first_answer_only,
                )
            )
        rows[name] = scored
        summaries[name] = summarize(scored)
    if "novel" in summaries and "broken_context" in summaries:
        summaries["context_lift_proxy"] = {
            "target_logit_difference": summaries["novel"]["mean_target_logit"] - summaries["broken_context"]["mean_target_logit"],
            "margin_difference": summaries["novel"]["mean_first_margin"] - summaries["broken_context"]["mean_first_margin"],
            "free_exact_difference": summaries["novel"]["free_exact"] - summaries["broken_context"]["free_exact"],
        }
    return {"summaries": summaries, "rows": rows, "protected_material_opened": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    raise SystemExit("Use train.py for checkpoint loading; evaluate.py is an importable evaluator.")


if __name__ == "__main__":
    main()
