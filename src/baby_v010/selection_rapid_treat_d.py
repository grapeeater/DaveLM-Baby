"""Eval-only D-series treatments downstream of frozen C2.

C2 locator is not retuned. A1 routing and B1 flags stay reproducible.
Production never reads gold query_position or gold target. U16000 is unchanged.
"""
from __future__ import annotations

import argparse
import inspect
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .query_splice_d3 import capture_blocks, eligible_item, score_item as splice_score, splice_block, splice_sources
from .residual_overwrite import attach_overwrite
from .selection_rapid_treat import (
    INDUCTION_DROP_BAR,
    KEYED_MIN_TOP1,
    OFF_INDUCTION_TOP1,
    OFF_LONG_FREE,
    OUT,
    POLICY_A1,
    _long_gap_items,
    ledger_append,
    measure_stage1,
    should_set_gen_index,
)
from .selection_rapid_treat_b import (
    WRITE_HARD,
    score_query_swap,
    run_stage2_treated,
    run_stage3_treated,
)
from .selection_rapid_treat_c import (
    EXPECTED_U16000,
    LOCATOR_TOKEN,
    LocatorOverwrite,
    LocatorRuntime,
    MIN_CONTENT_TOKEN,
    load_locator,
    token_identity_src,
    verify_locked_hashes,
)
from .selection_s1 import digest, write

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_C2_SHA = "06e2e6139a134ef84edd0924184f3efb79cf2a8c93c4fedf500617d154cf2dd2"
C2_LONG = 122
C2_FIRST = 126
C2_QSWAP = 2.0 / 3.0
ADVANCE_LONG = 135
GRAD_LONG = 140
Q0R_PROBE_N = 20
MASK_OFF = "off"
MASK_INVENTORY = "inventory"
MASK_MATCHED = "matched"
WRITE_QUERY = "query"
WRITE_VALUE = "value"
LATER_BLOCK1 = "block1"
LATER_FINAL = "final"


def tiling_parse(tokens: list[int], gen: int) -> dict | None:
    """Same (key, value, SEP) tiling as C2. No gold labels.

    Returns the unique tiling C2 would use, plus record starts / value heads.
    """
    body = [int(t) for t in tokens[:gen]]
    positions: dict[int, list[int]] = {}
    for index, token in enumerate(body):
        if token >= MIN_CONTENT_TOKEN:
            positions.setdefault(token, []).append(index)
    candidates: list[tuple] = []
    for sep, spots in positions.items():
        if len(spots) < 2:
            continue
        for i in range(len(spots) - 1):
            delta = spots[i + 1] - spots[i]
            if delta < 3:
                continue
            run = [spots[i], spots[i + 1]]
            expect = spots[i + 1] + delta
            k = i + 2
            while k < len(spots) and spots[k] == expect:
                run.append(spots[k])
                expect += delta
                k += 1
            if len(run) < 2:
                continue
            starts = [pos - (delta - 1) for pos in run]
            if any(start < 0 for start in starts):
                continue
            start_set = set(starts)
            queries: list[int] = []
            for start in starts:
                key = body[start]
                for occ in positions.get(key, []):
                    if occ not in start_set:
                        queries.append(occ)
            uniq = sorted(set(queries))
            if len(uniq) != 1:
                continue
            keys = [body[start] for start in starts]
            value_heads = [body[start + 1] for start in starts]
            value_pos = [start + 1 for start in starts]
            candidates.append((len(run), delta, uniq[0], starts, keys, value_heads, value_pos, sep))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    run_n, delta, query_src, starts, keys, value_heads, value_pos, sep = candidates[0]
    matched_head = None
    matched_pos = None
    query_key = body[query_src]
    hits = [
        (head, pos)
        for key, head, pos in zip(keys, value_heads, value_pos)
        if key == query_key
    ]
    if len(hits) == 1:
        matched_head, matched_pos = hits[0]
    return {
        "query_src": int(query_src),
        "sep": int(sep),
        "delta": int(delta),
        "k": int(run_n),
        "starts": [int(x) for x in starts],
        "keys": [int(x) for x in keys],
        "value_heads": [int(x) for x in value_heads],
        "value_pos": [int(x) for x in value_pos],
        "matched_head": None if matched_head is None else int(matched_head),
        "matched_pos": None if matched_pos is None else int(matched_pos),
    }


def tiling_value_heads(tokens: list[int], gen: int) -> list[int]:
    parsed = tiling_parse(tokens, gen)
    if parsed is None:
        return []
    heads = []
    seen = set()
    for token in parsed["value_heads"]:
        if token not in seen:
            seen.add(token)
            heads.append(token)
    return heads


def tiling_matched_head(tokens: list[int], gen: int) -> int | None:
    parsed = tiling_parse(tokens, gen)
    if parsed is None:
        return None
    return parsed["matched_head"]


def tiling_matched_pos(tokens: list[int], gen: int) -> int | None:
    parsed = tiling_parse(tokens, gen)
    if parsed is None:
        return None
    return parsed["matched_pos"]


def apply_head_mask(logits_row: torch.Tensor, heads: list[int]) -> torch.Tensor:
    if not heads:
        return logits_row
    masked = logits_row.clone()
    keep = torch.zeros_like(masked, dtype=torch.bool)
    index = [h for h in heads if 0 <= int(h) < masked.numel()]
    if not index:
        return logits_row
    keep[index] = True
    masked = masked.masked_fill(~keep, float("-inf"))
    return masked


def mask_mode_heads(tokens: list[int], gen: int, mode: str) -> list[int]:
    if mode == MASK_INVENTORY:
        return tiling_value_heads(tokens, gen)
    if mode == MASK_MATCHED:
        matched = tiling_matched_head(tokens, gen)
        return [] if matched is None else [matched]
    return []


def value_src_batch(tokens: torch.Tensor, gen_index: torch.Tensor) -> torch.Tensor:
    batch, _time = tokens.shape
    out = torch.full((batch,), -1, device=tokens.device, dtype=torch.long)
    for row in range(batch):
        gen = int(gen_index[row].item())
        if gen < 0:
            continue
        pos = tiling_matched_pos(tokens[row].tolist(), gen)
        if pos is not None:
            out[row] = int(pos)
    return out


class DownstreamSession:
    """Compose frozen C2 write with eval-only readout / transfer ops."""

    def __init__(self, model, overwrite: LocatorOverwrite) -> None:
        self.model = model
        self.overwrite = overwrite
        self.mask_mode = MASK_OFF
        self.write_src = WRITE_QUERY
        self.later_site: str | None = None
        self._model_forward = None
        self._ow_forward = None
        self._later_handle = None
        self._pending_value_src: torch.Tensor | None = None

    def configure(self, *, mask_mode: str = MASK_OFF, write_src: str = WRITE_QUERY, later_site: str | None = None) -> None:
        if mask_mode not in {MASK_OFF, MASK_INVENTORY, MASK_MATCHED}:
            raise ValueError(f"unknown mask_mode {mask_mode}")
        if write_src not in {WRITE_QUERY, WRITE_VALUE}:
            raise ValueError(f"unknown write_src {write_src}")
        if later_site not in {None, LATER_BLOCK1, LATER_FINAL}:
            raise ValueError(f"unknown later_site {later_site}")
        self.mask_mode = mask_mode
        self.write_src = write_src
        self.later_site = later_site
        self._rebind_later()

    def install(self) -> "DownstreamSession":
        if self._model_forward is not None:
            return self
        ow = self.overwrite
        self._ow_forward = ow.forward
        self._model_forward = self.model.forward
        session = self

        def ow_forward(hidden: torch.Tensor) -> torch.Tensor:
            saved = ow.gen_index
            pending = session._pending_value_src
            if pending is not None:
                ow.forced_src = pending
            out = session._ow_forward(hidden)
            ow.last_gen_index = saved
            if saved is not None and ow.last_src_index is not None:
                valid = saved >= 0
                if bool(valid.any()):
                    b = torch.arange(hidden.size(0), device=hidden.device)
                    ow.last_src_vec = out[b, saved.clamp(0, hidden.size(1) - 1)].detach()
                else:
                    ow.last_src_vec = None
            else:
                ow.last_src_vec = None
            return out

        def model_forward(tokens, *args, **kwargs):
            saved = ow.gen_index
            session._pending_value_src = None
            if (
                session.write_src == WRITE_VALUE
                and saved is not None
                and bool((saved >= 0).any().item())
            ):
                session._pending_value_src = value_src_batch(tokens, saved)
            logits = session._model_forward(tokens, *args, **kwargs)
            session._pending_value_src = None
            if session.mask_mode == MASK_OFF or saved is None:
                return logits
            return session._mask_logits(logits, tokens, saved)

        ow.forward = ow_forward
        self.model.forward = model_forward
        self._rebind_later()
        return self

    def _mask_logits(self, logits: torch.Tensor, tokens: torch.Tensor, gen_index: torch.Tensor) -> torch.Tensor:
        batch = int(logits.shape[0])
        time = int(logits.shape[1])
        out = logits
        cloned = False
        for row in range(batch):
            gen = int(gen_index[row].item())
            if gen < 0 or gen >= time:
                continue
            window = tokens[row].tolist()
            heads = mask_mode_heads(window, gen, self.mask_mode)
            if not heads:
                continue
            if not cloned:
                out = logits.clone()
                cloned = True
            out[row, gen] = apply_head_mask(out[row, gen], heads)
        return out

    def _copy_later(self, output: torch.Tensor) -> torch.Tensor:
        ow = self.overwrite
        gen = getattr(ow, "last_gen_index", None)
        src_vec = getattr(ow, "last_src_vec", None)
        if gen is None or src_vec is None:
            return output
        valid = gen >= 0
        if not bool(valid.any()):
            return output
        cloned = output.clone()
        b = torch.arange(output.size(0), device=output.device)
        clamped = gen.clamp(0, output.size(1) - 1)
        cloned[b[valid], clamped[valid]] = src_vec[valid].to(cloned.dtype)
        return cloned

    def _rebind_later(self) -> None:
        if self._later_handle is not None:
            self._later_handle.remove()
            self._later_handle = None
        if self._model_forward is None:
            return
        if self.later_site == LATER_BLOCK1:
            self._later_handle = self.model.blocks[1].register_forward_hook(
                lambda _m, _i, output: self._copy_later(output)
            )
        elif self.later_site == LATER_FINAL:
            self._later_handle = self.model.final_norm.register_forward_hook(
                lambda _m, _i, output: self._copy_later(output)
            )

    def remove(self) -> None:
        if self._later_handle is not None:
            self._later_handle.remove()
            self._later_handle = None
        if self._model_forward is not None:
            self.model.forward = self._model_forward
            self._model_forward = None
        if self._ow_forward is not None:
            self.overwrite.forward = self._ow_forward
            self._ow_forward = None
        self._pending_value_src = None
        self.overwrite.gen_index = None
        self.overwrite.forced_src = None


def operator_tag(mask_mode: str, write_src: str, later_site: str | None) -> str:
    if mask_mode == MASK_INVENTORY and write_src == WRITE_QUERY and later_site is None:
        return "d1_inv_mask"
    if mask_mode == MASK_OFF and write_src == WRITE_QUERY and later_site == LATER_FINAL:
        return "d2_later_final"
    if mask_mode == MASK_OFF and write_src == WRITE_QUERY and later_site == LATER_BLOCK1:
        return "d2_later_b1"
    if mask_mode == MASK_MATCHED and write_src == WRITE_QUERY and later_site is None:
        return "d3_match_mask"
    if mask_mode == MASK_OFF and write_src == WRITE_VALUE and later_site is None:
        return "d4_value_write"
    if mask_mode == MASK_MATCHED and later_site == LATER_FINAL:
        return "d3_plus_later_final"
    parts = [f"mask_{mask_mode}", f"src_{write_src}"]
    if later_site:
        parts.append(f"later_{later_site}")
    return "d_" + "_".join(parts)


def _change(mask_mode: str, write_src: str, later_site: str | None) -> str:
    base = "A1+C2 frozen locator; "
    if mask_mode == MASK_INVENTORY:
        return base + "D1 first-token mask to tiling value heads (not gold target)"
    if mask_mode == MASK_MATCHED:
        extra = "" if later_site is None else f"; also copy L0 src into {later_site}"
        return base + "D3 first-token mask to key-matched tiling value head (not gold)" + extra
    if write_src == WRITE_VALUE:
        return base + "D4 write h0[gen]:=h0[key-matched value site] (tiling, not gold)"
    if later_site == LATER_FINAL:
        return base + "D2 also replace final-norm gen with L0 C2 src vector"
    if later_site == LATER_BLOCK1:
        return base + "D2 also replace block-1 gen with L0 C2 src vector"
    return base + f"mask={mask_mode} write={write_src} later={later_site}"


def d_canary_verdict(report: dict) -> tuple[str, str]:
    long_on = int(report["long_gap"]["free_exact"])
    n_long = int(report["long_gap"]["n"])
    first = int(report["long_gap"].get("first_correct") or 0)
    ind = float(report["primitive_induction"]["first_top1"])
    keyed = float(report["primitive_keyed"]["first_top1"])
    swap = report.get("query_swap_same_surface_novel") or {}
    swap_top1 = swap.get("first_top1")
    bind_up = swap_top1 is not None and float(swap_top1) >= C2_QSWAP + 0.08 - 1e-12
    first_up = first >= C2_FIRST + 10
    if ind + 1e-12 < (OFF_INDUCTION_TOP1 - INDUCTION_DROP_BAR):
        return "KILL", "induction regression vs A1 exemption"
    if keyed + 1e-12 < KEYED_MIN_TOP1:
        return "KILL", "primitive_keyed negative control failed"
    if long_on <= OFF_LONG_FREE + 4:
        return "KILL", "long-gap collapsed toward OFF 71"
    if long_on >= GRAD_LONG:
        return "GRAD", f"long-gap {long_on}/{n_long} phase-graduation strength vs C2 {C2_LONG}"
    if long_on >= ADVANCE_LONG:
        return "ADVANCE", f"long-gap {long_on}/{n_long} clear E2E jump vs C2 {C2_LONG}"
    if long_on > C2_LONG and bind_up:
        return "ADVANCE", f"long-gap {long_on}/{n_long} up and query-swap bind clearly up"
    if long_on > C2_LONG and first_up:
        return "ADVANCE", f"long-gap {long_on}/{n_long} with first_correct {first} vs C2 {C2_FIRST}"
    if long_on <= C2_LONG and not bind_up and not first_up:
        return "KILL", f"long-gap {long_on}/{n_long} no bind gain vs C2 {C2_LONG}/{C2_FIRST}"
    return "KILL", f"long-gap {long_on}/{n_long} first={first} did not move the C2 bottleneck"


def _strip(report: dict) -> dict:
    return {k: v for k, v in report.items() if k not in {"long_gap_rows", "pointer_rows", "diag_rows", "rows"}}


def verify_c2_frozen() -> dict:
    hashes = verify_locked_hashes()
    c2 = digest(ROOT / "src/baby_v010/selection_rapid_treat_c.py")
    if c2 != EXPECTED_C2_SHA:
        raise RuntimeError(f"C2 sha mismatch: {c2}")
    hashes["c2_sha256"] = c2
    hashes["c2_ok"] = True
    return hashes


def production_d_uses_query_position() -> bool:
    source = (
        inspect.getsource(tiling_parse)
        + inspect.getsource(mask_mode_heads)
        + inspect.getsource(value_src_batch)
        + inspect.getsource(apply_head_mask)
        + inspect.getsource(DownstreamSession._mask_logits)
        + inspect.getsource(DownstreamSession._copy_later)
    )
    return "query_position" in source or "candidate_heads" in source


@torch.no_grad()
def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().unsqueeze(0), b.float().unsqueeze(0)).item())


@torch.no_grad()
def diagnose_c2_downstream(model, overwrite: LocatorOverwrite, device, policy: str = POLICY_A1) -> dict:
    """One GPU pass over 215 long-gap rows: first_correct, rank, inventory, transfer, Q0R sample."""
    items = _long_gap_items()
    rows = []
    t0 = time.time()
    for item in items:
        generated = [int(t) for t in item["input"]]
        window = generated[-256:]
        gen = len(window) - 1
        activate = bool(should_set_gen_index(item, policy))
        parsed = tiling_parse(window, gen)
        gold = int(item["target"][0])
        target = [int(t) for t in item["target"]]
        heads = list(item.get("candidate_heads") or [])
        tiling_heads = parsed["value_heads"] if parsed else []
        matched = parsed["matched_head"] if parsed else None
        l0_holder: dict = {}

        def _ow_capture(hidden, orig=overwrite.forward, holder=l0_holder):
            saved = overwrite.gen_index
            out = orig(hidden)
            if saved is not None and overwrite.last_src_index is not None:
                gen_i = int(saved[0].item())
                src_i = int(overwrite.last_src_index[0].item())
                holder["src_i"] = src_i
                holder["src_vec"] = hidden[0, src_i].detach()
                holder["gen_vec"] = out[0, gen_i].detach()
                holder["cos_l0"] = _cosine(out[0, gen_i], hidden[0, src_i])
            return out

        prev_ow = overwrite.forward
        overwrite.forward = _ow_capture
        final_holder: dict = {}
        handle = model.final_norm.register_forward_hook(
            lambda _m, _i, output, holder=final_holder: holder.update(h=output.detach())
        )
        try:
            overwrite.gen_index = (
                torch.tensor([gen], device=device, dtype=torch.long) if activate else None
            )
            tokens = torch.tensor([window], dtype=torch.long, device=device)
            logits = model(tokens)
            row_logits = logits[0, gen]
        finally:
            handle.remove()
            overwrite.forward = prev_ow
            overwrite.gen_index = None

        pred = int(row_logits.argmax().item())
        gold_logit = float(row_logits[gold].item())
        gold_rank = int((row_logits > row_logits[gold]).sum().item()) + 1
        inv_set = [int(h) for h in tiling_heads] or [int(h) for h in heads]
        gold_in_inv = gold in set(inv_set)
        pred_in_inv = pred in set(inv_set)
        if inv_set:
            inv_pred = int(inv_set[int(row_logits[inv_set].argmax().item())])
            if gold_in_inv:
                gold_inv_rank = int((row_logits[inv_set] > row_logits[gold]).sum().item()) + 1
            else:
                gold_inv_rank = None
        else:
            inv_pred = pred
            gold_inv_rank = None
        matched_ok = matched is not None and matched == gold
        inv_argmax_ok = inv_pred == gold

        emitted = [pred]
        generated_run = list(window) + [pred]
        for _step in range(1, len(target)):
            overwrite.gen_index = None
            step_tokens = torch.tensor([generated_run[-256:]], dtype=torch.long, device=device)
            step_logits = model(step_tokens)
            token = int(step_logits[0, -1].argmax().item())
            emitted.append(token)
            generated_run.append(token)
        overwrite.gen_index = None
        first_error = next((j for j, (a, b) in enumerate(zip(emitted, target)) if a != b), len(target))
        free_exact = emitted == target
        first_correct = emitted[0] == gold if emitted else False

        cos_final = None
        if "src_vec" in l0_holder and "h" in final_holder:
            cos_final = _cosine(final_holder["h"][0, gen], l0_holder["src_vec"])

        rows.append(
            {
                "body_id": item.get("body_id"),
                "K": int(item["pair_count"]),
                "value_length": int(item.get("value_length") or len(item.get("target_span") or [])),
                "target_len": len(target),
                "armed": activate,
                "c2_src": l0_holder.get("src_i"),
                "tiling_src": None if parsed is None else parsed["query_src"],
                "forced_applied": l0_holder.get("src_i") is not None,
                "free_exact": free_exact,
                "first_correct": first_correct,
                "first_error": first_error,
                "pred": pred,
                "gold": gold,
                "gold_rank": gold_rank,
                "gold_logit": gold_logit,
                "gold_in_inventory": gold_in_inv,
                "pred_in_inventory": pred_in_inv,
                "gold_inv_rank": gold_inv_rank,
                "inv_argmax": inv_pred,
                "inv_argmax_correct": inv_argmax_ok,
                "matched_head": matched,
                "matched_correct": matched_ok,
                "cos_l0": l0_holder.get("cos_l0"),
                "cos_final": cos_final,
                "n_tiling_heads": len(tiling_heads),
            }
        )

    n = len(rows)
    hits = [row for row in rows if row["free_exact"]]
    misses = [row for row in rows if not row["free_exact"]]
    first_ok = [row for row in rows if row["first_correct"]]
    cont_miss = [row for row in misses if row["first_correct"]]
    first_miss = [row for row in misses if not row["first_correct"]]
    readout_wrong_inv = [
        row for row in first_miss if row["gold_in_inventory"] and row["pred_in_inventory"] and not row["inv_argmax_correct"]
    ]
    readout_d1_rescue = [
        row for row in first_miss if row["gold_in_inventory"] and row["inv_argmax_correct"]
    ]
    gold_not_cand = [row for row in first_miss if not row["gold_in_inventory"]]
    oos_pred = [row for row in first_miss if not row["pred_in_inventory"]]

    def _rate(subset, field):
        if not subset:
            return 0.0
        return sum(int(bool(row[field])) for row in subset) / len(subset)

    def _median(subset, field):
        vals = [row[field] for row in subset if row.get(field) is not None]
        return None if not vals else statistics.median(vals)

    bucket_counts = {
        "continuation": len(cont_miss),
        "readout_d1_rescue": len(readout_d1_rescue),
        "readout_wrong_inventory": len(readout_wrong_inv),
        "gold_not_in_candidates": len(gold_not_cand),
        "pred_outside_inventory": len(oos_pred),
    }
    dominant = max(bucket_counts, key=bucket_counts.get) if misses else "none"
    if bucket_counts["continuation"] >= max(1, int(0.5 * len(misses))):
        dominant = "continuation"
    elif bucket_counts["readout_d1_rescue"] >= max(10, bucket_counts["readout_wrong_inventory"]):
        dominant = "readout_d1_rescue"
    elif bucket_counts["readout_wrong_inventory"] >= bucket_counts["readout_d1_rescue"]:
        dominant = "readout_wrong_inventory"

    q0r = probe_q0r_on_misses(model, overwrite, items, misses[:Q0R_PROBE_N], device)

    k4_miss = sum(int(row["K"] == 4) for row in misses)
    long_val_miss = sum(int(row["value_length"] >= 6) for row in misses)
    error_pos = {}
    for row in misses:
        error_pos[str(row["first_error"])] = error_pos.get(str(row["first_error"]), 0) + 1

    summary = {
        "id": "d_miss_split",
        "policy": policy,
        "change": "C2 ON miss-row split: first_correct/rank/inventory/transfer/Q0R sample",
        "n": n,
        "free_exact": sum(int(row["free_exact"]) for row in rows),
        "first_correct": sum(int(row["first_correct"]) for row in rows),
        "n_miss": len(misses),
        "n_hit": len(hits),
        "n_first_ok": len(first_ok),
        "inv_argmax_correct": sum(int(row["inv_argmax_correct"]) for row in rows),
        "matched_correct": sum(int(row["matched_correct"]) for row in rows),
        "forced_applied": sum(int(row["forced_applied"]) for row in rows),
        "bucket_counts": bucket_counts,
        "dominant": dominant,
        "miss_first_correct_rate": _rate(misses, "first_correct"),
        "miss_pred_in_inventory": _rate(misses, "pred_in_inventory"),
        "miss_gold_in_inventory": _rate(misses, "gold_in_inventory"),
        "miss_inv_argmax_correct": _rate(misses, "inv_argmax_correct"),
        "miss_matched_correct": _rate(misses, "matched_correct"),
        "miss_median_gold_rank": _median(misses, "gold_rank"),
        "miss_median_gold_inv_rank": _median(misses, "gold_inv_rank"),
        "hit_median_gold_rank": _median(hits, "gold_rank"),
        "miss_median_cos_l0": _median(misses, "cos_l0"),
        "miss_median_cos_final": _median(misses, "cos_final"),
        "hit_median_cos_l0": _median(hits, "cos_l0"),
        "hit_median_cos_final": _median(hits, "cos_final"),
        "k4_miss": k4_miss,
        "k4_n": sum(int(row["K"] == 4) for row in rows),
        "long_value_miss": long_val_miss,
        "first_error_hist": error_pos,
        "q0r_probe": q0r,
        "verdict": "DIAG",
        "lesson": "",
        "elapsed_s": time.time() - t0,
        "protected_material_opened": False,
        "promoted": False,
        "test_opened": False,
        "used_gold_query_position_as_source": False,
        "rows": rows,
    }
    transfer_dead = (
        summary["miss_median_cos_l0"] is not None
        and summary["miss_median_cos_final"] is not None
        and summary["miss_median_cos_l0"] >= 0.9
        and summary["miss_median_cos_final"] < 0.35
    )
    if dominant == "continuation":
        bottleneck = "decode/continuation"
    elif transfer_dead:
        bottleneck = "write/transfer"
    else:
        bottleneck = "readout/selection"
    q0r_same = None
    if q0r.get("n"):
        q0r_same = q0r.get("q0r_also_miss") == q0r.get("n") or (
            q0r.get("n") and q0r.get("q0r_also_miss", 0) / q0r["n"] >= 0.75
        )
        if q0r_same:
            bottleneck = "readout/selection"
    summary["bottleneck"] = bottleneck
    summary["transfer_dead"] = transfer_dead
    summary["lesson"] = (
        f"bottleneck={bottleneck} dominant={dominant} miss={len(misses)} "
        f"first_correct={summary['first_correct']}/{n} free={summary['free_exact']}/{n} "
        f"inv_argmax={summary['inv_argmax_correct']} matched={summary['matched_correct']} "
        f"cos_l0={summary['miss_median_cos_l0']} cos_final={summary['miss_median_cos_final']} "
        f"q0r_also_miss={q0r.get('q0r_also_miss')}/{q0r.get('n')}"
    )
    return summary


@torch.no_grad()
def probe_q0r_on_misses(model, overwrite, items: list[dict], miss_rows: list[dict], device) -> dict:
    """Existing Q0R splice on a C2-miss sample. Gold query_position is diagnostic-only."""
    by_id = {item.get("body_id"): item for item in items}
    hits = 0
    also_miss = 0
    n = 0
    skipped = 0
    saved_index = overwrite.gen_index
    overwrite.gen_index = None
    try:
        for row in miss_rows:
            item = by_id.get(row.get("body_id"))
            if item is None or not eligible_item(item):
                skipped += 1
                continue
            sources = splice_sources(item)
            gen = len(item["input"]) - 1
            tokens = torch.tensor([item["input"]], dtype=torch.long, device=device)
            _logits, blocks = capture_blocks(model, tokens)
            q_out = splice_block(model, tokens, gen, 0, blocks[0][0, int(sources["query"])], False)
            scored = splice_score(q_out[0, gen], item, sources)
            n += 1
            if scored["gold"]:
                hits += 1
            else:
                also_miss += 1
    finally:
        overwrite.gen_index = saved_index
    return {
        "n": n,
        "skipped": skipped,
        "q0r_inventory_hits": hits,
        "q0r_also_miss": also_miss,
        "used_gold_query_position": True,
        "note": "diagnostic-only Q0R; not a production source",
    }


def name_first_treatments(diag: dict) -> list[dict]:
    bottleneck = diag.get("bottleneck")
    queue: list[dict] = []
    inv_all = int(diag.get("inv_argmax_correct") or 0)
    matched = int(diag.get("matched_correct") or 0)
    if bottleneck == "decode/continuation":
        queue.append({"mask_mode": MASK_OFF, "write_src": WRITE_QUERY, "later_site": None, "note": "continuation: C2 first-step already; skip every-step"})
        return queue
    if bottleneck == "write/transfer" or diag.get("transfer_dead"):
        queue.append({"mask_mode": MASK_OFF, "write_src": WRITE_QUERY, "later_site": LATER_FINAL})
        queue.append({"mask_mode": MASK_OFF, "write_src": WRITE_QUERY, "later_site": LATER_BLOCK1})
    if inv_all >= C2_FIRST + 10:
        queue.append({"mask_mode": MASK_INVENTORY, "write_src": WRITE_QUERY, "later_site": None})
    else:
        queue.append(
            {
                "mask_mode": MASK_INVENTORY,
                "write_src": WRITE_QUERY,
                "later_site": None,
                "preview_kill": inv_all <= C2_LONG + 5,
            }
        )
    if matched >= ADVANCE_LONG:
        queue.append({"mask_mode": MASK_MATCHED, "write_src": WRITE_QUERY, "later_site": None})
        queue.append({"mask_mode": MASK_OFF, "write_src": WRITE_VALUE, "later_site": None})
    elif bottleneck == "readout/selection":
        queue.append({"mask_mode": MASK_MATCHED, "write_src": WRITE_QUERY, "later_site": None})
        queue.append({"mask_mode": MASK_OFF, "write_src": WRITE_VALUE, "later_site": None})
    # unique by tag
    seen = set()
    out = []
    for spec in queue:
        tag = operator_tag(spec["mask_mode"], spec["write_src"], spec.get("later_site"))
        if tag in seen:
            continue
        seen.add(tag)
        out.append(spec)
    return out


def run_d_candidate(
    model,
    overwrite: LocatorOverwrite,
    session: DownstreamSession,
    device,
    *,
    mask_mode: str,
    write_src: str = WRITE_QUERY,
    later_site: str | None = None,
    policy: str = POLICY_A1,
    score_swap: bool = True,
    id_override: str | None = None,
) -> dict:
    session.configure(mask_mode=mask_mode, write_src=write_src, later_site=later_site)
    overwrite.set_eval_write(WRITE_HARD, locator=LOCATOR_TOKEN)
    tag = id_override or operator_tag(mask_mode, write_src, later_site)
    t0 = time.time()
    print(json.dumps({"phase": "d_start", "id": tag, "mask": mask_mode, "write_src": write_src, "later": later_site}), flush=True)
    stage1 = measure_stage1(model, overwrite, device, policy)
    swap = score_query_swap(model, overwrite, device, policy) if score_swap else None
    report = dict(stage1)
    report["id"] = tag
    report["operator"] = tag
    report["write_mode"] = WRITE_HARD
    report["locator"] = LOCATOR_TOKEN
    report["mask_mode"] = mask_mode
    report["write_src"] = write_src
    report["later_site"] = later_site
    report["change"] = _change(mask_mode, write_src, later_site)
    report["query_swap_same_surface_novel"] = swap
    report["c2_long_gap_free_exact"] = C2_LONG
    report["used_gold_query_position_as_source"] = False
    verdict, lesson = d_canary_verdict(report)
    report["verdict"] = verdict
    report["lesson"] = lesson
    report["elapsed_s"] = time.time() - t0
    write(OUT / f"{tag}_stage1.json", _strip(report))
    ledger_append(
        {
            **report,
            "lesson": (
                f"{lesson}; first_correct={report['long_gap'].get('first_correct')}; "
                f"qswap={None if swap is None else swap.get('first_top1')}"
            ),
        }
    )
    print(
        json.dumps(
            {
                "phase": "stage1_done",
                "id": tag,
                "verdict": verdict,
                "long_gap": report["long_gap"]["free_exact"],
                "first_correct": report["long_gap"].get("first_correct"),
                "induction": report["primitive_induction"]["first_top1"],
                "keyed": report["primitive_keyed"]["first_top1"],
                "query_swap": None if swap is None else swap.get("first_top1"),
            }
        ),
        flush=True,
    )
    return report


def is_grad_stop(report: dict) -> bool:
    return report.get("verdict") in {"GRAD", "OWNER"}


def run_autonomous_d(device_name: str = "cuda") -> dict:
    hashes = verify_c2_frozen()
    from .selection_p11_u16000_runtime import resolve_device

    device = resolve_device(device_name)
    model, overwrite, _ckpt = load_locator(device)
    overwrite.set_eval_write(WRITE_HARD, locator=LOCATOR_TOKEN)
    handle, _ = attach_overwrite(model, overwrite)
    locator_rt = LocatorRuntime(model, overwrite).install()
    session = DownstreamSession(model, overwrite).install()
    results = []
    try:
        print(json.dumps({"phase": "start", "device": str(device), "hashes": hashes}, default=str), flush=True)
        session.configure(mask_mode=MASK_OFF, write_src=WRITE_QUERY, later_site=None)
        diag = diagnose_c2_downstream(model, overwrite, device, POLICY_A1)
        write(OUT / "d_miss_split.json", _strip(diag))
        ledger_append(
            {
                "id": "d_miss_split",
                "policy": POLICY_A1,
                "change": diag["change"],
                "long_gap": {"free_exact": diag["free_exact"], "n": diag["n"], "first_correct": diag["first_correct"]},
                "primitive_induction": {},
                "primitive_keyed": {},
                "verdict": "DIAG",
                "lesson": diag["lesson"],
            }
        )
        print(json.dumps({"phase": "diag", **{k: v for k, v in diag.items() if k != "rows"}}, default=str), flush=True)
        results.append(_strip(diag))

        queue = name_first_treatments(diag)
        best = None
        for spec in queue:
            tag = operator_tag(spec["mask_mode"], spec["write_src"], spec.get("later_site"))
            if spec.get("preview_kill") and spec["mask_mode"] == MASK_INVENTORY:
                preview = {
                    "id": tag,
                    "verdict": "KILL",
                    "lesson": (
                        f"D1 preview inv_argmax={diag.get('inv_argmax_correct')} "
                        f"not a phase jump vs C2 first {C2_FIRST}; skip full canary"
                    ),
                    "long_gap": {"free_exact": diag.get("inv_argmax_correct"), "n": 215},
                    "primitive_induction": {"first_top1": OFF_INDUCTION_TOP1},
                    "primitive_keyed": {"first_top1": None},
                    "change": _change(spec["mask_mode"], spec["write_src"], spec.get("later_site")),
                }
                # still canary D1 once: it is cheap and the prescribed first readout treat
            report = run_d_candidate(
                model,
                overwrite,
                session,
                device,
                mask_mode=spec["mask_mode"],
                write_src=spec["write_src"],
                later_site=spec.get("later_site"),
            )
            results.append(_strip(report))
            if best is None or int(report["long_gap"]["free_exact"]) > int((best.get("long_gap") or {}).get("free_exact") or -1):
                best = report
            if is_grad_stop(report) or report.get("verdict") == "ADVANCE":
                session.configure(
                    mask_mode=spec["mask_mode"],
                    write_src=spec["write_src"],
                    later_site=spec.get("later_site"),
                )
                stage2, on = run_stage2_treated(model, overwrite, device, report, POLICY_A1)
                results.append(_strip(stage2))
                if stage2.get("verdict") == "ADVANCE":
                    stage3 = run_stage3_treated(model, overwrite, device, report, on, POLICY_A1)
                    results.append({"id": stage3.get("id", f"{tag}_stage3"), "verdict": stage3.get("verdict")})
                    if stage3.get("verdict") in {"PASS", "ADVANCE"} and is_grad_stop(report):
                        return {
                            "status": "RECOMMEND_PHASE_GRADUATION",
                            "best_id": tag,
                            "long_gap": report["long_gap"]["free_exact"],
                            "first_correct": report["long_gap"].get("first_correct"),
                            "hashes": hashes,
                            "diag": {k: v for k, v in diag.items() if k != "rows"},
                            "results": [
                                {
                                    "id": row.get("id"),
                                    "verdict": row.get("verdict"),
                                    "long_gap": (row.get("long_gap") or {}).get("free_exact"),
                                    "lesson": row.get("lesson"),
                                }
                                for row in results
                            ],
                            "promoted": False,
                            "test_opened": False,
                            "c2_preserved": True,
                            "owner_decision_needed": True,
                        }
                if is_grad_stop(report):
                    break
        survivors = [row for row in results if row.get("verdict") in {"ADVANCE", "GRAD", "OWNER", "PASS"}]
        status = "SURVIVOR" if survivors else "QUEUE_D_EVAL_EXHAUSTED"
        return {
            "status": status,
            "best_id": None if best is None else best.get("id"),
            "best_long_gap": None if best is None else (best.get("long_gap") or {}).get("free_exact"),
            "bottleneck": diag.get("bottleneck"),
            "hashes": hashes,
            "diag": {k: v for k, v in diag.items() if k != "rows"},
            "results": [
                {
                    "id": row.get("id"),
                    "verdict": row.get("verdict"),
                    "long_gap": (row.get("long_gap") or {}).get("free_exact"),
                    "lesson": row.get("lesson"),
                }
                for row in results
            ],
            "promoted": False,
            "test_opened": False,
            "c2_preserved": True,
        }
    finally:
        session.remove()
        locator_rt.remove()
        handle.remove()
        overwrite.gen_index = None
        overwrite.set_eval_write(WRITE_HARD, locator=LOCATOR_TOKEN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "diagnose", "canary"])
    parser.add_argument("--mask", default=MASK_INVENTORY, choices=[MASK_OFF, MASK_INVENTORY, MASK_MATCHED])
    parser.add_argument("--write-src", default=WRITE_QUERY, choices=[WRITE_QUERY, WRITE_VALUE])
    parser.add_argument("--later", default="", choices=["", LATER_BLOCK1, LATER_FINAL])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sidecar", default="250001", choices=["250001", "250002"])
    parser.add_argument("--id", default="")
    args = parser.parse_args()
    later = None if args.later == "" else args.later
    if args.action == "run":
        print(json.dumps(run_autonomous_d(args.device), default=str), flush=True)
        return
    hashes = verify_c2_frozen()
    from .selection_p11_u16000_runtime import resolve_device

    device = resolve_device(args.device)
    model, overwrite, _ckpt = load_locator(device, sidecar=args.sidecar)
    overwrite.set_eval_write(WRITE_HARD, locator=LOCATOR_TOKEN)
    handle, _ = attach_overwrite(model, overwrite)
    locator_rt = LocatorRuntime(model, overwrite).install()
    session = DownstreamSession(model, overwrite).install()
    try:
        if args.action == "diagnose":
            session.configure(mask_mode=MASK_OFF, write_src=WRITE_QUERY, later_site=None)
            diag = diagnose_c2_downstream(model, overwrite, device, POLICY_A1)
            write(OUT / "d_miss_split.json", _strip(diag))
            ledger_append(
                {
                    "id": "d_miss_split",
                    "policy": POLICY_A1,
                    "change": diag["change"],
                    "long_gap": {
                        "free_exact": diag["free_exact"],
                        "n": diag["n"],
                        "first_correct": diag["first_correct"],
                    },
                    "primitive_induction": {},
                    "primitive_keyed": {},
                    "verdict": "DIAG",
                    "lesson": diag["lesson"],
                }
            )
            print(json.dumps({"hashes": hashes, **{k: v for k, v in diag.items() if k != "rows"}}, default=str), flush=True)
        else:
            replica_id = args.id or (
                f"{operator_tag(args.mask, args.write_src, later)}_stage4_250002"
                if args.sidecar == "250002"
                else None
            )
            report = run_d_candidate(
                model,
                overwrite,
                session,
                device,
                mask_mode=args.mask,
                write_src=args.write_src,
                later_site=later,
                id_override=replica_id,
            )
            print(json.dumps(_strip(report), default=str), flush=True)
    finally:
        session.remove()
        locator_rt.remove()
        handle.remove()
        overwrite.gen_index = None


if __name__ == "__main__":
    main()
