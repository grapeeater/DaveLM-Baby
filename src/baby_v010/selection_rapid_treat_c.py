"""Eval-only C-series locators on A1 routing + B1 hard replace.

Does not mutate A1 (`selection_rapid_treat.py`) or B1 (`selection_rapid_treat_b.py`).
U16000 Baby is never replaced. Production source is never item['query_position'].
Gold query_position is diagnostic-only (pointer match labels).
"""
from __future__ import annotations

import argparse
import inspect
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_overwrite import attach_overwrite, pointer_aux
from .selection_p11 import GATE_BIAS
from .selection_p11_u16000_runtime import (
    TREATMENT_SHA,
    load_u16000_with_learned_overwrite,
    resolve_device,
)
from .selection_rapid_treat import (
    INDUCTION_DROP_BAR,
    KEYED_MIN_TOP1,
    OFF_INDUCTION_TOP1,
    OFF_LONG_FREE,
    OUT,
    POLICY_A1,
    REPLICA_CKPT,
    REPLICA_SHA,
    _long_gap_items,
    ledger_append,
    load_u16000_with_sidecar,
    measure_stage1,
    should_set_gen_index,
)
from .selection_rapid_treat_b import (
    A1_LONG_FREE,
    A1_QUERY_SWAP_NOVEL_TOP1,
    OWNER_LONG,
    POINTER_DIAG_N,
    Q0R_LONG_FREE,
    QUERY_SWAP_LIFT,
    WRITE_CONF,
    WRITE_GATED,
    WRITE_HARD,
    WRITE_MIX,
    WRITE_MODES,
    TreatedOverwrite,
    diagnose_pointer,
    run_pretest,
    run_stage2_treated,
    run_stage3_treated,
    score_query_swap,
    window_query_position,
)
from .selection_s1 import digest, write

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_A1_SHA = "cfe65d3c6c4ecc150c7ffae874b40bdb6e5b4e99cd9d77da31d0edb137979907"
EXPECTED_B1_PREFIX = "52450f58"
EXPECTED_B1_SUFFIX = "a1787"
EXPECTED_U16000 = "94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827"
B1_LONG = 109
ADVANCE_OVER_B1 = 110
MIN_CONTENT_TOKEN = 5
WRITE_ADD = "additive"

LOCATOR_SLOT = "slot"
LOCATOR_L0 = "l0"
LOCATOR_HYBRID = "hybrid"
LOCATOR_AGREE = "agree"
LOCATOR_PREV2 = "prev2"
LOCATOR_TOKEN = "token"
LOCATOR_CONTENT = "content"
LOCATORS = (
    LOCATOR_SLOT,
    LOCATOR_L0,
    LOCATOR_HYBRID,
    LOCATOR_AGREE,
    LOCATOR_PREV2,
    LOCATOR_TOKEN,
    LOCATOR_CONTENT,
)


class LastAttentionMap:
    """Capture bucket that keeps the latest appended attention tensor."""

    def __init__(self) -> None:
        self.weights: torch.Tensor | None = None

    def append(self, weights: torch.Tensor) -> None:
        self.weights = weights

    def clear(self) -> None:
        self.weights = None


def src_from_attn(weights: torch.Tensor, gen_index: torch.Tensor) -> torch.Tensor:
    """Mean-over-heads causal argmax at gen, excluding self. Gold labels unused."""
    mass = weights.mean(dim=1)
    batch, time, _ = mass.shape
    clamped = gen_index.clamp(0, time - 1)
    row = mass[torch.arange(batch, device=mass.device), clamped].clone()
    row[torch.arange(batch, device=mass.device), clamped] = float("-inf")
    return row.argmax(dim=-1)


def token_identity_src(tokens: list[int], gen: int) -> int | None:
    """Query copy site from (key, value, SEP) tiling. No gold query labels.

    Finds a repeated delimiter with constant spacing (body records), treats the
    token just after the previous delimiter-gap as a key, and returns the unique
    occurrence of those keys that is *not* a record start. Falls back to None
    when the tiling is not unique (caller uses B1 slot).
    """
    body = [int(t) for t in tokens[:gen]]
    positions: dict[int, list[int]] = {}
    for index, token in enumerate(body):
        if token >= MIN_CONTENT_TOKEN:
            positions.setdefault(token, []).append(index)
    candidates: list[tuple[int, int, int]] = []
    for _tok, spots in positions.items():
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
            if len(uniq) == 1:
                candidates.append((len(run), delta, uniq[0]))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def token_identity_batch(tokens: torch.Tensor, gen_index: torch.Tensor) -> torch.Tensor:
    batch, _time = tokens.shape
    out = torch.full((batch,), -1, device=tokens.device, dtype=torch.long)
    for row in range(batch):
        gen = int(gen_index[row].item())
        if gen < 0:
            continue
        src = token_identity_src(tokens[row].tolist(), gen)
        if src is not None:
            out[row] = int(src)
    return out


class LocatorOverwrite(TreatedOverwrite):
    """B1 write operator with a selectable production locator.

    Extra locators are eval-time; P11 scorer/gate weights stay the B1 sidecar.
    """

    def __init__(self, d_model: int, gate_bias: float = -4.0, gen_only: bool = False) -> None:
        super().__init__(d_model, gate_bias=gate_bias, gen_only=gen_only)
        self.locator = LOCATOR_SLOT
        self.attn_layer = 0
        self.attn_map = LastAttentionMap()
        self.forced_src: torch.Tensor | None = None
        self.locate_only = False
        self.last_slot_src: torch.Tensor | None = None
        self.last_l0_src: torch.Tensor | None = None
        self.last_agree: torch.Tensor | None = None

    def needs_attn_capture(self) -> bool:
        return self.locator in {LOCATOR_L0, LOCATOR_HYBRID, LOCATOR_AGREE}

    def set_eval_write(
        self,
        mode: str = WRITE_GATED,
        temperature: float = 1.0,
        mix_beta: float = 0.0,
        conf_tau: float = 0.5,
        locator: str | None = None,
        attn_layer: int | None = None,
    ) -> None:
        if mode == WRITE_ADD:
            self.write_mode = WRITE_ADD
            self.score_temperature = float(temperature)
            self.mix_beta = float(mix_beta)
            self.conf_tau = float(conf_tau)
        else:
            if mode not in WRITE_MODES:
                raise ValueError(f"unknown write_mode {mode}")
            super().set_eval_write(mode, temperature=temperature, mix_beta=mix_beta, conf_tau=conf_tau)
        if locator is not None:
            if locator not in LOCATORS:
                raise ValueError(f"unknown locator {locator}")
            self.locator = locator
        if attn_layer is not None:
            self.attn_layer = int(attn_layer)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, time, width = hidden.shape
        nxt = torch.cat([hidden[:, 1:], hidden[:, -1:]], dim=1)
        slot = self.scorer(torch.cat([hidden, nxt], dim=-1)).squeeze(-1)
        causal = torch.ones((time, time), dtype=torch.bool, device=hidden.device).tril()
        temp = max(float(self.score_temperature), 1e-8)
        scaled = slot / temp
        scores = scaled.unsqueeze(1).expand(batch, time, time).masked_fill(~causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        read = torch.matmul(attn, hidden)
        raw_gate = torch.sigmoid(self.gate(hidden))
        index = self.gen_index
        self.gen_index = None

        valid = hidden.new_zeros(batch, dtype=torch.bool)
        clamped = hidden.new_zeros(batch, dtype=torch.long)
        if index is not None and int(index.shape[0]) == batch:
            valid = index >= 0
            clamped = index.clamp(0, time - 1)

        if self.gen_only:
            mask = hidden.new_zeros(batch, time, 1)
            if bool(valid.any()):
                mask[torch.arange(batch, device=hidden.device)[valid], clamped[valid]] = 1.0
            gate = raw_gate * mask
        else:
            gate = raw_gate

        self.last_attn = attn
        self.last_gate = gate.squeeze(-1)
        gated_write = hidden + gate * (read - hidden)

        slot_src = None
        l0_src = None
        if bool(valid.any()):
            raw_scores = slot.unsqueeze(1).expand(batch, time, time).masked_fill(~causal, float("-inf"))
            gen_scores = raw_scores[torch.arange(batch, device=hidden.device), clamped]
            slot_src = gen_scores.argmax(dim=-1)
            self.last_slot_src = slot_src.detach()
            b_ix = torch.arange(batch, device=hidden.device)
            attn_gen = attn[b_ix, clamped]
            top2 = attn_gen.topk(k=min(2, time), dim=-1)
            self.last_pointer_mass = top2.values[:, 0].detach()
            if top2.values.shape[-1] > 1:
                self.last_second_mass = top2.values[:, 1].detach()
            else:
                self.last_second_mass = hidden.new_zeros(batch)
            captured = self.attn_map.weights
            if captured is not None and captured.shape[0] == batch and captured.shape[-1] == time:
                l0_src = src_from_attn(captured, clamped)
                self.last_l0_src = l0_src.detach()
            else:
                self.last_l0_src = None
        else:
            self.last_slot_src = None
            self.last_l0_src = None
            self.last_src_index = None
            self.last_pointer_mass = None
            self.last_second_mass = None
            self.last_agree = None
            rewritten = hidden if (self.gen_only or self.locate_only) else gated_write
            self.last_hidden = rewritten
            return rewritten

        src = self._pick_src(slot_src, l0_src, valid, clamped)
        forced = self.forced_src
        self.forced_src = None
        if forced is not None and slot_src is not None:
            forced = forced.to(device=hidden.device)
            use_forced = (forced >= 0) & valid
            src = torch.where(use_forced, forced.clamp(0, time - 1), src)
        self.last_src_index = src.detach() if src is not None else None
        if slot_src is not None and l0_src is not None:
            self.last_agree = (slot_src == l0_src) & valid
        else:
            self.last_agree = None

        if self.locate_only:
            self.last_hidden = hidden
            return hidden

        hard = hidden.clone()
        additive = hidden.clone()
        b_ix = torch.arange(batch, device=hidden.device)
        write_b = b_ix[valid]
        hard[write_b, clamped[valid]] = hidden[write_b, src[valid]]
        additive[write_b, clamped[valid]] = hidden[write_b, clamped[valid]] + hidden[write_b, src[valid]]

        if self.locator == LOCATOR_AGREE:
            if self.last_agree is None:
                rewritten = gated_write
            else:
                rewritten = gated_write.clone()
                rewritten[self.last_agree] = hard[self.last_agree]
        elif self.write_mode == WRITE_HARD:
            rewritten = hard
        elif self.write_mode == WRITE_ADD:
            rewritten = additive
        elif self.write_mode == WRITE_MIX:
            beta = float(self.mix_beta)
            rewritten = (1.0 - beta) * gated_write + beta * hard
        elif self.write_mode == WRITE_CONF:
            peak = self.last_pointer_mass
            use_hard = (peak >= float(self.conf_tau)) & valid
            rewritten = gated_write.clone()
            rewritten[use_hard] = hard[use_hard]
        else:
            rewritten = gated_write
        self.last_hidden = rewritten
        return rewritten

    def _pick_src(
        self,
        slot_src: torch.Tensor,
        l0_src: torch.Tensor | None,
        valid: torch.Tensor,
        clamped: torch.Tensor,
    ) -> torch.Tensor:
        locator = self.locator
        if locator == LOCATOR_PREV2:
            return (clamped - 2).clamp(min=0)
        if locator == LOCATOR_L0 and l0_src is not None:
            return l0_src
        if locator == LOCATOR_HYBRID and l0_src is not None and self.last_pointer_mass is not None:
            use_l0 = (self.last_pointer_mass < float(self.conf_tau)) & valid
            return torch.where(use_l0, l0_src, slot_src)
        if locator == LOCATOR_CONTENT:
            content_src = self._content_src(slot_src, clamped)
            if content_src is not None:
                return content_src
        return slot_src

    def _content_src(self, slot_src: torch.Tensor, clamped: torch.Tensor) -> torch.Tensor | None:
        return None


class ContentAddressOverwrite(LocatorOverwrite):
    """C3: bilinear content-address pointer from gen, independent of P11 slot scorer."""

    def __init__(self, d_model: int, gate_bias: float = -4.0, gen_only: bool = False) -> None:
        super().__init__(d_model, gate_bias=gate_bias, gen_only=gen_only)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        self.last_content_attn: torch.Tensor | None = None
        self.last_hidden_for_content: torch.Tensor | None = None
        self.locator = LOCATOR_CONTENT

    def _content_src(self, slot_src: torch.Tensor, clamped: torch.Tensor) -> torch.Tensor | None:
        hidden = self.last_hidden_for_content
        if hidden is None:
            return None
        batch, time, width = hidden.shape
        scale = width ** -0.5
        scores = torch.matmul(self.query(hidden), self.key(hidden).transpose(-2, -1)) * scale
        causal = torch.ones((time, time), dtype=torch.bool, device=hidden.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        self.last_content_attn = attn
        row = scores[torch.arange(batch, device=hidden.device), clamped].clone()
        row[torch.arange(batch, device=hidden.device), clamped] = float("-inf")
        return row.argmax(dim=-1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        self.last_hidden_for_content = hidden
        try:
            return super().forward(hidden)
        finally:
            self.last_hidden_for_content = None


class LocatorRuntime:
    """Wire L0/later-layer capture or C2 token scan only while gen_index is armed.

    Unarmed forwards (induction / later decode steps) keep SDPA. Does not wrap
    identity A1/B1 when locator=slot.
    """

    def __init__(self, model, overwrite: LocatorOverwrite) -> None:
        self.model = model
        self.overwrite = overwrite
        self._orig_forward = model.forward

    def install(self) -> "LocatorRuntime":
        ow = self.overwrite
        model = self.model
        orig = self._orig_forward

        def wrapped(tokens, *args, **kwargs):
            armed = ow.gen_index is not None and bool((ow.gen_index >= 0).any().item())
            if not armed:
                return orig(tokens, *args, **kwargs)
            if ow.locator == LOCATOR_TOKEN:
                ow.forced_src = token_identity_batch(tokens, ow.gen_index)
                try:
                    return orig(tokens, *args, **kwargs)
                finally:
                    ow.forced_src = None
            if not ow.needs_attn_capture():
                return orig(tokens, *args, **kwargs)
            layer = int(ow.attn_layer)
            attn_mod = model.blocks[layer].attention
            if layer == 0:
                attn_mod._capture = ow.attn_map
                try:
                    return orig(tokens, *args, **kwargs)
                finally:
                    attn_mod._capture = None
                    ow.attn_map.clear()
            saved = ow.gen_index
            ow.gen_index = torch.full_like(saved, -1)
            attn_mod._capture = ow.attn_map
            try:
                orig(tokens, *args, **kwargs)
            finally:
                attn_mod._capture = None
                ow.gen_index = saved
            if ow.attn_map.weights is not None:
                ow.forced_src = src_from_attn(ow.attn_map.weights, saved)
            ow.attn_map.clear()
            try:
                return orig(tokens, *args, **kwargs)
            finally:
                ow.forced_src = None

        model.forward = wrapped
        return self

    def remove(self) -> None:
        self.model.forward = self._orig_forward
        for block in self.model.blocks:
            block.attention._capture = None
        self.overwrite.attn_map.clear()
        self.overwrite.forced_src = None


def verify_locked_hashes() -> dict:
    a1 = digest(ROOT / "src/baby_v010/selection_rapid_treat.py")
    b1 = digest(ROOT / "src/baby_v010/selection_rapid_treat_b.py")
    parent = digest(ROOT / "runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt")
    ok = (
        a1 == EXPECTED_A1_SHA
        and b1.startswith(EXPECTED_B1_PREFIX)
        and b1.endswith(EXPECTED_B1_SUFFIX)
        and parent == EXPECTED_U16000
    )
    report = {
        "a1_sha256": a1,
        "b1_sha256": b1,
        "u16000_sha256": parent,
        "a1_ok": a1 == EXPECTED_A1_SHA,
        "b1_ok": b1.startswith(EXPECTED_B1_PREFIX) and b1.endswith(EXPECTED_B1_SUFFIX),
        "u16000_ok": parent == EXPECTED_U16000,
        "ok": ok,
    }
    if not ok:
        raise RuntimeError(f"locked hash mismatch: {report}")
    return report


def load_locator(device, sidecar: str = "250001") -> tuple:
    if sidecar == "250002":
        model, base, ckpt = load_u16000_with_sidecar(device, REPLICA_CKPT, REPLICA_SHA)
    else:
        model, base, ckpt = load_u16000_with_learned_overwrite(device)
    d_model = int(base.gate.in_features)
    treated = LocatorOverwrite(d_model, gate_bias=GATE_BIAS, gen_only=True).to(device)
    treated.load_state_dict(base.state_dict())
    treated.eval()
    treated.set_eval_write(WRITE_HARD, locator=LOCATOR_SLOT, attn_layer=0)
    return model, treated, ckpt


def operator_tag(locator: str, mode: str, conf_tau: float, attn_layer: int) -> str:
    if locator == LOCATOR_L0 and attn_layer == 0 and mode == WRITE_HARD:
        return "c1b"
    if locator == LOCATOR_L0 and attn_layer != 0:
        return f"c1b_l{attn_layer}"
    if locator == LOCATOR_HYBRID:
        ttag = str(conf_tau).replace(".", "p")
        layer = "" if attn_layer == 0 else f"_l{attn_layer}"
        return f"c1c_{ttag}{layer}"
    if locator == LOCATOR_AGREE:
        layer = "" if attn_layer == 0 else f"_l{attn_layer}"
        return f"c1d{layer}"
    if locator == LOCATOR_PREV2:
        return "c1a"
    if locator == LOCATOR_TOKEN:
        return "c2"
    if locator == LOCATOR_CONTENT:
        return "c3_content"
    if mode == WRITE_ADD:
        return "c4_add"
    return f"c_{locator}_{mode}"


def _change(locator: str, mode: str, conf_tau: float, attn_layer: int) -> str:
    base = "A1 induction-exempt first-step; "
    if locator == LOCATOR_L0:
        return base + (
            f"C1b hard replace h0[gen]:=h0[mean-head L{attn_layer} attn argmax] "
            "(not query_position)"
        )
    if locator == LOCATOR_HYBRID:
        return base + (
            f"C1c hard replace; L{attn_layer} attn argmax if slot mass<{conf_tau} else B1 slot argmax"
        )
    if locator == LOCATOR_AGREE:
        return base + (
            f"C1d hard replace iff B1 slot argmax agrees with L{attn_layer} attn argmax; else A1 gated"
        )
    if locator == LOCATOR_PREV2:
        return base + "C1a hard replace src=gen-2"
    if locator == LOCATOR_TOKEN:
        return base + "C2 structural (key,value,SEP) query site; else B1 slot; hard replace"
    if locator == LOCATOR_CONTENT:
        return base + "C3 content-address Q/K from gen residual; hard replace"
    if mode == WRITE_ADD:
        return base + "C4 additive h0[gen]+=h0[src] with B1 slot src"
    return base + f"locator={locator} write={mode}"


def c_canary_verdict(report: dict) -> tuple[str, str]:
    long_on = int(report["long_gap"]["free_exact"])
    n_long = int(report["long_gap"]["n"])
    ind = float(report["primitive_induction"]["first_top1"])
    keyed = float(report["primitive_keyed"]["first_top1"])
    swap = report.get("query_swap_same_surface_novel") or {}
    swap_top1 = swap.get("first_top1")
    bind_up = (
        swap_top1 is not None and float(swap_top1) >= A1_QUERY_SWAP_NOVEL_TOP1 + QUERY_SWAP_LIFT - 1e-12
    )
    if ind + 1e-12 < (OFF_INDUCTION_TOP1 - INDUCTION_DROP_BAR):
        return "KILL", "induction regression vs A1 exemption"
    if keyed + 1e-12 < KEYED_MIN_TOP1:
        return "KILL", "primitive_keyed negative control failed"
    if long_on <= OFF_LONG_FREE + 4:
        return "KILL", "long-gap collapsed toward OFF 71"
    if long_on >= OWNER_LONG:
        return "OWNER", f"long-gap {long_on}/{n_long} past owner bar ~{OWNER_LONG}"
    if long_on >= ADVANCE_OVER_B1:
        return "ADVANCE", f"long-gap {long_on}/{n_long} lifts over B1 {B1_LONG}"
    if long_on == B1_LONG and bind_up:
        return "ADVANCE", "long-gap held at B1 109 with query-swap bind clearly up"
    if long_on > A1_LONG_FREE and bind_up:
        return "ADVANCE", f"long-gap {long_on}/{n_long} below B1 but bind clearly up"
    return "KILL", f"long-gap {long_on}/{n_long} did not beat B1 {B1_LONG} without bind rescue"


def _strip(report: dict) -> dict:
    return {k: v for k, v in report.items() if k not in {"long_gap_rows", "pointer_rows", "diag_rows"}}


@torch.no_grad()
def diagnose_c_pointer(
    model,
    overwrite: LocatorOverwrite,
    items: list[dict],
    device,
    policy: str = POLICY_A1,
) -> dict:
    rows = []
    n_layers = len(model.blocks)
    for item in items:
        generated = [int(t) for t in item["input"]]
        window = generated[-256:]
        activate = bool(should_set_gen_index(item, policy))
        gen_t = len(window) - 1
        q_win = window_query_position(item, len(window))
        captured: list = []
        model.set_attention_capture(captured)
        overwrite.locate_only = True
        overwrite.gen_index = (
            torch.tensor([gen_t], device=device, dtype=torch.long) if activate else None
        )
        tokens = torch.tensor([window], dtype=torch.long, device=device)
        model(tokens)
        model.set_attention_capture(None)
        overwrite.locate_only = False
        slot_src = int(overwrite.last_slot_src[0].item()) if overwrite.last_slot_src is not None else None
        peak = float(overwrite.last_pointer_mass[0].item()) if overwrite.last_pointer_mass is not None else None
        layer_src = []
        for weights in captured[:n_layers]:
            src = src_from_attn(weights, torch.tensor([gen_t], device=device, dtype=torch.long))
            layer_src.append(int(src[0].item()))
        l0_src = layer_src[0] if layer_src else None
        c2_src = token_identity_src(window, gen_t) if activate else None
        c1a_src = max(gen_t - 2, 0) if activate else None
        hybrid_src = None
        if slot_src is not None and l0_src is not None and peak is not None:
            hybrid_src = l0_src if peak < float(overwrite.conf_tau) else slot_src
        agree = bool(slot_src is not None and l0_src is not None and slot_src == l0_src)
        rows.append(
            {
                "body_id": item.get("body_id"),
                "armed": activate,
                "slot_src": slot_src,
                "l0_src": l0_src,
                "layer_src": layer_src,
                "c2_src": c2_src,
                "c1a_src": c1a_src,
                "hybrid_src": hybrid_src,
                "agree": agree,
                "slot_peak": peak,
                "query_window": q_win,
                "slot_match": bool(slot_src is not None and q_win is not None and slot_src == q_win),
                "l0_match": bool(l0_src is not None and q_win is not None and l0_src == q_win),
                "c2_match": bool(c2_src is not None and q_win is not None and c2_src == q_win),
                "c1a_match": bool(c1a_src is not None and q_win is not None and c1a_src == q_win),
                "hybrid_match": bool(hybrid_src is not None and q_win is not None and hybrid_src == q_win),
            }
        )
    armed = [row for row in rows if row["armed"]]
    slot_miss = [row for row in armed if not row["slot_match"]]
    slot_hit = [row for row in armed if row["slot_match"]]

    def _rate(field: str, subset: list[dict]) -> float:
        if not subset:
            return 0.0
        return sum(int(row[field]) for row in subset) / len(subset)

    layer_rates = []
    miss_layer_rates = []
    n_layer = max((len(row["layer_src"]) for row in armed), default=0)
    for layer in range(n_layer):
        hits = 0
        miss_hits = 0
        for row in armed:
            src = row["layer_src"][layer] if layer < len(row["layer_src"]) else None
            if src is not None and row["query_window"] is not None and src == row["query_window"]:
                hits += 1
                if not row["slot_match"]:
                    miss_hits += 1
        layer_rates.append(hits / len(armed) if armed else 0.0)
        miss_layer_rates.append(miss_hits / len(slot_miss) if slot_miss else 0.0)
    return {
        "n": len(rows),
        "n_armed": len(armed),
        "n_slot_match": sum(int(row["slot_match"]) for row in armed),
        "n_slot_miss": len(slot_miss),
        "slot_match_rate": _rate("slot_match", armed),
        "l0_match_rate": _rate("l0_match", armed),
        "l0_on_slot_miss": _rate("l0_match", slot_miss),
        "l0_on_slot_hit": _rate("l0_match", slot_hit),
        "hybrid_match_rate": _rate("hybrid_match", armed),
        "c2_match_rate": _rate("c2_match", armed),
        "c2_on_slot_miss": _rate("c2_match", slot_miss),
        "c2_coverage": (sum(row["c2_src"] is not None for row in armed) / len(armed)) if armed else 0.0,
        "c1a_match_rate": _rate("c1a_match", armed),
        "agree_rate": (sum(int(row["agree"]) for row in armed) / len(armed)) if armed else 0.0,
        "agree_and_slot_match": (
            sum(int(row["agree"] and row["slot_match"]) for row in armed) / len(armed) if armed else 0.0
        ),
        "layer_match_rate": layer_rates,
        "layer_on_slot_miss": miss_layer_rates,
        "median_slot_peak_match": statistics.median(
            [row["slot_peak"] for row in slot_hit if row["slot_peak"] is not None]
        )
        if slot_hit
        else None,
        "median_slot_peak_miss": statistics.median(
            [row["slot_peak"] for row in slot_miss if row["slot_peak"] is not None]
        )
        if slot_miss
        else None,
        "used_gold_query_position_as_source": False,
        "rows": rows,
    }


def run_c_candidate(
    model,
    overwrite: LocatorOverwrite,
    device,
    *,
    locator: str,
    mode: str = WRITE_HARD,
    conf_tau: float = 0.5,
    attn_layer: int = 0,
    policy: str = POLICY_A1,
    pretest: bool = True,
    score_swap: bool = True,
    pointer_n: int = POINTER_DIAG_N,
    id_override: str | None = None,
    pointer_floor: float = 0.65,
) -> dict:
    overwrite.set_eval_write(mode, temperature=1.0, mix_beta=0.0, conf_tau=conf_tau, locator=locator, attn_layer=attn_layer)
    tag = id_override or operator_tag(locator, mode, conf_tau, attn_layer)
    t0 = time.time()
    long_items = _long_gap_items()
    pointer = diagnose_pointer(model, overwrite, long_items[:pointer_n], device, policy)
    print(
        json.dumps(
            {
                "phase": "pointer",
                "id": tag,
                "operator": tag,
                "match_rate": pointer["match_rate"],
                "median_mass_src_match": pointer.get("median_mass_src_match"),
                "median_mass_src_miss": pointer.get("median_mass_src_miss"),
            }
        ),
        flush=True,
    )
    if pretest:
        pre = run_pretest(model, overwrite, device, policy=policy)
        print(
            json.dumps(
                {
                    "phase": "pretest_long",
                    "id": tag,
                    "free_exact": pre["long_gap"]["free_exact"],
                    "n": pre["long_gap"]["n"],
                    "kill": pre["kill"],
                }
            ),
            flush=True,
        )
        if pre["kill"]:
            report = {
                "id": tag,
                "operator": tag,
                "policy": policy,
                "write_mode": mode,
                "locator": locator,
                "attn_layer": attn_layer,
                "conf_tau": conf_tau,
                "change": _change(locator, mode, conf_tau, attn_layer),
                "verdict": "KILL",
                "lesson": pre["lesson"],
                "pretest": pre,
                "pointer": {k: v for k, v in pointer.items() if k != "rows"},
                "long_gap": pre["long_gap"],
                "primitive_induction": pre["primitive_induction"],
                "primitive_keyed": {"first_top1": None, "n": 0},
                "baby_weights": "authoritative_u16000",
                "p11_model_state_dict_loaded": False,
                "overwrite_checkpoint_sha256": TREATMENT_SHA,
                "used_gold_query_position_as_source": False,
                "protected_material_opened": False,
                "promoted": False,
                "test_opened": False,
                "elapsed_s": time.time() - t0,
            }
            write(OUT / f"{tag}_pretest.json", _strip(report))
            ledger_append(report)
            return report

    if float(pointer["match_rate"]) + 1e-12 < pointer_floor:
        report = {
            "id": tag,
            "operator": tag,
            "policy": policy,
            "write_mode": mode,
            "locator": locator,
            "attn_layer": attn_layer,
            "conf_tau": conf_tau,
            "change": _change(locator, mode, conf_tau, attn_layer),
            "verdict": "KILL",
            "lesson": (
                f"pointer_match={pointer['match_rate']:.3f} below {pointer_floor} vs B1 0.712; "
                "skip full 215"
            ),
            "pointer": {k: v for k, v in pointer.items() if k != "rows"},
            "long_gap": {"free_exact": None, "n": 215},
            "primitive_induction": {"first_top1": None, "n": 0},
            "primitive_keyed": {"first_top1": None, "n": 0},
            "baby_weights": "authoritative_u16000",
            "p11_model_state_dict_loaded": False,
            "overwrite_checkpoint_sha256": TREATMENT_SHA,
            "used_gold_query_position_as_source": False,
            "protected_material_opened": False,
            "promoted": False,
            "test_opened": False,
            "elapsed_s": time.time() - t0,
        }
        write(OUT / f"{tag}_pointer_kill.json", _strip(report))
        ledger_append(report)
        print(json.dumps({"phase": "pointer_kill", "id": tag, "match_rate": pointer["match_rate"]}), flush=True)
        return report

    stage1 = measure_stage1(model, overwrite, device, policy)
    swap = score_query_swap(model, overwrite, device, policy) if score_swap else None
    report = dict(stage1)
    report["id"] = tag
    report["operator"] = tag
    report["write_mode"] = mode
    report["locator"] = locator
    report["attn_layer"] = attn_layer
    report["conf_tau"] = conf_tau
    report["change"] = _change(locator, mode, conf_tau, attn_layer)
    report["pointer"] = {k: v for k, v in pointer.items() if k != "rows"}
    report["query_swap_same_surface_novel"] = swap
    report["used_gold_query_position_as_source"] = False
    report["a1_long_gap_free_exact"] = A1_LONG_FREE
    report["b1_long_gap_free_exact"] = B1_LONG
    report["q0r_ceiling"] = Q0R_LONG_FREE
    verdict, lesson = c_canary_verdict(report)
    report["verdict"] = verdict
    report["lesson"] = lesson
    report["elapsed_s"] = time.time() - t0
    write(OUT / f"{tag}_stage1.json", _strip(report))
    ledger_append(
        {
            **report,
            "lesson": (
                f"{lesson}; pointer_match={pointer['match_rate']:.3f}; "
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
                "induction": report["primitive_induction"]["first_top1"],
                "keyed": report["primitive_keyed"]["first_top1"],
                "query_swap": None if swap is None else swap.get("first_top1"),
                "pointer_match": pointer["match_rate"],
            }
        ),
        flush=True,
    )
    return report


def is_owner_stop(report: dict) -> bool:
    return report.get("verdict") == "OWNER"


def production_forward_uses_query_position() -> bool:
    source = inspect.getsource(LocatorOverwrite.forward) + inspect.getsource(LocatorOverwrite._pick_src)
    return "query_position" in source


def run_c3_content(model, overwrite: LocatorOverwrite, device, until: int = 400) -> dict:
    from .selection_p11 import OUT as P11_OUT, TRAIN_SEED, PTR_LAMBDA, OVERWRITE_LR
    from .selection_p4 import copy_specs
    from . import selection_s1
    from .selection_s1 import pack

    schedule_path = P11_OUT / f"SCHEDULE_{TRAIN_SEED}.json"
    if not schedule_path.exists():
        report = {
            "id": "c3_content",
            "verdict": "BLOCKED",
            "lesson": f"P11 schedule missing: {schedule_path}",
            "long_gap": {},
            "primitive_induction": {},
            "primitive_keyed": {},
            "protected_material_opened": False,
            "promoted": False,
            "test_opened": False,
        }
        ledger_append(report)
        return report

    dest = OUT / "c3_content"
    dest.mkdir(parents=True, exist_ok=True)
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trained = ContentAddressOverwrite(int(overwrite.gate.in_features), gate_bias=GATE_BIAS, gen_only=True).to(device)
    trained.load_state_dict(overwrite.state_dict(), strict=False)
    trained.train()
    trained.set_eval_write(WRITE_HARD, locator=LOCATOR_CONTENT)
    for parameter in trained.scorer.parameters():
        parameter.requires_grad_(False)
    for parameter in trained.gate.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        list(trained.query.parameters()) + list(trained.key.parameters()),
        lr=OVERWRITE_LR,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    handle, _ = attach_overwrite(model, trained)
    previous_hook = selection_s1.PACK_HOOK
    selection_s1.PACK_HOOK = lambda items, x, module=trained: module.set_gen_index_routed(items, x.device, POLICY_A1)
    t0 = time.time()
    try:
        last_step = 0
        for step in range(1, until + 1):
            spec = schedule[step - 1]
            if spec["task"] == "language":
                last_step = step
                continue
            model.train()
            trained.train()
            trained.set_eval_write(WRITE_GATED, locator=LOCATOR_CONTENT)
            items = spec["items"]
            x, y, mask, _first = pack(items, device)
            hidden = model.forward_hidden(x)
            logits = model.language_head(hidden.detach())
            loss = F.cross_entropy(logits[mask], y[mask])
            specs = copy_specs(items, min_gap=2)
            extra_ptr = hidden.new_zeros(())
            if trained.last_content_attn is not None and specs:
                extra_ptr, _extra_gate = pointer_aux(
                    trained.last_content_attn, trained.last_gate, specs, use_gate=False
                )
                loss = loss + PTR_LAMBDA * extra_ptr
            optimizer.zero_grad(set_to_none=True)
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(trained.query.parameters()) + list(trained.key.parameters()), 2.0
                )
                optimizer.step()
            last_step = step
            if step % 50 == 0:
                print(
                    json.dumps(
                        {
                            "phase": "c3_train",
                            "step": step,
                            "loss": float(loss.detach().item()),
                            "ptr": float(extra_ptr.detach().item()) if specs else 0.0,
                        }
                    ),
                    flush=True,
                )
            if step % 200 == 0:
                model.eval()
                trained.eval()
                trained.set_eval_write(WRITE_HARD, locator=LOCATOR_CONTENT)
                canary = run_c_candidate(
                    model,
                    trained,
                    device,
                    locator=LOCATOR_CONTENT,
                    mode=WRITE_HARD,
                    pretest=True,
                    score_swap=False,
                    id_override=f"c3_step{step}",
                )
                write(dest / f"eval_{step:04d}.json", _strip(canary))
                if is_owner_stop(canary) or canary.get("verdict") == "ADVANCE":
                    torch.save(
                        {
                            "overwrite_state_dict": trained.state_dict(),
                            "step": step,
                            "parent_checkpoint_sha256": TREATMENT_SHA,
                        },
                        dest / f"checkpoint_{step}.pt",
                    )
                    if is_owner_stop(canary):
                        return canary
                torch.save({"overwrite_state_dict": trained.state_dict(), "step": step}, dest / f"checkpoint_{step}.pt")
        model.eval()
        trained.eval()
        trained.set_eval_write(WRITE_HARD, locator=LOCATOR_CONTENT)
        final = run_c_candidate(
            model, trained, device, locator=LOCATOR_CONTENT, mode=WRITE_HARD, pretest=False, id_override="c3_content"
        )
        final["train_steps"] = last_step
        final["elapsed_s"] = time.time() - t0
        write(dest / "FINAL.json", _strip(final))
        ledger_append(final)
        return final
    finally:
        selection_s1.PACK_HOOK = previous_hook
        handle.remove()
        overwrite.eval()
        overwrite.set_eval_write(WRITE_HARD, locator=LOCATOR_SLOT)


def _best_miss_layer(diag: dict) -> tuple[int, float]:
    rates = diag.get("layer_on_slot_miss") or []
    if not rates:
        return 0, 0.0
    best = max(range(len(rates)), key=lambda i: rates[i])
    return best, float(rates[best])


def run_autonomous_c(device_name: str = "cuda") -> dict:
    hashes = verify_locked_hashes()
    device = resolve_device(device_name)
    model, overwrite, _ckpt = load_locator(device)
    handle, _ = attach_overwrite(model, overwrite)
    runtime = LocatorRuntime(model, overwrite).install()
    results = []
    try:
        print(json.dumps({"phase": "start", "device": str(device), "hashes": hashes}, default=str), flush=True)
        overwrite.set_eval_write(WRITE_HARD, locator=LOCATOR_SLOT, conf_tau=0.3)
        diag = diagnose_c_pointer(model, overwrite, _long_gap_items(), device, POLICY_A1)
        slim_diag = {k: v for k, v in diag.items() if k != "rows"}
        write(OUT / "c_locator_diag.json", slim_diag)
        ledger_append(
            {
                "id": "c_locator_diag",
                "policy": POLICY_A1,
                "change": "eval-only locator pointer diagnostic (query_position labels only)",
                "long_gap": {"free_exact": None, "n": diag["n_armed"]},
                "primitive_induction": {},
                "primitive_keyed": {},
                "verdict": "DIAG",
                "lesson": (
                    f"slot={diag['slot_match_rate']:.3f} l0={diag['l0_match_rate']:.3f} "
                    f"l0_on_miss={diag['l0_on_slot_miss']:.3f} hybrid={diag['hybrid_match_rate']:.3f} "
                    f"c2={diag['c2_match_rate']:.3f} cover={diag['c2_coverage']:.3f} "
                    f"agree={diag['agree_rate']:.3f} layer_miss={diag['layer_on_slot_miss']}"
                ),
            }
        )
        print(json.dumps({"phase": "diag", **slim_diag}, default=str), flush=True)

        queue = [
            {"locator": LOCATOR_L0, "mode": WRITE_HARD, "conf_tau": 0.5, "attn_layer": 0},
            {"locator": LOCATOR_HYBRID, "mode": WRITE_HARD, "conf_tau": 0.3, "attn_layer": 0},
            {"locator": LOCATOR_HYBRID, "mode": WRITE_HARD, "conf_tau": 0.5, "attn_layer": 0},
        ]
        if float(diag.get("agree_rate") or 0.0) >= 0.05:
            queue.append({"locator": LOCATOR_AGREE, "mode": WRITE_HARD, "conf_tau": 0.5, "attn_layer": 0})
        else:
            skip_d = {
                "id": "c1d",
                "verdict": "KILL",
                "lesson": "C1d agree_rate=0 so write is A1 gated on every row; not a new locator",
                "long_gap": {"free_exact": A1_LONG_FREE, "n": 215},
                "primitive_induction": {"first_top1": OFF_INDUCTION_TOP1},
                "primitive_keyed": {"first_top1": None},
            }
            results.append(skip_d)
            ledger_append(
                {
                    "id": "c1d",
                    "policy": POLICY_A1,
                    "change": _change(LOCATOR_AGREE, WRITE_HARD, 0.5, 0),
                    "long_gap": {"free_exact": A1_LONG_FREE, "n": 215},
                    "primitive_induction": {},
                    "primitive_keyed": {},
                    "verdict": "KILL",
                    "lesson": skip_d["lesson"],
                }
            )
        queue.append({"locator": LOCATOR_TOKEN, "mode": WRITE_HARD, "conf_tau": 0.5, "attn_layer": 0})
        for spec in queue:
            report = run_c_candidate(model, overwrite, device, **spec)
            results.append(_strip(report))
            if is_owner_stop(report):
                break

        survivors = [row for row in results if row.get("verdict") in {"ADVANCE", "OWNER"}]
        if not survivors:
            best_layer, miss_rate = _best_miss_layer(diag)
            if best_layer > 0 and miss_rate >= 0.20:
                report = run_c_candidate(
                    model,
                    overwrite,
                    device,
                    locator=LOCATOR_L0,
                    mode=WRITE_HARD,
                    attn_layer=best_layer,
                )
                results.append(_strip(report))
                if is_owner_stop(report):
                    return _stop("OWNER_C1_LAYER", results, report, hashes)
                if report.get("verdict") in {"ADVANCE", "OWNER"}:
                    survivors.append(report)
                hybrid = run_c_candidate(
                    model,
                    overwrite,
                    device,
                    locator=LOCATOR_HYBRID,
                    mode=WRITE_HARD,
                    conf_tau=0.3,
                    attn_layer=best_layer,
                )
                results.append(_strip(hybrid))
                if is_owner_stop(hybrid):
                    return _stop("OWNER_C1C_LAYER", results, hybrid, hashes)
                if hybrid.get("verdict") in {"ADVANCE", "OWNER"}:
                    survivors.append(hybrid)

            c1a = run_c_candidate(model, overwrite, device, locator=LOCATOR_PREV2, mode=WRITE_HARD)
            results.append(_strip(c1a))
            if is_owner_stop(c1a):
                return _stop("OWNER_C1A", results, c1a, hashes)

            if float(diag.get("c2_match_rate") or 0.0) < 0.40 and not any(
                row.get("id") == "c2" for row in results
            ):
                skip = {
                    "id": "c2",
                    "verdict": "SKIP",
                    "lesson": (
                        f"C2 structural match too low "
                        f"(match={diag.get('c2_match_rate')} cover={diag.get('c2_coverage')})"
                    ),
                    "long_gap": {},
                }
                results.append(skip)
                ledger_append(
                    {
                        "id": "c2",
                        "policy": POLICY_A1,
                        "change": _change(LOCATOR_TOKEN, WRITE_HARD, 0.5, 0),
                        "long_gap": {},
                        "primitive_induction": {},
                        "primitive_keyed": {},
                        "verdict": "SKIP",
                        "lesson": skip["lesson"],
                    }
                )

            if not any(row.get("verdict") in {"ADVANCE", "OWNER"} for row in results):
                handle.remove()
                handle = None
                runtime.remove()
                c3 = run_c3_content(model, overwrite, device, until=400)
                results.append(_strip(c3) if isinstance(c3, dict) else {"id": "c3_content", "verdict": "BLOCKED"})
                if is_owner_stop(c3):
                    return _stop("OWNER_C3", results, c3, hashes)
                handle, _ = attach_overwrite(model, overwrite)
                runtime = LocatorRuntime(model, overwrite).install()
                if c3.get("verdict") not in {"ADVANCE", "OWNER"}:
                    c4 = run_c_candidate(
                        model, overwrite, device, locator=LOCATOR_SLOT, mode=WRITE_ADD
                    )
                    results.append(_strip(c4))
                    if is_owner_stop(c4):
                        return _stop("OWNER_C4", results, c4, hashes)

        survivors = [row for row in results if row.get("verdict") in {"ADVANCE", "OWNER"}]
        if survivors:
            best = max(survivors, key=lambda row: int((row.get("long_gap") or {}).get("free_exact") or -1))
            overwrite.set_eval_write(
                best.get("write_mode") or WRITE_HARD,
                conf_tau=float(best.get("conf_tau") or 0.5),
                locator=best.get("locator") or LOCATOR_SLOT,
                attn_layer=int(best.get("attn_layer") or 0),
            )
            stage2, on = run_stage2_treated(model, overwrite, device, best, POLICY_A1)
            results.append(_strip(stage2))
            if stage2.get("verdict") == "ADVANCE":
                stage3 = run_stage3_treated(model, overwrite, device, best, on, POLICY_A1)
                results.append({"id": stage3.get("id", f"{best.get('id')}_stage3"), "verdict": stage3.get("verdict")})
                if stage3.get("verdict") in {"PASS", "ADVANCE"} and int((best.get("long_gap") or {}).get("free_exact") or 0) >= OWNER_LONG:
                    return _stop("OWNER_STAGE3", results, best, hashes)
            status = "SURVIVOR"
        else:
            best = max(results, key=lambda row: int((row.get("long_gap") or {}).get("free_exact") or -1), default={})
            status = "QUEUE_C_EVAL_EXHAUSTED"
        return {
            "status": status,
            "best_id": best.get("id"),
            "best_long_gap": (best.get("long_gap") or {}).get("free_exact"),
            "diag": slim_diag,
            "hashes": hashes,
            "results": [
                {
                    "id": row.get("id"),
                    "verdict": row.get("verdict"),
                    "long_gap": (row.get("long_gap") or {}).get("free_exact"),
                    "induction": (row.get("primitive_induction") or {}).get("first_top1"),
                    "keyed": (row.get("primitive_keyed") or {}).get("first_top1"),
                    "lesson": row.get("lesson"),
                }
                for row in results
            ],
            "promoted": False,
            "test_opened": False,
            "a1_preserved": True,
            "b1_preserved": True,
        }
    finally:
        runtime.remove()
        if handle is not None:
            handle.remove()
        overwrite.gen_index = None
        overwrite.set_eval_write(WRITE_GATED, locator=LOCATOR_SLOT)


def _stop(status: str, results: list[dict], report: dict, hashes: dict) -> dict:
    return {
        "status": status,
        "stop_id": report.get("id"),
        "long_gap": (report.get("long_gap") or {}).get("free_exact"),
        "induction": (report.get("primitive_induction") or {}).get("first_top1"),
        "lesson": report.get("lesson"),
        "hashes": hashes,
        "results": results,
        "promoted": False,
        "test_opened": False,
        "a1_preserved": True,
        "b1_preserved": True,
        "owner_decision_needed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "diagnose", "canary", "pretest", "c3"])
    parser.add_argument("--locator", default=LOCATOR_L0, choices=list(LOCATORS))
    parser.add_argument("--write", default=WRITE_HARD, choices=list(WRITE_MODES) + [WRITE_ADD])
    parser.add_argument("--conf-tau", type=float, default=0.3)
    parser.add_argument("--attn-layer", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy", default=POLICY_A1)
    parser.add_argument("--sidecar", default="250001", choices=["250001", "250002"])
    args = parser.parse_args()
    if args.action == "run":
        print(json.dumps(run_autonomous_c(args.device), default=str), flush=True)
        return
    hashes = verify_locked_hashes()
    device = resolve_device(args.device)
    model, overwrite, _ckpt = load_locator(device, sidecar=args.sidecar)
    handle, _ = attach_overwrite(model, overwrite)
    runtime = LocatorRuntime(model, overwrite).install()
    try:
        overwrite.set_eval_write(
            args.write, conf_tau=args.conf_tau, locator=args.locator, attn_layer=args.attn_layer
        )
        if args.action == "diagnose":
            diag = diagnose_c_pointer(model, overwrite, _long_gap_items(), device, args.policy)
            write(OUT / "c_locator_diag.json", {k: v for k, v in diag.items() if k != "rows"})
            print(json.dumps({"hashes": hashes, **{k: v for k, v in diag.items() if k != "rows"}}, default=str), flush=True)
        elif args.action == "pretest":
            pre = run_pretest(model, overwrite, device, policy=args.policy)
            print(json.dumps(pre, default=str), flush=True)
        elif args.action == "c3":
            handle.remove()
            handle = None
            runtime.remove()
            print(json.dumps(_strip(run_c3_content(model, overwrite, device)), default=str), flush=True)
        else:
            report = run_c_candidate(
                model,
                overwrite,
                device,
                locator=args.locator,
                mode=args.write,
                conf_tau=args.conf_tau,
                attn_layer=args.attn_layer,
                policy=args.policy,
            )
            print(json.dumps(_strip(report), default=str), flush=True)
    finally:
        runtime.remove()
        if handle is not None:
            handle.remove()
        overwrite.gen_index = None
        overwrite.set_eval_write(WRITE_GATED, locator=LOCATOR_SLOT)


if __name__ == "__main__":
    main()
