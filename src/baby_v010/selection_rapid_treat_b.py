"""Eval-only B-series writes on top of A1 routing.

Does not mutate frozen P11 files or the A1 default path. U16000 Baby is
never replaced. Treatments are selectable write operators; A1 gated P11
write remains reproducible by leaving write_mode at gated / T=1 / β=0.
Never uses item["query_position"] as the production write source.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from .residual_overwrite import LocalSlotOverwrite, attach_overwrite
from .selection_p11 import GATE_BIAS
from .selection_p11_u16000_runtime import (
    ISOLATION,
    TREATMENT_SHA,
    adjudicate,
    load_u16000_with_learned_overwrite,
    resolve_device,
    slim_arm,
)
from .selection_rapid_treat import (
    INDUCTION_DROP_BAR,
    KEYED_MIN_TOP1,
    OFF_INDUCTION_TOP1,
    OFF_LONG_FREE,
    OUT,
    POLICY_A1,
    FROZEN_FIRSTSTEP_ON,
    _long_gap_items,
    _panel_slice,
    ledger_append,
    load_off_baseline,
    measure_arm_routed,
    measure_stage1,
    isolation_snapshot,
    score_items_routed,
    score_long_gap,
    should_set_gen_index,
    stage2_from_on,
)
from .selection_s1 import write
from .evaluate import summarize

WRITE_GATED = "gated"
WRITE_HARD = "hard_replace"
WRITE_SHARP = "temperature"
WRITE_MIX = "mix"
WRITE_CONF = "conf_hard"
WRITE_MODES = (WRITE_GATED, WRITE_HARD, WRITE_SHARP, WRITE_MIX, WRITE_CONF)

A1_LONG_FREE = FROZEN_FIRSTSTEP_ON  # 102
A1_QUERY_SWAP_NOVEL_TOP1 = 0.4270833333333333
Q0R_LONG_FREE = 127
ADVANCE_LONG = 110
OWNER_LONG = 115
PRETEST_N = 40
PRETEST_KILL_MAX = 12  # 12/40 ≈ 0.30, at/below OFF rate
POINTER_DIAG_N = 215
QUERY_SWAP_LIFT = 0.04  # ~4 hits / 96 vs A1 0.427


class TreatedOverwrite(LocalSlotOverwrite):
    """P11 slot scorer with selectable eval writes.

    Production source is argmax of causal slot scores at gen, never gold
    query_position. Induction exemption is the caller's gen_index (A1).
    """

    def __init__(self, d_model: int, gate_bias: float = -4.0, gen_only: bool = False) -> None:
        super().__init__(d_model, gate_bias=gate_bias, gen_only=gen_only)
        self.write_mode = WRITE_GATED
        self.score_temperature = 1.0
        self.mix_beta = 0.0
        self.conf_tau = 0.5
        self.last_src_index: torch.Tensor | None = None
        self.last_pointer_mass: torch.Tensor | None = None
        self.last_second_mass: torch.Tensor | None = None

    def set_eval_write(
        self,
        mode: str = WRITE_GATED,
        temperature: float = 1.0,
        mix_beta: float = 0.0,
        conf_tau: float = 0.5,
    ) -> None:
        if mode not in WRITE_MODES:
            raise ValueError(f"unknown write_mode {mode}")
        self.write_mode = mode
        self.score_temperature = float(temperature)
        self.mix_beta = float(mix_beta)
        self.conf_tau = float(conf_tau)

    def set_gen_index_routed(self, items, device, policy: str = POLICY_A1) -> None:
        idxs = []
        for item in items:
            if should_set_gen_index(item, policy):
                idxs.append(len(item["input"]) - 1)
            else:
                idxs.append(-1)
        self.gen_index = torch.tensor(idxs, device=device, dtype=torch.long)

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

        src = None
        if bool(valid.any()):
            raw_scores = slot.unsqueeze(1).expand(batch, time, time).masked_fill(~causal, float("-inf"))
            gen_scores = raw_scores[torch.arange(batch, device=hidden.device), clamped]
            src = gen_scores.argmax(dim=-1)
            self.last_src_index = src.detach()
            b_ix = torch.arange(batch, device=hidden.device)
            attn_gen = attn[b_ix, clamped]
            top2 = attn_gen.topk(k=min(2, time), dim=-1)
            self.last_pointer_mass = top2.values[:, 0].detach()
            if top2.values.shape[-1] > 1:
                self.last_second_mass = top2.values[:, 1].detach()
            else:
                self.last_second_mass = hidden.new_zeros(batch)
        else:
            self.last_src_index = None
            self.last_pointer_mass = None
            self.last_second_mass = None

        if not bool(valid.any()):
            rewritten = hidden if self.gen_only else gated_write
            self.last_hidden = rewritten
            return rewritten

        hard = hidden.clone()
        b_ix = torch.arange(batch, device=hidden.device)
        write_b = b_ix[valid]
        hard[write_b, clamped[valid]] = hidden[write_b, src[valid]]

        mode = self.write_mode
        if mode == WRITE_HARD:
            rewritten = hard
        elif mode == WRITE_MIX:
            beta = float(self.mix_beta)
            rewritten = (1.0 - beta) * gated_write + beta * hard
        elif mode == WRITE_CONF:
            peak = self.last_pointer_mass
            use_hard = (peak >= float(self.conf_tau)) & valid
            rewritten = gated_write.clone()
            rewritten[use_hard] = hard[use_hard]
        else:
            rewritten = gated_write
        self.last_hidden = rewritten
        return rewritten


def attach_overwrite_at(model, module, block_index: int = 0):
    bucket: dict = {}

    def hook(_block, _inputs, output):
        rewritten = module(output)
        bucket["h"] = rewritten
        return rewritten

    handle = model.blocks[block_index].register_forward_hook(hook)
    return handle, bucket


def load_treated(device) -> tuple:
    model, base, ckpt = load_u16000_with_learned_overwrite(device)
    d_model = int(base.gate.in_features)
    treated = TreatedOverwrite(d_model, gate_bias=GATE_BIAS, gen_only=True).to(device)
    treated.load_state_dict(base.state_dict())
    treated.eval()
    treated.set_eval_write(WRITE_GATED, temperature=1.0, mix_beta=0.0)
    return model, treated, ckpt


def window_query_position(item: dict, window_len: int) -> int | None:
    """Diagnostic only. Maps gold query_position into the decode window."""
    if "query_position" not in item:
        return None
    generated = len(item["input"])
    offset = generated - window_len
    qpos = int(item["query_position"]) - offset
    if qpos < 0 or qpos >= window_len:
        return None
    return qpos


@torch.no_grad()
def diagnose_pointer(model, overwrite: TreatedOverwrite, items: list[dict], device, policy: str = POLICY_A1) -> dict:
    rows = []
    for item in items:
        generated = [int(t) for t in item["input"]]
        window = generated[-256:]
        activate = bool(should_set_gen_index(item, policy))
        overwrite.gen_index = (
            torch.tensor([len(window) - 1], device=device, dtype=torch.long) if activate else None
        )
        tokens = torch.tensor([window], dtype=torch.long, device=device)
        model(tokens)
        q_win = window_query_position(item, len(window))
        src = int(overwrite.last_src_index[0].item()) if overwrite.last_src_index is not None else None
        attn = overwrite.last_attn
        gen_t = len(window) - 1
        mass_src = float(attn[0, gen_t, src].item()) if src is not None and attn is not None else None
        mass_q = float(attn[0, gen_t, q_win].item()) if q_win is not None and attn is not None else None
        mass_second = (
            float(overwrite.last_second_mass[0].item()) if overwrite.last_second_mass is not None else None
        )
        gate = float(overwrite.last_gate[0, gen_t].item()) if overwrite.last_gate is not None else None
        rows.append(
            {
                "body_id": item.get("body_id"),
                "armed": activate,
                "src": src,
                "query_position": item.get("query_position"),
                "query_window": q_win,
                "match": bool(src is not None and q_win is not None and src == q_win),
                "mass_src": mass_src,
                "mass_query": mass_q,
                "mass_second": mass_second,
                "gate": gate,
            }
        )
    armed = [row for row in rows if row["armed"]]
    matched = [row for row in armed if row["match"]]
    masses_q = [row["mass_query"] for row in armed if row["mass_query"] is not None]
    masses_s = [row["mass_src"] for row in armed if row["mass_src"] is not None]
    masses_2 = [row["mass_second"] for row in armed if row["mass_second"] is not None]
    gates = [row["gate"] for row in armed if row["gate"] is not None]
    matched_mass = [row["mass_src"] for row in matched if row["mass_src"] is not None]
    miss = [row for row in armed if not row["match"]]
    miss_mass = [row["mass_src"] for row in miss if row["mass_src"] is not None]
    return {
        "n": len(rows),
        "n_armed": len(armed),
        "n_match_query": len(matched),
        "match_rate": (len(matched) / len(armed)) if armed else 0.0,
        "median_mass_query": statistics.median(masses_q) if masses_q else None,
        "median_mass_src": statistics.median(masses_s) if masses_s else None,
        "median_mass_second": statistics.median(masses_2) if masses_2 else None,
        "median_mass_src_match": statistics.median(matched_mass) if matched_mass else None,
        "median_mass_src_miss": statistics.median(miss_mass) if miss_mass else None,
        "median_gate": statistics.median(gates) if gates else None,
        "rows": rows,
    }


def score_query_swap(model, overwrite, device, policy: str = POLICY_A1) -> dict | None:
    if not ISOLATION.exists():
        return None
    items = json.loads(ISOLATION.read_text(encoding="utf-8")).get("query_swap_same_surface_novel")
    if not items:
        return None
    scored = []
    for start in range(0, len(items), 16):
        scored.extend(
            score_items_routed(
                model,
                items[start : start + 16],
                device,
                overwrite=overwrite,
                first_answer_only=True,
                policy=policy,
            )
        )
    summary = summarize(scored)
    summary["n_overwrite_armed"] = sum(int(should_set_gen_index(item, policy)) for item in items)
    summary["a1_first_top1"] = A1_QUERY_SWAP_NOVEL_TOP1
    summary["delta_vs_a1"] = summary["first_top1"] - A1_QUERY_SWAP_NOVEL_TOP1
    return summary


def candidate_tag(mode: str, temperature: float, mix_beta: float, block_index: int, conf_tau: float = 0.5) -> str:
    if mode == WRITE_HARD:
        return "b1_hard_replace"
    if mode == WRITE_SHARP:
        ttag = str(temperature).replace(".", "p")
        return f"b2_temp_{ttag}"
    if mode == WRITE_MIX:
        btag = str(mix_beta).replace(".", "p")
        return f"b3_mix_b{btag}"
    if mode == WRITE_CONF:
        ttag = str(conf_tau).replace(".", "p")
        return f"b7_conf_{ttag}"
    if block_index != 0:
        return f"b5_block{block_index}_{mode}"
    return "b0_gated_a1"


def b_canary_verdict(report: dict) -> tuple[str, str]:
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
    if long_on >= ADVANCE_LONG:
        return "ADVANCE", f"long-gap {long_on}/{n_long} CI-worthy vs A1 {A1_LONG_FREE}"
    if long_on > A1_LONG_FREE:
        return "ADVANCE", f"long-gap {long_on}/{n_long} exceeds A1 {A1_LONG_FREE}"
    if long_on == A1_LONG_FREE and bind_up:
        return "ADVANCE", "long-gap held at A1 102 and query-swap bind clearly up"
    if long_on == A1_LONG_FREE:
        return "KILL", "long-gap held at A1 102 with no bind gain"
    return "KILL", f"long-gap {long_on}/{n_long} below A1 {A1_LONG_FREE} without bind rescue"


def _strip(report: dict) -> dict:
    return {k: v for k, v in report.items() if k not in {"long_gap_rows", "pointer_rows"}}


def run_pretest(
    model,
    overwrite: TreatedOverwrite,
    device,
    *,
    policy: str = POLICY_A1,
    n: int = PRETEST_N,
) -> dict:
    items = _long_gap_items()[:n]
    long_gap = score_long_gap(
        model, items, device, overwrite=overwrite, first_answer_only=True, policy=policy
    )
    induction = _panel_slice("primitive_induction")
    ind_scored = []
    for start in range(0, len(induction), 16):
        ind_scored.extend(
            score_items_routed(
                model,
                induction[start : start + 16],
                device,
                overwrite=overwrite,
                first_answer_only=True,
                policy=policy,
            )
        )
    ind_sum = summarize(ind_scored)
    kill = int(long_gap["free_exact"]) <= PRETEST_KILL_MAX or (
        float(ind_sum["first_top1"]) + 1e-12 < (OFF_INDUCTION_TOP1 - INDUCTION_DROP_BAR)
    )
    return {
        "long_gap": {k: v for k, v in long_gap.items() if k != "rows"},
        "primitive_induction": {
            "n": ind_sum["n"],
            "first_top1": ind_sum["first_top1"],
            "n_overwrite_armed": 0,
        },
        "kill": kill,
        "lesson": (
            "pretest collapsed"
            if kill
            else "pretest not catastrophic; continue to 215"
        ),
    }


def run_candidate(
    model,
    overwrite: TreatedOverwrite,
    device,
    *,
    mode: str,
    temperature: float = 1.0,
    mix_beta: float = 0.0,
    conf_tau: float = 0.5,
    policy: str = POLICY_A1,
    pretest: bool = True,
    score_swap: bool = True,
    pointer_n: int = POINTER_DIAG_N,
    id_override: str | None = None,
) -> dict:
    overwrite.set_eval_write(mode, temperature=temperature, mix_beta=mix_beta, conf_tau=conf_tau)
    tag = id_override or candidate_tag(mode, temperature, mix_beta, 0, conf_tau=conf_tau)
    t0 = time.time()
    long_items = _long_gap_items()
    pointer = diagnose_pointer(model, overwrite, long_items[:pointer_n], device, policy)
    print(
        json.dumps(
            {
                "phase": "pointer",
                "id": tag,
                "match_rate": pointer["match_rate"],
                "median_mass_query": pointer["median_mass_query"],
                "median_mass_src_match": pointer.get("median_mass_src_match"),
                "median_mass_src_miss": pointer.get("median_mass_src_miss"),
                "median_gate": pointer["median_gate"],
            }
        ),
        flush=True,
    )
    if pretest:
        pre = run_pretest(model, overwrite, device, policy=policy)
        print(json.dumps({"phase": "pretest", "id": tag, **{k: v for k, v in pre.items() if k != "long_gap"}}, default=str), flush=True)
        print(
            json.dumps(
                {
                    "phase": "pretest_long",
                    "id": tag,
                    "free_exact": pre["long_gap"]["free_exact"],
                    "n": pre["long_gap"]["n"],
                }
            ),
            flush=True,
        )
        if pre["kill"]:
            report = {
                "id": tag,
                "policy": policy,
                "write_mode": mode,
                "score_temperature": temperature,
                "mix_beta": mix_beta,
                "conf_tau": conf_tau,
                "change": _change(mode, temperature, mix_beta, 0, conf_tau),
                "verdict": "KILL",
                "lesson": pre["lesson"],
                "pretest": pre,
                "pointer": {k: v for k, v in pointer.items() if k != "rows"},
                "pointer_rows": pointer["rows"],
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

    stage1 = measure_stage1(model, overwrite, device, policy)
    swap = score_query_swap(model, overwrite, device, policy) if score_swap else None
    report = dict(stage1)
    report["id"] = tag
    report["write_mode"] = mode
    report["score_temperature"] = temperature
    report["mix_beta"] = mix_beta
    report["conf_tau"] = conf_tau
    report["change"] = _change(mode, temperature, mix_beta, 0, conf_tau)
    report["pointer"] = {k: v for k, v in pointer.items() if k != "rows"}
    report["pointer_rows"] = pointer["rows"]
    report["query_swap_same_surface_novel"] = swap
    report["used_gold_query_position_as_source"] = False
    report["a1_long_gap_free_exact"] = A1_LONG_FREE
    report["q0r_ceiling"] = Q0R_LONG_FREE
    verdict, lesson = b_canary_verdict(report)
    report["verdict"] = verdict
    report["lesson"] = lesson
    report["elapsed_s"] = time.time() - t0
    write(OUT / f"{tag}_stage1.json", _strip(report))
    ledger_append(
        {
            **report,
            "lesson": (
                f"{lesson}; pointer_match={pointer['match_rate']:.3f} "
                f"mass_q={pointer['median_mass_query']}; "
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


def _change(mode: str, temperature: float, mix_beta: float, block_index: int, conf_tau: float = 0.5) -> str:
    base = "A1 induction-exempt first-step; "
    if mode == WRITE_HARD:
        return base + "B1 hard replace h0[gen]:=h0[argmax causal slot scores] (not query_position)"
    if mode == WRITE_SHARP:
        return base + f"B2 temperature-sharpened P11 write T={temperature}"
    if mode == WRITE_MIX:
        return base + f"B3 mix (1-β)·P11 + β·hard_replace β={mix_beta}"
    if mode == WRITE_CONF:
        return base + f"B7 hard replace only if pointer peak mass≥{conf_tau}, else P11 gated"
    if block_index != 0:
        return base + f"B5 write at block {block_index}"
    return base + "gated P11 write (A1 default)"


def is_owner_stop(report: dict) -> bool:
    return report.get("verdict") == "OWNER"


def run_stage2_treated(model, overwrite: TreatedOverwrite, device, stage1: dict, policy: str = POLICY_A1):
    """Retention battery. Does not open TEST. Uses live ON vs frozen OFF baseline."""
    print(json.dumps({"phase": "stage2_start", "id": stage1.get("id")}), flush=True)
    overwrite.set_eval_write(
        stage1.get("write_mode", WRITE_HARD),
        temperature=float(stage1.get("score_temperature") or 1.0),
        mix_beta=float(stage1.get("mix_beta") or 0.0),
        conf_tau=float(stage1.get("conf_tau") or 0.5),
    )
    on = measure_arm_routed(model, device, overwrite, policy, first_answer_only=True, skip_long_gap=True)
    on["long_gap"] = stage1["long_gap"]
    off = load_off_baseline()
    stage2 = stage2_from_on(off, on)
    iso = isolation_snapshot(off, on)
    report = {
        "id": f"{stage1.get('id')}_stage2",
        "policy": policy,
        "write_mode": stage1.get("write_mode"),
        "change": stage1.get("change"),
        "verdict": stage2["verdict"],
        "lesson": stage2["lesson"],
        "retention": stage2["retention"],
        "long_gap": stage1["long_gap"],
        "primitive_induction": on["frozen"]["summaries"]["primitive_induction"],
        "primitive_keyed": stage2["primitive_keyed"],
        "short_keyed": stage2["short_keyed"],
        "rest_lock": stage2["rest_lock"],
        "value_absent": stage2["value_absent"],
        "broken_context": stage2["broken_context"],
        "broken_order": stage2["broken_order"],
        "language_dev_ce": stage2["language_dev_ce"],
        "same_surface_novel": stage2["same_surface_novel"],
        "isolation": iso,
        "protected_material_opened": False,
        "promoted": False,
        "test_opened": False,
    }
    write(OUT / f"{stage1.get('id')}_stage2.json", report)
    ledger_append(report)
    print(
        json.dumps(
            {
                "phase": "stage2_done",
                "id": report["id"],
                "verdict": report["verdict"],
                "short_keyed": stage2["short_keyed"].get("first_top1"),
                "rest_lock": stage2["rest_lock"],
                "query_swap": (iso.get("query_swap_same_surface_novel") or {}).get("on_first_top1"),
            },
            default=str,
        ),
        flush=True,
    )
    return report, on


def run_stage3_treated(model, overwrite: TreatedOverwrite, device, stage1: dict, on: dict | None = None, policy: str = POLICY_A1) -> dict:
    print(json.dumps({"phase": "stage3_start", "id": stage1.get("id")}), flush=True)
    overwrite.set_eval_write(
        stage1.get("write_mode", WRITE_HARD),
        temperature=float(stage1.get("score_temperature") or 1.0),
        mix_beta=float(stage1.get("mix_beta") or 0.0),
        conf_tau=float(stage1.get("conf_tau") or 0.5),
    )
    if on is None:
        on = measure_arm_routed(model, device, overwrite, policy, first_answer_only=True, skip_long_gap=True)
    if "rows" not in (stage1.get("long_gap") or {}) and "long_gap_rows" not in stage1:
        long_on = score_long_gap(
            model, _long_gap_items(), device, overwrite=overwrite, first_answer_only=True, policy=policy
        )
        on["long_gap"] = {k: v for k, v in long_on.items() if k != "rows"}
        on["long_gap_rows"] = long_on["rows"]
    else:
        on["long_gap"] = stage1["long_gap"]
        on["long_gap_rows"] = stage1.get("long_gap_rows") or []
        if not on["long_gap_rows"]:
            long_on = score_long_gap(
                model, _long_gap_items(), device, overwrite=overwrite, first_answer_only=True, policy=policy
            )
            on["long_gap"] = {k: v for k, v in long_on.items() if k != "rows"}
            on["long_gap_rows"] = long_on["rows"]
    off = dict(load_off_baseline())
    off_long = score_long_gap(
        model, _long_gap_items(), device, overwrite=overwrite, first_answer_only=False, policy=policy
    )
    off["long_gap"] = {k: v for k, v in off_long.items() if k != "rows"}
    off["long_gap_rows"] = off_long["rows"]
    decision = adjudicate(off, on)
    decision["policy"] = policy
    decision["write_mode"] = stage1.get("write_mode")
    decision["routing"] = stage1.get("change")
    decision["protocol"] = "RAPID_TREAT_B_STAGE3_P11_RUNTIME_ADAPTED"
    tag = stage1.get("id", "treated")
    write(OUT / f"{tag}_OFF.json", slim_arm(off))
    write(OUT / f"{tag}_ON.json", slim_arm(on))
    write(OUT / f"{tag}_stage3.json", decision)
    ledger_append(
        {
            "id": f"{tag}_stage3",
            "policy": policy,
            "change": str(stage1.get("change")) + " ; full P11-runtime battery",
            "long_gap": {
                "free_exact": decision["long_gap"]["on_free_exact"],
                "n": decision["n_long"],
                "n_overwrite_armed": on["long_gap"].get("n_overwrite_armed"),
            },
            "primitive_induction": on["frozen"]["summaries"]["primitive_induction"],
            "primitive_keyed": on["frozen"]["summaries"]["primitive_keyed"],
            "verdict": decision["verdict"],
            "lesson": f"cause={decision['cause']}; benefit_ok={decision['benefit_ok']}; "
            f"on_retention={decision['on_retention_pass']}; ci={decision['long_gap']['bootstrap_ci95_on_minus_off']}",
        }
    )
    print(
        json.dumps(
            {
                "phase": "stage3_done",
                "id": tag,
                "verdict": decision["verdict"],
                "long_gap": decision["long_gap"],
                "cause": decision["cause"],
            }
        ),
        flush=True,
    )
    return decision


def run_b5_block1(model, overwrite: TreatedOverwrite, device, policy: str = POLICY_A1) -> dict:
    overwrite.set_eval_write(WRITE_GATED, temperature=1.0, mix_beta=0.0)
    handle, _ = attach_overwrite_at(model, overwrite, block_index=1)
    tag = "b5_block1_gated"
    try:
        pre = run_pretest(model, overwrite, device, policy=policy)
        report = {
            "id": tag,
            "policy": policy,
            "write_mode": WRITE_GATED,
            "block_index": 1,
            "change": _change(WRITE_GATED, 1.0, 0.0, 1),
            "pretest": pre,
            "long_gap": pre["long_gap"],
            "primitive_induction": pre["primitive_induction"],
            "primitive_keyed": {"first_top1": None, "n": 0},
            "used_gold_query_position_as_source": False,
            "protected_material_opened": False,
            "promoted": False,
            "test_opened": False,
        }
        if pre["kill"] or int(pre["long_gap"]["free_exact"]) <= 16:
            report["verdict"] = "KILL"
            report["lesson"] = "B5 block-1 L0-trained sidecar is NULL or collapsed"
        else:
            stage1 = measure_stage1(model, overwrite, device, policy)
            report.update(stage1)
            report["id"] = tag
            report["change"] = _change(WRITE_GATED, 1.0, 0.0, 1)
            report["verdict"], report["lesson"] = b_canary_verdict(report)
        write(OUT / f"{tag}_stage1.json", _strip(report))
        ledger_append(report)
        print(json.dumps({"phase": "b5_done", "id": tag, "verdict": report["verdict"]}), flush=True)
        return report
    finally:
        handle.remove()


def run_autonomous_b(device_name: str = "cuda") -> dict:
    device = resolve_device(device_name)
    model, overwrite, _ckpt = load_treated(device)
    handle, _ = attach_overwrite(model, overwrite)
    results = []
    try:
        print(json.dumps({"phase": "start", "device": str(device), "sha": TREATMENT_SHA}), flush=True)

        b1 = run_candidate(model, overwrite, device, mode=WRITE_HARD)
        results.append(_strip(b1))
        if is_owner_stop(b1):
            return _stop("OWNER_B1", results, b1)

        b1_hits = int(b1.get("long_gap", {}).get("free_exact") or 0)
        b1_collapsed = b1_hits <= OFF_LONG_FREE + 4
        # Mixed: hard replace moved hits but did not win or collapse — β-blend may help.
        b1_mixed = (not b1_collapsed) and b1.get("verdict") not in {"ADVANCE", "OWNER"} and b1_hits != A1_LONG_FREE

        temps = [0.5, 0.25, 0.1]
        if b1_collapsed:
            temps = [0.5]
        for temp in temps:
            b2 = run_candidate(model, overwrite, device, mode=WRITE_SHARP, temperature=temp)
            results.append(_strip(b2))
            if is_owner_stop(b2):
                return _stop("OWNER_B2", results, b2)

        if b1_mixed:
            for beta in (0.5, 1.0):
                if beta == 1.0 and b1.get("write_mode") == WRITE_HARD:
                    # β=1 is B1; skip duplicate full canary, record alias.
                    alias = {
                        "id": "b3_mix_b1p0",
                        "verdict": b1.get("verdict"),
                        "lesson": "alias of B1 hard replace; skipped duplicate",
                        "long_gap": b1.get("long_gap"),
                    }
                    results.append(alias)
                    ledger_append(
                        {
                            "id": "b3_mix_b1p0",
                            "policy": POLICY_A1,
                            "change": _change(WRITE_MIX, 1.0, 1.0, 0),
                            "long_gap": b1.get("long_gap") or {},
                            "primitive_induction": b1.get("primitive_induction") or {},
                            "primitive_keyed": b1.get("primitive_keyed") or {},
                            "verdict": b1.get("verdict"),
                            "lesson": "β=1.0 is B1; not rerun",
                        }
                    )
                    continue
                b3 = run_candidate(model, overwrite, device, mode=WRITE_MIX, mix_beta=beta)
                results.append(_strip(b3))
                if is_owner_stop(b3):
                    return _stop("OWNER_B3", results, b3)

        pointer_rate = float((b1.get("pointer") or {}).get("match_rate") or 0.0)
        survivors = [r for r in results if r.get("verdict") in {"ADVANCE", "OWNER"}]
        eval_best = max(
            (int((r.get("long_gap") or {}).get("free_exact") or 0) for r in results),
            default=0,
        )

        if not survivors and pointer_rate < 0.60:
            print(
                json.dumps(
                    {
                        "phase": "b4_licensed",
                        "reason": "eval-only exhausted or pointer-limited",
                        "pointer_match": pointer_rate,
                        "eval_best": eval_best,
                    }
                ),
                flush=True,
            )
            handle.remove()
            handle = None
            b4 = run_b4(model, overwrite, device)
            results.append(_strip(b4) if isinstance(b4, dict) else {"id": "b4", "verdict": "BLOCKED"})
            if is_owner_stop(b4):
                return _stop("OWNER_B4", results, b4)
            handle, _ = attach_overwrite(model, overwrite)
        elif not survivors:
            print(
                json.dumps(
                    {
                        "phase": "b4_skip",
                        "reason": "pointer not the bottleneck",
                        "pointer_match": pointer_rate,
                    }
                ),
                flush=True,
            )

        if not survivors:
            if handle is not None:
                handle.remove()
                handle = None
            b5 = run_b5_block1(model, overwrite, device)
            results.append(_strip(b5))
            if is_owner_stop(b5):
                return _stop("OWNER_B5", results, b5)

        best = max(results, key=lambda r: int((r.get("long_gap") or {}).get("free_exact") or -1))
        status = "SURVIVOR" if any(r.get("verdict") in {"ADVANCE", "OWNER"} for r in results) else "QUEUE_B_EVAL_EXHAUSTED"
        return {
            "status": status,
            "best_id": best.get("id"),
            "best_long_gap": (best.get("long_gap") or {}).get("free_exact"),
            "pointer_match_b1": pointer_rate,
            "results": [
                {
                    "id": r.get("id"),
                    "verdict": r.get("verdict"),
                    "long_gap": (r.get("long_gap") or {}).get("free_exact"),
                    "induction": (r.get("primitive_induction") or {}).get("first_top1"),
                    "lesson": r.get("lesson"),
                }
                for r in results
            ],
            "promoted": False,
            "test_opened": False,
            "a1_preserved": True,
        }
    finally:
        if handle is not None:
            handle.remove()
        overwrite.gen_index = None
        overwrite.set_eval_write(WRITE_GATED, 1.0, 0.0)


def _stop(status: str, results: list[dict], report: dict) -> dict:
    return {
        "status": status,
        "stop_id": report.get("id"),
        "long_gap": (report.get("long_gap") or {}).get("free_exact"),
        "induction": (report.get("primitive_induction") or {}).get("first_top1"),
        "lesson": report.get("lesson"),
        "results": results,
        "promoted": False,
        "test_opened": False,
        "a1_preserved": True,
        "owner_decision_needed": True,
    }


def run_b4(model, overwrite: TreatedOverwrite, device) -> dict:
    """Train a stronger first-step scorer with A1 routing. Freeze Baby.

    Induction never gets gen_index. Gold query_position is aux-only.
    """
    from .selection_p11 import OUT as P11_OUT, TRAIN_SEED, PTR_LAMBDA, GATE_LAMBDA, GATE_ON_AFTER, OVERWRITE_LR
    from .selection_p4 import copy_specs
    from .residual_overwrite import pointer_aux
    from . import selection_s1
    from .selection_s1 import pack

    schedule_path = P11_OUT / f"SCHEDULE_{TRAIN_SEED}.json"
    if not schedule_path.exists():
        report = {
            "id": "b4_stronger_scorer",
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

    dest = OUT / "b4b_longgap_ptr"
    dest.mkdir(parents=True, exist_ok=True)
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trained = TreatedOverwrite(int(overwrite.gate.in_features), gate_bias=GATE_BIAS, gen_only=True).to(device)
    trained.load_state_dict(overwrite.state_dict())
    trained.train()
    trained.set_eval_write(WRITE_GATED, 1.0, 0.0)
    optimizer = torch.optim.AdamW(trained.parameters(), lr=OVERWRITE_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    handle, _ = attach_overwrite(model, trained)
    previous_hook = selection_s1.PACK_HOOK
    selection_s1.PACK_HOOK = lambda items, x, module=trained: module.set_gen_index_routed(items, x.device, POLICY_A1)
    until = 400
    t0 = time.time()
    try:
        last_step = 0
        for step in range(1, until + 1):
            spec = schedule[step - 1]
            if spec["task"] == "language":
                last_step = step
                if step % 200 != 0:
                    continue
            else:
                model.train()
                trained.train()
                trained.set_eval_write(WRITE_GATED, 1.0, 0.0)
                items = spec["items"]
                x, y, mask, _first = pack(items, device)
                hidden = model.forward_hidden(x)
                logits = model.language_head(hidden.detach())
                loss = F.cross_entropy(logits[mask], y[mask])
                specs = copy_specs(items, min_gap=13)
                gate_lam = GATE_LAMBDA if step > GATE_ON_AFTER else 0.0
                extra_ptr, extra_gate = pointer_aux(trained.last_attn, trained.last_gate, specs, use_gate=gate_lam > 0)
                loss = loss + PTR_LAMBDA * extra_ptr
                if gate_lam > 0:
                    loss = loss + gate_lam * extra_gate
                optimizer.zero_grad(set_to_none=True)
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trained.parameters(), 2.0)
                    optimizer.step()
                last_step = step
                if step % 50 == 0:
                    print(
                        json.dumps(
                            {
                                "phase": "b4_train",
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
                trained.set_eval_write(WRITE_HARD, 1.0, 0.0)
                canary = run_candidate(
                    model,
                    trained,
                    device,
                    mode=WRITE_HARD,
                    pretest=True,
                    score_swap=False,
                    pointer_n=POINTER_DIAG_N,
                    id_override=f"b4_step{step}_hard",
                )
                write(dest / f"eval_{step:04d}.json", _strip(canary))
                print(
                    json.dumps(
                        {
                            "phase": "b4_eval",
                            "step": step,
                            "verdict": canary.get("verdict"),
                            "long_gap": (canary.get("long_gap") or {}).get("free_exact"),
                            "pointer": (canary.get("pointer") or {}).get("match_rate"),
                        }
                    ),
                    flush=True,
                )
                if is_owner_stop(canary):
                    torch.save(
                        {"overwrite_state_dict": trained.state_dict(), "step": step, "parent_checkpoint_sha256": TREATMENT_SHA},
                        dest / f"checkpoint_{step}.pt",
                    )
                    return canary
                torch.save(
                    {"overwrite_state_dict": trained.state_dict(), "step": step},
                    dest / f"checkpoint_{step}.pt",
                )
        model.eval()
        trained.eval()
        trained.set_eval_write(WRITE_HARD, 1.0, 0.0)
        final = run_candidate(
            model, trained, device, mode=WRITE_HARD, pretest=False, id_override="b4b_longgap_ptr"
        )
        final["change"] = "B4b overwrite-only, A1 PACK_HOOK, pointer aux gap>=13, hard-replace eval"
        final["train_steps"] = last_step
        final["elapsed_s"] = time.time() - t0
        write(dest / "FINAL.json", _strip(final))
        ledger_append(final)
        return final
    finally:
        selection_s1.PACK_HOOK = previous_hook
        handle.remove()
        overwrite.eval()
        overwrite.set_eval_write(WRITE_GATED, 1.0, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["canary", "pretest", "diagnose", "run", "b5", "stage2"])
    parser.add_argument("--write", default=WRITE_HARD, choices=list(WRITE_MODES))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mix-beta", type=float, default=0.0)
    parser.add_argument("--conf-tau", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy", default=POLICY_A1)
    args = parser.parse_args()
    device = resolve_device(args.device)
    if args.action == "run":
        print(json.dumps(run_autonomous_b(args.device), default=str), flush=True)
        return
    model, overwrite, _ckpt = load_treated(device)
    handle, _ = attach_overwrite(model, overwrite)
    try:
        overwrite.set_eval_write(args.write, temperature=args.temperature, mix_beta=args.mix_beta, conf_tau=args.conf_tau)
        if args.action == "diagnose":
            pointer = diagnose_pointer(model, overwrite, _long_gap_items()[:POINTER_DIAG_N], device, args.policy)
            write(OUT / "b_pointer_diag.json", {k: v for k, v in pointer.items() if k != "rows"})
            print(json.dumps({k: v for k, v in pointer.items() if k != "rows"}, default=str), flush=True)
        elif args.action == "pretest":
            pre = run_pretest(model, overwrite, device, policy=args.policy)
            print(json.dumps(pre, default=str), flush=True)
        elif args.action == "stage2":
            stage1 = json.loads((OUT / f"{candidate_tag(args.write, args.temperature, args.mix_beta, 0, args.conf_tau)}_stage1.json").read_text(encoding="utf-8"))
            s2, _on = run_stage2_treated(model, overwrite, device, stage1, args.policy)
            print(json.dumps(s2, default=str), flush=True)
        elif args.action == "b5":
            handle.remove()
            handle = None
            print(json.dumps(_strip(run_b5_block1(model, overwrite, device, args.policy)), default=str), flush=True)
        else:
            report = run_candidate(
                model,
                overwrite,
                device,
                mode=args.write,
                temperature=args.temperature,
                mix_beta=args.mix_beta,
                conf_tau=args.conf_tau,
                policy=args.policy,
            )
            print(json.dumps(_strip(report), default=str), flush=True)
    finally:
        if handle is not None:
            handle.remove()
        overwrite.gen_index = None
        overwrite.set_eval_write(WRITE_GATED, 1.0, 0.0)


if __name__ == "__main__":
    main()
