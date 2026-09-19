"""Eval-time P11 routing treatments. Does not mutate frozen P11 protocol files.

Loads authoritative U16000 Baby + P11 overwrite_state_dict only. Refuses P11
Baby weights. First-step overwrite only. Never sets gen_index on language_ce.
TEST / FINAL / SACRED stay closed. Does not promote Baby.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .evaluate import language_ce, score_items, summarize
from .residual_overwrite import LocalSlotOverwrite, attach_overwrite
from .selection_p2 import RETENTION_DROP
from .selection_p11 import GATE_BIAS, OUT as P11_OUT
from .selection_p11_decode import MIN_GAP, SUCCESS_DELTA, long_items
from .selection_p11_u16000_runtime import (
    EXPECTED_N_LONG,
    FROZEN_FIRSTSTEP_OFF,
    FROZEN_FIRSTSTEP_ON,
    ISOLATION,
    OUT as P11_RUNTIME_OUT,
    PARENT_SHA,
    SKIP_PANELS,
    S2_DIAGNOSTIC,
    TREATMENT_SHA,
    adjudicate,
    arm_retention,
    load_parent_model,
    load_u16000_with_learned_overwrite,
    resolve_device,
    rest_lock_mean,
    slim_arm,
)
from .selection_s1 import PANEL, digest, write
from .train_v2r4 import DEV_STREAM

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/rapid_treat"
LEDGER = OUT / "LEDGER.md"

POLICY_KIND_BLIND = "kind_blind"
POLICY_A1 = "a1_induction_exempt"
POLICY_A2 = "a2_gap_gated"
POLICY_A3 = "a3_keyed_only"
POLICIES = (POLICY_KIND_BLIND, POLICY_A1, POLICY_A2, POLICY_A3)

OFF_LONG_FREE = FROZEN_FIRSTSTEP_OFF
OFF_INDUCTION_TOP1 = 19 / 64
INDUCTION_DROP_BAR = 0.05
KEYED_MIN_TOP1 = 0.95
STAGE1_LONG_MIN = OFF_LONG_FREE + SUCCESS_DELTA * EXPECTED_N_LONG  # 92.5 → need ≥93
REPLICA_CKPT = P11_OUT / "treatment_250002" / "checkpoint_16800.pt"
REPLICA_SHA = "ac49bf4138b3bda125cd989392e46ee70fbf3d80362dae8323c56fe2ace19629"


def should_set_gen_index(item: dict, policy: str) -> bool:
    """Whether first-step overwrite may write on this row.

    Identity unless the caller also requests first_answer_only.
    """
    if policy == POLICY_KIND_BLIND:
        return True
    if policy == POLICY_A1:
        return item.get("kind") != "induction"
    if policy == POLICY_A2:
        if "query_position" not in item:
            return False
        gap = len(item["input"]) - 1 - int(item["query_position"])
        return gap >= MIN_GAP
    if policy == POLICY_A3:
        return item.get("kind") == "keyed"
    raise ValueError(f"unknown policy {policy}")


def gen_index_for_item(item: dict, policy: str, *, first_answer_only: bool) -> int | None:
    if not first_answer_only or not should_set_gen_index(item, policy):
        return None
    return len(item["input"]) - 1


def greedy_decode(
    model,
    item: dict,
    device,
    n_tokens: int,
    overwrite=None,
    first_answer_only: bool = False,
    policy: str = POLICY_A1,
) -> list[int]:
    generated = [int(t) for t in item["input"]]
    emitted: list[int] = []
    with torch.no_grad():
        for step in range(n_tokens):
            if overwrite is not None:
                activate = (
                    first_answer_only
                    and step == 0
                    and should_set_gen_index(item, policy)
                )
                overwrite.gen_index = (
                    torch.tensor([len(generated) - 1], device=device, dtype=torch.long)
                    if activate
                    else None
                )
            tokens = torch.tensor([generated[-256:]], dtype=torch.long, device=device)
            logits = model(tokens)
            token = int(logits[0, -1].argmax().item())
            emitted.append(token)
            generated.append(token)
    return emitted


def score_items_routed(
    model,
    items: list[dict],
    device: torch.device,
    overwrite=None,
    first_answer_only: bool = False,
    policy: str = POLICY_A1,
) -> list[dict]:
    """score_items with per-row gen_index routing. Splits mixed batches.

    Skipped rows use the frozen identity scorer. Activated rows use first-step
    write. Does not mutate evaluate.py.
    """
    if not items:
        return []
    if not first_answer_only or overwrite is None:
        return score_items(model, items, device, overwrite=overwrite, first_answer_only=False)
    skip_idx = [i for i, item in enumerate(items) if not should_set_gen_index(item, policy)]
    on_idx = [i for i, item in enumerate(items) if should_set_gen_index(item, policy)]
    result: list[dict | None] = [None] * len(items)
    if skip_idx:
        skipped = score_items(
            model,
            [items[i] for i in skip_idx],
            device,
            overwrite=overwrite,
            first_answer_only=False,
        )
        for i, row in zip(skip_idx, skipped):
            result[i] = row
    if on_idx:
        activated = score_items(
            model,
            [items[i] for i in on_idx],
            device,
            overwrite=overwrite,
            first_answer_only=True,
        )
        for i, row in zip(on_idx, activated):
            result[i] = row
    return result  # type: ignore[return-value]


def evaluate_panels_routed(
    model,
    panels: dict,
    device: torch.device,
    overwrite=None,
    first_answer_only: bool = False,
    policy: str = POLICY_A1,
    limit: int | None = None,
) -> dict:
    model.eval()
    summaries: dict[str, dict] = {}
    rows: dict[str, list[dict]] = {}
    for name, items in panels.items():
        if name == "all_intact":
            continue
        selected = items if limit is None else items[:limit]
        scored: list[dict] = []
        for start in range(0, len(selected), 16):
            scored.extend(
                score_items_routed(
                    model,
                    selected[start : start + 16],
                    device,
                    overwrite=overwrite,
                    first_answer_only=first_answer_only,
                    policy=policy,
                )
            )
        rows[name] = scored
        summaries[name] = summarize(scored)
    return {"summaries": summaries, "rows": rows, "protected_material_opened": False}


def score_long_gap(
    model,
    items: list[dict],
    device,
    overwrite=None,
    first_answer_only: bool = False,
    policy: str = POLICY_A1,
) -> dict:
    rows = []
    n_activated = 0
    for item in items:
        target = [int(t) for t in item["target"]]
        activated = bool(first_answer_only and should_set_gen_index(item, policy))
        n_activated += int(activated)
        emitted = greedy_decode(
            model,
            item,
            device,
            len(target),
            overwrite=overwrite,
            first_answer_only=first_answer_only,
            policy=policy,
        )
        pred_head = emitted[0] if emitted else None
        rows.append(
            {
                "body_id": item["body_id"],
                "query_index": item["query_index"],
                "K": item["pair_count"],
                "free_exact": emitted == target,
                "first_correct": pred_head == target[0],
                "inventory_correct": emitted == target,
                "overwrite_armed": activated,
            }
        )
    n = len(rows)
    free = sum(int(row["free_exact"]) for row in rows)
    first = sum(int(row["first_correct"]) for row in rows)
    return {
        "n": n,
        "free_exact": free,
        "free_accuracy": free / n if n else 0.0,
        "first_correct": first,
        "first_accuracy": first / n if n else 0.0,
        "n_overwrite_armed": n_activated,
        "rows": rows,
    }


@torch.no_grad()
def probe_write(model, overwrite, item: dict, device, policy: str, *, first_answer_only: bool = True) -> dict:
    generated = [int(t) for t in item["input"]]
    activate = bool(first_answer_only and should_set_gen_index(item, policy))
    overwrite.gen_index = (
        torch.tensor([len(generated) - 1], device=device, dtype=torch.long) if activate else None
    )
    tokens = torch.tensor([generated[-256:]], dtype=torch.long, device=device)
    logits = model(tokens)
    gate = overwrite.last_gate
    gen = min(len(generated), tokens.shape[1]) - 1
    gate_at_gen = float(gate[0, gen].item()) if gate is not None else 0.0
    wrote = activate and abs(gate_at_gen) > 1e-8
    pred = int(logits[0, gen].argmax().item())
    return {
        "kind": item.get("kind"),
        "armed": activate,
        "wrote": wrote,
        "gate_at_gen": gate_at_gen,
        "pred": pred,
        "target0": int(item["target"][0]) if item.get("target") else None,
    }


def _panel_slice(name: str) -> list[dict]:
    panels = json.loads(PANEL.read_text(encoding="utf-8"))
    return panels[name]


def _long_gap_items() -> list[dict]:
    return long_items(json.loads(S2_DIAGNOSTIC.read_text(encoding="utf-8")))


def _score_named_panel(model, device, overwrite, policy: str, name: str, *, first_answer_only: bool) -> dict:
    items = _panel_slice(name)
    scored = []
    for start in range(0, len(items), 16):
        scored.extend(
            score_items_routed(
                model,
                items[start : start + 16],
                device,
                overwrite=overwrite,
                first_answer_only=first_answer_only,
                policy=policy,
            )
        )
    summary = summarize(scored)
    summary["n_overwrite_armed"] = sum(
        int(first_answer_only and should_set_gen_index(item, policy)) for item in items
    )
    return summary


def stage1_verdict(report: dict) -> tuple[str, str]:
    long_on = report["long_gap"]["free_exact"]
    ind = report["primitive_induction"]["first_top1"]
    keyed = report["primitive_keyed"]["first_top1"]
    long_armed = report["long_gap"]["n_overwrite_armed"]
    ind_armed = report["primitive_induction"]["n_overwrite_armed"]
    mech_long = report["mechanism"]["long_gap"]["wrote"]
    long_ok = long_on + 1e-12 >= STAGE1_LONG_MIN and long_armed == report["long_gap"]["n"] and mech_long
    ind_ok = ind + 1e-12 >= (OFF_INDUCTION_TOP1 - INDUCTION_DROP_BAR) and ind_armed == 0
    keyed_ok = keyed + 1e-12 >= KEYED_MIN_TOP1
    if long_ok and ind_ok and keyed_ok:
        return "ADVANCE", "induction restored; long-gap held; keyed intact; write armed on long-gap only"
    if long_on <= OFF_LONG_FREE + 4 and not mech_long:
        return "KILL", "long-gap benefit disappeared or overwrite never armed"
    if ind < (OFF_INDUCTION_TOP1 - INDUCTION_DROP_BAR) - 1e-12:
        if long_ok:
            return "KILL", "induction still damaged despite exemption"
        return "KILL", "induction damaged and long-gap lost"
    if not long_ok and ind_ok:
        return "KILL", "induction restored but long-gap died — routing skipped keyed write"
    if not keyed_ok:
        return "KILL", "primitive_keyed negative control failed"
    return "KILL", "stage-1 thresholds not met"


def measure_stage1(model, overwrite, device, policy: str) -> dict:
    t0 = time.time()
    long_items_ = _long_gap_items()
    if len(long_items_) != EXPECTED_N_LONG:
        raise RuntimeError(f"long-gap n={len(long_items_)} expected={EXPECTED_N_LONG}")
    mechanism = {
        "long_gap": probe_write(model, overwrite, long_items_[0], device, policy),
        "induction": probe_write(model, overwrite, _panel_slice("primitive_induction")[0], device, policy),
        "keyed": probe_write(model, overwrite, _panel_slice("primitive_keyed")[0], device, policy),
    }
    print(json.dumps({"phase": "stage1_probe", "policy": policy, "mechanism": mechanism}), flush=True)
    long_gap = score_long_gap(
        model, long_items_, device, overwrite=overwrite, first_answer_only=True, policy=policy
    )
    print(
        json.dumps(
            {
                "phase": "stage1_long_gap",
                "policy": policy,
                "free_exact": long_gap["free_exact"],
                "n": long_gap["n"],
                "n_overwrite_armed": long_gap["n_overwrite_armed"],
            }
        ),
        flush=True,
    )
    induction = _score_named_panel(
        model, device, overwrite, policy, "primitive_induction", first_answer_only=True
    )
    keyed = _score_named_panel(
        model, device, overwrite, policy, "primitive_keyed", first_answer_only=True
    )
    report = {
        "id": policy,
        "policy": policy,
        "change": _policy_change(policy),
        "baby_weights": "authoritative_u16000",
        "p11_model_state_dict_loaded": False,
        "parent_sha256": PARENT_SHA,
        "overwrite_checkpoint_sha256": TREATMENT_SHA,
        "first_answer_only": True,
        "long_gap": {key: value for key, value in long_gap.items() if key != "rows"},
        "long_gap_rows": long_gap["rows"],
        "primitive_induction": induction,
        "primitive_keyed": keyed,
        "mechanism": mechanism,
        "baselines": {
            "long_gap_off_free_exact": OFF_LONG_FREE,
            "long_gap_kindblind_on_free_exact": FROZEN_FIRSTSTEP_ON,
            "induction_off_first_top1": OFF_INDUCTION_TOP1,
            "induction_kindblind_on_first_top1": 10 / 64,
        },
        "elapsed_s": time.time() - t0,
        "protected_material_opened": False,
        "promoted": False,
        "test_opened": False,
    }
    verdict, lesson = stage1_verdict(report)
    report["verdict"] = verdict
    report["lesson"] = lesson
    return report


def _policy_change(policy: str) -> str:
    return {
        POLICY_A1: "induction-exempt gen_index: kind==induction identity; else first-step write",
        POLICY_A2: "gap-gated write: query_position present AND gap>=13; else identity",
        POLICY_A3: "keyed-only first_answer_only: kind==keyed write; else identity",
        POLICY_KIND_BLIND: "kind-blind first-step write (P11 collision control)",
    }[policy]


def ledger_append(report: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not LEDGER.exists():
        LEDGER.write_text(
            "# Rapid treat ledger\n\nEval-time P11 routing. U16000 Baby unchanged. TEST closed.\n\n",
            encoding="utf-8",
        )
    long_gap = report.get("long_gap", {})
    ind = report.get("primitive_induction", {})
    keyed = report.get("primitive_keyed", {})
    lines = [
        f"## {report.get('id', report.get('policy', 'unknown'))}",
        "",
        f"- ID: `{report.get('policy')}`",
        f"- Change: {report.get('change', '')}",
        f"- long-gap: ON {long_gap.get('free_exact')}/{long_gap.get('n')} vs OFF {OFF_LONG_FREE}/215 "
        f"(kind-blind ON {FROZEN_FIRSTSTEP_ON}/215); armed={long_gap.get('n_overwrite_armed')}",
        f"- induction: first_top1 {ind.get('first_top1')} n={ind.get('n')} armed={ind.get('n_overwrite_armed')} "
        f"(OFF {OFF_INDUCTION_TOP1:.3f}; kind-blind ON 0.156)",
        f"- primitive_keyed: first_top1 {keyed.get('first_top1')} n={keyed.get('n')}",
        f"- verdict: **{report.get('verdict')}**",
        f"- lesson: {report.get('lesson', '')}",
        "",
    ]
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def measure_arm_routed(
    model,
    device,
    overwrite,
    policy: str,
    *,
    first_answer_only: bool,
    skip_long_gap: bool = False,
) -> dict:
    from .data import read_u16

    diagnostic_items = json.loads(S2_DIAGNOSTIC.read_text(encoding="utf-8"))
    if skip_long_gap:
        long_gap = {"n": 0, "free_exact": 0, "free_accuracy": 0.0, "first_correct": 0, "first_accuracy": 0.0, "n_overwrite_armed": 0, "rows": []}
    else:
        long_gap = score_long_gap(
            model,
            long_items(diagnostic_items),
            device,
            overwrite=overwrite,
            first_answer_only=first_answer_only,
            policy=policy,
        )
    scored = []
    for start in range(0, len(diagnostic_items), 16):
        scored.extend(
            score_items_routed(
                model,
                diagnostic_items[start : start + 16],
                device,
                overwrite=overwrite,
                first_answer_only=first_answer_only,
                policy=policy,
            )
        )
    panels = json.loads(PANEL.read_text(encoding="utf-8"))
    panels = {name: rows for name, rows in panels.items() if name not in SKIP_PANELS}
    frozen = evaluate_panels_routed(
        model,
        panels,
        device,
        overwrite=overwrite,
        first_answer_only=first_answer_only,
        policy=policy,
    )
    isolation = evaluate_panels_routed(
        model,
        json.loads(ISOLATION.read_text(encoding="utf-8")),
        device,
        overwrite=overwrite,
        first_answer_only=first_answer_only,
        policy=policy,
    )
    stream = torch.tensor(read_u16(DEV_STREAM), dtype=torch.long)
    if overwrite is not None:
        overwrite.gen_index = None
    ce = language_ce(model, stream, list(range(0, 32 * 256, 256)), device, limit=32)
    rest_lock = rest_lock_mean(scored, diagnostic_items)
    return {
        "baby_weights": "authoritative_u16000",
        "first_answer_only": first_answer_only,
        "policy": policy,
        "overwrite_attached": overwrite is not None,
        "summary": {"rest_lock": rest_lock},
        "frozen": {"summaries": frozen["summaries"]},
        "isolation": {"summaries": isolation["summaries"]},
        "language_dev_ce": ce,
        "long_gap": {key: value for key, value in long_gap.items() if key != "rows"},
        "long_gap_rows": long_gap["rows"],
        "protected_material_opened": False,
    }


def stage2_from_on(off: dict, on: dict) -> dict:
    retention = arm_retention(off, on)
    frozen_on = on["frozen"]["summaries"]
    frozen_off = off["frozen"]["summaries"]
    novel_off = frozen_off["same_surface_novel"]["free_exact"]
    novel_on = frozen_on["same_surface_novel"]["free_exact"]
    watch = []
    if novel_off - novel_on > RETENTION_DROP:
        watch.append(
            {
                "panel": "same_surface_novel",
                "metric": "free_exact",
                "parent": novel_off,
                "treatment": novel_on,
                "bar_drop": RETENTION_DROP,
                "note": "not in retention_failures; informational only",
            }
        )
    ce_delta = on["language_dev_ce"] - off["language_dev_ce"]
    verdict = "ADVANCE" if retention["pass"] else "KILL"
    lesson = "retention bars held" if retention["pass"] else "retention or negative-control fail"
    return {
        "id": f"{on.get('policy', 'routed')}_stage2",
        "policy": on.get("policy"),
        "verdict": verdict,
        "lesson": lesson,
        "retention": retention,
        "same_surface_novel": {
            "off_free_exact": novel_off,
            "on_free_exact": novel_on,
            "drop": novel_off - novel_on,
        },
        "same_surface_novel_watch": watch,
        "language_dev_ce": {"off": off["language_dev_ce"], "on": on["language_dev_ce"], "delta": ce_delta},
        "rest_lock": {"off": off["summary"]["rest_lock"], "on": on["summary"]["rest_lock"]},
        "primitive_keyed": frozen_on["primitive_keyed"],
        "short_keyed": frozen_on["short_keyed"],
        "value_absent": on["isolation"]["summaries"]["value_absent_same_surface_novel"],
        "broken_context": frozen_on["broken_context"],
        "broken_order": frozen_on["broken_order"],
        "protected_material_opened": False,
        "promoted": False,
        "test_opened": False,
    }


def load_off_baseline() -> dict:
    path = P11_RUNTIME_OUT / "OFF.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_u16000_with_sidecar(device, ckpt_path: Path, expected_sha: str):
    """U16000 Baby + named overwrite sidecar. Refuses P11 Baby weights."""
    if digest(ckpt_path) != expected_sha:
        raise RuntimeError(f"overwrite checkpoint hash mismatch: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("parent_checkpoint_sha256") != PARENT_SHA:
        raise RuntimeError("overwrite checkpoint parent mismatch")
    if ckpt.get("protected_material_opened"):
        raise RuntimeError("P11 checkpoint opened protected material")
    model, config, parent_ckpt = load_parent_model(device)
    delta = 0.0
    for key, tensor in parent_ckpt["model_state_dict"].items():
        p11_tensor = ckpt["model_state_dict"][key]
        delta = max(delta, float((tensor - p11_tensor).abs().max().item()))
    if delta > 0.0:
        raise RuntimeError(
            f"sidecar model_state_dict differs from U16000 (max abs delta {delta}); "
            "refusing to load P11 Baby weights"
        )
    overwrite = LocalSlotOverwrite(config.d_model, gate_bias=GATE_BIAS, gen_only=True).to(device)
    overwrite.load_state_dict(ckpt["overwrite_state_dict"])
    overwrite.eval()
    return model, overwrite, ckpt


def isolation_snapshot(off: dict, on: dict) -> dict:
    names = (
        "query_swap_same_surface_novel",
        "query_swap_short_keyed",
        "body_reorder_query_first_same_surface_novel",
        "body_reorder_query_last_same_surface_novel",
        "body_reorder_query_first_heldout_surface",
    )
    out = {}
    for name in names:
        off_row = off.get("isolation", {}).get("summaries", {}).get(name)
        on_row = on.get("isolation", {}).get("summaries", {}).get(name)
        if not off_row or not on_row:
            continue
        out[name] = {
            "off_first_top1": off_row["first_top1"],
            "on_first_top1": on_row["first_top1"],
            "off_free_exact": off_row["free_exact"],
            "on_free_exact": on_row["free_exact"],
        }
    held_off = off["frozen"]["summaries"].get("heldout_surface")
    held_on = on["frozen"]["summaries"].get("heldout_surface")
    if held_off and held_on:
        out["heldout_surface"] = {
            "off_first_top1": held_off["first_top1"],
            "on_first_top1": held_on["first_top1"],
            "off_free_exact": held_off["free_exact"],
            "on_free_exact": held_on["free_exact"],
        }
    return out


def run_stage4(device_name: str = "cuda", policy: str = POLICY_A1) -> dict:
    """Replication sidecar 250002 + query-swap/held-out snapshot. No TEST."""
    if not REPLICA_CKPT.exists():
        report = {
            "id": f"{policy}_stage4",
            "policy": policy,
            "verdict": "BLOCKED",
            "lesson": f"replica checkpoint missing: {REPLICA_CKPT}",
            "protected_material_opened": False,
            "promoted": False,
            "test_opened": False,
        }
        write(OUT / f"{policy}_stage4.json", report)
        ledger_append(
            {
                **report,
                "change": "stage4 replica 250002",
                "long_gap": {},
                "primitive_induction": {},
                "primitive_keyed": {},
            }
        )
        return report
    device = resolve_device(device_name)
    model, overwrite, _ckpt = load_u16000_with_sidecar(device, REPLICA_CKPT, REPLICA_SHA)
    handle, _ = attach_overwrite(model, overwrite)
    try:
        print(json.dumps({"phase": "stage4_start", "sidecar": "250002", "sha256": REPLICA_SHA}), flush=True)
        stage1 = measure_stage1(model, overwrite, device, policy)
        stage1["id"] = f"{policy}_stage4_250002"
        stage1["sidecar"] = "250002"
        stage1["overwrite_checkpoint_sha256"] = REPLICA_SHA
        write(OUT / f"{policy}_stage4_stage1.json", {k: v for k, v in stage1.items() if k != "long_gap_rows"})
        ledger_append(stage1)
        print(
            json.dumps({"phase": "stage4_stage1_done", "verdict": stage1["verdict"], "lesson": stage1["lesson"]}),
            flush=True,
        )
        if stage1["verdict"] != "ADVANCE":
            return {
                "stage1": {k: v for k, v in stage1.items() if k != "long_gap_rows"},
                "verdict": "KILL",
                "promoted": False,
                "test_opened": False,
            }
        on = measure_arm_routed(model, device, overwrite, policy, first_answer_only=True, skip_long_gap=True)
        on["long_gap"] = stage1["long_gap"]
        on["long_gap_rows"] = stage1["long_gap_rows"]
        off = load_off_baseline()
        stage2 = stage2_from_on(off, on)
        iso = isolation_snapshot(off, on)
        report = {
            "id": f"{policy}_stage4",
            "policy": policy,
            "sidecar": "250002",
            "overwrite_checkpoint_sha256": REPLICA_SHA,
            "change": "A1 routing on P11 replica sidecar 250002",
            "stage1_verdict": stage1["verdict"],
            "stage2_verdict": stage2["verdict"],
            "long_gap": stage1["long_gap"],
            "primitive_induction": stage1["primitive_induction"],
            "primitive_keyed": stage1["primitive_keyed"],
            "retention": stage2["retention"],
            "isolation": iso,
            "verdict": "PASS" if stage2["verdict"] == "ADVANCE" else "KILL",
            "lesson": "replica reproduces A1" if stage2["verdict"] == "ADVANCE" else stage2["lesson"],
            "protected_material_opened": False,
            "promoted": False,
            "test_opened": False,
        }
        write(OUT / f"{policy}_stage4.json", report)
        ledger_append(report)
        print(json.dumps({"phase": "stage4_done", "verdict": report["verdict"], "isolation": iso}), flush=True)
        return report
    finally:
        handle.remove()
        overwrite.gen_index = None


def run_policy(device, model, overwrite, policy: str) -> dict:
    print(json.dumps({"phase": "start", "policy": policy, "device": str(device)}), flush=True)
    stage1 = measure_stage1(model, overwrite, device, policy)
    write(OUT / f"{policy}_stage1.json", {k: v for k, v in stage1.items() if k != "long_gap_rows"})
    ledger_append(stage1)
    print(json.dumps({"phase": "stage1_done", "verdict": stage1["verdict"], "lesson": stage1["lesson"]}), flush=True)
    if stage1["verdict"] != "ADVANCE":
        return {"policy": policy, "stage1": stage1, "advanced": False}

    print(json.dumps({"phase": "stage2_start", "policy": policy}), flush=True)
    on = measure_arm_routed(model, device, overwrite, policy, first_answer_only=True, skip_long_gap=True)
    on["long_gap"] = stage1["long_gap"]
    on["long_gap_rows"] = stage1["long_gap_rows"]
    off = load_off_baseline()
    stage2 = stage2_from_on(off, on)
    write(OUT / f"{policy}_stage2.json", stage2)
    ledger_append(
        {
            **stage2,
            "change": stage1["change"],
            "long_gap": stage1["long_gap"],
            "primitive_induction": stage1["primitive_induction"],
            "primitive_keyed": stage1["primitive_keyed"],
        }
    )
    print(json.dumps({"phase": "stage2_done", "verdict": stage2["verdict"], "lesson": stage2["lesson"]}), flush=True)
    if stage2["verdict"] != "ADVANCE":
        return {"policy": policy, "stage1": stage1, "stage2": stage2, "advanced": False}

    print(json.dumps({"phase": "stage3_start", "policy": policy}), flush=True)
    off_live = dict(off)
    off_long = score_long_gap(
        model,
        _long_gap_items(),
        device,
        overwrite=overwrite,
        first_answer_only=False,
        policy=policy,
    )
    off_live["long_gap"] = {key: value for key, value in off_long.items() if key != "rows"}
    off_live["long_gap_rows"] = off_long["rows"]
    decision = adjudicate(off_live, on)
    decision["policy"] = policy
    decision["routing"] = stage1["change"]
    decision["protocol"] = "RAPID_TREAT_STAGE3_P11_RUNTIME_ADAPTED"
    write(OUT / f"{policy}_OFF.json", slim_arm(off_live))
    write(OUT / f"{policy}_ON.json", slim_arm(on))
    write(OUT / f"{policy}_stage3.json", decision)
    ledger_append(
        {
            "id": f"{policy}_stage3",
            "policy": policy,
            "change": stage1["change"] + " ; full P11-runtime battery",
            "long_gap": {
                "free_exact": decision["long_gap"]["on_free_exact"],
                "n": decision["n_long"],
                "n_overwrite_armed": on["long_gap"].get("n_overwrite_armed"),
            },
            "primitive_induction": on["frozen"]["summaries"]["primitive_induction"],
            "primitive_keyed": on["frozen"]["summaries"]["primitive_keyed"],
            "verdict": decision["verdict"],
            "lesson": f"cause={decision['cause']}; benefit_ok={decision['benefit_ok']}; "
            f"on_retention={decision['on_retention_pass']}",
        }
    )
    print(
        json.dumps(
            {
                "phase": "stage3_done",
                "verdict": decision["verdict"],
                "long_gap": decision["long_gap"],
                "cause": decision["cause"],
            }
        ),
        flush=True,
    )
    return {
        "policy": policy,
        "stage1": {k: v for k, v in stage1.items() if k != "long_gap_rows"},
        "stage2": stage2,
        "stage3": decision,
        "advanced": True,
        "promoted": False,
        "test_opened": False,
    }


def next_policy_after_fail(failed: str, stage1: dict) -> str | None:
    lesson = stage1.get("lesson", "")
    if failed == POLICY_A1:
        if "long-gap died" in lesson or "never armed" in lesson or "benefit disappeared" in lesson:
            return POLICY_A2
        if "induction still damaged" in lesson:
            return POLICY_A2
        return POLICY_A2
    if failed == POLICY_A2:
        return POLICY_A3
    if failed == POLICY_A3:
        return None
    return POLICY_A1


def run_autonomous(device_name: str = "cuda") -> dict:
    device = resolve_device(device_name)
    model, overwrite, _ckpt = load_u16000_with_learned_overwrite(device)
    handle, _ = attach_overwrite(model, overwrite)
    results = []
    try:
        policy = POLICY_A1
        while policy is not None:
            result = run_policy(device, model, overwrite, policy)
            results.append(result)
            if result.get("stage3", {}).get("verdict") == "PASS":
                stage4 = run_stage4(device_name, policy)
                return {
                    "status": "SURVIVOR" if stage4.get("verdict") == "PASS" else "SURVIVOR_STAGE3_STAGE4_FAIL",
                    "results": _slim_results(results),
                    "stage4_verdict": stage4.get("verdict"),
                    "promoted": False,
                    "test_opened": False,
                }
            stage1 = result["stage1"]
            if result.get("stage2", {}).get("verdict") == "KILL":
                policy = next_policy_after_fail(policy, stage1)
                continue
            if stage1.get("verdict") != "ADVANCE":
                policy = next_policy_after_fail(policy, stage1)
                continue
            # Stage 3 did not PASS
            policy = next_policy_after_fail(policy, stage1)
        return {"status": "QUEUE_A_EXHAUSTED", "results": _slim_results(results), "promoted": False, "test_opened": False}
    finally:
        handle.remove()
        overwrite.gen_index = None


def _slim_results(results: list[dict]) -> list[dict]:
    slim = []
    for result in results:
        stage1 = result.get("stage1", {})
        slim.append(
            {
                "policy": result.get("policy"),
                "stage1_verdict": stage1.get("verdict"),
                "stage1_lesson": stage1.get("lesson"),
                "stage2_verdict": result.get("stage2", {}).get("verdict"),
                "stage3_verdict": result.get("stage3", {}).get("verdict"),
                "long_gap_on": stage1.get("long_gap", {}).get("free_exact"),
                "induction_top1": stage1.get("primitive_induction", {}).get("first_top1"),
            }
        )
    return slim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["canary", "run", "stage2", "stage3", "stage4"])
    parser.add_argument("--policy", default=POLICY_A1, choices=list(POLICIES))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = resolve_device(args.device)
    if args.action == "run":
        print(json.dumps(run_autonomous(args.device), default=str), flush=True)
        return
    if args.action == "stage4":
        print(json.dumps(run_stage4(args.device, args.policy), default=str), flush=True)
        return
    model, overwrite, _ckpt = load_u16000_with_learned_overwrite(device)
    handle, _ = attach_overwrite(model, overwrite)
    try:
        if args.action == "canary":
            report = measure_stage1(model, overwrite, device, args.policy)
            write(OUT / f"{args.policy}_stage1.json", {k: v for k, v in report.items() if k != "long_gap_rows"})
            ledger_append(report)
            print(json.dumps({k: v for k, v in report.items() if k != "long_gap_rows"}, default=str), flush=True)
        elif args.action == "stage2":
            on = measure_arm_routed(model, device, overwrite, args.policy, first_answer_only=True)
            stage2 = stage2_from_on(load_off_baseline(), on)
            write(OUT / f"{args.policy}_stage2.json", stage2)
            print(json.dumps(stage2, default=str), flush=True)
        else:
            off = measure_arm_routed(model, device, overwrite, args.policy, first_answer_only=False)
            on = measure_arm_routed(model, device, overwrite, args.policy, first_answer_only=True)
            decision = adjudicate(off, on)
            write(OUT / f"{args.policy}_stage3.json", decision)
            print(json.dumps(decision, default=str), flush=True)
    finally:
        handle.remove()
        overwrite.gen_index = None


if __name__ == "__main__":
    main()
