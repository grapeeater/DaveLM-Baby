"""P10 aux-only slot overwrite versus matched λ=0.

Protocol: `design/V010_SELECTION_REPAIR_P10_AUX_ONLY.md`.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import time
from pathlib import Path

from .query_presence import median
from .residual_overwrite import LocalSlotOverwrite, attach_overwrite, pointer_aux
from .selection_p1 import _draw, bootstrap_delta, long_rows
from .selection_p2 import absorb_keyed_row, annotate_keys, bind_fell_half, negative_controls, retention_failures
from .selection_p3 import DEBUG_LOG, SESSION
from .selection_p4 import copy_specs
from .selection_p9 import denials as p9_denials
from .selection_s1 import PANEL, PARENT, PARENT_SHA, digest, model_load, pack, write
from .selection_t1 import gap_strata, salience

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/selection_p10"
PROTOCOL = ROOT / "design/V010_SELECTION_REPAIR_P10_AUX_ONLY.md"
S2_DIAGNOSTIC = ROOT / "runs/selection_s2/DIAGNOSTIC.json"
P9_ADJUDICATION = ROOT / "runs/selection_p9/ADJUDICATION_230001_800.json"
P9_SCHEDULES = (
    ROOT / "runs/selection_p9/SCHEDULE_230001.json",
    ROOT / "runs/selection_p9/SCHEDULE_230002.json",
)

TRAIN_SEED = 240001
DATA_SEED = 240100
BOOTSTRAP_SEED = 240300
CE_TO_OVERWRITE = False
P9_LICENSE_VERDICT = "MECHANISM SUPPORTED, DOSE INSUFFICIENT"
MAX_UPDATES = 800
KEYED_PER_BATCH = 8
PRIMITIVE_PER_BATCH = 3
INDUCTION_PER_BATCH = 5
BATCH = KEYED_PER_BATCH + PRIMITIVE_PER_BATCH + INDUCTION_PER_BATCH
LANGUAGE_PROBABILITY = 0.20
PTR_LAMBDA = 1.0
GATE_LAMBDA = 1.0
GATE_ON_AFTER = 200
GATE_BIAS = -4.0
OVERWRITE_LR = 1e-3
MIN_GAP = 2
SUCCESS_EXCESS = 0.10
SUCCESS_DELTA = 0.07
SUCCESS_COSINE = 0.40
SUCCESS_COSINE_GAIN = 0.15
MECHANISM_COSINE = 0.20
MECHANISM_COSINE_GAIN = 0.10
MECHANISM_MASS = 0.50
FALSIFIED_MASS = 0.20
NULL_EXCESS = 0.02
FUTILITY_EXCESS = 0.02
FUTILITY_COSINE = 0.05
FUTILITY_MASS_200 = 0.05
FUTILITY_MASS_400 = 0.50
LANGUAGE_CE_HARD = 0.20
INIT_HIT_TOLERANCE = 8
PARENT_LONG_HITS = 74


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


def assert_p9_licenses_p10() -> dict:
    adj = json.loads(P9_ADJUDICATION.read_text(encoding="utf-8"))
    if adj.get("protocol") != "V010_SELECTION_REPAIR_P9_SLOT_GATEON":
        raise RuntimeError("P10 requires P9 adjudication")
    if adj.get("verdict") != P9_LICENSE_VERDICT:
        raise RuntimeError("P10 is licensed only by P9 MECHANISM SUPPORTED, DOSE INSUFFICIENT")
    return adj


def denials() -> tuple[set[tuple], set[tuple]]:
    denied_inputs, denied_spans = p9_denials()
    for path in P9_SCHEDULES:
        if not path.exists():
            raise RuntimeError(f"missing P9 schedule {path}")
        for spec in json.loads(path.read_text(encoding="utf-8")):
            if spec.get("task") != "structured":
                continue
            for row in spec["items"]:
                denied_inputs.add(tuple(row["input"]))
    return denied_inputs, denied_spans


def overwrite_stats(model, overwrite, items, device) -> dict:
    import torch
    import torch.nn.functional as F

    long_c, long_m, long_g = [], [], []
    model.eval()
    overwrite.eval()
    with torch.no_grad():
        for start in range(0, len(items), 8):
            batch = items[start : start + 8]
            x, _, _, _ = pack(batch, device)
            model.forward_hidden(x)
            h0 = overwrite.last_hidden
            attn = overwrite.last_attn
            gate = overwrite.last_gate
            for i, item in enumerate(batch):
                if item.get("kind") != "keyed" or "query_position" not in item:
                    continue
                gen = len(item["input"]) - 1
                query = int(item["query_position"])
                if gen - query < 13:
                    continue
                long_c.append(float(F.cosine_similarity(h0[i, gen], h0[i, query], dim=0).item()))
                long_m.append(float(attn[i, gen, query].item()))
                long_g.append(float(gate[i, gen].item()))
    return {
        "n_long": len(long_c),
        "median_long": median(long_c),
        "median_long_mass": median(long_m),
        "median_long_gate": median(long_g),
    }


def measure(model, overwrite, dev_items, diagnostic, full: bool = True) -> dict:
    from .selection_s2 import measure as s2_measure

    report = s2_measure(model, dev_items, full=full)
    report["gap_strata"] = gap_strata(report["rows"], diagnostic)
    report["summary"].update(salience(report["rows"]))
    report["summary"]["primary_long_gap_excess"] = report["gap_strata"]["long"]["excess"]
    device = next(model.parameters()).device
    stats = overwrite_stats(model, overwrite, dev_items, device)
    report["l0_cosine"] = {"n_long": stats["n_long"], "median_long": stats["median_long"]}
    report["overwrite_pointer"] = stats
    report["summary"]["l0_long_query_cosine"] = stats["median_long"]
    report["summary"]["overwrite_long_query_mass"] = stats["median_long_mass"]
    return report


def decide_verdict(
    *,
    treatment_excess: float,
    control_excess: float,
    ci_lo: float,
    treatment_cosine: float,
    cosine_gain: float,
    treatment_mass: float,
    regressions: list,
    ptr_halved: bool,
    negative_ok: bool,
) -> str:
    if regressions or not negative_ok:
        return "REGRESSION"
    delta = treatment_excess - control_excess
    mechanism_cos = treatment_cosine >= SUCCESS_COSINE or cosine_gain >= SUCCESS_COSINE_GAIN
    if treatment_excess >= SUCCESS_EXCESS and delta >= SUCCESS_DELTA and ci_lo > 0 and mechanism_cos:
        return "SUCCESS"
    if ptr_halved and treatment_mass < FALSIFIED_MASS and treatment_excess < NULL_EXCESS:
        return "FALSIFIED"
    if (
        treatment_mass >= MECHANISM_MASS
        or treatment_cosine >= MECHANISM_COSINE
        or cosine_gain >= MECHANISM_COSINE_GAIN
    ):
        return "MECHANISM SUPPORTED, DOSE INSUFFICIENT"
    if treatment_excess < NULL_EXCESS and cosine_gain < FUTILITY_COSINE and treatment_mass < FUTILITY_MASS_200:
        return "NULL"
    return "MIXED"


def generate() -> None:
    from .data import LANG_TRAIN, build_banks, read_u16
    from .v2r4_provenance import FROZEN_PANELS_SHA256, require_frozen_file

    require_frozen_file(PANEL, FROZEN_PANELS_SHA256, "panels")
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    p9 = assert_p9_licenses_p10()
    if (OUT / "MANIFEST.json").exists():
        raise RuntimeError("P10 already frozen")
    banks = build_banks(read_u16(LANG_TRAIN))
    base_inputs, base_spans = denials()
    audits = {}
    for seed in (TRAIN_SEED, TRAIN_SEED + 1):
        dest = OUT / f"SCHEDULE_{seed}.json"
        denied_inputs = set(base_inputs)
        denied_spans = set(base_spans)
        for earlier in range(TRAIN_SEED, seed):
            earlier_path = OUT / f"SCHEDULE_{earlier}.json"
            if earlier_path.exists():
                for spec in json.loads(earlier_path.read_text(encoding="utf-8")):
                    if spec.get("task") != "structured":
                        continue
                    for row in spec["items"]:
                        denied_inputs.add(tuple(row["input"]))
        rng = random.Random(DATA_SEED + seed - TRAIN_SEED)
        schedule = None
        if dest.exists():
            loaded = json.loads(dest.read_text(encoding="utf-8"))
            if len(loaded) == MAX_UPDATES:
                schedule = loaded
        if schedule is None:
            schedule = []
            for update in range(1, MAX_UPDATES + 1):
                if rng.random() < LANGUAGE_PROBABILITY:
                    schedule.append({"task": "language", "rng_seed": rng.randrange(2**31)})
                    continue
                items = []
                for _ in range(KEYED_PER_BATCH):
                    item = annotate_keys(
                        _draw(rng, banks, denied_inputs, denied_spans, kind="keyed", difficulty="full")
                    )
                    absorb_keyed_row(denied_inputs, denied_spans, item)
                    items.append(item)
                for _ in range(PRIMITIVE_PER_BATCH):
                    prim = annotate_keys(
                        _draw(rng, banks, denied_inputs, denied_spans, kind="keyed", difficulty="primitive")
                    )
                    absorb_keyed_row(denied_inputs, denied_spans, prim)
                    items.append(prim)
                for _ in range(INDUCTION_PER_BATCH):
                    ind = _draw(rng, banks, denied_inputs, denied_spans, kind="induction", difficulty="full")
                    denied_inputs.add(tuple(ind["input"]))
                    items.append(ind)
                assert len(items) == BATCH
                schedule.append({"task": "structured", "update": update, "items": items})
                if update % 50 == 0:
                    write(dest, schedule)
                    print(json.dumps({"generate_seed": seed, "update": update, "rows": len(schedule)}), flush=True)
            write(dest, schedule)
        rows = [r for s in schedule if s["task"] == "structured" for r in s["items"]]
        keyed_full = [r for r in rows if r["kind"] == "keyed" and r["difficulty"] == "full"]
        audits[str(seed)] = {
            "updates": len(schedule),
            "language_updates": sum(s["task"] == "language" for s in schedule),
            "keyed_full_rows": len(keyed_full),
            "induction_rows": sum(r["kind"] == "induction" for r in rows),
        }
    files = [
        PROTOCOL,
        Path(__file__),
        PANEL,
        OUT / f"SCHEDULE_{TRAIN_SEED}.json",
        OUT / f"SCHEDULE_{TRAIN_SEED + 1}.json",
        S2_DIAGNOSTIC,
        P9_ADJUDICATION,
        PARENT,
        ROOT / "src/baby_v010/residual_overwrite.py",
        ROOT / "src/baby_v010/model.py",
    ]
    write(
        OUT / "MANIFEST.json",
        {
            "protocol": "V010_SELECTION_REPAIR_P10_AUX_ONLY",
            "parent_sha256": PARENT_SHA,
            "ptr_lambda_treatment": PTR_LAMBDA,
            "gate_lambda_treatment_after_200": GATE_LAMBDA,
            "lambda_control": 0.0,
            "ce_to_overwrite": CE_TO_OVERWRITE,
            "licensed_by": "V010_SELECTION_REPAIR_P9_SLOT_GATEON",
            "p9_verdict": p9["verdict"],
            "pointer": "local_slot_h0_t_tplus1",
            "futility_mass_200": FUTILITY_MASS_200,
            "overwrite_lr": OVERWRITE_LR,
            "baby_weights_frozen": True,
            "gate_bias_init": GATE_BIAS,
            "batch": {
                "keyed_full": KEYED_PER_BATCH,
                "primitive_keyed": PRIMITIVE_PER_BATCH,
                "induction_full": INDUCTION_PER_BATCH,
            },
            "p2_launched": False,
            "files": {posix(path): digest(path) for path in files},
            "audits": audits,
            "protected_material_opened": False,
        },
    )
    print(json.dumps({"generated": True, "audits": audits}), flush=True)


def verify() -> dict:
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError("hash mismatch " + rel)
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    assert_p9_licenses_p10()
    return manifest


def run(arm: str, seed: int, until: int) -> None:
    import torch
    import torch.nn.functional as F

    from .data import LANG_TRAIN, read_u16
    from .train_v2r4 import language_batch, set_seed

    verify()
    dest = OUT / f"{arm}_{seed}"
    if dest.exists():
        raise RuntimeError("refuse overwrite " + str(dest))
    dest.mkdir(parents=True)
    set_seed(seed)
    model, config = model_load()
    model.set_attention_backend("sdpa")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    overwrite = LocalSlotOverwrite(config.d_model, gate_bias=GATE_BIAS).to(
        next(model.parameters()).device
    )
    optimizer = torch.optim.AdamW(overwrite.parameters(), lr=OVERWRITE_LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    stream = torch.tensor(read_u16(LANG_TRAIN), dtype=torch.long)
    dev_items = json.loads(S2_DIAGNOSTIC.read_text(encoding="utf-8"))
    diagnostic = {(r["body_id"], r["query_index"]): r for r in dev_items}
    schedule = json.loads((OUT / f"SCHEDULE_{seed}.json").read_text(encoding="utf-8"))
    ptr_lam = PTR_LAMBDA if arm == "treatment" else 0.0
    gate_lam_full = GATE_LAMBDA if arm == "treatment" else 0.0
    device = torch.device("cuda")
    start = time.monotonic()
    handle, _bucket = attach_overwrite(model, overwrite)
    try:
        baseline = measure(model, overwrite, dev_items, diagnostic)
        write(dest / "eval_0000.json", baseline)
        debug_log(
            "P10",
            "selection_p10.py:run",
            "baseline",
            {
                "arm": arm,
                "long": baseline["gap_strata"]["long"],
                "l0": baseline["l0_cosine"],
                "ow": baseline["overwrite_pointer"],
            },
        )
        print(
            json.dumps(
                {
                    "arm": arm,
                    "step": 0,
                    "primary": baseline["gap_strata"]["long"],
                    "l0": baseline["l0_cosine"],
                    "ow": baseline["overwrite_pointer"],
                    "init_ok": abs(baseline["gap_strata"]["long"]["hit"] - PARENT_LONG_HITS) <= INIT_HIT_TOLERANCE,
                }
            ),
            flush=True,
        )
        if abs(baseline["gap_strata"]["long"]["hit"] - PARENT_LONG_HITS) > INIT_HIT_TOLERANCE:
            write(
                dest / "RECEIPT_0000.json",
                {"reason": "invalid_init", "last_step": 0, "arm": arm, "seed": seed, "hits": baseline["gap_strata"]["long"]["hit"]},
            )
            print(json.dumps({"arm": arm, "reason": "invalid_init"}), flush=True)
            return
        initial_ce = baseline["language_dev_ce"]
        baseline_long = baseline["gap_strata"]["long"]["excess"]
        baseline_l0 = baseline["l0_cosine"]["median_long"]
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        reason = "terminal"
        last_step = 0
        ptr_history: list[float] = []
        for step in range(1, until + 1):
            spec = schedule[step - 1]
            if shutil.disk_usage(ROOT).free < 10 * 2**30:
                reason, last_step = "hard_stop_disk", step - 1
                break
            if time.monotonic() - start > 7200:
                reason, last_step = "hard_stop_runtime", step - 1
                break
            model.train()
            overwrite.train()
            model.set_attention_backend("sdpa")
            optimizer.zero_grad(set_to_none=True)
            ptr_value = 0.0
            gate_value = 0.0
            gate_lam = gate_lam_full if step > GATE_ON_AFTER else 0.0
            if spec["task"] == "language":
                x, y = language_batch(stream, random.Random(spec["rng_seed"]), 16, 256, device)
                with torch.no_grad():
                    loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
                last_step = step
                with (dest / "events.jsonl").open("a", encoding="utf-8") as handle_events:
                    handle_events.write(
                        json.dumps(
                            {
                                "step": step,
                                "task": spec["task"],
                                "loss": loss.item(),
                                "ptr": 0.0,
                                "gate": 0.0,
                                "lambda_ptr": ptr_lam,
                                "lambda_gate": gate_lam,
                                "gradient_norm": 0.0,
                                "ce_to_overwrite": CE_TO_OVERWRITE,
                            }
                        )
                        + "\n"
                    )
                if step % 50 == 0:
                    print(
                        json.dumps(
                            {
                                "arm": arm,
                                "step": step,
                                "loss": loss.item(),
                                "ptr": 0.0,
                                "gate": 0.0,
                                "elapsed_s": round(time.monotonic() - start),
                            }
                        ),
                        flush=True,
                    )
                if step % 200 != 0:
                    continue
            else:
                items = spec["items"]
                x, y, mask, _first = pack(items, device)
                hidden = model.forward_hidden(x)
                logits = model.language_head(hidden.detach() if not CE_TO_OVERWRITE else hidden)
                loss = F.cross_entropy(logits[mask], y[mask])
                specs = copy_specs(items)
                extra_ptr, extra_gate = pointer_aux(overwrite.last_attn, overwrite.last_gate, specs, use_gate=gate_lam > 0)
                if ptr_lam > 0:
                    ptr_value = float(extra_ptr.detach().item())
                    loss = loss + ptr_lam * extra_ptr
                    ptr_history.append(ptr_value)
                if gate_lam > 0:
                    gate_value = float(extra_gate.detach().item())
                    loss = loss + gate_lam * extra_gate
                if len(ptr_history) <= 3 and ptr_lam > 0:
                    debug_log(
                        "A",
                        "selection_p10.py:run",
                        "first_ptr_batches",
                        {
                            "arm": arm,
                            "step": step,
                            "n_specs": len(specs),
                            "ptr": ptr_value,
                            "mass0": float(overwrite.last_attn[specs[0][0], specs[0][1], specs[0][2]].detach()) if specs else None,
                            "ce_to_overwrite": CE_TO_OVERWRITE,
                        },
                    )
                if not torch.isfinite(loss):
                    reason, last_step = "hard_stop_nonfinite_loss", step
                    break
                if loss.requires_grad:
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(overwrite.parameters(), 2.0)
                    if not torch.isfinite(norm):
                        reason, last_step = "hard_stop_nonfinite_gradient", step
                        break
                    optimizer.step()
                else:
                    norm = 0.0
                last_step = step
                with (dest / "events.jsonl").open("a", encoding="utf-8") as handle_events:
                    handle_events.write(
                        json.dumps(
                            {
                                "step": step,
                                "task": spec["task"],
                                "loss": loss.item(),
                                "ptr": ptr_value,
                                "gate": gate_value,
                                "lambda_ptr": ptr_lam,
                                "lambda_gate": gate_lam,
                                "gradient_norm": float(norm),
                            }
                        )
                        + "\n"
                    )
                if step % 50 == 0:
                    print(
                        json.dumps(
                            {
                                "arm": arm,
                                "step": step,
                                "loss": loss.item(),
                                "ptr": ptr_value,
                                "gate": gate_value,
                                "elapsed_s": round(time.monotonic() - start),
                            }
                        ),
                        flush=True,
                    )
            if step % 200 == 0:
                model.eval()
                overwrite.eval()
                report = measure(model, overwrite, dev_items, diagnostic, full=True)
                write(dest / f"eval_{step:04d}.json", report)
                torch.save(
                    {
                        "config": config.to_dict(),
                        "model_state_dict": model.state_dict(),
                        "overwrite_state_dict": overwrite.state_dict(),
                        "parent_checkpoint_sha256": PARENT_SHA,
                        "update": 16000 + step,
                        "seed": seed,
                        "arm": arm,
                        "lambda_ptr": ptr_lam,
                        "protocol": "V010_SELECTION_REPAIR_P10_AUX_ONLY",
                        "baby_weights_frozen": True,
                        "ce_to_overwrite": CE_TO_OVERWRITE,
                        "overwrite_lr": OVERWRITE_LR,
                        "protected_material_opened": False,
                        "p2_launched": False,
                        "manifest_sha256": digest(OUT / "MANIFEST.json"),
                    },
                    dest / f"checkpoint_{16000 + step}.pt",
                )
                debug_log(
                    "P10",
                    "selection_p10.py:run",
                    "eval",
                    {"arm": arm, "step": step, "long": report["gap_strata"]["long"], "ow": report["overwrite_pointer"]},
                )
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "step": step,
                            "primary": report["gap_strata"]["long"],
                            "l0": report["l0_cosine"],
                            "ow": report["overwrite_pointer"],
                            "ce": report["language_dev_ce"],
                        }
                    ),
                    flush=True,
                )
                if report["language_dev_ce"] > initial_ce + LANGUAGE_CE_HARD:
                    reason = "hard_stop_language"
                    break
                mass = report["overwrite_pointer"]["median_long_mass"]
                if step == 200 and arm == "treatment" and mass < FUTILITY_MASS_200:
                    reason = "futility"
                    break
                if step == 400 and arm == "treatment":
                    gain = report["gap_strata"]["long"]["excess"] - baseline_long
                    cosine_gain = report["l0_cosine"]["median_long"] - baseline_l0
                    if gain < FUTILITY_EXCESS and cosine_gain < FUTILITY_COSINE and mass < FUTILITY_MASS_400:
                        reason = "futility"
                        break
        write(
            dest / f"RECEIPT_{last_step:04d}.json",
            {
                "reason": reason,
                "last_step": last_step,
                "arm": arm,
                "seed": seed,
                "lambda_ptr": ptr_lam,
                "parent_sha256": digest(PARENT),
                "mean_ptr_first50": statistics.mean(ptr_history[:50]) if len(ptr_history) >= 50 else None,
                "mean_ptr_last50": statistics.mean(ptr_history[-50:]) if ptr_history else None,
                "bind_fell_half": bind_fell_half(ptr_history),
                "artifacts": {p.name: digest(p) for p in dest.iterdir() if p.is_file()},
            },
        )
        print(json.dumps({"arm": arm, "reason": reason, "last_step": last_step}), flush=True)
    finally:
        handle.remove()


def adjudicate() -> dict:
    p9 = assert_p9_licenses_p10()
    diagnostic_items = json.loads(S2_DIAGNOSTIC.read_text(encoding="utf-8"))
    diagnostic = {(r["body_id"], r["query_index"]): r for r in diagnostic_items}
    treat_dir = OUT / f"treatment_{TRAIN_SEED}"
    control_dir = OUT / f"control_{TRAIN_SEED}"
    receipt = json.loads(max(treat_dir.glob("RECEIPT_*.json")).read_text(encoding="utf-8"))
    last = receipt["last_step"]
    step = last if last % 200 == 0 else (last // 200) * 200
    treat = json.loads((treat_dir / f"eval_{step:04d}.json").read_text(encoding="utf-8"))
    control = json.loads((control_dir / f"eval_{step:04d}.json").read_text(encoding="utf-8"))
    parent = json.loads((treat_dir / "eval_0000.json").read_text(encoding="utf-8"))
    t_long = treat["gap_strata"]["long"]
    c_long = control["gap_strata"]["long"]
    p_long = parent["gap_strata"]["long"]
    t_cos = treat["l0_cosine"]["median_long"]
    p_cos = parent["l0_cosine"]["median_long"]
    t_mass = treat["overwrite_pointer"]["median_long_mass"]
    ci = bootstrap_delta(long_rows(treat, diagnostic), long_rows(control, diagnostic), seed=BOOTSTRAP_SEED)
    regressions = retention_failures(parent, treat)
    controls = negative_controls(treat)
    verdict = decide_verdict(
        treatment_excess=t_long["excess"],
        control_excess=c_long["excess"],
        ci_lo=ci[0],
        treatment_cosine=t_cos,
        cosine_gain=t_cos - p_cos,
        treatment_mass=t_mass,
        regressions=regressions,
        ptr_halved=bool(receipt.get("bind_fell_half")),
        negative_ok=all(row["pass"] for row in controls.values()),
    )
    decision = {
        "protocol": "V010_SELECTION_REPAIR_P10_AUX_ONLY",
        "licensed_by": "V010_SELECTION_REPAIR_P9_SLOT_GATEON",
        "p9_verdict": p9["verdict"],
        "ce_to_overwrite": CE_TO_OVERWRITE,
        "step": step,
        "reason": receipt["reason"],
        "verdict": verdict,
        "parent_sha256": PARENT_SHA,
        "primary_endpoint": {
            "parent_hit": p_long["hit"],
            "treatment_hit": t_long["hit"],
            "control_hit": c_long["hit"],
            "parent_excess": p_long["excess"],
            "treatment_excess": t_long["excess"],
            "control_excess": c_long["excess"],
            "treatment_minus_control": t_long["excess"] - c_long["excess"],
            "bootstrap_ci95": list(ci),
        },
        "mechanism": {
            "parent_l0_cosine": p_cos,
            "treatment_l0_cosine": t_cos,
            "control_l0_cosine": control["l0_cosine"]["median_long"],
            "cosine_gain": t_cos - p_cos,
            "treatment_overwrite_mass": t_mass,
            "control_overwrite_mass": control["overwrite_pointer"]["median_long_mass"],
        },
        "regressions": regressions,
        "retention_other": {
            "language_dev_ce": {
                "parent": parent["language_dev_ce"],
                "treatment": treat["language_dev_ce"],
                "control": control["language_dev_ce"],
            },
            "primitive_induction_first_top1": {
                "parent": parent["frozen"]["summaries"]["primitive_induction"]["first_top1"],
                "treatment": treat["frozen"]["summaries"]["primitive_induction"]["first_top1"],
                "control": control["frozen"]["summaries"]["primitive_induction"]["first_top1"],
            },
        },
        "negative_controls": controls,
        "trained": True,
        "p2_launched": False,
        "gates_changed": False,
        "protected_material_opened": False,
        "authoritative_parent_unchanged": True,
    }
    write(OUT / f"ADJUDICATION_{TRAIN_SEED}_{step}.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "run", "adjudicate"])
    parser.add_argument("--arm", choices=["control", "treatment"])
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--until", type=int, default=MAX_UPDATES)
    args = parser.parse_args()
    if args.action == "generate":
        generate()
    elif args.action == "adjudicate":
        print(json.dumps(adjudicate()), flush=True)
    else:
        if not args.arm:
            raise SystemExit("--arm required")
        run(args.arm, args.seed, args.until)


if __name__ == "__main__":
    main()
