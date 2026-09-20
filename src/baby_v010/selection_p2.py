"""P2 residual query-bind auxiliary versus a matched λ=0 control.

Protocol: `design/V010_SELECTION_REPAIR_P2_RESIDUAL_BIND.md`.

CE is identical in both arms. Treatment adds InfoNCE so the generation residual
matches the query-token residual and not other in-context key residuals.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .query_presence import body_key_positions, median
from .selection_p1 import _draw, bootstrap_delta, denials as p1_denials, long_rows, pointer_stats
from .selection_s1 import PANEL, PARENT, PARENT_SHA, digest, model_load, pack, write
from .selection_t1 import gap_strata, salience

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/selection_p2"
PROTOCOL = ROOT / "design/V010_SELECTION_REPAIR_P2_RESIDUAL_BIND.md"
S2_DIAGNOSTIC = ROOT / "runs/selection_s2/DIAGNOSTIC.json"
D1B_DIAGNOSTIC = ROOT / "runs/query_presence_d1b/DIAGNOSTIC.json"
D2_ADJUDICATION = ROOT / "runs/query_presence_d2/ADJUDICATION.json"
P1_SCHEDULES = (
    ROOT / "runs/selection_p1/SCHEDULE_150001.json",
    ROOT / "runs/selection_p1/SCHEDULE_150002.json",
)

TRAIN_SEED = 160001
DATA_SEED = 160100
BOOTSTRAP_SEED = 160300
MAX_UPDATES = 800
KEYED_PER_BATCH = 10
PRIMITIVE_PER_BATCH = 3
INDUCTION_PER_BATCH = 3
BATCH = KEYED_PER_BATCH + PRIMITIVE_PER_BATCH + INDUCTION_PER_BATCH
LANGUAGE_PROBABILITY = 0.20
BIND_LAMBDA = 0.25
BIND_TAU = 0.10
MIN_GAP = 2
RETENTION_DROP = 0.05
SUCCESS_EXCESS = 0.10
SUCCESS_DELTA = 0.07
SUCCESS_RESIDUAL = 0.99
MECHANISM_RESIDUAL_DROP = 0.005
FALSIFIED_RESIDUAL = 0.999
NULL_EXCESS = 0.02
FUTILITY_GAIN = 0.02
LANGUAGE_CE_FLAG = 0.05
LANGUAGE_CE_HARD = 0.20


def posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def assert_d2_licenses_p2() -> dict:
    if not D2_ADJUDICATION.exists():
        raise RuntimeError("D2 adjudication missing; P2 is licensed only by D2 MIXED")
    adj = json.loads(D2_ADJUDICATION.read_text(encoding="utf-8"))
    if adj.get("protocol") != "V010_QUERY_PRESENCE_D2":
        raise RuntimeError("D2 adjudication protocol mismatch")
    if adj.get("headline") != "MIXED":
        raise RuntimeError("P2 residual-bind is licensed only by D2 MIXED")
    if adj.get("control", {}).get("verdict") != "B":
        raise RuntimeError("P2 requires D2 control to have stayed B")
    return adj


def absorb_keyed_row(denied_inputs: set[tuple], denied_spans: set[tuple], row: dict) -> None:
    from .isolation_transforms import parse_records

    denied_inputs.add(tuple(row["input"]))
    denied_spans.add(tuple(row["target_span"]))
    if row.get("kind") == "keyed":
        denied_spans.update(tuple(value) for _, value in parse_records(row))


def denials() -> tuple[set[tuple], set[tuple]]:
    denied_inputs, denied_spans = p1_denials()
    for path in P1_SCHEDULES:
        if not path.exists():
            raise RuntimeError(f"P2 refuses to generate without P1 schedule {path}")
        for spec in json.loads(path.read_text(encoding="utf-8")):
            if spec.get("task") != "structured":
                continue
            for row in spec["items"]:
                absorb_keyed_row(denied_inputs, denied_spans, row)
    return denied_inputs, denied_spans


def annotate_keys(item: dict) -> dict:
    if item.get("kind") != "keyed":
        return item
    positions, _queried = body_key_positions(item)
    item["body_key_positions"] = [int(p) for p in positions]
    return item


def body_keys_for(item: dict) -> list[int]:
    stored = item.get("body_key_positions")
    if stored:
        return [int(p) for p in stored]
    positions, _queried = body_key_positions(item)
    return [int(p) for p in positions]


def bind_specs(items: list[dict], min_gap: int = MIN_GAP) -> list[tuple[int, int, list[int]]]:
    specs = []
    for i, item in enumerate(items):
        if item.get("kind") != "keyed" or "query_position" not in item:
            continue
        gen = len(item["input"]) - 1
        query = int(item["query_position"])
        if gen - query < min_gap:
            continue
        keys = []
        seen = {query}
        for pos in body_keys_for(item):
            if pos != query and 0 <= pos <= gen and pos not in seen:
                keys.append(pos)
                seen.add(pos)
        if not keys:
            continue
        specs.append((i, gen, [query, *keys]))
    return specs


def residual_bind_loss(hidden, specs: list[tuple[int, int, list[int]]], tau: float = BIND_TAU):
    import torch.nn.functional as F

    if not specs:
        return hidden.new_zeros(())
    terms = []
    for batch_i, gen_pos, positions in specs:
        anchor = hidden[batch_i, gen_pos]
        keys = hidden[batch_i, positions].detach()
        scores = F.cosine_similarity(anchor.unsqueeze(0), keys, dim=-1) / tau
        terms.append(-F.log_softmax(scores, dim=0)[0])
    return torch_stack_mean(terms)


def torch_stack_mean(terms):
    import torch

    return torch.stack(terms).mean()


def d1b_twin_residual(model, items, device) -> dict:
    import torch.nn.functional as F

    bodies = defaultdict(list)
    for item in items:
        bodies[item["body_id"]].append(item)
    long_cos, short_cos, all_cos = [], [], []
    model.eval()
    import torch

    with torch.no_grad():
        for group in bodies.values():
            x, _, _, _ = pack(group, device)
            hidden = model.forward_hidden(x)
            for i, left in enumerate(group):
                gen = len(left["input"]) - 1
                gap = gen - int(left["query_position"])
                for j in range(len(group)):
                    if i == j:
                        continue
                    value = float(F.cosine_similarity(hidden[i, gen], hidden[j, gen], dim=0).item())
                    all_cos.append(value)
                    if gap >= 13:
                        long_cos.append(value)
                    if gap <= 1:
                        short_cos.append(value)
    return {
        "n_long": len(long_cos),
        "median_long": median(long_cos),
        "n_short": len(short_cos),
        "median_short": median(short_cos),
        "n_all": len(all_cos),
        "median_all": median(all_cos),
    }


def measure(model, dev_items, diagnostic, d1b_items, full: bool = True) -> dict:
    from .selection_s2 import measure as s2_measure

    report = s2_measure(model, dev_items, full=full)
    report["gap_strata"] = gap_strata(report["rows"], diagnostic)
    report["summary"].update(salience(report["rows"]))
    report["summary"]["primary_long_gap_excess"] = report["gap_strata"]["long"]["excess"]
    report["pointer"] = pointer_stats(model, dev_items, next(model.parameters()).device)
    report["summary"]["long_gap_query_track"] = report["pointer"]["long"].get(
        "fraction_any_query_tracking_head", float("nan")
    )
    report["twin_residual"] = d1b_twin_residual(model, d1b_items, next(model.parameters()).device)
    report["summary"]["d1b_long_residual_cosine"] = report["twin_residual"]["median_long"]
    return report


def generate() -> None:
    from .data import LANG_TRAIN, build_banks, read_u16
    from .v2r4_provenance import FROZEN_PANELS_SHA256, require_frozen_file

    require_frozen_file(PANEL, FROZEN_PANELS_SHA256, "panels")
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    d2 = assert_d2_licenses_p2()
    if (OUT / "MANIFEST.json").exists():
        raise RuntimeError("P2 already frozen")
    banks = build_banks(read_u16(LANG_TRAIN))
    denied_inputs, denied_spans = denials()

    audits = {}
    for seed in (TRAIN_SEED, TRAIN_SEED + 1):
        rng = random.Random(DATA_SEED + seed - TRAIN_SEED)
        schedule: list[dict] = []
        gap_hist: dict[str, int] = defaultdict(int)
        bindable = 0
        for update in range(1, MAX_UPDATES + 1):
            if rng.random() < LANGUAGE_PROBABILITY:
                schedule.append({"task": "language", "rng_seed": rng.randrange(2**31)})
                continue
            items: list[dict] = []
            for _ in range(KEYED_PER_BATCH):
                item = annotate_keys(
                    _draw(rng, banks, denied_inputs, denied_spans, kind="keyed", difficulty="full")
                )
                gap = len(item["input"]) - 1 - int(item["query_position"])
                gap_hist[str(gap)] += 1
                if gap >= MIN_GAP:
                    bindable += 1
                absorb_keyed_row(denied_inputs, denied_spans, item)
                items.append(item)
            for _ in range(PRIMITIVE_PER_BATCH):
                prim = annotate_keys(
                    _draw(
                        rng, banks, denied_inputs, denied_spans,
                        kind="keyed", difficulty="primitive",
                    )
                )
                absorb_keyed_row(denied_inputs, denied_spans, prim)
                items.append(prim)
            for _ in range(INDUCTION_PER_BATCH):
                ind = _draw(
                    rng, banks, denied_inputs, denied_spans,
                    kind="induction", difficulty="full",
                )
                absorb_keyed_row(denied_inputs, denied_spans, ind)
                items.append(ind)
            assert len(items) == BATCH
            schedule.append({"task": "structured", "update": update, "items": items})
            if update % 100 == 0:
                print(json.dumps({"generate_seed": seed, "update": update}), flush=True)
        write(OUT / f"SCHEDULE_{seed}.json", schedule)
        rows = [r for s in schedule if s["task"] == "structured" for r in s["items"]]
        keyed_full = [r for r in rows if r["kind"] == "keyed" and r["difficulty"] == "full"]
        audits[str(seed)] = {
            "updates": len(schedule),
            "language_updates": sum(s["task"] == "language" for s in schedule),
            "rows": len(rows),
            "keyed_full_rows": len(keyed_full),
            "primitive_keyed_rows": sum(
                r["kind"] == "keyed" and r["difficulty"] == "primitive" for r in rows
            ),
            "induction_rows": sum(r["kind"] == "induction" for r in rows),
            "bindable_full_keyed_gap_ge2": bindable,
            "natural_gap_histogram": dict(sorted(gap_hist.items(), key=lambda kv: int(kv[0]))),
            "keyed_pair_count": {
                str(k): sum(r["pair_count"] == k for r in keyed_full) for k in (2, 3, 4)
            },
            "keyed_variant": {
                v: sum(r["variant"] == v for r in keyed_full)
                for v in sorted({r["variant"] for r in keyed_full})
            },
            "no_exact_input_or_candidate_span_leakage": True,
            "arms_share_items_and_order": True,
        }

    files = [
        PROTOCOL,
        Path(__file__),
        PANEL,
        OUT / f"SCHEDULE_{TRAIN_SEED}.json",
        OUT / f"SCHEDULE_{TRAIN_SEED + 1}.json",
        S2_DIAGNOSTIC,
        D1B_DIAGNOSTIC,
        D2_ADJUDICATION,
        ROOT / "src/baby_v010/data.py",
        ROOT / "src/baby_v010/data_v2.py",
        ROOT / "src/baby_v010/model.py",
        ROOT / "src/baby_v010/config.py",
        ROOT / "src/baby_v010/evaluate.py",
        ROOT / "src/baby_v010/train_v2r4.py",
        ROOT / "src/baby_v010/query_presence.py",
        ROOT / "src/baby_v010/selection_s1.py",
        ROOT / "src/baby_v010/selection_t1.py",
        ROOT / "src/baby_v010/selection_p1.py",
    ]
    write(
        OUT / "MANIFEST.json",
        {
            "protocol": "V010_SELECTION_REPAIR_P2_RESIDUAL_BIND",
            "parent_sha256": PARENT_SHA,
            "bind_lambda_treatment": BIND_LAMBDA,
            "bind_lambda_control": 0.0,
            "bind_tau": BIND_TAU,
            "min_gap": MIN_GAP,
            "licensed_by": "V010_QUERY_PRESENCE_D2",
            "d2_headline": d2["headline"],
            "batch_composition": {
                "keyed_full": KEYED_PER_BATCH,
                "primitive_keyed": PRIMITIVE_PER_BATCH,
                "induction_full": INDUCTION_PER_BATCH,
            },
            "language_probability": LANGUAGE_PROBABILITY,
            "natural_gaps": True,
            "p1_schedule_sha256": {posix(path): digest(path) for path in P1_SCHEDULES},
            "files": {posix(path): digest(path) for path in files},
            "audits": audits,
            "protected_material_opened": False,
            "p1_checkpoint_used": False,
            "d1_reopened": False,
            "s1_data_mutated": False,
            "s2_data_mutated": False,
        },
    )
    print(
        json.dumps(
            {
                "generated": True,
                "bindable": {k: v["bindable_full_keyed_gap_ge2"] for k, v in audits.items()},
            }
        ),
        flush=True,
    )


def verify() -> dict:
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError("hash mismatch " + rel)
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    assert_d2_licenses_p2()
    return manifest


def bind_fell_half(history: list[float]) -> bool:
    if len(history) < 100:
        return False
    first = statistics.mean(history[:50])
    last = statistics.mean(history[-50:])
    if first <= 0:
        return False
    return last <= 0.5 * first


def residual_moved(treatment_residual: float, parent_residual: float) -> bool:
    return (
        treatment_residual <= SUCCESS_RESIDUAL
        or (parent_residual - treatment_residual) >= MECHANISM_RESIDUAL_DROP
    )


def retention_failures(parent: dict, treat: dict) -> list[dict]:
    failures = []
    p_panels = parent["frozen"]["summaries"]
    t_panels = treat["frozen"]["summaries"]
    checks = (
        ("primitive_induction", "first_top1"),
        ("primitive_keyed", "first_top1"),
        ("short_keyed", "free_exact"),
    )
    for panel, metric in checks:
        p_val = p_panels[panel][metric]
        t_val = t_panels[panel][metric]
        if p_val - t_val > RETENTION_DROP:
            failures.append(
                {
                    "panel": panel,
                    "metric": metric,
                    "parent": p_val,
                    "treatment": t_val,
                    "bar_drop": RETENTION_DROP,
                }
            )
    p_lock = parent["summary"]["rest_lock"]
    t_lock = treat["summary"]["rest_lock"]
    if p_lock - t_lock > RETENTION_DROP:
        failures.append(
            {
                "panel": "diagnostic",
                "metric": "rest_lock",
                "parent": p_lock,
                "treatment": t_lock,
                "bar_drop": RETENTION_DROP,
            }
        )
    return failures


def negative_controls(treat: dict) -> dict:
    iso = treat["isolation"]["summaries"]
    frozen = treat["frozen"]["summaries"]
    value_absent = iso["value_absent_same_surface_novel"]["free_exact"]
    broken_context = frozen["broken_context"]["free_exact"]
    broken_order = frozen["broken_order"]["first_top1"]
    return {
        "value_absent_same_surface_novel.free_exact": {
            "value": value_absent,
            "bar": 0.0,
            "pass": value_absent == 0.0,
        },
        "broken_context.free_exact": {
            "value": broken_context,
            "bar": 0.05,
            "pass": broken_context <= 0.05,
        },
        "broken_order.first_top1": {
            "value": broken_order,
            "bar": 0.20,
            "pass": broken_order <= 0.20,
        },
    }


def decide_verdict(
    *,
    treatment_excess: float,
    control_excess: float,
    ci_lo: float,
    treatment_residual: float,
    parent_residual: float,
    regressions: list,
    bind_halved: bool,
    negative_ok: bool,
) -> str:
    if regressions or not negative_ok:
        return "REGRESSION"
    delta = treatment_excess - control_excess
    if (
        treatment_excess >= SUCCESS_EXCESS
        and delta >= SUCCESS_DELTA
        and ci_lo > 0
        and treatment_residual <= SUCCESS_RESIDUAL
    ):
        return "SUCCESS"
    if bind_halved and treatment_residual >= FALSIFIED_RESIDUAL and treatment_excess < NULL_EXCESS:
        return "FALSIFIED"
    if residual_moved(treatment_residual, parent_residual):
        return "MECHANISM SUPPORTED, DOSE INSUFFICIENT"
    if treatment_excess < NULL_EXCESS:
        return "NULL"
    return "MIXED"


def run(arm: str, seed: int, until: int) -> None:
    import torch
    import torch.nn.functional as F

    from .data import LANG_TRAIN, read_u16
    from .train_v2r4 import capability_optimizer, language_batch, set_seed

    verify()
    if arm not in {"control", "treatment"}:
        raise RuntimeError("arm must be control or treatment")
    dest = OUT / f"{arm}_{seed}"
    if dest.exists():
        raise RuntimeError("refuse overwrite " + str(dest))
    dest.mkdir(parents=True)
    set_seed(seed)
    model, config = model_load()
    model.set_attention_backend("sdpa")
    optimizer, _, _ = capability_optimizer(model)
    stream = torch.tensor(read_u16(LANG_TRAIN), dtype=torch.long)
    dev_items = json.loads(S2_DIAGNOSTIC.read_text(encoding="utf-8"))
    d1b_items = json.loads(D1B_DIAGNOSTIC.read_text(encoding="utf-8"))
    diagnostic = {(r["body_id"], r["query_index"]): r for r in dev_items}
    schedule = json.loads((OUT / f"SCHEDULE_{seed}.json").read_text(encoding="utf-8"))
    lam = BIND_LAMBDA if arm == "treatment" else 0.0
    device = torch.device("cuda")

    start = time.monotonic()
    baseline = measure(model, dev_items, diagnostic, d1b_items)
    write(dest / "eval_0000.json", baseline)
    print(
        json.dumps(
            {
                "arm": arm,
                "step": 0,
                "primary": baseline["gap_strata"]["long"],
                "twin_long": baseline["twin_residual"],
                "ce": baseline["language_dev_ce"],
            }
        ),
        flush=True,
    )
    initial_ce = baseline["language_dev_ce"]
    baseline_long = baseline["gap_strata"]["long"]["excess"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    reason = "terminal"
    last_step = 0
    bind_history: list[float] = []
    for step in range(1, until + 1):
        spec = schedule[step - 1]
        if shutil.disk_usage(ROOT).free < 10 * 2**30:
            reason, last_step = "hard_stop_disk", step - 1
            break
        if time.monotonic() - start > 7200:
            reason, last_step = "hard_stop_runtime", step - 1
            break
        model.train()
        model.set_attention_backend("sdpa")
        optimizer.zero_grad(set_to_none=True)
        bind_value = 0.0
        if spec["task"] == "language":
            x, y = language_batch(stream, random.Random(spec["rng_seed"]), 16, 256, device)
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        else:
            items = spec["items"]
            x, y, mask, _first = pack(items, device)
            hidden = model.forward_hidden(x)
            logits = model.language_head(hidden)
            loss = F.cross_entropy(logits[mask], y[mask])
            if lam > 0:
                extra = residual_bind_loss(hidden, bind_specs(items), tau=BIND_TAU)
                bind_value = float(extra.detach().item())
                loss = loss + lam * extra
                bind_history.append(bind_value)
        if not torch.isfinite(loss):
            reason, last_step = "hard_stop_nonfinite_loss", step
            break
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        if not torch.isfinite(norm):
            reason, last_step = "hard_stop_nonfinite_gradient", step
            break
        optimizer.step()
        last_step = step
        with (dest / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "step": step,
                        "task": spec["task"],
                        "loss": loss.item(),
                        "bind": bind_value,
                        "lambda": lam,
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
                        "bind": bind_value,
                        "elapsed_s": round(time.monotonic() - start),
                    }
                ),
                flush=True,
            )
        if step % 200 == 0:
            report = measure(model, dev_items, diagnostic, d1b_items, full=True)
            write(dest / f"eval_{step:04d}.json", report)
            torch.save(
                {
                    "config": config.to_dict(),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "parent_checkpoint_sha256": PARENT_SHA,
                    "update": 16000 + step,
                    "seed": seed,
                    "arm": arm,
                    "lambda": lam,
                    "protocol": "V010_SELECTION_REPAIR_P2_RESIDUAL_BIND",
                    "protected_material_opened": False,
                    "p1_checkpoint_used": False,
                    "manifest_sha256": digest(OUT / "MANIFEST.json"),
                },
                dest / f"checkpoint_{16000 + step}.pt",
            )
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "step": step,
                        "primary": report["gap_strata"]["long"],
                        "short": report["gap_strata"]["short"],
                        "twin_long": report["twin_residual"],
                        "body_macro": report["summary"]["body_accuracy"],
                        "ce": report["language_dev_ce"],
                    }
                ),
                flush=True,
            )
            if report["language_dev_ce"] > initial_ce + LANGUAGE_CE_HARD:
                reason = "hard_stop_language"
                break
            if step == 400 and arm == "treatment":
                gain = report["gap_strata"]["long"]["excess"] - baseline_long
                resid = report["twin_residual"]["median_long"]
                if gain < FUTILITY_GAIN and resid >= FALSIFIED_RESIDUAL:
                    reason = "futility"
                    break
    write(
        dest / f"RECEIPT_{last_step:04d}.json",
        {
            "reason": reason,
            "last_step": last_step,
            "arm": arm,
            "seed": seed,
            "lambda": lam,
            "parent_sha256": digest(PARENT),
            "manifest_sha256": digest(OUT / "MANIFEST.json"),
            "elapsed_s": time.monotonic() - start,
            "mean_bind_first50": statistics.mean(bind_history[:50]) if len(bind_history) >= 50 else None,
            "mean_bind_last50": statistics.mean(bind_history[-50:]) if bind_history else None,
            "bind_fell_half": bind_fell_half(bind_history),
            "n_bind_terms": len(bind_history),
            "artifacts": {p.name: digest(p) for p in dest.iterdir() if p.is_file()},
        },
    )
    print(json.dumps({"arm": arm, "reason": reason, "last_step": last_step}), flush=True)


def adjudicate() -> dict:
    d2 = assert_d2_licenses_p2()
    diagnostic_items = json.loads(S2_DIAGNOSTIC.read_text(encoding="utf-8"))
    diagnostic = {(r["body_id"], r["query_index"]): r for r in diagnostic_items}
    treat_dir = OUT / f"treatment_{TRAIN_SEED}"
    control_dir = OUT / f"control_{TRAIN_SEED}"
    treat_receipt = max(treat_dir.glob("RECEIPT_*.json"))
    receipt = json.loads(treat_receipt.read_text(encoding="utf-8"))
    last = receipt["last_step"]
    step = last if last % 200 == 0 else (last // 200) * 200
    if step == 0:
        step = last
    treat = json.loads((treat_dir / f"eval_{step:04d}.json").read_text(encoding="utf-8"))
    control = json.loads((control_dir / f"eval_{step:04d}.json").read_text(encoding="utf-8"))
    parent = json.loads((treat_dir / "eval_0000.json").read_text(encoding="utf-8"))
    t_long = treat["gap_strata"]["long"]
    c_long = control["gap_strata"]["long"]
    p_long = parent["gap_strata"]["long"]
    t_res = treat["twin_residual"]["median_long"]
    p_res = parent["twin_residual"]["median_long"]
    c_res = control["twin_residual"]["median_long"]
    ci = bootstrap_delta(
        long_rows(treat, diagnostic),
        long_rows(control, diagnostic),
        seed=BOOTSTRAP_SEED,
    )
    delta = t_long["excess"] - c_long["excess"]
    regressions = retention_failures(parent, treat)
    controls = negative_controls(treat)
    negative_ok = all(row["pass"] for row in controls.values())
    ce_flag = treat["language_dev_ce"] > parent["language_dev_ce"] + LANGUAGE_CE_FLAG
    verdict = decide_verdict(
        treatment_excess=t_long["excess"],
        control_excess=c_long["excess"],
        ci_lo=ci[0],
        treatment_residual=t_res,
        parent_residual=p_res,
        regressions=regressions,
        bind_halved=bool(receipt.get("bind_fell_half")),
        negative_ok=negative_ok,
    )
    decision = {
        "protocol": "V010_SELECTION_REPAIR_P2_RESIDUAL_BIND",
        "licensed_by": "V010_QUERY_PRESENCE_D2",
        "d2_headline": d2["headline"],
        "step": step,
        "reason": receipt["reason"],
        "verdict": verdict,
        "parent_sha256": PARENT_SHA,
        "primary_endpoint": {
            "stratum": "frozen diagnostic rows with query->generation gap >= 13",
            "parent_excess": p_long["excess"],
            "treatment_excess": t_long["excess"],
            "control_excess": c_long["excess"],
            "treatment_minus_control": delta,
            "bootstrap_ci95": list(ci),
        },
        "mechanism": {
            "parent_d1b_long_residual": p_res,
            "treatment_d1b_long_residual": t_res,
            "control_d1b_long_residual": c_res,
            "residual_moved": residual_moved(t_res, p_res),
            "bind_fell_half": bool(receipt.get("bind_fell_half")),
            "mean_bind_first50": receipt.get("mean_bind_first50"),
            "mean_bind_last50": receipt.get("mean_bind_last50"),
        },
        "regressions": regressions,
        "language_ce_flag": ce_flag,
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
            "primitive_keyed_first_top1": {
                "parent": parent["frozen"]["summaries"]["primitive_keyed"]["first_top1"],
                "treatment": treat["frozen"]["summaries"]["primitive_keyed"]["first_top1"],
                "control": control["frozen"]["summaries"]["primitive_keyed"]["first_top1"],
            },
            "short_keyed_free_exact": {
                "parent": parent["frozen"]["summaries"]["short_keyed"]["free_exact"],
                "treatment": treat["frozen"]["summaries"]["short_keyed"]["free_exact"],
                "control": control["frozen"]["summaries"]["short_keyed"]["free_exact"],
            },
            "rest_lock": {
                "parent": parent["summary"]["rest_lock"],
                "treatment": treat["summary"]["rest_lock"],
                "control": control["summary"]["rest_lock"],
            },
        },
        "negative_controls": controls,
        "trained": True,
        "gates_changed": False,
        "protected_material_opened": False,
        "p1_checkpoint_used": False,
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
