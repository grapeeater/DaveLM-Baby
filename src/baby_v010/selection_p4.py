"""P4 direct L0 query-residual copy versus matched λ=0.

Protocol: `design/V010_SELECTION_REPAIR_P4_EARLY_COPY.md`.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
import time
from pathlib import Path

from .selection_p1 import _draw, bootstrap_delta, long_rows
from .selection_p2 import absorb_keyed_row, annotate_keys, bind_fell_half, negative_controls, retention_failures
from .selection_p3 import (
    DEBUG_LOG,
    SESSION,
    denials as p3_denials,
    forward_with_block0,
    measure,
)
from .selection_s1 import PANEL, PARENT, PARENT_SHA, digest, model_load, pack, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/selection_p4"
PROTOCOL = ROOT / "design/V010_SELECTION_REPAIR_P4_EARLY_COPY.md"
S2_DIAGNOSTIC = ROOT / "runs/selection_s2/DIAGNOSTIC.json"
P3_ADJUDICATION = ROOT / "runs/selection_p3/ADJUDICATION_170001_400.json"
P3_SCHEDULES = (
    ROOT / "runs/selection_p3/SCHEDULE_170001.json",
    ROOT / "runs/selection_p3/SCHEDULE_170002.json",
)

TRAIN_SEED = 180001
DATA_SEED = 180100
BOOTSTRAP_SEED = 180300
MAX_UPDATES = 800
KEYED_PER_BATCH = 8
PRIMITIVE_PER_BATCH = 3
INDUCTION_PER_BATCH = 5
BATCH = KEYED_PER_BATCH + PRIMITIVE_PER_BATCH + INDUCTION_PER_BATCH
LANGUAGE_PROBABILITY = 0.20
COPY_LAMBDA = 1.0
MIN_GAP = 2
SUCCESS_EXCESS = 0.10
SUCCESS_DELTA = 0.07
SUCCESS_COSINE = 0.40
SUCCESS_COSINE_GAIN = 0.15
MECHANISM_COSINE = 0.20
MECHANISM_COSINE_GAIN = 0.10
FALSIFIED_COSINE = 0.10
NULL_EXCESS = 0.02
FUTILITY_EXCESS = 0.02
FUTILITY_COSINE = 0.05
LANGUAGE_CE_HARD = 0.20
_COPY_LOGS = 0


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


def assert_p3_licenses_p4() -> dict:
    adj = json.loads(P3_ADJUDICATION.read_text(encoding="utf-8"))
    if adj.get("protocol") != "V010_SELECTION_REPAIR_P3_EARLY_BIND":
        raise RuntimeError("P4 requires P3 adjudication")
    if adj.get("verdict") != "REGRESSION":
        raise RuntimeError("P4 is licensed only by P3 REGRESSION")
    return adj


def denials() -> tuple[set[tuple], set[tuple]]:
    denied_inputs, denied_spans = p3_denials()
    for path in P3_SCHEDULES:
        if not path.exists():
            raise RuntimeError(f"missing P3 schedule {path}")
        for spec in json.loads(path.read_text(encoding="utf-8")):
            if spec.get("task") != "structured":
                continue
            for row in spec["items"]:
                denied_inputs.add(tuple(row["input"]))
    return denied_inputs, denied_spans


def copy_specs(items: list[dict], min_gap: int = MIN_GAP) -> list[tuple[int, int, int]]:
    specs = []
    for i, item in enumerate(items):
        if item.get("kind") != "keyed" or "query_position" not in item:
            continue
        gen = len(item["input"]) - 1
        query = int(item["query_position"])
        if gen - query < min_gap:
            continue
        specs.append((i, gen, query))
    return specs


def residual_copy_loss(hidden, specs: list[tuple[int, int, int]]):
    import torch
    import torch.nn.functional as F

    global _COPY_LOGS
    if not specs:
        # #region agent log
        if _COPY_LOGS < 6:
            debug_log(
                "A",
                "selection_p4.py:residual_copy_loss",
                "empty_copy_specs",
                {"n_specs": 0, "hidden_shape": list(hidden.shape)},
            )
            _COPY_LOGS += 1
        # #endregion
        return hidden.new_zeros(())
    terms = []
    cos_values = []
    for batch_i, gen_pos, query_pos in specs:
        cos = F.cosine_similarity(hidden[batch_i, gen_pos], hidden[batch_i, query_pos].detach(), dim=0)
        cos_values.append(float(cos.detach().item()))
        terms.append(1.0 - cos)
    loss = torch.stack(terms).mean()
    # #region agent log
    if _COPY_LOGS < 6:
        debug_log(
            "A",
            "selection_p4.py:residual_copy_loss",
            "copy_loss",
            {
                "n_specs": len(specs),
                "first_spec": list(specs[0]),
                "hidden_shape": list(hidden.shape),
                "mean_cosine": sum(cos_values) / len(cos_values),
                "min_cosine": min(cos_values),
                "loss": float(loss.detach().item()),
            },
        )
        _COPY_LOGS += 1
    # #endregion
    return loss


def decide_verdict(
    *,
    treatment_excess: float,
    control_excess: float,
    ci_lo: float,
    treatment_cosine: float,
    cosine_gain: float,
    regressions: list,
    copy_halved: bool,
    negative_ok: bool,
) -> str:
    if regressions or not negative_ok:
        return "REGRESSION"
    delta = treatment_excess - control_excess
    mechanism = treatment_cosine >= SUCCESS_COSINE or cosine_gain >= SUCCESS_COSINE_GAIN
    if treatment_excess >= SUCCESS_EXCESS and delta >= SUCCESS_DELTA and ci_lo > 0 and mechanism:
        return "SUCCESS"
    if copy_halved and treatment_cosine < FALSIFIED_COSINE and treatment_excess < NULL_EXCESS:
        return "FALSIFIED"
    if treatment_cosine >= MECHANISM_COSINE or cosine_gain >= MECHANISM_COSINE_GAIN:
        return "MECHANISM SUPPORTED, DOSE INSUFFICIENT"
    if treatment_excess < NULL_EXCESS and cosine_gain < FUTILITY_COSINE:
        return "NULL"
    return "MIXED"


def generate() -> None:
    from .data import LANG_TRAIN, build_banks, read_u16
    from .v2r4_provenance import FROZEN_PANELS_SHA256, require_frozen_file

    require_frozen_file(PANEL, FROZEN_PANELS_SHA256, "panels")
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    p3 = assert_p3_licenses_p4()
    if (OUT / "MANIFEST.json").exists():
        raise RuntimeError("P4 already frozen")
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
        bindable = sum(
            1 for r in keyed_full if len(r["input"]) - 1 - int(r["query_position"]) >= MIN_GAP
        )
        audits[str(seed)] = {
            "updates": len(schedule),
            "language_updates": sum(s["task"] == "language" for s in schedule),
            "keyed_full_rows": len(keyed_full),
            "bindable_full_keyed_gap_ge2": bindable,
            "induction_rows": sum(r["kind"] == "induction" for r in rows),
        }
    files = [
        PROTOCOL,
        Path(__file__),
        PANEL,
        OUT / f"SCHEDULE_{TRAIN_SEED}.json",
        OUT / f"SCHEDULE_{TRAIN_SEED + 1}.json",
        S2_DIAGNOSTIC,
        P3_ADJUDICATION,
        PARENT,
        ROOT / "src/baby_v010/selection_p3.py",
        ROOT / "src/baby_v010/model.py",
    ]
    write(
        OUT / "MANIFEST.json",
        {
            "protocol": "V010_SELECTION_REPAIR_P4_EARLY_COPY",
            "parent_sha256": PARENT_SHA,
            "copy_lambda_treatment": COPY_LAMBDA,
            "copy_lambda_control": 0.0,
            "licensed_by": "V010_SELECTION_REPAIR_P3_EARLY_BIND",
            "p3_verdict": p3["verdict"],
            "bind_layer": 0,
            "batch": {
                "keyed_full": KEYED_PER_BATCH,
                "primitive_keyed": PRIMITIVE_PER_BATCH,
                "induction_full": INDUCTION_PER_BATCH,
            },
            "p1_checkpoint_used": False,
            "p2_launched": False,
            "files": {posix(path): digest(path) for path in files},
            "audits": audits,
            "protected_material_opened": False,
        },
    )
    debug_log("D", "selection_p4.py:generate", "audits", audits)
    print(json.dumps({"generated": True, "audits": audits}), flush=True)


def verify() -> dict:
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError("hash mismatch " + rel)
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    assert_p3_licenses_p4()
    return manifest


def run(arm: str, seed: int, until: int) -> None:
    import torch
    import torch.nn.functional as F

    from .data import LANG_TRAIN, read_u16
    from .train_v2r4 import capability_optimizer, language_batch, set_seed

    verify()
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
    diagnostic = {(r["body_id"], r["query_index"]): r for r in dev_items}
    schedule = json.loads((OUT / f"SCHEDULE_{seed}.json").read_text(encoding="utf-8"))
    lam = COPY_LAMBDA if arm == "treatment" else 0.0
    device = torch.device("cuda")
    start = time.monotonic()
    baseline = measure(model, dev_items, diagnostic)
    write(dest / "eval_0000.json", baseline)
    debug_log("P4", "selection_p4.py:run", "baseline", {"arm": arm, "long": baseline["gap_strata"]["long"], "l0": baseline["l0_cosine"]})
    print(json.dumps({"arm": arm, "step": 0, "primary": baseline["gap_strata"]["long"], "l0": baseline["l0_cosine"]}), flush=True)
    initial_ce = baseline["language_dev_ce"]
    baseline_long = baseline["gap_strata"]["long"]["excess"]
    baseline_l0 = baseline["l0_cosine"]["median_long"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    reason = "terminal"
    last_step = 0
    copy_history: list[float] = []
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
        copy_value = 0.0
        if spec["task"] == "language":
            x, y = language_batch(stream, random.Random(spec["rng_seed"]), 16, 256, device)
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        else:
            items = spec["items"]
            x, y, mask, _first = pack(items, device)
            hidden, h0 = forward_with_block0(model, x)
            logits = model.language_head(hidden)
            loss = F.cross_entropy(logits[mask], y[mask])
            if lam > 0:
                specs = copy_specs(items)
                extra = residual_copy_loss(h0, specs)
                copy_value = float(extra.detach().item())
                loss = loss + lam * extra
                copy_history.append(copy_value)
                if len(copy_history) <= 3:
                    debug_log(
                        "A",
                        "selection_p4.py:run",
                        "first_copy_batches",
                        {
                            "arm": arm,
                            "step": step,
                            "n_specs": len(specs),
                            "copy": copy_value,
                            "seq_lens": [len(item["input"]) for item in items],
                            "gaps": [len(item["input"]) - 1 - int(item["query_position"]) for item in items if item.get("kind") == "keyed" and "query_position" in item],
                        },
                    )
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
            handle.write(json.dumps({"step": step, "task": spec["task"], "loss": loss.item(), "copy": copy_value, "lambda": lam, "gradient_norm": float(norm)}) + "\n")
        if step % 50 == 0:
            print(json.dumps({"arm": arm, "step": step, "loss": loss.item(), "copy": copy_value, "elapsed_s": round(time.monotonic() - start)}), flush=True)
        if step % 200 == 0:
            report = measure(model, dev_items, diagnostic, full=True)
            write(dest / f"eval_{step:04d}.json", report)
            torch.save(
                {
                    "config": config.to_dict(),
                    "model_state_dict": model.state_dict(),
                    "parent_checkpoint_sha256": PARENT_SHA,
                    "update": 16000 + step,
                    "seed": seed,
                    "arm": arm,
                    "lambda": lam,
                    "protocol": "V010_SELECTION_REPAIR_P4_EARLY_COPY",
                    "protected_material_opened": False,
                    "p2_launched": False,
                    "manifest_sha256": digest(OUT / "MANIFEST.json"),
                },
                dest / f"checkpoint_{16000 + step}.pt",
            )
            debug_log("P4", "selection_p4.py:run", "eval", {"arm": arm, "step": step, "long": report["gap_strata"]["long"], "l0": report["l0_cosine"]})
            print(json.dumps({"arm": arm, "step": step, "primary": report["gap_strata"]["long"], "l0": report["l0_cosine"], "ce": report["language_dev_ce"]}), flush=True)
            if report["language_dev_ce"] > initial_ce + LANGUAGE_CE_HARD:
                reason = "hard_stop_language"
                break
            if step == 400 and arm == "treatment":
                gain = report["gap_strata"]["long"]["excess"] - baseline_long
                cosine_gain = report["l0_cosine"]["median_long"] - baseline_l0
                if gain < FUTILITY_EXCESS and cosine_gain < FUTILITY_COSINE:
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
            "mean_copy_first50": statistics.mean(copy_history[:50]) if len(copy_history) >= 50 else None,
            "mean_copy_last50": statistics.mean(copy_history[-50:]) if copy_history else None,
            "bind_fell_half": bind_fell_half(copy_history),
            "artifacts": {p.name: digest(p) for p in dest.iterdir() if p.is_file()},
        },
    )
    print(json.dumps({"arm": arm, "reason": reason, "last_step": last_step}), flush=True)


def adjudicate() -> dict:
    p3 = assert_p3_licenses_p4()
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
    ci = bootstrap_delta(long_rows(treat, diagnostic), long_rows(control, diagnostic), seed=BOOTSTRAP_SEED)
    regressions = retention_failures(parent, treat)
    controls = negative_controls(treat)
    verdict = decide_verdict(
        treatment_excess=t_long["excess"],
        control_excess=c_long["excess"],
        ci_lo=ci[0],
        treatment_cosine=t_cos,
        cosine_gain=t_cos - p_cos,
        regressions=regressions,
        copy_halved=bool(receipt.get("bind_fell_half")),
        negative_ok=all(row["pass"] for row in controls.values()),
    )
    decision = {
        "protocol": "V010_SELECTION_REPAIR_P4_EARLY_COPY",
        "licensed_by": "V010_SELECTION_REPAIR_P3_EARLY_BIND",
        "p3_verdict": p3["verdict"],
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
