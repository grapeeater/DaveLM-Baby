"""P3 early residual query-bind at block 0 versus a matched λ=0 control.

Protocol: `design/V010_SELECTION_REPAIR_P3_EARLY_BIND.md`.
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

from .query_presence import UNIFORM_TRACK_MULTIPLIER, median
from .selection_p1 import _draw, bootstrap_delta, denials as p1_denials, long_rows
from .selection_p2 import (
    absorb_keyed_row,
    annotate_keys,
    bind_fell_half,
    negative_controls,
    residual_bind_loss,
    retention_failures,
)
from .selection_s1 import PANEL, PARENT, PARENT_SHA, digest, model_load, pack, write
from .selection_t1 import gap_strata, salience

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/selection_p3"
PROTOCOL = ROOT / "design/V010_SELECTION_REPAIR_P3_EARLY_BIND.md"
S2_DIAGNOSTIC = ROOT / "runs/selection_s2/DIAGNOSTIC.json"
D3B_ADJUDICATION = ROOT / "runs/query_splice_d3b/ADJUDICATION.json"
P1_SCHEDULES = (
    ROOT / "runs/selection_p1/SCHEDULE_150001.json",
    ROOT / "runs/selection_p1/SCHEDULE_150002.json",
)
DEBUG_LOG = ROOT / "debug-145edc.log"
SESSION = "145edc"

TRAIN_SEED = 170001
DATA_SEED = 170100
BOOTSTRAP_SEED = 170300
MAX_UPDATES = 800
KEYED_PER_BATCH = 10
PRIMITIVE_PER_BATCH = 3
INDUCTION_PER_BATCH = 3
BATCH = KEYED_PER_BATCH + PRIMITIVE_PER_BATCH + INDUCTION_PER_BATCH
LANGUAGE_PROBABILITY = 0.20
BIND_LAMBDA = 0.25
BIND_TAU = 0.10
MIN_GAP = 2
BIND_LAYER = 0
RETENTION_DROP = 0.05
SUCCESS_EXCESS = 0.10
SUCCESS_DELTA = 0.07
SUCCESS_COSINE_GAIN = 0.10
SUCCESS_EARLY_TRACK = 0.50
MECHANISM_COSINE_GAIN = 0.05
MECHANISM_TRACK_GAIN = 0.20
NULL_EXCESS = 0.02
FUTILITY_GAIN = 0.02
LANGUAGE_CE_FLAG = 0.05
LANGUAGE_CE_HARD = 0.20
EARLY_LAYERS = 3


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


def assert_d3b_licenses_p3() -> dict:
    if not D3B_ADJUDICATION.exists():
        raise RuntimeError("D3b adjudication missing")
    adj = json.loads(D3B_ADJUDICATION.read_text(encoding="utf-8"))
    if adj.get("protocol") != "V010_QUERY_SPLICE_D3B":
        raise RuntimeError("D3b protocol mismatch")
    if adj.get("verdict") != "SPLICED_IDENTITY":
        raise RuntimeError("P3 is licensed only by D3b SPLICED_IDENTITY")
    return adj


def denials() -> tuple[set[tuple], set[tuple]]:
    denied_inputs, denied_spans = p1_denials()
    for path in P1_SCHEDULES:
        if not path.exists():
            raise RuntimeError(f"P3 refuses to generate without P1 schedule {path}")
        for spec in json.loads(path.read_text(encoding="utf-8")):
            if spec.get("task") != "structured":
                continue
            for row in spec["items"]:
                denied_inputs.add(tuple(row["input"]))
                if row.get("kind") == "keyed":
                    absorb_keyed_row(denied_inputs, denied_spans, row)
    return denied_inputs, denied_spans


def competitor_key_positions(item: dict) -> list[int]:
    from .isolation_transforms import recover_rendered_pairs

    query_key = int(item["query_key"])
    positions = []
    seen = set()
    for row in recover_rendered_pairs(item):
        if int(row["original_key"]) == query_key or int(row["key"]) == query_key:
            continue
        start = int(row["start"])
        if start not in seen:
            positions.append(start)
            seen.add(start)
    return positions


def bind_specs(items: list[dict], min_gap: int = MIN_GAP) -> list[tuple[int, int, list[int]]]:
    specs = []
    for i, item in enumerate(items):
        if item.get("kind") != "keyed" or "query_position" not in item:
            continue
        gen = len(item["input"]) - 1
        query = int(item["query_position"])
        if gen - query < min_gap:
            continue
        competitors = [
            pos for pos in competitor_key_positions(item) if 0 <= pos <= gen and pos != query
        ]
        if not competitors:
            continue
        specs.append((i, gen, [query, *competitors]))
    return specs


def forward_with_block0(model, tokens):
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output)
        return output

    handle = model.blocks[BIND_LAYER].register_forward_hook(hook)
    try:
        hidden = model.forward_hidden(tokens)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("block 0 hook missed")
    return hidden, captured[0]


def l0_query_cosines(model, items, device) -> dict:
    import torch
    import torch.nn.functional as F

    long_c, short_c = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(items), 8):
            batch = items[start : start + 8]
            x, _, _, _ = pack(batch, device)
            _hidden, h0 = forward_with_block0(model, x)
            for i, item in enumerate(batch):
                if item.get("kind") != "keyed" or "query_position" not in item:
                    continue
                gen = len(item["input"]) - 1
                query = int(item["query_position"])
                value = float(F.cosine_similarity(h0[i, gen], h0[i, query], dim=0).item())
                gap = gen - query
                if gap >= 13:
                    long_c.append(value)
                if gap <= 1:
                    short_c.append(value)
    return {
        "n_long": len(long_c),
        "median_long": median(long_c),
        "n_short": len(short_c),
        "median_short": median(short_c),
    }


def early_pointer_stats(model, items, device) -> dict:
    import torch

    from .query_locality import bucket as gap_bucket

    model.eval()
    model.set_attention_backend("reference")
    rows = []
    with torch.no_grad():
        for start in range(0, len(items), 8):
            batch = items[start : start + 8]
            captured: list = []
            model.set_attention_capture(captured)
            x, _, _, _ = pack(batch, device)
            model(x)
            model.set_attention_capture(None)
            for i, item in enumerate(batch):
                if item.get("kind") != "keyed" or "query_position" not in item:
                    continue
                gen = len(item["input"]) - 1
                query = int(item["query_position"])
                masses = [float(w[i, :, gen, query].max().item()) for w in captured[:EARLY_LAYERS]]
                mx = max(masses) if masses else 0.0
                uniform = 1.0 / (gen + 1)
                rows.append(
                    {
                        "gap": gen - query,
                        "bucket": gap_bucket(gen - query),
                        "max_query_mass": mx,
                        "tracks_query": mx >= UNIFORM_TRACK_MULTIPLIER * uniform,
                    }
                )
    model.set_attention_backend("sdpa")

    def subset(pred):
        chosen = [row for row in rows if pred(row)]
        if not chosen:
            return {"n": 0}
        return {
            "n": len(chosen),
            "fraction_any_query_tracking_head": sum(float(r["tracks_query"]) for r in chosen)
            / len(chosen),
            "median_max_query_mass": statistics.median([r["max_query_mass"] for r in chosen]),
        }

    return {"all": subset(lambda r: True), "long": subset(lambda r: r["gap"] >= 13)}


def measure(model, dev_items, diagnostic, full: bool = True) -> dict:
    from .selection_s2 import measure as s2_measure

    report = s2_measure(model, dev_items, full=full)
    report["gap_strata"] = gap_strata(report["rows"], diagnostic)
    report["summary"].update(salience(report["rows"]))
    report["summary"]["primary_long_gap_excess"] = report["gap_strata"]["long"]["excess"]
    device = next(model.parameters()).device
    report["l0_cosine"] = l0_query_cosines(model, dev_items, device)
    report["early_pointer"] = early_pointer_stats(model, dev_items, device)
    report["summary"]["l0_long_query_cosine"] = report["l0_cosine"]["median_long"]
    report["summary"]["early_long_query_track"] = report["early_pointer"]["long"].get(
        "fraction_any_query_tracking_head", float("nan")
    )
    return report


def generate() -> None:
    from .data import LANG_TRAIN, build_banks, read_u16
    from .v2r4_provenance import FROZEN_PANELS_SHA256, require_frozen_file

    require_frozen_file(PANEL, FROZEN_PANELS_SHA256, "panels")
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    d3b = assert_d3b_licenses_p3()
    if (OUT / "MANIFEST.json").exists():
        raise RuntimeError("P3 already frozen")
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
                denied_inputs.add(tuple(ind["input"]))
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
            "bindable_full_keyed_gap_ge2": bindable,
            "natural_gap_histogram": dict(sorted(gap_hist.items(), key=lambda kv: int(kv[0]))),
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
        D3B_ADJUDICATION,
        *P1_SCHEDULES,
        ROOT / "src/baby_v010/data.py",
        ROOT / "src/baby_v010/data_v2.py",
        ROOT / "src/baby_v010/model.py",
        ROOT / "src/baby_v010/selection_p2.py",
        ROOT / "src/baby_v010/selection_p1.py",
    ]
    write(
        OUT / "MANIFEST.json",
        {
            "protocol": "V010_SELECTION_REPAIR_P3_EARLY_BIND",
            "parent_sha256": PARENT_SHA,
            "bind_lambda_treatment": BIND_LAMBDA,
            "bind_lambda_control": 0.0,
            "bind_tau": BIND_TAU,
            "min_gap": MIN_GAP,
            "bind_layer": BIND_LAYER,
            "licensed_by": "V010_QUERY_SPLICE_D3B",
            "d3b_verdict": d3b["verdict"],
            "language_probability": LANGUAGE_PROBABILITY,
            "natural_gaps": True,
            "p1_checkpoint_used": False,
            "p2_launched": False,
            "files": {posix(path): digest(path) for path in files},
            "audits": audits,
            "protected_material_opened": False,
        },
    )
    print(json.dumps({"generated": True, "bindable": {k: v["bindable_full_keyed_gap_ge2"] for k, v in audits.items()}}), flush=True)


def verify() -> dict:
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError("hash mismatch " + rel)
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    assert_d3b_licenses_p3()
    return manifest


def decide_verdict(
    *,
    treatment_excess: float,
    control_excess: float,
    ci_lo: float,
    cosine_gain: float,
    early_track: float,
    early_track_gain: float,
    regressions: list,
    bind_halved: bool,
    negative_ok: bool,
) -> str:
    if regressions or not negative_ok:
        return "REGRESSION"
    delta = treatment_excess - control_excess
    mechanism = cosine_gain >= SUCCESS_COSINE_GAIN or early_track >= SUCCESS_EARLY_TRACK
    if (
        treatment_excess >= SUCCESS_EXCESS
        and delta >= SUCCESS_DELTA
        and ci_lo > 0
        and mechanism
    ):
        return "SUCCESS"
    if bind_halved and cosine_gain < FUTILITY_GAIN and treatment_excess < NULL_EXCESS:
        return "FALSIFIED"
    if cosine_gain >= MECHANISM_COSINE_GAIN or early_track_gain >= MECHANISM_TRACK_GAIN:
        return "MECHANISM SUPPORTED, DOSE INSUFFICIENT"
    if treatment_excess < NULL_EXCESS and cosine_gain < MECHANISM_COSINE_GAIN:
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
    diagnostic = {(r["body_id"], r["query_index"]): r for r in dev_items}
    schedule = json.loads((OUT / f"SCHEDULE_{seed}.json").read_text(encoding="utf-8"))
    lam = BIND_LAMBDA if arm == "treatment" else 0.0
    device = torch.device("cuda")
    start = time.monotonic()
    baseline = measure(model, dev_items, diagnostic)
    write(dest / "eval_0000.json", baseline)
    debug_log("H1", "selection_p3.py:run", "p3_baseline", {"arm": arm, "long": baseline["gap_strata"]["long"], "l0": baseline["l0_cosine"]})
    print(json.dumps({"arm": arm, "step": 0, "primary": baseline["gap_strata"]["long"], "l0": baseline["l0_cosine"]}), flush=True)
    initial_ce = baseline["language_dev_ce"]
    baseline_long = baseline["gap_strata"]["long"]["excess"]
    baseline_l0 = baseline["l0_cosine"]["median_long"]
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
            hidden, h0 = forward_with_block0(model, x)
            logits = model.language_head(hidden)
            loss = F.cross_entropy(logits[mask], y[mask])
            if lam > 0:
                extra = residual_bind_loss(h0, bind_specs(items), tau=BIND_TAU)
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
            handle.write(json.dumps({"step": step, "task": spec["task"], "loss": loss.item(), "bind": bind_value, "lambda": lam, "gradient_norm": float(norm)}) + "\n")
        if step % 50 == 0:
            print(json.dumps({"arm": arm, "step": step, "loss": loss.item(), "bind": bind_value, "elapsed_s": round(time.monotonic() - start)}), flush=True)
        if step % 200 == 0:
            report = measure(model, dev_items, diagnostic, full=True)
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
                    "protocol": "V010_SELECTION_REPAIR_P3_EARLY_BIND",
                    "protected_material_opened": False,
                    "p1_checkpoint_used": False,
                    "p2_launched": False,
                    "manifest_sha256": digest(OUT / "MANIFEST.json"),
                },
                dest / f"checkpoint_{16000 + step}.pt",
            )
            debug_log("H1", "selection_p3.py:run", "p3_eval", {"arm": arm, "step": step, "long": report["gap_strata"]["long"], "l0": report["l0_cosine"], "early": report["early_pointer"]["long"]})
            print(json.dumps({"arm": arm, "step": step, "primary": report["gap_strata"]["long"], "l0": report["l0_cosine"], "ce": report["language_dev_ce"]}), flush=True)
            if report["language_dev_ce"] > initial_ce + LANGUAGE_CE_HARD:
                reason = "hard_stop_language"
                break
            if step == 400 and arm == "treatment":
                gain = report["gap_strata"]["long"]["excess"] - baseline_long
                cosine_gain = report["l0_cosine"]["median_long"] - baseline_l0
                if gain < FUTILITY_GAIN and cosine_gain < FUTILITY_GAIN:
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
    d3b = assert_d3b_licenses_p3()
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
    t_cos = treat["l0_cosine"]["median_long"]
    p_cos = parent["l0_cosine"]["median_long"]
    c_cos = control["l0_cosine"]["median_long"]
    t_track = treat["early_pointer"]["long"].get("fraction_any_query_tracking_head", 0.0)
    p_track = parent["early_pointer"]["long"].get("fraction_any_query_tracking_head", 0.0)
    ci = bootstrap_delta(long_rows(treat, diagnostic), long_rows(control, diagnostic), seed=BOOTSTRAP_SEED)
    regressions = retention_failures(parent, treat)
    controls = negative_controls(treat)
    negative_ok = all(row["pass"] for row in controls.values())
    verdict = decide_verdict(
        treatment_excess=t_long["excess"],
        control_excess=c_long["excess"],
        ci_lo=ci[0],
        cosine_gain=t_cos - p_cos,
        early_track=t_track,
        early_track_gain=t_track - p_track,
        regressions=regressions,
        bind_halved=bool(receipt.get("bind_fell_half")),
        negative_ok=negative_ok,
    )
    decision = {
        "protocol": "V010_SELECTION_REPAIR_P3_EARLY_BIND",
        "licensed_by": "V010_QUERY_SPLICE_D3B",
        "d3b_verdict": d3b["verdict"],
        "step": step,
        "reason": receipt["reason"],
        "verdict": verdict,
        "parent_sha256": PARENT_SHA,
        "primary_endpoint": {
            "parent_excess": p_long["excess"],
            "treatment_excess": t_long["excess"],
            "control_excess": c_long["excess"],
            "treatment_minus_control": t_long["excess"] - c_long["excess"],
            "bootstrap_ci95": list(ci),
            "parent_hit": p_long["hit"],
            "treatment_hit": t_long["hit"],
            "control_hit": c_long["hit"],
        },
        "mechanism": {
            "parent_l0_cosine": p_cos,
            "treatment_l0_cosine": t_cos,
            "control_l0_cosine": c_cos,
            "cosine_gain": t_cos - p_cos,
            "treatment_early_track": t_track,
            "parent_early_track": p_track,
            "early_track_gain": t_track - p_track,
            "bind_fell_half": bool(receipt.get("bind_fell_half")),
        },
        "regressions": regressions,
        "negative_controls": controls,
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
        "trained": True,
        "p2_launched": False,
        "p1_checkpoint_used": False,
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
