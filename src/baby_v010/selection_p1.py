"""P1 pointer-attention auxiliary versus a matched λ=0 control.

Protocol: `design/V010_SELECTION_REPAIR_P1_POINTER.md` (frozen before generation).

Both arms train on the same items in the same order with the same answer-span
CE. The only difference is λ on the pointer term. Natural gaps. No T1 resume.
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

from .query_presence import UNIFORM_TRACK_MULTIPLIER
from .selection_s1 import PANEL, PARENT, PARENT_SHA, digest, model_load, pack, write
from .selection_t1 import gap_strata, salience

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/selection_p1"
PROTOCOL = ROOT / "design/V010_SELECTION_REPAIR_P1_POINTER.md"
S1_DIAGNOSTIC = ROOT / "runs/selection_s1/DIAGNOSTIC.json"
S2_DIAGNOSTIC = ROOT / "runs/selection_s2/DIAGNOSTIC.json"
S2_MANIFEST = ROOT / "runs/selection_s2/MANIFEST.json"
D1B_DIAGNOSTIC = ROOT / "runs/query_presence_d1b/DIAGNOSTIC.json"
D1B_ADJUDICATION = ROOT / "runs/query_presence_d1b/ADJUDICATION.json"

TRAIN_SEED = 150001
DATA_SEED = 150100
BOOTSTRAP_SEED = 150300
MAX_UPDATES = 800
KEYED_PER_BATCH = 10
PRIMITIVE_PER_BATCH = 3
INDUCTION_PER_BATCH = 3
BATCH = KEYED_PER_BATCH + PRIMITIVE_PER_BATCH + INDUCTION_PER_BATCH
LANGUAGE_PROBABILITY = 0.20
POINTER_LAMBDA = 0.25
POINTER_EPS = 1e-8
N_LAYERS = 12


def posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def assert_d1b_licenses_p1() -> dict:
    if not D1B_ADJUDICATION.exists():
        raise RuntimeError("D1b adjudication missing; P1 is licensed only by D1b B")
    adj = json.loads(D1B_ADJUDICATION.read_text(encoding="utf-8"))
    if adj.get("protocol") != "V010_QUERY_PRESENCE_D1B":
        raise RuntimeError("D1b adjudication protocol mismatch")
    if not adj.get("valid") or adj.get("verdict") != "B":
        raise RuntimeError("P1 pointer treatment is licensed only by a valid D1b B")
    return adj


def denials() -> tuple[set[tuple], set[tuple]]:
    from .isolation_transforms import parse_records

    denied_inputs: set[tuple] = set()
    denied_spans: set[tuple] = set()

    def absorb(rows) -> None:
        for row in rows:
            denied_inputs.add(tuple(row["input"]))
            denied_spans.add(tuple(row["target_span"]))
            if row.get("kind") == "keyed":
                denied_spans.update(tuple(value) for _, value in parse_records(row))

    for group in json.loads(PANEL.read_text(encoding="utf-8")).values():
        absorb(group)
    for path in (S1_DIAGNOSTIC, S2_DIAGNOSTIC, D1B_DIAGNOSTIC):
        if not path.exists():
            raise RuntimeError(f"missing prior diagnostic {path}; refusing to generate")
        absorb(json.loads(path.read_text(encoding="utf-8")))
    return denied_inputs, denied_spans


def _draw(rng, banks, denied_inputs, denied_spans, *, kind, difficulty):
    from .data_v2 import make_item
    from .isolation_transforms import locate_query, parse_records

    for _ in range(20000):
        item = make_item(
            rng,
            banks,
            kind=kind,
            difficulty=difficulty,
            low_prior=rng.random() < 0.20,
        )
        if tuple(item["input"]) in denied_inputs:
            continue
        if tuple(item["target_span"]) in denied_spans:
            continue
        if kind == "keyed":
            pairs = parse_records(item)
            if any(tuple(value) in denied_spans for _, value in pairs):
                continue
            if len({value[0] for _, value in pairs}) != len(pairs):
                continue
            _body, query = locate_query(item)
            if query is None:
                continue
            item["query_position"] = int(query)
            item["candidate_heads"] = [int(value[0]) for _, value in pairs]
        return item
    raise RuntimeError("could not draw a legal item")


def generate() -> None:
    from .data import LANG_TRAIN, build_banks, read_u16
    from .v2r4_provenance import FROZEN_PANELS_SHA256, require_frozen_file

    require_frozen_file(PANEL, FROZEN_PANELS_SHA256, "panels")
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    d1b = assert_d1b_licenses_p1()
    if (OUT / "MANIFEST.json").exists():
        raise RuntimeError("P1 already frozen")
    banks = build_banks(read_u16(LANG_TRAIN))
    denied_inputs, denied_spans = denials()

    audits = {}
    for seed in (TRAIN_SEED, TRAIN_SEED + 1):
        rng = random.Random(DATA_SEED + seed - TRAIN_SEED)
        schedule: list[dict] = []
        gap_hist: dict[str, int] = defaultdict(int)
        for update in range(1, MAX_UPDATES + 1):
            if rng.random() < LANGUAGE_PROBABILITY:
                schedule.append({"task": "language", "rng_seed": rng.randrange(2**31)})
                continue
            items: list[dict] = []
            for _ in range(KEYED_PER_BATCH):
                item = _draw(
                    rng, banks, denied_inputs, denied_spans, kind="keyed", difficulty="full"
                )
                gap = len(item["input"]) - 1 - int(item["query_position"])
                gap_hist[str(gap)] += 1
                denied_inputs.add(tuple(item["input"]))
                denied_spans.add(tuple(item["target_span"]))
                items.append(item)
            for _ in range(PRIMITIVE_PER_BATCH):
                items.append(
                    _draw(
                        rng, banks, denied_inputs, denied_spans,
                        kind="keyed", difficulty="primitive",
                    )
                )
            for _ in range(INDUCTION_PER_BATCH):
                items.append(
                    _draw(
                        rng, banks, denied_inputs, denied_spans,
                        kind="induction", difficulty="full",
                    )
                )
            assert len(items) == BATCH
            schedule.append({"task": "structured", "update": update, "items": items})
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
        D1B_ADJUDICATION,
        D1B_DIAGNOSTIC,
        ROOT / "src/baby_v010/data.py",
        ROOT / "src/baby_v010/data_v2.py",
        ROOT / "src/baby_v010/model.py",
        ROOT / "src/baby_v010/config.py",
        ROOT / "src/baby_v010/evaluate.py",
        ROOT / "src/baby_v010/train_v2r4.py",
        ROOT / "src/baby_v010/selection_s1.py",
        ROOT / "src/baby_v010/selection_t1.py",
    ]
    write(
        OUT / "MANIFEST.json",
        {
            "protocol": "V010_SELECTION_REPAIR_P1_POINTER",
            "parent_sha256": PARENT_SHA,
            "pointer_lambda_treatment": POINTER_LAMBDA,
            "pointer_lambda_control": 0.0,
            "licensed_by": "V010_QUERY_PRESENCE_D1B",
            "d1b_verdict": d1b["verdict"],
            "batch_composition": {
                "keyed_full": KEYED_PER_BATCH,
                "primitive_keyed": PRIMITIVE_PER_BATCH,
                "induction_full": INDUCTION_PER_BATCH,
            },
            "language_probability": LANGUAGE_PROBABILITY,
            "natural_gaps": True,
            "files": {posix(path): digest(path) for path in files},
            "audits": audits,
            "protected_material_opened": False,
            "d1_reopened": False,
            "s1_data_mutated": False,
            "s2_data_mutated": False,
        },
    )
    print(json.dumps({"audits": {k: {kk: vv for kk, vv in v.items() if kk != "natural_gap_histogram"} for k, v in audits.items()}}), flush=True)


def verify() -> dict:
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError("hash mismatch " + rel)
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    assert_d1b_licenses_p1()
    return manifest


def keyed_pointers(items: list[dict]) -> list[tuple[int, int, int]]:
    pointers = []
    for i, row in enumerate(items):
        if row.get("kind") != "keyed":
            continue
        if "query_position" not in row:
            continue
        gen_pos = len(row["input"]) - 1
        query_pos = int(row["query_position"])
        if 0 <= query_pos <= gen_pos:
            pointers.append((i, gen_pos, query_pos))
    return pointers


def pointer_loss(weights: list, pointers: list[tuple[int, int, int]], eps: float = POINTER_EPS):
    import torch

    if not pointers or not weights:
        return weights[0].new_zeros(()) if weights else None
    terms = []
    for batch_i, gen_pos, query_pos in pointers:
        layer_max = [w[batch_i, :, gen_pos, query_pos].max() for w in weights]
        mass = torch.stack(layer_max).max()
        terms.append(-torch.log(mass + eps))
    return torch.stack(terms).mean()


def pointer_stats(model, items, device) -> dict:
    import torch

    from .query_locality import bucket as gap_bucket

    model.eval()
    model.set_attention_backend("reference")
    rows = []
    with torch.no_grad():
        for start in range(0, len(items), 16):
            batch = items[start : start + 16]
            captured: list = []
            model.set_attention_capture(captured)
            x, _, _, _ = pack(batch, device)
            model(x)
            model.set_attention_capture(None)
            if len(captured) != N_LAYERS:
                raise RuntimeError("expected one attention map per layer")
            for i, item in enumerate(batch):
                if item.get("kind") != "keyed" or "query_position" not in item:
                    continue
                gen_pos = len(item["input"]) - 1
                query_pos = int(item["query_position"])
                gap = gen_pos - query_pos
                masses = [w[i, :, gen_pos, query_pos].max().item() for w in captured]
                mx = max(masses)
                uniform = 1.0 / (gen_pos + 1)
                rows.append(
                    {
                        "body_id": item["body_id"],
                        "query_index": int(item["query_index"]),
                        "gap": gap,
                        "bucket": gap_bucket(gap),
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
            "fraction_any_query_tracking_head": mean_f([float(r["tracks_query"]) for r in chosen]),
            "median_max_query_mass": statistics.median([r["max_query_mass"] for r in chosen]),
        }

    return {
        "all": subset(lambda r: True),
        "g0_1": subset(lambda r: r["gap"] <= 1),
        "long": subset(lambda r: r["gap"] >= 13),
        "g31p": subset(lambda r: r["gap"] >= 31),
    }


def mean_f(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def measure(model, dev_items, diagnostic, full: bool = True) -> dict:
    from .selection_s2 import measure as s2_measure

    report = s2_measure(model, dev_items, full=full)
    report["gap_strata"] = gap_strata(report["rows"], diagnostic)
    report["summary"].update(salience(report["rows"]))
    report["summary"]["primary_long_gap_excess"] = report["gap_strata"]["long"]["excess"]
    report["pointer"] = pointer_stats(model, dev_items, next(model.parameters()).device)
    report["summary"]["long_gap_query_track"] = report["pointer"]["long"].get(
        "fraction_any_query_tracking_head", float("nan")
    )
    return report


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
    model.set_attention_backend("reference")
    optimizer, _, _ = capability_optimizer(model)
    stream = torch.tensor(read_u16(LANG_TRAIN), dtype=torch.long)
    dev_items = json.loads(S2_DIAGNOSTIC.read_text(encoding="utf-8"))
    diagnostic = {(r["body_id"], r["query_index"]): r for r in dev_items}
    schedule = json.loads((OUT / f"SCHEDULE_{seed}.json").read_text(encoding="utf-8"))
    lam = POINTER_LAMBDA if arm == "treatment" else 0.0

    start = time.monotonic()
    baseline = measure(model, dev_items, diagnostic)
    write(dest / "eval_0000.json", baseline)
    print(
        json.dumps(
            {
                "arm": arm,
                "step": 0,
                "primary": baseline["gap_strata"]["long"],
                "pointer_long": baseline["pointer"]["long"],
                "ce": baseline["language_dev_ce"],
            }
        ),
        flush=True,
    )
    initial_ce = baseline["language_dev_ce"]
    baseline_long = baseline["gap_strata"]["long"]["excess"]
    baseline_track = baseline["pointer"]["long"].get("fraction_any_query_tracking_head", 0.0)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    reason = "terminal"
    last_step = 0
    ptr_window: list[float] = []
    for step in range(1, until + 1):
        spec = schedule[step - 1]
        if shutil.disk_usage(ROOT).free < 10 * 2**30:
            reason, last_step = "hard_stop_disk", step - 1
            break
        if time.monotonic() - start > 7200:
            reason, last_step = "hard_stop_runtime", step - 1
            break
        model.train()
        model.set_attention_backend("reference")
        optimizer.zero_grad(set_to_none=True)
        pointer_value = 0.0
        if spec["task"] == "language":
            x, y = language_batch(
                stream, random.Random(spec["rng_seed"]), 16, 256, torch.device("cuda")
            )
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        else:
            items = spec["items"]
            x, y, mask, _first = pack(items, torch.device("cuda"))
            captured: list = []
            if lam > 0:
                model.set_attention_capture(captured)
            logits = model(x)
            model.set_attention_capture(None)
            loss = F.cross_entropy(logits[mask], y[mask])
            if lam > 0:
                extra = pointer_loss(captured, keyed_pointers(items))
                if extra is not None:
                    pointer_value = float(extra.detach().item())
                    loss = loss + lam * extra
                    ptr_window.append(pointer_value)
                    if len(ptr_window) > 50:
                        ptr_window.pop(0)
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
                        "pointer": pointer_value,
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
                        "pointer": pointer_value,
                        "elapsed_s": round(time.monotonic() - start),
                    }
                ),
                flush=True,
            )
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
                    "protocol": "V010_SELECTION_REPAIR_P1_POINTER",
                    "protected_material_opened": False,
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
                        "pointer_long": report["pointer"]["long"],
                        "body_macro": report["summary"]["body_accuracy"],
                        "ce": report["language_dev_ce"],
                    }
                ),
                flush=True,
            )
            if report["language_dev_ce"] > initial_ce + 0.20:
                reason = "hard_stop_language"
                break
            if step == 400 and arm == "treatment":
                gain = report["gap_strata"]["long"]["excess"] - baseline_long
                track_now = report["pointer"]["long"].get("fraction_any_query_tracking_head", 0.0)
                track_gain = track_now - baseline_track
                if gain < 0.02 and track_gain < 0.10:
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
            "mean_pointer_last50": statistics.mean(ptr_window) if ptr_window else None,
            "artifacts": {p.name: digest(p) for p in dest.iterdir() if p.is_file()},
        },
    )
    print(json.dumps({"arm": arm, "reason": reason, "last_step": last_step}), flush=True)


def bootstrap_delta(treat_rows, control_rows, n: int = 10000, seed: int = BOOTSTRAP_SEED):
    rng = random.Random(seed)
    treat = defaultdict(list)
    control = defaultdict(list)
    for row in treat_rows:
        treat[row["body_id"]].append(row)
    for row in control_rows:
        control[row["body_id"]].append(row)
    bodies = sorted(set(treat) & set(control))

    def excess(store, chosen):
        hit = n_rows = chance = 0.0
        for body in chosen:
            for row in store[body]:
                if row.get("gap", 0) < 13 and "gap" in row:
                    continue
                # rows from s2 diagnostic don't include gap; caller must pass long rows only
                hit += float(row["inventory_correct"])
                n_rows += 1
                chance += 1.0 / row["K"]
        if n_rows == 0:
            return 0.0
        return hit / n_rows - chance / n_rows

    # Use only pre-filtered long-gap rows.
    deltas = []
    for _ in range(n):
        chosen = [bodies[rng.randrange(len(bodies))] for _ in bodies]
        deltas.append(excess(treat, chosen) - excess(control, chosen))
    deltas.sort()
    lo = deltas[int(0.025 * n)]
    hi = deltas[int(0.975 * n)]
    return lo, hi


def long_rows(eval_report: dict, diagnostic: dict) -> list[dict]:
    out = []
    for row in eval_report["rows"]:
        item = diagnostic.get((row["body_id"], row["query_index"]))
        if item is None:
            continue
        gap = len(item["input"]) - 1 - int(item["query_position"])
        if gap >= 13:
            rec = dict(row)
            rec["gap"] = gap
            out.append(rec)
    return out


def adjudicate() -> dict:
    d1b = assert_d1b_licenses_p1()
    diagnostic_items = json.loads(S2_DIAGNOSTIC.read_text(encoding="utf-8"))
    diagnostic = {(r["body_id"], r["query_index"]): r for r in diagnostic_items}
    treat_dir = OUT / f"treatment_{TRAIN_SEED}"
    control_dir = OUT / f"control_{TRAIN_SEED}"
    treat_receipt = max(treat_dir.glob("RECEIPT_*.json"))
    last = json.loads(treat_receipt.read_text(encoding="utf-8"))["last_step"]
    step = last if last % 200 == 0 else (last // 200) * 200
    if step == 0:
        step = last
    treat = json.loads((treat_dir / f"eval_{step:04d}.json").read_text(encoding="utf-8"))
    control = json.loads((control_dir / f"eval_{step:04d}.json").read_text(encoding="utf-8"))
    parent = json.loads((treat_dir / "eval_0000.json").read_text(encoding="utf-8"))
    t_long = treat["gap_strata"]["long"]
    c_long = control["gap_strata"]["long"]
    p_long = parent["gap_strata"]["long"]
    t_track = treat["pointer"]["long"]["fraction_any_query_tracking_head"]
    p_track = parent["pointer"]["long"]["fraction_any_query_tracking_head"]
    t_mass = treat["pointer"]["long"]["median_max_query_mass"]
    ci = bootstrap_delta(long_rows(treat, diagnostic), long_rows(control, diagnostic))
    delta = t_long["excess"] - c_long["excess"]
    track_gain = t_track - p_track
    reason = json.loads(treat_receipt.read_text(encoding="utf-8"))["reason"]

    if t_long["excess"] >= 0.10 and delta >= 0.07 and ci[0] > 0 and t_track >= 0.50:
        verdict = "SUCCESS"
    elif track_gain >= 0.20 or t_mass >= 0.15:
        verdict = "MECHANISM SUPPORTED, DOSE INSUFFICIENT"
    elif t_long["excess"] < 0.02 and track_gain < 0.05:
        verdict = "NULL"
    else:
        verdict = "MIXED"

    decision = {
        "protocol": "V010_SELECTION_REPAIR_P1_POINTER",
        "licensed_by": "V010_QUERY_PRESENCE_D1B",
        "d1b_verdict": d1b["verdict"],
        "step": step,
        "reason": reason,
        "verdict": verdict,
        "primary_endpoint": {
            "parent_excess": p_long["excess"],
            "treatment_excess": t_long["excess"],
            "control_excess": c_long["excess"],
            "treatment_minus_control": delta,
            "bootstrap_ci95": list(ci),
        },
        "mechanism": {
            "parent_query_track": p_track,
            "treatment_query_track": t_track,
            "track_gain": track_gain,
            "treatment_median_max_query_mass": t_mass,
        },
        "trained": True,
        "gates_changed": False,
        "protected_material_opened": False,
        "d1_reopened": False,
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
