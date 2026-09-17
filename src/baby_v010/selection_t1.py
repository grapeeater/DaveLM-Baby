"""T1 query-transport gap curriculum versus a perfectly matched natural-gap control.

Protocol: `design/V010_SELECTION_REPAIR_T1_GAP.md` (frozen before generation).

The two arms train on the **same items in the same order** with the **same loss**
(answer-span cross entropy, nothing added). The only difference is where the
template's own nuisance filler sits relative to the query key: the treatment
relocates filler tokens from after the query into the template's `filler_a` slot
to hit a scheduled query-to-generation gap, the control leaves the natural split
alone. Token multiset, body, rendered pair order, markers, separator, query key,
and target are identical in both arms.

Nothing here mutates frozen panels, S1 artifacts, or S2 artifacts.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path

from .selection_s1 import PANEL, PARENT, PARENT_SHA, digest, model_load, pack, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/selection_t1"
PROTOCOL = ROOT / "design/V010_SELECTION_REPAIR_T1_GAP.md"
S1_DIAGNOSTIC = ROOT / "runs/selection_s1/DIAGNOSTIC.json"
S2_DIAGNOSTIC = ROOT / "runs/selection_s2/DIAGNOSTIC.json"

TRAIN_SEED = 130001
DATA_SEED = 130100
BOOTSTRAP_SEED = 130300
MAX_UPDATES = 800
KEYED_PER_BATCH = 10
PRIMITIVE_PER_BATCH = 3
INDUCTION_PER_BATCH = 3
BATCH = KEYED_PER_BATCH + PRIMITIVE_PER_BATCH + INDUCTION_PER_BATCH
LANGUAGE_PROBABILITY = 0.20
GAP_VARIANTS = ("keyed_1", "keyed_3", "keyed_4")
MIN_GAP = {"keyed_1": 0, "keyed_3": 1, "keyed_4": 0}

# (last update of block, maximum allowed gap). None means "natural, no relocation".
SCHEDULE_BLOCKS = ((150, 1), (300, 3), (450, 8), (600, 20), (MAX_UPDATES, None))


def posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def allowed_gap(update: int) -> int | None:
    for last, cap in SCHEDULE_BLOCKS:
        if update <= last:
            return cap
    return None


def body_length(item: dict) -> int:
    return int(item["pair_count"]) * (int(item["value_length"]) + 2)


def insertion_index(item: dict) -> int:
    """Index of the template's own ``filler_a`` slot, which is always before the query."""
    query = int(item["query_position"])
    variant = item["variant"]
    if variant == "keyed_1":
        return query - 1 - body_length(item)
    if variant == "keyed_3":
        return query
    if variant == "keyed_4":
        return query - 1
    raise ValueError(f"variant {variant} has no free gap parameter")


def natural_gap(item: dict) -> int:
    return len(item["input"]) - 1 - int(item["query_position"])


def set_gap(item: dict, target: int) -> dict | None:
    """Relocate nuisance filler so the query sits ``target`` tokens from the end.

    Returns ``None`` when ``target`` is not reachable for this item. The returned
    item has the same length, the same token multiset, and the same target.
    """
    variant = item["variant"]
    if variant not in MIN_GAP:
        return None
    natural = natural_gap(item)
    if target < MIN_GAP[variant] or target > natural:
        return None
    move = natural - target
    out = copy.deepcopy(item)
    if move == 0:
        return out
    inp = [int(token) for token in item["input"]]
    index = insertion_index(item)
    if index < 1:
        return None
    kept, moved = inp[: len(inp) - move], inp[len(inp) - move :]
    out["input"] = kept[:index] + moved + kept[index:]
    out["query_position"] = int(item["query_position"]) + move
    if out["input"][out["query_position"]] != int(item["query_key"]):
        return None
    if natural_gap(out) != target:
        return None
    return out


def sorted_multiset(tokens) -> list[int]:
    return sorted(int(token) for token in tokens)


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
    for path in (S1_DIAGNOSTIC, S2_DIAGNOSTIC):
        if not path.exists():
            raise RuntimeError(f"missing prior diagnostic {path}; refusing to generate")
        absorb(json.loads(path.read_text(encoding="utf-8")))
    return denied_inputs, denied_spans


def _draw(rng, banks, denied_inputs, denied_spans, *, kind, difficulty, want_gap_variant):
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
            if want_gap_variant and item["variant"] not in GAP_VARIANTS:
                continue
        return item
    raise RuntimeError("could not draw a legal item")


def generate() -> None:
    from .data import LANG_TRAIN, build_banks, read_u16
    from .v2r4_provenance import FROZEN_PANELS_SHA256, require_frozen_file

    require_frozen_file(PANEL, FROZEN_PANELS_SHA256, "panels")
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    if (OUT / "MANIFEST.json").exists():
        raise RuntimeError("T1 already frozen")
    banks = build_banks(read_u16(LANG_TRAIN))
    denied_inputs, denied_spans = denials()

    audits = {}
    for seed in (TRAIN_SEED, TRAIN_SEED + 1):
        rng = random.Random(DATA_SEED + seed - TRAIN_SEED)
        schedule: list[dict] = []
        reachable = 0
        unreachable = 0
        gap_hist: dict[str, int] = defaultdict(int)
        for update in range(1, MAX_UPDATES + 1):
            if rng.random() < LANGUAGE_PROBABILITY:
                schedule.append({"task": "language", "rng_seed": rng.randrange(2**31)})
                continue
            cap = allowed_gap(update)
            items: list[dict] = []
            for _ in range(KEYED_PER_BATCH):
                item = _draw(
                    rng,
                    banks,
                    denied_inputs,
                    denied_spans,
                    kind="keyed",
                    difficulty="full",
                    want_gap_variant=cap is not None,
                )
                if cap is None:
                    item["t1_gap"] = None
                else:
                    low = MIN_GAP[item["variant"]]
                    high = min(cap, natural_gap(item))
                    if high < low:
                        item["t1_gap"] = None
                        unreachable += 1
                    else:
                        item["t1_gap"] = rng.randint(low, high)
                        reachable += 1
                gap_hist[str(item["t1_gap"])] += 1
                denied_inputs.add(tuple(item["input"]))
                denied_spans.add(tuple(item["target_span"]))
                items.append(item)
            for _ in range(PRIMITIVE_PER_BATCH):
                items.append(
                    _draw(
                        rng, banks, denied_inputs, denied_spans,
                        kind="keyed", difficulty="primitive", want_gap_variant=False,
                    )
                )
            for _ in range(INDUCTION_PER_BATCH):
                items.append(
                    _draw(
                        rng, banks, denied_inputs, denied_spans,
                        kind="induction", difficulty="full", want_gap_variant=False,
                    )
                )
            assert len(items) == BATCH
            schedule.append({"task": "structured", "update": update, "cap": cap, "items": items})
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
            "gap_targets_assigned": reachable,
            "gap_targets_unreachable": unreachable,
            "t1_gap_histogram": dict(sorted(gap_hist.items(), key=lambda kv: (kv[0] == "None", kv[0]))),
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
        ROOT / "src/baby_v010/data.py",
        ROOT / "src/baby_v010/data_v2.py",
        ROOT / "src/baby_v010/model.py",
        ROOT / "src/baby_v010/config.py",
        ROOT / "src/baby_v010/evaluate.py",
        ROOT / "src/baby_v010/train_v2r4.py",
        ROOT / "src/baby_v010/selection_s1.py",
        ROOT / "src/baby_v010/query_locality.py",
    ]
    write(
        OUT / "MANIFEST.json",
        {
            "protocol": "V010_SELECTION_REPAIR_T1_GAP",
            "parent_sha256": PARENT_SHA,
            "schedule_blocks": [list(block) for block in SCHEDULE_BLOCKS],
            "batch_composition": {
                "keyed_full": KEYED_PER_BATCH,
                "primitive_keyed": PRIMITIVE_PER_BATCH,
                "induction_full": INDUCTION_PER_BATCH,
            },
            "language_probability": LANGUAGE_PROBABILITY,
            "extra_loss_terms": [],
            "files": {posix(path): digest(path) for path in files},
            "audits": audits,
            "protected_material_opened": False,
            "s1_data_mutated": False,
            "s2_data_mutated": False,
        },
    )
    print(json.dumps(audits, indent=2), flush=True)


def verify() -> dict:
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError("hash mismatch " + rel)
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    return manifest


def arm_items(spec: dict, arm: str) -> list[dict]:
    """Treatment relocates filler to the scheduled gap; control keeps the natural split."""
    if arm == "control":
        return spec["items"]
    out = []
    for item in spec["items"]:
        target = item.get("t1_gap")
        if target is None:
            out.append(item)
            continue
        moved = set_gap(item, int(target))
        out.append(moved if moved is not None else item)
    return out


def gap_strata(rows: list[dict], diagnostic: dict) -> dict:
    from .query_locality import bucket

    cells = defaultdict(lambda: {"hit": 0, "n": 0, "chance": 0.0})
    for row in rows:
        item = diagnostic.get((row["body_id"], row["query_index"]))
        if item is None:
            continue
        gap = len(item["input"]) - 1 - int(item["query_position"])
        hit = int(bool(row["inventory_correct"]))
        for label in (bucket(gap), "long" if gap >= 13 else "short", "all"):
            cell = cells[label]
            cell["hit"] += hit
            cell["n"] += 1
            cell["chance"] += 1.0 / row["K"]
    return {
        label: {
            "hit": cell["hit"],
            "n": cell["n"],
            "accuracy": cell["hit"] / cell["n"],
            "chance": cell["chance"] / cell["n"],
            "excess": cell["hit"] / cell["n"] - cell["chance"] / cell["n"],
        }
        for label, cell in sorted(cells.items())
    }


def salience(rows: list[dict]) -> dict:
    bodies = defaultdict(list)
    for row in rows:
        bodies[row["body_id"]].append(row)
    bias, dev = [], []
    for group in bodies.values():
        k = group[0]["K"]
        if len(group) != k:
            continue
        logits = [row["candidate_logits"] for row in group]
        means = [sum(r[c] for r in logits) / len(logits) for c in range(k)]
        bias.append(max(means) - min(means))
        centered = [r[c] - means[c] for r in logits for c in range(k)]
        dev.append(max(centered) - min(centered))
    if not bias:
        return {}
    return {
        "salience_spread_median": statistics.median(bias),
        "query_dependent_spread_median": statistics.median(dev),
    }


def measure(model, dev_items, diagnostic, full: bool = True) -> dict:
    from .selection_s2 import measure as s2_measure

    report = s2_measure(model, dev_items, full=full)
    report["gap_strata"] = gap_strata(report["rows"], diagnostic)
    report["summary"].update(salience(report["rows"]))
    report["summary"]["primary_long_gap_excess"] = report["gap_strata"]["long"]["excess"]
    return report


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
    optimizer, _, _ = capability_optimizer(model)
    stream = torch.tensor(read_u16(LANG_TRAIN), dtype=torch.long)
    dev_items = json.loads(S2_DIAGNOSTIC.read_text(encoding="utf-8"))
    diagnostic = {(r["body_id"], r["query_index"]): r for r in dev_items}
    schedule = json.loads((OUT / f"SCHEDULE_{seed}.json").read_text(encoding="utf-8"))

    start = time.monotonic()
    baseline = measure(model, dev_items, diagnostic)
    write(dest / "eval_0000.json", baseline)
    print(
        json.dumps(
            {"arm": arm, "step": 0, "primary": baseline["gap_strata"]["long"],
             "ce": baseline["language_dev_ce"]}
        ),
        flush=True,
    )
    initial_ce = baseline["language_dev_ce"]
    baseline_long = baseline["gap_strata"]["long"]["excess"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    reason = "terminal"
    last_step = 0
    for step in range(1, until + 1):
        spec = schedule[step - 1]
        if shutil.disk_usage(ROOT).free < 10 * 2**30:
            reason, last_step = "hard_stop_disk", step - 1
            break
        if time.monotonic() - start > 7200:
            reason, last_step = "hard_stop_runtime", step - 1
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if spec["task"] == "language":
            x, y = language_batch(
                stream, random.Random(spec["rng_seed"]), 16, 256, torch.device("cuda")
            )
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        else:
            x, y, mask, _first = pack(arm_items(spec, arm), torch.device("cuda"))
            loss = F.cross_entropy(model(x)[mask], y[mask])
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
                    {"step": step, "task": spec["task"], "cap": spec.get("cap"),
                     "loss": loss.item(), "gradient_norm": float(norm)}
                )
                + "\n"
            )
        if step % 50 == 0:
            print(
                json.dumps({"arm": arm, "step": step, "loss": loss.item(),
                            "elapsed_s": round(time.monotonic() - start)}),
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
                    "protocol": "V010_SELECTION_REPAIR_T1_GAP",
                    "protected_material_opened": False,
                    "manifest_sha256": digest(OUT / "MANIFEST.json"),
                },
                dest / f"checkpoint_{16000 + step}.pt",
            )
            print(
                json.dumps(
                    {"arm": arm, "step": step, "primary": report["gap_strata"]["long"],
                     "short": report["gap_strata"]["short"],
                     "body_macro": report["summary"]["body_accuracy"],
                     "ce": report["language_dev_ce"]}
                ),
                flush=True,
            )
            if report["language_dev_ce"] > initial_ce + 0.20:
                reason = "hard_stop_language"
                break
            if step == 400 and arm == "treatment":
                gain = report["gap_strata"]["long"]["excess"] - baseline_long
                if gain < 0.02:
                    reason = "futility"
                    break
    write(
        dest / f"RECEIPT_{last_step:04d}.json",
        {
            "reason": reason,
            "last_step": last_step,
            "arm": arm,
            "seed": seed,
            "parent_sha256": digest(PARENT),
            "manifest_sha256": digest(OUT / "MANIFEST.json"),
            "elapsed_s": time.monotonic() - start,
            "artifacts": {p.name: digest(p) for p in dest.iterdir() if p.is_file()},
        },
    )
    print(json.dumps({"arm": arm, "reason": reason, "last_step": last_step}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "run"])
    parser.add_argument("--arm", choices=["control", "treatment"])
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--until", type=int, default=MAX_UPDATES)
    args = parser.parse_args()
    if args.action == "generate":
        generate()
    else:
        if not args.arm:
            raise SystemExit("--arm required")
        run(args.arm, args.seed, args.until)


if __name__ == "__main__":
    main()
