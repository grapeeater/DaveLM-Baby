"""S2 paired query-contrast vs matched all-K full-answer CE.

Does not mutate frozen S1 artifacts, gates, or protected eval.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

from .selection_s1 import (
    PANEL,
    PARENT,
    PARENT_SHA,
    diagnostic,
    digest,
    measure as s1_measure,
    model_load,
    pack,
    queries as s1_queries,
    write,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/selection_s2"
PROTOCOL = ROOT / "design/V010_SELECTION_REPAIR_S2.md"
S1_DIAGNOSTIC = ROOT / "runs/selection_s1/DIAGNOSTIC.json"
CONTRAST_MARGIN = 2.0
CONTRAST_WEIGHT = 1.0
TRAIN_SEED = 120001
DATA_SEED = 120100
DIAGNOSTIC_SEED = 120200
BOOTSTRAP_SEED = 120300
MAX_UPDATES = 400


def posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def contrast_loss_from_candidate_logits(candidate_logits, query_indices, margin: float = CONTRAST_MARGIN):
    """Paired query-contrast on a single body group.

    candidate_logits: [G, K] first-answer logits of the K inventory heads.
    query_indices: [G] which head is gold for that query row.
    """
    import torch
    import torch.nn.functional as F

    group = candidate_logits.shape[0]
    if group < 2:
        return candidate_logits.new_zeros(())
    gathered = candidate_logits[:, query_indices]
    effect = gathered.diag().unsqueeze(1) - gathered - gathered.t() + gathered.diag().unsqueeze(0)
    idx = torch.triu_indices(group, group, offset=1, device=candidate_logits.device)
    paired = effect[idx[0], idx[1]]
    return F.softplus(margin - paired).mean()


def inventory_rank(logits: list[float], query_index: int) -> int:
    queried = logits[query_index]
    return 1 + sum(1 for value in logits if value > queried)


def enrich(report: dict) -> dict:
    rows = report["rows"]
    bodies: dict[str, list] = defaultdict(list)
    for row in rows:
        bodies[row["body_id"]].append(row)
    same = 0
    rank1 = 0
    rank2 = 0
    for row in rows:
        rank = inventory_rank(row["candidate_logits"], row["query_index"])
        rank1 += int(rank == 1)
        rank2 += int(rank == 2)
    for group in bodies.values():
        firsts = {row["emitted"][0] for row in group if row["emitted"]}
        same += int(len(firsts) == 1)
    n = max(1, len(rows))
    report["summary"]["same_first_token_rate"] = same / max(1, len(bodies))
    report["summary"]["queried_inventory_rank1"] = rank1 / n
    report["summary"]["queried_inventory_rank2"] = rank2 / n
    report["summary"]["n_bodies"] = len(bodies)
    return report


def measure(model, dev, full: bool = True) -> dict:
    return enrich(s1_measure(model, dev, full=full))


def attach_body(rows: list[dict], body_id: str) -> list[dict]:
    out = []
    for row in rows:
        item = copy.deepcopy(row)
        item["body_id"] = body_id
        out.append(item)
    return out


def denied_from_panels(frozen: dict) -> tuple[set[tuple], set[tuple]]:
    from .isolation_transforms import parse_records

    denied_inputs: set[tuple] = set()
    denied_spans: set[tuple] = set()
    for group in frozen.values():
        for row in group:
            denied_inputs.add(tuple(row["input"]))
            denied_spans.add(tuple(row["target_span"]))
            if row.get("kind") == "keyed":
                denied_spans.update(tuple(value) for _, value in parse_records(row))
    return denied_inputs, denied_spans


def load_s1_denials(denied_inputs: set[tuple], denied_spans: set[tuple]) -> None:
    from .isolation_transforms import parse_records

    if not S1_DIAGNOSTIC.exists():
        raise RuntimeError("S1 diagnostic missing; refusing to generate overlapping S2 data")
    for row in json.loads(S1_DIAGNOSTIC.read_text(encoding="utf-8")):
        denied_inputs.add(tuple(row["input"]))
        denied_spans.add(tuple(row["target_span"]))
        if row.get("kind") == "keyed":
            denied_spans.update(tuple(value) for _, value in parse_records(row))


def try_keyed_group(rng: random.Random, banks, denied_inputs: set[tuple], denied_spans: set[tuple], k: int, body_id: str):
    from .data_v2 import make_item
    from .isolation_transforms import parse_records

    item = make_item(rng, banks, kind="keyed", difficulty="full", low_prior=rng.random() < 0.20)
    pairs = parse_records(item)
    if item["pair_count"] != k or len({value[0] for _, value in pairs}) != k:
        return None
    if any(tuple(value) in denied_spans for _, value in pairs):
        return None
    group = attach_body(s1_queries(item), body_id)
    if any(tuple(row["input"]) in denied_inputs for row in group) or len(group) != k:
        return None
    return group, pairs


def keyed_group(rng: random.Random, banks, denied_inputs: set[tuple], denied_spans: set[tuple], k: int, body_id: str):
    for _ in range(10000):
        got = try_keyed_group(rng, banks, denied_inputs, denied_spans, k, body_id)
        if got is not None:
            return got
    raise RuntimeError(f"could not sample a legal K={k} body")


def generate() -> None:
    from .data import LANG_TRAIN, build_banks, read_u16
    from .data_v2 import make_item
    from .v2r4_provenance import FROZEN_PANELS_SHA256, require_frozen_file

    require_frozen_file(PANEL, FROZEN_PANELS_SHA256, "panels")
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    if (OUT / "MANIFEST.json").exists():
        raise RuntimeError("S2 already frozen")
    banks = build_banks(read_u16(LANG_TRAIN))
    frozen = json.loads(PANEL.read_text(encoding="utf-8"))
    denied_inputs, denied_spans = denied_from_panels(frozen)
    load_s1_denials(denied_inputs, denied_spans)

    rng = random.Random(DIAGNOSTIC_SEED)
    dev = []
    for k in (2, 3, 4):
        for count in range(48):
            body_id = f"K{k}_{count}"
            group, pairs = keyed_group(rng, banks, denied_inputs, denied_spans, k, body_id)
            dev.extend(group)
            denied_spans.update(tuple(value) for _, value in pairs)
            denied_inputs.update(tuple(row["input"]) for row in group)
    write(OUT / "DIAGNOSTIC.json", dev)

    audits = {}
    for seed in (TRAIN_SEED, 120002):
        rng = random.Random(DATA_SEED + seed - TRAIN_SEED)
        schedule = []
        rejected = 0
        body_serial = 0
        for _step in range(MAX_UPDATES):
            if rng.random() < 0.20:
                schedule.append({"task": "language", "rng_seed": rng.randrange(2**31)})
                continue
            k = rng.choice((2, 3, 4))
            n_bodies = 12 // k
            items = []
            for _ in range(n_bodies):
                got = None
                for _attempt in range(10000):
                    got = try_keyed_group(
                        rng, banks, denied_inputs, denied_spans, k, f"train_{seed}_{body_serial}"
                    )
                    if got is None:
                        rejected += 1
                        continue
                    break
                if got is None:
                    raise RuntimeError(f"could not sample a legal K={k} train body")
                group, _pairs = got
                items.extend(group)
                body_serial += 1
            while len(items) < 16:
                item = make_item(rng, banks, kind="induction", difficulty="full", low_prior=rng.random() < 0.20)
                if tuple(item["input"]) in denied_inputs or tuple(item["target_span"]) in denied_spans:
                    rejected += 1
                    continue
                items.append(item)
            assert len(items) == 16
            assert sum(row["kind"] == "keyed" for row in items) == 12
            schedule.append({"task": "structured", "k": k, "items": items})
        write(OUT / f"SCHEDULE_{seed}.json", schedule)
        rows = [row for spec in schedule if spec["task"] == "structured" for row in spec["items"]]
        keyed = [row for row in rows if row["kind"] == "keyed"]
        audits[str(seed)] = {
            "updates": len(schedule),
            "language_updates": sum(spec["task"] == "language" for spec in schedule),
            "rows": len(rows),
            "keyed_rows": len(keyed),
            "induction_rows": sum(row["kind"] == "induction" for row in rows),
            "rejected": rejected,
            "keyed_pair_count": {str(k): sum(row["pair_count"] == k for row in keyed) for k in (2, 3, 4)},
            "all_k_groups": True,
            "no_exact_input_or_candidate_span_leakage": True,
        }
    files = [
        PROTOCOL,
        Path(__file__),
        PANEL,
        OUT / "DIAGNOSTIC.json",
        OUT / "SCHEDULE_120001.json",
        OUT / "SCHEDULE_120002.json",
        ROOT / "src/baby_v010/data.py",
        ROOT / "src/baby_v010/data_v2.py",
        ROOT / "src/baby_v010/model.py",
        ROOT / "src/baby_v010/config.py",
        ROOT / "src/baby_v010/evaluate.py",
        ROOT / "src/baby_v010/train_v2r4.py",
        ROOT / "src/baby_v010/isolation_transforms.py",
        ROOT / "src/baby_v010/selection_s1.py",
    ]
    write(
        OUT / "MANIFEST.json",
        {
            "protocol": "V010_SELECTION_REPAIR_S2",
            "parent_sha256": PARENT_SHA,
            "contrast_margin": CONTRAST_MARGIN,
            "contrast_weight": CONTRAST_WEIGHT,
            "files": {posix(path): digest(path) for path in files},
            "audits": audits,
            "diagnostic_rows": len(dev),
            "diagnostic_bodies": 144,
            "protected_material_opened": False,
            "s1_data_mutated": False,
        },
    )
    print(json.dumps(audits), flush=True)


def verify() -> dict:
    manifest = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        if digest(ROOT / rel) != expected:
            raise RuntimeError("hash mismatch " + rel)
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    return manifest


def keyed_contrast_loss(logits, items, first, margin: float = CONTRAST_MARGIN):
    import torch

    by_body: dict[str, list] = defaultdict(list)
    index = {row_i: item for row_i, item in enumerate(items) if item.get("kind") == "keyed"}
    for row_i, pos, _gold in first:
        item = index[row_i]
        by_body[item["body_id"]].append((row_i, pos, item))
    losses = []
    for group in by_body.values():
        if len(group) < 2:
            continue
        heads = group[0][2]["candidate_heads"]
        stacked = torch.stack([logits[row_i, pos, heads] for row_i, pos, _ in group], dim=0)
        query_indices = torch.tensor([item["query_index"] for _r, _p, item in group], device=logits.device)
        losses.append(contrast_loss_from_candidate_logits(stacked, query_indices, margin=margin))
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def futility_at_200(baseline: dict, treatment: dict) -> bool:
    parent = baseline["summary"]
    now = treatment["summary"]
    return (
        now["query_logit_effect"] - parent["query_logit_effect"] < 0.15
        and parent["same_first_token_rate"] - now["same_first_token_rate"] < 0.05
        and now["queried_inventory_rank1"] - parent["queried_inventory_rank1"] < 0.05
    )


def run(arm: str, seed: int, until: int, resume: bool) -> None:
    import torch
    import torch.nn.functional as F

    from .data import LANG_TRAIN, read_u16
    from .train_v2r4 import capability_optimizer, language_batch, set_seed

    verify()
    dest = OUT / f"{arm}_{seed}"
    if dest.exists() and not resume:
        raise RuntimeError("refuse overwrite " + str(dest))
    dest.mkdir(exist_ok=resume)
    set_seed(seed)
    model, config = model_load()
    optimizer, _, _ = capability_optimizer(model)
    stream = torch.tensor(read_u16(LANG_TRAIN), dtype=torch.long)
    dev = json.loads((OUT / "DIAGNOSTIC.json").read_text(encoding="utf-8"))
    schedule = json.loads((OUT / f"SCHEDULE_{seed}.json").read_text(encoding="utf-8"))
    start = time.monotonic()
    start_step = 0
    if resume:
        raise RuntimeError("S2 resume is not part of the frozen protocol")
    baseline = measure(model, dev)
    write(dest / "eval_0000.json", baseline)
    print(json.dumps({"arm": arm, "step": 0, "summary": baseline["summary"], "ce": baseline["language_dev_ce"]}), flush=True)
    initial_ce = baseline["language_dev_ce"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    reason = "terminal"
    last_step = 0
    for step in range(start_step + 1, until + 1):
        spec = schedule[step - 1]
        if shutil.disk_usage(ROOT).free < 10 * 2**30:
            reason = "hard_stop_disk"
            last_step = step - 1
            break
        if time.monotonic() - start > 7200:
            reason = "hard_stop_runtime"
            last_step = step - 1
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if spec["task"] == "language":
            x, y = language_batch(stream, random.Random(spec["rng_seed"]), 16, 256, torch.device("cuda"))
            z = model(x)
            loss = F.cross_entropy(z.flatten(0, 1), y.flatten())
        else:
            x, y, mask, first = pack(spec["items"], torch.device("cuda"))
            z = model(x)
            loss = F.cross_entropy(z[mask], y[mask])
            if arm == "treatment" and first:
                loss = loss + CONTRAST_WEIGHT * keyed_contrast_loss(z, spec["items"], first)
        if not torch.isfinite(loss):
            reason = "hard_stop_nonfinite_loss"
            last_step = step
            break
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        if not torch.isfinite(norm):
            reason = "hard_stop_nonfinite_gradient"
            last_step = step
            break
        optimizer.step()
        last_step = step
        with (dest / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, "task": spec["task"], "loss": loss.item(), "gradient_norm": float(norm)}) + "\n")
        if step % 25 == 0:
            print(json.dumps({"arm": arm, "step": step, "loss": loss.item(), "elapsed_s": time.monotonic() - start}), flush=True)
        if step % 200 == 0:
            checkpoint = {
                "config": config.to_dict(),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "parent_checkpoint_sha256": PARENT_SHA,
                "update": 16000 + step,
                "seed": seed,
                "arm": arm,
                "protocol": "V010_SELECTION_REPAIR_S2",
                "contrast_margin": CONTRAST_MARGIN,
                "protected_material_opened": False,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "manifest_sha256": digest(OUT / "MANIFEST.json"),
            }
            torch.save(checkpoint, dest / f"checkpoint_{16000 + step}.pt")
            report = measure(model, dev, full=True)
            write(dest / f"eval_{step:04d}.json", report)
            print(json.dumps({"arm": arm, "step": step, "summary": report["summary"], "ce": report["language_dev_ce"]}), flush=True)
            if report["language_dev_ce"] > initial_ce + 0.20:
                reason = "hard_stop_language"
                break
            if step == 200 and arm == "treatment" and futility_at_200(baseline, report):
                reason = "futility"
                break
    write(
        dest / f"RECEIPT_{last_step:04d}.json",
        {
            "reason": reason,
            "last_step": last_step,
            "parent_sha256": digest(PARENT),
            "manifest_sha256": digest(OUT / "MANIFEST.json"),
            "elapsed_s": time.monotonic() - start,
            "arm": arm,
            "seed": seed,
            "artifacts": {path.name: digest(path) for path in dest.iterdir() if path.is_file()},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["generate", "run"])
    parser.add_argument("--arm", choices=["control", "treatment"])
    parser.add_argument("--seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--until", type=int, default=MAX_UPDATES)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.until not in (200, 400):
        raise SystemExit("until must be 200 or 400")
    if args.action == "generate":
        generate()
    else:
        if not args.arm:
            raise SystemExit("--arm required")
        run(args.arm, args.seed, args.until, args.resume)


if __name__ == "__main__":
    main()
