r"""Stage-0 read-only measurement for the seed-87002 factual-supervision run.

1. Verifies frozen v8/eval hashes, parent/tokenizer identity, factual-500
   checkpoint identity, and the factual TRAIN corpus identity.
2. Recomputes the persisted Primary aggregates from RAW_RESULTS.jsonl and
   compares them to the corrected authoritative records (AGGREGATES_CORRECTED,
   STRATA). On material disagreement it stops before any TRAIN inference.
3. Only if 1-2 pass: scores factual-500 on the factual arm's own 384-item
   TRAIN partition using the exact frozen EVALUATE.py candidate-scoring
   semantics (English base_model path, candidate tokens + EOS, signed margin,
   correct iff margin>0, tie==0). No new metrics, no normalization, no ranks,
   no perplexity, no thresholds, no greedy/EOS format diagnostics.

No checkpoint, corpus, schedule, receipt, or historical file is modified.
Output is written only to the additive directory below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r"C:\DaveLM-CADAVER")
V8 = ROOT / "fact_supervision_87001_corrected_v8"
EVAL = ROOT / "fact_supervision_87001_eval_v1"
OUT_DIR = ROOT / "fact_supervision_87001_stage0_measurement"

PARENT = ROOT / "language_pilot_1_early_block_protection_seed8380" / "pilot_run" / "checkpoints" / "seed_8380" / "latest.pt"
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
FACTUAL_500 = V8 / "run_factual" / "checkpoint_500.pt"
ITEMS = V8 / "ITEMS.jsonl"
SCHEDULE = V8 / "MATERIALIZED_SCHEDULE_SEED87002.json"
RAW = EVAL / "RAW_RESULTS.jsonl"
AGG_CORRECTED = EVAL / "AGGREGATES_CORRECTED.json"
STRATA_FILE = EVAL / "STRATA.json"
REPORT = EVAL / "REPORT.md"

EXPECTED = {
    "parent_sha256": "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb",
    "tokenizer_sha256": "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b",
    "items_sha256": "2dab70739e9c0ffe1ad3154076ec613a10a4964fa1138e9c7b40d8a5e55f518d",
    "factual_500_observed": "3ae30ae847d3129d7715173ddbfbbaf6b7490772dc850e361a5a2d785b991123",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_file_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    require(actual == expected, f"{label} hash mismatch: expected {expected}, observed {actual}")
    print(f"[hash] {label}: {actual}")


def verify_sums_file(bundle: Path) -> None:
    for line in (bundle / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        h, name = line.split("  ", 1)
        target = bundle / name
        if target.is_file():
            require(sha256_file(target) == h, f"bundle file hash mismatch: {name}")
    receipt = json.loads((bundle / "FREEZE_RECEIPT.json").read_text(encoding="utf-8"))
    require(sha256_file(bundle / "FREEZE_RECEIPT.json")
            == (bundle / "FREEZE_RECEIPT.sha256").read_text().split()[0],
            "FREEZE_RECEIPT chain")
    require(sha256_file(bundle / "SHA256SUMS.txt") == receipt["manifest_sha256"],
            "manifest_sha256 chain")


def parse_key(row) -> tuple:
    # id encodes ...:a{assignment}q{query}o{fact_order}
    tail = row["id"].rsplit(":", 1)[-1]
    return (int(tail[1]), int(tail[3]), int(tail[5]))


def row_key(row) -> tuple:
    if "assignment" in row:
        return (int(row["assignment"]), int(row["query"]), int(row["fact_order"]))
    return parse_key(row)


def aggregate(rows, reversal_pairs_total_by_family=4):
    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family_id"]].append(r)
    items = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    ties = sum(1 for r in rows if r["tie"])
    families_complete = sum(all(r["correct"] for r in group) for group in by_family.values())
    margins = [r["margin"] for r in rows]
    rev_both = 0
    rev_pairs = 0
    for group in by_family.values():
        mp = {row_key(r): r for r in group}
        for key, item in mp.items():
            a, q, o = key
            partner = mp.get((1 - a, q, o))
            if partner is not None and a == 0:
                rev_pairs += 1
                rev_both += int(item["correct"] and partner["correct"])
    require(rev_pairs == reversal_pairs_total_by_family * len(by_family),
            f"reversal pairs != expected total ({rev_pairs})")
    return {
        "items": items,
        "correct": correct,
        "accuracy": correct / items if items else 0.0,
        "ties": ties,
        "families_complete": families_complete,
        "families_total": len(by_family),
        "reversal_pairs": rev_pairs,
        "reversal_both_correct": rev_both,
        "reversal_rate": rev_both / rev_pairs if rev_pairs else 0.0,
        "mean_margin": statistics.mean(margins) if margins else 0.0,
    }


def recompute_primary_from_raw():
    rows = [json.loads(line) for line in RAW.read_text(encoding="utf-8").splitlines()]
    require(len(rows) == 576, f"raw rows != 576 ({len(rows)})")
    result = {}
    for checkpoint in ("Pilot1 parent", "Factual", "Control"):
        group = [r for r in rows if r["checkpoint"] == checkpoint]
        require(len(group) == 192, f"{checkpoint}: raw rows != 192")
        # self-consistency: correct flag == margin > 0 (no ties in persisted set)
        require(all((r["correct"] == (r["margin"] > 0)) and (r["tie"] == (r["margin"] == 0))
                    for r in group), f"{checkpoint}: persisted correct/tie not implied by margin")
        result[checkpoint] = aggregate(group)
        result[checkpoint]["greedy_exact"] = sum(1 for r in group if r["greedy"]["exact"])
        result[checkpoint]["object"] = aggregate([r for r in group if r["stratum"] == "object"])
        result[checkpoint]["predicate"] = aggregate([r for r in group if r["stratum"] == "predicate"])
    return rows, result


def compare_primary(recomputed, corrected, strata):
    mismatches = []
    for name in corrected:
        expect = corrected[name]
        got = recomputed[name]
        for field in ("items", "correct", "families_complete", "families_total",
                      "reversal_both_correct", "greedy_exact"):
            if expect.get(field) != got[field]:
                mismatches.append(f"{name}.{field}: expected {expect.get(field)}, got {got[field]}")
        if abs(expect.get("mean_margin", 0.0) - got["mean_margin"]) > 1e-6:
            mismatches.append(f"{name}.mean_margin expected {expect.get('mean_margin')}, got {got['mean_margin']}")
    s = json.loads(strata.read_text(encoding="utf-8"))
    for name in corrected:
        got = recomputed[name]
        for stratum in ("object", "predicate"):
            expect = s[name][stratum]
            if expect["items"] != got[stratum]["items"] or expect["correct"] != got[stratum]["correct"]:
                mismatches.append(f"{name}.{stratum}: expected {expect}, got "
                                  f"{{'items': {got[stratum]['items']}, 'correct': {got[stratum]['correct']}}}")
            if expect.get("reversal", -1) != got[stratum]["reversal_both_correct"]:
                mismatches.append(f"{name}.{stratum}.reversal expected {expect.get('reversal')}, "
                                  f"got {got[stratum]['reversal_both_correct']}")
    return mismatches


def load_model_factual_500(dev):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, r"C:\DaveLM-v0.9")
    sys.path.insert(0, str(EVAL))
    from treatment13_model import Treatment13Model
    from PINNED_PILOT1_BINDING_IMPLEMENTATION import OrthoLocalizer
    raw = torch.load(FACTUAL_500, map_location=dev, weights_only=True)
    sd = dict(raw["model_state_dict"])
    keys = ["u", "q", "bs", "ba"]
    vals = [sd.pop("localizer." + k) for k in keys]
    m = Treatment13Model()
    m.load_state_dict(sd, strict=False)
    m.localizer = OrthoLocalizer(*vals)
    return m.to(dev)


def score_items(items, model, dev, tokenizer):
    import torch
    torch.set_grad_enabled(False)
    rows = []
    for row in items:
        cands = row["candidate_token_ids"]
        ci = int(row["correct_index"])
        out = []
        prefix = [2] + row["prompt_token_ids"]
        for c in cands:
            ids = prefix + list(c) + [3]
            x = torch.tensor([ids], device=dev)
            logits = model.base_model(x)[0]
            lp = logits[len(prefix) - 1: len(prefix) - 1 + len(c) + 1].log_softmax(-1)
            idx = torch.tensor(list(c) + [3], device=dev)
            vals = lp[torch.arange(len(idx), device=dev), idx].detach().cpu().tolist()
            out.append({"candidate_ll": sum(vals[:-1]), "eos_ll": sum(vals), "tokens": vals})
        margin = out[0]["candidate_ll"] - out[1]["candidate_ll"]
        signed = margin if ci == 0 else -margin
        rows.append({
            "id": row["id"],
            "family_id": row["family_id"],
            "stratum": row["stratum"],
            "assignment": int(row["assignment"]),
            "query": int(row["query"]),
            "fact_order": int(row["fact_order"]),
            "correct_index": ci,
            "scores": out,
            "margin": signed,
            "correct": signed > 0,
            "tie": signed == 0,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    require(torch.cuda.is_available() if args.device == "cuda" else True, "GPU unavailable")
    dev = torch.device(args.device)

    # 1. provenance verification
    verify_file_hash(PARENT, EXPECTED["parent_sha256"], "parent checkpoint")
    verify_file_hash(TOKENIZER, EXPECTED["tokenizer_sha256"], "tokenizer")
    verify_file_hash(ITEMS, EXPECTED["items_sha256"], "ITEMS.jsonl")
    verify_file_hash(FACTUAL_500, EXPECTED["factual_500_observed"], "factual-500 checkpoint")
    verify_sums_file(V8)
    verify_sums_file(EVAL)
    schedule_expected = next(
        line.split("  ", 1)[0] for line in (V8 / "SHA256SUMS.txt").read_text().splitlines()
        if line.endswith("MATERIALIZED_SCHEDULE_SEED87002.json"))
    verify_file_hash(SCHEDULE, schedule_expected, "MATERIALIZED_SCHEDULE_SEED87002.json")

    print("Provisionally verified v8 and eval bundles, parent, tokenizer, schedule, "
          "ITEMS corpus identity, and factual-500 observed hash.")

    # 2. recompute persisted Primary aggregates
    raw_rows, recomputed = recompute_primary_from_raw()
    corrected = json.loads(AGG_CORRECTED.read_text(encoding="utf-8"))
    mismatches = compare_primary(recomputed, corrected, STRATA_FILE)
    require(not mismatches, "Primary recomputation disagrees with corrected authoritative record: "
            + "; ".join(mismatches[:10]))
    print("Primary recomputation matches AGGREGATES_CORRECTED.json and STRATA.json for all checkpoints.")

    # 3. factual TRAIN partition (factual arm)
    items = [json.loads(line) for line in ITEMS.read_text(encoding="utf-8").splitlines()]
    train = [r for r in items if r["partition"] == "train" and r["arm"] == "factual"]
    require(len(train) == 384, f"factual train items != 384 ({len(train)})")
    families = Counter(r["family_id"] for r in train)
    require(len(families) == 48 and all(v == 8 for v in families.values()),
            "factual train families != 48x8")

    model = load_model_factual_500(dev)
    model.eval()
    scored = score_items(train, model, dev, None)

    train_agg = {"overall": aggregate(scored),
                 "object": aggregate([r for r in scored if r["stratum"] == "object"]),
                 "predicate": aggregate([r for r in scored if r["stratum"] == "predicate"])}
    asym = {}
    for label, key in (("assignment", "assignment"), ("query", "query"), ("fact_order", "fact_order")):
        counts = Counter()
        total = Counter()
        for r in scored:
            total[r[key]] += 1
            counts[r[key]] += int(r["correct"])
        asym[label] = {str(k): {"correct": counts[k], "total": total[k]}
                       for k in sorted(set(total))}
    asym_by_stratum = {}
    for stratum in ("object", "predicate"):
        rows = [r for r in scored if r["stratum"] == stratum]
        asym_by_stratum[stratum] = {}
        for label, key in (("assignment", "assignment"), ("query", "query"), ("fact_order", "fact_order")):
            counts, total = Counter(), Counter()
            for r in rows:
                total[r[key]] += 1
                counts[r[key]] += int(r["correct"])
            asym_by_stratum[stratum][label] = {str(k): {"correct": counts[k], "total": total[k]}
                                               for k in sorted(set(total))}
    margins = [r["margin"] for r in scored]
    margin_summary = {
        "n": len(margins),
        "mean": statistics.mean(margins),
        "median": statistics.median(margins),
        "min": min(margins),
        "max": max(margins),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "STAGE0_MEASUREMENT_COMPLETE",
        "provenance": {
            "parent_sha256": sha256_file(PARENT),
            "tokenizer_sha256": sha256_file(TOKENIZER),
            "items_sha256": sha256_file(ITEMS),
            "schedule_sha256": sha256_file(SCHEDULE),
            "factual_500_sha256": sha256_file(FACTUAL_500),
            "v8_shasums_verified": True,
            "eval_shasums_verified": True,
            "raw_rows": len(raw_rows),
        },
        "primary_recompute_matches_authoritative": True,
        "primary_recompute": recomputed,
        "train": {
            "corpus": {"items": len(train), "families": len(families), "items_per_family": 8,
                       "balanced_correct_index": Counter(r["correct_index"] for r in train)},
            "aggregates": train_agg,
            "margin_summary": margin_summary,
            "asymmetry_by_factor": asym,
            "asymmetry_by_factor_and_stratum": asym_by_stratum,
        },
        "per_item": scored,
    }
    out_json = OUT_DIR / "stage0_measurement.json"
    out_json.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    md_lines = ["# Fact-supervision (seed 87002) Stage-0 read-only measurement", "",
                "## Provenance verified", ""]
    for k, v in payload["provenance"].items():
        if isinstance(v, bool):
            md_lines.append(f"- {k}: {v}")
        else:
            md_lines.append(f"- {k}: `{v}`")
    md_lines += ["", "## Primary recomputation (matches corrected authoritative record)", ""]
    for name, agg in recomputed.items():
        md_lines.append(f"### {name}")
        md_lines.append(json.dumps({k: agg[k] for k in
                                    ("items", "correct", "accuracy", "ties", "families_complete",
                                     "families_total", "reversal_pairs", "reversal_both_correct",
                                     "mean_margin", "greedy_exact")}, indent=1))
    md_lines += ["", "## Factual-500 on factual TRAIN partition (384 items)", "",
                 "### Overall", json.dumps(train_agg["overall"], indent=1),
                 "### Object stratum", json.dumps(train_agg["object"], indent=1),
                 "### Predicate stratum", json.dumps(train_agg["predicate"], indent=1),
                 "### Margin summary", json.dumps(margin_summary, indent=1),
                 "### Asymmetry by factor", json.dumps(asym, indent=1),
                 "### Asymmetry by factor and stratum", json.dumps(asym_by_stratum, indent=1), ""]
    (OUT_DIR / "stage0_measurement_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    (OUT_DIR / "SHA256SUMS.txt").write_text(
        f"{sha256_file(out_json)}  {out_json.name}\n", encoding="utf-8")

    print(json.dumps({"train_overall": train_agg["overall"],
                      "train_object": train_agg["object"],
                      "train_predicate": train_agg["predicate"],
                      "margin_summary": margin_summary}, indent=1))
    return 0


if __name__ == "__main__":
    import torch
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"STAGE0 ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
