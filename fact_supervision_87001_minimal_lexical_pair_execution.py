r"""Read-only execution of the frozen Fork L2 minimal-lexical paired diagnostic.

ONE (48 items) and TWO (96 items) x 3 pinned checkpoints. Scorer = frozen
candidate conditional LL over response tokens + EOS (English base_model path),
signed margin, correct iff margin>0, tie==0, greedy argmax<=32 stop EOS.
Additive outputs only.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(r"C:\DaveLM-CADAVER")
PKG = ROOT / "fact_supervision_87001_minimal_lexical_pair_seed87002"
OUT = ROOT / "fact_supervision_87001_minimal_lexical_pair_execution"
TOKENIZER_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
AC_AGG = ROOT / "fact_supervision_87001_single_fact_AC_execution" / "aggregates.json"
GLOSS_AGG = ROOT / "fact_supervision_87001_structural_gloss_execution" / "aggregates.json"

CHECKPOINTS = {
    "Pilot1 parent": ROOT / "language_pilot_1_early_block_protection_seed8380" / "pilot_run" / "checkpoints" / "seed_8380" / "latest.pt",
    "factual-500": ROOT / "fact_supervision_87001_corrected_v8" / "run_factual" / "checkpoint_500.pt",
    "control-500": ROOT / "fact_supervision_87001_corrected_v8" / "run_control" / "checkpoint_500.pt",
}
EXPECTED_CK = {
    "Pilot1 parent": "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb",
    "factual-500": "3ae30ae847d3129d7715173ddbfbbaf6b7490772dc850e361a5a2d785b991123",
    "control-500": "c888e3b4cf20860b8d1d0651e418be931da7e8db2acc1c32625edb29701f1530",
}
EXPECTED_TOKENIZER = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
EXPECTED_EVALUATOR = "3563cb58334cd9753ca8ebff7fe49987a0f0ccb535abdfdf9a149cc1541a497b"
EOS = 3
BOS = 2


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def load_model(path, dev):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, r"C:\DaveLM-v0.9")
    sys.path.insert(0, str(ROOT / "fact_supervision_87001_eval_v1"))
    from treatment13_model import Treatment13Model
    from PINNED_PILOT1_BINDING_IMPLEMENTATION import OrthoLocalizer
    raw = torch.load(path, map_location=dev, weights_only=True)
    sd = dict(raw["model_state_dict"])
    vals = [sd.pop("localizer." + k) for k in ("u", "q", "bs", "ba")]
    m = Treatment13Model()
    m.load_state_dict(sd, strict=False)
    m.localizer = OrthoLocalizer(*vals)
    return m.to(dev)


@torch.no_grad()
def item_score(m, row, dev, ci, tok):
    prefix = [BOS] + tok.encode(row["prompt"]).ids
    out = []
    for c in row["candidate_token_ids"]:
        ids = prefix + list(c) + [EOS]
        x = torch.tensor([ids], device=dev)
        logits = m.base_model(x)[0]
        lp = logits[len(prefix) - 1: len(prefix) - 1 + len(c) + 1].log_softmax(-1)
        idx = torch.tensor(list(c) + [EOS], device=dev)
        vals = lp[torch.arange(len(idx), device=dev), idx].detach().cpu().tolist()
        out.append({"candidate_ll": sum(vals[:-1]), "eos_ll": sum(vals), "tokens": vals})
    margin = out[0]["candidate_ll"] - out[1]["candidate_ll"]
    signed = margin if ci == 0 else -margin
    return {"id": row["id"], "family_id": row["family_id"], "stratum": row["stratum"],
            "query": row["query"], "row_order": row.get("row_order"), "kind": row["kind"],
            "correct_index": ci, "correct_name": row["correct_name"],
            "scores": out, "margin": signed, "correct": signed > 0, "tie": signed == 0}


@torch.no_grad()
def greedy(m, row, dev, ci, tok):
    target = row["candidate_token_ids"][ci] + [EOS]
    gen = [BOS] + tok.encode(row["prompt"]).ids
    for _ in range(32):
        x = torch.tensor([gen[-256:]], device=dev)
        n = int(m.base_model(x)[0, -1].argmax())
        gen.append(n)
        if n == EOS:
            break
    return {"exact": gen[len([BOS] + tok.encode(row["prompt"]).ids):] == target,
            "generated_ids": gen[len([BOS] + tok.encode(row["prompt"]).ids):]}


def agg(rows):
    by_family = defaultdict(list)
    for r in rows:
        by_family[r["family_id"]].append(r)
    fam_complete = sum(all(r["correct"] for r in v) for v in by_family.values())
    return {
        "n": len(rows), "correct": sum(r["correct"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "ties": sum(r["tie"] for r in rows),
        "families_complete": fam_complete, "families_total": len(by_family),
        "mean_margin": statistics.mean(r["margin"] for r in rows),
        "median_margin": statistics.median(r["margin"] for r in rows),
        "min_margin": min(r["margin"] for r in rows),
        "max_margin": max(r["margin"] for r in rows),
        "correct_index": dict(sorted(Counter(r["correct_index"] for r in rows).items())),
        "correct_name": dict(sorted(Counter(r["correct_name"] for r in rows).items())),
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    require(torch.cuda.is_available() if args.device == "cuda" else True, "GPU unavailable")
    dev = torch.device(args.device)

    sums = {n: h for h, n in (l.split("  ", 1) for l in (PKG / "SHA256SUMS.txt").read_text().splitlines() if l.strip())}
    for f in PKG.iterdir():
        if f.is_file() and f.name != "SHA256SUMS.txt":
            require(sha256_file(f) == sums[f.name], f"package hash mismatch {f.name}")
    receipt = json.loads((PKG / "FREEZE_RECEIPT.json").read_text())
    require((PKG / "FREEZE_RECEIPT.sha256").read_text().split()[0] == sha256_file(PKG / "FREEZE_RECEIPT.json"),
            "receipt chain")
    require(receipt["items_one_sha256"] == sha256_file(PKG / "ITEMS_ONE.jsonl"), "ONE hash")
    require(receipt["items_two_sha256"] == sha256_file(PKG / "ITEMS_TWO.jsonl"), "TWO hash")
    require(sha256_file(TOKENIZER_PATH) == EXPECTED_TOKENIZER, "tokenizer")
    require(sha256_file(ROOT / "fact_supervision_87001_eval_v1/EVALUATE.py") == EXPECTED_EVALUATOR, "evaluator")
    for k, p in CHECKPOINTS.items():
        require(sha256_file(p) == EXPECTED_CK[k], f"checkpoint {k}")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    one = [json.loads(l) for l in (PKG / "ITEMS_ONE.jsonl").read_text().splitlines()]
    two = [json.loads(l) for l in (PKG / "ITEMS_TWO.jsonl").read_text().splitlines()]
    for it in one:
        it["kind"] = "ONE"
    for it in two:
        it["kind"] = "TWO"
    require(len(one) == 48 and len(two) == 96, "counts")

    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "raw_per_item.jsonl"
    raw.write_text("", encoding="utf-8")
    results = {"provenance": {}, "per_checkpoint": {}}

    for ck_name, path in CHECKPOINTS.items():
        m = load_model(path, dev)
        m.eval()
        scored = []
        for row in one + two:
            z = item_score(m, row, dev, row["correct_index"], tok)
            z["greedy"] = greedy(m, row, dev, row["correct_index"], tok)
            z["checkpoint"] = ck_name
            scored.append(z)
        with raw.open("a", encoding="utf-8") as fh:
            for z in scored:
                fh.write(json.dumps(z) + "\n")
        one_s = [r for r in scored if r["kind"] == "ONE"]
        two_s = [r for r in scored if r["kind"] == "TWO"]
        res = {
            "ONE": {"all": agg(one_s),
                    "object": agg([r for r in one_s if r["stratum"] == "object"]),
                    "predicate": agg([r for r in one_s if r["stratum"] == "predicate"]),
                    "greedy_exact": sum(1 for r in one_s if r["greedy"]["exact"])},
            "TWO": {"all": agg(two_s),
                    "object": agg([r for r in two_s if r["stratum"] == "object"]),
                    "predicate": agg([r for r in two_s if r["stratum"] == "predicate"]),
                    "row_order_0": agg([r for r in two_s if r["row_order"] == 0]),
                    "row_order_1": agg([r for r in two_s if r["row_order"] == 1]),
                    "greedy_exact": sum(1 for r in two_s if r["greedy"]["exact"])},
        }
        results["per_checkpoint"][ck_name] = res

    # descriptive comparisons with frozen A, C, gloss
    ac = json.loads(AC_AGG.read_text(encoding="utf-8"))
    gloss = json.loads(GLOSS_AGG.read_text(encoding="utf-8"))
    results["comparisons"] = {}
    for ck in results["per_checkpoint"]:
        one = results["per_checkpoint"][ck]["ONE"]["all"]
        two = results["per_checkpoint"][ck]["TWO"]["all"]
        results["comparisons"][ck] = {
            "ONE_accuracy": one["accuracy"], "TWO_accuracy": two["accuracy"],
            "A_accuracy": ac["per_checkpoint"][ck]["A"]["all"]["accuracy"],
            "C_accuracy": ac["per_checkpoint"][ck]["C"]["all"]["accuracy"],
            "gloss_accuracy": gloss["per_checkpoint"][ck]["all"]["accuracy"],
            "ONE_greedy_exact": results["per_checkpoint"][ck]["ONE"]["greedy_exact"],
            "TWO_greedy_exact": results["per_checkpoint"][ck]["TWO"]["greedy_exact"],
            "A_greedy_exact": ac["per_checkpoint"][ck]["A"]["greedy_exact"],
            "C_greedy_exact": ac["per_checkpoint"][ck]["C"]["greedy_exact"],
            "gloss_greedy_exact": gloss["per_checkpoint"][ck]["greedy_exact"],
            "ONE_mean_margin": one["mean_margin"], "TWO_mean_margin": two["mean_margin"],
        }

    results["provenance"] = {
        "package": str(PKG),
        "items_one_sha256": sha256_file(PKG / "ITEMS_ONE.jsonl"),
        "items_two_sha256": sha256_file(PKG / "ITEMS_TWO.jsonl"),
        "tokenizer_sha256": sha256_file(TOKENIZER_PATH),
        "evaluator_sha256": sha256_file(ROOT / "fact_supervision_87001_eval_v1/EVALUATE.py"),
        "checkpoints": {k: sha256_file(v) for k, v in CHECKPOINTS.items()},
        "ac_aggregates_sha256": sha256_file(AC_AGG),
        "gloss_aggregates_sha256": sha256_file(GLOSS_AGG),
        "device": str(dev),
        "training": False, "sacred_access": False, "confirmation_access": False, "seed_87003_access": False,
    }
    (OUT / "aggregates.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    md = ["# Fork L2 minimal-lexical paired diagnostic - read-only execution", ""]
    for ck in results["per_checkpoint"]:
        md.append(f"## {ck}")
        md.append(json.dumps(results["per_checkpoint"][ck], indent=1))
    md.append("## Descriptive comparisons (A/C/gloss)")
    md.append(json.dumps(results["comparisons"], indent=1))
    (OUT / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "SHA256SUMS.txt").write_text(
        f"{sha256_file(OUT / 'aggregates.json')}  aggregates.json\n", encoding="utf-8")

    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"EXEC ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
