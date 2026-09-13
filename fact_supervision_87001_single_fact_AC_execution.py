r"""Read-only execution of the frozen single-fact A+C diagnostic (seed-87002 study).

Uses exactly the frozen package at fact_supervision_87001_single_fact_AC_seed87002
and the three pinned checkpoints. Scoring reproduces EVALUATE.py semantics:
candidate conditional LL over response tokens + EOS on the English base_model
path, signed margin, correct iff margin>0, tie==0, greedy argmax<=32 stop EOS.

No training, no modification, no forbidden partitions/checkpoints/material.
Outputs are written only to a new additive evaluation directory.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(r"C:\DaveLM-CADAVER")
PKG = ROOT / "fact_supervision_87001_single_fact_AC_seed87002"
OUT = ROOT / "fact_supervision_87001_single_fact_AC_execution"
TOKENIZER_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")

CHECKPOINTS = {
    "Pilot1 parent": ROOT / "language_pilot_1_early_block_protection_seed8380" / "pilot_run" / "checkpoints" / "seed_8380" / "latest.pt",
    "factual-500": ROOT / "fact_supervision_87001_corrected_v8" / "run_factual" / "checkpoint_500.pt",
    "control-500": ROOT / "fact_supervision_87001_corrected_v8" / "run_control" / "checkpoint_500.pt",
}
EXPECTED = {
    "parent": "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb",
    "factual": "3ae30ae847d3129d7715173ddbfbbaf6b7490772dc850e361a5a2d785b991123",
    "control": "c888e3b4cf20860b8d1d0651e418be931da7e8db2acc1c32625edb29701f1530",
    "tokenizer": "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b",
    "evaluator": "3563cb58334cd9753ca8ebff7fe49987a0f0ccb535abdfdf9a149cc1541a497b",
}
EOS = 3
BOS = 2


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_model(path: Path, dev):
    sys_path_insert = True
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, r"C:\DaveLM-v0.9")
    sys.path.insert(0, str(ROOT / "fact_supervision_87001_eval_v1"))
    from treatment13_model import Treatment13Model
    from PINNED_PILOT1_BINDING_IMPLEMENTATION import OrthoLocalizer
    raw = torch.load(path, map_location=dev, weights_only=True)
    sd = dict(raw["model_state_dict"])
    keys = ["u", "q", "bs", "ba"]
    vals = [sd.pop("localizer." + k) for k in keys]
    m = Treatment13Model()
    m.load_state_dict(sd, strict=False)
    m.localizer = OrthoLocalizer(*vals)
    return m.to(dev)


@torch.no_grad()
def item_score(m, row, dev, correct_index, tok):
    cands = row["candidate_token_ids"]
    prompt_token_ids = tok.encode(row["prompt"]).ids
    prefix = [BOS] + prompt_token_ids
    out = []
    for c in cands:
        ids = prefix + list(c) + [EOS]
        x = torch.tensor([ids], device=dev)
        logits = m.base_model(x)[0]
        lp = logits[len(prefix) - 1: len(prefix) - 1 + len(c) + 1].log_softmax(-1)
        idx = torch.tensor(list(c) + [EOS], device=dev)
        vals = lp[torch.arange(len(idx), device=dev), idx].detach().cpu().tolist()
        out.append({"candidate_ll": sum(vals[:-1]), "eos_ll": sum(vals), "tokens": vals})
    margin = out[0]["candidate_ll"] - out[1]["candidate_ll"]
    signed = margin if correct_index == 0 else -margin
    return {"id": row["id"], "family_id": row["family_id"], "stratum": row["stratum"],
            "correct_index": correct_index, "correct_name": row["correct_name"],
            "other_name": row["other_name"], "neutral_position": row.get("neutral_position"),
            "query": row.get("query"), "prompt_token_ids": prompt_token_ids,
            "scores": out, "margin": signed,
            "correct": signed > 0, "tie": signed == 0}


@torch.no_grad()
def greedy(m, row, dev, correct_index, tok):
    target = row["candidate_token_ids"][correct_index] + [EOS]
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
    families_complete = sum(all(r["correct"] for r in v) for v in by_family.values())
    ci = Counter(r["correct_index"] for r in rows)
    name = Counter(r["correct_name"] for r in rows)
    return {
        "n": len(rows), "correct": sum(r["correct"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "ties": sum(r["tie"] for r in rows),
        "families_complete": families_complete, "families_total": len(by_family),
        "mean_margin": statistics.mean(r["margin"] for r in rows),
        "median_margin": statistics.median(r["margin"] for r in rows),
        "min_margin": min(r["margin"] for r in rows),
        "max_margin": max(r["margin"] for r in rows),
        "correct_index": dict(sorted(ci.items())), "correct_name": dict(sorted(name.items())),
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    require(torch.cuda.is_available() if args.device == "cuda" else True, "GPU unavailable")
    dev = torch.device(args.device)

    # Re-verify package
    sums = {name.split("  ", 1)[1]: name.split("  ", 1)[0]
            for name in (PKG / "SHA256SUMS.txt").read_text().splitlines() if name.strip()}
    for f in PKG.iterdir():
        if f.is_file() and f.name != "SHA256SUMS.txt":
            require(sha256_file(f) == sums[f.name], f"package hash mismatch: {f.name}")
    receipt = json.loads((PKG / "FREEZE_RECEIPT.json").read_text())
    require((PKG / "FREEZE_RECEIPT.sha256").read_text().split()[0] == sha256_file(PKG / "FREEZE_RECEIPT.json"),
            "freeze receipt chain")
    require(receipt["items_a_sha256"] == sha256_file(PKG / "ITEMS_A.jsonl")
            and receipt["items_c_sha256"] == sha256_file(PKG / "ITEMS_C.jsonl"), "item hash mismatch")
    require(sha256_file(TOKENIZER_PATH) == EXPECTED["tokenizer"], "tokenizer hash")
    require(sha256_file(ROOT / "fact_supervision_87001_eval_v1/EVALUATE.py") == EXPECTED["evaluator"],
            "evaluator hash")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    for key, path in CHECKPOINTS.items():
        require(sha256_file(path) == EXPECTED[key.replace("Pilot1 parent", "parent").replace("factual-500", "factual").replace("control-500", "control")],
                f"checkpoint hash mismatch: {key}")
    require(receipt["status"] == "SINGLE_FACT_AC_FREEZE_COMPLETE_PREEXECUTION_ONLY", "freeze status")

    rows_a = [json.loads(l) for l in (PKG / "ITEMS_A.jsonl").read_text().splitlines()]
    rows_c = [json.loads(l) for l in (PKG / "ITEMS_C.jsonl").read_text().splitlines()]
    require(len(rows_a) == 48 and len(rows_c) == 96, "item count")

    OUT.mkdir(parents=True, exist_ok=True)
    results = {"provenance": {}, "per_checkpoint": {}}
    raw_path = OUT / "raw_per_item.jsonl"
    raw_path.write_text("", encoding="utf-8")

    for ck_name, path in CHECKPOINTS.items():
        m = load_model(path, dev)
        m.eval()
        ck = {"A": {}, "C": {}}
        for tag, rows, kind in (("A", rows_a, "A"), ("C", rows_c, "C")):
            scored = []
            for row in rows:
                z = item_score(m, row, dev, row["correct_index"], tok)
                z["greedy"] = greedy(m, row, dev, row["correct_index"], tok)
                z["checkpoint"] = ck_name
                z["kind"] = kind
                scored.append(z)
            with raw_path.open("a", encoding="utf-8") as fh:
                for z in scored:
                    fh.write(json.dumps(z) + "\n")
            ck[kind]["all"] = agg(scored)
            ck[kind]["object"] = agg([r for r in scored if r["stratum"] == "object"])
            ck[kind]["predicate"] = agg([r for r in scored if r["stratum"] == "predicate"])
            ck[kind]["greedy_exact"] = sum(1 for r in scored if r["greedy"]["exact"])
            if kind == "C":
                ck[kind]["factual_first"] = agg([r for r in scored if r["neutral_position"] == 0])
                ck[kind]["neutral_first"] = agg([r for r in scored if r["neutral_position"] == 1])
            results["per_checkpoint"][ck_name] = ck

    # Paired A-vs-C difference (descriptive, per checkpoint)
    for ck in results["per_checkpoint"]:
        a = results["per_checkpoint"][ck]["A"]["all"]
        c = results["per_checkpoint"][ck]["C"]["all"]
        results["per_checkpoint"][ck]["paired_diff"] = {
            "A_correct": a["correct"], "C_correct": c["correct"],
            "A_total": a["n"], "C_total": c["n"],
            "A_mean_margin": a["mean_margin"], "C_mean_margin": c["mean_margin"],
            "A_greedy_exact": results["per_checkpoint"][ck]["A"]["greedy_exact"],
            "C_greedy_exact": results["per_checkpoint"][ck]["C"]["greedy_exact"],
        }

    results["provenance"] = {
        "package": str(PKG),
        "items_a_sha256": sha256_file(PKG / "ITEMS_A.jsonl"),
        "items_c_sha256": sha256_file(PKG / "ITEMS_C.jsonl"),
        "tokenizer_sha256": sha256_file(TOKENIZER_PATH),
        "evaluator_sha256": sha256_file(ROOT / "fact_supervision_87001_eval_v1/EVALUATE.py"),
        "checkpoints": {k: sha256_file(v) for k, v in CHECKPOINTS.items()},
        "device": str(dev),
        "sacred_access": False, "confirmation_access": False, "seed_87003_access": False,
        "training": False,
    }

    (OUT / "aggregates.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    md = []
    md.append("# Single-fact A+C diagnostic (seed-87002) — read-only execution")
    md.append("")
    for ck in results["per_checkpoint"]:
        for kind in ("A", "C"):
            md.append(f"## {ck} — {kind}")
            md.append(json.dumps(results["per_checkpoint"][ck][kind], indent=1))
        md.append(f"## {ck} — paired A-vs-C difference")
        md.append(json.dumps(results["per_checkpoint"][ck]["paired_diff"], indent=1))
    (OUT / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "SHA256SUMS.txt").write_text(
        f"{sha256_file(OUT / 'aggregates.json')}  aggregates.json\n", encoding="utf-8")

    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    import sys
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"EXEC ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
