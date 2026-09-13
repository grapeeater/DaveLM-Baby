r"""Read-only mechanistic localization study: representational census (+ gated causal patch).

Stage 1 (always): answer-position final_norm hidden census on frozen L2 ONE/TWO,
per-family actor direction from matched ONE pairs, correct-side projection rates,
ONE/TWO cosine similarity, name-embedding alignment.
Stage 2 (gated BEFORE results): if TWO correct-side >= 0.90 AND frozen TWO
behavioral accuracy <= 0.60, patch the TWO answer-position hidden with the matched
ONE answer hidden (controls: sham, wrong-actor) and report first-token
correct-actor selection under the frozen first-token LL.

No weight modification, no training, no outcome-selected subsets.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(r"C:\DaveLM-CADAVER")
PKG = ROOT / "fact_supervision_87001_minimal_lexical_pair_seed87002"
EXEC = ROOT / "fact_supervision_87001_minimal_lexical_pair_execution"
STUDY = ROOT / "fact_supervision_87001_mechanistic_study_seed87002"
OUT = ROOT / "fact_supervision_87001_mechanistic_study_execution"
TOKENIZER_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")

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
GATE = {"two_correct_side_rate_min": 0.90, "two_behavioral_accuracy_max": 0.60}


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


def find_name_pos(pids, name, tok):
    for nids in (tok.encode(" " + name).ids, tok.encode(name).ids):
        if not nids:
            continue
        for i in range(len(pids) - len(nids) + 1):
            if pids[i:i + len(nids)] == nids:
                return i
    return None


@torch.no_grad()
def hidden_for(model, tok, item, dev):
    pids = tok.encode(item["prompt"]).ids
    x = torch.tensor([[BOS] + pids], device=dev)
    captured = []
    handle = model.base_model.final_norm.register_forward_hook(
        lambda m, i, o: captured.append(o))
    model.base_model(x)
    handle.remove()
    h = captured[0][0].float()  # [L,H]; answer-position hidden = last token hidden
    return pids, h[-1]


def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    require(torch.cuda.is_available() if args.device == "cuda" else True, "GPU unavailable")
    dev = torch.device(args.device)

    # integrity
    sums = {n: h for h, n in (l.split("  ", 1) for l in (STUDY / "SHA256SUMS.txt").read_text().splitlines() if l.strip())}
    for f in STUDY.iterdir():
        if f.is_file() and f.name != "SHA256SUMS.txt":
            require(sha256_file(f) == sums[f.name], f"study hash mismatch {f.name}")
    require(sha256_file(TOKENIZER_PATH) == EXPECTED_TOKENIZER, "tokenizer")
    require(sha256_file(ROOT / "fact_supervision_87001_eval_v1/EVALUATE.py") == EXPECTED_EVALUATOR, "evaluator")
    for k, p in CHECKPOINTS.items():
        require(sha256_file(p) == EXPECTED_CK[k], f"checkpoint {k}")

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    one = [json.loads(l) for l in (PKG / "ITEMS_ONE.jsonl").read_text().splitlines()]
    two = [json.loads(l) for l in (PKG / "ITEMS_TWO.jsonl").read_text().splitlines()]
    l2 = json.loads((EXEC / "aggregates.json").read_text())

    OUT.mkdir(parents=True, exist_ok=True)
    report = {"provenance": {}, "per_checkpoint": {}}
    raw = OUT / "raw_per_item.jsonl"
    raw.write_text("", encoding="utf-8")

    for ck_name, path in CHECKPOINTS.items():
        m = load_model(path, dev)
        m.eval()
        one_h, two_h = {}, {}
        one_meta, two_meta = {}, {}
        for it in one:
            pids, h = hidden_for(m, tok, it, dev)
            key = (it["family_id"], it["query"])
            one_h[key] = h
            one_meta[key] = it
        for it in two:
            pids, h = hidden_for(m, tok, it, dev)
            key = (it["family_id"], it["query"], it["row_order"])
            two_h[key] = h
            two_meta[key] = it

        # per-family actor direction from matched ONE pairs
        fam_dirs = {}
        for (fam, q), h in one_h.items():
            pass
        fams = sorted({fam for fam, q in one_h})
        for fam in fams:
            (q0, q1) = sorted(q for f, q in one_h if f == fam)
            h0, h1 = one_h[(fam, q0)], one_h[(fam, q1)]
            d = h0 - h1
            n = d.norm()
            if n > 0:
                d = d / n
            fam_dirs[fam] = (d, one_meta[(fam, q0)]["correct_name"], one_meta[(fam, q1)]["correct_name"])

        # projections
        def project(item_h, fam):
            d, A, B = fam_dirs[fam]
            return float(torch.dot(item_h, d)), A, B

        census_rows = []
        for kind, hdict, metadict in (("ONE", one_h, one_meta), ("TWO", two_h, two_meta)):
            for key, h in hdict.items():
                meta = metadict[key]
                fam = key[0]
                s, A, B = project(h, fam)
                correct_name = meta["correct_name"]
                correct_side = (s > 0) if correct_name == A else (s < 0)
                row = {"kind": kind, "checkpoint": ck_name, "family_id": fam,
                       "query": key[1], "row_order": key[2] if kind == "TWO" else None,
                       "correct_name": correct_name, "stratum": meta["stratum"],
                       "projection": s, "correct_side": bool(correct_side)}
                # name-embedding alignment
                pids = tok.encode(meta["prompt"]).ids
                ans_pos = len(pids)  # hidden index of last prompt token
                h_ans = h
                emb = m.base_model.token_embedding.weight
                ci = meta["correct_index"]
                correct_tok = meta["candidate_token_ids"][ci][0]
                other_tok = meta["candidate_token_ids"][1 - ci][0]
                row["cos_correct_name_emb"] = cos(h_ans, emb[correct_tok])
                row["cos_other_name_emb"] = cos(h_ans, emb[other_tok])
                row["cos_one_two"] = None
                if kind == "TWO":
                    ok = (fam, key[1])
                    if ok in one_h:
                        row["cos_one_two"] = cos(h_ans, one_h[ok])
                census_rows.append(row)
                raw.write_text(json.dumps(row) + "\n", encoding="utf-8")

        def rate(rows, kind=None, stratum=None, row_order=None):
            sel = rows
            if kind is not None:
                sel = [r for r in sel if r["kind"] == kind]
            if stratum is not None:
                sel = [r for r in sel if r["stratum"] == stratum]
            if row_order is not None:
                sel = [r for r in sel if r["row_order"] == row_order]
            if not sel:
                return None
            return sum(r["correct_side"] for r in sel) / len(sel)

        one_all = rate(census_rows, "ONE")
        two_all = rate(census_rows, "TWO")
        two_ro0 = rate(census_rows, "TWO", row_order=0)
        two_ro1 = rate(census_rows, "TWO", row_order=1)
        ck = {
            "correct_side_rate": {
                "ONE_all": one_all,
                "ONE_object": rate(census_rows, "ONE", "object"),
                "ONE_predicate": rate(census_rows, "ONE", "predicate"),
                "TWO_all": two_all,
                "TWO_object": rate(census_rows, "TWO", "object"),
                "TWO_predicate": rate(census_rows, "TWO", "predicate"),
                "TWO_row_order_0": two_ro0,
                "TWO_row_order_1": two_ro1,
            },
            "cos_one_two": {
                "all": statistics.mean(r["cos_one_two"] for r in census_rows if r["cos_one_two"] is not None),
                "row_order_0": statistics.mean(r["cos_one_two"] for r in census_rows if r["row_order"] == 0 and r["cos_one_two"] is not None),
                "row_order_1": statistics.mean(r["cos_one_two"] for r in census_rows if r["row_order"] == 1 and r["cos_one_two"] is not None),
            },
            "name_embedding_alignment": {
                "ONE_correct": statistics.mean(r["cos_correct_name_emb"] for r in census_rows if r["kind"] == "ONE"),
                "ONE_other": statistics.mean(r["cos_other_name_emb"] for r in census_rows if r["kind"] == "ONE"),
                "TWO_correct": statistics.mean(r["cos_correct_name_emb"] for r in census_rows if r["kind"] == "TWO"),
                "TWO_other": statistics.mean(r["cos_other_name_emb"] for r in census_rows if r["kind"] == "TWO"),
            },
        }
        # behavioral accuracy from frozen L2 execution
        two_acc = l2["per_checkpoint"][ck_name]["TWO"]["all"]["accuracy"]
        ck["frozen_TWO_behavioral_accuracy"] = two_acc
        trigger = (two_all is not None and two_all >= GATE["two_correct_side_rate_min"]
                   and two_acc <= GATE["two_behavioral_accuracy_max"])
        ck["stage2_triggered"] = bool(trigger)
        ck["stage2"] = {"ran": False, "note": "gate not met"}

        if trigger:
            # gated causal patch: first-token correct-actor selection
            rows2 = []
            for (fam, q, ro), h in two_h.items():
                meta = two_meta[(fam, q, ro)]
                ci = meta["correct_index"]
                c_tok = meta["candidate_token_ids"][ci][0]
                o_tok = meta["candidate_token_ids"][1 - ci][0]
                one_key = (fam, q)
                patches = {"one_patch": one_h[one_key], "sham": h,
                           "wrong_actor": one_h[(fam, 1 - q)]}
                for pname, hp in patches.items():
                    logits = m.base_model.language_head(hp)
                    lls = torch.log_softmax(logits, dim=-1)
                    llc = float(lls[c_tok])
                    llo = float(lls[o_tok])
                    rows2.append({"patch": pname, "family_id": fam, "query": q,
                                  "row_order": ro, "correct": llc > llo,
                                  "margin": llc - llo})
            def sel(pname=None, ro=None):
                r = rows2
                if pname is not None:
                    r = [x for x in r if x["patch"] == pname]
                if ro is not None:
                    r = [x for x in r if x["row_order"] == ro]
                if not r:
                    return None
                return {"n": len(r), "correct_rate": sum(x["correct"] for x in r) / len(r),
                        "mean_margin": statistics.mean(x["margin"] for x in r)}
            ck["stage2"] = {"ran": True,
                            "one_patch": sel("one_patch"), "sham": sel("sham"),
                            "wrong_actor": sel("wrong_actor"),
                            "one_patch_ro0": sel("one_patch", 0), "one_patch_ro1": sel("one_patch", 1)}
            raw2 = OUT / f"raw_stage2_{ck_name.replace(' ', '_')}.jsonl"
            raw2.write_text("".join(json.dumps(x) + "\n" for x in rows2), encoding="utf-8")

        report["per_checkpoint"][ck_name] = ck

    report["provenance"] = {
        "study_package": str(STUDY),
        "input_package": str(PKG),
        "input_execution": str(EXEC),
        "tokenizer_sha256": sha256_file(TOKENIZER_PATH),
        "evaluator_sha256": sha256_file(ROOT / "fact_supervision_87001_eval_v1/EVALUATE.py"),
        "checkpoints": {k: sha256_file(v) for k, v in CHECKPOINTS.items()},
        "l2_aggregates_sha256": sha256_file(EXEC / "aggregates.json"),
        "device": str(dev),
        "training": False, "sacred_access": False, "confirmation_access": False, "seed_87003_access": False,
        "gate": GATE,
    }
    (OUT / "aggregates.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    md = ["# Mechanistic localization study - read-only execution", ""]
    for ck in report["per_checkpoint"]:
        md.append(f"## {ck}")
        md.append(json.dumps(report["per_checkpoint"][ck], indent=1))
    (OUT / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "SHA256SUMS.txt").write_text(
        f"{sha256_file(OUT / 'aggregates.json')}  aggregates.json\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"EXEC ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
