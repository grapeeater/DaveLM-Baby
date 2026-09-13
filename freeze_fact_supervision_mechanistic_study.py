r"""Freeze + structural validation for the mechanistic localization study.

Writes the pre-execution protocol/manifest/validation/preexecution/receipt and
SHA256SUMS for the census study over the frozen L2 ONE/TWO diagnostic. No model
or checkpoint is loaded here; validation is tokenizer/position/pairing only.

FROZEN DESIGN (recorded in PROTOCOL.md):
Stage 1 (always): read-only representational census.
  - Semantic positions per item (tokenizer-aligned): answer position =
    len([BOS]+prompt_tokens)-1; correct-name token position; other-name token
    position (TWO only); these are located by the frozen name token sequences.
  - Representation: base_model.final_norm hidden state (eval mode, no_grad).
  - Per family, reference actor direction d = h_ans(ONE q with actor A) -
    h_ans(ONE q with actor B) (normalized), where {A,B} are the two correct
    actors across the family's two ONE queries. Full set; no outcome selection.
  - Measurements: projection score = dot(h_ans, d); correct-side = projection
    positive for actor-A items and negative for actor-B items; correct-side rate
    for ONE and for TWO (overall, by stratum, by row order); cosine between
    matched ONE/TWO answer hidden states; cosine of answer hidden vs
    token_embedding[correct-name token] and [other-name token].
Stage 2 (conditional causal patch, gated BEFORE results):
  - Gate: for a checkpoint, run Stage 2 iff (TWO correct-side rate >= 0.90)
    AND (frozen TWO behavioral accuracy <= 0.60). Otherwise Stage 2 is skipped
    for that checkpoint and recorded as not triggered.
  - Patch: replace the TWO item's answer-position hidden with the answer hidden
    of its matched ONE item (same family/query), then recompute the answer
    position logits via the frozen language_head; report first-token
    correct-actor selection (first candidate token LL from the frozen scorer
    semantics; margins use the frozen first-token LL). Controls:
      sham = patch with its own hidden (identity);
      wrong-actor = patch with the matched ONE hidden of the OTHER query of the
      same family;
      row-order = both TWO row orders reported separately.
No training, no weight modification, no outcome-selected subsets.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(r"C:\DaveLM-CADAVER")
PKG = ROOT / "fact_supervision_87001_minimal_lexical_pair_seed87002"
EXEC = ROOT / "fact_supervision_87001_minimal_lexical_pair_execution"
OUT = ROOT / "fact_supervision_87001_mechanistic_study_seed87002"
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")

EXPECTED_TOKENIZER_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
EXPECTED_EVALUATOR_SHA = "3563cb58334cd9753ca8ebff7fe49987a0f0ccb535abdfdf9a149cc1541a497b"

CHECKPOINTS = {
    "Pilot1 parent": "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb",
    "factual-500": "3ae30ae847d3129d7715173ddbfbbaf6b7490772dc850e361a5a2d785b991123",
    "control-500": "c888e3b4cf20860b8d1d0651e418be931da7e8db2acc1c32625edb29701f1530",
}
GATE = {"two_correct_side_rate_min": 0.90, "two_behavioral_accuracy_max": 0.60}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def main() -> int:
    require(sha256_file(TOKENIZER) == EXPECTED_TOKENIZER_SHA, "tokenizer")
    require(sha256_file(ROOT / "fact_supervision_87001_eval_v1/EVALUATE.py") == EXPECTED_EVALUATOR_SHA, "evaluator")

    tok = Tokenizer.from_file(str(TOKENIZER))
    one = [json.loads(l) for l in (PKG / "ITEMS_ONE.jsonl").read_text().splitlines()]
    two = [json.loads(l) for l in (PKG / "ITEMS_TWO.jsonl").read_text().splitlines()]
    require(len(one) == 48 and len(two) == 96, "counts")

    # pairing and token-position validation (tokenizer only)
    by_fam_one = defaultdict(list)
    for it in one:
        by_fam_one[it["family_id"]].append(it)
    require(len(by_fam_one) == 24 and all(len(v) == 2 for v in by_fam_one.values()), "ONE pairing")
    for fam, grp in by_fam_one.items():
        names = {g["correct_name"] for g in grp}
        require(len(names) == 2, f"{fam}: ONE actor set != 2")

    def find_name_occ(pids, name):
        forms = [tok.encode(" " + name).ids, tok.encode(name).ids]
        for nids in forms:
            if not nids:
                continue
            occ = [i for i in range(len(pids) - len(nids) + 1) if pids[i:i + len(nids)] == nids]
            if occ:
                return occ[0], nids
        return None, None

    for it in one + two:
        pids = tok.encode(it["prompt"]).ids
        pos, _ = find_name_occ(pids, it["correct_name"])
        require(pos is not None, f"{it['id']}: correct-name token sequence missing")

    # family/query matched TWO pairing for causal stage
    two_keys = defaultdict(set)
    for it in two:
        two_keys[it["family_id"]].add((it["query"], it["row_order"]))
    require(len(two_keys) == 24 and all(len(v) == 4 for v in two_keys.values()), "TWO pairing")

    protocol = (
        "# Mechanistic localization study - pre-execution freeze\n\n"
        "Scientific question: when a competing mapping is added (ONE->TWO), is usable "
        "query-conditioned correct-actor information absent from the answer-position hidden "
        "state, or present but overridden by competition/position?\n\n"
        "Stage 1 (always): read-only representational census on the frozen L2 ONE/TWO items "
        "using the answer-position final_norm hidden state, a per-family actor direction built "
        "from matched ONE pairs (full preregistered set, no outcome selection), correct-side "
        "projection rates, matched ONE/TWO cosine similarity, and cosine alignment with the "
        "frozen token embeddings of the two names.\n\n"
        "Stage 2 (conditional, gated BEFORE results): patch the TWO answer-position hidden with "
        "the matched ONE answer hidden and recompute the answer-position logits through the "
        "frozen language_head; report first-token correct-actor selection under the frozen "
        "first-token LL. Controls: sham (own hidden), wrong-actor (other query's ONE hidden), "
        "and both row orders. Stage 2 runs for a checkpoint only if TWO correct-side rate >= 0.90 "
        "AND frozen TWO behavioral accuracy <= 0.60; otherwise it is recorded as not triggered.\n\n"
        "No training, no weight modification, no outcome-selected subsets, no new behavioral "
        "items, no trained probes, no thresholds beyond the preregistered gate.\n"
    )
    manifest = {
        "protocol_id": "fact_supervision_87001_mechanistic_study_seed87002",
        "status": "PREEXECUTION_ONLY",
        "input_package": str(PKG),
        "input_execution": str(EXEC),
        "checkpoints": CHECKPOINTS,
        "tokenizer_sha256": EXPECTED_TOKENIZER_SHA,
        "evaluator_sha256": EXPECTED_EVALUATOR_SHA,
        "gate": GATE,
        "checkpoint_loaded": False, "inference_run": False,
        "sacred_access": False, "confirmation_access": False, "seed_87003_access": False,
        "l1_forkA_dev_train_forbidden": True,
    }
    validation = {
        "status": "VALIDATION_PASS",
        "one_items": len(one), "two_items": len(two),
        "one_families": len(by_fam_one), "one_items_per_family": 2,
        "two_families": len(two_keys), "two_items_per_family": 4,
        "one_actor_set_size_2_per_family": True,
        "correct_name_token_sequence_present_in_prompt": True,
        "notes": ("correct-name token sequences located by the frozen candidate token ids (name "
                  "tokens without the trailing period); no model was loaded"),
    }
    preexecution = {
        "status": "PREEXECUTION_ONLY",
        "checkpoint_loaded": False, "inference_run": False, "optimizer_created": False,
        "sacred_access": False, "confirmation_access": False, "seed_87003_access": False,
        "gate": GATE,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "PROTOCOL.md").write_text(protocol, encoding="utf-8")
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (OUT / "VALIDATION_REPORT.json").write_text(json.dumps(validation, indent=1), encoding="utf-8")
    (OUT / "PREEXECUTION.json").write_text(json.dumps(preexecution, indent=1), encoding="utf-8")

    receipt = {
        "status": "MECHANISTIC_STUDY_FREEZE_COMPLETE_PREEXECUTION_ONLY",
        "bundle": "fact_supervision_87001_mechanistic_study_seed87002",
        "protocol_sha256": sha256_file(OUT / "PROTOCOL.md"),
        "manifest_sha256": sha256_file(OUT / "MANIFEST.json"),
        "validation_sha256": sha256_file(OUT / "VALIDATION_REPORT.json"),
        "preexecution_sha256": sha256_file(OUT / "PREEXECUTION.json"),
        "checkpoint_loaded": False, "inference_run": False,
        "sacred_access": False, "confirmation_access": False, "seed_87003_access": False,
    }
    (OUT / "FREEZE_RECEIPT.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    (OUT / "FREEZE_RECEIPT.sha256").write_text(
        sha256_file(OUT / "FREEZE_RECEIPT.json") + "  FREEZE_RECEIPT.json\n", encoding="utf-8")
    sums = sorted(f"{sha256_file(OUT / f.name)}  {f.name}"
                  for f in OUT.iterdir() if f.is_file() and f.name != "SHA256SUMS.txt")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "validation": validation["status"]}, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(f"FREEZE ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
