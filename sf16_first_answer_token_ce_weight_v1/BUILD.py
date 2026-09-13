"""Build prospective SF16 solely from sealed SF15 plus a 0.125 first-answer CE weight."""
import difflib
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parent
SF15 = ROOT / "sf15_partial_first_token_ce_v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def verify_predecessor():
    receipt = json.loads((SF15 / "FREEZE_RECEIPT.json").read_text(encoding="utf-8"))
    assert sha(SF15 / "FREEZE_RECEIPT.json") == (SF15 / "FREEZE_RECEIPT.sha256").read_text().split()[0]
    assert sha(SF15 / "SHA256SUMS.txt") == receipt["manifest_sha256"]
    for line in (SF15 / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert sha(SF15 / name) == expected, name
    return receipt


def main():
    assert not any(p.name != "BUILD.py" for p in H.iterdir())
    predecessor_receipt = verify_predecessor()

    payloads = [
        "D3_SELECTION.json", "DEV_ORDER.json", "DEV_SURFACE.json", "EXTERNAL_INPUTS.json",
        "KL_POOL_MANIFEST.json", "KL_POOL.json", "SCHEDULE.json", "SF13_KL_SCHEDULE.json",
        "SF2_ENGINE.py", "SF2_PROTOCOL.json", "TRAIN.json", "TRAIN16_RETENTION.json",
    ]
    for name in payloads:
        shutil.copyfile(SF15 / name, H / name)
    for directory in ["data", "sources"]:
        shutil.copytree(SF15 / directory, H / directory)

    old = (SF15 / "CONTROLLER.py").read_text(encoding="utf-8")
    new = old.replace(
        '"""SF15: sealed SF14 machinery with first-answer-token causal CE weight restored to 0.25.\n\nThe sole scientific change from SF14 is the first answer-name CE weight. All later CE,\nmargin, broad KL, curriculum, optimizer, scope, gates and persistence remain unchanged."""',
        '"""SF16: sealed SF15 machinery with first-answer-token causal CE weight interpolated to 0.125.\n\nThe sole scientific change from SF15 is the first answer-name CE weight (0.25 -> 0.125). All later CE,\nmargin, broad KL, curriculum, optimizer, scope, gates and persistence remain unchanged."""',
        1,
    )
    assert new != old
    old_weight_line = "FIRST_ANSWER_CE_WEIGHT = 0.25"
    new_weight_line = "FIRST_ANSWER_CE_WEIGHT = 0.125"
    assert new.count(old_weight_line) == 1
    new = new.replace(old_weight_line, new_weight_line, 1)
    old_denom = "assert abs(ce_meta['ce_weight_sum'] - len(u['ids']) * 4.25) < 1e-5"
    new_denom = "assert abs(ce_meta['ce_weight_sum'] - len(u['ids']) * 4.125) < 1e-5"
    assert new.count(old_denom) == 1
    new = new.replace(old_denom, new_denom, 1)
    new = new.replace("SF15_PROSPECTIVE_PREFLIGHT_PASS", "SF16_PROSPECTIVE_PREFLIGHT_PASS")
    new = new.replace("SF15_PRE_PARENT_LOAD_PASS", "SF16_PRE_PARENT_LOAD_PASS")
    # LAMBDA_MARGIN = 0.25 (line 20) must remain untouched: it is the unrelated pairwise margin weight.
    assert new.count("LAMBDA_MARGIN = 0.25") == 1
    (H / "CONTROLLER.py").write_text(new, encoding="utf-8", newline="\n")
    diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile="SF15/CONTROLLER.py", tofile="SF16/CONTROLLER.py"))
    (H / "CONTROLLER_DIFF.patch").write_text(diff, encoding="utf-8", newline="\n")

    protocol = json.loads((SF15 / "PROTOCOL.json").read_text(encoding="utf-8"))
    parents = [
        (87041, 87017, "9e293af6d16cb642ba8e2bf1aeb5ad392ff4b27399ebbf0b7b8fd2ef7339b81d", 0.007036717671962123),
        (87042, 87018, "839f7f5a60b33a1736376c8a68a02985fb05ba3230e56b7aaf20d1eba5b35102", 0.006672408242356376),
        (87043, 87019, "eb6a725173ff26516d876914ea4357cce7e3758616c3d659ff059d2354404517", 0.007839491795780695),
    ]
    protocol.update({
        "study": "SF16_FIRST_ANSWER_TOKEN_CE_INTERPOLATION_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": [x[0] for x in parents],
        "runs": [
            {
                "seed": seed,
                "arm": "curriculum",
                "parent_checkpoint": str(ROOT / f"sf8_margin_dose_comparison_v1/runs/seed_{parent}_low/checkpoint_200.pt"),
                "parent_checkpoint_sha256": parent_hash,
                "d3_reference": d3,
                "parent_sf8_seed": parent,
            }
            for seed, parent, parent_hash, d3 in parents
        ],
        "hypothesis": "A 0.125 first-answer-name CE weight can preserve the SF15 exact-generation recovery while restoring durable D3 containment (<=0.010) at U100 and U200.",
        "manipulated_variable": "Relative to sealed SF15, change only the first answer-name causal CE position weight from 0.25 to 0.125 on every English update.",
        "english_objective": "Weighted-mean causal CE with per-record response weights [0.125,1,1,1,1] +1.0 full-vocabulary forward KL on the frozen broad-coverage 160 positions +0.25 first-answer-token hinge with M=1.0. No R_name.",
        "first_answer_ce": {
            "weight": 0.125,
            "scope": "all 36 records in each widening English update, including the four scheduled TRAIN16 preservation records",
            "normalization": "sum weighted token CE divided by sum of supervised token weights (36*4.125 per English update)",
            "later_candidate_punctuation_eos_weight": 1.0,
            "margin_at_first_token": "unchanged",
            "binding_updates": "unchanged",
            "endpoint_equivalence": "weight 0 exactly reproduces sealed SF14 CE; weight 1 exactly reproduces sealed SF13 CE",
        },
        "classification": {
            "priority1": "MECHANICAL_INCOMPLETE if execution or integrity prevents valid classification",
            "priority2": "RETENTION_REGRESSION if any frozen TRAIN16, language, or binding retention gate fails",
            "priority3": "FIRST_TOKEN_CE_INTERPOLATION_SUPPORTED if >=2/3 branches reach u200 and pass all frozen endpoint gates including D3<=0.010",
            "priority4": "FIRST_TOKEN_CE_INTERPOLATION_INSUFFICIENT if >=2/3 valid branches either fail D3 while retention and widening survive, or reproduce SF14-like exact-generation failure (surface exact<=2/16 and order exact<=2/16) while D3 and retention remain green",
            "otherwise": "MIXED_OR_UNRESOLVED",
        },
        "interpretation": "Success supports only that a 0.125 first-answer-token CE weight was sufficient under SF16 to preserve D3 while recovering exact factual generation/retention and maintaining tested language, binding, and widening behavior. Failure reports the observed tradeoff; it does not establish optimality, sole causation, a scaling law, or a capacity limit.",
        "next_action_rule": "Freeze and report SF16 only. SF16 is the final simple first-answer-token CE dose interpolation in this program: no automatic micro-dose sweep (0.10/0.15/0.1875/etc.) follows a failure. If insufficient, preserve the result and recommend SF17 move to a different treatment mechanism class. Do not run SF17 automatically.",
    })
    protocol.pop("first_answer_ce_ablation", None)
    write_json(H / "PROTOCOL.json", protocol)
    write_json(H / "PROVENANCE.json", {
        "study": "SF16_FIRST_ANSWER_TOKEN_CE_INTERPOLATION_V1",
        "predecessor": str(SF15),
        "predecessor_receipt_status": predecessor_receipt["status"],
        "predecessor_receipt_sha256": sha(SF15 / "FREEZE_RECEIPT.json"),
        "predecessor_manifest_sha256": sha(SF15 / "SHA256SUMS.txt"),
        "unchanged_payload_hashes": {name: sha(H / name) for name in payloads},
        "sole_scientific_change": "first answer-name causal CE weight changed from 0.25 to 0.125 during English updates",
        "weighted_mean_definition": "sum(w_i*CE_i)/sum(w_i), weights [0.125,1,1,1,1] per response",
        "fresh_seeds": [87041, 87042, 87043],
        "final_sacred": "LOCKED_NOT_ACCESSED",
    })
    print("SF16_BUILD_COMPLETE")


if __name__ == "__main__":
    main()
