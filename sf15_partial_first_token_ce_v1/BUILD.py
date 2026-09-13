"""Build prospective SF15 solely from sealed SF14 plus a 0.25 first-answer CE weight."""
import difflib
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

H = Path(__file__).resolve().parent
ROOT = H.parent
SF14 = ROOT / "sf14_first_answer_token_ce_ablation_v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def verify_predecessor():
    receipt = json.loads((SF14 / "FREEZE_RECEIPT.json").read_text(encoding="utf-8"))
    assert sha(SF14 / "FREEZE_RECEIPT.json") == (SF14 / "FREEZE_RECEIPT.sha256").read_text().split()[0]
    assert sha(SF14 / "SHA256SUMS.txt") == receipt["manifest_sha256"]
    for line in (SF14 / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert sha(SF14 / name) == expected, name
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
        shutil.copyfile(SF14 / name, H / name)
    for directory in ["data", "sources"]:
        shutil.copytree(SF14 / directory, H / directory)

    old = (SF14 / "CONTROLLER.py").read_text(encoding="utf-8")
    new = old.replace(
        '"""SF13: broad-coverage KL retention at fixed retention compute (sampling geometry change).\n\nONE scientific variable vs SF11/SF12: retention position coverage / sampling geometry\n(contiguous sequential 160-entry slice -> frozen row-distributed 160-entry schedule).\nAll loss terms, curriculum, optimizer, scope, gates and persistence are unchanged from SF11."""',
        '"""SF15: sealed SF14 machinery with first-answer-token causal CE weight restored to 0.25.\n\nThe sole scientific change from SF14 is the first answer-name CE weight. All later CE,\nmargin, broad KL, curriculum, optimizer, scope, gates and persistence remain unchanged."""',
        1,
    )
    start = new.index("NAME_TOKEN_IDS = (314, 536, 925, 512)")
    end = new.index("def verify(sealed=True):", start)
    weighted = '''NAME_TOKEN_IDS = (314, 536, 925, 512)\nFIRST_ANSWER_CE_WEIGHT = 0.25\n\n\ndef english_ce_weights(labels, ids_in_batch, idx, first_weight=FIRST_ANSWER_CE_WEIGHT):\n    """Return frozen per-label CE weights; ignored labels remain zero."""\n    assert 0.0 <= float(first_weight) <= 1.0\n    weights = (labels != -100).to(dtype=torch.float32)\n    for k, rid in enumerate(ids_in_batch):\n        record = idx[rid]\n        pos = len(record['prompt_token_ids'])\n        expected = record['candidate_token_ids'][record['correct_index']][0]\n        assert expected in NAME_TOKEN_IDS\n        assert int(labels[k, pos]) == expected\n        weights[k, pos] = float(first_weight)\n    return weights\n\n\ndef weighted_english_ce(logits, labels, ids_in_batch, idx, first_weight=FIRST_ANSWER_CE_WEIGHT):\n    """Weighted mean causal CE; weight 0 reproduces SF14 and weight 1 reproduces SF13."""\n    weights = english_ce_weights(labels, ids_in_batch, idx, first_weight).to(device=logits.device, dtype=logits.dtype)\n    raw = F.cross_entropy(\n        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),\n        ignore_index=-100, reduction='none'\n    ).reshape_as(labels)\n    denominator = weights.sum()\n    assert float(denominator.detach()) > 0.0\n    return (raw * weights).sum() / denominator, {\n        'first_answer_positions': len(ids_in_batch),\n        'later_response_eos_positions': int((labels != -100).sum()) - len(ids_in_batch),\n        'ce_weight_sum': float(denominator.detach()),\n+    }\n\n\n'''
    # Strip an accidental patch marker if this source string is edited mechanically.
    weighted = weighted.replace("\n+    }", "\n    }")
    new = new[:start] + weighted + new[end:]
    new = new.replace("SF14_PROSPECTIVE_PREFLIGHT_PASS", "SF15_PROSPECTIVE_PREFLIGHT_PASS")
    new = new.replace("SF14_PRE_PARENT_LOAD_PASS", "SF15_PRE_PARENT_LOAD_PASS")
    new = new.replace("'first_answer_token_ce_weight': 0.0", "'first_answer_token_ce_weight': FIRST_ANSWER_CE_WEIGHT")
    old_train = '''            x, y = E.pad_batch(batch, u['pad'])\n            y = mask_first_answer_ce_labels(y, u['ids'], idx)\n            assert int((y != -100).sum()) == len(u['ids']) * 4\n            x = x.to(device); y = y.to(device)\n            logits = m.base_model(x)\n            ce = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1), ignore_index=-100)'''
    new_train = '''            x, y = E.pad_batch(batch, u['pad'])\n            assert int((y != -100).sum()) == len(u['ids']) * 5\n            x = x.to(device); y = y.to(device)\n            logits = m.base_model(x)\n            ce, ce_meta = weighted_english_ce(logits, y, u['ids'], idx)\n            assert ce_meta['first_answer_positions'] == len(u['ids'])\n            assert ce_meta['later_response_eos_positions'] == len(u['ids']) * 4\n            assert abs(ce_meta['ce_weight_sum'] - len(u['ids']) * 4.25) < 1e-5'''
    assert new.count(old_train) == 1
    new = new.replace(old_train, new_train, 1)
    old_metric = "if ce is not None: rec.update(ce=float(ce.detach()), kl=float(kl_val.detach()), shard=u.get('shard'), first_answer_ce_masked=len(u['ids']), ce_supervised_tokens=len(u['ids'])*4)"
    new_metric = "if ce is not None: rec.update(ce=float(ce.detach()), kl=float(kl_val.detach()), shard=u.get('shard'), first_answer_ce_weight=FIRST_ANSWER_CE_WEIGHT, later_response_ce_weight=1.0, ce_weight_sum=ce_meta['ce_weight_sum'], ce_supervised_positions=len(u['ids'])*5)"
    assert new.count(old_metric) == 1
    new = new.replace(old_metric, new_metric, 1)
    assert "mask_first_answer_ce_labels" not in new
    assert new.count("weighted_english_ce(logits, y, u['ids'], idx)") == 1
    (H / "CONTROLLER.py").write_text(new, encoding="utf-8", newline="\n")
    diff = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile="SF14/CONTROLLER.py", tofile="SF15/CONTROLLER.py"))
    (H / "CONTROLLER_DIFF.patch").write_text(diff, encoding="utf-8", newline="\n")

    protocol = json.loads((SF14 / "PROTOCOL.json").read_text(encoding="utf-8"))
    parents = [
        (87038, 87017, "9e293af6d16cb642ba8e2bf1aeb5ad392ff4b27399ebbf0b7b8fd2ef7339b81d", 0.007036717671962123),
        (87039, 87018, "839f7f5a60b33a1736376c8a68a02985fb05ba3230e56b7aaf20d1eba5b35102", 0.006672408242356376),
        (87040, 87019, "eb6a725173ff26516d876914ea4357cce7e3758616c3d659ff059d2354404517", 0.007839491795780695),
    ]
    protocol.update({
        "study": "SF15_PARTIAL_FIRST_ANSWER_TOKEN_CE_RESTORATION_V1",
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
        "hypothesis": "A 0.25 first-answer-name CE weight can retain SF14-like D3 containment while restoring unrestricted exact factual generation and TRAIN16 exact retention.",
        "manipulated_variable": "Relative to sealed SF14, change only the first answer-name causal CE position weight from 0.0 to 0.25 on every English update.",
        "english_objective": "Weighted-mean causal CE with per-record response weights [0.25,1,1,1,1] +1.0 full-vocabulary forward KL on the frozen broad-coverage 160 positions +0.25 first-answer-token hinge with M=1.0. No R_name.",
        "first_answer_ce": {
            "weight": 0.25,
            "scope": "all 36 records in each widening English update, including the four scheduled TRAIN16 preservation records",
            "normalization": "sum weighted token CE divided by sum of supervised token weights (36*4.25 per English update)",
            "later_candidate_punctuation_eos_weight": 1.0,
            "margin_at_first_token": "unchanged",
            "binding_updates": "unchanged",
            "endpoint_equivalence": "weight 0 exactly reproduces sealed SF14 CE; weight 1 exactly reproduces sealed SF13 CE",
        },
        "classification": {
            "priority1": "MECHANICAL_INCOMPLETE if execution or integrity prevents valid classification",
            "priority2": "RETENTION_REGRESSION if any frozen TRAIN16, language, or binding retention gate fails",
            "priority3": "PARTIAL_FIRST_TOKEN_CE_SUPPORTED if >=2/3 branches reach u200 and pass all frozen endpoint gates including D3<=0.010",
            "priority4": "PARTIAL_FIRST_TOKEN_CE_INSUFFICIENT if >=2/3 valid branches either fail D3 while retention and widening survive, or reproduce SF14-like exact-generation failure (surface exact<=2/16 and order exact<=2/16) while D3 and retention remain green",
            "otherwise": "MIXED_OR_UNRESOLVED",
        },
        "interpretation": "Success supports only that a 0.25 first-answer-token CE weight was sufficient under SF15 to preserve D3 while recovering exact factual generation/retention and maintaining tested language, binding, and widening behavior. Failure reports the observed tradeoff; it does not establish optimality, sole causation, a scaling law, or a capacity limit.",
        "next_action_rule": "Freeze and report SF15 only; no rescue, replacement seed, sweep, or follow-up treatment.",
    })
    protocol.pop("first_answer_ce_ablation", None)
    write_json(H / "PROTOCOL.json", protocol)
    write_json(H / "PROVENANCE.json", {
        "study": "SF15_PARTIAL_FIRST_ANSWER_TOKEN_CE_RESTORATION_V1",
        "predecessor": str(SF14),
        "predecessor_receipt_status": predecessor_receipt["status"],
        "predecessor_receipt_sha256": sha(SF14 / "FREEZE_RECEIPT.json"),
        "predecessor_manifest_sha256": sha(SF14 / "SHA256SUMS.txt"),
        "unchanged_payload_hashes": {name: sha(H / name) for name in payloads},
        "sole_scientific_change": "first answer-name causal CE weight changed from 0.0 to 0.25 during English updates",
        "weighted_mean_definition": "sum(w_i*CE_i)/sum(w_i), weights [0.25,1,1,1,1] per response",
        "fresh_seeds": [87038, 87039, 87040],
        "final_sacred": "LOCKED_NOT_ACCESSED",
    })
    print("SF15_BUILD_COMPLETE")


if __name__ == "__main__":
    main()
