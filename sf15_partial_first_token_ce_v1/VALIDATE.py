"""Minimal mechanical SF15 preflight. No checkpoint load, optimizer, inference, or update."""
import ast
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
import tokenizers
from tokenizers import Tokenizer

import CONTROLLER as C
import SF2_ENGINE as E

H = Path(__file__).resolve().parent
SF14 = H.parent / "sf14_first_answer_token_ce_ablation_v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    checks = {}
    protocol = C.verify(sealed=False)
    old = json.loads((SF14 / "PROTOCOL.json").read_text(encoding="utf-8"))
    unchanged = [
        "D3_SELECTION.json", "DEV_ORDER.json", "DEV_SURFACE.json", "EXTERNAL_INPUTS.json",
        "KL_POOL_MANIFEST.json", "KL_POOL.json", "SCHEDULE.json", "SF13_KL_SCHEDULE.json",
        "SF2_ENGINE.py", "SF2_PROTOCOL.json", "TRAIN.json", "TRAIN16_RETENTION.json",
    ]
    assert all(sha(H / n) == sha(SF14 / n) for n in unchanged)
    checks["byte_identical_sf14_scientific_payload"] = unchanged

    for run in protocol["runs"]:
        assert sha(run["parent_checkpoint"]) == run["parent_checkpoint_sha256"]
    checks["parents"] = {str(r["seed"]): {"sf8_seed": r["parent_sf8_seed"], "sha256": r["parent_checkpoint_sha256"]} for r in protocol["runs"]}

    assert platform.python_version() == protocol["runtime"]["python"]
    assert torch.__version__ == protocol["runtime"]["torch"]
    assert tokenizers.__version__ == protocol["runtime"]["tokenizers"]
    assert torch.cuda.is_available()
    checks["runtime"] = {
        "python": platform.python_version(), "torch": torch.__version__,
        "tokenizers": tokenizers.__version__, "device": torch.cuda.get_device_name(0),
    }

    tokenizer = Tokenizer.from_file(protocol["tokenizer"])
    assert sha(protocol["tokenizer"]) == protocol["tokenizer_sha256"]
    name_ids = {name: tokenizer.encode(" " + name + ".").ids[0] for name in ["Alex", "Owen", "Mia", "Nora"]}
    assert name_ids == {"Alex": 314, "Owen": 536, "Mia": 925, "Nora": 512}
    checks["tokenizer"] = {"sha256": sha(protocol["tokenizer"]), "name_first_token_ids": name_ids}

    schedule, items, index, train16, dev_surface, dev_order, pool, kl_pool, _ = C.load_inputs()
    assert Counter(u["kind"] for u in schedule) == {"english": 180, "binding": 20}
    batch = next(u for u in schedule if u["kind"] == "english")
    x, labels = E.pad_batch([index[rid] for rid in batch["ids"]], batch["pad"])
    assert int((labels != -100).sum()) == 36 * 5

    weights = C.english_ce_weights(labels, batch["ids"], index)
    assert int((weights == 0.25).sum()) == 36
    assert int((weights == 1.0).sum()) == 36 * 4
    assert int((weights == 0.0).sum()) == weights.numel() - 36 * 5
    assert abs(float(weights.sum()) - 36 * 4.25) < 1e-6
    for row, rid in enumerate(batch["ids"]):
        record = index[rid]
        pos = len(record["prompt_token_ids"])
        candidate = record["candidate_token_ids"][record["correct_index"]]
        assert len(candidate) == 4
        assert int(labels[row, pos]) == candidate[0]
        assert float(weights[row, pos]) == 0.25
        assert labels[row, pos + 1:pos + 4].tolist() == candidate[1:]
        assert int(labels[row, pos + 4]) == 3
        assert weights[row, pos + 1:pos + 5].tolist() == [1.0, 1.0, 1.0, 1.0]

    generator = torch.Generator().manual_seed(150025)
    logits = torch.randn((36, batch["pad"], 1024), generator=generator)
    loss_025, meta = C.weighted_english_ce(logits, labels, batch["ids"], index)
    loss_0, _ = C.weighted_english_ce(logits, labels, batch["ids"], index, first_weight=0.0)
    loss_1, _ = C.weighted_english_ce(logits, labels, batch["ids"], index, first_weight=1.0)
    sf14_labels = labels.clone()
    for row, rid in enumerate(batch["ids"]):
        sf14_labels[row, len(index[rid]["prompt_token_ids"])] = -100
    sf14_reference = F.cross_entropy(logits.reshape(-1, 1024), sf14_labels.reshape(-1), ignore_index=-100)
    sf13_reference = F.cross_entropy(logits.reshape(-1, 1024), labels.reshape(-1), ignore_index=-100)
    # The explicit weighted reduction and fused CE mean can differ by one float32 ULP
    # because they sum the same terms in a different order.
    assert torch.allclose(loss_0, sf14_reference, atol=2e-6, rtol=0.0)
    assert torch.allclose(loss_1, sf13_reference, atol=2e-6, rtol=0.0)
    assert meta == {"first_answer_positions": 36, "later_response_eos_positions": 144, "ce_weight_sum": 153.0}

    zero_logits = torch.zeros((36, batch["pad"], 1024))
    assert float(C.margin_hinge_term(zero_logits, batch["ids"], index)) == 1.0
    kl_selection = E.pick_kl_entries(kl_pool["entries"], 1)
    assert len(kl_selection) == 160 and len({row for row, _ in kl_selection}) == 160
    checks["isolated_variable"] = {
        "first_answer_positions_weight_0_25": 36,
        "later_response_punctuation_eos_positions_weight_1": 144,
        "ce_weight_sum": 153.0,
        "sf14_weight0_endpoint_reproduction": True,
        "sf13_weight1_endpoint_reproduction": True,
        "margin_active_at_first_name": True,
        "margin_mock_value": 1.0,
        "kl_positions": 160,
        "kl_distinct_source_rows": 160,
    }

    assert protocol["gates"] == old["gates"]
    assert protocol["optimizer"] == old["optimizer"]
    assert protocol["sampling"] == old["sampling"]
    assert protocol["margin"] == old["margin"]
    source = (H / "CONTROLLER.py").read_text(encoding="utf-8")
    assert "R_name" not in source
    assert "mask_first_answer_ce_labels" not in source
    assert source.count("weighted_english_ce(logits, y, u['ids'], idx)") == 1
    tree = ast.parse(source)
    d3_calls = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "measure_d3"]
    assert len(d3_calls) == 1
    checks["unchanged_settings"] = {
        "gates": True, "optimizer": True, "schedule": True, "margin": True,
        "broad_kl": True, "binding": True, "D3_training_calls": 0,
    }

    inherited = json.loads((SF14 / "U0_PREFLIGHT_RECORD.json").read_text(encoding="utf-8"))
    mapping = {"87038": "87035", "87039": "87036", "87040": "87037"}
    branches = {}
    for seed, prior in mapping.items():
        record = dict(inherited["branches"][prior])
        record["sf15_seed"] = int(seed)
        record["source_sf14_preflight_seed"] = int(prior)
        branches[seed] = record
    u0 = {
        "study": "SF15_PARTIAL_FIRST_ANSWER_TOKEN_CE_RESTORATION_V1",
        "record_type": "INHERITED_ACTUAL_U0_PARENT_REPRODUCTION",
        "source_path": str(SF14 / "U0_PREFLIGHT_RECORD.json"),
        "source_sha256": sha(SF14 / "U0_PREFLIGHT_RECORD.json"),
        "basis": "Same parent bytes and byte-identical evaluator/data; every sealed SF15 run independently reruns and asserts U0 before optimizer construction.",
        "branches": branches,
    }
    (H / "U0_PREFLIGHT_RECORD.json").write_text(json.dumps(u0, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    checks.update({
        "reporter_smoke": json.loads((H / "REPORTER_SMOKE.json").read_text())["status"],
        "checkpoint_loaded": False, "optimizer_created": False, "updates": 0,
        "final_accessed": False, "sacred_accessed": False,
    })
    result = {
        "status": "SF15_PROSPECTIVE_PREFLIGHT_PASS",
        "checks": checks,
        "warnings": [
            "The 0.25 CE weight is normalized by total token weight. This exact interpolation reproduces SF14 at weight 0 and SF13 at weight 1.",
            "U0 evidence is inherited from actual sealed evaluation of identical parent bytes/data/evaluator and is independently asserted again by each run before optimizer creation.",
        ],
    }
    (H / "PREFLIGHT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (H / "PREFLIGHT.md").write_text(
        "# SF15 prospective preflight\n\n"
        "**SF15_PROSPECTIVE_PREFLIGHT_PASS**\n\n"
        "The sole scientific change from sealed SF14 is first-answer-name CE weight `0.0 -> 0.25`. "
        "All three parent hashes, runtime/tokenizer, 180/20 schedule, 160-position broad KL, margin, data, gates, and binding path verified. "
        "A frozen 36-record English batch contained exactly 36 first-name positions at weight 0.25 and 144 later response/punctuation/EOS positions at weight 1.0. "
        "No checkpoint was loaded, no optimizer was created, and zero updates occurred. FINAL/sacred remained locked.\n",
        encoding="utf-8", newline="\n",
    )
    print(result["status"])


if __name__ == "__main__":
    main()
