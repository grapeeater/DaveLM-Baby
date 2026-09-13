"""Complete bounded SF20 v2 preflight. No checkpoint load, optimizer, or inference."""

from __future__ import annotations

import ast
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import torch
import tokenizers
from tokenizers import Tokenizer

import BUILD as B
import CONTROLLER as C
import SF2_ENGINE as E


H = Path(__file__).resolve().parent
SF13 = H.parent / "sf13_broad_coverage_kl_retention_v1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    checks = {}
    protocol = C.verify(sealed=False)
    assert protocol["study"] == "SF20_IDENTITY_HELDOUT_BALANCED_ENTITY_ROTATION_V2"

    # Exact inherited artifacts and authoritative identities.
    identical = [
        "D3_SELECTION.json", "DEV_ORDER.json", "DEV_SURFACE.json", "KL_POOL_MANIFEST.json",
        "KL_POOL.json", "SF13_KL_SCHEDULE.json", "SF2_ENGINE.py", "SF2_PROTOCOL.json",
        "TRAIN16_RETENTION.json",
    ]
    assert all(sha(H / name) == sha(SF13 / name) for name in identical)
    checks["byte_identical_sf13_payload"] = identical
    assert sha(protocol["tokenizer"]) == protocol["tokenizer_sha256"] == B.TOKENIZER_SHA256
    for run in protocol["runs"]:
        assert sha(Path(run["parent_checkpoint"])) == run["parent_checkpoint_sha256"]
    checks["parents"] = {str(run["seed"]): run["parent_checkpoint_sha256"] for run in protocol["runs"]}
    assert protocol["seeds"] == [87053, 87054, 87055]

    # Runtime compatibility without model loading.
    assert platform.python_version() == protocol["runtime"]["python"]
    assert torch.__version__ == protocol["runtime"]["torch"]
    assert tokenizers.__version__ == protocol["runtime"]["tokenizers"]
    assert torch.cuda.is_available()
    checks["runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
                         "tokenizers": tokenizers.__version__, "gpu": torch.cuda.get_device_name(0)}

    tokenizer = Tokenizer.from_file(protocol["tokenizer"])
    names = B.read(H / "NAME_CENSUS.json")["selected"]
    assignment = B.read(H / "IDENTITY_ASSIGNMENT.json")
    token_match = B.read(H / "TOKENIZATION_MATCH.json")
    assert names == assignment["selected_names"] == protocol["selected_names"]
    assert len(names) == len(set(names)) == 16 and not set(names) & set(B.ORIGINAL_NAMES)
    assert all(len(tokenizer.encode(name).ids) == 3 for name in names)
    assert all(len(tokenizer.encode(" " + name).ids) == 3 for name in names)
    assert all(len(tokenizer.encode(" " + name + ".").ids) == 4 for name in names)
    assert all(tokenizer.decode(tokenizer.encode(name).ids) == name for name in names)
    assert all(tokenizer.decode(tokenizer.encode(" " + name).ids) == " " + name for name in names)
    assert all(tokenizer.decode(tokenizer.encode(" " + name + ".").ids) == " " + name + "." for name in names)
    original_first = {tokenizer.encode(" " + name + ".").ids[0] for name in B.ORIGINAL_NAMES}
    assert not original_first & {tokenizer.encode(" " + name + ".").ids[0] for name in names}
    checks["names"] = {"count": 16, "selected": names, "geometry": "3/3/4",
                       "original_first_token_collision": False}

    # Exact identity-only data transformation.
    source = {row["id"]: row for row in B.read(SF13 / "TRAIN.json")}
    items = B.read(H / "TRAIN.json")
    idx = {row["id"]: row for row in items}
    assert len(items) == len(idx) == 48
    preservation = [row for row in items if row["id"].startswith("TRAIN:")]
    assert json.dumps(preservation, sort_keys=True) == json.dumps(
        [row for row in B.read(SF13 / "TRAIN.json") if row["id"].startswith("TRAIN:")], sort_keys=True)
    assert sha(H / "TRAIN16_RETENTION.json") == sha(SF13 / "TRAIN16_RETENTION.json")
    assert token_match["all_surrounding_tokenization_unchanged"]
    assert token_match["all_prompt_lengths_unchanged"]
    assert token_match["all_first_answer_tokens_distinct_within_item"]
    assert len(token_match["records"]) == 32
    for row in items:
        if not row["id"].startswith("SF20:"):
            continue
        old = source[row["source_id"]]
        old_pair = [candidate.strip()[:-1] for candidate in old["candidates"]]
        new_pair = [candidate.strip()[:-1] for candidate in row["candidates"]]
        assert B.skeleton(old["prompt"], old_pair, tokenizer) == B.skeleton(row["prompt"], new_pair, tokenizer)
        assert len(old["prompt_token_ids"]) == len(row["prompt_token_ids"])
        assert row["correct_index"] == old["correct_index"]
        for key in ["subgroup", "arm", "assignment", "object", "predicate", "shard"]:
            assert row.get(key) == old.get(key), (row["id"], key)
    assert sha(H / "DEV_SURFACE.json") == sha(SF13 / "DEV_SURFACE.json")
    assert sha(H / "DEV_ORDER.json") == sha(SF13 / "DEV_ORDER.json")
    checks["identity_only_transformation"] = {"preservation_items_byte_equivalent": 16,
                                               "widening_items_template_equivalent": 32,
                                               "dev_panels_byte_identical": True}

    # Exact balance proof.
    balance = assignment["balance"]
    assert all(balance["candidate_appearances"][name] == 4 for name in names)
    assert all(balance["correct"][name] == 2 and balance["distractor"][name] == 2 for name in names)
    for stratum in ["Surface", "Order"]:
        assert all(balance["per_stratum"][stratum]["candidate"][name] == 2 for name in names)
        assert all(balance["per_stratum"][stratum]["correct"][name] == 1 for name in names)
        assert all(balance["per_stratum"][stratum]["distractor"][name] == 1 for name in names)
    records = assignment["records"]
    assert Counter(row["stratum"] for row in records) == {"Surface": 16, "Order": 16}
    assert Counter(row["correct_index"] for row in records) == {0: 16, 1: 16}
    assert Counter(row["subgroup"] for row in records) == {
        "cloze": 4, "active_qa": 4, "passive_cloze": 4, "passive_qa": 4,
        "order0:fact": 4, "order0:copy": 4, "order1:fact": 4, "order1:copy": 4,
    }
    checks["balance"] = {"candidate_appearances_each": 4, "correct_each": 2, "distractor_each": 2,
                         "surface_correct_each": 1, "surface_distractor_each": 1,
                         "order_correct_each": 1, "order_distractor_each": 1,
                         "correct_index": {"0": 16, "1": 16}}

    # Schedule, padding, labels, and objective isolation.
    schedule, loaded_items, loaded_idx, train16, dev_surface, dev_order, pool, kl, _ = C.load_inputs()
    assert loaded_items == items and loaded_idx == idx
    assert Counter(update["kind"] for update in schedule) == {"english": 180, "binding": 20}
    source_schedule = B.read(SF13 / "SCHEDULE.json")
    reverse = {row["item_id"]: row["source_id"] for row in records}
    for old_update, update in zip(source_schedule, schedule):
        assert update["update"] == old_update["update"] and update["kind"] == old_update["kind"]
        assert update["pad"] == old_update["pad"]
        if update["kind"] == "english":
            assert [reverse.get(item_id, item_id) for item_id in update["ids"]] == old_update["ids"]
    max_length = 0
    for update in schedule:
        if update["kind"] != "english":
            continue
        batch = [idx[item_id] for item_id in update["ids"]]
        x, y = E.pad_batch(batch, update["pad"])
        assert x.shape == y.shape == (36, update["pad"])
        assert int((y != -100).sum()) == 36 * 5
        for k, row in enumerate(batch):
            labels = y[k][y[k] != -100].tolist()
            # PINNED_MASKING.prepare_example appends the authoritative EOS token id 3.
            expected = row["candidate_token_ids"][row["correct_index"]] + [3]
            assert labels == expected
        max_length = max(max_length, update["pad"])
    assert max_length <= 256
    assert len(E.pick_kl_entries(kl["entries"], 1)) == 160
    controller_source = (H / "CONTROLLER.py").read_text(encoding="utf-8")
    expected_controller = (SF13 / "CONTROLLER.py").read_text(encoding="utf-8").replace("SF13", "SF20")
    expected_controller = expected_controller.replace(
        "broad-coverage KL retention at fixed retention compute (sampling geometry change)",
        "identity-heldout balanced entity rotation with SF13 objective")
    expected_controller = expected_controller.replace(
        "ONE scientific variable vs SF11/SF12: retention position coverage / sampling geometry\n(contiguous sequential 160-entry slice -> frozen row-distributed 160-entry schedule).\nAll loss terms, curriculum, optimizer, scope, gates and persistence are unchanged from SF11.",
        "ONE scientific variable vs SF13: widening identity set. Full CE, broad KL160, margin,\noptimizer, scope, schedule, gates, binding, evaluation and persistence remain unchanged."
    )
    assert controller_source == expected_controller
    ast.parse(controller_source)
    assert "mask_first_answer_ce_labels" not in controller_source
    assert "candidate_membership_hinge" not in controller_source
    assert "R_name" not in controller_source and "autograd.grad" not in controller_source
    assert controller_source.count("F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1), ignore_index=-100)") == 1
    assert C.LAMBDA_MARGIN == 0.25 and C.MARGIN_M == 1.0 and E.LAMBDA_KL == 1.0
    checks["schedule_and_objective"] = {"updates": 200, "english": 180, "binding": 20,
                                         "batch": 36, "supervised_tokens_per_english_batch": 180,
                                         "full_first_answer_ce": True, "membership_hinge": False,
                                         "lambda_margin": 0.25, "M": 1.0, "KL_positions": 160,
                                         "lambda_KL": 1.0, "max_pad": max_length,
                                         "controller_exact_sf13_mechanical_derivative": True}

    # Gates/evaluator and contamination fences are inherited exactly.
    old_protocol = B.read(SF13 / "PROTOCOL.json")
    assert protocol["gates"] == old_protocol["gates"]
    assert protocol["optimizer"] == old_protocol["optimizer"]
    assert protocol["scope"] == old_protocol["scope"]
    assert protocol["binding"] == old_protocol["binding"]
    assert protocol["evaluation"] == old_protocol["evaluation"]
    assert not (H / "runs").exists() and not (H / "RUN_LEDGER.json").exists()
    checks["fences"] = {"gates_unchanged": True, "evaluator_unchanged": True,
                         "D3_training_calls": 0, "historical_transfer": "LOCKED_UNSCORED",
                         "final_or_sacred": "PROHIBITED", "run_directory_exists": False}
    checks.update({"checkpoint_loaded": False, "optimizer_created": False, "updates": 0,
                   "treatment_outcomes_scored": False, "final_accessed": False, "sacred_accessed": False})
    output = {"status": "SF20_STATIC_PREFLIGHT_PASS", "checks": checks, "warnings": [
        "Name selection excludes the four original first-answer token IDs so the D3 token classes receive no direct widening target pressure.",
        "TinyStories occurrence frequency is a lexical eligibility/ranking input only; no model logits or evaluation outcomes were used.",
        "Optional new-name probability telemetry was omitted.",
    ]}
    (H / "STATIC_PREFLIGHT.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8", newline="\n")
    print(output["status"])


if __name__ == "__main__":
    main()
