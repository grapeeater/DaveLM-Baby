"""Run the sealed HR-3 nonsacred evaluator on the fixed comparison set."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer


ROOT = Path(r"C:\DaveLM-CADAVER")
BUNDLE = ROOT / "human_readiness_hr3_block3_causal_seed87006_v7"
RUN = ROOT / "human_readiness_hr3_causal_seed87006_execution_v7"
HR1 = ROOT / "human_readiness_hr1_causal_aligned_seed87006_execution" / "run_factual" / "checkpoint_500.pt"
PILOT1 = ROOT / "language_pilot_1_early_block_protection_seed8380" / "pilot_run" / "checkpoints" / "seed_8380" / "latest.pt"


def load_eval():
    source = BUNDLE / "sources" / "EVALUATE_HR3.py"
    spec = importlib.util.spec_from_file_location("hr3_eval_comparison", source)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.path.insert(0, str(source.parent))
    spec.loader.exec_module(module)
    return module


def main() -> None:
    evaluator = load_eval()
    evaluator.verify_integrity(BUNDLE)
    protocol = evaluator.read_json(BUNDLE / "HR3_PROTOCOL.json")
    evaluator.configure_runtime(protocol["training"]["seed"])
    device = torch.device("cuda")
    tok = Tokenizer.from_file(protocol["tokenizer_path"])
    pinned = evaluator.pinned_binding(BUNDLE)
    checkpoints = [
        ("Pilot1_parent", PILOT1),
        ("HR1_causal_aligned_500", HR1),
        ("HR3_block3_update_100", RUN / "checkpoint_100.pt"),
        ("HR3_block3_update_250", RUN / "checkpoint_250.pt"),
        ("HR3_block3_update_500", RUN / "checkpoint_500.pt"),
    ]
    out = {
        "status": "HR3_NONSACRED_DEV_COMPARISON_COMPLETE",
        "bundle": str(BUNDLE),
        "bundle_receipt_sha256": evaluator.sha(BUNDLE / "FREEZE_RECEIPT.json"),
        "final_readiness_accessed": False,
        "sacred_accessed": False,
        "checkpoints": [
            evaluator.evaluate_one(name, path, BUNDLE, protocol, tok, device, pinned)
            for name, path in checkpoints
        ],
    }
    # Relative parameter drift is grouped from state tensors; this is an
    # audit diagnostic and does not influence checkpoint selection.
    states = {}
    for name, path in checkpoints:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        states[name] = raw["model_state_dict"]
    base = states["Pilot1_parent"]
    groups = {"embeddings_positions": [], "block0_2": [], "block3_7": [], "final_norm": [], "language_head": [], "localizer_retrieval": []}
    for key, value in states["HR3_block3_update_500"].items():
        if key not in base or not torch.is_floating_point(value):
            continue
        if key.startswith("base_model.token_embedding") or key.startswith("base_model.position"):
            group = "embeddings_positions"
        elif key.startswith("base_model.blocks."):
            index = int(key.split(".")[2])
            group = "block0_2" if index < 3 else "block3_7"
        elif key.startswith("base_model.final_norm"):
            group = "final_norm"
        elif key.startswith("base_model.language_head"):
            group = "language_head"
        else:
            group = "localizer_retrieval"
        groups[group].append((key, value, base[key]))
    drift = {}
    for group, entries in groups.items():
        before = sum(float(old.float().norm()) ** 2 for _, _, old in entries) ** 0.5
        delta = sum(float((new.float() - old.float()).norm()) ** 2 for _, new, old in entries) ** 0.5
        drift[group] = {"parameter_tensors": len(entries), "relative_l2_delta_percent": (100.0 * delta / before if before else None)}
    out["pilot1_to_hr3_update500_parameter_drift"] = drift
    evaluator.atomic_json(out, RUN / "NONSACRED_DEV_COMPARISON.json")
    print(json.dumps({"status": out["status"], "checkpoints": [x["name"] for x in out["checkpoints"]], "drift": drift}, indent=2))


if __name__ == "__main__":
    main()
