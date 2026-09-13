from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from baby_vnext import BabyVNextConfig, BabyVNextWithBinding, BindingConfig
from baby_vnext.binding import legacy_t13_layout
from baby_vnext.checkpoint import (
    _state_digest,
    load_initialized_checkpoint,
    save_initialized_checkpoint,
)
from baby_vnext.parameter_count import analytical_counts, physical_counts


ROOT = Path(__file__).resolve().parent
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
EXPECTED_TOKENIZER_SHA256 = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
INITIALIZATION_SEED = 600001

AUTHORITATIVE_INPUTS = {
    "research_model_source": Path(r"C:\DaveLM-v0.9\v0_7\model.py"),
    "research_config_source": Path(r"C:\DaveLM-v0.9\v0_8\config.py"),
    "t13_model_source": Path(r"C:\DaveLM-CADAVER\treatment13_model.py"),
    "pinned_pilot1_binding": Path(
        r"C:\DaveLM-CADAVER\fact_supervision_87001_eval_v1\PINNED_PILOT1_BINDING_IMPLEMENTATION.py"
    ),
    "pilot1_runner": Path(
        r"C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\run.py"
    ),
    "pilot1_language_train": Path(
        r"C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\language_train.jsonl"
    ),
    "pilot1_language_dev": Path(
        r"C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\language_dev.jsonl"
    ),
    "pilot1_checkpoint": Path(
        r"C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt"
    ),
    "t13_graduate_checkpoint": Path(
        r"C:\DaveLM-CADAVER\treatment13_orthogonal_shared_unbounded_seed8380\checkpoints\orthogonal_shared_unbounded\seed_8380\latest.pt"
    ),
    "vnext_audit_facts": Path(
        r"C:\DaveLM-CADAVER\vnext_60m_audit\VNEXT_AUDIT_FACTS.json"
    ),
}

EXPECTED_CHECKPOINT_HASHES = {
    "pilot1_checkpoint": "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb",
    "t13_graduate_checkpoint": "fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def candidate(
    name: str, *, d_model: int, n_layers: int, n_heads: int, d_mlp: int, retrieval_dim: int
) -> dict[str, object]:
    config = BabyVNextConfig(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_mlp=d_mlp,
        binding=BindingConfig(retrieval_dim=retrieval_dim),
    )
    config.validate()
    return {"name": name, "config": config.to_dict(), "counts": analytical_counts(config)}


def main() -> None:
    started = time.time()
    before = {name: file_record(path) for name, path in AUTHORITATIVE_INPUTS.items()}
    if before["pilot1_checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_HASHES["pilot1_checkpoint"]:
        raise RuntimeError("Pilot1 checkpoint identity mismatch")
    if before["t13_graduate_checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_HASHES["t13_graduate_checkpoint"]:
        raise RuntimeError("T13 graduate checkpoint identity mismatch")
    if sha256(TOKENIZER) != EXPECTED_TOKENIZER_SHA256:
        raise RuntimeError("tokenizer identity mismatch")

    candidates = [
        candidate(
            "balanced_12x640_recommended",
            d_model=640,
            n_layers=12,
            n_heads=10,
            d_mlp=2560,
            retrieval_dim=128,
        ),
        candidate(
            "deeper_narrower_16x560",
            d_model=560,
            n_layers=16,
            n_heads=10,
            d_mlp=2240,
            retrieval_dim=112,
        ),
        candidate(
            "wider_shallower_10x704",
            d_model=704,
            n_layers=10,
            n_heads=11,
            d_mlp=2816,
            retrieval_dim=128,
        ),
    ]
    (ROOT / "ARCHITECTURE_CANDIDATES.json").write_text(
        json.dumps(candidates, indent=2) + "\n", encoding="utf-8"
    )

    config = BabyVNextConfig.from_dict(candidates[0]["config"])
    config.save(ROOT / "BABY_VNEXT_CONFIG.json")
    torch.manual_seed(INITIALIZATION_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(INITIALIZATION_SEED)
    torch.use_deterministic_algorithms(True)
    model = BabyVNextWithBinding(config)
    analytical = analytical_counts(config)
    physical = physical_counts(model)
    if analytical["base_total"] != physical["base_total"]:
        raise RuntimeError("analytical and physical base parameter counts differ")
    if analytical["binding_total"] != physical["binding_total"]:
        raise RuntimeError("analytical and physical binding parameter counts differ")
    if analytical["trained_total"] != physical["trained_total"]:
        raise RuntimeError("analytical and physical total parameter counts differ")
    counts = {"analytical": analytical, "physical": physical}
    (ROOT / "PARAMETER_COUNTS.json").write_text(
        json.dumps(counts, indent=2) + "\n", encoding="utf-8"
    )

    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    tokenizer_checks = {
        "sha256": sha256(TOKENIZER),
        "vocab_size": tokenizer.get_vocab_size(),
        "special_token_ids": {
            token: tokenizer.token_to_id(token)
            for token in ("<pad>", "<unk>", "<bos>", "<eos>", "<doc>")
        },
        "config_vocab_matches": tokenizer.get_vocab_size() == config.vocab_size,
        "input_output_weights_tied": (
            model.base_model.token_embedding.weight.data_ptr()
            == model.base_model.language_head.weight.data_ptr()
        ),
    }
    if not tokenizer_checks["config_vocab_matches"]:
        raise RuntimeError("tokenizer and model vocabulary sizes differ")
    if tokenizer_checks["input_output_weights_tied"]:
        raise RuntimeError("candidate unexpectedly ties input and output weights")

    tests: dict[str, object] = {}
    expected_shapes = {
        "token_embedding": [1024, 640],
        "position_embedding": [256, 640],
        "block0_qkv": [1920, 640],
        "block0_attention_projection": [640, 640],
        "block0_mlp_in": [2560, 640],
        "block0_mlp_out": [640, 2560],
        "final_norm": [640],
        "language_head": [1024, 640],
        "localizer_u": [640],
        "localizer_q": [639],
        "retrieval_wq": [128, 640],
        "retrieval_wv": [640, 640],
    }
    actual_shapes = {
        "token_embedding": list(model.base_model.token_embedding.weight.shape),
        "position_embedding": list(model.base_model.position_embedding.weight.shape),
        "block0_qkv": list(model.base_model.blocks[0].attention.qkv.weight.shape),
        "block0_attention_projection": list(
            model.base_model.blocks[0].attention.projection.weight.shape
        ),
        "block0_mlp_in": list(model.base_model.blocks[0].feed_forward.network[0].weight.shape),
        "block0_mlp_out": list(model.base_model.blocks[0].feed_forward.network[2].weight.shape),
        "final_norm": list(model.base_model.final_norm.weight.shape),
        "language_head": list(model.base_model.language_head.weight.shape),
        "localizer_u": list(model.localizer.u.shape),
        "localizer_q": list(model.localizer.q.shape),
        "retrieval_wq": list(model.wq.weight.shape),
        "retrieval_wv": list(model.wv.weight.shape),
    }
    tests["shape_validation"] = {
        "pass": actual_shapes == expected_shapes,
        "expected": expected_shapes,
        "actual": actual_shapes,
    }
    if actual_shapes != expected_shapes:
        raise RuntimeError("shape validation failed")

    model.eval()
    torch.manual_seed(INITIALIZATION_SEED + 1)
    first = torch.randint(0, config.vocab_size, (1, 16))
    second = first.clone()
    second[:, 8:] = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        first_logits, _ = model(first)
        second_logits, _ = model(second)
    causal_max_difference = float((first_logits[:, :8] - second_logits[:, :8]).abs().max())
    tests["causal_mask"] = {
        "pass": causal_max_difference <= 1e-6,
        "prefix_max_abs_difference": causal_max_difference,
    }
    if not tests["causal_mask"]["pass"]:
        raise RuntimeError("causal masking failed")

    with torch.no_grad():
        model.base_model.set_attention_backend("sdpa")
        sdpa_logits, _ = model(first)
        model.base_model.set_attention_backend("reference")
        reference_logits, _ = model(first)
        model.base_model.set_attention_backend("sdpa")
    backend_difference = float((sdpa_logits - reference_logits).abs().max())
    tests["sdpa_reference_equivalence"] = {
        "pass": backend_difference <= 2e-5,
        "max_abs_difference": backend_difference,
        "tolerance": 2e-5,
    }
    if not tests["sdpa_reference_equivalence"]["pass"]:
        raise RuntimeError("SDPA/reference equivalence failed")

    checkpoint_path = ROOT / "validation" / "initialized_checkpoint_roundtrip.pt"
    digest_before_backward = _state_digest(model)
    saved = save_initialized_checkpoint(
        checkpoint_path,
        model,
        tokenizer_sha256=EXPECTED_TOKENIZER_SHA256,
        initialization_seed=INITIALIZATION_SEED,
    )
    loaded_model, loaded_payload = load_initialized_checkpoint(
        checkpoint_path, expected_tokenizer_sha256=EXPECTED_TOKENIZER_SHA256
    )
    tests["config_roundtrip"] = {
        "pass": loaded_model.config == config and loaded_payload["config_sha256"] == config.sha256(),
        "config_sha256": config.sha256(),
    }
    tests["checkpoint_roundtrip"] = {
        "pass": _state_digest(loaded_model) == digest_before_backward,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "model_state_sha256": saved["model_state_sha256"],
        "optimizer_updates": saved["optimizer_updates"],
    }
    if not tests["config_roundtrip"]["pass"] or not tests["checkpoint_roundtrip"]["pass"]:
        raise RuntimeError("configuration/checkpoint round-trip failed")
    del loaded_model, loaded_payload, saved

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    torch.manual_seed(INITIALIZATION_SEED + 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(INITIALIZATION_SEED + 2)
    input_ids = torch.randint(0, config.vocab_size, (2, 16), device=device)
    language_logits, _ = model(input_ids)
    language_loss = F.cross_entropy(
        language_logits[:, :-1].reshape(-1, config.vocab_size), input_ids[:, 1:].reshape(-1)
    )
    model.zero_grad(set_to_none=True)
    language_loss.backward()
    language_gradient_checks = {
        "token_embedding": model.base_model.token_embedding.weight.grad is not None,
        "position_embedding": model.base_model.position_embedding.weight.grad is not None,
        "all_attention_blocks": all(
            block.attention.qkv.weight.grad is not None for block in model.base_model.blocks
        ),
        "all_mlp_blocks": all(
            block.feed_forward.network[0].weight.grad is not None
            for block in model.base_model.blocks
        ),
        "final_norm": model.base_model.final_norm.weight.grad is not None,
        "language_head": model.base_model.language_head.weight.grad is not None,
    }
    tests["synthetic_language_backward"] = {
        "pass": all(language_gradient_checks.values()),
        "loss": float(language_loss.detach().cpu()),
        "gradient_checks": language_gradient_checks,
        "optimizer_created": False,
        "optimizer_step": False,
    }
    if not tests["synthetic_language_backward"]["pass"]:
        raise RuntimeError("language gradient validation failed")

    model.zero_grad(set_to_none=True)
    query_positions = torch.tensor([10, 11], device=device)
    answer_positions = torch.tensor([14, 15], device=device)
    layout = legacy_t13_layout(query_positions, answer_positions, 16)
    binding_logits, extras = model(input_ids, layout)
    answer_targets = torch.tensor([7, 11], device=device)
    rows = torch.arange(2, device=device)
    binding_loss = F.cross_entropy(binding_logits[rows, answer_positions], answer_targets)
    binding_loss.backward()
    binding_gradient_checks = {
        "localizer_u": model.localizer.u.grad is not None,
        "localizer_q": model.localizer.q.grad is not None,
        "localizer_biases": model.localizer.bs.grad is not None and model.localizer.ba.grad is not None,
        "wq": model.wq.weight.grad is not None,
        "wk": model.wk.weight.grad is not None,
        "wv": model.wv.weight.grad is not None,
        "wo": model.wo.weight.grad is not None,
        "base_model": model.base_model.blocks[0].attention.qkv.weight.grad is not None,
        "two_slots": list(extras["localization_attention"].shape) == [2, 11, 2],
        "explicit_layout": layout.key_positions.shape == layout.value_positions.shape,
    }
    tests["synthetic_binding_backward"] = {
        "pass": all(binding_gradient_checks.values()),
        "loss": float(binding_loss.detach().cpu()),
        "gradient_checks": binding_gradient_checks,
        "optimizer_created": False,
        "optimizer_step": False,
    }
    if not tests["synthetic_binding_backward"]["pass"]:
        raise RuntimeError("binding interface/gradient validation failed")
    model.zero_grad(set_to_none=True)
    model.to("cpu")
    digest_after_backward = _state_digest(model)
    tests["weights_unchanged_by_backward"] = {
        "pass": digest_after_backward == digest_before_backward,
        "before": digest_before_backward,
        "after": digest_after_backward,
    }
    if not tests["weights_unchanged_by_backward"]["pass"]:
        raise RuntimeError("weights changed despite no optimizer step")

    bytes_per_parameter = {
        "weights_fp32": 4,
        "gradients_fp32": 4,
        "adam_first_moment_fp32": 4,
        "adam_second_moment_fp32": 4,
    }
    parameter_bytes = analytical["trained_total"] * 4
    memory = {
        "trained_parameters": analytical["trained_total"],
        "weights_fp32_bytes": parameter_bytes,
        "gradients_fp32_bytes": parameter_bytes,
        "adam_moments_fp32_bytes": parameter_bytes * 2,
        "persistent_training_state_bytes": parameter_bytes * 4,
        "persistent_training_state_mib": parameter_bytes * 4 / (1024**2),
        "additional_no_grad_teacher_weights_mib": parameter_bytes / (1024**2),
        "persistent_with_teacher_mib": parameter_bytes * 5 / (1024**2),
        "activation_note": "Input dependent; validate microbatch 8 first, then 16. SDPA avoids materializing the Python per-head mask path.",
        "checkpoint_model_only_mib": checkpoint_path.stat().st_size / (1024**2),
        "precision_policy": "float32 baseline; mixed precision deferred to a separately validated training preflight",
        "accounting": bytes_per_parameter,
    }
    (ROOT / "MEMORY_ESTIMATE.json").write_text(
        json.dumps(memory, indent=2) + "\n", encoding="utf-8"
    )

    after = {name: file_record(path) for name, path in AUTHORITATIVE_INPUTS.items()}
    historical_unchanged = all(
        before[name]["sha256"] == after[name]["sha256"]
        and before[name]["size_bytes"] == after[name]["size_bytes"]
        and before[name]["mtime_ns"] == after[name]["mtime_ns"]
        for name in before
    )
    tests["historical_artifacts_unchanged"] = {"pass": historical_unchanged}
    if not historical_unchanged:
        raise RuntimeError("an authoritative historical input changed during validation")

    if torch.cuda.is_available():
        device_record = {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
        }
    else:
        device_record = {"available": False, "name": None, "count": 0}
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "tokenizers": __import__("tokenizers").__version__,
        "executable": sys.executable,
        "device": device_record,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    tests["gpu_device_smoke"] = {
        "pass": device_record["available"],
        "device": device_record,
        "selected_model_forward_backward": True,
    }
    all_pass = all(bool(value.get("pass")) for value in tests.values())
    validation = {
        "status": "PASS" if all_pass else "FAIL",
        "training_performed": False,
        "real_baby_data_used_for_backward": False,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "locked_or_sacred_material_accessed": False,
        "tests": tests,
        "tokenizer": tokenizer_checks,
        "runtime": runtime,
        "elapsed_seconds": time.time() - started,
    }
    (ROOT / "VALIDATION_RESULTS.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "status": "PASS" if all_pass else "FAIL",
        "authoritative_inputs_before": before,
        "authoritative_inputs_after": after,
        "historical_artifacts_unchanged": historical_unchanged,
        "tokenizer": file_record(TOKENIZER),
        "runtime": runtime,
        "initialization_seed": INITIALIZATION_SEED,
        "forbidden_material_accessed": False,
    }
    (ROOT / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    if not all_pass:
        raise RuntimeError("one or more validation checks failed")
    print(json.dumps({"status": "PASS", "counts": analytical, "tests": tests}, indent=2))


if __name__ == "__main__":
    main()
