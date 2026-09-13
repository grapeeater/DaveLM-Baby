from __future__ import annotations

from array import array
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent
ARCH = Path(r"C:\DaveLM-CADAVER\baby_vnext_60m_design_v1")
PILOT = Path(r"C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380")
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
sys.path.insert(0, str(ARCH))

from baby_vnext import BabyVNextConfig, BabyVNextWithBinding  # noqa: E402
from baby_vnext.checkpoint import save_initialized_checkpoint  # noqa: E402


TOKENIZER_SHA256 = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
TRAIN_SHA256 = "450f78bdb437bde98f9a1ff48364b835df517b7d8e24a4f52256c114fa34c41c"
DEV_SHA256 = "deff4fc7ed18e6e1f0b6f32faccc80f1eb44d0e58f3f38d512ffdfab798d6ac4"
RUN_SEED = 610001
TRAIN_SCHEDULE_SEED = 61000101
EVAL_SELECTION_SEED = 61000102
RESERVED_REPLICATION_SEEDS = [610002, 610003]
MAX_UPDATES = 6000
EFFECTIVE_BATCH = 64
CONTEXT = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_chain(root: Path) -> dict[str, object]:
    detached = (root / "FREEZE_RECEIPT.sha256").read_text(encoding="ascii").split()[0]
    receipt = root / "FREEZE_RECEIPT.json"
    if sha256(receipt) != detached:
        raise RuntimeError("architecture receipt mismatch")
    parsed = json.loads(receipt.read_text(encoding="utf-8"))
    manifest = root / "SHA256SUMS.txt"
    if sha256(manifest) != parsed["manifest_sha256"]:
        raise RuntimeError("architecture manifest mismatch")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if sha256(root / relative) != expected:
            raise RuntimeError(f"architecture payload mismatch: {relative}")
    return {"receipt_sha256": detached, "manifest_sha256": parsed["manifest_sha256"]}


def encode_stream(tokenizer: Tokenizer, rows: list[dict[str, object]]) -> list[int]:
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    document = tokenizer.token_to_id("<doc>")
    if None in (bos, eos, document):
        raise RuntimeError("tokenizer is missing a required stream token")
    stream: list[int] = []
    for index, row in enumerate(rows):
        if index:
            stream.append(int(document))
        stream.append(int(bos))
        stream.extend(tokenizer.encode(str(row["text"])).ids)
        stream.append(int(eos))
    return stream


def write_u16(path: Path, values: list[int]) -> None:
    if min(values) < 0 or max(values) > 65535:
        raise ValueError("u16 stream value out of range")
    packed = array("H", values)
    if sys.byteorder != "little":
        packed.byteswap()
    path.write_bytes(packed.tobytes())


def write_u32(path: Path, values: list[int]) -> None:
    packed = array("I", values)
    if packed.itemsize != 4:
        raise RuntimeError("platform unsigned-int is not 32 bits")
    if sys.byteorder != "little":
        packed.byteswap()
    path.write_bytes(packed.tobytes())


def starts(generator: torch.Generator, count: int, stream_length: int) -> list[int]:
    return torch.randint(
        0, stream_length - (CONTEXT + 1), size=(count,), generator=generator
    ).tolist()


def main() -> None:
    architecture_chain = verify_chain(ARCH)
    if sha256(TOKENIZER) != TOKENIZER_SHA256:
        raise RuntimeError("tokenizer mismatch")
    train_path = PILOT / "language_train.jsonl"
    dev_path = PILOT / "language_dev.jsonl"
    if sha256(train_path) != TRAIN_SHA256 or sha256(dev_path) != DEV_SHA256:
        raise RuntimeError("Pilot1 language corpus mismatch")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    if tokenizer.get_vocab_size() != 1024:
        raise RuntimeError("unexpected vocabulary size")
    train_rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    dev_rows = [json.loads(line) for line in dev_path.read_text(encoding="utf-8").splitlines()]
    train_stream = encode_stream(tokenizer, train_rows)
    dev_stream = encode_stream(tokenizer, dev_rows)
    if len(train_rows) != 9000 or len(train_stream) != 3576861:
        raise RuntimeError("physical Pilot1 train counts differ from the verified design evidence")
    train_stream_path = ROOT / "data" / "LANGUAGE_TRAIN_STREAM.u16"
    dev_stream_path = ROOT / "data" / "LANGUAGE_DEV_STREAM.u16"
    write_u16(train_stream_path, train_stream)
    write_u16(dev_stream_path, dev_stream)

    generator = torch.Generator().manual_seed(TRAIN_SCHEDULE_SEED)
    train_starts: list[int] = []
    for _ in range(MAX_UPDATES):
        train_starts.extend(starts(generator, EFFECTIVE_BATCH, len(train_stream)))
    schedule_path = ROOT / "data" / "TRAIN_WINDOW_STARTS.u32"
    write_u32(schedule_path, train_starts)

    eval_generator = torch.Generator().manual_seed(EVAL_SELECTION_SEED)
    dev_starts = starts(eval_generator, 20 * EFFECTIVE_BATCH, len(dev_stream))
    train_eval_starts = starts(eval_generator, 5 * EFFECTIVE_BATCH, len(train_stream))
    eval_path = ROOT / "data" / "EVAL_WINDOW_STARTS.u32"
    write_u32(eval_path, dev_starts + train_eval_starts)

    architecture_config = BabyVNextConfig.load(ARCH / "BABY_VNEXT_CONFIG.json")
    torch.manual_seed(RUN_SEED)
    torch.cuda.manual_seed_all(RUN_SEED)
    initialized_model = BabyVNextWithBinding(architecture_config)
    init_path = ROOT / "initialization" / f"seed_{RUN_SEED}_initialized.pt"
    init_payload = save_initialized_checkpoint(
        init_path,
        initialized_model,
        tokenizer_sha256=TOKENIZER_SHA256,
        initialization_seed=RUN_SEED,
    )
    if init_payload["optimizer_updates"] != 0:
        raise RuntimeError("initialized checkpoint records an optimizer update")
    del initialized_model, init_payload

    total_positions = MAX_UPDATES * EFFECTIVE_BATCH * CONTEXT
    config = {
        "status": "PROSPECTIVE_READY_TO_TRAIN",
        "study": "BABY_VNEXT_PHASE1_LANGUAGE_V1",
        "scientific_purpose": "ordinary-language acquisition baseline for the 61.52M capacity successor",
        "architecture": {
            "bundle": str(ARCH),
            "receipt_sha256": architecture_chain["receipt_sha256"],
            "manifest_sha256": architecture_chain["manifest_sha256"],
            "config_path": str(ARCH / "BABY_VNEXT_CONFIG.json"),
            "config_sha256": architecture_config.sha256(),
            "trained_parameters": 61520385,
            "base_parameters": 60536064,
            "binding_parameters": 984321,
        },
        "runtime": {
            "python": r"C:\DaveLM-CADAVER\sf2_runtime\sf2venv\Scripts\python.exe",
            "python_version": "3.12.14",
            "torch": "2.12.0+rocm7.14.0",
            "tokenizers": "0.23.1",
            "device": "AMD Radeon RX 9060 XT",
            "deterministic_algorithms": True,
            "tf32": False,
            "autocast": False,
            "precision": "float32",
            "sdpa_expected_backend": "math",
        },
        "seed_policy": {
            "primary_run_seed": RUN_SEED,
            "initialization_seed": RUN_SEED,
            "dropout_and_runtime_seed": RUN_SEED,
            "training_schedule_seed": TRAIN_SCHEDULE_SEED,
            "evaluation_selection_seed": EVAL_SELECTION_SEED,
            "reserved_replication_run_seeds": RESERVED_REPLICATION_SEEDS,
            "replication_status": "reserved_not_authorized",
        },
        "initialization": {
            "strategy": "from_scratch_pytorch_module_defaults_v1",
            "checkpoint": str(init_path),
            "checkpoint_sha256": sha256(init_path),
            "model_state_sha256": None,
            "optimizer_created": False,
            "optimizer_updates": 0,
        },
        "data": {
            "tokenizer_path": str(TOKENIZER),
            "tokenizer_sha256": TOKENIZER_SHA256,
            "train_source": str(train_path),
            "train_source_sha256": TRAIN_SHA256,
            "dev_source": str(dev_path),
            "dev_source_sha256": DEV_SHA256,
            "train_documents": len(train_rows),
            "dev_documents": len(dev_rows),
            "train_stream": str(train_stream_path),
            "train_stream_sha256": sha256(train_stream_path),
            "train_stream_tokens": len(train_stream),
            "dev_stream": str(dev_stream_path),
            "dev_stream_sha256": sha256(dev_stream_path),
            "dev_stream_tokens": len(dev_stream),
            "stream_format": "little-endian unsigned 16-bit token IDs",
            "stream_construction": "<doc> between documents; <bos> + text tokens + <eos> per document",
            "window_sampling": "materialized Pilot1-compatible torch.randint starts over concatenated stream",
            "padding": "none",
            "truncation": "fixed 256-token windows only",
            "train_dev_separate": True,
        },
        "schedule": {
            "path": str(schedule_path),
            "sha256": sha256(schedule_path),
            "format": "little-endian unsigned 32-bit starts, row-major [6000,64]",
            "shape": [MAX_UPDATES, EFFECTIVE_BATCH],
            "regeneration_during_training": False,
            "max_updates": MAX_UPDATES,
            "context": CONTEXT,
            "microbatch": 16,
            "gradient_accumulation": 4,
            "effective_batch": EFFECTIVE_BATCH,
            "supervised_positions_per_update": EFFECTIVE_BATCH * CONTEXT,
            "total_supervised_positions": total_positions,
            "train_stream_equivalents": total_positions / len(train_stream),
        },
        "objective": {
            "type": "full_vocabulary_causal_next_token_cross_entropy",
            "reduction": "mean over each equal-sized microbatch, divided by accumulation steps before backward",
            "supervision": "all 256 next-token positions in every window",
            "binding_path_active": False,
        },
        "parameter_scope": {
            "train": "all base_model parameters: token/position embeddings, blocks 0-11, final norm, language head",
            "freeze": "orthogonal localizer and wq/wk/wv/wo binding projections",
            "trainable_parameters": 60536064,
            "frozen_parameters": 984321,
            "old_block_protection_mapping": "no base blocks protected because Phase1 begins from scratch; Pilot1 protection preserved pre-existing binding, which does not yet exist in vNext",
        },
        "optimizer": {
            "type": "AdamW",
            "lr_peak": 0.0003,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.05,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
            "gradient_clip_norm": 2.0,
        },
        "lr_schedule": {
            "type": "linear_warmup_then_cosine_decay",
            "warmup_updates": 200,
            "warmup_positions": 200 * EFFECTIVE_BATCH * CONTEXT,
            "peak_lr": 0.0003,
            "minimum_lr_ratio": 0.1,
            "minimum_lr": 0.00003,
            "formula": "u<=200: peak*u/200; otherwise peak*(0.1+0.9*0.5*(1+cos(pi*(u-200)/5800)))",
            "schedule_change_classification": "SCALE-JUSTIFIED: from-scratch run is 6.67x longer than Pilot1 continuation; repository v0.8 already specified the same 3,276,800-position warmup and 0.1 minimum ratio",
        },
        "evaluation": {
            "points": list(range(0, MAX_UPDATES + 1, 500)),
            "eval_microbatch": 16,
            "dev_windows": 20 * EFFECTIVE_BATCH,
            "train_windows": 5 * EFFECTIVE_BATCH,
            "selection_path": str(eval_path),
            "selection_sha256": sha256(eval_path),
            "selection_format": "first 1280 dev starts, then 320 train starts; little-endian u32",
            "metrics": ["aligned causal CE", "perplexity", "train-dev CE gap", "fixed greedy diagnostics"],
            "binding": "NOT_APPLICABLE_UNTRAINED; mechanical interface only",
        },
        "gates": {
            "phase1_language_ready": {
                "best_dev_ce_max": 3.0,
                "final_dev_ce_max": 3.1,
                "two_consecutive_dev_evaluations_max": 3.1,
                "selected_checkpoint_train_dev_gap_max": 0.5,
                "all_values_finite": True,
                "completion_updates": MAX_UPDATES,
            },
            "descriptive_oh_fuck_signal": "substantial monotonic DEV CE/PPL fall, two consecutive DEV CE <=3.1, no widening train-dev gap, finite gradients",
            "not_a_gate": "generation diagnostics and binding interface smoke",
        },
        "checkpointing": {
            "rolling_restart_interval": 100,
            "permanent_model_interval": 500,
            "best_definition": "lowest frozen DEV CE; exact ties select earlier update",
            "atomic": True,
            "resume": "restore model, optimizer, completed update, Python/torch/GPU RNG, frozen hashes and metrics; next update is completed_update+1",
            "uncommitted_replay": "updates after the latest 100-update rolling commit are not committed and are deterministically replayed",
        },
        "stopping": {
            "maximum_update": MAX_UPDATES,
            "dev_early_stopping": False,
            "hard_stop": [
                "NaN or Inf in inputs/logits/loss/gradients/gradient norm/model/optimizer",
                "integrity or resume-provenance mismatch",
                "deterministic algorithm failure",
                "GPU OOM",
                "checkpoint commit failure",
            ],
            "no_same_run_adaptation": True,
        },
        "locks": {
            "factual_supervision": False,
            "historical_transfer": "LOCKED_UNSCORED",
            "final": "LOCKED_UNACCESSED",
            "sacred": "LOCKED_UNACCESSED",
        },
    }
    init_raw = torch.load(init_path, map_location="cpu", weights_only=False)
    config["initialization"]["model_state_sha256"] = init_raw["model_state_sha256"]
    config_path = ROOT / "PHASE1_LANGUAGE_CONFIG.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    prompts = {
        "status": "FROZEN_DESCRIPTIVE_ONLY",
        "source": "exact nonsacred Pilot1 diagnostic prompt list",
        "prompts": [
            "Hello",
            "Hello, my name is",
            "What is your name?",
            "The dog is",
            "The girl went to",
            "Yesterday I",
            "I am happy because",
            "Why did the dog run?",
            "Tell me a short story about a dog.",
            "he womputeld"
        ],
        "generation": {"strategy": "greedy", "max_new_tokens": 32, "stop": "normal EOS"},
        "gate": None,
    }
    (ROOT / "data" / "GENERATION_PROMPTS.json").write_text(
        json.dumps(prompts, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "architecture_chain": architecture_chain,
        "architecture_config_sha256": architecture_config.sha256(),
        "tokenizer": {"path": str(TOKENIZER), "sha256": sha256(TOKENIZER)},
        "pilot1_train": {"path": str(train_path), "sha256": sha256(train_path)},
        "pilot1_dev": {"path": str(dev_path), "sha256": sha256(dev_path)},
        "pilot1_training_spec": {
            "path": str(PILOT / "pilot_run" / "TRAINING_SPEC.json"),
            "sha256": sha256(PILOT / "pilot_run" / "TRAINING_SPEC.json"),
        },
        "pilot1_runner": {
            "path": str(PILOT / "run.py"),
            "sha256": sha256(PILOT / "run.py"),
        },
        "pilot1_material_builder": {
            "path": str(PILOT / "prepare.py"),
            "sha256": sha256(PILOT / "prepare.py"),
        },
        "warmup_cosine_reference": {
            "path": r"C:\DaveLM-v0.8\v0_8\token_schedule.py",
            "sha256": "0cb0083016a8f57966190f95170b415ca683f07983801a05e23dfaf95b525745",
        },
        "warmup_budget_reference": {
            "path": r"C:\DaveLM-v0.8\v0_7_1\config.py",
            "sha256": "0ff9dc509b02c2065d12d5b8fb0f742bcd4ff5ab4271fa321121032449de99bd",
        },
        "initialized_checkpoint": {"path": str(init_path), "sha256": sha256(init_path)},
        "optimizer_created": False,
        "optimizer_updates": 0,
        "training_performed": False,
        "forbidden_material_accessed": False,
    }
    (ROOT / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PHASE1_BUILD_ZERO_UPDATE_PASS",
        "train_stream_tokens": len(train_stream),
        "dev_stream_tokens": len(dev_stream),
        "schedule_starts": len(train_starts),
        "total_supervised_positions": total_positions,
        "stream_equivalents": total_positions / len(train_stream),
        "initialized_checkpoint_sha256": sha256(init_path),
        "optimizer_created": False,
        "optimizer_updates": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
