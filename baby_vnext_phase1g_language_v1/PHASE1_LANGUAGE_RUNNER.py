from __future__ import annotations

import argparse
from array import array
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer


BUNDLE_DEFAULT = Path(r"C:\DaveLM-CADAVER\baby_vnext_phase1_language_v1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_integrity_chain(bundle: Path) -> dict[str, Any]:
    detached_path = bundle / "FREEZE_RECEIPT.sha256"
    receipt_path = bundle / "FREEZE_RECEIPT.json"
    manifest_path = bundle / "SHA256SUMS.txt"
    expected_receipt = detached_path.read_text(encoding="ascii").split()[0]
    if sha256(receipt_path) != expected_receipt:
        raise RuntimeError("Phase1 detached receipt hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["status"] != "BABY_VNEXT_PHASE1_LANGUAGE_READY_TO_TRAIN":
        raise RuntimeError("Phase1 receipt does not authorize the frozen training protocol")
    if sha256(manifest_path) != receipt["manifest_sha256"]:
        raise RuntimeError("Phase1 manifest hash mismatch")
    checked = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = bundle / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Phase1 payload mismatch: {relative}")
        checked += 1
    return {"receipt_sha256": expected_receipt, "payload_files_checked": checked}


def verify_architecture_chain(root: Path, expected_receipt: str) -> None:
    detached = (root / "FREEZE_RECEIPT.sha256").read_text(encoding="ascii").split()[0]
    if detached != expected_receipt or sha256(root / "FREEZE_RECEIPT.json") != detached:
        raise RuntimeError("architecture receipt mismatch")
    receipt = json.loads((root / "FREEZE_RECEIPT.json").read_text(encoding="utf-8"))
    manifest = root / "SHA256SUMS.txt"
    if sha256(manifest) != receipt["manifest_sha256"]:
        raise RuntimeError("architecture manifest mismatch")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if sha256(root / relative) != expected:
            raise RuntimeError(f"architecture payload mismatch: {relative}")


def load_u16(path: Path, expected_length: int) -> torch.Tensor:
    raw = array("H")
    with path.open("rb") as handle:
        raw.fromfile(handle, expected_length)
        if handle.read(1):
            raise RuntimeError(f"unexpected extra data in {path}")
    if raw.itemsize != 2:
        raise RuntimeError("platform unsigned-short is not 16 bits")
    if sys.byteorder != "little":
        raw.byteswap()
    tensor = torch.tensor(raw, dtype=torch.long)
    if tensor.numel() != expected_length:
        raise RuntimeError(f"stream length mismatch: {path}")
    return tensor


def load_u32(path: Path, expected_length: int) -> torch.Tensor:
    raw = array("I")
    with path.open("rb") as handle:
        raw.fromfile(handle, expected_length)
        if handle.read(1):
            raise RuntimeError(f"unexpected extra data in {path}")
    if raw.itemsize != 4:
        raise RuntimeError("platform unsigned-int is not 32 bits")
    if sys.byteorder != "little":
        raw.byteswap()
    tensor = torch.tensor(raw, dtype=torch.long)
    if tensor.numel() != expected_length:
        raise RuntimeError(f"index length mismatch: {path}")
    return tensor


def configure_runtime(config: dict[str, Any]) -> None:
    seed = int(config["seed_policy"]["primary_run_seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


def import_architecture(config: dict[str, Any]):
    root = Path(config["architecture"]["bundle"])
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from baby_vnext.checkpoint import load_initialized_checkpoint
    return load_initialized_checkpoint


def set_phase1_scope(model: torch.nn.Module) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.base_model.parameters():
        parameter.requires_grad_(True)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    return {"trainable": trainable, "frozen": frozen}


def lr_for_update(config: dict[str, Any], update: int) -> float:
    schedule = config["lr_schedule"]
    peak = float(schedule["peak_lr"])
    warmup = int(schedule["warmup_updates"])
    maximum = int(config["schedule"]["max_updates"])
    minimum_ratio = float(schedule["minimum_lr_ratio"])
    if not 1 <= update <= maximum:
        raise ValueError("update is outside the frozen schedule")
    if update <= warmup:
        return peak * update / warmup
    progress = (update - warmup) / (maximum - warmup)
    return peak * (minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.AdamW:
    settings = config["optimizer"]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(
        parameters,
        lr=float(settings["lr_peak"]),
        betas=tuple(float(value) for value in settings["betas"]),
        eps=float(settings["eps"]),
        weight_decay=float(settings["weight_decay"]),
        amsgrad=bool(settings["amsgrad"]),
        foreach=bool(settings["foreach"]),
        fused=bool(settings["fused"]),
    )


def windows(stream: torch.Tensor, starts: torch.Tensor, context: int) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = torch.arange(context)
    return stream[starts[:, None] + offsets[None, :]], stream[starts[:, None] + offsets[None, :] + 1]


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    stream: torch.Tensor,
    starts: torch.Tensor,
    *,
    context: int,
    microbatch: int,
    device: torch.device,
) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    tokens = 0
    try:
        for begin in range(0, starts.numel(), microbatch):
            selected = starts[begin : begin + microbatch]
            inputs, targets = windows(stream, selected, context)
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits, _ = model(inputs)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"
            )
            total += float(loss_sum.cpu())
            tokens += targets.numel()
    finally:
        model.train(was_training)
    value = total / tokens
    if not math.isfinite(value):
        raise RuntimeError("evaluation produced non-finite loss")
    return value


@torch.no_grad()
def greedy_diagnostics(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompts: list[str],
    *,
    context: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    was_training = model.training
    model.eval()
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    rows = []
    try:
        for prompt in prompts:
            prompt_ids = tokenizer.encode(prompt).ids
            generated = [int(bos), *prompt_ids]
            new_tokens = []
            for _ in range(32):
                inputs = torch.tensor([generated[-context:]], device=device)
                logits, _ = model(inputs)
                token = int(logits[0, -1].argmax())
                generated.append(token)
                new_tokens.append(token)
                if token == eos:
                    break
            trigrams = [tuple(new_tokens[index : index + 3]) for index in range(max(0, len(new_tokens) - 2))]
            rows.append({
                "prompt": prompt,
                "prompt_ids": prompt_ids,
                "generated_ids": new_tokens,
                "decoded": tokenizer.decode(new_tokens, skip_special_tokens=True),
                "stopped_on_eos": bool(new_tokens and new_tokens[-1] == eos),
                "immediate_eos": bool(new_tokens and new_tokens[0] == eos),
                "three_identical_token_run": any(
                    new_tokens[index] == new_tokens[index + 1] == new_tokens[index + 2]
                    for index in range(max(0, len(new_tokens) - 2))
                ),
                "repeated_token_trigram": len(trigrams) != len(set(trigrams)),
            })
    finally:
        model.train(was_training)
    return rows


def load_all(bundle: Path, *, require_sealed: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    chain = verify_integrity_chain(bundle) if require_sealed else {"preseal": True}
    config_path = bundle / "PHASE1_LANGUAGE_CONFIG.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    verify_architecture_chain(
        Path(config["architecture"]["bundle"]), config["architecture"]["receipt_sha256"]
    )
    if sha256(Path(config["data"]["tokenizer_path"])) != config["data"]["tokenizer_sha256"]:
        raise RuntimeError("tokenizer mismatch")
    for key in ("train_source", "dev_source", "train_stream", "dev_stream"):
        expected = config["data"][key + "_sha256"]
        if sha256(Path(config["data"][key])) != expected:
            raise RuntimeError(f"data mismatch: {key}")
    if sha256(Path(config["schedule"]["path"])) != config["schedule"]["sha256"]:
        raise RuntimeError("training schedule mismatch")
    if sha256(Path(config["evaluation"]["selection_path"])) != config["evaluation"]["selection_sha256"]:
        raise RuntimeError("evaluation selection mismatch")
    init_path = Path(config["initialization"]["checkpoint"])
    if sha256(init_path) != config["initialization"]["checkpoint_sha256"]:
        raise RuntimeError("initialized checkpoint mismatch")
    return config, chain


def load_streams_and_indices(bundle: Path, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    train = load_u16(Path(config["data"]["train_stream"]), int(config["data"]["train_stream_tokens"]))
    dev = load_u16(Path(config["data"]["dev_stream"]), int(config["data"]["dev_stream_tokens"]))
    shape = config["schedule"]["shape"]
    schedule = load_u32(Path(config["schedule"]["path"]), int(shape[0]) * int(shape[1])).view(shape)
    eval_count = int(config["evaluation"]["dev_windows"]) + int(config["evaluation"]["train_windows"])
    evaluation = load_u32(Path(config["evaluation"]["selection_path"]), eval_count)
    if int(schedule.max()) >= train.numel() - 256 or int(evaluation[: int(config["evaluation"]["dev_windows"])].max()) >= dev.numel() - 256:
        raise RuntimeError("materialized window start exceeds stream")
    return {"train": train, "dev": dev, "schedule": schedule, "evaluation": evaluation}


def load_initial_model(config: dict[str, Any]):
    loader = import_architecture(config)
    model, payload = loader(
        config["initialization"]["checkpoint"],
        expected_tokenizer_sha256=config["data"]["tokenizer_sha256"],
    )
    if payload["optimizer_updates"] != 0 or payload["model_state_sha256"] != config["initialization"]["model_state_sha256"]:
        raise RuntimeError("initialized model metadata mismatch")
    scope = set_phase1_scope(model)
    if scope != {"trainable": 60536064, "frozen": 984321}:
        raise RuntimeError(f"Phase1 scope mismatch: {scope}")
    return model, payload, scope


def evaluate_checkpoint(
    bundle: Path,
    config: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    model: torch.nn.Module,
    update: int,
    device: torch.device,
) -> dict[str, Any]:
    dev_count = int(config["evaluation"]["dev_windows"])
    dev_starts = tensors["evaluation"][:dev_count]
    train_starts = tensors["evaluation"][dev_count:]
    kwargs = {
        "context": int(config["schedule"]["context"]),
        "microbatch": int(config["evaluation"]["eval_microbatch"]),
        "device": device,
    }
    dev_ce = evaluate_loss(model, tensors["dev"], dev_starts, **kwargs)
    train_ce = evaluate_loss(model, tensors["train"], train_starts, **kwargs)
    tokenizer = Tokenizer.from_file(config["data"]["tokenizer_path"])
    prompt_record = json.loads((bundle / "data" / "GENERATION_PROMPTS.json").read_text(encoding="utf-8"))
    generations = greedy_diagnostics(
        model,
        tokenizer,
        prompt_record["prompts"],
        context=int(config["schedule"]["context"]),
        device=device,
    )
    return {
        "update": update,
        "dev_ce": dev_ce,
        "dev_perplexity": math.exp(dev_ce),
        "train_ce": train_ce,
        "train_perplexity": math.exp(train_ce),
        "train_dev_gap": dev_ce - train_ce,
        "generation_descriptive": {
            "prompts": len(generations),
            "non_immediate_eos": sum(not row["immediate_eos"] for row in generations),
            "no_three_identical_run": sum(not row["three_identical_token_run"] for row in generations),
            "no_repeated_token_trigram": sum(not row["repeated_token_trigram"] for row in generations),
            "rows": generations,
        },
        "binding_status": "NOT_APPLICABLE_UNTRAINED",
    }


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    with temporary.open("r+b") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def provenance_hashes(bundle: Path, config: dict[str, Any]) -> dict[str, str]:
    return {
        "phase1_config_sha256": sha256(bundle / "PHASE1_LANGUAGE_CONFIG.json"),
        "architecture_receipt_sha256": config["architecture"]["receipt_sha256"],
        "tokenizer_sha256": config["data"]["tokenizer_sha256"],
        "train_stream_sha256": config["data"]["train_stream_sha256"],
        "dev_stream_sha256": config["data"]["dev_stream_sha256"],
        "schedule_sha256": config["schedule"]["sha256"],
        "evaluation_selection_sha256": config["evaluation"]["selection_sha256"],
        "initialized_checkpoint_sha256": config["initialization"]["checkpoint_sha256"],
    }


def validate_restart_metadata(payload: dict[str, Any], expected: dict[str, str], run_seed: int) -> None:
    if payload.get("schema") != "baby_vnext_phase1_restart_v1":
        raise RuntimeError("restart schema mismatch")
    if payload.get("provenance_hashes") != expected:
        raise RuntimeError("restart provenance mismatch")
    if int(payload.get("run_seed", -1)) != run_seed:
        raise RuntimeError("restart seed mismatch")
    if int(payload.get("completed_update", -1)) < 0:
        raise RuntimeError("restart update is invalid")
    for key in ("model_state_dict", "optimizer_state_dict", "python_rng_state", "torch_rng_state", "cuda_rng_states", "scheduler_state"):
        if key not in payload:
            raise RuntimeError(f"restart missing {key}")


def make_restart(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    completed_update: int,
    config: dict[str, Any],
    hashes: dict[str, str],
    best: dict[str, Any],
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": "baby_vnext_phase1_restart_v1",
        "run_seed": int(config["seed_policy"]["primary_run_seed"]),
        "completed_update": completed_update,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all(),
        "scheduler_state": {"last_completed_update": completed_update},
        "provenance_hashes": hashes,
        "best": best,
        "evaluations": evaluations,
    }


def restore_restart(
    payload: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    expected: dict[str, str],
    run_seed: int,
) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    validate_restart_metadata(payload, expected, run_seed)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    random.setstate(payload["python_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
    if payload["scheduler_state"]["last_completed_update"] != payload["completed_update"]:
        raise RuntimeError("restart scheduler/update mismatch")
    return int(payload["completed_update"]), dict(payload["best"]), list(payload["evaluations"])


def finite_model_and_optimizer(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    for parameter in model.parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise RuntimeError("non-finite model parameter")
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                raise RuntimeError("non-finite optimizer state")


def save_model_only(path: Path, model: torch.nn.Module, update: int, hashes: dict[str, str], evaluation: dict[str, Any]) -> None:
    atomic_torch_save(path, {
        "schema": "baby_vnext_phase1_model_v1",
        "completed_update": update,
        "model_state_dict": model.state_dict(),
        "provenance_hashes": hashes,
        "evaluation": evaluation,
    })


def run_training(bundle: Path, config: dict[str, Any], output: Path, resume: Path | None) -> None:
    if os.environ.get("PYTHONHASHSEED") != str(config["seed_policy"]["primary_run_seed"]):
        raise RuntimeError("train mode requires the frozen PYTHONHASHSEED")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    if resume is None and output.exists():
        raise RuntimeError("fresh output directory already exists")
    output.mkdir(parents=True, exist_ok=resume is not None)
    (output / "checkpoints").mkdir(exist_ok=True)
    config_hashes = provenance_hashes(bundle, config)
    tensors = load_streams_and_indices(bundle, config)
    model, _, _ = load_initial_model(config)
    model.to("cuda")
    baseline = evaluate_checkpoint(bundle, config, tensors, model, 0, torch.device("cuda"))
    frozen_baseline = json.loads((bundle / "PHASE1_LANGUAGE_BASELINES.json").read_text(encoding="utf-8"))["primary_u0"]
    if abs(baseline["dev_ce"] - frozen_baseline["dev_ce"]) > 1e-6 or abs(baseline["train_ce"] - frozen_baseline["train_ce"]) > 1e-6:
        raise RuntimeError("U0 baseline reproduction failed")
    configure_runtime(config)
    optimizer = build_optimizer(model, config)
    completed = 0
    best = {"dev_ce": baseline["dev_ce"], "update": 0}
    evaluations = [baseline]
    if resume is not None:
        payload = torch.load(resume, map_location="cuda", weights_only=False)
        completed, best, evaluations = restore_restart(
            payload,
            model,
            optimizer,
            expected=config_hashes,
            run_seed=int(config["seed_policy"]["primary_run_seed"]),
        )
    metrics_path = output / "training_metrics.jsonl"
    if metrics_path.exists():
        retained = []
        for line in metrics_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if int(row["update"]) <= completed:
                retained.append(line)
        metrics_path.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
    evaluation_points = set(int(value) for value in config["evaluation"]["points"])
    if completed in evaluation_points and not (output / f"evaluation_{completed:04d}.json").exists():
        missing = evaluate_checkpoint(bundle, config, tensors, model, completed, torch.device("cuda"))
        atomic_json(output / f"evaluation_{completed:04d}.json", missing)
        evaluations = [row for row in evaluations if row["update"] != completed] + [missing]
    microbatch = int(config["schedule"]["microbatch"])
    accumulation = int(config["schedule"]["gradient_accumulation"])
    context = int(config["schedule"]["context"])
    maximum = int(config["schedule"]["max_updates"])
    clip = float(config["optimizer"]["gradient_clip_norm"])
    base_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for update in range(completed + 1, maximum + 1):
        lr = lr_for_update(config, update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        starts = tensors["schedule"][update - 1]
        for begin in range(0, starts.numel(), microbatch):
            selected = starts[begin : begin + microbatch]
            inputs, targets = windows(tensors["train"], selected, context)
            inputs = inputs.to("cuda")
            targets = targets.to("cuda")
            logits, _ = model(inputs)
            raw_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            if not bool(torch.isfinite(raw_loss)):
                raise RuntimeError(f"non-finite loss at update {update}")
            (raw_loss / accumulation).backward()
            update_loss += float(raw_loss.detach().cpu()) / accumulation
        if any(parameter.grad is not None for name, parameter in model.named_parameters() if not name.startswith("base_model.")):
            raise RuntimeError("frozen binding parameter received a gradient")
        for parameter in base_parameters:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"missing/non-finite base gradient at update {update}")
        gradient_norm = torch.nn.utils.clip_grad_norm_(base_parameters, clip)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"non-finite gradient norm at update {update}")
        optimizer.step()
        row = {
            "update": update,
            "train_batch_ce": update_loss,
            "lr": lr,
            "gradient_norm_before_clip": float(gradient_norm),
        }
        with metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row) + "\n")
        if update % int(config["checkpointing"]["rolling_restart_interval"]) == 0:
            finite_model_and_optimizer(model, optimizer)
            restart = make_restart(
                model,
                optimizer,
                completed_update=update,
                config=config,
                hashes=config_hashes,
                best=best,
                evaluations=evaluations,
            )
            atomic_torch_save(output / "rolling_restart.pt", restart)
        if update in evaluation_points:
            evaluation = evaluate_checkpoint(bundle, config, tensors, model, update, torch.device("cuda"))
            atomic_json(output / f"evaluation_{update:04d}.json", evaluation)
            evaluations.append(evaluation)
            save_model_only(
                output / "checkpoints" / f"checkpoint_{update:04d}.pt",
                model,
                update,
                config_hashes,
                evaluation,
            )
            if evaluation["dev_ce"] < best["dev_ce"]:
                best = {"dev_ce": evaluation["dev_ce"], "update": update}
                save_model_only(output / "checkpoints" / "best.pt", model, update, config_hashes, evaluation)
            restart = make_restart(
                model,
                optimizer,
                completed_update=update,
                config=config,
                hashes=config_hashes,
                best=best,
                evaluations=evaluations,
            )
            atomic_torch_save(output / "rolling_restart.pt", restart)
    final_eval = next(row for row in evaluations if row["update"] == maximum)
    ordered = sorted(evaluations, key=lambda row: row["update"])
    consecutive = any(
        ordered[index - 1]["dev_ce"] <= 3.1 and ordered[index]["dev_ce"] <= 3.1
        for index in range(1, len(ordered))
    )
    best_eval = next(row for row in ordered if row["update"] == best["update"])
    gates = {
        "completed_6000": True,
        "best_dev_ce_le_3": best["dev_ce"] <= 3.0,
        "final_dev_ce_le_3_1": final_eval["dev_ce"] <= 3.1,
        "two_consecutive_dev_ce_le_3_1": consecutive,
        "best_train_dev_gap_le_0_5": best_eval["train_dev_gap"] <= 0.5,
    }
    status = "PHASE1_LANGUAGE_READY" if all(gates.values()) else "PHASE1_LANGUAGE_ACQUISITION_FAIL"
    atomic_json(output / "FINAL_STATUS.json", {
        "status": status,
        "gates": gates,
        "best": best,
        "final": final_eval,
        "provenance_hashes": config_hashes,
        "transfer": "LOCKED_UNSCORED",
        "final_and_sacred": "LOCKED_UNACCESSED",
    })


def validate_runner(bundle: Path, config: dict[str, Any], *, preseal: bool) -> dict[str, Any]:
    tensors = load_streams_and_indices(bundle, config)
    model, payload, scope = load_initial_model(config)
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    required = {
        "verify_integrity_chain", "load_all", "configure_runtime", "build_optimizer",
        "evaluate_checkpoint", "make_restart", "restore_restart", "run_training",
        "atomic_torch_save", "lr_for_update", "set_phase1_scope",
    }
    hashes = provenance_hashes(bundle, config)
    fixture_path = bundle / "VALIDATION_RESTART_FIXTURE.pt"
    if preseal:
        fixture = {
            "schema": "baby_vnext_phase1_restart_v1",
            "run_seed": int(config["seed_policy"]["primary_run_seed"]),
            "completed_update": 100,
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": [],
            "scheduler_state": {"last_completed_update": 100},
            "provenance_hashes": hashes,
            "best": {"dev_ce": 7.0, "update": 0},
            "evaluations": [],
        }
        atomic_torch_save(fixture_path, fixture)
    fixture = torch.load(fixture_path, map_location="cpu", weights_only=False)
    validate_restart_metadata(
        fixture,
        hashes,
        int(config["seed_policy"]["primary_run_seed"]),
    )
    invalid_rejected = False
    bad = dict(fixture)
    bad["provenance_hashes"] = dict(hashes, schedule_sha256="invalid")
    try:
        validate_restart_metadata(bad, hashes, int(config["seed_policy"]["primary_run_seed"]))
    except RuntimeError:
        invalid_rejected = True
    result = {
        "status": "PASS",
        "sealed_chain_checked": not preseal,
        "model_loaded": True,
        "initialized_checkpoint_optimizer_updates": payload["optimizer_updates"],
        "scope": scope,
        "stream_lengths": {"train": tensors["train"].numel(), "dev": tensors["dev"].numel()},
        "schedule_shape": list(tensors["schedule"].shape),
        "schedule_all_resolves": bool((tensors["schedule"] < tensors["train"].numel() - 256).all()),
        "lr_boundaries": {
            "update1": lr_for_update(config, 1),
            "update200": lr_for_update(config, 200),
            "update201": lr_for_update(config, 201),
            "update6000": lr_for_update(config, 6000),
        },
        "runner_required_functions_present": sorted(required),
        "runner_completeness": required.issubset(function_names),
        "restart_fixture_roundtrip": True,
        "restart_provenance_mismatch_rejected": invalid_rejected,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "training_performed": False,
    }
    if not result["runner_completeness"] or not invalid_rejected:
        raise RuntimeError("runner structural validation failed")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validate", "u0", "train"), required=True)
    parser.add_argument("--bundle", type=Path, default=BUNDLE_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-preseal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = args.bundle.resolve()
    if args.allow_preseal and args.mode == "train":
        raise RuntimeError("train mode can never bypass the sealed integrity chain")
    config, chain = load_all(bundle, require_sealed=not args.allow_preseal)
    if args.mode == "validate":
        result = validate_runner(bundle, config, preseal=args.allow_preseal)
        result["chain"] = chain
        if args.allow_preseal:
            atomic_json(bundle / "RUNNER_VALIDATION.json", result)
        print(json.dumps(result, indent=2))
        return
    tensors = load_streams_and_indices(bundle, config)
    model, _, scope = load_initial_model(config)
    if args.mode == "u0":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable")
        model.to("cuda")
        baseline = evaluate_checkpoint(bundle, config, tensors, model, 0, torch.device("cuda"))
        result = {
            "status": "PHASE1_U0_BASELINE_FROZEN",
            "primary_u0": baseline,
            "scope": scope,
            "optimizer_created": False,
            "optimizer_steps": 0,
            "training_performed": False,
        }
        if args.allow_preseal:
            atomic_json(bundle / "PHASE1_LANGUAGE_BASELINES.json", result)
        print(json.dumps(result, indent=2))
        return
    if args.output_dir is None:
        raise RuntimeError("train mode requires --output-dir")
    run_training(bundle, config, args.output_dir.resolve(), args.resume.resolve() if args.resume else None)


if __name__ == "__main__":
    main()
