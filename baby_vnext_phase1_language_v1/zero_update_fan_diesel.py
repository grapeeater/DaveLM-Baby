from __future__ import annotations

import contextlib
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
ARCH = Path(r"C:\DaveLM-CADAVER\baby_vnext_60m_design_v1")
sys.path.insert(0, str(ARCH))

from baby_vnext import BabyVNextConfig, BabyVNextWithBinding  # noqa: E402
from baby_vnext.checkpoint import _state_digest  # noqa: E402


DISPOSABLE_SEED = 620999
CONTEXT = 256


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def process_memory() -> dict[str, float]:
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb
    )
    if not ok:
        return {}
    mib = 1024**2
    return {
        "working_set_mib": counters.WorkingSetSize / mib,
        "peak_working_set_mib": counters.PeakWorkingSetSize / mib,
        "private_mib": counters.PrivateUsage / mib,
    }


def all_base_gradients_valid(model: BabyVNextWithBinding) -> tuple[bool, int, int]:
    present = 0
    finite = 0
    for parameter in model.base_model.parameters():
        if parameter.grad is not None:
            present += 1
            if bool(torch.isfinite(parameter.grad).all()):
                finite += 1
    return present > 0 and present == finite, present, finite


def run_iteration(
    model: BabyVNextWithBinding,
    microbatch: int,
    *,
    precision: str,
    accumulate_without_clear: bool = False,
) -> dict[str, object]:
    device = next(model.parameters()).device
    if not accumulate_without_clear:
        model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cpu").manual_seed(
        DISPOSABLE_SEED + microbatch + (1000 if precision == "bf16_autocast" else 0)
    )
    inputs = torch.randint(
        0, model.config.vocab_size, (microbatch, CONTEXT), generator=generator
    ).to(device)
    targets = torch.randint(
        0, model.config.vocab_size, (microbatch, CONTEXT), generator=generator
    ).to(device)
    if precision == "fp32":
        precision_context = contextlib.nullcontext()
    elif precision == "bf16_autocast":
        precision_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        raise ValueError(precision)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with precision_context:
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - start
    backward_start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - backward_start
    valid, gradients_present, gradients_finite = all_base_gradients_valid(model)
    binding_gradients_absent = all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith("base_model.")
    )
    result = {
        "microbatch": microbatch,
        "context": CONTEXT,
        "precision": precision,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "iteration_seconds": forward_seconds + backward_seconds,
        "loss": float(loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradient_tensors_present": gradients_present,
        "gradient_tensors_finite": gradients_finite,
        "gradients_valid": valid,
        "binding_gradients_absent": binding_gradients_absent,
        "max_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "max_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        "allocated_after_backward_mib": torch.cuda.memory_allocated() / (1024**2),
        "reserved_after_backward_mib": torch.cuda.memory_reserved() / (1024**2),
        "host_memory": process_memory(),
        "optimizer_created": False,
        "optimizer_step": False,
        "accumulate_without_clear": accumulate_without_clear,
    }
    del inputs, targets, logits, loss
    return result


def identify_sdpa_backend(model: BabyVNextWithBinding) -> list[str]:
    model.eval()
    tokens = torch.randint(0, model.config.vocab_size, (1, 64), device="cuda")
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.no_grad(), torch.profiler.profile(activities=activities) as profile:
        model(tokens)
        torch.cuda.synchronize()
    names = sorted(
        event.key for event in profile.key_averages() if "scaled_dot_product" in event.key
    )
    del tokens
    model.train()
    return names


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Fan Diesel GPU is unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(DISPOSABLE_SEED)
    torch.cuda.manual_seed_all(DISPOSABLE_SEED)
    config = BabyVNextConfig.load(ARCH / "BABY_VNEXT_CONFIG.json")
    model = BabyVNextWithBinding(config)
    initial_digest = _state_digest(model)
    model.to("cuda").train()
    backend_events = identify_sdpa_backend(model)
    records: list[dict[str, object]] = []
    captured_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        records.append(run_iteration(model, 8, precision="fp32"))
        records.append(
            run_iteration(
                model, 8, precision="fp32", accumulate_without_clear=True
            )
        )
        safe_fraction = float(records[0]["max_reserved_mib"]) / (
            torch.cuda.get_device_properties(0).total_memory / (1024**2)
        )
        if safe_fraction < 0.70:
            model.zero_grad(set_to_none=True)
            records.append(run_iteration(model, 16, precision="fp32"))
            records.append(
                run_iteration(
                    model, 16, precision="fp32", accumulate_without_clear=True
                )
            )
        if torch.cuda.is_bf16_supported():
            model.zero_grad(set_to_none=True)
            records.append(run_iteration(model, 8, precision="bf16_autocast"))
        captured_warnings.extend(str(item.message) for item in warning_records)
    model.zero_grad(set_to_none=True)
    model.to("cpu")
    final_digest = _state_digest(model)
    total_memory_mib = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    result = {
        "status": "PASS",
        "disposable_seed": DISPOSABLE_SEED,
        "eventual_training_seed_used": False,
        "model_saved": False,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "real_corpus_used": False,
        "device": torch.cuda.get_device_name(0),
        "total_device_memory_mib": total_memory_mib,
        "torch": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_enabled": False,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "sdpa_profiler_events": backend_events,
        "warnings": captured_warnings,
        "measurements": records,
        "weight_digest_before": initial_digest,
        "weight_digest_after": final_digest,
        "weights_unchanged": initial_digest == final_digest,
    }
    if not result["weights_unchanged"]:
        raise RuntimeError("weights changed during zero-update preflight")
    if not all(record["gradients_valid"] for record in records):
        raise RuntimeError("non-finite or missing gradients")
    if not all(record["binding_gradients_absent"] for record in records):
        raise RuntimeError("binding parameters received language gradients")
    (ROOT / "FAN_DIESEL_ZERO_UPDATE_PREFLIGHT.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
