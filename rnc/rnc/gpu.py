"""GPU environment for the AMD Radeon RX 9060 XT (gfx1200) under ROCm 7.2 on Windows.

Call :func:`configure_environment` BEFORE importing torch. It encodes three hard-won facts about this machine:

1. The HIP runtime needs a writable TEMP; a sandboxed TEMP makes ``torch.cuda.mem_get_info`` raise
   "HIP error: invalid argument". We point TEMP/TMP at a workspace directory.
2. The fused flash / memory-efficient SDPA kernels are disabled on this GPU unless
   ``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1``. Measured 2026-10-16: 10x faster than the math kernel, agreement
   within one bf16 ulp. The math kernel remains available as a config switch.
3. The historical 118M runs hit a HIP OOM at 13-16 GB; RNC keeps peak memory far below the card by configuration.

GPU-first policy: substantial compute must run on the GPU. :func:`require_gpu` refuses to continue otherwise.
"""
from __future__ import annotations

import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
GPU_TEMP = Path(r"C:\DaveLM-Training\tmp\gpu_temp")


def configure_environment() -> None:
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        GPU_TEMP.mkdir(parents=True, exist_ok=True)
        probe = GPU_TEMP / "write_probe.tmp"
        probe.write_text("ok")
        probe.unlink()
        os.environ["TEMP"] = str(GPU_TEMP)
        os.environ["TMP"] = str(GPU_TEMP)
    except OSError:
        pass  # fall back to the inherited TEMP; require_gpu() will surface a HIP failure if it is unwritable
    os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    # Note: expandable_segments is NOT supported by the HIP allocator on Windows (torch warns and ignores it), so no
    # allocator tuning is applied; peak memory is kept well below the card's 16 GB by the config instead.


def require_gpu(device: str = "cuda"):
    import torch

    if device != "cuda":
        raise SystemExit(f"RNC is GPU-first; device={device!r} is not permitted for model compute.")
    if not torch.cuda.is_available():
        raise SystemExit("No ROCm/HIP GPU visible to torch. Refusing CPU fallback.")
    free, total = torch.cuda.mem_get_info()  # raises on the unwritable-TEMP HIP fault
    return torch.device("cuda")


def describe_gpu() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"available": False}
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "name": p.name,
        "arch": p.gcnArchName,
        "total_memory_bytes": int(p.total_memory),
        "free_memory_bytes": int(free),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "aotriton_experimental": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"),
    }


def memory_stats() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def synchronize_quietly() -> None:
    """Every GPU entry point calls this in ``finally``: a process exiting with HIP work in flight hangs forever."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
