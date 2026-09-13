# Fan Diesel zero-update preflight

Status: **PASS**

The test used disposable seed **620999**, synthetic language-shaped tensors, full context 256, and an initialized vNext model that was never saved as a training parent. It created no optimizer, took no optimizer step, used no real corpus data, and left the model-state digest unchanged at `21b51216ae54860e4b35797932233a528c17672d8da360b41dd815854ffce42b`.

## Physical result

Device: AMD Radeon RX 9060 XT, 16,304 MiB reported VRAM. Runtime: Python 3.12.14, torch 2.12.0+rocm7.14.0. Deterministic algorithms were enabled and TF32 disabled.

| Microbatch | Precision | Forward | Backward | Iteration | Peak allocated | Peak reserved | Gradients |
|---:|---|---:|---:|---:|---:|---:|---|
| 8 cold | FP32 | 0.098 s | 2.735 s | 2.833 s | 1,918.6 MiB | 1,998 MiB | 138/138 finite |
| 8 warmed/accumulated | FP32 | 0.059 s | 0.101 s | 0.161 s | 2,156.2 MiB | 2,250 MiB | 138/138 finite |
| 16 | FP32 | 0.133 s | 0.186 s | 0.318 s | 3,338.8 MiB | 3,538 MiB | 138/138 finite |
| 16 second accumulated | FP32 | 0.117 s | 0.190 s | 0.307 s | 3,569.2 MiB | 3,804 MiB | 138/138 finite |
| 8 | BF16 autocast | 0.949 s | 0.843 s | 1.792 s | 1,811.3 MiB | 1,900 MiB | 138/138 finite |

The profiler reported `aten::scaled_dot_product_attention` backed by `aten::_scaled_dot_product_attention_math`; fused flash/memory-efficient kernels were not used. Host-RAM telemetry was unavailable in the runtime, so no host-memory figure is claimed.

## Frozen recommendation

Use **FP32 microbatch 16 with accumulation 4**, preserving effective batch 64. Its accumulated backward path used about 3.8 GiB reserved VRAM, leaving substantial nominal headroom for Adam states, evaluation, and runtime variance. BF16 was numerically valid in this bounded pass but slower and is therefore not selected. Activation checkpointing is unnecessary for Phase 1 at this measured batch shape.

The raw measurements and all tensor checks are in `FAN_DIESEL_ZERO_UPDATE_PREFLIGHT.json`.
