# Baby vNext Phase-1 language protocol

Status: **prospectively frozen; training not performed**

## Scientific purpose

Phase 1 gives the 61,520,385-parameter Baby vNext a bounded ordinary-language acquisition run before binding or factual training. It preserves the Research Baby tokenizer, context length, corpus family, document encoding, next-token objective, AdamW family, float32 policy, and core evaluation semantics. Historical factual, transfer, FINAL, and sacred materials remain locked and are not part of this phase.

## Parent and seed policy

The model starts from the sealed deterministic initialized checkpoint `initialization/seed_610001_initialized.pt`. It contains zero optimizer updates and is not a descendant of Research Baby. The primary run seed is **610001**. Seeds **610002** and **610003** are reserved for later replication and are not authorized by this package.

The primary seed controls initialization, dropout, and runtime stochasticity. Literal training-window starts were generated once with schedule seed **61000101** and are consumed directly. Frozen evaluation windows use seed **61000102**. Training must launch with `PYTHONHASHSEED=610001`; Python, torch CPU, and all GPU RNG states are set to 610001.

## Data and objective

The authoritative Pilot 1 train and DEV JSONL files remain the sources. The frozen binary streams reproduce Pilot 1 encoding exactly: `<doc>` separates documents and each document is encoded as `<bos> + text + <eos>`. Train and DEV remain separate.

Training uses fixed 256-token input windows and their 256 next tokens. There is no padding. Full-vocabulary causal cross-entropy supervises every next-token position. Four equal microbatches are averaged by dividing each microbatch mean loss by four before backward, which preserves effective-batch mean semantics.

| Quantity | Frozen value |
|---|---:|
| Train documents | 9,000 |
| Train stream tokens | 3,576,861 |
| DEV documents | 1,000 |
| DEV stream tokens | 390,340 |
| Updates | 6,000 |
| Effective windows/update | 64 |
| Context / supervised positions per window | 256 |
| Total supervised positions | 98,304,000 |
| Train-stream equivalents | 27.4833 |

The 27.5-fold stream reuse is a material overfitting risk. The fixed train/DEV gap and DEV trajectory diagnose it; no same-run data or schedule change is allowed.

## Runtime and batching

| Setting | Frozen value |
|---|---|
| Runtime | Python 3.12.14; torch 2.12.0+rocm7.14.0; tokenizers 0.23.1 |
| Device | AMD Radeon RX 9060 XT |
| Precision | float32; no autocast; TF32 disabled |
| Determinism | `torch.use_deterministic_algorithms(True)` |
| Attention | PyTorch SDPA; observed math backend |
| Microbatch | 16 |
| Accumulation | 4 |
| Effective batch | 64 |
| DataLoader | none; literal window starts |

## Parameter scope

All **60,536,064 base-model parameters** train: token and position embeddings, blocks 0-11, final normalization, and untied language head. The **984,321 binding/localization parameters** are frozen and must retain `grad=None`.

Pilot 1 protected blocks 0-3 to preserve an already-learned binding capability. Phase 1 starts from scratch, so no learned binding state exists to preserve. Copying that block mask would arbitrarily deny half of the old network, or one-third of vNext, language acquisition. Training all 12 base blocks preserves the functional intent: train the ordinary-language model while leaving the untrained binding adapter inactive.

## Optimizer and learning-rate schedule

The future runner constructs one fresh optimizer only after sealed verification and U0 reproduction:

`AdamW(lr=3e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=0.05, amsgrad=False, foreach=False, fused=False)`

The global gradient norm over trainable base parameters is clipped to 2.0.

Updates 1-200 linearly warm from 0 to 3e-4. Updates 201-6000 use cosine decay from 3e-4 to 3e-5. No adaptive or outcome-driven schedule change is allowed. Warmup spans exactly 3,276,800 supervised positions, matching the physical v0.8 from-scratch recipe. This schedule change from Pilot 1 is **SCALE-JUSTIFIED** because Phase 1 is a 6.67-times-longer from-scratch run rather than a short continuation.

## Evaluation and frozen readiness gate

Evaluation runs at U0 and every 500 updates through U6000 on 1,280 frozen DEV windows and 320 frozen train windows. It records aligned causal CE, perplexity, train-versus-DEV CE gap, and ten fixed nonsacred greedy prompts. Generation is descriptive. Learned binding is **not applicable before binding training**; only its interface and frozen parameter scope are checked.

Phase 1 is classified `PHASE1_LANGUAGE_READY` only if all conditions hold:

- 6,000 updates complete;
- best frozen DEV CE is at most 3.0;
- final DEV CE is at most 3.1;
- two consecutive frozen evaluations have DEV CE at most 3.1;
- the selected checkpoint's DEV-minus-train CE gap is at most 0.5;
- all monitored numeric values remain finite.

Pilot 1's authoritative final DEV CE was **2.903933**. The 3.0 best / 3.1 sustained-final thresholds require vNext to enter and remain near that demonstrated healthy-language regime rather than counting any improvement from random initialization as success.

The plain-English “OH FUCK, BIG BABY IS ACTUALLY LEARNING” signal is a sustained DEV CE/perplexity fall from the U0 value of 7.0866, followed by two consecutive evaluations at or below 3.1 without a widening train/DEV gap or numerical instability. It is a progress signal; the complete frozen gate above determines readiness.

There is no DEV-based early stop or same-run adaptation. Hard stops are NaN/Inf, deterministic-operation failure, GPU OOM, integrity or restart mismatch, or checkpoint commit failure.

## Checkpointing and resume

A rolling full restart is atomically committed every 100 updates. Permanent model checkpoints and evaluations are committed every 500 updates. The best checkpoint is the lowest frozen DEV CE; exact ties choose the earlier update.

The rolling state stores model, optimizer, completed update, scheduler position, Python/torch/GPU RNG states, evaluations, best-state metadata, and all frozen provenance hashes. Resume verifies those hashes and restores every state before executing `completed_update + 1`. Updates after the latest committed 100-update state are uncommitted and are deterministically replayed. Evaluation artifacts are independent; a missing scheduled evaluation after a committed update is generated without replaying that update.

## Changes from Pilot 1

| Change | Classification | Reason |
|---|---|---|
| Deterministic from-scratch vNext initialization | REQUIRED | Shapes differ and Phase 1 establishes the new lineage. |
| Train all 12 base blocks; freeze binding adapter | REQUIRED | No learned binding capability exists yet to protect. |
| 6,000 updates instead of 900 language updates | SCALE-JUSTIFIED | Matches the historical positions-per-parameter neighborhood. |
| Warmup plus cosine decay | SCALE-JUSTIFIED | Long from-scratch training needs a bounded schedule; repository precedent fixes the warmup scale. |
| Microbatch 16 x accumulation 4 | SAFETY-JUSTIFIED | Preserves effective batch 64 within measured GPU headroom. |
| Literal training and evaluation window files | SAFETY-JUSTIFIED | Makes ordering and resume identity auditable and immutable. |

No optimizer has been created and no optimizer update has occurred during construction or validation.

## Authorized future launch interface

After separate human authorization, the frozen primary run command is:

```powershell
$env:PYTHONHASHSEED='610001'
& 'C:\DaveLM-CADAVER\sf2_runtime\sf2venv\Scripts\python.exe' -B 'C:\DaveLM-CADAVER\baby_vnext_phase1_language_v1\PHASE1_LANGUAGE_RUNNER.py' --mode train --bundle 'C:\DaveLM-CADAVER\baby_vnext_phase1_language_v1' --output-dir 'C:\DaveLM-CADAVER\baby_vnext_phase1_language_v1_run_seed610001'
```

This package does not authorize the reserved replication seeds.
