# RNC — Racina Neural Core: architecture decision record

Project DaveLM · family Mangaris · codename Baby · architecture **RNC-1** (successor to MRCN-Alpha) · 2026-10-16.
Philosophy: FUNCTION FIRST. INTELLIGENCE NEXT. PERFECTION OVER TIME.

## 1. What RNC is for

A from-scratch, English-learning autoregressive language model that is trainable, inspectable and reproducible on
one AMD Radeon RX 9060 XT (16 GB, gfx1200, ROCm 7.2 Windows, PyTorch 2.9.1+rocmsdk20260116). Ordinary next-token
prediction on real English text is the whole training objective. No task scaffolding, no answer routes, no sidecar
heads, no synthetic-id curriculum.

## 2. What the history taught (sources: MRCN-Alpha brief and 61M v0.10 brief, both compiled read-only on 2026-10-16)

| Lesson from the lineage | RNC response |
|---|---|
| MRCN-Alpha (118M) never saw real text: 1024 synthetic ids, copy tasks, seven training stages planned, one reached. | Train on real English from update 0 with a real tokenizer. |
| Learned absolute positions (61M, Bigger Baby) produced position/length failures; RoPE (MRCN-Alpha) was the one architectural choice that worked. | RoPE, no learned position table. |
| Heavy-tailed gradients: 36–41 % of MRCN steps hit the clip; meltdown machinery, LR throttles, update skipping. | QK-normalisation (RMSNorm on q and k before the dot product) removes the attention-logit growth that drives most such spikes; plain clip 1.0; no bespoke safety layer. |
| Tied embedding "drifted 8–12 % per 100 updates" at LR 1e-3. | Peak LR 6e-4 with warm-up and cosine decay (the standard ~100M recipe); weight decay off for embeddings and norms. |
| 61M PropMatchHead / binding sidecar: 984K parameters never trained, readout picked recency, always-on copy boosts destroyed sentences. | No auxiliary heads. One residual stream, one tied readout. |
| MRCN-Alpha peak memory 13–16 GB at batch 48, seq ≤ 296, math attention kernel; HIP OOM at u551. | Fused SDPA kernels verified on gfx1200 with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (10× faster, one-bf16-ulp agreement). Memory is measured by `bench`, not assumed. |
| Processes with un-synchronised GPU work hang at exit; HIP needs a writable TEMP. | Every GPU entry point synchronises in `finally`; the launcher sets TEMP to a workspace directory. |
| Old 16K tokenizer migration: remapping a trained model destroyed behaviour. | RNC is born with its tokenizer; nothing is transplanted. |
| Old 61M line: "healthy language cross-entropy did not mean the model could copy"; selecting the right pair was the bottleneck. | RNC-1 makes no capability claim beyond what the held-out loss and the sampled text show. |

## 3. The design: RNC-1

Decoder-only pre-norm transformer.

| Component | Choice | Why |
|---|---|---|
| d_model | 768 | 12 heads × 64; the well-charted ~100M operating point. |
| Layers | 12 | Depth matters more than width at this scale; 12 keeps activations small on 16 GB. |
| Heads | 12 × head_dim 64 | head_dim 64 is the fast path for the fused SDPA kernels. |
| MLP | SwiGLU, hidden 2048 (≈ 8/3·d rounded to 64) | Better quality per parameter than ReLU/GELU; already validated numerically on this GPU by MRCN-Alpha. |
| Norm | RMSNorm (eps 1e-6), pre-norm, final norm; fp32 reduction | Cheaper than LayerNorm, no bias. |
| QK-norm | RMSNorm over head_dim on q and k, learnable per-dim scale | Stability without throttles (see §2). |
| Positions | RoPE, base 10000, full head_dim, applied in fp32 | Relative positions; extrapolates better than learned absolute. |
| Attention | causal SDPA; `auto` (fused), `math`, and an explicit fp32 `reference` path | The reference path makes causality and kernel parity testable. |
| Biases | none | Fewer parameters, no benefit at this scale. |
| Embedding / readout | tied | Saves one V×d matrix; standard for ≤ 150M models. |
| Dropout | 0 | Far less than one epoch of data per run; no regularisation needed. |
| Init | N(0, 0.02); attention-out and MLP-down scaled by 1/√(2·L) | GPT-2-style depth scaling keeps the residual-stream variance bounded. |
| Vocab | **8,192** (measured, see docs/TOKENIZER.md) | TinyStories has a small vocabulary; 16K left 62 % of ids seen < 10 times on 4.1M held-out tokens for 0.8 % better compression. |
| Context | 512 tokens | TinyStories documents average ~200 tokens; 512 packs 2–3 stories per row. |

### Parameter count (analytic; the module count and the test suite agree exactly)

| Block | Formula | Params |
|---|---|---:|
| Token embedding (tied readout) | 8,192 × 768 | 6,291,456 |
| Attention per layer | 4·768² + 2·64 (QK-norm scales) | 2,359,424 |
| MLP per layer | 3·768·2048 | 4,718,592 |
| Norms per layer | 2·768 | 1,536 |
| 12 layers | 12 × 7,079,552 | 84,954,624 |
| Final norm | 768 | 768 |
| **Total** | | **91,246,848** |

≈ 91.2M parameters, 85.0M non-embedding. Inside the 80–150M practical range; nothing was added to make the number
larger. The 16K variant would have been 97.5M with the extra 6.3M sitting in rarely-used embedding rows.

### Why not bigger or smaller

- Smaller (≈ 50M) trains faster but we want a core that can be extended; 85M non-embedding parameters is the
  smallest size at which the well-known ~100M behaviours (fluent short prose after a few hundred million tokens)
  are reliably observed in the literature on this kind of corpus.
- Bigger (≈ 150M) halves throughput on a ~5 TFLOPS card for no demonstrated need in a first experiment.

### Expected memory

fp32 master weights 0.37 GB + AdamW moments 0.73 GB + fp32 grads 0.37 GB ≈ 1.5 GB resident. Activations in bf16
autocast at micro-batch 16 × 512 ≈ 2–3 GB, plus the fp32 logit/loss tensor 16 × 512 × 8,192 × 4 B ≈ 0.27 GB
(doubled by its gradient). Estimated peak 4–6 GB. Measured numbers are in docs/TRAINING.md.

## 4. Training recipe (first run)

AdamW β=(0.9, 0.95), eps 1e-8, weight decay 0.1 on matrices only; peak LR 6e-4, linear warm-up, cosine to 10 %;
grad-clip 1.0; bf16 autocast with fp32 master weights (fp32 is a config switch); a fixed number of tokens per
optimizer step; deterministic data order from a seed; checkpoints with optimizer + RNG + statistics; fixed held-out
validation windows at fixed intervals; fixed-prompt samples at the same intervals; step-0 baseline always recorded.

## 5. Out of scope for RNC-1

Mixture-of-experts, sliding-window attention, multi-token prediction, grouped-query attention (no inference
pressure yet), document-masked packing (noted as a later refinement), KV-cache generation, any instruction or
dialogue formatting.
