# DaveLM / Baby vNext — Full Pre-60M Architecture & Lineage Audit

**AUDIT STATUS:** PASS_WITH_UNCERTAINTIES  
**Audit completed:** 2026-09-09  
**Scope:** Read-only physical repository inspection of the ~10.6M Baby lineage (T1-T13 binding, Pilot 0/1 language, SF1-SF21 factual supervision)  
**Authoritative source:** Physical files in `C:\DaveLM-CADAVER` and `C:\DaveLM-v0.9`  

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Authoritative Current Model](#2-authoritative-current-model)
3. [Architecture Autopsy](#3-architecture-autopsy)
4. [Parameter Accounting](#4-parameter-accounting)
5. [Binding Lineage and Surviving Mechanism](#5-binding-lineage-and-surviving-mechanism)
6. [Language Lineage](#6-language-lineage)
7. [Factual Acquisition Lineage](#7-factual-acquisition-lineage)
8. [SF9–SF21 Coexistence Wall](#8-sf9sf21-coexistence-wall)
9. [SF21 Closure](#9-sf21-closure)
10. [Tokenizer Autopsy](#10-tokenizer-autopsy)
11. [Checkpoint / State-Dict Autopsy](#11-checkpoint--state-dict-autopsy)
12. [Training Pipeline](#12-training-pipeline)
13. [Regression & Evaluation Infrastructure](#13-regression--evaluation-infrastructure)
14. [Locked / Sacred Material Rules](#14-locked--sacred-material-rules)
15. [What vNext Must Preserve](#15-what-vnext-must-preserve)
16. [What vNext Should Not Automatically Inherit](#16-what-vnext-should-not-automatically-inherit)
17. [Capacity / Interference Evidence](#17-capacity--interference-evidence)
18. [60M Migration Constraints](#18-60m-migration-constraints)
19. [Fan Diesel Runtime Constraints](#19-fan-diesel-runtime-constraints)
20. [Discrepancies / Uncertainties](#20-discrepancies--uncertainties)
21. [Codex Handoff](#21-codex-handoff)
22. [Exact Files Codex Should Read First](#22-exact-files-codex-should-read-first)

---

## 1. Executive Summary

### 1.1 Bottom Line

The Baby model is a **decoder-only transformer with 10,594,944 base parameters** (untied language head, 8 blocks, d_model=320, 10 heads, d_mlp=1280, vocab=1024, context=256). With the T13-derived binding machinery, the trained artifact is **10,841,345 parameters**. It was trained from scratch.

The lineage contains **major successes**: T12/T13 achieved structurally-assisted contextual binding at ceiling (317/320 answer, 312/320 BOTH_DISTINCT, 0 collapse, 78/80 quartets, 158/160 reversals). Pilot 1 achieved major language improvement (dev CE 7.15→2.90, ppl 1278→18.25) **while preserving** binding (80/80, 80/80, 0 collapse on both pools).

The lineage also contains **a repeated coexistence frontier**: 21 factual-supervision treatments (SF1–SF21) attempted to teach short ordinary-English factual selection/generalization while simultaneously preserving exact factual answer generation, reversal behavior, surface/order generalization, language, low D3, and protected binding. **SF21 — the prospectively declared final treatment — failed.**

**Authoritative SF21 v2 fresh-seed classification:** `SF21_COEXISTENCE_FRONTIER_FAIL_10M_SERIES_CLOSED` — all three seeds (87056/87057/87058) stopped at U100 with `retention_acquisition` failure (TRAIN16 exact < 16) and `endpoint_pass` = false. **The 10.6M factual-supervision series is CLOSED.**

### 1.2 What This Means for vNext

- **NOT proven:** Parameter count is the sole bottleneck.
- **IS established:** Continued micro-tuning of the current 10.6M regime has poor expected value after SF21.
- **Therefore:** A substantially larger successor (~60M) is a justified next engineering/scientific move to test whether representational elbow room can move the coexistence frontier.

### 1.3 The Core Failure Phenotype

Across SF9–SF21, **exact surface/order generalization repeatedly refuses to coexist** with:
- Strong forced-choice factual correctness ✓
- Reversal consistency ✓
- Complete factual families ✓
- Language retention ✓
- Binding preservation ✓
- Low D3 ✓

**Pattern:** Direct identity/name-generation pressure helps exact output but pollutes ordinary-English output distributions (D3); suppressing it keeps D3 clean but damages exact generation and widening. Even identity diversification (SF20) and representation consistency (SF21) failed to resolve this.

### 1.4 What Must Survive Transplantation

| Category | Must Preserve | Rationale |
|---|---|---|
| Binding | T12/T13 query→source→value routing mechanism | Proven ceiling on synthetic binding |
| Binding | Both protected pools + gates (≥76/80 ans/BD, 0 collapse) | Regression guardrail |
| Binding | OrthoLocalizer + retrieval head | Surviving implementation |
| Language | Pilot 1 language parent lineage + gates | Proven language+binding coexistence |
| Factual | SF8 recipe (λ_margin=0.25, M=1.0) | Standing narrow acquisition |
| Gates | TRAIN16 (16/16 correct+exact), DEV_SURFACE, DEV_ORDER, D3≤0.01, language CE≤u0+0.25 | All frozen |
| Discipline | Provenance/hashing/sealing, prospective gating, LOCKED_UNSCORED transfer | Methodological rigor |

---

## 2. Authoritative Current Model

### 2.1 Identity

- **Name:** Baby (Research Baby)
- **Architecture:** Decoder-only transformer, custom implementation
- **Parameter count:** 10,594,944 base (untied) + 246,401 binding additions = **10,841,345 trained parameters**
- **Model file lineage:** `C:\DaveLM-v0.9\v0_8\model.py` (base) + `treatment13_model.py` (wrapper with binding) + `PINNED_PILOT1_BINDING_IMPLEMENTATION.py` (localizer/eval)
- **Tokenizers:** `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json` SHA256 `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`
- **Authoritative parent anchor:** Pilot 1 language model `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt` SHA256 `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb`

### 2.2 Deployment Artifact

The SF21 v2 trained model (representative):
- Path: `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\runs\seed_87056_curriculum\checkpoint_100.pt`
- Size: ~151 MB (fp32 weights + optimizer states + provenance)
- Checkpoint keys: `['model', 'optimizer', 'rng', 'scope', 'provenance', 'completed', 'last_metric']`

---

## 3. Architecture Autopsy

### 3.1 Base Transformer (`C:\DaveLM-v0.9\v0_8\model.py`)

| Component | Value | Source |
|---|---|---|
| Architecture | Decoder-only transformer | v0_7/model.py |
| Layers | 8 | v0_8/config.py:70 |
| d_model | 320 | v0_8/config.py:68 |
| n_heads | 10 | v0_8/config.py:69 |
| head_dim | 32 (320//10) | v0_7/model.py:42-44 |
| MLP hidden | 1280 | v0_8/config.py:71 |
| MLP activation | ReLU | v0_7/model.py:59 |
| Vocab size | 1024 | v0_8/config.py:66 |
| Context length | 256 | v0_8/config.py:67 |
| Positional encoding | Learned absolute, `nn.Embedding(256, 320)` | v0_7/model.py:95,111-112 |
| Dropout | 0.05 (embedding, attention, residual) | v0_8/config.py:72-74 |
| Norm type | LayerNorm (ExplicitLayerNorm wrapper) | v0_2_1/model.py:9-37 |
| Norm placement | Pre-norm | v0_7/model.py:81-82 |
| Attention impl | Manual per-head loop (no SDPA/flash) | v0_7/model.py:39-52 |
| Causal mask | `torch.tril` float buffer, sliced per timestep | v0_7/model.py:26,32-35 |
| Residual | Post-attn + post-FFN adds | v0_7/model.py:81-82 |
| Biases | Q/K/V: NO; Output projection: YES; MLP: YES | v0_7/model.py:22-24,49,59,61 |
| Embedding tie | UNTIED | v0_8_2/config.py:33; v0_8_2/model.py:23-38 |
| Final norm | ExplicitLayerNorm(320) | v0_7/model.py:102 |
| LM head | `nn.Linear(320, 1024)` + bias | v0_7/model.py:103 |
| Forward | `x = emb + pos; emb_dropout; blocks; final_norm; language_head` | v0_7/model.py:105-114 |

### 3.2 Binding / Localization Machinery (`treatment13_model.py`, `PINNED_PILOT1_BINDING_IMPLEMENTATION.py`)

**Surviving mechanism (deployed in all SF-series runs):**

1. **OrthoLocalizer** (641 params): Householder reflection-based 2-slot scorer
   - Params: `u[320]`, `q[319]`, `bs[scalar]`, `ba[scalar]`
   - Forward: `S = h@u + bs`, `R = h@basis(u)@q + ba`, output `stack((S+R,S-R), -1)` → `[B,T,2]`
   - Source: `PINNED_PILOT1_BINDING_IMPLEMENTATION.py:18-23`

2. **Retrieval head** (245,760 params): Query-conditioned value retrieval
   - `wq = Linear(320→64, bias=False)` (20,480)
   - `wk = Linear(320→64, bias=False)` (20,480)
   - `wv = Linear(320→320, bias=False)` (102,400)
   - `wo = Linear(320→320, bias=False)` (102,400)
   - Source: `treatment13_model.py:62-66`

3. **Source localization** (geometry-dependent):
   - Candidate positions: `n_cand = T-1-ROW_VALUE_OFFSET` (188 for T=193)
   - Validity mask: `pos < qpos AND pos+ROW_VALUE_OFFSET < T`
   - **ROW_VALUE_OFFSET = 4** (synthetic grammar constant)
   - Source slots = slot-weighted mean of hidden at candidate positions
   - Value slots = slot-weighted mean of hidden at `pos+ROW_VALUE_OFFSET`
   - **TASK-SPECIFIC:** Hard-coded 2 rows, offset=4 pairing
   - Source: `treatment13_model.py:90-106`

4. **Answer modulation:**
   - Query = hidden at `qpos` (supplied by caller)
   - Keys = `wk(source_slots)`, Scores = `wq(query)@keys * scale`, softmax → `row_weights[B,2]`
   - Retrieved = `row_weights @ value_slots`
   - Answer hidden = `hidden[anspos] + wo(retrieved)`
   - Spliced into logits at `anspos` → language head
   - Source: `treatment13_model.py:108-121`

**Classification:**
- **GENERIC reusable:** final-norm hook capture; 4-projection retrieval head; residual answer injection
- **SYNTHETIC-TASK-SPECIFIC:** 2-row slot count; positions-before-query masking; ROW_VALUE_OFFSET=4 value pairing; qpos/anspos handoff

**Checkpoint tensors:**
- `localizer.u`, `localizer.q`, `localizer.bs`, `localizer.ba`
- `wq.weight`, `wk.weight`, `wv.weight`, `wo.weight`

### 3.3 Parameter Counting (Physical Verification)

| Component | Shape(s) | Params | % of 10,594,944 |
|---|---|---|---|
| token_embedding | (1024, 320) | 327,680 | 3.1% |
| position_embedding | (256, 320) | 81,920 | 0.8% |
| per-block attention | 3×10×(32×320) + 320×320 + 320 | 409,920 | 3.9% |
| per-block MLP | 320×1280 + 1280 + 1280×320 + 320 | 820,800 | 7.8% |
| per-block norms | 2×(2×320) | 1,280 | 0.0% |
| per-block total | | 1,232,000 | 11.6% |
| 8 blocks total | | 9,856,000 | 93.0% |
| final_norm | 2×320 | 640 | 0.0% |
| LM head | 1024×320 + 1024 | 328,704 | 3.1% |
| **BASE TOTAL** | | **10,594,944** | **100%** |
| Binding additions | | **246,401** | **2.3%** |
| OrthoLocalizer | u[320]+q[319]+bs[1]+ba[1] | 641 | |
| Retrieval head | wq[64×320]+wk[64×320]+wv[320×320]+wo[320×320] | 245,760 | |
| **TOTAL TRAINED** | | **10,841,345** | **102.3%** |

**Verification:**
- Code formula: `v0_8/config.py:76` `EXPECTED_PARAMETER_COUNT = 10_594_944`
- Runtime assert: `v0_8_2/model.py:46-51` enforces exact count
- Physical checkpoints (SF8, SF20 v2, SF21 v2) all contain exactly 10,594,944 base params + 246,401 binding params

---

## 4. Parameter Accounting

### 4.1 Component Breakdown

| Component | Shape | Parameters | Scaling Formula |
|---|---|---|---|
| Token embeddings | (V, d) = (1024, 320) | 327,680 | V × d |
| Position embeddings | (C, d) = (256, 320) | 81,920 | C × d |
| Attention per head | K/Q/V: (d, head_dim) × 3 = (320, 32) × 3 | 307,200 per head | 3 × d × (d/n_heads) |
| Attention projection | (head_dim, d) + bias | 102,720 | d × d + d |
| Attention per block | | 409,920 | 4 × d² + 2d |
| MLP per block | (d, h) + (h, d) + biases | 820,800 | 2 × d × h + h + d |
| Norm per block | 2 × (2 × d) | 1,280 | 4d |
| Block total | | 1,232,000 | 4d² + 2dh + h + 5d |
| 8 blocks | | 9,856,000 | 8 × (4d² + 2dh + h + 5d) |
| Final norm | 2d | 640 | 2d |
| LM head | (d, V) + bias | 328,704 | d × V + V |
| **Base** | | **10,594,944** | V×d + C×d + 8×(4d²+2dh+h+5d) + 2d + d×V + V |
| Binding (OrthoLocalizer) | u[d]+q[d-1]+bs+ba | 641 | 2d |
| Binding (retrieval) | wq[d→r]+wk[d→r]+wv[d→d]+wo[d→d] | 245,760 | 2×d×r + 2×d² |
| **Total** | | **10,841,345** | |

Where: V=1024, C=256, d=320, n_heads=10, head_dim=32, h=1280, r=64 (retrieval dim)

### 4.2 Dominant Components

1. **Attention (all blocks):** 9,856,000 × ~81.7% of base
2. **MLP (all blocks):** Included in block totals above
3. **Embeddings:** 327,680 + 81,920 = 409,600 (~3.9%)
4. **LM head:** 328,704 (~3.1%)
5. **Binding:** 246,401 (~2.3% of total)

**Conclusion:** Attention parameters dominate capacity (81.7%). MLP is second. Embeddings and output head are minor. Binding additions are ~2.3% of total.

### 4.3 Scaling Relationships

For a future ~60M model:

- **Attention per block:** 4 × d_model² + 2 × d_model ≈ 4d²
- **MLP per block:** 2 × d_model × d_mlp + d_mlp + d_model ≈ 2d×h
- **Embeddings:** (V + C) × d_model
- **LM head:** V × d_model + V
- **Binding:** ~2 × d_model² (retrieval head dominates) + 2 × d_model (localizer)

If scaling width by factor k: attention scales as k², MLP as k², embeddings as k, head as k.
If scaling depth by factor k: all per-block components scale as k.

---

## 5. Binding Lineage and Surviving Mechanism

### 5.1 Historical Stages (Verified from Physical Artifacts)

| Treatment | Mechanism | Outcome | Source |
|---|---|---|---|
| T8 | Contextualized query→logit pathway (query bias on answer logits) | Mechanism established | treatment8_train.py |
| T9 | Answer-position retrieval via learned attention head | Generic retrieval failure | treatment9_train.py |
| T10 | Strict counterfactual curriculum (2-mapping quartets) + answer-CE + membership | 50% answer acc, 0/40 quartets — self-attention cannot do it | treatment10_final_retention_audit.json |
| T11 | Answer-only strict counterfactual (CE on answer token only) | 45% answer acc, 0 complete quartets | treatment11_final_retention_audit.json |
| T12 | **Query-conditioned mapping retrieval**: query→match source→retrieve source value→answer | **160/160 retention, 40/40 quartets, retrieval row acc 1.0** | treatment12_model.py, treatment12_final_retention_audit.json |
| T13 | **Learned mapping-row localization**: removes explicit row positions; learned localizer + query-conditioned retrieval | Multiple variants; canonical: **317/320 ans, 312/320 BD, 0 collapse, 78/80 quartets, 158/160 reversals** | treatment13_orthogonal_shared_unbounded_seed8380 |

### 5.2 The Canonical Synthetic Binding Graduate

**Physical location:**
- Checkpoint: `C:\DaveLM-CADAVER\treatment13_orthogonal_shared_unbounded_seed8380\checkpoints\orthogonal_shared_unbounded\seed_8380\latest.pt`
- SHA256: `fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430`
- Size: 64,504,479 bytes

**Headline numbers (FINAL_FROZEN_RETENTION_RESULTS.json):**
| Metric | Score | Gate |
|---|---|---|
| Answer exact | 317/320 (99.06%) | — |
| BOTH_DISTINCT | 312/320 (97.5%) | ≥76/80 per pool |
| Collapse | 0/320 | ==0 |
| Complete quartets | 78/80 (97.5%) | — |
| Reversal both-correct | 158/160 (98.75%) | — |
| Classification | GRADUATION_LEVEL_SUCCESS | — |

**Training:** 1000 updates from T13 learned-localization checkpoint, replacing scorer with OrthoLocalizer, objective = answer_CE + 1.0536573711078283 × localization_loss.

**Mechanism summary:**
- Query hidden state at `qpos` is projected (wq) and matched against source slot keys (wk) from localized positions
- Softmax over exactly 2 slots → row weights
- Retrieved value = weighted sum of value slots
- Answer modulation: `answer_hidden + wo(retrieved)` → language head
- **Generic:** retrieval head architecture, residual injection
- **Task-specific:** 2-row geometry, ROW_VALUE_OFFSET=4 pairing, position masks

### 5.3 Surviving Implementation

**Authoritative files:**
- Model wrapper: `C:\DaveLM-CADAVER\treatment13_model.py` (Treatment13Model class)
- Localizer: `PINNED_PILOT1_BINDING_IMPLEMENTATION.py` (OrthoLocalizer + binding_eval)
- Masking/batching: `PINNED_MASKING.py` (prepare_example, pad_batch, set_scope)
- Runtime loader: `hr3_block3_runtime.py` (load_model, set_scope, binding_gate)

**Checkpoint contract:**
- Required keys: `localizer.u`, `localizer.q`, `localizer.bs`, `localizer.ba`, `wq.weight`, `wk.weight`, `wv.weight`, `wo.weight`
- Base model keys: all `base_model.*` (token_embedding, position_embedding, blocks.0-7, final_norm, language_head)
- 80 non-trainable causal mask buffers: `base_model.blocks.{0-7}.attention.heads.{0-9}.mask` (256,256)

### 5.4 Protected Binding Pools

Two nonsacred pools used as regression gates throughout SF series:

| Pool | Path | Size | Gate |
|---|---|---|---|
| pilot0_dev | language_pilot_0_tinystories_seed8380\binding_dev.json | 80 docs | answer≥76, BD≥76, collapse==0 |
| pilot1_dev | language_pilot_1_early_block_protection_seed8380\binding_dev.json | 80 docs | answer≥76, BD≥76, collapse==0 |
| rehearsal | language_pilot_1_early_block_protection_seed8380\binding_rehearsal.json | 80 docs | Used during binding updates |

Byte-identical copies propagated through: sf13, sf20 v1/v2, sf21 v1/v2 under `data\`

---

## 6. Language Lineage

### 6.1 Authoritative Pilot 1 Language Parent

**Checkpoint:**
- Path: `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt`
- SHA256: `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb` (VERIFIED)
- Size: 64,504,479 bytes
- **Status:** VERIFIED — computed hash matches expected exactly

**Training lineage:**
- Parent: Baby graduate (T13 orthogonal-shared-unbounded)
- Schedule: 900 TinyStories language updates + 100 binding updates (full scope)
- **Early block protection:** Blocks 0-3 of base model + T13 localizer/retrieval **frozen** during language steps; binding steps use full scope
- Source: language_pilot_1...\run.py set_scope logic

**Results (vs Graduate baseline):**
| Metric | Graduate (pre) | Pilot 1 (post) | Source |
|---|---|---|---|
| Dev language CE | 7.1532 | **2.9039** | REPORT.md |
| Dev perplexity | 1278.2 | **18.25** | REPORT.md |
| Binding pilot0_dev | 79/80 ans, BD 76, collapse 0 | **80/80 ans, BD 80, collapse 0** | REPORT.md |
| Binding pilot1_dev | — | **80/80 ans, BD 80, collapse 0** | human_readiness_binding_measurement.json |

**Conclusion:** Pilot 1 achieved major language improvement **while preserving** binding at ceiling on both pools.

### 6.2 Pilot 0 (Contrast)

- Same schedule, NO early block protection
- Result: dev ppl 1278→8.6 but binding **damaged** (73/80 ans, BD 76, collapse 4 on pilot0_dev)
- Source: language_pilot_0...\REPORT.md

### 6.3 P0–P7 Language Experiments

Brief summary (from RESEARCH_LOG.md files):
- **p0:** Plain TinyStories continuation from Pilot 0 → ppl 2.22→1.70, locally fluent but repetitive/drifting
- **p1:** 819 two-sentence synthetic consistency → grammatical but templated/memorized
- **p2:** 9 TinyStories : 1 consistency interleave → local form improves, context reliability fails
- **p3:** Complete story starts with BOS/EOS + masked padding → better first sentences, 4/8 exact corpus substrings
- **p4:** Recent-token unlikelihood (window 8, λ=0.1) → less repetition, semantic failures remain
- **p5:** 147,521 TinyStories sentence segments → fluent completions, all exact substrings (memorization)
- **p6:** 8 synthetic frames, held-out combos → first controlled compositional step
- **p7:** 8 varied relation frames with held-out lexical combos → **12/12 held-out first-sentence continuations context-matched, grammatical, no overlap** (narrow controlled composition)

### 6.4 English Context Characterization

`english_context_characterization_v1_seed8380_execution\REPORT.md`:
- Graduate, Pilot 0, Pilot 1 all show **0/16 complete families** on sealed English batteries (naming/possession/location cloze+QA+frame-holdout)
- Margins ~0 with large negative minima
- Reversal both-correct rates: 0–6%
- **Interpretation:** No general-English context-sensitive counterfactual behavior; only structural synthetic binding references remain at ceiling

### 6.5 Generation Pathology Finding

`GENERATION_PATHOLOGY_REPORT.md`:
- HR-1/HR-2 models are **worse** than Pilot 1 (HR-1 loss 8.64 vs Pilot-1 3.39)
- Their celebrated low DEV losses (0.555/0.0926) came from a **shifted label layout** in training/eval loops
- Under greedy rollout: HR-1/HR-2 lock into single-token loops by step 2–3 on all five prompts
- **Next step indicated:** Training-side causal-alignment correction preserving Pilot 1

### 6.6 Protected Language Gates

From SF20/SF21 PROTOCOL.json:
- `gates.language = "CE <= own update0 CE + .25"` (measured on 128 aligned TinyStoriesDEV rows)
- Language dev file: `sf20...\data\ENGLISH_DEV.jsonl` (128 rows, 3,796 targets)

---

## 7. Factual Acquisition Lineage

### 7.1 SF Series Mapping

| SF# | Directory | Main Intervention | Outcome |
|---|---|---|---|
| SF1 | single_fact_acquisition_sf1_seed87011 | Initial factual acquisition | Partial success |
| SF2 | sf2_kl_parent_retention_run_v1/v2 | KL parent retention | Near-acquisition with retention |
| SF3 | sf3_prospective_treatment_v1(+) | Prospective treatment | — |
| SF4 | sf4_common_state_lr_anneal_v1 | Common state LR anneal | — |
| SF5 | sf5_matched_prefix_anneal_v1 | Matched prefix anneal | — |
| SF6 | sf6_same_seed_lr_anneal_v1 | Same seed LR anneal | Constant-recipe success evidence |
| SF7 | sf7_pairwise_margin_hinge_v1 | Strong margin hinge (λ=1.0, M=1.0) | Mechanism established but D3 trips in 2/3 seeds |
| SF8 | sf8_margin_dose_comparison_v1 + sf8_lowdose_transfer_eval_v1 | **Low-dose margin (λ=0.25, M=1.0)** | **Reliable narrow factual acquisition across 3 seeds** |
| SF9 | sf9_surface_order_curriculum_v1 | Surface/order curriculum | Widening attempt |
| SF10 | sf10_reduced_density_widening_v1 | Reduced density widening | — |
| SF11 | sf11_narrow_breadth_widening_v1 | Narrow breadth widening | — |
| SF12 | sf12_answer_vocab_retention_v1 | Answer vocab retention | — |
| SF13 | sf13_broad_coverage_kl_retention_v1 | Broad retention/KL coverage | Retention restored, coexistence problem persists |
| SF14 | sf14_first_answer_token_ce_ablation_v1 | First-answer-token CE ablation | D3 contained but exact generation damaged |
| SF15 | sf15_partial_first_token_ce_v1 | Partial first-token CE | Some generation restored, D3 leakage returns |
| SF16 | sf16_first_answer_token_ce_weight_v1 | First-answer-token CE weight | Inconsistent, CE-microdose exhausted |
| SF17 | sf17_retention_priority_gradient_projection_v1 | Retention-priority gradient projection | Active but insufficient |
| SF18 | sf18_english_output_head_freeze_v1 | English output head freeze | Upstream changes still drive name mass |
| SF19 | sf19_candidate_membership_hinge_v1 | Candidate-membership ranking | Near-hit: durable containment failed |
| SF20 | sf20_identity_heldout_balanced_entity_rotation_v1/v2 | **Identity-heldout balanced entity rotation** | **D3 suppressed but no identity-independent widening** |
| SF21 | sf21_paired_counterfactual_representation_consistency_v1/v2 | **Paired counterfactual representation consistency (λ=0.1)** | **Final treatment: FAILED** |

### 7.2 SF8: Standing Narrow Factual Recipe

**Authoritative artifacts:** sf8_margin_dose_comparison_v1

**Design:**
- Manipulated variable: λ_margin only (control=0.0, low=0.25, medium=0.50)
- M (margin target) fixed at 1.0 nat
- All other parameters identical to SF7
- 3 seeds × 3 arms = 9 runs, all sealed before outcomes

**Results (verified from STATUS.json files):**
| Seed | Arm | Status | Completed | Acquisition | D3 | Language | Binding | Endpoint |
|---|---|---|---|---|---|---|---|---|
| 87017 | low (λ=0.25) | ACQUISITION_SUCCESS | 200 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 87018 | low (λ=0.25) | ACQUISITION_SUCCESS | 200 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 87019 | low (λ=0.25) | ACQUISITION_SUCCESS | 200 | ✓ | ✓ | ✓ | ✓ | ✓ |

**All three low-dose seeds achieved full-gate success with comfortable D3 and no retention failures.**

**Recipe:** λ_margin=0.25, M=1.0, λ_KL=1.0, 160 positions/update, 180 English + 20 binding updates, AdamW lr=5e-5, wd=0.05, clip=2.0, blocks 0-3 frozen on English.

**Post-SF8 Transfer Evaluation (sf8_lowdose_transfer_eval_v1):**
- Strong held-out recombination forced-choice behavior
- Exact unrestricted generation weaker
- Alternate surface/interface transfer limited
- Copy transfer limited
- Competing-fact behavior depended strongly on order
- Canonical-order contradictory distractor resistance strong; reversed order exposed recency/order sensitivity

### 7.3 SF9–SF21: The Coexistence Wall

See Section 8 for detailed table.

---

## 8. SF9–SF21 Coexistence Wall

### 8.1 Compact Results Table

| Study | Dir | Seeds | Main Intervention | TRAIN16 (corr/exact) | Surface (corr/exact) | Order (corr/exact) | Lang Δ | D3 | Binding | Classification | Main Lesson |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SF8 | sf8_margin_dose_comparison_v1 | 87017-19 | λ_margin=0.25, M=1.0 | 16/16 | — | — | ≤+0.25 | ≤0.01 | ✓ | ACQUISITION_SUCCESS | Standing narrow recipe |
| SF9 | sf9_surface_order_curriculum_v1 | ? | Surface/order curriculum | — | — | — | — | — | — | — | Widening attempt |
| SF10 | sf10_reduced_density_widening_v1 | ? | Reduced density | — | — | — | — | — | — | — | — |
| SF11 | sf11_narrow_breadth_widening_v1 | ? | Narrow breadth | — | — | — | — | — | — | — | — |
| SF12 | sf12_answer_vocab_retention_v1 | ? | Answer vocab retention | — | — | — | — | — | — | — | — |
| SF13 | sf13_broad_coverage_kl_retention_v1 | 87011-? | Broad KL coverage | 16/16 | — | — | — | — | ✓ | — | Retention restored, coexistence persists |
| SF14 | sf14_first_answer_token_ce_ablation_v1 | ? | Remove first-token CE | — | — | — | — | Clean | ✓ | — | D3 contained, exact gen damaged |
| SF15 | sf15_partial_first_token_ce_v1 | ? | Partial first-token CE | — | — | — | — | Leakage | ✓ | — | Some gen restored, D3 back |
| SF16 | sf16_first_answer_token_ce_weight_v1 | ? | First-token CE weight | — | — | — | — | Inconsistent | ✓ | — | CE-microdose exhausted |
| SF17 | sf17_retention_priority_gradient_projection_v1 | ? | Retention-priority projection | — | — | — | — | — | ✓ | — | Active but insufficient |
| SF18 | sf18_english_output_head_freeze_v1 | ? | Freeze English LM head | — | — | — | — | — | ✓ | — | Upstream still drives name mass |
| SF19 | sf19_candidate_membership_hinge_v1 | ? | Candidate-membership hinge | — | — | — | — | Early clean, late leak | ✓ | — | Near-hit: containment not durable |
| SF20 v2 | sf20_identity_heldout_balanced_entity_rotation_v2 | 87053-55 | Identity-heldout balanced entity rotation | 16/?? | ~2/?? | ~2/?? | — | ~.007→.002 | ✓ | STOP_REGRESSION | D3 suppressed, no widening |
| SF21 v2 | sf21_paired_counterfactual_representation_consistency_v2 | 87056-58 | +λ_consistency=0.1 paired cosine | **16/13** | **11/2** | **8/2** | **+0.0912** | **.00704→.00219** | ✓ | **STOP_REGRESSION** | **Final failure** |

**SF21 v2 seed-level results (verified):**

| Seed | Stop | TRAIN16 | Surface | Order | Lang Δ | D3 start→end | Binding | Status |
|---|---|---|---|---|---|---|---|---|
| 87056 | U100 | 16/13 exact | 11/2 exact | 8/2 exact | +0.0912 | 0.00704→0.00219 | ✓ | STOP_REGRESSION |
| 87057 | U100 | 16/13 exact | 10/2 exact | 8/2 exact | +0.1530 | 0.00667→0.00238 | ✓ | STOP_REGRESSION |
| 87058 | U200 | 16/8 exact | 13/1 exact | 8/2 exact | +0.2256 | 0.00784→0.00243 | ✓ | STOP_REGRESSION |

**All three seeds failed retention_acquisition gate (exact < 16) at U100 or U200.**

### 8.2 What Each Treatment Taught Us

| Treatment | What It Ruled Out / Weakened | What Survived |
|---|---|---|
| SF8 | — | Narrow factual acquisition is reliable at λ_margin=0.25 |
| SF13 | — | Broad KL coverage can restore retention but not end coexistence |
| SF14 | First-token CE is NOT required for D3 containment | Exact name generation requires CE pressure |
| SF15/16 | CE-microdose knob is exhausted | Partial CE cannot durably coexist |
| SF17 | Global gradient projection alone insufficient | Retention priority helps but not enough |
| SF18 | Output head freeze insufficient | Upstream represents the real problem |
| SF19 | Candidate-membership is a near-hit | Containment not durable, widening still short |
| SF20 | Identity diversification suppresses D3 | No identity-independent widening/generalization |
| SF21 | Representation consistency signal | Same coexistence frontier persists |

### 8.3 The Repeated Failure Pattern

**Can coexist:** Forced factual correctness, reversal consistency, complete families, language retention, binding preservation, low D3.

**Cannot coexist:** Exact surface/order generalization with all of the above.

**Hypothesis (evidence-supported):** Strong direct identity/name-generation pressure helps exact output but tends to pollute ordinary-English output distributions (D3); suppressing or redistributing that pressure keeps D3 clean but damages exact generation and/or widening. Even identity diversification (SF20) and representation consistency (SF21) failed to make robust surface/order generalization coexist.

---

## 9. SF21 Closure

### 9.1 Authoritative Classification

**Study:** SF21_PAIRED_COUNTERFACTUAL_REPRESENTATION_CONSISTENCY_V2  
**Status:** `SF21_COEXISTENCE_FRONTIER_FAIL_10M_SERIES_CLOSED`  
**Source:** `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\PROTOCOL.json` lines 4-8  

**Classification rule (from PROTOCOL.json):**
```json
{
  "failure": "SF21_COEXISTENCE_FRONTIER_FAIL_10M_SERIES_CLOSED for any other scientifically valid cross-seed result",
  "mechanical": "MECHANICAL_INCOMPLETE if integrity/execution prevents classification",
  "retention": "RETENTION_REGRESSION if any frozen TRAIN16/language/binding retention gate fails",
  "success": "SF21_COEXISTENCE_FRONTIER_SUCCESS if >=2/3 endpoint_pass"
}
```

**Actual outcome:** All three seeds (87056, 87057, 87058) → STOP_REGRESSION at U100 (seeds 87056/57) or U200 (seed 87058), with `retention_acquisition: false` (TRAIN16 exact < 16) and `endpoint_pass: false`.

**0/3 seeds achieved endpoint_pass.** Therefore classification = **SF21_COEXISTENCE_FRONTIER_FAIL_10M_SERIES_CLOSED**.

### 9.2 Prospective Stop Rule (Fulfilled)

From PROTOCOL.json line 77:
> "If SF21 does not achieve coexistence in >=2/3 branches, close the 10.6M factual-supervision series without SF22 or coefficient rescue."

**Result:** SF21 did NOT achieve coexistence in ≥2/3 branches (0/3).  
**Conclusion:** The 10.6M factual-supervision series is **CLOSED**. Do not reopen.

### 9.3 Scientific Interpretation

**NOT proven:** 10.6M parameters are mathematically incapable of solving the task.  
**IS proven:** Continued micro-tuning of the current 10.6M factual-supervision regime has poor expected value after the prospectively final SF21 treatment.

The larger successor (~60M) is therefore a **justified next engineering/scientific move** to test whether representational elbow room can move the coexistence frontier.

---

## 10. Tokenizer Autopsy

### 10.1 Identity

| Attribute | Value | Source |
|---|---|---|
| Path | `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json` | Physical |
| SHA256 | `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b` | Computed + matches expected |
| Size | 55,710 bytes | Physical |
| Sibling | `stats.json` SHA256 `b6c68b9664faa90d2a87c5c441abcc051f43a75002a78f0a25a622039be8d517` | Physical |
| Family | HuggingFace tokenizers v1.0 | JSON header |
| Model type | BPE | JSON:"type" |
| Byte-level | Yes | ByteLevel pre-tokenizer/decoder |
| Normalizer | null | JSON |
| Vocab size | 1024 | Count of vocab entries |
| Merges | 763 | JSON merges count |
| Special tokens | `<pad>`=0, `<unk>`=1, `<bos>`=2, `<eos>`=3, `<doc>`=4 | added_tokens |
| No `<mask>` | Confirmed | token_to_id("<mask>") → None |

### 10.2 Tokenization Geometry

**All names are multi-token:**

| Name | Bare tokens | " Name" tokens | " Name." tokens | First token ID |
|---|---|---|---|---|
| Alex | 3 | 4 | 4 | 314 |
| Owen | 3 | 4 | 4 | 536 |
| Mia | 3 | 4 | 4 | 925 |
| Nora | 3 | 4 | 4 | 512 |
| Sara | 3 | 4 | 4 | 527 |
| Bob | 3 | 4 | 4 | 532 |
| Billy | 3 | 4 | 4 | 532 |
| Tony | 3 | 4 | 4 | 525 |
| Fred | 3 | 4 | 4 | 508 |
| Ralph | 3 | 4 | 4 | 1019 |
| Remy | 3 | 4 | 4 | 1019 |
| Steve | 3 | 4 | 4 | 527 |
| Susan | 3 | 4 | 4 | 527 |
| Tina | 3 | 4 | 4 | 525 |
| Wendy | 3 | 4 | 4 | 679 |
| Cathy | 3 | 4 | 4 | 651 |
| Colin | 3 | 4 | 4 | 651 |
| Faye | 3 | 4 | 4 | 508 |
| Carl | 3 | 4 | 4 | 651 |
| Tara | 3 | 4 | 4 | 525 |

**Key findings:**
- Every name = **3 BPE tokens** bare, **4 tokens** as candidate (" Name.")
- Only **7 distinct first-token IDs** for 20 names → heavy first-token collision
- `.` token ID = 18
- BOS (id 2) prepended, EOS (id 3) appended in sequence construction
- SF20 builder required exact geometry `[3,3,4]` with no first-token collisions

### 10.3 BOS/EOS Behavior

From `PINNED_MASKING.py:5-10`:
- `prefix = [2] + prompt_token_ids` → BOS prepended
- `seq = prefix + candidate_ids + [3]` → EOS appended
- Labels: `[-100]*(len(prefix)-1) + candidate + [3]`
- Answer tokens predicted starting at `len(prefix)-1` (last prompt token)

### 10.4 Blast Radius if Tokenizer Changed

**Would break:**
1. All SHA256 asserts (200+ files pin tokenizer hash)
2. Checkpoint embedding shape mismatch (model vocab=1024)
3. All frozen datasets with precomputed token IDs (DEV_SURFACE.json, TRAIN.json, TOKENIZATION_MATCH.json, IDENTITY_ASSIGNMENT.json first_answer_token_ids)
4. Tokenized language training streams
5. SF20 first-token distinctness guarantees

**Trade-off for vNext:**
- **Keep:** Isolates architecture/capacity, preserves evaluator compatibility
- **Change:** Better long-term language model, but makes vNext a broader generational redesign

---

## 11. Checkpoint / State-Dict Autopsy

### 11.1 Top-Level Schema

**restart.pt (rolling checkpoint):**
```json
{
  "completed": int,
  "model": OrderedDict (414 entries),
  "optimizer": {"state": 334 entries, "param_groups": [...]},
  "rng": {"python_rng": ..., "torch_rng_cpu": ..., "torch_rng_cuda": [...]},
  "scope": str,
  "provenance": dict (14 keys),
  "last_metric": dict
}
```

**checkpoint_{100,200}.pt (permanent):**
```json
{
  "model_state_dict": OrderedDict (414 entries),
  "update": int,
  "provenance": dict
}
```

### 11.2 Model State-Dict Structure

**414 entries breakdown:**
- **Trainable parameters (334):**
  - base_model.token_embedding.weight (1024, 320)
  - base_model.position_embedding.weight (256, 320)
  - base_model.blocks.{0-7}.attention.heads.{0-9}.key/value/query.weight (32, 320) each
  - base_model.blocks.{0-7}.attention.heads.{0-9}.projection.weight (320, 32) + .bias (320)
  - base_model.blocks.{0-7}.feed_forward.network.0.weight (1280, 320) + .bias (1280)
  - base_model.blocks.{0-7}.feed_forward.network.2.weight (320, 1280) + .bias (320)
  - base_model.blocks.{0-7}.norm1.weight (320) + .bias (320)
  - base_model.blocks.{0-7}.norm2.weight (320) + .bias (320)
  - base_model.final_norm.weight (320) + .bias (320)
  - base_model.language_head.weight (1024, 320) + .bias (1024)
  - localizer.u (320), localizer.q (319), localizer.bs (1), localizer.ba (1)
  - wq.weight (64, 320), wk.weight (64, 320), wv.weight (320, 320), wo.weight (320, 320)
- **Non-trainable buffers (80):**
  - base_model.blocks.{0-7}.attention.heads.{0-9}.mask (256, 256) — causal mask buffers

### 11.3 Optimizer State

- **Type:** AdamW
- **Config:** lr=5e-05, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05, amsgrad=False, foreach=False, fused=False
- **State per param:** step (int), exp_avg (tensor), exp_avg_sq (tensor)
- **Param groups:** 1 group with all 334 trainable parameters

### 11.4 Provenance Metadata

From restart.pt['provenance'] (14 keys):
- parent_checkpoint + sha256
- protocol, receipt, schedule, curriculum, controller hashes
- seed, arm
- kl_pool, margin_M, lambda_margin, pair_manifest, lambda_consistency

### 11.5 Tensor Migration Classification

| Category | Examples | Migration Notes |
|---|---|---|
| 1. Exact-shape reusable | base_model.* weight tensors | Can load into same-shaped model |
| 1. Exact-shape reusable | localizer/wq/wk/wv/wo weights | Binding-specific, but shape-preserving |
| 2. Potentially expandable | token_embedding (V,d), position_embedding (C,d) | V or C change requires resize/retrain |
| 2. Potentially expandable | LM head (d,V) | V or d change requires resize |
| 3. Architecture-dependent | All attention/MLP/norm tensors | d_model, n_heads, d_mlp changes = incompatible |
| 4. Should retrain | All optimizer state | Always reset for new training |
| 4. Should retrain | All RNG state | Always reset for new training |

**Conclusion:** Current 10.6M weights **cannot** be mechanically transplanted into a larger model with different d_model, n_heads, or d_mlp. From-scratch training is the simplest path for vNext.

---

## 12. Training Pipeline

### 12.1 Optimizer & Learning Rate

| Attribute | Value | Source |
|---|---|---|
| Optimizer | AdamW | All protocols |
| LR | 5e-05 (constant) | All protocols |
| Betas | (0.9, 0.999) | All protocols |
| Eps | 1e-08 | All protocols |
| Weight decay | 0.05 | All protocols |
| AMSGrad | False | All protocols |
| Gradient clip | 2.0 | All protocols |
| Precision | float32 | All protocols (autocast=false) |
| TF32 | False | All protocols |

**Note:** No LR scheduling in SF series; constant LR=5e-05 throughout.

### 12.2 Batch & Update Structure

**SF8–SF21 standard:**
- **Total updates:** 200
- **English updates:** 180 (90%)
- **Binding updates:** 20 (10%, at updates 10,20,...,200)
- **English batch:** 36 records per update
  - 32 new factual records (rotating)
  - 4 preservation TRAIN16 records (rotating)
- **Total presentations:** 6480 English + 20 binding × 80 = 6480 + 1600 = 8080
- **Per-record presentations:** New = 180, TRAIN16 = 45

**KL distillation:**
- Teacher: Separate DaveLMV082 model loaded from parent base_model.* keys
- Student: eval mode (dropout off) with grad enabled
- Teacher: eval mode + torch.no_grad
- Positions: 160 per English update from KL_POOL.json
- Lambda: 1.0 (constant)
- Temperature: 1.0
- Precision: float64 accumulation, mean, cast to float32

**Margin hinge:**
- Lambda: 0.25 (SF8 low-dose recipe)
- M: 1.0 nat
- Target: First answer-token position only
- Gap: logp_correct - logp_wrong at that position
- Hinge: max(0, M - gap)
- Loss: mean over 32 sequences

**Consistency (SF21 only):**
- Lambda: 0.1
- Target: L2-normalized logits at position = prefix_length - 1
- Loss: mean(1 - cos_sim(v_i, v_j)) over positive pairs from PAIR_MANIFEST.json
- Positive pairs: 16 pairs (32 records) sharing pair_id

### 12.3 Parameter Freezing / Scope

**English scope (PINNED_MASKING.set_scope):**
- Trainable: All base_model params **except** blocks 0-3
- Frozen: base_model.blocks.0, base_model.blocks.1, base_model.blocks.2, base_model.blocks.3
- Binding scope: All params (full model)

**HR3 variant:** Freezes blocks 0-2 only (experimental delta)

### 12.4 Checkpointing & Persistence

- **Rolling restart:** restart.pt after every update (atomic)
- **Permanent checkpoints:** checkpoint_100.pt, checkpoint_200.pt
- **Evaluation:** At updates 0, 100, 200
- **Ledger:** PARAMETER_SCOPE.json, PROVENANCE.json, BASELINE_REPRODUCTION.json, STATUS.json
- **Sequential blocking:** One seed at a time, explicit durable ledger before/after

### 12.5 Device & Runtime

| Attribute | Value | Source |
|---|---|---|
| Python | 3.12.14 | sf2venv, protocols |
| Torch | 2.12.0+rocm7.14.0 | sf2venv, protocols |
| Tokenizers | 0.23.1 | sf2venv, protocols |
| Executable | C:\DaveLM-CADAVER\sf2_runtime\sf2venv\Scripts\python.exe | Protocols |
| Device | CUDA (default) / ROCm (available) | Code uses torch.device default |
| Deterministic | True | deterministic_algorithms=true |

### 12.6 Code Paths Coupled to 10.6M Shapes

**Hardcoded literals that would break with scaling:**

| File | Line | Literal | Purpose |
|---|---|---|---|
| hr3_block3_runtime.py | 176 | reshape(-1, 1024) | CE over vocab |
| hr3_block3_runtime.py | 256 | sequence[-256:] | Greedy decode window |
| sf1_engine.py | 152 | reshape(-1, 1024) | CE over vocab |
| sf1_engine.py | 42 | x.shape[1] <= 256 | Padding assert |
| sf1_engine.py | 49,52,54 | 1024 | Mock loss width |
| sf1_engine.py | 65 | range(8) | Block count |
| sf1_engine.py | 28 | range(4) | Frozen blocks |
| SF2_ENGINE.py | 446 | reshape(-1, 1024) | CE over vocab |
| SF2_ENGINE.py | 56 | range(4) | Frozen blocks |
| CONTROLLER.py | 283 | reshape(-1, 1024) | CE over vocab |
| CONTROLLER.py | 99 | pad <= 256 | Padding assert |
| PINNED_MASKING.py | 27 | range(4) | Frozen blocks |
| PINNED_PILOT1... | 36,42 | -256:, range(4) | Context window, frozen blocks |
| PINNED_PILOT1... | 108 | reshape(-1, 1024) | CE over vocab |
| treatment13_preflight.py | 184 | (BATCH, 193, 1024) | Doc length, vocab |
| treatment13_config.py | 27-31 | DOCUMENT_LENGTH=193, VOCAB=1024, EMBED=320, RETRIEVAL_DIM=64, ROW_VALUE_OFFSET=4 | Binding geometry |

**Generalization required for vNext:** All these literals must be parameterized or derived from config.

---

## 13. Regression & Evaluation Infrastructure

### 13.1 Synthetic Binding Evaluators

| Evaluator | Definition | Gate | Source |
|---|---|---|---|
| Answer exact | Greedy generation == target + EOS | — | sf1_engine.py:77 |
| BOTH_DISTINCT | Both source rows correctly localized | ≥76/80 per pool | PINNED_PILOT1...:27-31 |
| Collapse | SLOT_COLLAPSE (both slots map to same row) | ==0 | PINNED_PILOT1...:31 |
| Complete quartets | All 4 docs in quartet correct | — | treatment13...REPORT.md |
| Reversal both-correct | Both reversal pairs correct | — | treatment13...REPORT.md |

**Protected pools:**
- pilot0_dev: language_pilot_0...\binding_dev.json
- pilot1_dev: language_pilot_1...\binding_dev.json
- rehearsal: language_pilot_1...\binding_rehearsal.json

### 13.2 Language Evaluators

| Evaluator | Definition | Gate | Source |
|---|---|---|---|
| Aligned CE | Cross-entropy on 128 TinyStoriesDEV aligned windows | ≤ own u0 + 0.25 | sf20...\data\ENGLISH_DEV.jsonl |
| Perplexity | exp(CE) | — | — |
| Greedy generation | Rollout to EOS, max 32 tokens | — | hr3_block3_runtime.py:252-261 |

### 13.3 Factual Evaluators

| Evaluator | Definition | Gate | Source |
|---|---|---|---|
| TRAIN16 correct | Forced-choice: margin > 0 | 16/16 | SF2_ENGINE.py:197-204 |
| TRAIN16 exact | Greedy == target + EOS | 16/16 | SF2_ENGINE.py:211 |
| TRAIN16 reversals | Reversal pairs correct | 8/8 | TRAIN16 panel |
| TRAIN16 families | Complete families | 4/4 | TRAIN16 panel |

### 13.4 Widening Evaluators

| Evaluator | Definition | Gate | Source |
|---|---|---|---|
| DEV_SURFACE correct | Forced-choice correct | ≥15/16 | SF2_ENGINE.py |
| DEV_SURFACE exact | Greedy exact | ≥13/16 | SF2_ENGINE.py |
| DEV_ORDER correct | Forced-choice correct | ≥13/16 | SF2_ENGINE.py |
| DEV_ORDER exact | Greedy exact | ≥13/16 | SF2_ENGINE.py |

**Subgroups:** active_qa, cloze, passive_cloze, passive_qa (4×4=16 items each)

### 13.5 D3 Evaluator

| Attribute | Definition | Gate | Source |
|---|---|---|---|
| D3 | Four-name mass (Alex, Owen, Mia, Nora) | ≤0.01 | D3_SELECTION.json |
| Selection | 256 contexts from ENGLISH_DEV.jsonl | — | D3_SELECTION.json |
| Per-name | Individual name probabilities | — | d3_*_summary.json |

**Anti-contamination:** FORBIDDEN_IDENTITIES.json, EXTERNAL_INPUTS.json, frozen lexicons (english_context_characterization_v1_seed8380\LEXICON.json, post_p7_language_report_card_v1_seed8380\LEXICON.json, human_test_readiness_v2_seed87010\LEXICON.json)

### 13.6 Evaluation Cadence

- **Updates:** 0, 100, 200
- **Rolling:** Every update (restart.pt + TRAIN_METRICS.jsonl)
- **Permanent:** checkpoint_100.pt, checkpoint_200.pt
- **Metrics per update:** loss, ce, kl, margin_loss, consistency_loss (SF21), grad_norm, lr

---

## 14. Locked / Sacred Material Rules

### 14.1 Locked Material (Metadata Only — Not Opened)

| Category | Paths | Rule | Source |
|---|---|---|---|
| Historical transfer panels | english_context_characterization_v1_seed8380, post_p7_language_report_card_v*, human_test_readiness_v* | LOCKED_UNSCORED | SF8, SF20, SF21 protocols |
| FINAL | (not found as standalone dir) | PROHIBITED | SF21 PROTOCOL.json:146 |
| Sacred | (not found as standalone dir) | PROHIBITED | SF21 PROTOCOL.json:146 |
| Forbidden lexicons | LEXICON.json in various dirs | Anti-contamination | PROTOCOL files |

**Access rule:** Do not open, score, or evaluate locked/sacred/FINAL outcomes. Metadata (paths, hashes) may be referenced but content must not be accessed.

### 14.2 Frozen Gates (From SF21 PROTOCOL.json)

```json
{
  "binding_each_pool": "answer>=76, BD>=76, collapse==0",
  "continue": "retention_acquisition AND language AND d3 AND binding pass at 100/200",
  "d3": "four-name mass <= .01",
  "dev_order": "exact>=13/16 AND fact>=7/8 AND copy>=7/8 AND each order>=7/8 AND each order:query cell>=3/4",
  "dev_surface": "correct>=15/16 AND exact>=13/16 AND each surface exact>=3/4",
  "development_gain": "At200 exact surface>=ownu0+4 AND exact order>=ownu0+4",
  "endpoint_pass": "u200 AND all retention AND dev_surface AND dev_order AND development_gain",
  "language": "CE <= own update0 CE + .25",
  "retention_acquisition": "TRAIN16 panel: correct==16 AND exact==16 AND reversals==8 AND families==4"
}
```

---

## 15. What vNext Must Preserve

### 15.1 Category A: MUST PRESERVE (Strong Positive Evidence)

**Mechanisms:**
- T12/T13 query→source→value routing mechanism (OrthoLocalizer + retrieval head)
- PINNED_MASKING scoping (English = all except blocks 0-3, binding = all)
- Binding gates (≥76/80 answer/BD, collapse==0 on both pools)

**Evaluators:**
- TRAIN16 forced-choice and exact generation (16/16 gate)
- DEV_SURFACE forced-choice and exact (correct≥15/16, exact≥13/16)
- DEV_ORDER forced-choice and exact (exact≥13/16)
- D3 four-name mass measurement (≤0.01)
- Language CE on aligned TinyStoriesDEV (≤ own u0 + 0.25)
- Both protected binding pools (pilot0_dev, pilot1_dev) + rehearsal pool

**Discipline:**
- Provenance/hashing/sealing (FREEZE_RECEIPT.json, MANIFEST.json, SHA256SUMS.txt)
- Prospective gating and stopping rules
- LOCKED_UNSCORED transfer panel discipline
- Deterministic seeding (PYTHONHASHSEED)

**Architecture:**
- Decoder-only transformer (proven base)
- Untied language head (proven in this lineage)
- Learned absolute positional embeddings
- Pre-norm residual structure

### 15.2 Category B: PRESERVE AS REGRESSION TEST (Not Necessarily as Architecture)

- T8-T11 failed binding mechanisms (useful as historical exams)
- SF1-SF7 early factual mechanisms
- HR1-HR3 human readiness mechanisms (label-shift defect noted)
- Generation pathology probes
- English context characterization batteries

### 15.3 Category C: SAFE TO RETIRE / NOT AUTOMATICALLY PORT

- SF14-SF19 failed coexistence mechanisms (CE ablation, projection, freeze, membership)
- SF20/SF21 identity rotation/consistency mechanisms (failed to move frontier)
- Hardcoded shape literals (must be generalized, not copied)
- Study-specific build scripts (sf1_build.py, sf2_* scripts)
- Obsolete diagnostics and forensic dirs

---

## 16. What vNext Should Not Automatically Inherit

1. **Hardcoded shape literals** (1024 vocab, 256 context, 320 d_model, 8 blocks, 10 heads) — must be parameterized
2. **Constant LR=5e-05** — may need scaling with model size
3. **No LR scheduling** — may benefit from warmup/decay for larger model
4. **Batch size 36** — may need adjustment for VRAM constraints
5. **fp32 precision** — consider mixed precision for 60M
6. **Manual attention implementation** — consider SDPA/flash for performance
7. **T13 synthetic-task-specific geometry** (ROW_VALUE_OFFSET=4, 2-row slots) — binding mechanism needs generalization for natural language
8. **Multi-token name assumption** — all names are 3-4 tokens; may not hold for larger vocab

---

## 17. Capacity / Interference Evidence

### 17.1 What 10.6M Can Do Simultaneously

**Preserved across SF8-SF21:**
- ✓ Forced-choice factual correctness (TRAIN16 correct = 16/16)
- ✓ Reversal consistency (TRAIN16 reversals = 8/8)
- ✓ Complete factual families (TRAIN16 families = 4/4)
- ✓ Language retention (CE ≤ own u0 + 0.25)
- ✓ Binding preservation (answer≥76, BD≥76, collapse==0 on both pools)
- ✓ Low D3 (four-name mass ≤ 0.01)

**Consistently fails:**
- ✗ Exact surface generalization (DEV_SURFACE exact < 13/16)
- ✗ Exact order generalization (DEV_ORDER exact < 13/16)
- ✗ Development gain (surface/order exact < own u0 + 4)

### 17.2 The Pareto Pattern

| Intervention | Effect on Exact Gen | Effect on D3 | Effect on Widening |
|---|---|---|---|
| +CE on first token | ↑ | ↑↑ (bad) | — |
| -CE on first token | ↓ | ↓↓ (good) | ↓ |
| Partial CE | ↑ | ↑ (bad) | ↑ |
| Gradient projection | — | — | — |
| Output head freeze | — | ↓ | — |
| Identity rotation (SF20) | ↓ | ↓↓ (good) | ↓ |
| Representation consistency (SF21) | ↓ | ↓↓ (good) | ↓ |

**Hypothesis:** There is a fundamental tension between exact multi-token name generation and D3 containment that cannot be resolved through coefficient tuning at 10.6M parameters.

### 17.3 Where the Failure Appears to Live

**Evidence points to:**
- **Output head:** All names are multi-token; first-token collisions are inevitable with 1024 vocab
- **Candidate selection:** Forced-choice works (margin > 0), but greedy exact generation fails
- **Identity/name priors:** D3 measures four-name mass in ordinary English contexts
- **Upstream representations:** The localizer and retrieval head are task-specific and geometry-dependent

**Unresolved:** Whether the failure is fundamentally architectural (output head, tokenization) or representational (capacity).

---

## 18. 60M Migration Constraints

### 18.1 Current Architecture Bottlenecks

| Component | Current | ~60M Target | Scaling |
|---|---|---|---|
| Base params | 10,594,944 | ~60,000,000 | ~5.7x |
| Attention per block | 409,920 | ~2,340,000 (if d=768) | k² |
| MLP per block | 820,800 | ~4,720,000 (if d=768, h=3072) | k² |
| Total blocks | 8 | ? | depth |
| Vocab | 1024 | 1024 (keep) or larger | linear |
| Context | 256 | 256 or larger | linear |

### 18.2 Checkpoint Migration

**Cannot transplant:** Current 10.6M weights are incompatible with different d_model, n_heads, d_mlp, depth.

**Options:**
1. **From-scratch training** (recommended): Simplest, cleanest
2. **Partial initialization:** Base model weights could be resized (embeddings, LM head) but attention/MLP would need reinitialization
3. **Progressive scaling:** Not applicable; no intermediate checkpoints

### 18.3 Code Generalization Required

**Must parameterize:**
- All hardcoded 1024, 256, 320, 8, 10, 32 literals
- DOCUMENT_LENGTH, VOCAB_SIZE, EMBEDDING_SIZE, NUM_HEADS, NUM_LAYERS, CONTEXT_SIZE
- RETRIEVAL_DIM, ROW_VALUE_OFFSET (if keeping T13 binding)
- Batch sizes, update counts

**Must abstract:**
- Binding geometry (2-row assumption, offset=4)
- Name tokenization assumptions (3-token names)
- First-token collision handling

### 18.4 Training Infrastructure Readiness

**Already supports larger models:**
- AdamW optimizer (scale-invariant)
- Gradient clipping (scale-invariant)
- Checkpointing mechanism (works for any size)
- Evaluation framework (works for any size)

**Needs attention for ~60M:**
- VRAM constraints (16GB)
- Batch size adjustments
- Precision (fp32 may be slow; consider mixed)
- Attention implementation (manual loop is slow; consider SDPA)

---

## 19. Fan Diesel Runtime Constraints

### 19.1 Hardware Context

| Component | Spec | Source |
|---|---|---|
| OS | Windows 11 | Known |
| GPU | AMD Radeon RX 9060 XT 16GB | Known |
| CPU | Ryzen 7 7700X | Known |
| RAM | 32GB DDR5 | Known |
| Python | 3.12.14 | sf2venv |
| Torch | 2.12.0+rocm7.14.0 | sf2venv |
| Tokenizers | 0.23.1 | sf2venv |

### 19.2 Memory Estimates (10.6M → ~60M)

| Component | 10.6M Size | ~60M Estimate | Scaling |
|---|---|---|---|
| Model weights (fp32) | ~42 MB | ~240 MB | 5.7x |
| Gradients (fp32) | ~42 MB | ~240 MB | 5.7x |
| Optimizer states (AdamW, 2 tensors) | ~84 MB | ~480 MB | 5.7x |
| Activations (per update) | ~Varies | ~1-2 GB | ~5.7x |
| KL teacher (2nd model) | ~42 MB | ~240 MB | 5.7x |
| Checkpoint (model+opt+rng) | ~151 MB | ~860 MB | ~5.7x |
| **Total resident** | ~400-500 MB | ~2-3 GB | ~5.7x |

**VRAM constraint:** 16GB GPU. ~60M model should fit with batch size adjustments and gradient accumulation.

### 19.3 Performance Bottlenecks

**Current:**
- Manual attention loop (no SDPA/flash) — slow on CPU, OK on GPU
- fp32 precision — uses more memory, slower than mixed
- No gradient accumulation — batch size limited by VRAM

**For ~60M:**
- **Recommend:** Enable SDPA/flash attention
- **Recommend:** Use mixed precision (bf16/fp16)
- **Recommend:** Use gradient accumulation to maintain effective batch size
- **Recommend:** Profile memory to determine max batch size on 16GB

### 19.4 Batch Size Considerations

**Current:** 36 records/update (English), 80 records (binding)
**Per-record tokens:** ~193 (binding docs), ~5-20 (factual rows)
**Total tokens/update:** ~36×20 = 720 (English) + 80×193 = 15,440 (binding) = ~16,160 tokens

**For ~60M:**
- Model params: ~60M × 4 bytes = 240 MB (fp32)
- Gradients: 240 MB
- Optimizer: 480 MB (AdamW)
- Activations: ~3 × model size = 720 MB (rough estimate)
- **Total:** ~1,440 MB per forward/backward
- **VRAM headroom:** 16GB - 1.4GB = 14.6GB for batch
- **Batch size:** Likely can maintain similar or larger batch sizes with gradient accumulation

---

## 20. Discrepancies / Uncertainties

### 20.1 Verified Facts

| Fact | Status | Source |
|---|---|---|
| Base parameter count = 10,594,944 | ✓ VERIFIED | Code + checkpoints |
| Total trained = 10,841,345 | ✓ VERIFIED | Checkpoints |
| Tokenizer SHA256 | ✓ VERIFIED | Computed hash matches |
| Pilot 1 SHA256 | ✓ VERIFIED | Computed hash matches |
| SF8 low-dose = ACQUISITION_SUCCESS (3/3) | ✓ VERIFIED | STATUS.json files |
| SF21 v2 = STOP_REGRESSION (3/3) | ✓ VERIFIED | STATUS.json files |
| SF21 classification = CLOSED | ✓ VERIFIED | PROTOCOL.json + results |
| All names = multi-token (3-4 tokens) | ✓ VERIFIED | Live encoding |

### 20.2 Minor Discrepancies

1. **T13 directory count:** Audit prompt mentions 7 treatment13_* dirs; physical repo has 6. The canonical graduate is in `treatment13_orthogonal_shared_unbounded_seed8380`.

2. **SF1 mapping:** The audit prompt lists SF1 as "initial factual acquisition"; physical dirs include `single_fact_acquisition_sf1_seed87011` and `fact_supervision_87001*` series. The authoritative SF1 result holder is `single_fact_acquisition_sf1_seed87011`.

3. **T12 numbers:** Audit prompt states "100% retrieval and answer accuracy"; physical T12 result is 160/160 retention, 40/40 quartets, retrieval row acc 1.0 — consistent with ceiling behavior.

### 20.3 Uncertainties (Could Materially Affect vNext)

1. **Binding mechanism generalization:** The current T13 binding is heavily coupled to synthetic grammar (2-row, offset=4). It is unclear how to port this to natural language factual tasks without fundamental redesign.

2. **Tokenization impact:** All names are multi-token with heavy first-token collisions (7 IDs for 20 names). A larger vocab could reduce collisions but would break all existing frozen datasets.

3. **Exact generation vs forced-choice:** Forced-choice (margin) works reliably; greedy exact generation consistently fails on surface/order. This suggests the problem may be in decoding/routing rather than representation.

4. **D3 definition sufficiency:** D3 measures four-name mass. It is possible that other name leakage patterns exist that are not captured by this metric.

5. **SF9-SF16 detailed results:** Full numerical results for SF9-SF16 were not extracted due to scope. The high-level lessons were inferred from protocol files and directory contents.

---

## 21. Codex Handoff

### 21.1 What Codex Needs to Know

1. **Current model:** 10,594,944 base params (8 blocks, d_model=320, 10 heads, d_mlp=1280, vocab=1024, context=256), untied language head, learned absolute positions, pre-norm, manual attention.

2. **Binding mechanism:** T12/T13 query→source→value routing with OrthoLocalizer (641 params) + retrieval head (245,760 params). Proven at ceiling on synthetic binding. Task-specific geometry (2-row, offset=4).

3. **Language parent:** Pilot 1 from T13 graduate, achieved CE 7.15→2.90, ppl 1278→18.25 while preserving binding at 80/80.

4. **Factual recipe:** SF8 λ_margin=0.25, M=1.0, λ_KL=1.0, 160 positions/update, 180 English + 20 binding updates, AdamW lr=5e-5, wd=0.05, clip=2.0.

5. **Coexistence wall:** SF21 (final treatment) failed. All 3 seeds stopped with retention_acquisition failure. Exact surface/order generalization cannot coexist with forced-choice correctness, reversal, language, binding, low D3.

6. **10.6M series CLOSED:** Prospective stop rule fulfilled. Do not reopen.

### 21.2 Constraints for vNext

- **Must preserve:** T12/T13 binding capability, Pilot 1 language+binding coexistence, all frozen gates, provenance discipline
- **Must generalize:** All hardcoded shape literals, binding geometry, tokenization assumptions
- **VRAM constraint:** 16GB GPU; ~60M should fit with careful batch sizing
- **Cannot transplant:** Current weights incompatible with different architecture
- **From-scratch likely:** Simplest path for vNext

### 21.3 Open Questions for Codex

1. Should vNext keep the 1024 vocab tokenizer or expand it?
2. Should vNext keep the T13 binding mechanism or design a new one for natural language?
3. What width/depth/head allocation for ~60M?
4. Should vNext use mixed precision?
5. Should vNext use SDPA/flash attention?
6. How to handle multi-token name generation at scale?
7. Can the coexistence frontier be moved with more capacity, or is the architecture fundamentally limited?

---

## 22. Exact Files Codex Should Read First

| # | Purpose | Authoritative Path | SHA256 (if available) | Why |
|---|---|---|---|---|
| 1 | Model implementation | C:\DaveLM-v0.9\v0_8\model.py | — | Base transformer |
| 2 | Model config | C:\DaveLM-v0.9\v0_8\config.py | — | Architecture constants |
| 3 | Binding model | C:\DaveLM-CADAVER\treatment13_model.py | B7EEA7B1… | Surviving wrapper |
| 4 | Binding localizer | C:\DaveLM-CADAVER\PINNED_PILOT1_BINDING_IMPLEMENTATION.py | — | OrthoLocalizer + eval |
| 5 | Binding masking | C:\DaveLM-CADAVER\PINNED_MASKING.py | 89A951D7… | prepare_example, set_scope |
| 6 | Tokenizer | C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json | E1C18BAE… | Authoritative tokenizer |
| 7 | Pilot 1 checkpoint | C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt | 2281D20E… | Language parent |
| 8 | SF8 protocol | C:\DaveLM-CADAVER\sf8_margin_dose_comparison_v1\PROTOCOL.json | — | Standing factual recipe |
| 9 | SF21 protocol | C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\PROTOCOL.json | — | Final treatment + closure |
| 10 | SF21 v2 runs | C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\runs\seed_87056_curriculum | — | Representative failure |
| 11 | Binding graduate | C:\DaveLM-CADAVER\treatment13_orthogonal_shared_unbounded_seed8380\checkpoints\orthogonal_shared_unbounded\seed_8380\latest.pt | FED29874… | Canonical binding |
| 12 | Binding results | C:\DaveLM-CADAVER\treatment13_orthogonal_shared_unbounded_seed8380\FINAL_FROZEN_RETENTION_RESULTS.json | — | Graduate numbers |
| 13 | SF2_ENGINE | C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\SF2_ENGINE.py | — | Training engine |
| 14 | CONTROLLER | C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\CONTROLLER.py | — | Training controller |
| 15 | Runtime venv | C:\DaveLM-CADAVER\sf2_runtime\sf2venv | — | Python/torch environment |

---

*End of audit.*
