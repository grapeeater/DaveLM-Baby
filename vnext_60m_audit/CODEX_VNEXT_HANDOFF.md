# Codex vNext Handoff — Baby ~60M Design Brief

**Status:** READY FOR DESIGN  
**Last audit:** 2026-09-09  
**Physical evidence:** Verified against `C:\DaveLM-CADAVER` and `C:\DaveLM-v0.9`  

---

## EXECUTIVE HANDOFF

You are Codex. Your task: **Design a ~60M-parameter Baby successor** that tests whether representational elbow room can move the coexistence frontier that defeated the 10.6M regime.

**The 10.6M factual-supervision series is CLOSED.** SF21 — the prospectively declared final treatment — failed across all three fresh-seed branches (87056/87057/87058). Do not reopen it. Do not create SF22. Do not micro-tune the 10.6M regime further.

Your design must respect **what Research Baby has already proven** and **what it has already failed to achieve**.

---

## 1. VERIFIED CURRENT ARCHITECTURE

### 1.1 Baby ~10.6M — Exact Specification

| Component | Value | Physical Source |
|---|---|---|
| **Architecture** | Decoder-only transformer | `C:\DaveLM-v0.9\v0_8\model.py` |
| **Layers** | 8 | `v0_8/config.py:70` |
| **d_model** | 320 | `v0_8/config.py:68` |
| **n_heads** | 10 | `v0_8/config.py:69` |
| **head_dim** | 32 (320//10) | `v0_7/model.py:42-44` |
| **MLP** | 320 → 1280 → 320, ReLU | `v0_8/config.py:71`, `v0_7/model.py:59` |
| **Vocab** | 1024 | `v0_8/config.py:66` |
| **Context** | 256 | `v0_8/config.py:67` |
| **Positions** | Learned absolute, (256, 320) | `v0_7/model.py:95,111-112` |
| **Dropout** | 0.05 (embedding, attention, residual) | `v0_8/config.py:72-74` |
| **Norm** | Pre-norm LayerNorm | `v0_7/model.py:81-82` |
| **Attention** | Manual per-head loop (no SDPA) | `v0_7/model.py:39-52` |
| **Embedding tie** | UNTIED | `v0_8_2/config.py:33` |
| **LM head** | `nn.Linear(320, 1024)` + bias | `v0_7/model.py:103` |
| **Base params** | **10,594,944** | Code + checkpoint verification |
| **Binding params** | **246,401** | OrthoLocalizer + retrieval head |
| **Total trained** | **10,841,345** | All SF8/SF20/SF21 checkpoints |

### 1.2 Binding Machinery (T12/T13 Surviving Mechanism)

**What it does:** Query → match source → retrieve source value → answer routing.

**Components:**
- **OrthoLocalizer** (641 params): Householder reflection 2-slot scorer (`u[320]`, `q[319]`, `bs[1]`, `ba[1]`)
- **Retrieval head** (245,760 params): `wq(320→64)`, `wk(320→64)`, `wv(320→320)`, `wo(320→320)`
- **Source localization:** Geometry-dependent (ROW_VALUE_OFFSET=4, 2-row slots, position masks)
- **Answer modulation:** `answer_hidden + wo(retrieved)` spliced at answer position

**Status:** Proven at ceiling on synthetic binding:
- Answer: 317/320 (99.06%)
- BOTH_DISTINCT: 312/320 (97.5%)
- Collapse: 0/320
- Complete quartets: 78/80 (97.5%)
- Reversals: 158/160 (98.75%)

**Checkpoint:** `C:\DaveLM-CADAVER\treatment13_orthogonal_shared_unbounded_seed8380\checkpoints\orthogonal_shared_unbounded\seed_8380\latest.pt` SHA256 `fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430`

**Classification:** GENERIC reusable (retrieval head architecture, residual injection) vs SYNTHETIC-TASK-SPECIFIC (2-row geometry, offset=4 pairing, position masks).

---

## 2. VERIFIED TOKENIZER

| Attribute | Value | Physical Source |
|---|---|---|
| **Path** | `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json` | — |
| **SHA256** | `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b` | Computed + matches |
| **Family** | HuggingFace tokenizers v1.0, BPE, byte-level | JSON header |
| **Vocab size** | 1024 | Count of vocab entries |
| **Special tokens** | `<pad>`=0, `<unk>`=1, `<bos>`=2, `<eos>`=3, `<doc>`=4 | added_tokens |
| **No `<mask>`** | Confirmed | token_to_id returns None |

### 2.1 Critical Tokenization Fact

**ALL NAMES ARE MULTI-TOKEN.** The 16 identities used in SF20/SF21 (Sara, Bob, Billy, Tony, Fred, Tina, Steve, Susan, Remy, Ralph, Colin, Faye, Carl, Wendy, Cathy, Tara) plus Alex/Owen/Mia/Nora **all encode to 3 BPE tokens bare, 4 tokens as candidates** (" Name."). Only 7 distinct first-token IDs exist for 20 names → heavy first-token collision.

**Implication for vNext:**
- If you keep vocab=1024, multi-token names are inevitable
- If you expand vocab, all frozen datasets (TRAIN.json, DEV_SURFACE.json, etc.) must be re-tokenized
- The first-token collision problem may require architectural solutions (e.g., output routing that handles multi-token generation better)

---

## 3. AUTHORITATIVE CHECKPOINT LINEAGE

### 3.1 Language Parent (Pilot 1)

| Attribute | Value | Physical Source |
|---|---|---|
| **Path** | `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt` | — |
| **SHA256** | `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb` | Computed + matches |
| **Size** | 64,504,479 bytes | — |
| **Training** | 900 TinyStories + 100 binding updates from T13 graduate | — |
| **Early block protection** | Blocks 0-3 frozen during language steps | run.py set_scope |

**Results (vs Graduate baseline):**
- Dev language CE: 7.1532 → **2.9039**
- Dev perplexity: 1278.2 → **18.25**
- Binding pilot0_dev: 79/80 → **80/80** (ans, BD 80, collapse 0)
- Binding pilot1_dev: — → **80/80** (ans, BD 80, collapse 0)

**Conclusion:** Pilot 1 achieved major language improvement **while preserving** binding at ceiling.

### 3.2 SF8: Standing Narrow Factual Recipe

**All three low-dose seeds (87017, 87018, 87019) achieved ACQUISITION_SUCCESS:**
- Status: ACQUISITION_SUCCESS
- Completed: 200 updates
- All gates passed: acquisition, binding (both pools), language, D3, endpoint

**Recipe:**
- λ_margin = 0.25
- M (margin target) = 1.0 nat
- λ_KL = 1.0
- Positions per update: 160
- Updates: 180 English + 20 binding
- Optimizer: AdamW lr=5e-05, wd=0.05, clip=2.0
- Scope: English = all except blocks 0-3, binding = all

**This is the proven narrow factual acquisition capability that vNext must not lose.**

---

## 4. SF21 CLOSURE AND WHY THE 10M SERIES STOPPED

### 4.1 SF21 v2 — The Final Treatment

**Study:** SF21_PAIRED_COUNTERFACTUAL_REPRESENTATION_CONSISTENCY_V2  
**Sole scientific variable:** Add λ_consistency=0.1 paired cosine consistency on L2-normalized logits at position = prefix_length - 1 (pre-answer query state) over 16 frozen widening reversal pairs.

**Seeds:** 87056, 87057, 87058 (fresh seeds, parents = SF8 seeds 87017, 87018, 87019)

### 4.2 Results (Verified from Physical Artifacts)

| Seed | Stop | TRAIN16 | Surface | Order | Lang Δ | D3 | Binding | STATUS |
|---|---|---|---|---|---|---|---|---|
| 87056 | U100 | 16/13 exact | 11/2 exact | 8/2 exact | +0.0912 | 0.00704→0.00219 | ✓ | STOP_REGRESSION |
| 87057 | U100 | 16/13 exact | 10/2 exact | 8/2 exact | +0.1530 | 0.00667→0.00238 | ✓ | STOP_REGRESSION |
| 87058 | U200 | 16/8 exact | 13/1 exact | 8/2 exact | +0.2256 | 0.00784→0.00243 | ✓ | STOP_REGRESSION |

**All three seeds failed the `retention_acquisition` gate** (exact < 16 on TRAIN16) at U100 or U200.

### 4.3 Classification

From `SF21_PAIRED_COUNTERFACTUAL_REPRESENTATION_CONSISTENCY_V2\PROTOCOL.json`:

```json
{
  "classification": {
    "failure": "SF21_COEXISTENCE_FRONTIER_FAIL_10M_SERIES_CLOSED for any other scientifically valid cross-seed result",
    "success": "SF21_COEXISTENCE_FRONTIER_SUCCESS if >=2/3 endpoint_pass"
  }
}
```

**Actual:** 0/3 seeds achieved `endpoint_pass`.  
**Classification:** **SF21_COEXISTENCE_FRONTIER_FAIL_10M_SERIES_CLOSED**

From PROTOCOL.json line 77:
> "If SF21 does not achieve coexistence in >=2/3 branches, close the 10.6M factual-supervision series without SF22 or coefficient rescue."

**The 10.6M factual-supervision series is CLOSED.**

### 4.4 Prospective Interpretation

**NOT proven:** 10.6M parameters are mathematically incapable of solving the task.  
**IS established:** Continued micro-tuning of the current 10.6M regime has poor expected value after the prospectively final SF21 treatment failed to move the coexistence frontier.

**The repeated failure pattern:** Exact surface/order generalization **cannot coexist** with forced-choice correctness, reversal consistency, language retention, binding preservation, and low D3.

---

## 5. THE COEXISTENCE PHENOTYPE

### 5.1 What 10.6M Can Do Simultaneously

✅ **Preserved across SF8–SF21:**
- Forced-choice factual correctness (TRAIN16 correct = 16/16)
- Reversal consistency (TRAIN16 reversals = 8/8)
- Complete factual families (TRAIN16 families = 4/4)
- Language retention (CE ≤ own u0 + 0.25)
- Binding preservation (answer≥76, BD≥76, collapse==0 on both pools)
- Low D3 (four-name mass ≤ 0.01)

❌ **Consistently fails:**
- Exact surface generalization (DEV_SURFACE exact < 13/16)
- Exact order generalization (DEV_ORDER exact < 13/16)
- Development gain (surface/order exact < own u0 + 4)

### 5.2 The Pareto Pattern

| Intervention | Effect on Exact Gen | Effect on D3 | Effect on Widening |
|---|---|---|---|
| +First-token CE | ↑ | ↑↑ (bad) | — |
| -First-token CE | ↓ | ↓↓ (good) | ↓ |
| Partial CE | ↑ | ↑ (bad) | ↑ |
| Gradient projection | — | — | — |
| Output head freeze | — | ↓ | — |
| Identity rotation (SF20) | ↓ | ↓↓ (good) | ↓ |
| Representation consistency (SF21) | ↓ | ↓↓ (good) | ↓ |

**Hypothesis (evidence-supported):** There is a fundamental tension between exact multi-token name generation and D3 containment that cannot be resolved through coefficient tuning at 10.6M parameters. The output head and tokenization geometry are likely part of the problem.

---

## 6. PRESERVATION REQUIREMENTS

### 6.1 MUST PRESERVE (Non-Negotiable)

**Mechanisms:**
- T12/T13 query→source→value routing capability (ceiling on synthetic binding)
- Binding gates (≥76/80 answer/BD, collapse==0 on both protected pools)
- PINNED_MASKING scoping discipline (English = all except blocks 0-3, binding = all)

**Evaluators:**
- TRAIN16 forced-choice + exact (16/16 gate)
- DEV_SURFACE forced-choice + exact (correct≥15/16, exact≥13/16)
- DEV_ORDER forced-choice + exact (exact≥13/16)
- D3 four-name mass (≤0.01)
- Language CE on aligned TinyStoriesDEV (≤ own u0 + 0.25)
- Both protected binding pools (pilot0_dev, pilot1_dev) + rehearsal pool

**Discipline:**
- Provenance/hashing/sealing (FREEZE_RECEIPT.json, MANIFEST.json, SHA256SUMS.txt)
- Prospective gating and stopping rules
- LOCKED_UNSCORED transfer panel discipline
- Deterministic seeding

### 6.2 MUST GENERALIZE (Not Preserve Literally)

- Hardcoded shape literals (1024, 256, 320, 8, 10, 32)
- Binding geometry (2-row, offset=4, position masks)
- Tokenization assumptions (3-4 token names, first-token collisions)
- Manual attention implementation (consider SDPA/flash)
- fp32 precision (consider mixed for performance)

---

## 7. MIGRATION HAZARDS

### 7.1 Checkpoint Incompatibility

**Current 10.6M weights CANNOT be transplanted into a model with:**
- Different d_model
- Different n_heads
- Different d_mlp
- Different depth
- Different vocab size
- Different context size

**Recommendation:** From-scratch training for vNext. This is the simplest and cleanest path.

### 7.2 Hardcoded Shape Coupling

**Files with hardcoded literals that must be generalized:**
- `hr3_block3_runtime.py` (176, 256)
- `sf1_engine.py` (152, 42, 49, 52, 54, 65, 28)
- `SF2_ENGINE.py` (446, 56)
- `CONTROLLER.py` (283, 99, 88)
- `PINNED_MASKING.py` (27)
- `PINNED_PILOT1_BINDING_IMPLEMENTATION.py` (36, 42, 68-69, 81, 88-97, 108)
- `treatment13_preflight.py` (184)
- `treatment13_config.py` (27-31)

**All numeric literals related to shapes must be derived from config, not hardcoded.**

### 7.3 Tokenization Blast Radius

**If you change the tokenizer/vocab:**
- All SHA256 asserts fail (200+ files)
- All frozen datasets with precomputed token IDs become stale
- Checkpoint embedding shapes mismatch
- SF20 first-token distinctness guarantees must be re-verified

**Trade-off:**
- Keep vocab=1024: Preserves evaluator compatibility, isolates capacity test
- Expand vocab: Better language model, but broader redesign

---

## 8. RUNTIME CONSTRAINTS (Fan Diesel)

### 8.1 Hardware

| Component | Spec | Note |
|---|---|---|
| OS | Windows 11 | — |
| GPU | AMD Radeon RX 9060 XT 16GB | ROCm 7.14.0 |
| CPU | Ryzen 7 7700X | — |
| RAM | 32GB DDR5 | — |
| Python | 3.12.14 | sf2venv |
| Torch | 2.12.0+rocm7.14.0 | sf2venv |

### 8.2 Memory Estimates (10.6M → ~60M)

| Component | 10.6M | ~60M (5.7x) | Notes |
|---|---|---|---|
| Model weights (fp32) | ~42 MB | ~240 MB | — |
| Gradients (fp32) | ~42 MB | ~240 MB | — |
| Optimizer (AdamW, 2 states) | ~84 MB | ~480 MB | — |
| Activations | ~Varies | ~1-2 GB | Rough estimate |
| KL teacher (2nd model) | ~42 MB | ~240 MB | — |
| Checkpoint | ~151 MB | ~860 MB | model+opt+rng |
| **Total** | ~400-500 MB | ~2-3 GB | Should fit in 16GB |

### 8.3 Performance Recommendations

1. **Enable SDPA/flash attention** — Current manual attention loop is slow
2. **Use mixed precision (bf16/fp16)** — fp32 uses more memory and is slower
3. **Use gradient accumulation** — Maintain effective batch size with VRAM constraints
4. **Profile memory** — Determine max batch size on 16GB for ~60M

---

## 9. EXACT HIGH-VALUE FILE PATHS

| # | File | Purpose | SHA256 |
|---|---|---|---|
| 1 | `C:\DaveLM-v0.9\v0_8\model.py` | Base transformer implementation | — |
| 2 | `C:\DaveLM-v0.9\v0_8\config.py` | Architecture constants | — |
| 3 | `C:\DaveLM-CADAVER\treatment13_model.py` | Binding wrapper | B7EEA7B1… |
| 4 | `C:\DaveLM-CADAVER\PINNED_PILOT1_BINDING_IMPLEMENTATION.py` | OrthoLocalizer | — |
| 5 | `C:\DaveLM-CADAVER\PINNED_MASKING.py` | Masking/batching | 89A951D7… |
| 6 | `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json` | Tokenizer | E1C18BAE… |
| 7 | `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt` | Language parent | 2281D20E… |
| 8 | `C:\DaveLM-CADAVER\treatment13_orthogonal_shared_unbounded_seed8380\checkpoints\orthogonal_shared_unbounded\seed_8380\latest.pt` | Binding graduate | FED29874… |
| 9 | `C:\DaveLM-CADAVER\sf8_margin_dose_comparison_v1\PROTOCOL.json` | SF8 recipe | — |
| 10 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\PROTOCOL.json` | SF21 closure | — |
| 11 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\runs\seed_87056_curriculum` | SF21 failure | — |

---

## 10. UNRESOLVED DECISIONS YOU MUST MAKE

1. **Vocab size:** Keep 1024 or expand? Trade-off: evaluator compatibility vs language capability.
2. **Context length:** Keep 256 or expand? Trade-off: memory vs longer sequences.
3. **Binding mechanism:** Keep T13 (synthetic-specific) or design new for natural language?
4. **Width vs depth:** How to allocate ~60M? Current is 8×320×10×1280.
5. **Precision:** fp32 (current) or mixed? Trade-off: numerical stability vs speed/memory.
6. **Attention:** Manual loop (current) or SDPA/flash? Trade-off: control vs performance.
7. **Multi-token names:** How to handle exact generation of multi-token answers?
8. **From-scratch vs transplant:** Current weights cannot be transplanted; from-scratch recommended.

---

## 11. YOUR DESIGN BRIEF

**Goal:** Design a ~60M Baby that:
1. **Preserves** all proven capabilities (binding, language, SF8 factual acquisition)
2. **Tests** whether representational elbow room can move the coexistence frontier
3. **Avoids** the architectural coupling that limits the current 10.6M regime
4. **Fits** on 16GB VRAM with practical training time

**Non-goals:**
- Do not redesign the tokenizer unless you explicitly decide the trade-off is worth it
- Do not throw away the binding mechanism unless you have a better alternative
- Do not copy hardcoded literals from the 10.6M code
- Do not assume current weights can be transplanted

**Success criteria:**
- All frozen gates pass (TRAIN16, DEV_SURFACE, DEV_ORDER, D3, language, binding)
- Development gain achieved (surface/order exact ≥ own u0 + 4)
- Endpoint pass on ≥2/3 seeds
- Training practical on 16GB VRAM

**If you succeed:** You have moved the coexistence frontier and proven that capacity was a bottleneck.
**If you fail with the same pattern:** The architectural limitations (output head, tokenization, binding geometry) are the real bottleneck, not capacity.

Either outcome is scientifically valuable.

---

## 12. READY CHECKLIST

- [x] Current architecture verified (10,594,944 base params)
- [x] Binding mechanism verified (T12/T13, OrthoLocalizer + retrieval head)
- [x] Tokenizer verified (BPE, byte-level, vocab=1024, SHA256 confirmed)
- [x] Language parent verified (Pilot 1, SHA256 confirmed)
- [x] SF8 recipe verified (λ_margin=0.25, M=1.0, 3/3 ACQUISITION_SUCCESS)
- [x] SF21 closure verified (0/3 endpoint_pass, series CLOSED)
- [x] Coexistence phenotype characterized
- [x] Preservation requirements identified
- [x] Migration hazards documented
- [x] Runtime constraints estimated
- [x] High-value file paths provided

**Codex is now ready to design ~60M Baby.**

---

*End of handoff.*
