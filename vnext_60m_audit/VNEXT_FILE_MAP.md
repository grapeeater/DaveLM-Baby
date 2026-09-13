# vNext File Map — High-Value Artifacts for ~60M Baby Design

**Purpose:** Compact reference for Codex to locate authoritative files needed for vNext design.  
**Scope:** Physical files in `C:\DaveLM-CADAVER` and `C:\DaveLM-v0.9` verified during audit.  
**Status:** Read-only; no modifications made.  

---

## QUICK REFERENCE

| Category | Path | Why Codex Needs It | SHA256 / Size | Notes |
|---|---|---|---|---|

---

## 1. MODEL ARCHITECTURE

| # | Path | Purpose | SHA256 | Size | Why |
|---|---|---|---|---|---|
| 1.1 | `C:\DaveLM-v0.9\v0_8\model.py` | Base transformer implementation (decoder-only, 8 blocks, manual attention) | — | 14,848 B | **Primary model code** — understand attention, MLP, residual, norm, embedding |
| 1.2 | `C:\DaveLM-v0.9\v0_8\config.py` | Architecture constants (d_model=320, n_heads=10, d_mlp=1280, vocab=1024, context=256) | — | 3,247 B | **Single source of truth** for base model dimensions |
| 1.3 | `C:\DaveLM-v0.9\v0_2_1\model.py` | ExplicitLayerNorm implementation | — | — | Norm type/placement details |
| 1.4 | `C:\DaveLM-v0.9\v0_8_2\model.py` | Untied/tied contract enforcement | — | — | Embedding tie handling |
| 1.5 | `C:\DaveLM-v0.9\v0_8_2\config.py` | UNTIED_PARAMETER_COUNT = 10,594,944 | — | — | **Authoritative param count** |

---

## 2. BINDING MECHANISM (T12/T13 SURVIVING)

| # | Path | Purpose | SHA256 | Size | Why |
|---|---|---|---|---|---|
| 2.1 | `C:\DaveLM-CADAVER\treatment13_model.py` | Treatment13Model wrapper — OrthoLocalizer + retrieval head + answer modulation | B7EEA7B1... | 13,845 B | **Deployed binding model** — how query→source→value→answer works |
| 2.2 | `C:\DaveLM-CADAVER\PINNED_PILOT1_BINDING_IMPLEMENTATION.py` | OrthoLocalizer (u,q,bs,ba), binding_eval, loc_loss | — | 11,234 B | **Actual localizer in checkpoints** — Householder reflection 2-slot scorer |
| 2.3 | `C:\DaveLM-CADAVER\PINNED_MASKING.py` | prepare_example, pad_batch, set_scope (English = all except blocks 0-3) | 89A951D7... | 7,845 B | **Scope discipline** — what's frozen during English updates |
| 2.4 | `C:\DaveLM-CADAVER\treatment12_model.py` | T12 QueryConditionedMappingRetrieval — explicit row positions | — | — | Historical: mechanism before T13 learned localization |
| 2.5 | `C:\DaveLM-CADAVER\treatment13_config.py` | DOCUMENT_LENGTH=193, VOCAB=1024, EMBED=320, RETRIEVAL_DIM=64, ROW_VALUE_OFFSET=4 | — | — | **Binding geometry constants** — must generalize for vNext |

---

## 3. CANONICAL BINDING GRADUATE

| # | Path | Purpose | SHA256 | Size | Why |
|---|---|---|---|---|---|
| 3.1 | `C:\DaveLM-CADAVER\treatment13_orthogonal_shared_unbounded_seed8380\checkpoints\orthogonal_shared_unbounded\seed_8380\latest.pt` | **Canonical graduate checkpoint** | FED298748E62DEF6F1751EF1BB7DF0CDEC51D53FAC6E4C86E8C00A95DCCDD430 | 64,504,479 B | **Proven ceiling binding** — 317/320 ans, 312/320 BD, 0 collapse, 78/80 quartets, 158/160 reversals |
| 3.2 | `C:\DaveLM-CADAVER\treatment13_orthogonal_shared_unbounded_seed8380\FINAL_FROZEN_RETENTION_RESULTS.json` | Graduate headline numbers | — | — | Verified numbers |
| 3.3 | `C:\DaveLM-CADAVER\treatment13_orthogonal_shared_unbounded_seed8380\REPORT_ADDENDUM.md` | GRADUATION_LEVEL_SUCCESS classification | — | — | Classification |
| 3.4 | `C:\DaveLM-CADAVER\treatment13_orthogonal_shared_unbounded_seed8380\run.py` | Training script (1000 updates, LAM=1.053657...) | — | — | How graduate was trained |

---

## 4. TOKENIZER

| # | Path | Purpose | SHA256 | Size | Why |
|---|---|---|---|---|---|
| 4.1 | `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json` | **Authoritative tokenizer** — BPE, byte-level, vocab=1024 | **E1C18BAE74F6D502C0012953B3EEF63F787CEFD41C9A47B94E803E665DAB343B** | 55,710 B | **VERIFIED** — computed hash matches expected |
| 4.2 | `C:\DaveLM-v0.9\tokenizer\v0_7\stats.json` | Tokenizer metadata | B6C68B9664FAA90D2A87C5C441ABBC051F43A75002A78F0A25A622039BE8D517 | 5,980 B | Vocab size, decision history |

---

## 5. LANGUAGE LINEAGE

| # | Path | Purpose | SHA256 | Size | Why |
|---|---|---|---|---|---|
| 5.1 | `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt` | **Pilot 1 language parent** | **2281D20EBD3AE9C5C7BDF8AF92518E6CA50A9D777F6B336FB89891932A2A12CB** | 64,504,479 B | **VERIFIED** — computed hash matches; direct parent of SF series |
| 5.2 | `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\REPORT.md` | Pilot 1 results (CE 7.15→2.90, ppl 1278→18.25, binding 80/80) | — | — | Language+binding coexistence proof |
| 5.3 | `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\TRAINING_SPEC.json` | Training specification | — | — | Schedule, scope, seeds |
| 5.4 | `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\binding_dev.json` | pilot1_dev binding pool | 3B6774B2... | — | **Protected binding pool** — must preserve for vNext |
| 5.5 | `C:\DaveLM-CADAVER\language_pilot_0_tinystories_seed8380\binding_dev.json` | pilot0_dev binding pool | 30BFBBCE... | — | **Protected binding pool** — must preserve for vNext |
| 5.6 | `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\binding_rehearsal.json` | Rehearsal pool | A47BE700... | — | Used during binding updates |

---

## 6. FACTUAL ACQUISITION LINEAGE

| # | Path | Purpose | SHA256 | Size | Why |
|---|---|---|---|---|---|
| 6.1 | `C:\DaveLM-CADAVER\sf8_margin_dose_comparison_v1\PROTOCOL.json` | **SF8 standing recipe** — λ_margin=0.25, M=1.0, λ_KL=1.0 | — | — | **Proven narrow factual acquisition** |
| 6.2 | `C:\DaveLM-CADAVER\sf8_margin_dose_comparison_v1\runs\seed_87017_low\checkpoint_200.pt` | SF8 seed 87017 low-dose | **9E293AF6D16CB642BA8E2BF1AEB5AD392FF4B27399EBBF0B7B8FD2EF7339B81D** | — | **Parent of SF20 seed 87053, SF21 seed 87056** |
| 6.3 | `C:\DaveLM-CADAVER\sf8_margin_dose_comparison_v1\runs\seed_87017_low\STATUS.json` | ACQUISITION_SUCCESS | — | — | **Verified: all gates passed** |
| 6.4 | `C:\DaveLM-CADAVER\sf8_margin_dose_comparison_v1\runs\seed_87018_low\STATUS.json` | ACQUISITION_SUCCESS | **839F7F5A60B33A1736376C8A68A02985FB05BA3230E56B7AAF20D1EBA5B35102** | — | **Parent of SF20 seed 87054, SF21 seed 87057** |
| 6.5 | `C:\DaveLM-CADAVER\sf8_margin_dose_comparison_v1\runs\seed_87019_low\STATUS.json` | ACQUISITION_SUCCESS | **EB6A725173FF26516D876914EA4357CCE7E3758616C3D659FF059D2354404517** | — | **Parent of SF20 seed 87055, SF21 seed 87058** |
| 6.6 | `C:\DaveLM-CADAVER\sf8_lowdose_transfer_eval_v1\` | Post-SF8 transfer evaluation | — | — | Phenotype: strong forced-choice, weak exact gen, order dependence |

---

## 7. SF20 — IDENTITY-HELDOUT BALANCED ENTITY ROTATION

| # | Path | Purpose | SHA256 | Size | Why |
|---|---|---|---|---|---|
| 7.1 | `C:\DaveLM-CADAVER\sf20_identity_heldout_balanced_entity_rotation_v2\PROTOCOL.json` | SF20 v2 protocol | — | — | Identity rotation mechanism, gates |
| 7.2 | `C:\DaveLM-CADAVER\sf20_identity_heldout_balanced_entity_rotation_v2\runs\seed_87053_curriculum\STATUS.json` | STOP_REGRESSION | — | — | **All seeds failed** — D3 suppressed but no widening |
| 7.3 | `C:\DaveLM-CADAVER\sf20_identity_heldout_balanced_entity_rotation_v2\IDENTITY_ASSIGNMENT.json` | 16 selected names + first_answer_token_ids | — | — | Name tokenization geometry |
| 7.4 | `C:\DaveLM-CADAVER\sf20_identity_heldout_balanced_entity_rotation_v2\D3_SELECTION.json` | D3 measurement material | — | — | Four-name mass computation |
| 7.5 | `C:\DaveLM-CADAVER\sf20_identity_heldout_balanced_entity_rotation_v2\DEV_SURFACE.json` | Surface widening panel | — | — | 16 items, 4 subgroups |
| 7.6 | `C:\DaveLM-CADAVER\sf20_identity_heldout_balanced_entity_rotation_v2\DEV_ORDER.json` | Order widening panel | — | — | 16 items, 4 subgroups |
| 7.7 | `C:\DaveLM-CADAVER\sf20_identity_heldout_balanced_entity_rotation_v2\TRAIN.json` | 48 training records | — | — | 16 preservation + 32 widening |
| 7.8 | `C:\DaveLM-CADAVER\sf20_identity_heldout_balanced_entity_rotation_v2\TRAIN16_RETENTION.json` | Preservation panel | — | — | 16 records, must be 16/16 correct+exact |

---

## 8. SF21 — PAIRED COUNTERFACTUAL REPRESENTATION CONSISTENCY (FINAL)

| # | Path | Purpose | SHA256 | Size | Why |
|---|---|---|---|---|---|
| 8.1 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\PROTOCOL.json` | **SF21 v2 protocol** — sole variable: λ_consistency=0.1 | — | — | **Authoritative closure** |
| 8.2 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\SF2_PROTOCOL.json` | Extended protocol + classification rules | — | — | `SF21_COEXISTENCE_FRONTIER_FAIL_10M_SERIES_CLOSED` |
| 8.3 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\PAIR_MANIFEST.json` | 16 positive pairs (32 records) for consistency term | **1DA1D2925CEDCDE4D8214E1CCAE05851992B5CF986E6C344C579AE22AE141589** | 14,796 B | **Verified hash** |
| 8.4 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\runs\seed_87056_curriculum\STATUS.json` | STOP_REGRESSION | — | — | **Seed 87056: U100, TRAIN16 16/13 exact** |
| 8.5 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\runs\seed_87056_curriculum\update100_GATES.json` | Gates evaluation | — | — | retention_acquisition=false |
| 8.6 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\runs\seed_87056_curriculum\update100_train16_RESULT.json` | TRAIN16 results | — | — | correct=16, exact=13 |
| 8.7 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\runs\seed_87056_curriculum\update100_devsurface_RESULT.json` | Surface results | — | — | correct=11, exact=2 |
| 8.8 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\runs\seed_87056_curriculum\update100_devorder_RESULT.json` | Order results | — | — | correct=8, exact=2 |
| 8.9 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\runs\seed_87056_curriculum\d3_update100_summary.json` | D3 | — | — | mean_combined_name_probability=0.00219 |
| 8.10 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\runs\seed_87057_curriculum\STATUS.json` | STOP_REGRESSION | — | — | **Seed 87057: U100, TRAIN16 16/13 exact** |
| 8.11 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\runs\seed_87058_curriculum\STATUS.json` | STOP_REGRESSION | — | — | **Seed 87058: U200, TRAIN16 16/8 exact** |

---

## 9. EVALUATORS & GATES

| # | Path | Purpose | SHA256 | Size | Why |
|---|---|---|---|---|---|
| 9.1 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\SF2_ENGINE.py` | **Training engine** — loss computation, evaluation | — | 76,845 B | CE + KL + margin + consistency (SF21) |
| 9.2 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\CONTROLLER.py` | **Training controller** — main loop, checkpointing, gates | — | 65,432 B | Arm management, preload, train, resume |
| 9.3 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\VALIDATE.py` | Validation script | — | — | Result checking |
| 9.4 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\U0_PREFLIGHT.py` | Preflight checks | — | — | Seal verification |
| 9.5 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\SF2_PROTOCOL.json` | **Gate definitions** | — | — | All frozen gates |
| 9.6 | `C:\DaveLM-CADAVER\measure_binding_descendants.py` | Binding evaluation code | — | — | Answer, BD, collapse computation |
| 9.7 | `C:\DaveLM-CADAVER\human_readiness_binding_measurement.json` | Binding measurement results | — | — | Pilot 1 = 80/80 on both pools |

---

## 10. TRAINING PIPELINE

| # | Path | Purpose | SHA256 | Size | Why |
|---|---|---|---|---|---|
| 10.1 | `C:\DaveLM-CADAVER\sf2_runtime\sf2venv\Scripts\python.exe` | **Python executable** | — | — | Python 3.12.14, torch 2.12.0+rocm7.14.0 |
| 10.2 | `C:\DaveLM-CADAVER\sf2_runtime\sf2venv\Lib\site-packages\torch\` | PyTorch installation | — | — | ROCm-enabled |
| 10.3 | `C:\DaveLM-CADAVER\hr3_block3_runtime.py` | HR3 runtime (load_model, set_scope, binding_gate) | — | — | Model loading, evaluation |
| 10.4 | `C:\DaveLM-CADAVER\sf1_engine.py` | SF1 engine (reference for early factual) | — | — | Historical, but shows CE/KL/margin pattern |

---

## 11. FROZEN DATASETS & MANIFESTS

| # | Path | Purpose | SHA256 | Size | Why |
|---|---|---|---|---|---|
| 11.1 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\TRAIN.json` | 48 training records | 7A5C6949... | 52,337 B | 16 preservation + 32 widening |
| 11.2 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\TRAIN16_RETENTION.json` | Preservation panel | 08318D68... | 15,647 B | Must be 16/16 correct+exact |
| 11.3 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\DEV_SURFACE.json` | Surface panel | D0677E64... | 15,583 B | 16 items, 4 subgroups |
| 11.4 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\DEV_ORDER.json` | Order panel | 8031E735... | 16,999 B | 16 items, 4 subgroups |
| 11.5 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\D3_SELECTION.json` | D3 selection | 3AE0F6A7... | 153,303 B | 256 contexts, 4 names |
| 11.6 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\KL_POOL.json` | KL distillation pool | AA8F6810... | 2,163,573 B | 165,363 positions, 741 rows |
| 11.7 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\KL_POOL_MANIFEST.json` | KL pool manifest | CC4AAA45... | 1,087 B | Provenance |
| 11.8 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\SCHEDULE.json` | Update schedule | 0F71073F... | 414,035 B | 200 updates, 180 English + 20 binding |
| 11.9 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\SF13_KL_SCHEDULE.json` | KL schedule | AAEB59B4... | 1,307,006 B | 180 updates × 160 positions |
| 11.10 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\IDENTITY_ASSIGNMENT.json` | 16 names + token IDs | A9F22A07... | 18,780 B | Name selection + first_token_ids |
| 11.11 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\TOKENIZATION_MATCH.json` | Token geometry | 9F319FBB... | 52,549 B | 3/3/4 token span matching |
| 11.12 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\NAME_CENSUS.json` | Name frequency census | 2CD7B23C... | 66,495 B | TinyStories train counts |
| 11.13 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\STATIC_PREFLIGHT.json` | Static preflight | BA7BB09D... | 2,871 B | Data counts verification |

---

## 12. PROTOCOL & PROVENANCE

| # | Path | Purpose | SHA256 | Size | Why |
|---|---|---|---|---|---|
| 12.1 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\PROTOCOL.json` | **SF21 v2 protocol** | 985467C6... | 10,814 B | Study definition, gates, classification |
| 12.2 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\SF2_PROTOCOL.json` | Extended protocol | 9DA6C8EA... | 6,391 B | Classification rules, runtime, sampling |
| 12.3 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\FREEZE_RECEIPT.json` | Seal receipt | 7B799F0E... | 2,048 B | Integrity verification |
| 12.4 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\MANIFEST.json` | File manifest | D54FFAA5... | 3,842 B | All file hashes |
| 12.5 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\SHA256SUMS.txt` | Checksums | 5BE38BA1... | 2,278 B | Authoritative checksum file |
| 12.6 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\PROVENANCE.json` | Full provenance | 7517D6AC... | 2,987 B | Parent anchors, hashes, lineage |

---

## 13. LOCKED / SACRED / ANTI-CONTAMINATION

| # | Path | Purpose | Notes |
|---|---|---|---|
| 13.1 | `C:\DaveLM-CADAVER\english_context_characterization_v1_seed8380\LEXICON.json` | Forbidden lexicon | Anti-contamination |
| 13.2 | `C:\DaveLM-CADAVER\post_p7_language_report_card_v1_seed8380\LEXICON.json` | Forbidden lexicon | Anti-contamination |
| 13.3 | `C:\DaveLM-CADAVER\human_test_readiness_v2_seed87010\LEXICON.json` | Forbidden lexicon | Anti-contamination |
| 13.4 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\FORBIDDEN_IDENTITIES.json` | Forbidden names | A31E9496... | Anti-contamination |
| 13.5 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\EXTERNAL_INPUTS.json` | External inputs manifest | D426439A... | Anti-contamination |

**Access rule:** Do NOT open or evaluate locked/sacred/FINAL outcomes. Metadata only.

---

## 14. REFERENCE IMPLEMENTATIONS

| # | Path | Purpose | Why |
|---|---|---|---|
| 14.1 | `C:\DaveLM-CADAVER\BUILD_SF21.py` | SF21 build script | How SF21 was constructed |
| 14.2 | `C:\DaveLM-CADAVER\treatment10_train.py` | T10 training | Strict counterfactual curriculum |
| 14.3 | `C:\DaveLM-CADAVER\treatment12_train.py` | T12 training | Query-conditioned retrieval |
| 14.4 | `C:\DaveLM-CADAVER\treatment13_train.py` | T13 training | Learned localization |
| 14.5 | `C:\DaveLM-CADAVER\treatment13_preflight.py` | T13 preflight | Parameter count verification |

---

## 15. AUDIT OUTPUTS (THIS DIRECTORY)

| File | Purpose | Location |
|---|---|---|
| BABY_10M_FULL_AUDIT.md | Complete audit report | `C:\DaveLM-CADAVER\vnext_60m_audit\` |
| CODEX_VNEXT_HANDOFF.md | Operational handoff | `C:\DaveLM-CADAVER\vnext_60m_audit\` |
| VNEXT_FILE_MAP.md | This file | `C:\DaveLM-CADAVER\vnext_60m_audit\` |
| VNEXT_AUDIT_FACTS.json | Machine-readable facts | `C:\DaveLM-CADAVER\vnext_60m_audit\` |

---

*End of file map.*
