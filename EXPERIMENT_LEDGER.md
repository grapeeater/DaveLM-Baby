# Experiment ledger

Scientific directory names are frozen. This is an index, not a rename map. Checkpoints and `*_run_seed*` trees are local-only.

## Current parent

- **Phase1G** `baby_vnext_phase1g_language_v1` + run `baby_vnext_phase1g_language_v1_run_seed610001` (gitignored). U6000 `best.pt` SHA-256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`. Architecture: `baby_vnext_60m_design_v1`.

## Phase 2A — closed treatments (do not relaunch)

| ID | Directory | Terminal class (handoff) | Notes |
|---|---|---|---|
| T1 | `baby_vnext_phase2a_ctxbind_v1` | predecessor | History |
| T2 | `baby_vnext_phase2a_t2_ctxbind_v1` | `PHASE2A_T2_FAIL` | History |
| T3 design | `phase2a_t3_design_v1` | never implemented | |
| T3 committed v1/v2 | `phase2a_t3_committed_study_v1`, `_v2` | `T3_HARD_STOP` | Superseded; identical ckpts across seeds |
| T3 repair | `phase2a_t3_repair_v1` | infrastructure-blocked | |
| T3 ctxbind attempt | `baby_vnext_phase2a_t3_ctxbind_v1` | invalidated (name overlap) | Keep provenance note |
| **T3 rebuilt** | `phase2a_t3_rebuilt_study_v1` | **authoritative corpus** | TEST sealed here |
| T4 | `phase2a_t4_binding_sidecar_v1` | closed | Do not parent later OUTPUT_FAIL |
| T5 | `phase2a_t5_latebase_binding_v1` | closed | |
| T6 | `phase2a_t6_latebase_pointer_v1` | closed | |
| T7 | `phase2a_t7_paired_reversal_pointer_v1` | closed | |
| T8 | `phase2a_t8_latebase_loc_layout_v1` | closed | |
| T9 | `phase2a_t9_factclause_pointer_v1` | closed | |
| T10 | `phase2a_t10_factclause_pointer_1k_v1` | closed | |
| T11 | `phase2a_t11_factclause_pointer_4to1_750_v1` | closed | |
| T12 | `phase2a_t12_factclause_pointer_lr2p5e5_750_v1` | closed | |
| T13 | `phase2a_t13_factclause_pointer_lr2p5e5_1k_v1` | closed | |
| T14 | `phase2a_t14_factclause_pointer_lr3p75e5_750_v1` | `T14_FAIL_NO_REPRESENTATION` | Do not parent T14 / 730002 |
| T14X | `phase2a_t14x_latebase_decomp_v1` | read-only decomp, closed | |
| T15 | `phase2a_t15_latebase_blocks4to7_lr3p75e5_750_v1` | closed | Do not parent T15 |
| T15X | `phase2a_t15x_t14_t15_latebase_drift_v1` | closed | |
| T16 | `phase2a_t16_scaffold_revert811_lr3p75e5_750_v1` | closed | Do not parent T16 |
| T16X | `phase2a_t16x_reversal_fail_overlap_v1` | closed | |
| T17 | `phase2a_t17_scaffold_revert811_lr3p75e5_1k_v1` | `T17_REPRESENTATION_SUCCESS_OUTPUT_FAIL` | representation **2/3**, exact 0 |
| T17X | `phase2a_t17x_native_generation_failmode_v1` | COMPLETE | Exact 0 = Phase1G story continuation after `Answer:` |
| T18 | `phase2a_t18_t17recipe_answerce_lr3p75e5_1k_v1` | `T18_REPRESENTATION_SUCCESS_OUTPUT_FAIL` | language 3/3, representation **2/3**, exact **36/36/40** |
| T18X | `phase2a_t18x_generation_failmode_v1` | COMPLETE | First-token BOS 76–84; leftover diverge |
| T19 | `phase2a_t19_suffix_answerce_lr3p75e5_1k_v1` | `T19_FAIL_NO_REPRESENTATION` | representation **1/3**, exact **39/35/34**; suffix-weight 3.0 closed |
| T19X | `phase2a_t19x_generation_failmode_v1` | COMPLETE | Leftover still diverge |
| T20 | `phase2a_t20_readout_answerce_lr3p75e5_1k_v1` | `T20_REPRESENTATION_SUCCESS_OUTPUT_FAIL` | representation **2/3**, exact **9/10/13**; head-only CE closed |
| T20X | `phase2a_t20x_generation_failmode_v1` | COMPLETE | First-token BOS **42/34/37**; dominant `other_then_eos` |
| T21 | `phase2a_t21_hybrid_answerce_lr3p75e5_1k_v1` | `T21_REPRESENTATION_SUCCESS_OUTPUT_FAIL` | language 3/3, representation **3/3**, exact **27/29/26** |
| T21X | `phase2a_t21x_generation_failmode_v1` | COMPLETE | First-token BOS **79/59/72**; leftover diverge 52/30/46 + other_then_eos |
| T22 | `phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1` | `T22_REPRESENTATION_SUCCESS_OUTPUT_FAIL` | language 3/3, representation **2/3**, exact **21/31/26**; first-two through-base closed |
| T22X | `phase2a_t22x_generation_failmode_v1` | COMPLETE | BOS **66/65/73**; leftover still diverge 45/34/47 + other_then_eos; CE-span family closed |
| T23 | `phase2a_t23_candidate_unlikelihood_lr3p75e5_1k_v1` | `T23_REPRESENTATION_SUCCESS_OUTPUT_FAIL` | language 3/3, representation **3/3**, exact **34/34/38**; in-row unlikelihood closed |
| T23X | `phase2a_t23x_generation_failmode_v1` | COMPLETE | BOS **79/75/79**; leftover still diverge; off-row Walt/York/Sky |

Do **not** parent T14–T23 OUTPUT_FAIL checkpoints (including 820001–820003).

## Phase 2A — terminal / current

| ID | Directory | State |
|---|---|---|
| T24 | `phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1` | **TERMINAL: `T24_REPRESENTATION_SUCCESS_OUTPUT_FAIL`.** All three seeds finished U1000. Owner released the temporary pause restriction on 2026-09-13 and authorized autonomous successor selection. Historical pause record: `T24_PAUSE.md`. |
| T24X | `phase2a_t24x_generation_failmode_v1` | **`T24X_GENERATION_FAILMODE_COMPLETE`.** Read-only U1000 generation audit; dominant residual is first-token-correct-then-diverge (45–47/128) plus other-then-EOS (25–31/128). |
| T25 | `phase2a_t25_continuation_mechanism_forensic_v1` | Diagnostic/forensic; not a new training treatment. Do not relaunch as T29. |
| T28 | `phase2a_t28_fast_v2` | **`T28_SUFFIX_MARGIN_VALID_COMPLETION / OUTPUT_FAIL`.** Exact 36/35/40 of 128. |
| post-T28 tok | `phase2a_post_t28_tokenizer_diagnostic_v1` | **`TOKENIZER_ASSOCIATION_SUPPORTED`.** Compact reports in git; `ITEMS.json` local-only. |
| post-T28 suffix | `phase2a_post_t28_suffix_state_trace_v1` | **`GOLD_PREFIX_DIRECT_DECODABILITY_WITH_FREE_PREFIX_COLLAPSE`.** `TRACE.json` local-only. |
| mouth repair | `tokenizer_repair_v1` | **`NATIVE_BEAM_DOES_NOT_CLOSE_SHARED_PREFIX`.** v0_7 preserved. |
| Option C | `tokenizer_append_extension_v1` | **`APPEND_ONLY_EXTENSION_VALID_COMPLETION / NO_EFFECT`.** Tokenizer lineage paused. |
| Option B | `tokenizer_replacement_study_v1` | **`TOKENIZER_REPLACEMENT_NOT_JUSTIFIED`.** No candidate beat v0_7 DEV collision geometry. |
| post-T28 prefix rescue | `phase2a_post_t28_prefix_rescue_v1` | **`EARLY_TOKEN_ERROR_CAUSALLY_DOMINATES`.** Single-token gold oracle rescue then free greedy; DEV diverge population 39/40/45. Activation patching not run. |
| T29 | `phase2a_t29_disambiguation_fork_v1` | **`T29_REPRESENTATION_SUCCESS_OUTPUT_FAIL`.** Exact 35/35/35. Shared-prefix exact 2/2/0 of 64. Fork-objective hypothesis closed. TEST sealed. |
| T30 | phase2a_t30_persistent_pointer_route_v1 | **T30_REPRESENTATION_SUCCESS_OUTPUT_FAIL.** Persistent 640-parameter pointer route; exact 31/36/38 of 128; route active; native output gate missed. T3 TEST sealed. Do not launch T31. |

Question (historical; T24 already answered no): does inventory unlikelihood lift exact above T18 36–40 while holding language + representation ≥2/3?

## Closed historical lineages (Category B/C)

- **10.6M SF series** `sf1_*` … `sf21_*`: closed by SF21 (`SF21_COEXISTENCE_FRONTIER_FAIL_10M_SERIES_CLOSED`).
- **P0–P7 language ladder** `language_*`, post-P7 report cards.
- **Fact-supervision / HR / two-mapping / treatment4–13** families.
- **v0.9** nested snapshot: git tracks tokenizer only.
- Audits: `vnext_60m_audit/`, `BABY_REPO_MIGRATION_AUDIT.md`, `forensics/`, `mistral_treatment_design_audit_v1/`, `comparative_model_anatomy_post_sf2_v1/`, `overnight_reproducibility_resolution_v1/`.

Qwen/Smol notes are advisory. Baby evidence outranks them. Do not copy foreign weights.

