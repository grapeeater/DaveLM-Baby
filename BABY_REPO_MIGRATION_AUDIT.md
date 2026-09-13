# DaveLM "Baby" — Repository Migration Audit

**Audit type:** READ-ONLY filesystem audit (no files created/copied/moved/edited/deleted; no Git actions; no training)
**Source:** `C:\DaveLM-CADAVER`
**Future clean repository:** `C:\DaveLM - Baby Stage` (confirmed empty at audit time; 0 entries)
**Audit date:** 2026-09-11
**Authorized write:** this file only (`C:\DaveLM-CADAVER\BABY_REPO_MIGRATION_AUDIT.md`)

> **Live-system warning:** while this audit was running, a read-only diagnostic
> (`phase2a_t14x_latebase_decomp_v1`) was actively executing and writing new files in the
> source tree (status `DECOMP_RUNNING`, artifacts written 2026-09-11 22:51–23:02).
> `PHASE2A_CURRENT_HANDOFF.md` was also rewritten at 22:51 during the audit.
> The source is **not quiescent**. Take the migration snapshot only after the T14x process
> reaches a terminal status.

---

## 1. Executive summary

`C:\DaveLM-CADAVER` is a ~**103.2 GB / 36,859-file** research graveyard that also contains Baby's
current development head. It holds four distinct lineages:

1. **Current Baby — vNext ~61.5M "Phase 2A"** (2026-09-09 → present). This is the current architecture and
   the only lineage still being actively developed. It is built on the frozen architecture bundle
   `baby_vnext_60m_design_v1` and the Phase1G U6000 language parent.
2. **Closed 10.6M factual-supervision series (SF1–SF21)** — prospectively closed by SF21
   (`SF21_COEXISTENCE_FRONTIER_FAIL_10M_SERIES_CLOSED`). Important history, no longer current.
3. **DaveLM v0.9 / treatment 4–13 era** — historical baseline (10,594,944-param base + binding),
   plus a **stale nested snapshot** at `C:\DaveLM-CADAVER\DaveLM-v0.9` that duplicates (and is older
   than) the canonical `C:\DaveLM-v0.9`.
4. **P7 / language pilot / human-readiness / fact-supervision forensic families** — history only.

The overwhelming majority of bytes are **local-only artifacts**: checkpoints (`.pt`), rolling restart
states, run directories, and a 9.0 GB bundled ROCm Python runtime (`sf2_runtime`). A GitHub repo must
**not** carry these. The actual migratable source (code, configs, protocols, reports, manifests,
compact datasets) is a small fraction of the tree.

**Recommended canonical core:** `baby_vnext_60m_design_v1` (architecture) + `baby_vnext_phase1g_language_v1`
(data/protocol) + the `phase2a_*` treatment/diagnostic directories (code/data/reports) + the tokenizer
from `C:\DaveLM-v0.9\tokenizer\v0_7\` + handoff/audit documents. The Phase1G `best.pt` parent and the
T3–T14 checkpoints are required for reproduction but are large binaries → Git LFS or local artifact
storage, human decision.

**Blocking issues for migration:**

- Every Phase2A/Phase1G runner hardcodes absolute Windows paths (`C:\DaveLM-CADAVER\...`,
  `C:\DaveLM-v0.9\...`). A straight copy into `C:\DaveLM - Baby Stage` will **not run**. A path
  indirection refactor (env var / config root) is required.
- Secret-risk files exist (`.aider.chat.history.md`, `mistral_test.py`, `.config\opencode\tunnel.json`).
  None should be committed without review/scrubbing.
- Many checkpoints exceed GitHub's 100 MB hard file limit (largest single file: 696.8 MB).
- The closed 10.6M series and v0.9 baseline contain checkpoints whose preservation vs. exclusion is a
  research-value decision (Category F).

---

## 2. Best-supported current Baby lineage

**Verdict: the current Baby is the vNext ~61.5M "Phase 2A" lineage, currently in a read-only failure
decomposition (`phase2a_t14x_latebase_decomp_v1`). It is NOT the 10.6M SF series and NOT v0.9.**

Evidence chain (physical artifacts, newest state first):

| Step | Artifact | Evidence |
|---|---|---|
| Authoritative status | `C:\DaveLM-CADAVER\PHASE2A_CURRENT_HANDOFF.md` (rewritten 2026-09-11 22:51) | Declares parent, closed T3–T14, and **Live: `phase2a_t14x_latebase_decomp_v1`** (read-only) |
| Current live diagnostic | `C:\DaveLM-CADAVER\phase2a_t14x_latebase_decomp_v1\` | `PROTOCOL.json` `mode: READ_ONLY`, `does_not_train: true`; files written during this audit; depends on T10/T12/T13/T14 checkpoints |
| Last completed treatment | `C:\DaveLM-CADAVER\phase2a_t14_factclause_pointer_lr3p75e5_750_v1\` | `RUN_LEDGER.json`: `STUDY_COMPLETE`, `T14_FAIL_NO_REPRESENTATION`; `next_treatment: phase2a_t14x_latebase_decomp_v1` |
| Current parent | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001\checkpoints\best.pt` | SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1` cited by T3–T14 and T14x protocols |
| Current architecture | `C:\DaveLM-CADAVER\baby_vnext_60m_design_v1\` | `FREEZE_RECEIPT.json`: `BABY_VNEXT_60M_CANDIDATE_FROZEN`; 61,520,385 params; all Phase2A runners `sys.path.insert` this bundle and import `baby_vnext.config` / `baby_vnext.binding` |
| Current training data | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\` + `C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1\data\` | Every T3–T14 runner loads `LANGUAGE_TRAIN_STREAM.u16` from Phase1G and `qa_train.jsonl`/`qa_dev.jsonl`/`acq16.json`/`panels.json` from T3 rebuilt |
| Tokenizer | `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json` | SHA256 `e1c18bae…` cited by design receipt, Phase1G, T3, and every Phase2A runner |

**Why not the alternatives:**

- The **10.6M SF series is explicitly closed** by the Codex handoff
  (`vnext_60m_audit\CODEX_VNEXT_HANDOFF.md`, §4) and by SF21 v2 (`SF21_COEXISTENCE_FRONTIER_FAIL_10M_SERIES_CLOSED`).
  It is history (Category B), not current.
- **v0.9 is a historical baseline.** The mistral design audit states the nested
  `C:\DaveLM-CADAVER\DaveLM-v0.9` is a **stale snapshot**; the current 10.6M dev codebase is the external
  `C:\DaveLM-v0.9`. Neither is the current Baby.
- **"Newest timestamp" traps resolved:**
  - `baby_vnext_phase2a_t3_ctxbind_v1` (2026-09-10) looks newer than the authoritative rebuilt T3
    (2026-09-11), but its `audits.json` has train/dev/test **name overlap** and it is invalidated by the
    T3 rebuilt decision ledger. Do not treat it as canonical.
  - `phase2a_t3_committed_study_v1` (2.1 GB, 2026-09-10) ended `T3_HARD_STOP` with **identical
    checkpoints across seeds**; superseded.
  - `phase2a_t14x_latebase_decomp_v1` is the newest directory but is a **read-only diagnostic**, not a
    treatment. The last treatment is T14.

---

## 3. Category A — Core current source

Material required to continue Baby vNext development and reproduce Phase 2A.

| # | Original source path | What it is | Notes |
|---|---|---|---|
| A1 | `C:\DaveLM-CADAVER\baby_vnext_60m_design_v1\` | **Frozen vNext architecture bundle** | Migrate all except `__pycache__\` and `validation\initialized_checkpoint_roundtrip.pt` (246 MB, Category D) |
| A1a | …`\baby_vnext\` (`config.py`, `model.py`, `binding.py`, `checkpoint.py`, `parameter_count.py`, `__init__.py`) | Current model implementation | Required by every Phase2A runner |
| A1b | …`\BABY_VNEXT_ARCHITECTURE_SPEC.md`, `BABY_VNEXT_IMPLEMENTATION_REPORT.md`, `BABY_VNEXT_TRAINING_BLUEPRINT.md`, `BABY_VNEXT_REGRESSION_MAP.md`, `BABY_VNEXT_FACTS.json` | Architecture spec + design docs | Core design record |
| A1c | …`\BABY_VNEXT_CONFIG.json`, `ARCHITECTURE_CANDIDATES.json`, `PARAMETER_COUNTS.json`, `MEMORY_ESTIMATE.json` | Configs / counts | `BABY_VNEXT_CONFIG.json` is the semantic config source |
| A1d | …`\FREEZE_RECEIPT.json`, `FREEZE_RECEIPT.sha256`, `PROVENANCE.json`, `SHA256SUMS.txt`, `README.md` | Seal/provenance | Do not modify; verify hashes post-copy |
| A1e | …`\validate_design.py`, `verify_bundle.py`, `seal_bundle.py`, `VALIDATION_RESULTS.json` | Validation tooling | |
| A2 | `C:\DaveLM-CADAVER\PHASE2A_CURRENT_HANDOFF.md` | **Authoritative current status** | Migrate as `HANDOFF.md` (or similar) |
| A3 | `C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1\` (top-level files + `data\`) | Authoritative T3 corpus + runner/report | Exclude `run_seed620001/2/3` (Category D). `data\` includes sealed `qa_test.jsonl` + `TEST_SEAL.json` — migrate but do not open |
| A4 | `C:\DaveLM-CADAVER\phase2a_t4_binding_sidecar_v1\` (top-level) | T4 treatment source/report | Exclude `run_seed630001/2/3` |
| A5 | `C:\DaveLM-CADAVER\phase2a_t5_latebase_binding_v1\` (top-level) | T5 treatment source/report | Exclude run dirs |
| A6 | `C:\DaveLM-CADAVER\phase2a_t6_latebase_pointer_v1\` (top-level) | T6 treatment source/report | Exclude run dirs |
| A7 | `C:\DaveLM-CADAVER\phase2a_t7_paired_reversal_pointer_v1\` (top-level) | T7 treatment source/report | Exclude run dirs |
| A8 | `C:\DaveLM-CADAVER\phase2a_t8_latebase_loc_layout_v1\` (top-level) | T8 treatment source/report | Exclude run dirs |
| A9 | `C:\DaveLM-CADAVER\phase2a_t9_factclause_pointer_v1\` (top-level) | T9 treatment source/report | Exclude run dirs |
| A10 | `C:\DaveLM-CADAVER\phase2a_t10_factclause_pointer_1k_v1\` (top-level) | T10 treatment source/report | Exclude run dirs |
| A11 | `C:\DaveLM-CADAVER\phase2a_t11_factclause_pointer_4to1_750_v1\` (top-level) | T11 treatment source/report | Exclude run dirs |
| A12 | `C:\DaveLM-CADAVER\phase2a_t12_factclause_pointer_lr2p5e5_750_v1\` (top-level) | T12 treatment source/report | Exclude run dirs |
| A13 | `C:\DaveLM-CADAVER\phase2a_t13_factclause_pointer_lr2p5e5_1k_v1\` (top-level) | T13 treatment source/report | Exclude run dirs |
| A14 | `C:\DaveLM-CADAVER\phase2a_t14_factclause_pointer_lr3p75e5_750_v1\` (top-level + `data\`) | **Last completed treatment** | Exclude `run_seed730001/2/3` |
| A15 | `C:\DaveLM-CADAVER\phase2a_t14x_latebase_decomp_v1\` | **Current live read-only decomposition** | Migrate only after process reaches terminal status |
| A16 | `C:\DaveLM-CADAVER\phase2a_final_pretreatment_diagnostic_v1\` | Phase2A pre-treatment diagnostic (D1 result) | Small, all files |
| A17 | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\` (top-level + `data\`) | Phase1G bundle + frozen language streams | Exclude `initialization\seed_610001_initialized.pt` (234.7 MB, Category D/F) |
| A18 | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001\PHASE1G_FINAL_REPORT.md`, `FINAL_STATUS.json`, `training_metrics.jsonl`, `evaluation_*.json` | Parent run reports/metrics | Exclude `checkpoints\` and `rolling_restart.pt` (Category D) |
| A19 | `C:\DaveLM-CADAVER\baby_vnext_phase1_language_v1\` (top-level + `data\`) | Phase1 bundle (input to Phase1G build) | Exclude `initialization\seed_610001_initialized.pt` |
| A20 | `C:\DaveLM-CADAVER\baby_talk.py` | Interactive current-checkpoint demo | Paths hardcoded; needs refactor |
| A21 | `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json`, `…\stats.json` | **Authoritative tokenizer** (external dependency) | SHA256 `e1c18bae…`; an identical copy exists at `C:\DaveLM-CADAVER\DaveLM-v0.9\tokenizer\v0_7\` |
| A22 | `C:\DaveLM-CADAVER\vnext_60m_audit\CODEX_VNEXT_HANDOFF.md` | Current design/operations handoff | Partly Category B; migrate with core docs |

---

## 4. Category B — Important research history

Material future coding agents should read to avoid repeating closed experiments. Migrate source,
protocols, reports, manifests, compact datasets, and headline results — **not** run checkpoints.

| # | Original source path | What it is |
|---|---|---|
| B1 | `C:\DaveLM-CADAVER\vnext_60m_audit\` (`BABY_10M_FULL_AUDIT.md`, `CODEX_VNEXT_HANDOFF.md`, `VNEXT_FILE_MAP.md`, `VNEXT_AUDIT_FACTS.json`) | Full 10.6M audit + vNext design brief + file map |
| B2 | `C:\DaveLM-CADAVER\baby_vnext_phase2a_ctxbind_v1\` (top-level + `data\`) | First Phase2A ctxbind bundle (T1 predecessor) |
| B3 | `C:\DaveLM-CADAVER\baby_vnext_phase2a_t2_ctxbind_v1\` (top-level + `data\`) | Phase2A T2 bundle (`PHASE2A_T2_FAIL`) |
| B4 | `C:\DaveLM-CADAVER\phase2a_t3_design_v1\` | T3 prospective design (never implemented) |
| B5 | `C:\DaveLM-CADAVER\phase2a_t3_committed_study_v1\` top-level only (`T3_FINAL_RECEIPT.json`, `T3_FINAL_REPORT.md`, `T3_RUNNER.py`, `PROTOCOL.json`, `SHA256SUMS.txt`) | T3 v1 `T3_HARD_STOP` record |
| B6 | `C:\DaveLM-CADAVER\phase2a_t3_committed_study_v2\` top-level only | T3 v2 record |
| B7 | `C:\DaveLM-CADAVER\phase2a_t3_repair_v1\` | T3 infrastructure-blocked repair report |
| B8 | `C:\DaveLM-CADAVER\baby_vnext_phase2a_t3_ctxbind_v1\ARCHITECTURE_PROVENANCE_NOTE.md` | Resolves the config-hash discrepancy (hash/serialization only) — important provenance evidence |
| B9 | `C:\DaveLM-CADAVER\baby_vnext_phase2a_t3_ctxbind_v1_runs\` | Compact T3 ctxbind run metrics (no checkpoints) |
| B10 | `C:\DaveLM-CADAVER\mistral_treatment_design_audit_v1\` | Independent treatment-design audit (SF1 ordinary-language failure) |
| B11 | `C:\DaveLM-CADAVER\overnight_reproducibility_resolution_v1\` (reports + `path1_evidence\*.py/.json/.md` + `path3_design\`; exclude `*.pt`) | ROCm reproducibility root-cause study + preregistered SF5 design |
| B12 | `C:\DaveLM-CADAVER\comparative_model_anatomy_post_sf2_v1\` | Comparative anatomy vs Granite/Qwen/LFM2 |
| B13 | `C:\DaveLM-CADAVER\reference_model_research\nemotron_3_nano_4b_comparative_audit_v1\` | External reference-model research |
| B14 | `C:\DaveLM-CADAVER\forensics\` (`*.py`, `*.md`, `*.json`) | Bounded-shared-antisymmetric collapse autopsy, source-order asymmetry, freeze completion audit |
| B15 | Root-level experiment source (10.6M era): `treatment4_*` … `treatment13_*`, `query_swap_*.py`, `selection_margin_*.py`, `curriculum_bridge.py`, `brain_compare.py`, `sf1_build.py`, `sf1_engine.py`, `sf1_report.py`, `BUILD_SF21.py`, `t5_t8_structural_census.py`, `measure_binding_descendants.py`, `readiness_dev_baseline.py`, `summarize_results.py`, `analyze_pathology.py`, `build_*` / `validate_*` / `finalize_*` / `freeze_*` / `seal_*` / `run_*` / `execute_*` HR, fact-supervision, post-P7 and diagnostic scripts under `C:\DaveLM-CADAVER\` | Experiment source (all `C:\DaveLM-CADAVER\*.py` at root) |
| B16 | Root-level history reports: `P7_LINEAGE_AND_REPORT_CARD_ERRATUM_20260905.md`, `POST_P7_INTEGRITY_STOP_AUDIT.md`, `GENERATION_PATHOLOGY_REPORT.md`, `t5_t8_structural_census_report.md`, `human_readiness_binding_measurement.json`, `generation_pathology_*.json` (except the 133 MB census, see D) | Audits, errata, compact results |
| B17 | `C:\DaveLM-CADAVER\sf1_readout_selection_forensic_v1\`, `sf1_component_swap_forensic_v1\`, `sf1_upstream_localization_forensic_v1\`, `sf1_candidate_pool_hypothesis_audit_v1\`, `sf1_token_position_patching_forensic_v1\` (reports/protocols/data; exclude large `.pt`) | SF1 forensics |
| B18 | `C:\DaveLM-CADAVER\sf2_*` … `sf21_*` directories: all top-level protocols, engines, controllers, datasets, reports, `RESULTS.json`, `RUN_LEDGER.json`, `FINAL_REPORT.md`, `STATUS.json`; **exclude** `runs\`, `checkpoints\`, `restart.pt`, `*.pt` | Closed factual-supervision series (SF2–SF21) |
| B19 | `C:\DaveLM-CADAVER\language_pilot_0_tinystories_seed8380\`, `language_pilot_1_early_block_protection_seed8380\` (reports, pools `binding_dev.json`, `binding_rehearsal.json`, `language_train.jsonl`, `language_dev.jsonl`, `prepare.py`, `run.py`; exclude checkpoints) | Language parent lineage + protected binding pools |
| B20 | `C:\DaveLM-CADAVER\language_continuation_p0_v1\`, `language_consistency_p1\`, `language_mixed_p2\`, `language_storybound_p3\`, `language_unlikelihood_p4\`, `language_sentencebound_p5\`, `language_compositional_p6\`, `language_compositional_p7\` (reports/data; exclude `latest.pt`) | P0–P7 language ladder |
| B21 | `C:\DaveLM-CADAVER\post_p7_language_report_card_v1/v2/v3*_seed8380/8391\` and `..._execution*` (reports/data; exclude `.pt`) | P7 report-card battery + execution records |
| B22 | `C:\DaveLM-CADAVER\fact_supervision_87001*\` family (reports/protocols/data; exclude `.pt`/`rolling_restart.pt`) | Fact-supervision lineage |
| B23 | `C:\DaveLM-CADAVER\human_readiness_hr1*`, `human_readiness_hr2*`, `human_readiness_hr3*`, `human_test_readiness_v1`, `human_test_readiness_v2_seed87010` (reports/protocols/data; exclude checkpoints) | Human-readiness studies |
| B24 | `C:\DaveLM-CADAVER\hr3_context_autopsy_v1\`, `hr3_diagnostic_staircase_v1\`, `hr3_diagnostic_staircase_v1_results\`, `english_context_characterization_v1_seed8380*`, `single_fact_acquisition_sf1_seed87011*` (reports/data; exclude `.pt`) | HR3 context/autopsy and related |
| B25 | `C:\DaveLM-CADAVER\treatment4_*` … `treatment13_*` directories (top-level reports/protocols/configs; exclude `checkpoints\`) | 10.6M treatment experiments |
| B26 | `C:\DaveLM-CADAVER\two_mapping_curriculum_bridge_manual_seed8380\`, `two_mapping_membership_margin_manual_seed8380\`, `two_mapping_query_swap_manual_seed8380\`, `two_mapping_selection_margin_manual_seed8380\` (top-level; exclude `checkpoints\`) | Manual two-mapping curriculum experiments |
| B27 | `C:\DaveLM-CADAVER\archive\DAVELM_P7_MILESTONE_seed8380\ARCHIVE_RECORD.json`, `ARCHIVE_RECORD.md` | Archived P7 milestone record (the `.pt` is Category D) |
| B28 | `C:\DaveLM-CADAVER\treatment4_eval_dump.txt` (as reference only — see E for duplicate dumps) | Historical eval dump |

---

## 5. Category C — Historical baseline

Older DaveLM versions/components useful for reproducibility/reference but not the current architecture.

| # | Original source path | What it is |
|---|---|---|
| C1 | `C:\DaveLM-CADAVER\DaveLM-v0.9\` (whole tree) | **Stale nested snapshot** (copied ~2026-09-01 23:56). File-list diff vs `C:\DaveLM-v0.9`: nested has 338 files; external has 492. Nested is missing 154 later files (e.g. `experiments\counterfactual_identity_transport_ab`, `experiments\minimal_contextual_binding`, `experiments\relation_level_contextual_binding_ab`, `experiments\two_mapping_*`, `forensics\frozen_*`, plus later `__pycache__`). Core files verified byte-identical (tokenizer, `v0_8\config.py`, `v0_8_2\config.py`, `v0_9\dense_geometry.py`). |
| C2 | `C:\DaveLM-v0.9\v0_2`, `v0_2_1`, `v0_7`, `v0_8`, `v0_8_1`–`v0_8_4`, `v0_9`, `tests`, `tokenizer` (source only) | Canonical 10.6M baseline source. **Note:** `v0_8\model.py` does not exist; the operative model is `v0_7\model.py` (imported by `v0_8_2` wrapper) per `BABY_VNEXT_IMPLEMENTATION_REPORT.md`. |
| C3 | `C:\DaveLM-v0.9\experiments\*` (top-level configs/reports/corpora; exclude `checkpoints\`, `latest.pt`) | v0.8.4/v0.9 experiments (identity/tied-copy/transport families) |
| C4 | `C:\DaveLM-v0.9\forensics\*` (top-level; exclude checkpoints) | Untied stage forensics + frozen probes |
| C5 | `C:\DaveLM-v0.9\validation\dense_geometry_copy_gate\` (top-level; exclude checkpoints) | v0.9 integration gate |
| C6 | `C:\DaveLM-v0.9\README.md`, `V0_9_TECHNICAL_NOTE.md`, `V0_9_FILES.json`, `baby_playground.py` | Baseline docs + playground |
| C7 | `C:\DaveLM-CADAVER\language_pilot_0_tinystories_seed8380\pilot_run\checkpoints\seed_8380\latest.pt`, `language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt` | Pilot0/Pilot1 checkpoints (also Category D/F due to size) |
| C8 | `C:\DaveLM-CADAVER\treatment13_orthogonal_shared_unbounded_seed8380\` (report/addendum + `checkpoints\...\latest.pt`) | Canonical binding graduate (report = B; checkpoint = D/F) |
| C9 | `C:\DaveLM-CADAVER\sf8_margin_dose_comparison_v1\` (protocol/report; seed checkpoints = D/F) | SF8 standing recipe + seed parents of SF20/SF21 |

**Recommendation:** do not migrate both `C:\DaveLM-CADAVER\DaveLM-v0.9` and `C:\DaveLM-v0.9`.
Consolidate one baseline copy from the **canonical external tree** (C2–C6), and preserve the nested
snapshot only if its provenance value is judged worth it (see Category E/F).

---

## 6. Category D — Local-only artifacts

These should generally **not** be committed. Preserve locally (or with Git LFS / release assets) per
human decision. Original paths:

| # | Original source path pattern | Size / count | What it is |
|---|---|---|---|
| D1 | `C:\DaveLM-CADAVER\sf2_runtime\` | **9.0 GB** | Bundled CPython 3.12.14 (`cpython31214\`) + `sf2venv\` with torch 2.12.0+rocm7.14.0 |
| D2 | `C:\DaveLM-CADAVER\baby_vnext_phase1_language_v1_run_seed610001\` | 3.66 GB | Phase1 run (checkpoints, `rolling_restart.pt` 696.8 MB) |
| D3 | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001\` | 3.66 GB | Phase1G parent run (incl. `best.pt` 234.7 MB + 12 checkpoints + 696.8 MB restart) |
| D4 | All `C:\DaveLM-CADAVER\phase2a_*_run_seed*\` and `C:\DaveLM-CADAVER\baby_vnext_phase2a_*_run_seed*\` | ~1.0–1.7 GB each, ~40 dirs | T2–T14 run artifacts (`checkpoints\`, `rolling_restart.pt`, `item_results_*.json`, `training_metrics.jsonl`) |
| D5 | `C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1\run_seed620001/2/3\` | 1.37 GB each | T3 rebuilt runs |
| D6 | `C:\DaveLM-CADAVER\phase2a_t3_committed_study_v1\run_seed*\` + `phase2a_t3_committed_study_v2\run_seed*\` | 2.06 GB + 0.69 GB | Superseded T3 runs (`T3_HARD_STOP`, identical checkpoints) |
| D7 | `C:\DaveLM-CADAVER\baby_vnext_60m_design_v1\validation\initialized_checkpoint_roundtrip.pt` | 246.1 MB | Zero-update round-trip validation artifact |
| D8 | `C:\DaveLM-CADAVER\baby_vnext_phase1_language_v1\initialization\seed_610001_initialized.pt` and `…phase1g_language_v1\initialization\seed_610001_initialized.pt` | 246.1 MB each | Deterministic init checkpoints (byte-identical role) |
| D9 | `C:\DaveLM-CADAVER\t5_t8_structural_census.json` | **133 MB** | Large raw structural census JSON |
| D10 | `C:\DaveLM-CADAVER\sf8_margin_dose_comparison_v1\` (2.37 GB), `sf6_same_seed_lr_anneal_v1\` (1.58 GB), `sf7_pairwise_margin_hinge_v1\` (1.46 GB), `sf9`–`sf21` run/checkpoint dirs | ~10 GB total | Closed-series run artifacts |
| D11 | `C:\DaveLM-CADAVER\treatment5_*` … `treatment13_*` checkpoint/run dirs | ~15 GB total | 10.6M treatment run artifacts |
| D12 | `C:\DaveLM-CADAVER\language_*\latest.pt` (p0–p7), `human_readiness_*\` runs, `fact_supervision_*\` runs, `two_mapping_*\checkpoints\`, `archive\...\davelm_p7_latest.pt` (64.5 MB) | ~10 GB total | Historical checkpoints |
| D13 | `C:\DaveLM-CADAVER\DaveLM-v0.9\experiments\**\checkpoints\*.pt`, `forensics\**\*.pt`, `validation\**\*.pt` | ~3.5 GB | v0.9 checkpoint artifacts |
| D14 | `C:\DaveLM-CADAVER\**\__pycache__\` and `*.pyc` | ~0.3 GB+ | Bytecode caches |
| D15 | `C:\DaveLM-CADAVER\*.log`, `*_watchdog.log`, `*_train_seed*.log`, `*.err`, `*.out` | small | Training/watchdog logs |
| D16 | `C:\DaveLM-CADAVER\9bqwen3.5logs.txt` | 704 KB | External Qwen server logs (local-only) |
| D17 | `C:\DaveLM-CADAVER\overnight_reproducibility_resolution_v1\path1_evidence\probe_1a|probe_1b\*.pt` | ~0.4 GB | Probe checkpoints |
| D18 | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\LANGUAGE_TRAIN_SOURCE.jsonl` | 55.7 MB | Generated tokenizer source dataset (F) |

---

## 7. Category E — Redundant / probably unnecessary

| # | Original source path | Why |
|---|---|---|
| E1 | `C:\DaveLM-CADAVER\DaveLM-v0.9\` (entire nested tree) | Stale duplicate of `C:\DaveLM-v0.9` (see C1). Core files identical; 154 newer files missing. Only unique content is 2 stale `.pyc` files. |
| E2 | `C:\DaveLM-CADAVER\baby_vnext_phase2a_t3_ctxbind_v1\data\` and `…_runs\` | Invalidated T3 ctxbind attempt (train/dev/test name overlap). Keep only `ARCHITECTURE_PROVENANCE_NOTE.md` (B8). |
| E3 | `C:\DaveLM-CADAVER\phase2a_t3_committed_study_v1\run_seed620001*, run_seed620002, run_seed620003`, `phase2a_t3_committed_study_v2\run_seed620001` | Superseded T3 runs; `T3_HARD_STOP` with identical checkpoints. Keep top-level records only. |
| E4 | `C:\DaveLM-CADAVER\hr3_diagnostic_staircase_v1_build1_failed\`, `human_readiness_hr3_block3_causal_seed87006_v1_build_attempt{1,2,3}_failed\`, `…execution_v4_resume1_failed\`, `…execution_v5_failed_cuda_rng\`, `…execution_attempt1_uncommitted\` | Failed builds/attempts |
| E5 | `C:\DaveLM-CADAVER\human_readiness_hr1_seed87004_v2`–`v8`, `human_readiness_hr2_seed87005_v2`–`v7`, `human_readiness_hr3_block3_causal_seed87006_v2`–`v7` (superseded versions; keep latest + reports) | Iterative duplicate attempts |
| E6 | `C:\DaveLM-CADAVER\fact_supervision_87001_corrected_v2`–`v7` (keep `v8` and `_corrected`) | Superseded reseals |
| E7 | `C:\DaveLM-CADAVER\post_p7_language_report_card_v3b/v3c/v3d_seed8391` (superseded intermediates; v3d execution is the used one — human review) | Iterative report-card versions |
| E8 | `C:\DaveLM-CADAVER\sf12_answer_vocab_retention_v1_reporting_hotfix_v1\`, `sf14_first_answer_token_ce_ablation_v1_reporting_hotfix_v1\`, `sf15_partial_first_token_ce_v1_reporting_hotfix_v1\`, `sf16_execution_reporting_addendum_v1\`, `sf13_execution_v1\` | Reporting hotfixes / execution wrappers |
| E9 | `C:\DaveLM-CADAVER\sf3_prospective_treatment_v1+\` (directory name contains `+`) | Obsolete duplicate of `sf3_prospective_treatment_v1` |
| E10 | `C:\DaveLM-CADAVER\DaveLM-v0.9\experiments\identity_diversity_fixed_exposure_factorial\preflight_attempt_before_pool_matching_fix\`, `…preflight_attempt_before_initialization_matching_fix\`, and `…latest.pt.failed_atomic_save_step10` | Failed preflight attempt copies (141.7 MB each) |
| E11 | `C:\DaveLM-CADAVER\python1.txt`, `python2.txt`, `treatment4_eval_dump.txt` | Duplicated source-code dumps |
| E12 | `C:\DaveLM-CADAVER\.aider.chat.history.md`, `.aider.input.history` | Agent chat/input history (also secret-risk; see §11) |
| E13 | `C:\DaveLM-CADAVER\__pycache__\` and all other `__pycache__\` dirs / `*.pyc` | Build caches |
| E14 | `C:\DaveLM-CADAVER\9bqwen3.5logs.txt` | Local external-server logs |
| E15 | Duplicate `ENGLISH_TRAIN.jsonl` (29.4 MB each) across ~20 HR version dirs | Identical regenerated datasets in superseded dirs |
| E16 | `C:\DaveLM-CADAVER\baby_vnext_phase2a_t3_ctxbind_v1_runs\` (metrics only) | Invalidated T3 ctxbind attempt |

---

## 8. Category F — Needs human review

| # | Original source path | Question |
|---|---|---|
| F1 | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001\checkpoints\best.pt` (234.7 MB) | The current parent. Git LFS, release asset, or local-only? |
| F2 | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\initialization\seed_610001_initialized.pt` (234.7 MB) | Deterministic init artifact; needed for exact reproduction? |
| F3 | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\LANGUAGE_TRAIN_STREAM.u16` (16.7 MB), `LANGUAGE_DEV_STREAM.u16` (0.8 MB), `TRAIN_WINDOW_STARTS.u32` (1.5 MB), `EVAL_WINDOW_STARTS.u32`, `GENERATION_PROMPTS.json`, `LANGUAGE_TRAIN_SOURCE.jsonl` (55.7 MB) | Commit generated streams vs. regenerate from source JSONL? All are required by T3–T14. |
| F4 | `C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1\data\qa_test.jsonl` + `TEST_SEAL.json` | Sealed test set. Must migrate to preserve the seal, but never open/score. Confirm handling. |
| F5 | T3–T14 run checkpoints (`…\checkpoints\checkpoint_*.pt`, 234.7 MB each) | Which terminal checkpoints must be preserved (T14 U750, T13 U750/U1000, etc.) vs. excluded? |
| F6 | `C:\DaveLM-CADAVER\treatment13_orthogonal_shared_unbounded_seed8380\checkpoints\...\latest.pt` (64.5 MB) | Canonical binding graduate; keep with history? |
| F7 | `C:\DaveLM-CADAVER\sf8_margin_dose_comparison_v1\runs\seed_8701{7,8,9}_low\checkpoint_200.pt` | Parents of SF20/SF21; required for closed-series reproduction? |
| F8 | `C:\DaveLM-CADAVER\language_pilot_0/1...\latest.pt` (64.5 MB each) | Pilot language parents; historical value vs. size |
| F9 | `C:\DaveLM-CADAVER\t5_t8_structural_census.json` (133 MB) | Raw census data: commit (LFS), compress, or exclude with report only? |
| F10 | `C:\DaveLM-CADAVER\.aider.chat.history.md` (269.5 KB) | Contains API-key-like strings; scrub/archive privately vs. delete |
| F11 | `C:\DaveLM-CADAVER\.config\opencode\tunnel.json` | Local tunnel config (url/tunnelId); exclude from repo; confirm no live credential |
| F12 | `C:\DaveLM-CADAVER\mistral_test.py`, `mistral_baby_history.json` | External API test + history; check for embedded keys before any commit |
| F13 | `C:\DaveLM-CADAVER\generation_pathology_constraints.json` (610.8 KB) and related pathology JSONs | Compact enough to commit; confirm research value |
| F14 | `C:\DaveLM-CADAVER\DaveLM-v0.9\` vs `C:\DaveLM-v0.9\` | Which becomes the canonical baseline copy in the new repo? |
| F15 | `C:\DaveLM-CADAVER\sf21_paired_counterfactual_representation_consistency_v2\FINAL_REPORT.md`, `RESULTS.json`, `runs\...` | Closure evidence for the 10.6M series; keep compact results, exclude checkpoints? |
| F16 | `C:\DaveLM-CADAVER\post_p7_language_report_card_v3d_seed8391\` + execution | The actually-used v3d battery (vs. v3/v3b/v3c) |
| F17 | `C:\DaveLM-CADAVER\9bqwen3.5logs.txt` | May contain prompts/outputs; exclude or archive |
| F18 | `C:\DaveLM-CADAVER\phase2a_t3_committed_study_v1\data\corpus.json` / `baby_vnext_phase2a_t3_ctxbind_v1\data\corpus.json` (381 KB) | Superseded corpus; keep for audit trail? |

---

## 9. Dependency map

Absolute paths below are the **current hard-coded dependencies** in source. They are the reason a plain
copy of the directories is not sufficient.

```
CURRENT (vNext Phase 2A)
  phase2a_t3..t14 and t14x Runners
    ├─ sys.path.insert -> C:\DaveLM-CADAVER\baby_vnext_60m_design_v1
    │      └─ baby_vnext.config, baby_vnext.binding, baby_vnext.checkpoint
    ├─ PARENT -> C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001\checkpoints\best.pt
    │      └─ SHA256 c5406f80… (also embedded in every PROTOCOL.json)
    ├─ T3_DATA -> C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1\data
    │      ├─ qa_train.jsonl, qa_dev.jsonl, acq16.json, panels.json
    │      ├─ qa_schedules.json, rehearsal_index.json, update_kinds.json
    │      └─ qa_test.jsonl + TEST_SEAL.json (SEALED_UNOPENED; never read by runners)
    ├─ LANG_TRAIN/DEV/STARTS/PROMPTS -> C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\
    ├─ TOK -> C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json   [outside source root]
    └─ Runtime -> C:\DaveLM-CADAVER\sf2_runtime\sf2venv\Scripts\python.exe  [local-only]

  phase2a_t14x_latebase_decomp_v1 (READ-ONLY)
    ├─ same parent/T3_DATA/ARCH/tokenizer as above
    └─ reads terminal checkpoints of:
         phase2a_t10_..._run_seed690003\checkpoints\checkpoint_0500.pt, _0750.pt
         phase2a_t12_..._run_seed710001\checkpoints\checkpoint_0500.pt, _0750.pt
         phase2a_t13_..._run_seed720002/3\checkpoints\checkpoint_0750.pt (720003 also _1000.pt)
         phase2a_t14_..._run_seed730001/2/3\checkpoints\checkpoint_0500.pt, _0750.pt

  baby_vnext_phase1g build_phase1g_bundle.py
    ├─ ARCH -> baby_vnext_60m_design_v1
    ├─ P1  -> C:\DaveLM-CADAVER\baby_vnext_phase1_language_v1
    │      (reads data\LANGUAGE_TRAIN_STREAM.u16; copies dev stream, EVAL_WINDOW_STARTS,
    │       GENERATION_PROMPTS, initialization\seed_610001_initialized.pt)
    └─ TOKENIZER -> C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json

  baby_vnext_phase1 build_phase1_bundle.py
    ├─ ARCH -> baby_vnext_60m_design_v1
    ├─ sources -> C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\
    │      language_train.jsonl (24.98 MB), language_dev.jsonl (2.73 MB)
    └─ TOKENIZER -> C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json

  baby_vnext_phase2a_ctxbind / t2 builds
    ├─ ARCH -> baby_vnext_60m_design_v1
    ├─ P1/P1RUN -> phase1g bundle + run best.pt
    ├─ TOKENIZER -> C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json
    └─ t2 build also references phase2a_ctxbind_v1 (v1) data

  baby_talk.py
    ├─ ARCH -> baby_vnext_60m_design_v1
    ├─ CHECKPOINT -> phase1g run best.pt
    ├─ INIT_CHECKPOINT -> phase1g initialization\seed_610001_initialized.pt
    └─ TOKENIZER -> C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json

HISTORICAL (closed 10.6M series)
  SF1..SF21 -> language_pilot_1\pilot_run\checkpoints\seed_8380\latest.pt (SHA 2281d20e…)
      -> SF8 low-dose seeds 87017/87018/87019 (parents of SF20/SF21)
      -> v0_7.model.DaveLM + v0_8\config.py + v0_8_2 untied wrapper
  T4..T13   -> same Pilot1 parent (per vnext_60m_audit\VNEXT_FILE_MAP.md)
  P7 line   -> language_sentencebound_p5\latest.pt (immediate parent; see P7 erratum)
  tokenizer -> C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json
```

**Key migration implication:** the tokenizer lives **outside** `C:\DaveLM-CADAVER` (in
`C:\DaveLM-v0.9`). The clean repo must vendor the tokenizer (or vendor the identical nested copy) and
add a path-root indirection so runners stop depending on `C:\DaveLM-CADAVER` and `C:\DaveLM-v0.9`.

---

## 10. Large-file / GitHub-risk findings

- **Total source tree:** ~103.2 GB, 36,859 files. Almost all bytes are checkpoints/venv.
- **GitHub hard limit (100 MB/file)** is exceeded by many files; the largest are:

| Size | Original path (pattern) |
|---:|---|
| 696.8 MB | `baby_vnext_phase1g_language_v1_run_seed610001\rolling_restart.pt` |
| 696.8 MB | `baby_vnext_phase1_language_v1_run_seed610001\rolling_restart.pt` |
| 696.7 MB | `baby_vnext_phase2a_t2_ctxbind_v1_run_seed610005\rolling_restart.pt` |
| 696.7 MB | `baby_vnext_phase2a_ctxbind_v1_run_seed610002\rolling_restart.pt` |
| 696.7 MB ×3 | `phase2a_t3_rebuilt_study_v1_run_seed62000{1,2,3}\rolling_restart.pt` |
| 554 MB ×6 | `phase2a_t5_*` / `phase2a_t8_*` run `rolling_restart.pt` |
| 546.5 MB ×~30 | `phase2a_t6/t7/t9/t10/t11/t12/t13/t14_*_run_seed*\rolling_restart.pt` |
| 246.1 MB ×~15 | vNext `checkpoint_*.pt` (Phase1/1G/T3–T14) |
| 242.3 MB ×3 | `phase2a_t4_binding_sidecar_v1_run_seed63000{1,2,3}\rolling_restart.pt` |
| 234.7 MB | `baby_vnext_60m_design_v1\validation\initialized_checkpoint_roundtrip.pt` |
| 234.7 MB | Phase1/1G `initialization\seed_610001_initialized.pt` and `checkpoints\best.pt` |
| 151.8 MB | `forensics\bounded_internal_mechanism_diagnostic_seed8380\INTERNAL_MEASUREMENTS.pt` |
| 146.4 MB ×10 | `treatment9_answer_position_retrieval_seed8380\checkpoints\...\*.pt` |
| 145.4 MB ×10 | `treatment7_direct_query_logit_pathway_seed8380\checkpoints\...\*.pt` |
| 144.5 MB ×~90 | SF3–SF21 / HR / overnight / single-fact `restart.pt` |
| 141.7 MB ×~25 | v0.9 experiment checkpoints + `latest.pt.failed_atomic_save_step10` |
| **133 MB** | `C:\DaveLM-CADAVER\t5_t8_structural_census.json` (largest non-checkpoint file) |
| 55.7 MB | `baby_vnext_phase1g_language_v1\data\LANGUAGE_TRAIN_SOURCE.jsonl` |
| 42.2/35.5/27.2 MB ×3 | v0.9 `identity_diversity_fixed_exposure_factorial\corpus\arm_*.json` (+failed preflight copies) |
| 36.4 MB | `treatment5_counterfactual_pairs_seed8382\treatment5_full_document_frozen_schedule.json` |
| 29.4 MB ×~20 | Duplicate `ENGLISH_TRAIN.jsonl` in HR version dirs |

- **GitHub warning limit (50 MB)** is also exceeded by many small-checkpoint/dataset files.
- **Recommended:** `.gitignore` all `*.pt`/`*.pth`, `rolling_restart.pt`, `sf2_runtime/`, `__pycache__/`,
  and run directories. If any checkpoints must be versioned, use **Git LFS** or publish them as release
  assets; document SHA256s in manifests instead. Large JSON (>50 MB) should be compressed, sampled, or
  LFS-tracked.
- Note: `C:\DaveLM-CADAVER` is **not** a Git repo (no `.git` found anywhere in the tree).

---

## 11. Secret / security-risk findings

No `.env`, private keys (`id_rsa`, `*.pem`, `*.key`, `*.pfx`), or credential files were found in the
source tree (excluding the bundled Python venv's normal `pip/_vendor/certifi/cacert.pem`, which is not a
secret). The following files warrant review; **contents were not printed**:

| Risk | File | Finding | Recommendation |
|---|---|---|---|
| HIGH | `C:\DaveLM-CADAVER\.aider.chat.history.md` (269.5 KB) | Matches API-key-like patterns and provider names (Aider session history) | Do **not** commit. Inspect/scrub/archive privately. Treat as sensitive. |
| HIGH | `C:\DaveLM-CADAVER\mistral_test.py` | References an API-key environment variable and provider auth | Review for hard-coded key; commit only after scrubbing to `os.environ[...]` |
| MEDIUM | `C:\DaveLM-CADAVER\mistral_baby_history.json` | External model conversation history (root is a JSON array) | Review for sensitive content before any commit |
| MEDIUM | `C:\DaveLM-CADAVER\.config\opencode\tunnel.json` | Local tunnel config with `url`, `tunnelId`, `provider`, ports | Exclude from repo; confirm the tunnel is not a live credential |
| LOW | `C:\DaveLM-CADAVER\.aider.input.history` | Aider prompt history | Exclude |
| LOW | `C:\DaveLM-v0.9\opencode.json` (external) | Provider config only (`lmstudio.*`); no key values observed | Safe, but it is local tool config — do not vendor |
| INFO | `C:\DaveLM-CADAVER\sf2_runtime\...\pip\_vendor\certifi\cacert.pem` | Vendored CA bundle | Not a secret; excluded anyway |
| INFO | `C:\DaveLM-CADAVER\sf1_engine.py` | Uses `os.environ` (env-var read, no literal key observed) | Low risk |

**Recommendation:** before the first commit, run a secret scanner (e.g. gitleaks/trufflehog) over the
staged tree, and rotate any key that ever appeared in `.aider.chat.history.md`.

---

## 12. Proposed clean repository structure for `C:\DaveLM - Baby Stage`

Proposed layout (names illustrative; relative paths preserved where source imports allow, otherwise
refactored to a configurable root):

```
DaveLM-Baby-Stage\
├─ README.md                        (new; points to docs/handoff)
├─ HANDOFF.md                       <- PHASE2A_CURRENT_HANDOFF.md
├─ .gitignore
├─ .gitattributes                   (LFS rules, optional)
├─ docs\
│  ├─ design\                       <- baby_vnext_60m_design_v1\*.md, *.json (spec, config, freeze)
│  ├─ audits\                       <- vnext_60m_audit\, phase2a_final_pretreatment_diagnostic_v1\,
│  │                                   mistral_treatment_design_audit_v1\,
│  │                                   comparative_model_anatomy_post_sf2_v1\,
│  │                                   reference_model_research\, overnight_reproducibility_resolution_v1\
│  ├─ history\                      <- P7 erratum, POST_P7 audit, generation pathology, SF21 closure
│  └─ legacy_10m\                   <- 10.6M series reports/protocols (no checkpoints)
├─ src\
│  └─ baby_vnext\                   <- baby_vnext_60m_design_v1\baby_vnext\ (config/model/binding/checkpoint)
├─ tools\
│  └─ baby_talk.py
├─ experiments\
│  ├─ phase1\                       <- baby_vnext_phase1_language_v1\ (bundle, data; no init .pt)
│  ├─ phase1g\                      <- baby_vnext_phase1g_language_v1\ (bundle, data; no init .pt)
│  │  └─ run_seed610001_reports\    <- FINAL_STATUS, PHASE1G_FINAL_REPORT.md, metrics, eval json
│  ├─ phase2a\
│  │  ├─ pretreatment_diagnostic\   <- phase2a_final_pretreatment_diagnostic_v1\
│  │  ├─ t1_ctxbind\                <- baby_vnext_phase2a_ctxbind_v1\
│  │  ├─ t2_ctxbind\                <- baby_vnext_phase2a_t2_ctxbind_v1\
│  │  ├─ t3_design\                 <- phase2a_t3_design_v1\
│  │  ├─ t3_committed\              <- phase2a_t3_committed_study_v1|v2 (top-level only)
│  │  ├─ t3_repair\                 <- phase2a_t3_repair_v1\
│  │  ├─ t3_rebuilt\                <- phase2a_t3_rebuilt_study_v1\ (top-level + data\)
│  │  ├─ t4 ... t14\                <- phase2a_t4_binding_sidecar_v1\ ... phase2a_t14_...\ (top-level + data\)
│  │  └─ t14x_decomp\               <- phase2a_t14x_latebase_decomp_v1\
│  └─ legacy_10m\                   <- root treatment*/sf*/language_*/hr*/fact_supervision*/two_mapping*/
│                                      post_p7*/english_context*/single_fact*/archive*/forensics* source + reports
├─ data\
│  ├─ tokenizer\                    <- C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json + stats.json
│  └─ language\                     <- Phase1/1G streams, window starts, prompts, source jsonl
│                                      (or regenerate; see .gitignore/LFS)
├─ legacy\
│  └─ davelm_v0_9\                  <- canonical C:\DaveLM-v0.9 source (v0_2..v0_9, tests, tokenizer,
│                                      experiments/forensics/validation source only; no checkpoints)
├─ manifests\                       <- SHA256SUMS.txt, PROVENANCE.json copies, migration manifest
└─ artifacts\                       (gitignored; local checkpoints/runs; not created by migration)
```

**Path-portability requirement:** because all runners hard-code `C:\DaveLM-CADAVER` and
`C:\DaveLM-v0.9`, either (a) keep the source absolute-path layout inside the new repo and add a
one-time `Path` shim, or (b) refactor every runner to a `DAVELM_ROOT` environment variable. Option (b)
is strongly recommended before publishing.

---

## 13. Proposed migration manifest

Legend: **Action** — COPY = migrate as source; COPY-PARTIAL = migrate selected files only; EXCLUDE =
do not migrate; REVIEW = human decision required.

### 13.1 Core current (Category A)

| Action | Original source path | Proposed destination |
|---|---|---|
| COPY | `C:\DaveLM-CADAVER\baby_vnext_60m_design_v1\` | `src\baby_vnext\` + `docs\design\` + `manifests\` |
| EXCLUDE | `…\baby_vnext_60m_design_v1\__pycache__\` | — |
| EXCLUDE | `…\baby_vnext_60m_design_v1\validation\initialized_checkpoint_roundtrip.pt` | — (D7) |
| COPY | `C:\DaveLM-CADAVER\PHASE2A_CURRENT_HANDOFF.md` | `HANDOFF.md` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1\` (all top-level + `data\`) | `experiments\phase2a\t3_rebuilt\` |
| EXCLUDE | `…\phase2a_t3_rebuilt_study_v1\run_seed62000{1,2,3}\` | — (D5) |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\phase2a_t4_binding_sidecar_v1\` (top-level + `data\`) | `experiments\phase2a\t4\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\phase2a_t5_latebase_binding_v1\` (top-level + `data\`) | `experiments\phase2a\t5\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\phase2a_t6_latebase_pointer_v1\` (top-level + `data\`) | `experiments\phase2a\t6\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\phase2a_t7_paired_reversal_pointer_v1\` (top-level + `data\`) | `experiments\phase2a\t7\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\phase2a_t8_latebase_loc_layout_v1\` (top-level + `data\`) | `experiments\phase2a\t8\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\phase2a_t9_factclause_pointer_v1\` (top-level + `data\`) | `experiments\phase2a\t9\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\phase2a_t10_factclause_pointer_1k_v1\` (top-level + `data\`) | `experiments\phase2a\t10\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\phase2a_t11_factclause_pointer_4to1_750_v1\` (top-level + `data\`) | `experiments\phase2a\t11\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\phase2a_t12_factclause_pointer_lr2p5e5_750_v1\` (top-level + `data\`) | `experiments\phase2a\t12\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\phase2a_t13_factclause_pointer_lr2p5e5_1k_v1\` (top-level + `data\`) | `experiments\phase2a\t13\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\phase2a_t14_factclause_pointer_lr3p75e5_750_v1\` (top-level + `data\`) | `experiments\phase2a\t14\` |
| EXCLUDE | All `…\phase2a_t*_run_seed*\` | — (D4/D5) |
| COPY | `C:\DaveLM-CADAVER\phase2a_t14x_latebase_decomp_v1\` | `experiments\phase2a\t14x_decomp\` |
| COPY | `C:\DaveLM-CADAVER\phase2a_final_pretreatment_diagnostic_v1\` | `experiments\phase2a\pretreatment_diagnostic\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\` (top-level + `data\`) | `experiments\phase1g\` + `data\language\` |
| EXCLUDE | `…\baby_vnext_phase1g_language_v1\initialization\seed_610001_initialized.pt` | — (D8, F2) |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001\` (report/status/metrics/eval only) | `experiments\phase1g\run_seed610001_reports\` |
| EXCLUDE | `…\baby_vnext_phase1g_language_v1_run_seed610001\checkpoints\` and `rolling_restart.pt` | — (D3, F1) |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\baby_vnext_phase1_language_v1\` (top-level + `data\`) | `experiments\phase1\` |
| EXCLUDE | `…\baby_vnext_phase1_language_v1\initialization\seed_610001_initialized.pt` | — (D8) |
| COPY | `C:\DaveLM-CADAVER\baby_talk.py` | `tools\baby_talk.py` (refactor paths) |
| COPY | `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json` + `stats.json` | `data\tokenizer\` |
| COPY | `C:\DaveLM-CADAVER\vnext_60m_audit\CODEX_VNEXT_HANDOFF.md` | `docs\audits\` |

### 13.2 Research history (Category B)

| Action | Original source path | Destination |
|---|---|---|
| COPY | `C:\DaveLM-CADAVER\vnext_60m_audit\` | `docs\audits\vnext_60m_audit\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\baby_vnext_phase2a_ctxbind_v1\` (top-level + `data\`) | `experiments\phase2a\t1_ctxbind\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\baby_vnext_phase2a_t2_ctxbind_v1\` (top-level + `data\`) | `experiments\phase2a\t2_ctxbind\` |
| COPY | `C:\DaveLM-CADAVER\phase2a_t3_design_v1\` | `experiments\phase2a\t3_design\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\phase2a_t3_committed_study_v1\` (top-level only) | `experiments\phase2a\t3_committed_v1\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\phase2a_t3_committed_study_v2\` (top-level only) | `experiments\phase2a\t3_committed_v2\` |
| COPY | `C:\DaveLM-CADAVER\phase2a_t3_repair_v1\` | `experiments\phase2a\t3_repair\` |
| COPY | `C:\DaveLM-CADAVER\baby_vnext_phase2a_t3_ctxbind_v1\ARCHITECTURE_PROVENANCE_NOTE.md` | `docs\provenance\` |
| COPY | `C:\DaveLM-CADAVER\mistral_treatment_design_audit_v1\` | `docs\audits\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\overnight_reproducibility_resolution_v1\` (exclude `*.pt`) | `docs\audits\overnight_reproducibility\` |
| COPY | `C:\DaveLM-CADAVER\comparative_model_anatomy_post_sf2_v1\` | `docs\audits\` |
| COPY | `C:\DaveLM-CADAVER\reference_model_research\` | `docs\research\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\forensics\` (exclude `*.pt`) | `experiments\legacy_10m\forensics\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\sf1_*` … `sf21_*` (top-level + data/reports; exclude `runs\`, `checkpoints\`, `*.pt`, `restart.pt`) | `experiments\legacy_10m\sf\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\treatment4_*` … `treatment13_*` (top-level; exclude `checkpoints\`) | `experiments\legacy_10m\treatments\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\language_*` dirs and scripts (exclude `latest.pt`) | `experiments\legacy_10m\language\` |
| COPY-PARTIAL | `C:\DaveLM-CADAVER\fact_supervision_*`, `human_readiness_*`, `hr3_*`, `human_test_readiness_*`, `english_context_characterization_*`, `single_fact_acquisition_*`, `two_mapping_*`, `post_p7_*`, `archive\` (exclude `.pt`/runs) | `experiments\legacy_10m\` subtrees |
| COPY | Root `*.py` experiment/audit scripts (list in B15) | `experiments\legacy_10m\scripts\` |
| COPY | Root history `*.md`/`*.json` reports (list in B16) | `docs\history\` |

### 13.3 Historical baseline (Category C)

| Action | Original source path | Destination |
|---|---|---|
| COPY-PARTIAL | `C:\DaveLM-v0.9\` source (`v0_2`, `v0_2_1`, `v0_7`, `v0_8`, `v0_8_1`–`v0_8_4`, `v0_9`, `tests`, `tokenizer`, root docs) | `legacy\davelm_v0_9\` |
| COPY-PARTIAL | `C:\DaveLM-v0.9\experiments\*` (configs/reports/corpora; exclude `checkpoints\`/`*.pt`) | `legacy\davelm_v0_9\experiments\` |
| COPY-PARTIAL | `C:\DaveLM-v0.9\forensics\*`, `C:\DaveLM-v0.9\validation\*` (exclude `.pt`) | `legacy\davelm_v0_9\` |
| REVIEW | `C:\DaveLM-CADAVER\DaveLM-v0.9\` (nested snapshot) | E1; do not duplicate |

### 13.4 Local-only (Category D) and redundant (Category E)

All of §6 and §7 are recommended **EXCLUDE** from Git (keep locally under `artifacts\` if desired).
The only nuance is Category F checkpoints/datasets, which need a human keep/drop + LFS decision.

---

## 14. Proposed `.gitignore` rules

```gitignore
# --- Model weights / checkpoints / large binaries ---
*.pt
*.pth
*.ckpt
*.safetensors
*.gguf
*.bin
rolling_restart.*
**/checkpoints/
**/runs/
**/run_seed*/
**/*_run_seed*/

# --- Bundled runtimes / environments / caches ---
sf2_runtime/
**/venv/
**/.venv/
**/site-packages/
__pycache__/
*.py[cod]
*.pyo
.pytest_cache/
.mypy_cache/
.ruff_cache/

# --- Large generated data / logs / dumps ---
*.log
*.err
*.out
*_dump.txt
t5_t8_structural_census.json
**/ENGLISH_TRAIN.jsonl
**/LANGUAGE_TRAIN_SOURCE.jsonl
# (Decide separately: *.u16, *.u32, *.jsonl frozen streams — see F3)

# --- Agent / tool state (also secret-risk) ---
.aider*
.config/
.opencode/
.continue/
.cursor/
.vscode/
.idea/
tunnel.json

# --- Secrets ---
.env
.env.*
*.pem
*.key
*.p12
*.pfx
credentials*
secrets*
id_rsa*
*api_key*
*token*

# --- OS / editor ---
Thumbs.db
desktop.ini
.DS_Store
*.swp
New Text Document.txt

# --- Local artifacts directory ---
artifacts/
```

Optional `.gitattributes` for any deliberately versioned large binaries:

```gitattributes
*.pt filter=lfs diff=lfs merge=lfs -text
*.pt.* filter=lfs diff=lfs merge=lfs -text
*.u16 filter=lfs diff=lfs merge=lfs -text
*.u32 filter=lfs diff=lfs merge=lfs -text
```

---

## 15. Uncertainties / questions requiring human judgment

1. **Non-quiescent source.** `phase2a_t14x_latebase_decomp_v1` is still running and writing. When is it
   safe to freeze the snapshot? What is its expected terminal classification, and does the next
   treatment (T15?) change the migration set?
2. **Hard-coded absolute paths.** Every current runner references `C:\DaveLM-CADAVER` and
   `C:\DaveLM-v0.9`. Should the new repo (a) preserve those paths via a root shim, or (b) undergo a
   path-refactor (recommended)? This affects whether "COPY" is even runnable.
3. **Parent checkpoint policy.** Must `best.pt` (234.7 MB), initialization checkpoints (234.7 MB),
   and the T3–T14 terminal checkpoints live in Git (LFS/release) or locally only? The handoff and
   reproducibility bar suggest they matter.
4. **Generated data policy.** `LANGUAGE_TRAIN_SOURCE.jsonl` (55.7 MB) + `.u16` streams + window starts
   are derived but required. Commit (LFS), or regenerate via the build scripts at setup time?
5. **Sealed T3 TEST.** `qa_test.jsonl` + `TEST_SEAL.json` must migrate to preserve the seal, but any
   handling risks "opening" it. Confirm the acceptable procedure.
6. **Closed-series checkpoints.** Should any SF8/T13/Pilot0/Pilot1 checkpoints be preserved for the
   historical record, or is the report/hash record sufficient?
7. **Nested `DaveLM-v0.9`.** Discard as redundant (E1), or keep as an immutable provenance snapshot?
8. **Secret handling.** How should `.aider.chat.history.md`, `mistral_test.py`,
   `mistral_baby_history.json`, and `.config\opencode\tunnel.json` be archived/scrubbed? Rotate any key
   that may have appeared in the Aider history.
9. **10.6M history scope.** How much of the ~25 GB of legacy reports/data/checkpoints is worth carrying
   into a "clean canonical development repository" vs. leaving in CADAVER as an archive?
10. **v0.9 canonicalization.** The external `C:\DaveLM-v0.9` is current; the nested copy is stale. Should
    the migration read from the external tree (outside the stated SOURCE) or only from CADAVER's nested
    copy? The tokenizer exists identically in both, so either works for it, but the newer v0.9
    experiment source exists only externally.
11. **Repository licensing/provenance.** The repo will contain TinyStories-derived datasets and external
    model research notes; confirm licensing/attribution before publishing.
12. **Git history.** There is no existing Git history in CADAVER; the new repo starts fresh. Confirm the
    desired initial commit granularity (single "import" commit vs. staged commits by category).

---

*End of audit. No files were created, modified, copied, moved, or deleted other than this report.*
