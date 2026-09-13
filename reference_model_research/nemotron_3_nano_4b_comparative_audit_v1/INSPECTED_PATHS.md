# Inspected Paths — Provenance Record

This audit was strictly READ-ONLY. No file listed below was modified, moved, deleted, or renamed.
Nothing was trained. No checkpoints were touched. No Baby code, dataset, or frozen gate was altered.

## Repository identification (important finding)

- **`C:\DaveLM-v0.9`** (top-level, sibling of `C:\DaveLM-CADAVER`) is the **authoritative, currently-developed
  DaveLM v0.9 repository**. Confirmed by direct comparison: it contains 492 files vs. 339 in the CADAVER copy,
  and its newest files (e.g. `experiments\two_mapping_bootstrap_from_one_map_seed8380\*`,
  `v0_9\__pycache__\*.pyc` compiled 9/3/2026 7:20 PM) post-date every file in
  `C:\DaveLM-CADAVER\DaveLM-v0.9` (newest file there: `experiments\pointer_mixture_copy\*`, 8/31/2026).
- **`C:\DaveLM-CADAVER\DaveLM-v0.9`** is a **stale, frozen snapshot** of the v0.9 codebase copied in at some
  point on/before 8/31/2026 to seed the CADAVER research tree. It should not be cited as "current."
- **`C:\DaveLM-CADAVER`** (root) is the actively-worked research/forensic tree (treatments, language pilots,
  P0–P7, HR1/HR2/HR3, fact_supervision_87001, SF1 and its forensics), spanning roughly Sep 1–6, 2026.
- Other version directories (`C:\DaveLM`, `C:\DaveLM-v0.2.1` … `C:\DaveLM-v0.8.4`, `C:\DaveLM-v0.10`) exist as
  siblings on the machine. `C:\DaveLM` (no version suffix) is the earliest prototype (created 8/25/2026,
  contains `baby_davelm.pt`, `transformer_davelm.py`, etc.) and is NOT the current repo. `C:\DaveLM-v0.10`
  exists but is **empty** — no files were found in it.
- `V0_9_TECHNICAL_NOTE.md:24` references `C:\DaveLM-v0.8.4\experiments\dense_geometry_randomization_ab` as the
  immutable Phase-1 evidence store for the v0.9 gate; this directory was not separately re-audited in depth
  (only referenced) to keep scope bounded.

## Nemotron 3 Nano 4B — local files inspected

- `C:\Users\jdman\.lmstudio\hub\models\nvidia\nemotron-3-nano-4b\manifest.json`
- `C:\Users\jdman\.lmstudio\hub\models\nvidia\nemotron-3-nano-4b\model.yaml`
- `C:\Users\jdman\.lmstudio\hub\models\nvidia\nemotron-3-nano-4b\README.md`
- `C:\Users\jdman\.lmstudio\hub\models\nvidia\nemotron-3-nano-4b\thumbnail.png` (not opened — irrelevant)
- `C:\Users\jdman\.lmstudio\models\lmstudio-community\NVIDIA-Nemotron-3-Nano-4B-GGUF\NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf`
  — **metadata and tensor-info header only**, parsed with a purpose-built read-only GGUF header/KV parser
  (`gguf_meta.py`, run from a temp scratch directory, not saved into the repo). The multi-GB tensor **payload**
  was never read/loaded; only the GGUF header (magic/version/counts), the 36 metadata key-value pairs, and the
  263 tensor-info records (name/shape/ggml-type/offset) were parsed. This avoided downloading or reading any
  `.safetensors` weight data, per the mission's constraint.

## Nemotron 3 Nano 4B — remote/upstream sources fetched (HTML/JSON only, no weight files)

- `https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` (model card / README rendering)
- `https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/raw/main/config.json`
- `https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/raw/main/tokenizer_config.json`
- `https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16/raw/main/generation_config.json`
- `https://arxiv.org/abs/2511.16664` (Nemotron Elastic)
- `https://arxiv.org/abs/2504.03624` (Nemotron-H)
- (Referenced but not separately fetched in full: `NVIDIA Nemotron 3: Efficient and Open Intelligence`
  arXiv:2512.20856; `Nemotron 3 Nano` arXiv:2512.20848; `Nemotron 3 Super` NVIDIA research-report PDF;
  `NVIDIA Nemotron Nano 2` technical-report PDF — all cited by the model card's References section; their
  abstracts/titles are treated as AUTHORITATIVE UPSTREAM FACT via the model card citation, but their full text
  was not independently re-fetched in this pass.)

## Baby (DaveLM) architecture archaeology — files inspected (via delegated sub-agent, full read access)

Representative set (not exhaustive — see sub-agent citations reproduced in REPORT.md for full file:line detail):
- `C:\DaveLM-v0.9\v0_2\model.py`, `config.py`
- `C:\DaveLM-v0.9\v0_2_1\model.py`
- `C:\DaveLM-v0.9\v0_7\model.py`, `config.py`
- `C:\DaveLM-v0.9\v0_8\config.py`
- `C:\DaveLM-v0.9\v0_8_2\model.py`, `config.py`, `tied_embedding_routing.py`
- `C:\DaveLM-v0.9\v0_9\config.py`, `dense_geometry.py`, `geometry_helpers.py`, `integration_gate.py`
- `C:\DaveLM-v0.9\README.md`, `V0_9_TECHNICAL_NOTE.md`
- `C:\DaveLM-v0.9\experiments\matched_scale_tied_copy\REPORT.md`
- `C:\DaveLM-v0.9\experiments\straight_through_scaled_tied_copy\REPORT.md`
- `C:\DaveLM-v0.9\experiments\variance_scaled_tied_copy\REPORT.md`
- `C:\DaveLM-v0.9\experiments\straight_through_tied_recovery_extension\REPORT.md`
- `C:\DaveLM-v0.9\experiments\pointer_mixture_copy\REPORT.md`
- `C:\DaveLM-v0.9\experiments\aligned_output_identity_init_ab\REPORT.md`
- `C:\DaveLM-v0.9\experiments\aligned_output_rotation_init_abc\REPORT.md`
- `C:\DaveLM-v0.9\experiments\untied_aligned_output_initialization\` (confirmed empty)
- `C:\DaveLM-CADAVER\treatment5_train.py` … `treatment13_train.py`, and each `treatmentN_model.py`,
  `treatmentN_config.py`, `treatmentN_final_retention_audit.py`/`.json`
- `C:\DaveLM-CADAVER\treatment12_frozen_causal_audit.py`, `treatment12_geometry_census.py`,
  `treatment12_frozen_causal_validation\frozen_causal_report.md`,
  `treatment12_frozen_geometry_generalization\frozen_geometry_report.md`
- `C:\DaveLM-CADAVER\treatment13_bounded_shared_antisymmetric_seed8380\bounded_model.py`, `REPORT.md`
- `C:\DaveLM-CADAVER\treatment13_orthogonal_shared_unbounded_seed8380\REPORT.md`, `REPORT_ADDENDUM.md`
- `C:\DaveLM-CADAVER\treatment13_distinct_localization_supervision_seed8380\REPORT.md`
- `C:\DaveLM-CADAVER\treatment13_nonsource_localization_suppression_seed8380\REPORT.md`
- `C:\DaveLM-CADAVER\treatment13_source_recognition_seed8380\REPORT.md`
- `C:\DaveLM-CADAVER\hr1_build.py`, `hr3_block3_runtime.py` (and its bundled `treatment13_model.py`,
  `PINNED_PILOT1_BINDING_IMPLEMENTATION.py`)
- `C:\DaveLM-CADAVER\sf1_build.py`, `sf1_engine.py`, `sf1_report.py`

## Baby developmental history — files inspected (via delegated sub-agent, full read access)

- `C:\DaveLM-CADAVER\DaveLM-v0.9\README.md`, `V0_9_TECHNICAL_NOTE.md`
- `C:\DaveLM-CADAVER\language_pilot_0_tinystories_seed8380\pilot_run\RESULTS.json`
- `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\RESULTS.json`, `run.py`
- `C:\DaveLM-CADAVER\language_continuation_p0_v1\`, `language_consistency_p1\`, `language_mixed_p2\`,
  `language_storybound_p3\`, `language_unlikelihood_p4\`, `language_sentencebound_p5\`,
  `language_compositional_p6\`, `language_compositional_p7\` (RESEARCH_LOG.md / RESULTS.json / OVERLAP_AUDIT.json
  / NATURAL_TRANSFER_RESULTS.json in each, where present)
- `C:\DaveLM-CADAVER\archive\DAVELM_P7_MILESTONE_seed8380\ARCHIVE_RECORD.md`
- `C:\DaveLM-CADAVER\P7_LINEAGE_AND_REPORT_CARD_ERRATUM_20260905.md`
- `C:\DaveLM-CADAVER\POST_P7_INTEGRITY_STOP_AUDIT.md`
- `C:\DaveLM-CADAVER\post_p7_language_report_card_v1_seed8380_execution_p7\REPORT.md`
- `C:\DaveLM-CADAVER\post_p7_language_report_card_v2_seed8391_execution_p7\REPORT.md`
- `C:\DaveLM-CADAVER\post_p7_v3d_stage1_execution_retry\SUMMARY.json`
- `C:\DaveLM-CADAVER\fact_supervision_87001_corrected_v8\V3D_HISTORICAL_STATUS_ERRATUM.md`
- `C:\DaveLM-CADAVER\fact_supervision_87001\STOP_REPORT.md`
- `C:\DaveLM-CADAVER\fact_supervision_87001_corrected_v2\CORRECTION_STOP_REPORT.md`
- `C:\DaveLM-CADAVER\forensics\FREEZE_COMPLETION_AUDIT_20260905.md`
- `C:\DaveLM-CADAVER\fact_supervision_87001_eval_v1\REPORT.md`, `stage0_measurement_report.md`
- `C:\DaveLM-CADAVER\GENERATION_PATHOLOGY_REPORT.md` (root)
- `C:\DaveLM-CADAVER\hr_generation_pathology.py`, `hr_tf_diag.py`, `hr_decode_constraints.py`, `hr_eos_diag.py`
- `C:\DaveLM-CADAVER\hr3_context_autopsy_v1\AUTOPSY_REPORT.md`
- `C:\DaveLM-CADAVER\hr3_diagnostic_staircase_v1_results\REPORT.md`
- `C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011_run\REPORT.md`
- `C:\DaveLM-CADAVER\sf1_readout_selection_forensic_v1\REPORT.md`
- `C:\DaveLM-CADAVER\sf1_component_swap_forensic_v1\REPORT.md`

## Tooling used

- PowerShell `Get-ChildItem`/`Get-Item`/`Compare-Object` for directory/timestamp forensics (read-only).
- Python 3.13 one-off script (`gguf_meta.py`, in the OS temp scratch dir) for GGUF header/metadata parsing
  (no third-party `gguf` package was available/installed; a minimal from-scratch binary parser was written
  instead, reading only the header + KV section + tensor-info table, never tensor payload bytes).
- Two delegated read-only research sub-agents (this tool's `Task` mechanism) performed the deep Baby
  archaeology to keep this session's own context bounded; their full verbatim findings are reproduced/condensed
  into `REPORT.md`. Both were explicitly instructed not to modify any files.
