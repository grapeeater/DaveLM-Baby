# Provenance — SF2 preflight review v1

## Authoring context

- **Date:** 2026-09-07.
- **Mode:** read-only. No training, optimizer creation, autograd, weight mutation, checkpoint load/save,
  inference over a checkpoint, locked/held-out panel scoring, FINAL/sacred access, gate change, or new
  mechanistic experiment was performed. The deliverable directory and its five files are the only writes.
- **Tooling:** read-only filesystem inspection and full reads of the authoritative SF1/Pilot1 construction
  artifacts and the Mistral/Codestral treatment-design audit.

## Absence of the referenced DeepSeek V4 Flash review

The mission asked to read a "completed DeepSeek V4 Flash independent review" and to locate its artifacts **if
present**. An exhaustive search found **none**:
- Recursive directory-name and file-name search over `C:\DaveLM-CADAVER\` for `deepseek`/`DEEPSEEK`/`sf2`:
  no matches (only Windows WinSxS false positives on `sf2`).
- File-name search for `review`/`REVIEW` across `C:\DaveLM-CADAVER\`: only
  `human_test_readiness_v2_seed87010\HUMAN_REVIEW_RUBRIC.md` (unrelated).
- Content search for `DeepSeek V4 Flash` / `independent review` in root-level markdown: only matches in
  `.aider.chat.history.md` (a dev-tool log, not a review artifact).
- Search of `C:\Users\jdman\Documents`, `Temp`, `.cursor`, `Desktop`, and `C:\DaveLM-*` siblings: no match
  (only an unrelated AutoGPT `deepseek.png`).
Per the mission's instruction ("do not rely on assumptions or reconstruct missing conclusions"), this review did
**not** reconstruct the DeepSeek review's content. It is based solely on the present authoritative artifacts and
the Mistral/Codestral audit (`C:\DaveLM-CADAVER\mistral_treatment_design_audit_v1\`), whose five content files
were read (REPORT.md, EVIDENCE_LEDGER.md, PROPOSED_TREATMENT.md, INSPECTED_PATHS.md, PROVENANCE.md).

## Authoritative artifacts read for this review

- `C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011\ENGINE.py` (authoritative SF1 trainer; full read)
- `C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011\PROTOCOL.json`, `SCHEDULE.json`, `TRAIN.json` (read
  this and prior audits)
- `C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011\sources\PINNED_MASKING.py`,
  `sources\hr3_block3_runtime.py`, `sources\treatment13_model.py`, `sources\treatment13_config.py`,
  `sources\PINNED_PILOT1_BINDING_IMPLEMENTATION.py` (head read)
- `C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011_run\REPORT_SOURCE.py`, `STATUS.json`,
  `update0_acquisition_RESULT.json`, `update100_acquisition_RESULT.json`, `TRAIN_METRICS.jsonl` (head/tail)
- `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\TRAINING_SPEC.json`,
  `PILOT1_MATERIAL_MANIFEST.json`, `run.py` (grep), `language_train.jsonl` / `ENGLISH_DEV.jsonl` (row schema)
- `C:\DaveLM-v0.9\v0_7\model.py` (constructor/dropout), `v0_7\config.py`, `v0_8\config.py` (MODEL_CONFIG,
  dropout 0.05), `v0_8_2\config.py`, `v0_8_2\model.py`
- `C:\DaveLM-CADAVER\sf1_readout_selection_forensic_v1\REPORT.md`, `D3_SELECTION_RECEIPT.json`,
  `OPTIMIZER_METADATA.json`
- `C:\DaveLM-CADAVER\sf1_component_swap_forensic_v1\REPORT.md`
- `C:\DaveLM-CADAVER\sf1_upstream_localization_forensic_v1\REPORT.md`
- `C:\DaveLM-CADAVER\sf1_token_position_patching_forensic_v1\REPORT.md`
- `C:\DaveLM-CADAVER\sf1_candidate_pool_hypothesis_audit_v1\REPORT.md`, `EVIDENCE.md`, `NEXT_DIAGNOSTIC.md`

## Key artifact hashes (as published in the artifacts)

| Artifact | SHA-256 (first 16 shown) |
|---|---|
| Pilot1 parent `...\seed_8380\latest.pt` | `2281d20ebd3ae9c5...` |
| SF1 update-100 checkpoint | `550ce4306b450c2b...` |
| Tokenizer `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json` | `e1c18bae74f6d502...` |
| SF1 TRAIN.json | `08318d68e0d1b005...` |
| SF1 PROTOCOL.json | `5c236478bb7370f5...` |
| SF1 SCHEDULE.json | `91dad3510e7997b0...` |
| SF1 monitor `data\ENGLISH_DEV.jsonl` | `2053a8036d7d0f02...` |
| D3 selection manifest | `3ae0f6a748f5545b...` |
| Pilot1 language train corpus | `450f78bdb437bde9...` |
| Pilot1 language dev corpus | `deff4fc7ed18e6e1...` |

These are quoted from the artifacts' own provenance records; this review did not re-derive them.

## Integrity statement

- Only files written: the five deliverable files in `C:\DaveLM-CADAVER\sf2_kl_treatment_preflight_review_v1\`
  (`REVIEW.md`, `REQUIRED_CORRECTIONS.md`, `FROZEN_PROTOCOL_DRAFT.md`, `PROVENANCE.md`, `SHA256SUMS.txt`).
- No pre-existing artifact anywhere was modified. No training or mechanistic work was performed. No locked panel,
  FINAL, or sacred material was accessed. No gate was weakened.
