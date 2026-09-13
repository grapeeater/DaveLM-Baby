# Inspected Paths

Read-only audit. Nothing outside the deliverable directory
`C:\DaveLM-CADAVER\mistral_treatment_design_audit_v1\` was written. No locked/withheld panel, FINAL, or sacred
material was opened. No checkpoint, dataset, code, report, or gate was modified.

The deliverable directory already existed and contained a **stale placeholder set** (six files dated
2026-09-07 ~00:18, with a generic KL sketch and non-real hashes). This audit supersedes that placeholder with
evidence-linked content; provenance of the supersession is recorded in PROVENANCE.md.

## Repository/authority verification

- `C:\DaveLM-CADAVER\` — primary evidence/history tree (root; used as authority for all experiment artifacts).
- `C:\DaveLM-v0.9\` — current development codebase (top-level sibling). Confirmed by newer modification times
  than `C:\DaveLM-CADAVER\DaveLM-v0.9\` (stale snapshot). The stale snapshot was NOT used as authority.
- `C:\DaveLM\`, `C:\DaveLM-v0.2.1`…`C:\DaveLM-v0.8.4`, `C:\DaveLM-v0.10` — historical/empty siblings; not
  relevant to this audit beyond path confirmation.

## Mandated forensics (read in full where present)

- `C:\DaveLM-CADAVER\sf1_token_position_patching_forensic_v1\REPORT.md` (plus directory inventory)
- `C:\DaveLM-CADAVER\sf1_readout_selection_forensic_v1\REPORT.md`, `D3_SELECTION_RECEIPT.json`,
  `OPTIMIZER_METADATA.json`
- `C:\DaveLM-CADAVER\sf1_component_swap_forensic_v1\REPORT.md`
- `C:\DaveLM-CADAVER\sf1_upstream_localization_forensic_v1\REPORT.md`
- `C:\DaveLM-CADAVER\sf1_candidate_pool_hypothesis_audit_v1\REPORT.md`, `EVIDENCE.md`, `NEXT_DIAGNOSTIC.md`

## SF1 construction/training code and data (authoritative)

- `C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011\TRAIN.json` (16 records; full read)
- `C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011\PROTOCOL.json` (full read)
- `C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011\sources\PINNED_MASKING.py` (full read — scope/objective)
- `C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011\sources\hr3_block3_runtime.py` (full read — runtime,
  binding gates, aligned loss)
- `C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011\data\ENGLISH_DEV.jsonl` (line count only, 15,866)
- `C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011_run\REPORT.md`, `STATUS.json`,
  `update0_acquisition_RESULT.json`, `update100_acquisition_RESULT.json`, `TRAIN_METRICS.jsonl` (head/tail)

## Pilot1 parent/protection policy and corpora

- `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\TRAINING_SPEC.json`
- `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\PILOT1_MATERIAL_MANIFEST.json`
- `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\run.py` (grep of scope/schedule lines)
- Directory inventory of `C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\` (corpus files)

## Earlier factual-treatment / binding background (secondary, cited)

- `C:\DaveLM-CADAVER\sf1_candidate_pool_hypothesis_audit_v1\EVIDENCE.md` (contains hashes and summary of the
  earlier four-name factual-supervision treatment and of binding pool structure)
- Prior comparative audit output (referenced for context only):
  `C:\DaveLM-CADAVER\reference_model_research\nemotron_3_nano_4b_comparative_audit_v1\`

## Prior (stale) deliverable read for supersession

- `C:\DaveLM-CADAVER\mistral_treatment_design_audit_v1\REPORT.md`, `PROPOSED_TREATMENT.md` (placeholder; treated
  as unverified and replaced)

## NOT inspected / NOT opened (deliberately)

- `single_fact_acquisition_sf1_seed87011\HELDOUT.json`, `ALTERNATE.json`, `COPY.json`, `COMPETING.json` (locked
  transfer panels)
- Any `FINAL`/readiness/sacred battery
- Any `.pt` checkpoint payload beyond hash references already published in the artifacts
