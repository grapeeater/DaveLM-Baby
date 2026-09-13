# Provenance

## Authoring context

- **Date:** 2026-09-07 (audit performed after the 2026-09-06 evening forensics).
- **Tooling:** read-only filesystem inspection (PowerShell directory/hash queries), full reads of the mandated
  reports and SF1/Pilot1 construction artifacts. No Python, optimizer, autograd, inference over a checkpoint, or
  weight-touching operation was performed during this audit.
- **Method:** evidence-linked path traversal only (per mission cost/scope rule). Facts were taken from
  authoritative artifacts with published hashes wherever possible rather than recomputed.

## Authoritative artifact hashes referenced (as published in the artifacts themselves)

| Artifact | SHA-256 (first 16 shown) |
|---|---|
| Pilot1 parent checkpoint `language_pilot_1_...\pilot_run\checkpoints\seed_8380\latest.pt` | `2281d20ebd3ae9c5...` (PROTOCOL.json) |
| SF1 update-100 checkpoint `single_fact_acquisition_sf1_seed87011_run\checkpoint_100.pt` | `550ce4306b450c2b...` (checkpoint_100.sha256) |
| Tokenizer `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json` | `e1c18bae74f6d502...` |
| SF1 `TRAIN.json` | `08318d68e0d1b005...` (candidate-pool EVIDENCE.md) |
| SF1 `PROTOCOL.json` | `5c236478bb7370f5...` (candidate-pool EVIDENCE.md) |
| SF1 `SCHEDULE.json` | `91dad3510e7997b0...` (candidate-pool EVIDENCE.md) |
| SF1 monitor source `data\ENGLISH_DEV.jsonl` | `2053a8036d7d0f02...` (readout REPORT.md) |
| D3 selection manifest | `3ae0f6a748f5545b...` (D3_SELECTION_RECEIPT.json) |
| Pilot1 language train corpus | `450f78bdb437bde9...` (PILOT1_MATERIAL_MANIFEST.json) |
| Pilot1 language dev corpus | `deff4fc7ed18e6e1...` (PILOT1_MATERIAL_MANIFEST.json) |

These are quoted from the artifacts' own provenance records; this audit did not re-derive them.

## Supersession notice

`C:\DaveLM-CADAVER\mistral_treatment_design_audit_v1\` already contained six files created 2026-09-07 ~00:18
(recommendation-class `TRAIN_ONE_BOUNDED_TREATMENT`, a generic KL sketch whose design details referenced "same as
SF1" without specifying the KL source/disjointness/weight, and clearly non-real SHA-256 placeholders). That set
was treated as an unverified prior draft and replaced by the current audit's files. No other pre-existing
artifact anywhere was modified.

## Integrity statement

- The only files written by this audit are the six deliverable files in this directory
  (`REPORT.md`, `EVIDENCE_LEDGER.md`, `PROPOSED_TREATMENT.md`, `INSPECTED_PATHS.md`, `PROVENANCE.md`,
  `SHA256SUMS.txt`).
- No optimizer was created; no autograd run; no weights mutated; no checkpoint saved/loaded; no locked panel,
  FINAL, or sacred material accessed; no gate weakened; no training executed.
