# Agent instructions (DaveLM Baby / CADAVER)

Workspace root: `C:\DaveLM-CADAVER`. Do not open a different tree as the Baby workspace.

## Hard stops

1. **Do not load TEST bytes.** Never open, score, copy, or print `qa_test.jsonl` (any path). Git tracks only `phase2a_t3_rebuilt_study_v1/data/TEST_SEAL.json`.
   - Expected local path: `C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1\data\qa_test.jsonl`
   - SHA-256: `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`
   - Status: `SEALED_UNOPENED` until terminal DEV representation **and** native gates pass.
2. **Do not lower the v1.0 / 2/3 representation bar.**
3. **Do not parent** T14–T23 OUTPUT_FAIL checkpoints, or 730002 / 760001–820003.
4. **Do not retokenize.** Do not copy Qwen or Smol weights into this tree.
5. **Do not reorganize** frozen experiment directory names. Preserve scientific paths.
6. **Do not relaunch T4–T23.** CE-span-count, suffix-weight 3.0, and in-row unlikelihood are closed.
7. **GitHub:** may commit/push **doc-only** updates when a study is terminal. Never commit `.pt`, `*_run_seed*`, or `qa_test.jsonl`. Never force-push or skip hooks.

## Read first

- [`CANONICAL_STATE.md`](CANONICAL_STATE.md) — authoritative current state.
- [`PHASE2A_CURRENT_HANDOFF.md`](PHASE2A_CURRENT_HANDOFF.md) — scientific handoff.
- [`CURRENT_STATE.md`](CURRENT_STATE.md)
- [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md)
- [`artifacts/LOCAL_ONLY.md`](artifacts/LOCAL_ONLY.md)

## Parent checkpoint (local-only)

`C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001\checkpoints\best.pt`  
SHA-256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`

## Live

T22/T24 are terminal history (`REPRESENTATION_SUCCESS_OUTPUT_FAIL`). Do not resume T24. T28 FAST V2 is terminal `OUTPUT_FAIL`. Post-T28 diagnostics are complete. T29 is terminal `T29_REPRESENTATION_SUCCESS_OUTPUT_FAIL` (exact 35/35/35); the fork-objective hypothesis is closed. Do not overwrite v0_7, retune T29, or open TEST. T30 is terminal: T30_REPRESENTATION_SUCCESS_OUTPUT_FAIL (2/3 representation-positive, native output gate missed). T31 is terminal; do not launch T32.

When a terminal study earns a successor but no successor protocol exists, the autonomous agent is authorized and expected to design, prospectively freeze, implement, and execute the highest-information Baby-native successor. This authority was explicitly granted by the owner on 2026-09-13; all gates and protected-material locks still apply. Owner review is required before a full tokenizer-replacement retrain.

## What belongs in git vs disk

Tracked: source, protocols, reports, compact data, tokenizer under `DaveLM-v0.9/tokenizer/`, `TEST_SEAL.json`, onboarding docs.  
Not tracked (see `.gitignore`): weights, run-seed trees, `sf2_runtime/`, archive, venv, most of nested v0.9, logs, secrets, TEST jsonl.

Runners hard-code `C:\DaveLM-CADAVER`. Do not “fix” those paths as a drive-by refactor.


