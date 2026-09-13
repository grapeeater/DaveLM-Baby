# Agent instructions (DaveLM Baby / CADAVER)

Workspace root: `C:\DaveLM-CADAVER`. Do not open a different tree as the Baby workspace.

## Hard stops

1. **T22 is PAUSED.** Do not resume training. Do not launch 810002, 810003, T22X, or T23. Do not run `phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1\LAUNCH_WATCHDOG.cmd` until the user explicitly says **CONTINUE**.
2. **Do not load TEST bytes.** Never open, score, copy, or print `qa_test.jsonl` (any path). Git tracks only `phase2a_t3_rebuilt_study_v1/data/TEST_SEAL.json`.
   - Expected local path: `C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1\data\qa_test.jsonl`
   - SHA-256: `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`
   - Status: `SEALED_UNOPENED` until terminal DEV representation **and** native gates pass.
3. **Do not lower the v1.0 / 2/3 representation bar.**
4. **Do not parent** T14, T15, T16, T17, T18, T19, T20, T21 OUTPUT_FAIL checkpoints, or 730002.
5. **Do not retokenize.** Do not copy Qwen or Smol weights into this tree.
6. **Do not reorganize** frozen experiment directory names. Preserve scientific paths.
7. **GitHub:** do not `git commit`, `git push`, or `gh repo create` unless the user explicitly asks after reviewing the staging report. Never `git add -i`, `git rebase -i`, force-push, or rewrite published history. Never update git config.

## Read first

- [`CURRENT_STATE.md`](CURRENT_STATE.md) — physical pause (must match T22_PAUSE).
- [`PHASE2A_CURRENT_HANDOFF.md`](PHASE2A_CURRENT_HANDOFF.md) — authoritative scientific handoff.
- [`phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1/T22_PAUSE.md`](phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1/T22_PAUSE.md)
- [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md)
- [`artifacts/LOCAL_ONLY.md`](artifacts/LOCAL_ONLY.md) — checkpoint / tokenizer / TEST paths and hashes.

## Resume pair (local-only; do not use until CONTINUE)

- Run dir: `C:\DaveLM-CADAVER\phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1_run_seed810001`
- `rolling_restart.pt` `completed=550` phase A  
  SHA-256 `ff1000a8a0b117c7f9f28e866e9cf1d4920cceed59031c8c3bf69df8a10047b7`
- Last eval: `checkpoints\checkpoint_0500.pt`  
  SHA-256 `0ba71d0967fce73300170270b3593c7818f1e6deba8eb75059d2e9863dc8cfd7`

## Parent checkpoint (local-only)

`C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001\checkpoints\best.pt`  
SHA-256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`

## What belongs in git vs disk

Tracked: source, protocols, reports, compact data, tokenizer under `DaveLM-v0.9/tokenizer/`, `TEST_SEAL.json`, onboarding docs.  
Not tracked (see `.gitignore`): weights, run-seed trees, `sf2_runtime/`, archive, venv, most of nested v0.9, logs, secrets, TEST jsonl.

Runners hard-code `C:\DaveLM-CADAVER`. Do not “fix” those paths as a drive-by refactor.

## If asked to train

Refuse unless the user said CONTINUE for T22, or explicitly authorized a **new** named study that is not 810002/810003/T23 while T22 is paused.
