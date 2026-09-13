# Current state (physical)

**As of 2026-09-12.** Stay in `C:\DaveLM-CADAVER`. Do not open T3 TEST. Do not lower the v1.0 bar. **T22 remains PAUSED. Do not resume training.**

This file matches [`PHASE2A_CURRENT_HANDOFF.md`](PHASE2A_CURRENT_HANDOFF.md) and [`phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1/T22_PAUSE.md`](phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1/T22_PAUSE.md).

## Parent (Phase1G)

| | |
|---|---|
| Path (local-only) | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001\checkpoints\best.pt` |
| SHA-256 | `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1` |
| U0 DEV CE | `1.2040123894810677` (bit-identical on T22 U0) |

## TEST

Status **`SEALED_UNOPENED`**. Seal file: `phase2a_t3_rebuilt_study_v1/data/TEST_SEAL.json`. Bytes are **not** in git.

`qa_test.jsonl` SHA-256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`. Seal was read only; `qa_test.jsonl` was not opened for this pause/onboarding.

## Live — T22 PAUSED / physically quiescent

Bundle: `phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1`  
Run (local-only): `phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1_run_seed810001`

| Fact | Value |
|---|---|
| Seed | **810001 only** |
| Phase | **A** |
| Resume | `rolling_restart.pt` **`completed=550`**, includes `model_state_dict` + `optimizer` |
| `rolling_restart.pt` SHA-256 | `ff1000a8a0b117c7f9f28e866e9cf1d4920cceed59031c8c3bf69df8a10047b7` |
| Last eval | U500 `checkpoint_0500.pt` |
| `checkpoint_0500.pt` SHA-256 | `0ba71d0967fce73300170270b3593c7818f1e6deba8eb75059d2e9863dc8cfd7` |
| `evaluation_0500.json` SHA-256 | `c737dd662b7ec1a5c531f088f80627f4e6a8bf584cded960cb6130cdf2328096` |
| `STATUS.json` SHA-256 | `a9c3a85415df98815707614f75b15c95bea66ec63d8bf0a805eca5af28e69477` |
| `STATUS.json` contents | `TRAINING_RUNNING`, update **550**, phase **A** (stale label; processes are stopped) |
| Metrics | `training_metrics.jsonl` last line is update **580**; resume from `completed=550` will redo 551+ |
| Next eval | U750 (not written) |
| Watchdog / runner | PIDs 23396, 22592, 27036 **stopped**. No T22 processes live. |
| **810002 / 810003** | **Never started.** Directories absent. |
| T22X / T23 | **Never launched.** |

Do **not** run `LAUNCH_WATCHDOG.cmd` until the user says CONTINUE. Phase A→B remains armed for after U750 eval (`after_u750_eval`). Frozen recipe unchanged.

T14 leftover PIDs (2644 / 19768 / 14592) were **not** touched during the T22 stop.

## Closed lines (do not relaunch / do not parent OUTPUT_FAIL)

See [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md). Headline:

- T3–T16 / T16X / Qwen: closed as previously. Do not parent T14/T15/T16/730002.
- T17 `T17_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — representation **2/3**, exact 0.
- T17X COMPLETE. Exact 0 was Phase1G story continuation after `Answer:`.
- T18 `T18_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — language 3/3, representation **2/3**, exact **36/36/40**.
- T18X COMPLETE. First-token BOS 76–84. Leftover `first_token_correct_then_diverge`.
- T19 `T19_FAIL_NO_REPRESENTATION` — representation **1/3**, exact **39/35/34**. Suffix-weight 3.0 closed.
- T19X COMPLETE. Leftover still diverge.
- T20 `T20_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — representation **2/3**, exact **9/10/13**. Head-only CE closed.
- T20X COMPLETE. First-token BOS **42/34/37**. Dominant `other_then_eos`.
- T21 `T21_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — language 3/3, representation **3/3**, exact **27/29/26**. Report: `phase2a_t21_hybrid_answerce_lr3p75e5_1k_v1/T21_FINAL_REPORT.md`.
- T21X COMPLETE. First-token BOS **79/59/72**. Dominant leftover diverge 52/30/46 plus other_then_eos. Report: `phase2a_t21x_generation_failmode_v1/T21X_FINAL_REPORT.md`.
- SmolLM2-1.7B forensic is advisory only. Baby evidence outranks Smol. Qwen reference notes (markdown only) are advisory; do not copy weights.

## GitHub migration (this session)

Preparation only: onboarding files + `.gitignore` + local `git init` + **stage**. **No commit. No push. No GitHub remote.** T22 stays paused.
