# Current state (physical)

**As of 2026-09-13.** Stay in `C:\DaveLM-CADAVER`. Do not open T3 TEST. Do not lower the v1.0 bar. **T22 and T22X are terminal. T23 is the live train.**

This file matches [`PHASE2A_CURRENT_HANDOFF.md`](PHASE2A_CURRENT_HANDOFF.md).

## Parent (Phase1G)

| | |
|---|---|
| Path (local-only) | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001\checkpoints\best.pt` |
| SHA-256 | `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1` |
| U0 DEV CE | `1.2040123894810677` |

## TEST

Status **`SEALED_UNOPENED`**. Seal file: `phase2a_t3_rebuilt_study_v1/data/TEST_SEAL.json`. Bytes are **not** in git.

`qa_test.jsonl` SHA-256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`.

## Live — T23

Bundle: `phase2a_t23_candidate_unlikelihood_lr3p75e5_1k_v1`  
Seeds 820001–820003. T18 full-span answer CE + in-row rival unlikelihood. Detached `LAUNCH_WATCHDOG.cmd`. Do not parent T22 / 810001–810003.

GitHub: https://github.com/grapeeater/DaveLM-Baby.git

## Closed this session

- T22 `T22_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — language 3/3, representation **2/3**, exact **21/31/26**. U1000 SHA256: 810001 `22143cbf…d3d11b83`; 810002 `0115962e…b38ecf`; 810003 `ed1e6a8e…42a42aff`.
- T22X COMPLETE. BOS 66/65/73. Leftover still T18X diverge. CE-span-count family closed.

## Closed lines (do not relaunch / do not parent OUTPUT_FAIL)

See [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md). Headline T17–T22 as in the handoff. Do not parent T14–T22 OUTPUT_FAIL checkpoints.
