# Current state (physical)

**As of 2026-09-13.** Stay in `C:\DaveLM-CADAVER`. Do not open T3 TEST. Do not lower the v1.0 bar. **T24 is PAUSED / physically quiescent. Do not resume. Do not launch T24X / T25.**

This file matches [`PHASE2A_CURRENT_HANDOFF.md`](PHASE2A_CURRENT_HANDOFF.md). Pause record: [`phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1/T24_PAUSE.md`](phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1/T24_PAUSE.md).

## Parent (Phase1G)

| | |
|---|---|
| Path (local-only) | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001\checkpoints\best.pt` |
| SHA-256 | `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1` |
| U0 DEV CE | `1.2040123894810677` |

## TEST

Status **`SEALED_UNOPENED`**. `qa_test.jsonl` SHA-256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`.

## Live — T24 PAUSED (wait)

Bundle: `phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1`  
Seeds 830001–830003 all have `FINAL_STATUS.json` at update 1000 Phase B. Watchdog 34608 had already exited after `STUDY_COMPLETE` 2026-09-13T12:52:23Z. Leftover T24 cmd PID 1760 terminated. **No T24 processes remain.**

Last live restart artifact: `phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1_run_seed830003\rolling_restart.pt`  
size 415,491,817; SHA256 `e00f9b625d31c9c5615d3c639422362b585c92427828c254dc639704b9a186c9`; `completed=1000` phase B.

Do **not** adjudicate T24 scientifically in this pause. Do **not** parent 830001–830003. Do **not** run `LAUNCH_WATCHDOG.cmd`.

GitHub: https://github.com/grapeeater/DaveLM-Baby.git
