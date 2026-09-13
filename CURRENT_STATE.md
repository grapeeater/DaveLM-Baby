# Current state (physical)

**As of 2026-09-13.** Stay in `C:\DaveLM-CADAVER`. Do not open T3 TEST. Do not lower the v1.0 bar. T24 is terminal and adjudicated. The owner explicitly released the temporary pause restriction on 2026-09-13 and authorized autonomous successor studies. Historical pause wording is preserved in `T24_PAUSE.md`; it no longer blocks successor selection.

This file matches [`PHASE2A_CURRENT_HANDOFF.md`](PHASE2A_CURRENT_HANDOFF.md). Pause record: [`phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1/T24_PAUSE.md`](phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1/T24_PAUSE.md).

Owner authority (2026-09-13): when a terminal study earns a successor but no protocol exists, the autonomous agent may design, freeze, implement, and execute the highest-information Baby-native successor without waiting for further approval. Existing gates and protected-data locks remain mandatory.

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
`n## T28 FAST V2 (2026-09-13)`nRebuilt from untouched T24 runner; vectorized suffix hard-negative margin (M=1.0, lambda=0.5) only. Seeds 850001-850003 all completed update 1000. Pointer/forced-choice and TRAIN16/language/binding remained healthy; exact generation remained T24-like (36/35/40 of 128). Classification: T28_SUFFIX_MARGIN_VALID_COMPLETION / OUTPUT_FAIL. Protected T3_TEST, T2_EVAL_TEST, FINAL, sacred remained locked. Bundle: C:\DaveLM-CADAVER\phase2a_t28_fast_v2.`n
