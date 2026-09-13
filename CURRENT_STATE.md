# Current state (physical)

**As of 2026-09-13.** Stay in `C:\DaveLM-CADAVER`. Do not open T3 TEST. Do not lower the v1.0 bar. **T23 and T23X are terminal. T24 is the live train.**

This file matches [`PHASE2A_CURRENT_HANDOFF.md`](PHASE2A_CURRENT_HANDOFF.md).

## Parent (Phase1G)

| | |
|---|---|
| Path (local-only) | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001\checkpoints\best.pt` |
| SHA-256 | `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1` |
| U0 DEV CE | `1.2040123894810677` |

## TEST

Status **`SEALED_UNOPENED`**. `qa_test.jsonl` SHA-256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`.

## Live — T24

Bundle: `phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1`  
Seeds 830001–830003. T18 full-span CE + train/dev name-inventory unlikelihood (24 names; TEST never loaded). Do not parent T23 / 820001–820003.

## Closed this session

- T23 `T23_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — language 3/3, representation **3/3**, exact **34/34/38**. U1000 SHA256: 820001 `a68ffecf…f4cb0298`; 820002 `e77468e7…4728bcf5`; 820003 `f3644a71…7be0a888`.
- T23X COMPLETE. BOS 79/75/79. Leftover still diverge; off-row Walt/York/Sky.

GitHub: https://github.com/grapeeater/DaveLM-Baby.git
