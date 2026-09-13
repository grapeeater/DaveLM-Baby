# Phase 2A current handoff (authoritative)

Stay in `C:\DaveLM-CADAVER`. Do not open T3 TEST. Do not lower the v1.0 bar.

## Parent
Phase1G U6000 SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`
U0 DEV CE `1.2040123894810677`
TEST `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`)

## Closed lines
- T3–T16 / T16X / Qwen: as previously closed. Do not parent T14/T15/T16/730002.
- T17 `T17_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — representation **2/3**, exact 0.
- T17X COMPLETE. Exact 0 was Phase1G story continuation after `Answer:`.
- T18 `T18_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — language 3/3, representation **2/3**, exact **36/36/40**.
- T18X COMPLETE. First-token BOS 76–84. Leftover `first_token_correct_then_diverge`.
- T19 `T19_FAIL_NO_REPRESENTATION` — representation **1/3**, exact **39/35/34**. Suffix-weight 3.0 closed.
- T19X COMPLETE. Leftover still diverge.
- T20 `T20_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — representation **2/3**, exact **9/10/13**. Head-only CE closed.
- T20X COMPLETE. First-token BOS **42/34/37**. Dominant `other_then_eos`.
- T21 `T21_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — language 3/3, representation **3/3**, exact **27/29/26**. Report: `phase2a_t21_hybrid_answerce_lr3p75e5_1k_v1\T21_FINAL_REPORT.md`.
- T21X COMPLETE. First-token BOS **79/59/72**. Dominant leftover diverge 52/30/46 plus other_then_eos. Report: `phase2a_t21x_generation_failmode_v1\T21X_FINAL_REPORT.md`.
- SmolLM2-1.7B forensic is advisory only. Baby evidence outranks Smol.

Do not parent T14–T21 OUTPUT_FAIL checkpoints. Do not relaunch T4–T21. Do not copy Qwen or Smol. Do not retokenize.

## Live — T22 PAUSED / physically quiescent
T22 `phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1` is **PAUSED**. Do **not** resume until CONTINUE.

- Seed **810001** only. Phase **A**. Resume from `rolling_restart.pt` **`completed=550`**.
- SHA256 `rolling_restart.pt` `ff1000a8a0b117c7f9f28e866e9cf1d4920cceed59031c8c3bf69df8a10047b7`
- Last eval U500 `checkpoint_0500.pt` SHA256 `0ba71d0967fce73300170270b3593c7818f1e6deba8eb75059d2e9863dc8cfd7`
- T22 watchdog 23396 and runner 22592/27036 **stopped**. No T22 processes live.
- **810002 / 810003 never started.** No T22X / T23 launched.
- Pause record: `phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1\T22_PAUSE.md`
- TEST sealed. Frozen recipe unchanged. Phase A→B still after U750 eval when resumed.
