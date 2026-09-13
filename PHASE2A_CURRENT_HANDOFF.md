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
- T21 `T21_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — language 3/3, representation **3/3**, exact **27/29/26**.
- T21X COMPLETE. First-token BOS **79/59/72**. Dominant leftover diverge 52/30/46 plus other_then_eos.
- T22 `T22_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — language 3/3, representation **2/3**, exact **21/31/26**. First-two through-base closed.
- T22X COMPLETE. First-token BOS **66/65/73**. Leftover still diverge. CE-span-count family closed.
- T23 `T23_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — language 3/3, representation **3/3**, exact **34/34/38**. In-row rival unlikelihood did not beat T18 exact. Report: `phase2a_t23_candidate_unlikelihood_lr3p75e5_1k_v1\T23_FINAL_REPORT.md`.
- T23X COMPLETE. First-token BOS **79/75/79**. Leftover still diverge; Walt/York are off-row inventory names. Report: `phase2a_t23x_generation_failmode_v1\T23X_FINAL_REPORT.md`.
- SmolLM2-1.7B forensic is advisory only. Baby evidence outranks Smol.

Do not parent T14–T23 OUTPUT_FAIL checkpoints (760001–820003). Do not relaunch T4–T23. Do not copy Qwen or Smol. Do not retokenize. Do not raise suffix weight or `λ_unl`. Do not reopen CE-span-count.

## Live — T24 PAUSED (do not resume)
T24 `phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1` is **physically quiescent**. Seeds 830001–830003 all have `FINAL_STATUS.json` at U1000 Phase B. Watchdog 34608 had already exited after `STUDY_COMPLETE` 2026-09-13T12:52:23Z. Leftover T24 cmd 1760 stopped. No T24 processes remain.

This is **not** a scientific close-out. Do **not** adjudicate. Do **not** launch T24X / T25. Do **not** parent 830001–830003. TEST sealed.

Resume artifact (last seed): `phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1_run_seed830003\rolling_restart.pt` SHA256 `e00f9b625d31c9c5615d3c639422362b585c92427828c254dc639704b9a186c9` (415,491,817 bytes, `completed=1000` phase B). Pause record: `phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1\T24_PAUSE.md`.

Resume command (documented, **not run**):

```
cmd /c start "" /min C:\DaveLM-CADAVER\phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1\LAUNCH_WATCHDOG.cmd
```

That command would see all three `FINAL_STATUS` files and would not train. Do not run it until CONTINUE.
