# PHASE 2A T20 READOUT-ONLY ANSWER CE — FINAL REPORT

Status: **TERMINAL — study classification `T20_REPRESENTATION_SUCCESS_OUTPUT_FAIL`.**
Per-seed: 790001 `T20_REPRESENTATION_SUCCESS_OUTPUT_FAIL`; 790002 `T20_FAIL_NO_REPRESENTATION`; 790003 `T20_REPRESENTATION_SUCCESS_OUTPUT_FAIL`.
Frozen 2/3 rule: representation-class seeds = 2 ≥ 2. Language held 3/3. Exact/native did not. Do not lower the bar.
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

Watchdog PID 23460 launched 790001–790003 sequentially. Phase A→B (`after_u750_eval`, trainable 21,163,264) logged on all three. `STUDY_COMPLETE T20_REPRESENTATION_SUCCESS_OUTPUT_FAIL` at 2026-09-12T21:11:25Z. Independent U1000 adjudication agrees.

## Treatment
T17/T18 two-phase from Phase1G, uniform `λ_ans=1.0` answer-span CE, but CE **stop-grads through the base**: `logits = language_head(hidden.detach())`. Pointer still trains late-base. No T19 suffix weight. Fact-clause, 9:1, lr 3.75e-5, seeds 790001–790003.
Earned by T19 `T19_FAIL_NO_REPRESENTATION`: suffix-weight 3.0 through late-base missed representation 2/3 and did not lift exact. `language_head` is untied.

Parent SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`.

## Terminal U1000 (Phase B)

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U1000 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 790001 | **119** | 92 | **9** | **56/64** | **13/16** | **64** | **55** | 1.28675 | 0.4276 | T20_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `1341802700380d824b3f6289df97e9ced22e1326420a27a39afe91ffbc19d98e` |
| 790002 | 100 | 80 | **10** | **41/64** | 8/16 | 53 | **47** | 1.28081 | 0.4245 | T20_FAIL_NO_REPRESENTATION | `8ed4b3347ff2b4d8b9380f25f5cc9c84b66090591920f235b6d2f76973c6cee6` |
| 790003 | **112** | 90 | **13** | **48/64** | **11/16** | **58** | **54** | 1.28458 | 0.4306 | T20_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `3af1b2ebea46ba7c9b1e429882fc40b29cc8feef8934360ff32b731c58e7c957` |

U0 DEV CE **1.2040123894810677** bit-identical 3/3. Binding+early intact 3/3. Native reversals 28/17/28. ACQ16 exact+native **fail 3/3** (exact 10/9/12 of 16). TEST `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

U750 SHA256: 790001 `22aad274…e8d307fc`; 790002 `ea8ce3b0…b8ef1d48`; 790003 `98085e26…1bc539d8`.
U750: 790001 110/**46**/10/exact 13/CE 1.289 gap 0.546; 790002 108/**46**/10/exact 8/CE 1.284 gap 0.543; 790003 115/**52**/12/exact 16/CE 1.282 gap 0.534.
Phase B repaired 790001 reversals 46→56. 790002 already missed rev at U750 and dropped further (46→41, td 48→47).

## Frozen-gate adjudication
**Did language hold?** **Yes, 3/3.** CE 1.281–1.287 ≤ 1.30 and ≤ U0+0.25. Gap 0.425–0.431 ≤ 0.5. U750 gaps were 0.534–0.546 (above 0.5); Phase B repaired language as in T18.

**Did representation hold ≥2/3?** **Yes. 2/3.** Same pattern as T17/T18: 790001 and 790003 pass all representation gates; 790002 misses reversals (41) and template-disjoint (47). Stop-grad CE recovered the representation bar T19 lost (1/3).

**Did readout-only CE finish names / keep the T18 exact channel?** **No.** Exact **9 / 10 / 13** vs T18 **36 / 36 / 40** and T19 **39 / 35 / 34**. Native FC 92 / 80 / 90 — no gate pass (T18 was 86/81/86). ACQ16 exact failed 3/3 (T18/T19 passed 3/3). TEST is **not** authorized.

## Interpretation
**Observed in Baby.** Detaching hidden states before `language_head` on the CE term restored T17/T18’s language-safe representation **2/3** and collapsed the T18 generation channel. Suffix completion is **not** a head-only problem under this recipe. Through-base answer CE is what produced T18’s exact 36–40; it is also what T19 over-weighted into a representation tax.

Closed by this result: “uniform `λ_ans=1.0` stop-grad through the base finishes names, or even holds T18 exact, while recovering representation 2/3.” Do not relaunch T20. Do not parent T20/T19/T18 OUTPUT_FAIL checkpoints. Do not raise suffix weight. Do not retokenize.

## Next
T20X: read-only greedy fail-mode on the three T20 U1000 checkpoints. Question: did leftover revert to T17X TinyStories continuation, or stay T18X `first_token_correct_then_diverge` at much lower exact? That decides whether the next train must put some CE back through late-base.
