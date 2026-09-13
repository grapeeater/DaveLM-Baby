# PHASE 2A T22 FIRST-TWO ANSWER CE — FINAL REPORT

Status: **TERMINAL — study classification `T22_REPRESENTATION_SUCCESS_OUTPUT_FAIL`.**
Per-seed: 810001 `T22_REPRESENTATION_SUCCESS_OUTPUT_FAIL`; 810002 `T22_REPRESENTATION_SUCCESS_OUTPUT_FAIL`; 810003 `T22_FAIL_NO_REPRESENTATION`.
Frozen 2/3 rule: representation-class seeds = 2 ≥ 2. Language held 3/3. Exact/native did not. Do not lower the bar.
T3 TEST was **not** loaded.

Resumed from user CONTINUE: U550 `rolling_restart` SHA256 `ff1000a8…10047b7` unchanged. Watchdog 29808 then 31728. Phase A→B (`after_u750_eval`, trainable 21,163,264) on all three. `STUDY_COMPLETE T22_REPRESENTATION_SUCCESS_OUTPUT_FAIL` at 2026-09-13T06:14:55Z. Independent U1000 adjudication agrees.

## Treatment
T17/T18 two-phase from Phase1G, uniform mean `λ_ans=1.0`: **first TWO answer tokens CE through the base**, tail (rest of name + period + EOS) stop-grad. Fact-clause, 9:1, lr 3.75e-5, seeds 810001–810003.
Earned by T21/T21X: first-token through-base recovered BOS toward T18 and exact 27/29/26; leftover T18X diverge.

Parent SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`.

## Terminal U1000 (Phase B)

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U1000 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 810001 | **116** | 88 | **21** | **52/64** | **9/16** | **60** | **56** | 1.27667 | 0.4289 | T22_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `22143cbfb590f40142bf7a0652db0b97e64f989db8f2202c044b7745d3d11b83` |
| 810002 | **110** | 76 | **31** | **48/64** | **10/16** | **60** | **50** | 1.27978 | 0.4262 | T22_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `0115962ec85b8fb1de95632203afa772359860251dba6e61b86e92bb03b38ecf` |
| 810003 | 110 | 85 | **26** | **47/64** | 8/16 | 54 | 56 | 1.27467 | 0.4274 | T22_FAIL_NO_REPRESENTATION | `ed1e6a8ebc2a1754132746b004a4f57ed23a1cb723e5987b13d777dc42a42aff` |

U0 DEV CE **1.2040123894810677** bit-identical 3/3. Binding+early intact 3/3. Native reversals 25/12/25. ACQ16 fail **3/3**. TEST `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

U750 SHA256: 810001 `79833ae8…5fa68a1a`; 810002 `0fd6b889…b3d3222e`; 810003 `efa48077…ae62c738`.
U750: 810001 115/**53**/12/exact 15; 810002 112/**51**/10/exact 25; 810003 113/**50**/11/exact 14.
Phase B dropped 810003 reversals 50→47.

## Frozen-gate adjudication
**Did language hold?** **Yes, 3/3.** CE 1.275–1.280 ≤ 1.30. Gap 0.426–0.429 ≤ 0.5.

**Did representation hold ≥2/3?** **Yes. 2/3.** 810003 misses reversals (47). T21 was **3/3**; first-two through-base did not keep that.

**Did first-two-tokens-through-base recover T18 exact?** **No.** Exact **21 / 31 / 26** vs T18 **36 / 36 / 40** and T21 **27 / 29 / 26**. No material lift vs T21; 810001 is worse. Native FC 88 / 76 / 85. TEST is **not** authorized.

## Interpretation
**Observed in Baby.** Adding the second answer token to the through-base CE term did not buy T18’s extra ~9 exact and cost T21’s 3/3 representation bar. Dose-response on exact: T20 head-only 9–13 → T21 first-token 26–29 → T22 first-two 21–31 → T18 full-span 36–40. The second token is not the missing T18 term.

Closed: “first two answer tokens through the base recover T18 exact while holding representation ≥2/3.” Do not relaunch T22. Do not parent 810001–810003. Do not raise suffix weight. Do not retokenize.

## Next
T22X: read-only greedy fail-mode. Confirm leftover vs T21X/T18X before any further CE-span train.
