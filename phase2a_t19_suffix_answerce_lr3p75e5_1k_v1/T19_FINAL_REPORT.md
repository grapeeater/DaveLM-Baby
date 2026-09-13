# PHASE 2A T19 SUFFIX-WEIGHTED ANSWER CE — FINAL REPORT

Status: **TERMINAL — study classification `T19_FAIL_NO_REPRESENTATION`.**
Per-seed: 780001 `T19_REPRESENTATION_SUCCESS_OUTPUT_FAIL`; 780002 `T19_FAIL_NO_REPRESENTATION`; 780003 `T19_FAIL_NO_REPRESENTATION`.
Frozen 2/3 rule: representation-class seeds = 1 ≠ ≥2. Language held 3/3. Exact/native did not. Do not lower the bar.
T3 TEST was **not** loaded.

Watchdog PID 27980 launched 780001–780003 sequentially. Phase A→B (`after_u750_eval`) logged on all three. `STUDY_COMPLETE T19_FAIL_NO_REPRESENTATION` at 2026-09-12T18:07:47Z. Independent U1000 adjudication agrees.

## Treatment
T17/T18 two-phase from Phase1G, answer-span CE **suffix-weighted** (1.0 first answer token, 3.0 rest + period + EOS). Fact-clause, 9:1, lr 3.75e-5, seeds 780001–780003.
Earned by T18X: leftover exact fails were `first_token_correct_then_diverge` (Skye→Sky, Sal→Salt).

Parent SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`.

## Terminal U1000 (Phase B)

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U1000 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 780001 | **112** | 87 | **39** | **51/64** | **10/16** | **62** | **50** | 1.25557 | 0.4239 | T19_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `14c6596768ecabf9fcabb71cd65eb78b2ee5ac37cb9fef87f04ea4e9a427def4` |
| 780002 | 109 | 86 | **35** | **46/64** | 10/16 | 57 | 52 | 1.25631 | 0.4205 | T19_FAIL_NO_REPRESENTATION | `52738e4edc5b6eb6464442b32ee32d9752dfa5a4354ab4d9b4085d230e60c38d` |
| 780003 | 108 | 83 | **34** | **45/64** | 8/16 | 54 | 54 | 1.25674 | 0.4258 | T19_FAIL_NO_REPRESENTATION | `bb494bc0b18ce9e95ff813d8c95376020e56b0a3faeec19982aa4b0204a88f58` |

U0 DEV CE **1.2040123894810677** bit-identical 3/3. ACQ16 exact pass 3/3. TEST `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

U750 SHA256: 780001 `a7ddef51…73b461ef`; 780002 `cfcb9d62…31409716`; 780003 `840652ad…746f0c9b2`.
U750: 780001 114/54/11/exact 35; 780002 117/**53**/11/exact 32 (Phase B dropped rev 53→46); 780003 109/**46**/9/exact 34.

## Frozen-gate adjudication
**Did language hold?** **Yes, 3/3.** CE 1.256 ≤ 1.30. Gap 0.421–0.426 ≤ 0.5.

**Did representation hold ≥2/3?** **No. 1/3.** Only 780001 clears all representation gates. T17 and T18 were 2/3 on the same two-phase recipe. 780002 lost reversals in Phase B (53→46). 780003 never reached 48.

**Did suffix-weighted CE finish names after the correct first token?** **No.** Exact 39/35/34 vs T18’s 36/36/40. Native FC 87/86/83 vs T18’s 86/81/86. No material lift. TEST is **not** authorized.

## Interpretation
**Observed in Baby.** Heavier CE on later answer tokens did not close Sky/Salt-style completions and coincided with a representation-bar miss that T17/T18 did not have. U300 pointer on 780001/780002/780003 lagged T18 (~70 vs ~90–100). Answer-CE gradients go through trainable late-base (`language_head` is **untied**); extra suffix weight is extra hidden-state motion on the same modules the pointer needs.

Closed by this result: “suffix-weight 3.0 on the T17 recipe finishes names and holds representation 2/3.” Do not relaunch T19. Do not raise suffix weight. Do not parent T19/T18 OUTPUT_FAIL checkpoints. Do not retokenize.

## Next
T19X confirm leftover modes still Sky/Salt. Then T20: T18-style uniform `λ_ans=1.0` with **stop-grad through the base** on the CE term so only `language_head` is trained by answer CE. Pointer still trains late-base. Tests whether suffix completion is a readout problem without rewriting the representation circuit.
