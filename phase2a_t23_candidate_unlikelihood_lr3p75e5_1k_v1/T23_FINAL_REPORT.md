# PHASE 2A T23 CANDIDATE UNLIKELIHOOD — FINAL REPORT

Status: **TERMINAL — study classification `T23_REPRESENTATION_SUCCESS_OUTPUT_FAIL`.**
Per-seed: 820001 / 820002 / 820003 all `T23_REPRESENTATION_SUCCESS_OUTPUT_FAIL`.
Frozen 2/3 rule: representation-class seeds = 3 ≥ 2. Language held 3/3. Exact/native did not. Do not lower the bar.
T3 TEST was **not** loaded.

Watchdog PID **36984** (no successor) launched 820001 at 2026-09-13T06:20:03Z, 820002 at 07:23:14Z, 820003 at 08:25:56Z. Phase A→B (`after_u750_eval`) on all three. `STUDY_COMPLETE T23_REPRESENTATION_SUCCESS_OUTPUT_FAIL` at 2026-09-13T09:28:53Z. Independent U1000 adjudication agrees.

## Treatment
T17/T18 two-phase from Phase1G, T18 full-span through-base `λ_ans=1.0`, **plus** in-row wrong-candidate unlikelihood `λ_unl=1.0` at aligned answer positions. Fact-clause, 9:1, lr 3.75e-5, seeds 820001–820003.
Earned by T22/T22X: CE-span-count closed; leftover T18X diverge (`Wes.`→`Walt.`, `Skye.`→`Seth.`, `Ava.`→`York.`).

Parent SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`.

## Terminal U1000 (Phase B)

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U1000 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 820001 | **120** | 91 | **34** | **56/64** | **12/16** | **64** | **56** | 1.25785 | 0.4274 | T23_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `a68ffecff44630677d9a1693ce8f536a2b57250e3c0626aafcf24335f4cb0298` |
| 820002 | **115** | 87 | **34** | **53/64** | **9/16** | **59** | **56** | 1.25835 | 0.4235 | T23_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `e77468e71a3bcf9c1fb8962f5a3f85a2680ce274c13a4253822487a24728bcf5` |
| 820003 | **114** | 88 | **38** | **50/64** | **10/16** | **61** | **53** | 1.25890 | 0.4296 | T23_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `f3644a7175b9e51d7825816688d8e1c4768269d93dcb5016c9e535c87be0a888` |

U0 DEV CE **1.2040123894810677** bit-identical 3/3. Binding+early intact 3/3. Native reversals 27/24/26. ACQ16 runner-pass 3/3 (same T18 pattern: ACQ16 can pass while DEV exact stays ~30%). TEST `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

U750 SHA256: 820001 `2571ff49…3fa091dd`; 820002 `c00687ea…eb3f81a1`; 820003 `ec876085…cec10109`.
U750: 820001 111/**48**/9/exact 28/CE 1.275 gap 0.496; 820002 109/**45**/8/exact 30/CE 1.277 gap 0.496 (`FAIL_NO_REPRESENTATION`); 820003 113/**50**/9/exact 34/CE 1.273 gap **0.503** (`LANGUAGE_REGRESSION`). Phase B repaired language and 820002 reversals.

## Frozen-gate adjudication
**Did language hold?** **Yes, 3/3.** Terminal CE 1.258–1.259 ≤ 1.30. Gap 0.424–0.430 ≤ 0.5.

**Did representation hold ≥2/3?** **Yes. 3/3.** All representation gates pass at U1000 (T18 was 2/3).

**Did rival-token unlikelihood lift exact above T18 36–40?** **No.** Exact **34 / 34 / 38** vs T18 **36 / 36 / 40**. Native FC 91 / 87 / 88. TEST is **not** authorized.

## Interpretation
**Observed in Baby.** In-row unlikelihood held language and improved representation to 3/3 but did not buy T18’s exact ceiling. Exact is statistically T18, not above it. Raising `λ_unl` would repeat T19’s “more pressure on the same term” error.

Closed: “in-row wrong-candidate unlikelihood recovers T18 exact or lifts past it.” Do not relaunch T23. Do not parent 820001–820003. Do not raise `λ_unl`. Do not reopen CE-span-count. Do not retokenize.

## Next
T23X: read-only greedy fail-mode. Did unlikelihood change leftover vs T18X/T22X (Walt/Seth/York), or is residual still the same diverge family?
