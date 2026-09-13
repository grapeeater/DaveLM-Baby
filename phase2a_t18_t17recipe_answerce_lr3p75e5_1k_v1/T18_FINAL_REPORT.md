# PHASE 2A T18 T17-RECIPE + ANSWER-SPAN CE — FINAL REPORT

Status: **TERMINAL — study classification `T18_REPRESENTATION_SUCCESS_OUTPUT_FAIL`.**
Per-seed: 770001 `T18_REPRESENTATION_SUCCESS_OUTPUT_FAIL`; 770002 `T18_FAIL_NO_REPRESENTATION`; 770003 `T18_REPRESENTATION_SUCCESS_OUTPUT_FAIL`.
Frozen 2/3 rule: representation-class seeds = 2 ≥ 2. Language held 3/3. Native/exact did not. Do not lower the bar.
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

Watchdog PID 17912 launched 770001–770003 sequentially. Phase A→B (`after_u750_eval`, trainable 21,163,264) logged on all three seeds. `STUDY_COMPLETE T18_REPRESENTATION_SUCCESS_OUTPUT_FAIL` at 2026-09-12T14:38:37Z. Independent U1000 adjudication agrees.

## Treatment
T17 two-phase recipe from Phase1G, plus answer-span teacher-forced CE (`λ_ans=1.0`) on the correct candidate tokens + EOS during QA updates. Fact-clause pointer, 9:1, lr 3.75e-5, 1000 updates, seeds 770001–770003.
- **Phase A (1–750):** freeze 0–3 + binding; train 4–11 + embed/ln/head (40,849,664).
- **After U750 eval:** copy Phase1G into blocks 8–11, freeze them, rebuild AdamW.
- **Phase B (751–1000):** train 4–7 + embed/ln/head (21,163,264).
- QA loss = margin + `λ_ptr` pointer + `λ_ans` answer-span CE (prompt not supervised).

Tests whether teaching `language_head` to emit the answer converts T17’s language-safe representation into native/exact output. Do not parent T14/T15/T16/T17/730002/760001–760003. Mix unchanged.

Parent SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`.

## Terminal U1000 (Phase B)

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U1000 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 770001 | **116** | 86 | **36** | **52/64** | **10/16** | **60** | **56** | 1.25451 | 0.4248 | T18_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `308405b0432092002f90ff7fab476b1cb157982fe13dcaef7032f9556a83adf2` |
| 770002 | 108 | 81 | **36** | **46/64** | 9/16 | 55 | 53 | 1.25499 | 0.4205 | T18_FAIL_NO_REPRESENTATION | `6d963f3e48c9538b664e0cf18a02ab013120d467f07b8ee49e90371c2298f0df` |
| 770003 | **114** | 86 | **40** | **54/64** | **11/16** | **61** | **53** | 1.25461 | 0.4224 | T18_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `ec0c9bc0c1a47d3b429b325658efffcde4da78594131b73cda0b8c011187bccc` |

U0 DEV CE **1.2040123894810677** bit-identical 3/3. Binding+early intact 3/3. Native reversals 23/18/22. ACQ16 exact+native **pass 3/3** (16/16 exact on 770001 and 770003; 770002 16 exact / 15 ptr). TEST `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

U500 SHA256: 770001 `108d904d…869c8a06`; 770002 `b63318fb…9a08cb9a`; 770003 `3def68de…51e1a3a9`.
U750 SHA256: 770001 `087c1b27…2f55a669`; 770002 `6f9091c1…e9db1d38`; 770003 `badc1bc8…07cec83b`.

U750 (Phase A endpoint): 770001 119 ptr / **55** rev / 12 fam / exact 31 / CE 1.268 gap 0.488; 770002 108 / **45** / 9 / exact 32 / CE 1.268 gap 0.486; 770003 123 / **59** / 13 / exact 32 / CE 1.268 gap 0.488.

## Frozen-gate adjudication
**Did language hold?** **Yes, 3/3.** Terminal CE 1.255 ≤ 1.30 and ≤ U0+0.25. Gap 0.421–0.425 ≤ 0.5. Phase B repaired language vs U750 (CE 1.268 → 1.255; gap 0.486–0.488 → 0.421–0.425).

**Did representation hold ≥2/3?** **Yes. 2/3.** Same pattern as T17: 770001 and 770003 pass all representation gates; 770002 misses reversals (46). Answer-span CE did **not** destroy the T17 representation recipe.

**Did answer-span CE convert representation into native/exact output?** **No.**
- Exact moved from T17’s **0** to **36 / 36 / 40**. Real, replicated, far from 80.
- Native FC 86 / 81 / 86 — still below 96 (T17 was 88 / 90 / 86). No FC lift.
- Native reversals 23 / 18 / 22 — still far from 48.
- ACQ16 exact can pass while DEV exact stays ~30%. TEST is **not** authorized.

## Interpretation
**Observed in Baby.** Teacher-forced answer CE opens a generation channel the T17 pointer-only objective never had. It is enough for ACQ16 (16 short, in-template items) and about a third of DEV, and it does not cost language or the 2/3 representation recipe. It is not enough for DEV exact 80 or native ranking 96.

**Observed in Baby, fits Smol lesson 1 (not a Smol result).** Representation, ranking, and greedy token generation remain distinct stages. T17X’s TinyStories continuation after `Answer:` is only partly overwritten. Native FC barely moved, so the residual failure is not “more of the same CE will automatically lift ranking.”

Closed by this result: “λ_ans=1.0 answer-span CE on the T17 recipe yields native/exact 2/3.” Do not relaunch T18. Do not parent T18 OUTPUT_FAIL checkpoints. Do not raise `λ_ans` or add duration without a fail-mode split.

## Next
Read-only T18X generation fail-mode on T18 U1000 vs the T17X continuation baseline. No TEST. Distinguishes remaining story-continuation / wrong first token vs prefix-correct / missing-EOS. That decides whether the next train is first-token/anti-continuation, EOS/stop, or something else. Do not change the mix. Do not copy Qwen or Smol.
