# PHASE 2A T10 FACT-CLAUSE POINTER 1K — FINAL REPORT

Status: **TERMINAL — study classification `T10_LANGUAGE_REGRESSION` (3/3).**
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

## Treatment
T9 fact-clause late-base pointer + native margin, binding frozen, blocks 0–3 frozen, 9:1 language, **1000 updates**. Parent Phase1G U6000 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`. Seeds 690001–690003. Stop seed if DEV CE > 1.30.

Watchdog PID 3016. STUDY_COMPLETE 2026-09-11T18:23:38Z.

## Terminal U1000 (frozen study class)

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U1000 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 690001 | 98 | 83 | 0 | 40/64 | 7/16 | 51 | 47 | 1.30057 | 0.6216 | T10_LANGUAGE_REGRESSION | `3c58f0502ce6fdebdd14cc9897fc68b0dd9fbc8b8713f32ea047b25d524af0b1` |
| 690002 | 108 | 89 | 0 | 49/64 | 10/16 | 58 | 50 | 1.30680 | 0.6330 | T10_LANGUAGE_REGRESSION | `99e057f42be7712945e0e7ad5c891989deef36a11cb4c97472c6ccffb9baf9f2` |
| 690003 | 115 | 88 | 0 | 52/64 | 12/16 | 60 | 55 | 1.30149 | 0.6081 | T10_LANGUAGE_REGRESSION | `d416100e2f085f472ba66fe5138398ab6acf5bfa468f1668c4483c31d60fbea1` |

U0 DEV CE 1.2040123894810677 (690001 bit-identical). Early+binding intact 3/3. Exact 0. TEST `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

## Language-safe U500 (same horizon as T9; not the study class)

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U500 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 690001 | 104 | 86 | 0 | 42/64 | 9/16 | 58 | 46 | 1.27015 | 0.4695 | T10_FAIL_NO_REPRESENTATION | `5850860482acc079d4d80b838a4a2ef8ae6476f3df0af095c9fab092ebd3fb8a` |
| 690002 | 104 | 81 | 0 | 44/64 | 9/16 | 56 | 48 | 1.27295 | 0.4853 | T10_FAIL_NO_REPRESENTATION | `b9d3ac80e5c43abc7cec339d704bc2b5c9463ac549b3c9a9326440e23090180a` |
| 690003 | 105 | 80 | 0 | 46/64 | 9/16 | 56 | 49 | 1.27123 | 0.4581 | T10_FAIL_NO_REPRESENTATION | `2800c4b63e591d5851317e63ed07a05c9742be507998e4763ba9bded7e0b76b7` |

**0/3 representation at U500.** Combined with T9: **1/6** language-safe 500-update fact-clause seeds hit representation. 680001 was a late-takeoff outlier, not a replicable 500-update result.

## U750 (representation takeoff; language already broken)

| seed | ptr | ptr rev | fam | nd | td | DEV CE | gap | class |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 690001 | 100 | 39/64 | 9 | 54 | 46 | 1.28560 | 0.5518 | T10_LANGUAGE_REGRESSION |
| 690002 | 109 | **48/64** | 11 | 60 | 49 | 1.29146 | 0.5647 | T10_LANGUAGE_REGRESSION |
| 690003 | 112 | **51/64** | 12 | 60 | 52 | 1.29151 | 0.5327 | T10_LANGUAGE_REGRESSION |

U750 SHA256: 690001 `89091f03…f57f7984`; 690002 `71002ea4…c621aeec`; 690003 `e7564c0e…536d64ba`.

Representation **metrics** hit on 2/3 at U750, but the language gate (CE ≤ 1.30 **and** gap ≤ 0.5) already failed. Frozen study class uses terminal U1000 → `T10_LANGUAGE_REGRESSION`. Do not lower the bar to credit U750.

## Frozen-gate adjudication
Language **fail 3/3** (CE 1.30057–1.30680; all stop_code `language_ce_gt_1_30`). Gap already > 0.5 by U750. TEST scoring is **not** authorized. Do not parent U750/U1000 (language fail) or 680001 (unreplicated at 500).

## Interpretation
T10 asked undertraining vs 1/3 luck. **Both, and they do not cancel.** Fact-clause late-base retrieval is real: extra updates after 500 raise reversals (690002/690003). The 9:1 mix at 5e-5 cannot hold language through that takeoff. 500 updates is language-safe and **not** a 2/3 representation recipe.

## Next
T11: same fact-clause late-base pointer, Phase1G parent, new seeds 700001–700003, **750 updates**, **4:1 QA:language** (lang every 5th), eval 0/100/300/500/750, stop seed if DEV CE > 1.30. Tests whether extra language rehearsal keeps the U750 representation takeoff under the language cap. Bundle `phase2a_t11_factclause_pointer_4to1_750_v1`.
