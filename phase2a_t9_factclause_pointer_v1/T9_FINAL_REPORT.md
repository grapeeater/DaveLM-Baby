# PHASE 2A T9 FACT-CLAUSE POINTER — FINAL REPORT

Status: **TERMINAL — study classification `T9_FAIL_NO_REPRESENTATION` (1/3 representation success; 2/3 FAIL_NO_REPRESENTATION).**
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

## Treatment
T6 late-base pointer + native margin, binding frozen, blocks 0–3 frozen, 9:1 language. Pointer **keys = fact-clause** (name mention through next period token 18). Parent Phase1G U6000 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`. Seeds 680001–680003.

Watchdog PID 9924. STUDY_COMPLETE 2026-09-11T15:51:29Z.

## U500 results

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U500 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 680001 | 113 | 80 | 0 | 49/64 | 10/16 | 60 | 53 | 1.26826 | 0.4825 | T9_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `3769bf9e02248df977cc227bbf1ea47e2dc8abbc5af2e8685fc6fd6c6fbe0280` |
| 680002 | 97 | 75 | 0 | 36/64 | 5/16 | 51 | 46 | 1.27916 | 0.4891 | T9_FAIL_NO_REPRESENTATION | `be4da87d7a6d99079bf2b48dd72f828187a4f216b14dbfd2917eea3ea7952be1` |
| 680003 | 95 | 74 | 0 | 33/64 | 5/16 | 56 | 39 | 1.26562 | 0.4794 | T9_FAIL_NO_REPRESENTATION | `718168c0eb7112fdd324b58036bdcf4d1a16fd490ce1ccdcd42ff153be227afd` |

U0 DEV CE 1.2040123894810677. 680001 U0 pointer was 67/128 (fact-clause at init). Weight check: **early same, binding same, late base changed** (max |Δ| ≈ 0.010–0.011). Shortcuts near chance (not a shortcut class). `t3_test_loaded` absent/false. TEST seal `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

## Frozen-gate adjudication
Language **pass 3/3**. Representation requires pointer ≥96 **and** reversals ≥48 **and** families ≥8 **and** nd/td ≥48. **Only 680001** met all of those (113, 49, 10, 60, 53). Native FC 80<96, exact 0, ACQ16 fail → `T9_REPRESENTATION_SUCCESS_OUTPUT_FAIL`. 680002 met pointer (97) but missed reversals/families/td. 680003 missed pointer by 1 (95). Study rule needs ≥2 representation-class seeds → **`T9_FAIL_NO_REPRESENTATION`**. TEST scoring is **not** authorized. Do not lower the bar to 1/3.

## Interpretation vs T6
Fact-clause keys beat name-span keys on the reversal problem **when they work**: 680001 49/64 reversals vs T6 best 43/64, and 113 vs 105 pointer. That supports the assignment-invariant name-identity hypothesis. It did **not** replicate in 2/3 seeds at 500 updates. 680001 was still near chance at U300 (69 ptr / 13 rev) then jumped by U500 — consistent with late takeoff / undertraining, not with a dead mechanism. Do not parent on 680001 until representation replicates. Do not open TEST.

## Next
T10: identical T9 fact-clause late-base pointer, Phase1G parent, new seeds 690001–690003, **1000 updates** (eval 0/100/300/500/750/1000), stop seed if DEV CE > 1.30. Tests whether 500 updates undertrained 680002/680003. Bundle `phase2a_t10_factclause_pointer_1k_v1`.
