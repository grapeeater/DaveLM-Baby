# PHASE 2A T6 LATE-BASE POINTER — FINAL REPORT

Status: **TERMINAL — study classification `T6_FAIL_NO_REPRESENTATION` (3/3).**
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

## Treatment
T3 pointer + T2 native margin (no layout, no loc CE). Freeze `base_model.blocks.0–3` and all 984,321 binding params. Train late base only (40,849,664). Parent = Phase1G U6000 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`. 9:1 language, 500 updates, seeds 650001–650003.

Watchdog PID 21124. STUDY_COMPLETE 2026-09-11T12:00:47Z.

## U500 results

| seed | ptr | native | exact | ptr rev | nat rev | ptr fam | nd ptr | td ptr | DEV CE | gap | bind/early | class | U500 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 650001 | 92 | 72 | 0 | 28/64 | 12/64 | 5/16 | 51 | 41 | 1.26908 | 0.4687 | intact | T6_FAIL_NO_REPRESENTATION | `8e1ff05990a2a31bd851c4cdeb16f7335575afc7fe905d304934f2259f85d8e0` |
| 650002 | 105 | 94 | 0 | 43/64 | 33/64 | 9/16 | 55 | 50 | 1.26370 | 0.4612 | intact | T6_FAIL_NO_REPRESENTATION | `c975ce33285c71421a5d392c5a7648198dc8f456f38839ee1bb81dc7ae2579d4` |
| 650003 | 85 | 70 | 0 | 27/64 | 9/64 | 5/16 | 43 | 42 | 1.26442 | 0.4981 | intact | T6_FAIL_NO_REPRESENTATION | `a7ae2eb0e5deb994f86e7e8fb6b833490c8ce776f262fbb365c561cb6e4dd1a7` |

U0 DEV CE 1.2040123894810677. Weight check: **early same, binding same, late base changed** (max |Δ| ≈ 0.010–0.012). Distinct hashes. `t3_test_loaded: false`. TEST seal `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

## Frozen-gate adjudication
Language (CE ≤ 1.30 and gap ≤ 0.5): **pass 3/3**. Representation requires pointer ≥96 **and** reversals ≥48 **and** complete families ≥8 **and** name-disjoint ≥48 **and** template-disjoint ≥48. Only 650002 met pointer (105), families (9), nd (55), td (50); it missed reversals (43/64). 650001 missed pointer/reversals/families/td. 650003 missed all representation gates. Native exact 0/128 all seeds. Shortcuts near chance. Study class **`T6_FAIL_NO_REPRESENTATION`**. TEST not authorized.

## Interpretation vs T3/T5
T3 retrieval did **not** require early-block updates: late-base pointer can move held-out retrieval (one seed 105/128) while keeping CE under 1.30 (T3 could not). T5’s chance pointer (61–67) was the mixed loc/layout loss, not the freeze. Counterfactual reversals remain the failure: even 105/128 retrieval only got 43/64 reversals (T3: 35–39/64 at 97–102 pointer). Extra retrieval does not buy assignment completeness. Do not parent on T6 U500. Do not open TEST. Do not rerun T6 with more of the same independent-item SGD.

## Next
T7: same T6 scope/parent/objective, but every QA update is **16 complete assignment pairs** (32 items) so SGD cannot fit one assignment of a pair without the other. Bundle `phase2a_t7_paired_reversal_pointer_v1`.
