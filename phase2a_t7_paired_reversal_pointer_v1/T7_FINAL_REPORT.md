# PHASE 2A T7 PAIRED-REVERSAL POINTER — FINAL REPORT

Status: **TERMINAL — study classification `T7_FAIL_NO_REPRESENTATION` (3/3).**
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

## Treatment
T6 late-base pointer + native margin, binding frozen, blocks 0–3 frozen, 9:1 language. The only change vs T6: every QA update is 16 complete assignment pairs (32 items, pair members adjacent). Parent Phase1G U6000 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`. Seeds 660001–660003.

Watchdog PID 13660. STUDY_COMPLETE 2026-09-11T13:18:08Z.

## U500 results

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | bind/early | class | U500 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 660001 | 63 | 65 | 0 | 9/64 | 0/16 | 33 | 30 | 1.24350 | 0.4383 | intact | T7_FAIL_NO_REPRESENTATION | `cdc17d2413aa872fe110e2c1b20dc58aae796368df1422021b2587c4c7f33fe2` |
| 660002 | 87 | 75 | 0 | 29/64 | 3/16 | 52 | 35 | 1.26251 | 0.4327 | intact | T7_FAIL_NO_REPRESENTATION | `6c38fbda22772415e57f17ab48b5d3150d337d94f40722203797879e73fbca0d` |
| 660003 | 84 | 75 | 0 | 29/64 | 3/16 | 48 | 36 | 1.26076 | 0.4370 | intact | T7_FAIL_NO_REPRESENTATION | `26d9338ba213f1f02c75880e967a69e1dcec200de2059c9b4f33f21932a0b7f1` |

U0 DEV CE 1.2040123894810677. Weight check: **early same, binding same, late base changed** (max |Δ| ≈ 0.011). Distinct hashes. `t3_test_loaded: false`. TEST seal `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

## Frozen-gate adjudication
Language **pass 3/3**. Representation (pointer ≥96, reversals ≥48, families ≥8, nd/td ≥48): **fail 3/3**. No seed hit 96 pointer. Best reversals 29/64 (T6 best was 43/64). Exact 0. Study class **`T7_FAIL_NO_REPRESENTATION`**. TEST not authorized.

## Interpretation vs T6
Paired-assignment batches did **not** produce counterfactual reversals. They **weakened** the T6 retrieval signal (T6 85/92/105 vs T7 63/87/84). Co-present opposite assignments are consistent with gradient cancel on a single cosine pointer: the late-base feature cannot hold both assignments. Do not parent on T7 U500. Do not rerun T6/T7 batch-composition variants. Do not open TEST.

## Next
T8 isolates the remaining T5 confound: freeze 0–3, train late base **and** binding, **loc CE + layout native margin only** (no T3 pointer). Independent-item schedules (not T7 pairs). If loc still stays chance, this NL name-span layout cannot use late-base mentions either. Bundle `phase2a_t8_latebase_loc_layout_v1`.
