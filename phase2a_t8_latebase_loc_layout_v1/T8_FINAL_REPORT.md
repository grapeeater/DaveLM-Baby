# PHASE 2A T8 LATE-BASE LOC + LAYOUT — FINAL REPORT

Status: **TERMINAL — study classification `T8_FAIL_NO_ROUTING` (3/3).**
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

## Treatment
Freeze `base_model.blocks.0–3`. Train late base + all 984,321 binding params. Loss = T4 loc CE + layout native margin **only** (no T3 pointer). Independent-item 9:1 language. Parent Phase1G U6000 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`. Seeds 670001–670003.

Watchdog PID 15412. STUDY_COMPLETE 2026-09-11T14:32:45Z.

## U500 results

| seed | ptr (diag) | loc | native | exact | loc rev | nat rev | loc fam | DEV CE | gap | early | class | U500 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 670001 | 65 | 64 | 64 | 0 | 4/64 | 1/64 | 0/16 | 1.24149 | 0.4956 | intact | T8_FAIL_NO_ROUTING | `ccea4b31bd4e98a2014d6c0b9b7f3a95f751c40577703609fffd9285cebfb456` |
| 670002 | 75 | 64 | 71 | 0 | 7/64 | 9/64 | 0/16 | 1.24974 | 0.4867 | intact | T8_FAIL_NO_ROUTING | `246ff64b1c3b8d890713099ebff0cae16ee2e591b78210916d989382d287f276` |
| 670003 | 62 | 64 | 64 | 0 | 2/64 | 2/64 | 0/16 | 1.24585 | 0.4948 | intact | T8_FAIL_NO_ROUTING | `5f4dd814992edbc1e542ceda3609a0b7c4020a9aa99aea4907cbb3b11cbf6706` |

U0 DEV CE 1.2040123894810677. Weight check: **early same**; **late base and binding changed** (bind max |Δ| ≈ 0.008–0.009; late max ≈ 0.011). Valid updates, not a no-update failure. Distinct hashes. `t3_test_loaded: false`. TEST seal `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

## Frozen-gate adjudication
Language **pass 3/3** (CE 1.24–1.25, gap ≤ 0.5). Localization 96/128, loc reversals 48/64, families 8/16, native FC 96, native reversals 48: **all fail**. Localization stuck at **64/128 chance on all three seeds**. Study class **`T8_FAIL_NO_ROUTING`**. TEST not authorized.

## Interpretation vs T4/T5
T4 (frozen base) loc chance. T5 (late-base + pointer + loc + layout) loc chance. T8 (late-base + loc + layout, **no pointer**) loc still chance. T5’s failure was **not** pointer fighting loc. This T4 name-span `BindingLayout` does not produce held-out routing from Phase1G or late-base-plastic mentions at 500 updates. Sidecar weights moved; behavior did not. Do not parent on T8. Do not keep iterating loc/layout coefficient mixes on this layout. Do not open TEST.

The only language-safe retrieval that moved remains **T6 late-base cosine pointer** (85/92/105). Its reversal miss is the v1.0 hole.

## Next
T9: T6 scope/parent/pointer+margin/9:1, but pointer **keys = fact-clause mean-pool** (name through next period token 18), not name-span mean-pool. Tests whether T6 reversals failed because keys were assignment-invariant name identities. Bundle `phase2a_t9_factclause_pointer_v1`.
