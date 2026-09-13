# PHASE 2A T5 LATE-BASE + BINDING — FINAL REPORT

Status: **TERMINAL — study classification `T5_FAIL_NO_ROUTING` (2/3 FAIL_NO_ROUTING; 1/3 LANGUAGE_REGRESSION on the gap gate).**
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

## Treatment
Freeze `base_model.blocks.0–3` (19,686,400 params). Train remaining base + all 984,321 binding params (41,833,985 trainable). Parent = untouched Phase1G U6000 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`. Loss = T3 pointer + T2 layout margin + T4 localization CE, 9:1 language rehearsal, 500 updates, seeds 640001–640003. T4 name-span `BindingLayout`.

Watchdog PID 19920 (detached; Start-Process parent dead). STUDY_COMPLETE 2026-09-11T10:41:15Z.

## U500 results

| seed | ptr | loc | native | exact | loc rev | nat rev | loc fam | DEV CE | gap | early intact | class | U500 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 640001 | 63 | 64 | 61 | 0 | 5/64 | 0/64 | 0/16 | 1.24238 | 0.5032 | true | T5_LANGUAGE_REGRESSION | `b6c8f489abc942d4ff67b32abb5d5141662d27355c6dd515477e246b0a6ff3ee` |
| 640002 | 61 | 64 | 60 | 0 | 15/64 | 4/64 | 0/16 | 1.25158 | 0.4914 | true | T5_FAIL_NO_ROUTING | `1f802c026f658dcfffac0f5ac0d730c7979de7e60b89b477e0ceac48930b8938` |
| 640003 | 67 | 64 | 59 | 0 | 7/64 | 1/64 | 0/16 | 1.24547 | 0.4893 | true | T5_FAIL_NO_ROUTING | `308b156daba957dd377bdc0eab497ffc7dd2891f42e0d3b997331a001f41173d` |

U0 reproduction: DEV CE 1.2040123894810677, loc 64/128, native 63/128, pointer 64/128. `t3_test_loaded: false` on all evals.

Weight check vs parent: **early blocks identical**; **late base and binding changed** (bind max |Δ| ≈ 0.0086–0.0094; late max |Δ| ≈ 0.011). Valid training, not a no-update failure. Three distinct U500 hashes.

## Frozen-gate adjudication
Language CE ≤ 1.30 held on all three seeds (1.24–1.25). Gap ≤ 0.5 failed only on 640001 (0.5032), which is why that seed is `T5_LANGUAGE_REGRESSION` despite CE still under 1.30. Localization 96/128, loc reversals 48/64, complete families 8/16, native FC 96/128, native reversals 48/64: **all fail on all seeds** (localization stuck at 64/128 chance; pointer 61–67, unlike T3’s 97–102). Study rule: 2/3 `T5_FAIL_NO_ROUTING` and not ≥2 language/routing successes → **`T5_FAIL_NO_ROUTING`**. TEST scoring is not authorized. T3 TEST seal remains `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

## Interpretation vs T3 and T4
T3 (full base, binding frozen) moved pointer retrieval and nicked CE past 1.30. T4 (binding only) preserved language and left routing at chance. T5 jointly updated late base + binding under this NL layout and mixed losses: language mostly held, **routing did not move**, and **T3-style pointer retrieval also did not move**. That confounds two hypotheses (early-block plasticity vs mixed loc/layout loss). Do not parent on T5 U500. Do not reopen T3 TEST. Do not rerun T5 with coefficient tweaks.

## Next
T6 isolates the freeze: T3 pointer + T2 native margin (no layout/loc), binding frozen, **blocks 0–3 frozen**, 9:1 language, Phase1G parent, same T3 train/DEV. Question: did T3’s 97–102/128 retrieval require early-block updates? Bundle `phase2a_t6_latebase_pointer_v1`.
