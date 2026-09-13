# PHASE 2A T13 FACT-CLAUSE POINTER LR 2.5e-5 1K — FINAL REPORT

Status: **TERMINAL — study classification `T13_FAIL_NO_REPRESENTATION` (3/3).**
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

## Treatment
T12 recipe: T9 fact-clause late-base pointer + native margin, binding frozen, blocks 0–3 frozen, **9:1** language, **lr 2.5e-5**, **1000 updates**. Parent Phase1G U6000 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`. Seeds 720001–720003. Stop seed if DEV CE > 1.30. Isolates duration: do 250 more language-safe steps close T12's 44→48 reversals?

Watchdog PID 10540. STUDY_COMPLETE 2026-09-12T00:38:27Z.

## Terminal U1000 (frozen study class)

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U1000 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 720001 | 100 | 76 | 0 | 39/64 | 6/16 | 58 | 42 | 1.26655 | 0.4647 | T13_FAIL_NO_REPRESENTATION | `e261b9e788c4b74ac727ce3139079af44fc293b26673e925c6451e681659d8f9` |
| 720002 | 102 | 84 | 0 | 44/64 | 9/16 | 55 | 47 | 1.26255 | 0.4550 | T13_FAIL_NO_REPRESENTATION | `b543d227205f3ad898a35853c69770cbd7d473f3195886f5b777a04e3b0e2dfb` |
| 720003 | 105 | 77 | 0 | 43/64 | 8/16 | 56 | 49 | 1.26470 | 0.4453 | T13_FAIL_NO_REPRESENTATION | `59e1d34c0ffed2a4c84d70b7854ce99555ffb2e332e544551213d6d326305696` |

U0 DEV CE **1.2040123894810677** bit-identical 3/3. Early+binding intact 3/3. Exact 0. TEST `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

## Language-safe U750 (same horizon as T12; not the study class)

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U750 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 720001 | 95 | 74 | 0 | 35/64 | 5/16 | 54 | 41 | 1.23961 | 0.4214 | T13_FAIL_NO_REPRESENTATION | `2bbb797826712812640b126f58b9fe6d0045a30ed0b700c8c7440ebc13e98c98` |
| 720002 | 109 | 80 | 0 | 45/64 | 9/16 | 61 | 48 | 1.25245 | 0.4168 | T13_FAIL_NO_REPRESENTATION | `64880ddf06d604a7c1f41aac2d8666a704b95141b124e75eaa16ad950c12755f` |
| 720003 | 108 | 74 | 0 | **46/64** | 9/16 | 58 | 50 | 1.25095 | 0.4058 | T13_FAIL_NO_REPRESENTATION | `accf34a565fba60895c1df3cddd0d7e458b258066bd52316da442abacb30d135` |

U500 SHA256: 720001 `f617c7c8…172947c7`; 720002 `8e2c4739…0e4dec04`; 720003 `4b06577e…908e6e6c`. U500 still pre-takeoff (ptr 66/77/85, rev 7/13/25).

## Frozen-gate adjudication
**Did CE stay ≤1.30 through U1000? Yes.** Language **pass 3/3** (CE 1.263–1.267, gap 0.445–0.465). First language-safe 1000-update fact-clause study. Stop code null 3/3.
**Did representation replicate ≥2/3 under that constraint? No.** 0/3 at U1000. 0/3 at language-safe U750. Best language-safe point was 720003 U750 (rev 46/64, miss by 2). Extra 250 steps **did not close 44→48**; the two near-misses **regressed** (720002 45→44; 720003 46→43). TEST is **not** authorized. Do not parent T13 (no representation). Do not lower the bar.

## Interpretation
Duration at 2.5e-5 is a closed axis. T12 U750 and T13 U750/U1000 share a language-safe reversal ceiling of ~44–46. Combined with T9+T10 language-safe U500 (1/6), fact-clause late-base retrieval is real but does not replicate at the 2/3 bar when language holds. More of the same 2.5e-5 steps will not graduate. T10's 5e-5 still owns the only U750 representation *metrics* (2/3) and still owns the language death. The remaining isolated knob on this recipe is **step size between 2.5e-5 and 5e-5**.

## Next
T14: identical fact-clause / 9:1 / Phase1G / late-base recipe, **lr 3.75e-5** (arithmetic midpoint), **750 updates**, seeds 730001–730003, eval 0/100/300/500/750, stop if DEV CE > 1.30. Tests whether the midpoint LR reaches rev≥48 while keeping CE≤1.30 and gap≤0.5. Do not extend to 1000 (T13 closed that). Bundle `phase2a_t14_factclause_pointer_lr3p75e5_750_v1`.
