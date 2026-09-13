# PHASE 2A T12 FACT-CLAUSE POINTER LR 2.5e-5 750 — FINAL REPORT

Status: **TERMINAL — study classification `T12_FAIL_NO_REPRESENTATION` (3/3).**
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

## Treatment
T9 fact-clause late-base pointer + native margin, binding frozen, blocks 0–3 frozen, **9:1** language, **lr 2.5e-5**, **750 updates**. Parent Phase1G U6000 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`. Seeds 710001–710003. Stop seed if DEV CE > 1.30.

Watchdog PID 17560. STUDY_COMPLETE 2026-09-11T22:09:43Z.

## Terminal U750

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U750 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 710001 | 107 | 86 | 0 | 44/64 | 9/16 | 60 | 47 | 1.25289 | 0.4105 | T12_FAIL_NO_REPRESENTATION | `3b090f4d87b60778b0f6e17433355c38891d4ca90cca3b1cfc74acf6c217f8ba` |
| 710002 | 100 | 73 | 0 | 39/64 | 7/16 | 58 | 42 | 1.25128 | 0.4125 | T12_FAIL_NO_REPRESENTATION | `d7186f9d2552290074c2a2dc1c3bb7fee9fa41d1c5f060e36d289fc868ad8dfe` |
| 710003 | 95 | 68 | 0 | 35/64 | 5/16 | 51 | 44 | 1.24246 | 0.4162 | T12_FAIL_NO_REPRESENTATION | `50522323987d0af9637c1b5f5a3dae500f2ff33406ce404f9b54dfd3f61a0837` |

U0 DEV CE 1.2040123894810677 (710001 bit-identical). Early+binding intact 3/3. Exact 0. TEST `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

U500 SHA256: 710001 `71ef887f…da1635be`; 710002 `f431f167…418beb4e`; 710003 `bbb73293…5ebaee74`. U500 still pre-takeoff (ptr 77/69/78, rev 15/8/16) with CE ~1.22.

## Frozen-gate adjudication
**Did lower lr hold CE≤1.30 through U750? Yes.** Language **pass 3/3** (CE 1.242–1.253, gap 0.411–0.416). First language-safe 750-update fact-clause study.
**Did representation replicate ≥2/3 under that constraint? No.** 0/3. Best was 710001 (rev 44/64, td 47/64; both miss by 4 and 1). TEST is **not** authorized. Do not parent T12 (no representation). Do not lower the bar.

## Interpretation
Halving the step size from 5e-5 to 2.5e-5 **solved the T10/T11 language failure** at the 750 horizon and left headroom (CE ~1.25 ≪ 1.30). It also **slowed the reversal takeoff**: T10 U750 at 5e-5 hit rev 48–51 with broken language; T12 U750 is language-safe and short of the reversal/td gates. 710001 is a language-safe near-miss, not a success.

## Next
T13: identical T12 recipe (fact-clause, 9:1, lr 2.5e-5, Phase1G), **1000 updates**, seeds 720001–720003, eval 0/100/300/500/750/1000, stop if DEV CE > 1.30. Tests whether 250 more updates at the language-safe LR close 44→48 reversals without breaking the cap. Bundle `phase2a_t13_factclause_pointer_lr2p5e5_1k_v1`.
