# PHASE 2A T14 FACT-CLAUSE POINTER LR 3.75e-5 750 — FINAL REPORT

Status: **TERMINAL — study classification `T14_FAIL_NO_REPRESENTATION`.**
Per-seed: 730001 `FAIL_NO_REPRESENTATION`, 730002 `REPRESENTATION_SUCCESS_OUTPUT_FAIL`, 730003 `LANGUAGE_REGRESSION`.
Frozen 2/3 rule: 1 representation-class seed ≠ ≥2. Do not lower the bar.
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

Watchdog PID 14592 launched 730001–730003 sequentially. All three `FINAL_STATUS.json` present at U750. Watchdog log last line `WATCHDOG_LAUNCH seed=730003`; process 14592 remained alive after the third FINAL without writing `STUDY_COMPLETE` (hung post-terminal). Study class is taken from the three frozen FINALs + the published study rule, not from a missing watchdog stamp.

## Treatment
T9 fact-clause late-base pointer + native margin, binding frozen, blocks 0–3 frozen, **9:1** language, **lr 3.75e-5**, **750 updates**. Parent Phase1G U6000 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`. Seeds 730001–730003. Stop seed if DEV CE > 1.30. Isolates midpoint LR between language-safe 2.5e-5 and language-breaking 5e-5.

## Terminal U750

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U750 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 730001 | 110 | 87 | 0 | 46/64 | 11/16 | 62 | 48 | 1.27568 | 0.4812 | T14_FAIL_NO_REPRESENTATION | `08c88bed576b5055c2baa21d7072a6d4fe075136bdcbcd3cf6131e2463969368` |
| 730002 | 112 | 93 | 0 | **49/64** | 10/16 | 60 | 52 | 1.27593 | 0.4687 | T14_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `f1c81818cd9c5a2de40f8303c1826022e7e4cf1f102a8a31a89c70b94ead246b` |
| 730003 | 107 | 73 | 0 | 43/64 | 10/16 | 60 | 47 | 1.27559 | **0.5153** | T14_LANGUAGE_REGRESSION | `f8312207ef18bb4aac6c9ef3d7ca66a87396a57c893f2aa149d23e31506190a7` |

U0 DEV CE **1.2040123894810677** bit-identical 3/3. Early+binding intact 3/3. Exact 0. TEST `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

U500 SHA256: 730001 `3fa689eb…62175429`; 730002 `9ad4935d…3641475c`; 730003 `72d1df4b…3895c89f`.
U500: 730001 87/25/3 CE 1.251 gap 0.424; 730002 **109/45/11** CE 1.254 gap 0.409; 730003 64/7/0 CE 1.231 gap 0.445.

## Frozen-gate adjudication
**Did 3.75e-5 hold CE≤1.30 through U750?** CE yes 3/3 (1.276). Gap≤0.5 only **2/3** (730003 gap 0.515). Language is not a 3/3 pass.
**Did representation replicate ≥2/3 under the language constraint?** **No. 1/3.** Only 730002 is language-safe and representation-complete (rev 49). 730001 is a language-safe near-miss (rev 46). 730003 is language-fail (gap) and miss (rev 43). Native never cleared (best 93/128, exact 0). TEST is **not** authorized. Do not parent T14, including 730002.

## Interpretation
Midpoint LR sits on the knife-edge the sweep predicted: one language-safe representation hit, one language-safe near-miss at the familiar 46-rev ceiling, one gap breach. This is a scalar tradeoff, not a 2/3 recipe. Further LR/duration/mix tweaks on the same late-base fact-clause objective are not the next information. Relational gains live in trainable late base (blocks 4–11 + embed/ln/head), not the frozen sidecar.

## Next
Read-only late-base failure decomposition (`phase2a_t14x_latebase_decomp_v1`). No T15 training until that evidence localizes whether reversal gain and language cost occupy different late-base modules. Do not parent 730002/730001.
