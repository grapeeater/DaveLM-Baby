# PHASE 2A T15 LATE-BASE BLOCKS 4–7 LR 3.75e-5 750 — FINAL REPORT

Status: **TERMINAL — study classification `T15_FAIL_NO_REPRESENTATION`.**
Per-seed: 740001 / 740002 / 740003 all `T15_FAIL_NO_REPRESENTATION`.
Frozen 2/3 rule: representation-class seeds = 0 ≠ ≥2. Do not lower the bar.
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

Watchdog PID 27092 launched 740001–740003 sequentially and wrote `STUDY_COMPLETE T15_FAIL_NO_REPRESENTATION` at 2026-09-12T05:14:15Z after the third FINAL. Independent adjudication of the three U750 evaluations + the frozen study rule agrees.

## Treatment
T9 fact-clause late-base pointer + native margin, **9:1** language, **lr 3.75e-5**, **750 updates**, Phase1G parent. Isolates **scope**: freeze blocks **0–3 + 8–11** and all 984,321 binding params at Phase1G; train blocks **4–7 + embed + final_norm + language_head** (trainable 21,163,264 / frozen 40,357,121). Seeds 740001–740003. Stop seed if DEV CE > 1.30. Tests whether T14X’s post-hoc 8–11 revert is a *trainable* 2/3 recipe when 8–11 never move.

Parent Phase1G U6000 SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`.

## Terminal U750

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U750 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 740001 | 102 | 93 | 0 | **38/64** | 8/16 | 52 | 50 | 1.24607 | 0.3608 | T15_FAIL_NO_REPRESENTATION | `82fb9ef1166265af132162594732d44e36a55958ffed47a89fef926923aecf25` |
| 740002 | 104 | 78 | 0 | **41/64** | 8/16 | 57 | **47** | 1.24188 | 0.3606 | T15_FAIL_NO_REPRESENTATION | `a78e42d6bd5f4d2d5fbe518d16c485e0cc97f1c18612427f1e2c744878b68cf3` |
| 740003 | 103 | 76 | 0 | **43/64** | **7/16** | 60 | **43** | 1.23952 | 0.3609 | T15_FAIL_NO_REPRESENTATION | `b01dbd07af8ee3db16837ff0a4ba39378521d01205b8f9e8c32929f733ebd4a5` |

U0 DEV CE **1.2040123894810677** bit-identical 3/3. Early+binding intact 3/3 (`early_intact` here covers frozen blocks 0–3 **and** 8–11). Exact 0. TEST `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

U500 SHA256: 740001 `340b94e3c6f8cdd9d6864aa23eb9dde6abea7e8b5e8127ac598dd568b7de219a`; 740002 `c76ecde6238c0fde70b5a8c29756538dac5d3eaa045533816d943d0685e88531`; 740003 `9d58e908f769259001d82fddd088e8ddc59c4d8d5c436db187f8f2ea62f014ae`.
U500: 740001 102 ptr / **40** rev / 7 fam / CE 1.229 gap 0.331; 740002 87/26/4 CE 1.221 gap 0.330; 740003 88/26/3 CE 1.229 gap 0.331.

## Frozen-gate adjudication
**Did freezing 8–11 hold language?** **Yes, 3/3.** CE 1.240–1.246 (well under 1.30 and U0+0.25). Gap 0.361 3/3 (well under 0.5). Language is *better* than T14’s language-safe seeds (~1.276 / 0.47) and T14’s gap-fail seed (0.515). Lots of language headroom.

**Did representation replicate ≥2/3 under the language constraint?** **No. 0/3.** Every seed misses reversals (38 / 41 / 43 vs gate 48). 740002 also misses template-disjoint by 1 (47). 740003 also misses families (7) and template-disjoint (43). Pointer ≥96 3/3; name-disjoint ≥48 3/3. Native never cleared (best 93/128, exact 0). TEST is **not** authorized.

Reversals are *worse* than T14’s language-safe pair (46 and 49). The best T15 reversal (43) equals T14’s language-fail seed, not T14’s representation hit.

Do not parent T15. Do not parent any T15 checkpoint.

## Interpretation
T14X showed that *after joint 4–11 training*, reverting blocks 8–11 to Phase1G on 730002 improved CE 1.276→1.234, gap 0.469→0.367, **and** rev 49→51. That patch is still not a trained 2/3 recipe.

T15 tested the prospective version: freeze 8–11 at Phase1G from the start and train only 4–7 (+ embed / ln / head). The language half of T14X **replicated**. The representation half **did not**. Training 4–7 without 8–11 plasticity is **not** the same as jointly training 4–11 and then restoring 8–11.

Implication: blocks 8–11 appear to need to *move during training* for the 4–7 reversal pathway to form, even if their T14-trained endpoint is language-costly. Frozen Phase1G 8–11 preserve language and do not carry the T14 reversal circuit. Do not treat the T14X revert as a parentable treatment.

Closed by this result: “freeze 8–11, train 4–7” at T14’s recipe (fact-clause / 9:1 / 3.75e-5 / 750 / Phase1G). Do not relaunch T15. Do not resume the scalar LR sweep. Do not parent T14, 730002, or the T14X patched weights.

## Next
Do **not** launch T16 from this finalize. Authoritative next mission step is the read-only Qwen specimen study at `C:\AI\QwenLab`, then return here to synthesize T15 + T14X + T3–T14 + Qwen before choosing one Baby-native action. Baby evidence outranks Qwen analogy.
