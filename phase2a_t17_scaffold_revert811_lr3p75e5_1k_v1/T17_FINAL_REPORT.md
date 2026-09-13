# PHASE 2A T17 SCAFFOLD REVERT 8–11 AT U750 — FINAL REPORT

Status: **TERMINAL — study classification `T17_REPRESENTATION_SUCCESS_OUTPUT_FAIL`.**
Per-seed: 760001 `T17_REPRESENTATION_SUCCESS_OUTPUT_FAIL`; 760002 `T17_FAIL_NO_REPRESENTATION`; 760003 `T17_REPRESENTATION_SUCCESS_OUTPUT_FAIL`.
Frozen 2/3 rule: representation-class seeds = 2 ≥ 2. Language held 3/3. Native/exact did not. Do not lower the bar.
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

Watchdog PID 5524 launched 760001–760003 sequentially. Phase A→B (`after_u750_eval`, trainable 21,163,264 / frozen 40,357,121) logged on all three seeds. `STUDY_COMPLETE T17_REPRESENTATION_SUCCESS_OUTPUT_FAIL` at 2026-09-12T10:56:57Z. Independent U1000 adjudication agrees.

## Treatment
Two-phase from Phase1G. Fact-clause pointer, 9:1, lr 3.75e-5, 1000 updates, seeds 760001–760003.
- **Phase A (1–750):** freeze 0–3 + binding; train 4–11 + embed/ln/head (40,849,664).
- **After U750 eval:** copy Phase1G into blocks 8–11, freeze them, rebuild AdamW.
- **Phase B (751–1000):** train 4–7 + embed/ln/head (21,163,264).

Tests whether T16 failed because the 8–11 scaffold was cut at U500 (rev only 35–40). T17 keeps that scaffold through the T14 horizon, then reverts. Do not parent T14/T15/T16/730002. Same QA mix as T16 (no template reweight).

Parent SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`.

## Terminal U1000 (Phase B)

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U1000 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 760001 | **115** | 88 | 0 | **54/64** | **11/16** | **63** | **52** | 1.24780 | 0.4042 | T17_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `9925cf030649dd2ca7944109680d1b1e4eb72c37ce753bd4c47827a37f9537b0` |
| 760002 | 99 | 90 | 0 | **35/64** | **7/16** | 53 | **46** | 1.24667 | 0.4042 | T17_FAIL_NO_REPRESENTATION | `4b3f62b3488c24e229845b686e6a7c4b6718bb69476bb85644d3d2cd4e46324a` |
| 760003 | **112** | 86 | 0 | **48/64** | **11/16** | **60** | **52** | 1.25082 | 0.4069 | T17_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `50f21cbea6db2bf55b2ff9f21f6f2ca8455597efab1f435f9ef6ac3f9348b0af` |

U0 DEV CE **1.2040123894810677** bit-identical 3/3. Binding+early intact 3/3 (Phase B `early_intact` includes frozen 8–11). Exact 0. Native reversals 25/29/24. ACQ16 exact 0 so `acq16_pass` false 3/3. TEST `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

U500 SHA256: 760001 `4f944bf4…4e6e44ff`; 760002 `23c3c1c3…01830b15`; 760003 `d99d7c14…9fc8c32b`.
U750 SHA256: 760001 `71091f48…a8d74970`; 760002 `8c053c63…582a9afae`; 760003 `eccac13e…f045896e`.

U500 (Phase A mid): 760001 81 ptr / **20** rev / 2 fam / CE 1.241 gap 0.431; 760002 96 / **32** / 5 / CE 1.250 gap 0.429; 760003 105 / **41** / 7 / CE 1.253 gap 0.424.

U750 (Phase A endpoint, before revert): 760001 **122** / **59** / 12 / nd 64 / td 58 / CE 1.275 gap 0.492; 760002 102 / **39** / 8 / nd 54 / td 48 / CE 1.277 gap 0.486; 760003 109 / **45** / 10 / nd 58 / td 51 / CE 1.279 gap 0.486.

## Frozen-gate adjudication
**Did language hold?** **Yes, 3/3.** Terminal CE 1.247–1.251 ≤ 1.30 and ≤ U0+0.25. Gap 0.404–0.407 ≤ 0.5. Phase B *improved* language vs U750 (CE 1.275–1.279 → 1.247–1.251; gap 0.486–0.492 → 0.404–0.407), the same language-repair T16 and T14X saw after reverting 8–11.

**Did extending Phase A to the T14 horizon then revert/freeze 8–11 replicate representation ≥2/3?** **Yes. 2/3.**
- 760001: all representation gates pass (115 / 54 / 11 / 63 / 52).
- 760003: all representation gates pass (112 / 48 / 11 / 60 / 52). Phase B crossed the reversal bar (45→48) while holding language.
- 760002: language-safe miss (35 rev, 7 fam, 46 td). Not a parent.

Native never cleared (best FC 90/128, native rev ≤29, exact 0). TEST is **not** authorized.

## Interpretation
The A→B revert executed as frozen. T16’s hypothesis is confirmed: cutting 8–11 at U500 was too early. Keeping them plastic through U750 produces a language-safe representation circuit on 2/3 seeds; reverting 8–11 then repairs the language tax without destroying that circuit on the two hits (760001 rev 59→54 still ≥48; 760003 rev 45→48).

This is the first Phase 2A study to clear the frozen representation 2/3 bar while holding language. T9 was 1/3 luck. T14 was 1/3 and one language-regression seed. T15/T16 were 0/3.

Closed by this result: “T16 failed only because Phase A was too short” is **accepted**. The two-phase T14-horizon scaffold is a trained, replicable representation recipe. Do not relaunch T16 or T17. Do not parent T14/T15/T16/730002. Do not parent 760002. Do not parent 760001/760003 as if they were FULL_SUCCESS or a new Phase1G replacement — they are OUTPUT_FAIL checkpoints from a validated *recipe*.

QA updates remain pointer+margin only. `language_head` has no pointer gradient (T14X). Exact+EOS = 0 on every DEV row, including the 80–83 rows/seed that are already pointer-correct *and* native-forced-choice-correct. That is the remaining v1.0 bottleneck: usable native output, not another scaffold cut.

## Next
Read-only T17X native/generation fail-mode diagnostic on DEV `item_results` plus U1000 greedy traces. No TEST. Then one Baby-native native-output treatment from Phase1G using the T17 recipe, if traces confirm the pointer-only QA objective is the gap. Do not change the QA mix. Do not copy Qwen.
