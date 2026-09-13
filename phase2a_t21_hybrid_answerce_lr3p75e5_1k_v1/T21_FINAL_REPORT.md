# PHASE 2A T21 HYBRID ANSWER CE — FINAL REPORT

Status: **TERMINAL — study classification `T21_REPRESENTATION_SUCCESS_OUTPUT_FAIL`.**
Per-seed: 800001 / 800002 / 800003 all `T21_REPRESENTATION_SUCCESS_OUTPUT_FAIL`.
Frozen 2/3 rule: representation-class seeds = 3 ≥ 2. Language held 3/3. Exact/native did not. Do not lower the bar.
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

Watchdog 16468 launched 800001; host `STATUS.json` lock killed the runner at U350 after `rolling_restart.pt` wrote. Successor watchdog 29316 resumed 800001 then ran 800002–800003. Phase A→B (`after_u750_eval`, trainable 21,163,264) logged on all three. `STUDY_COMPLETE T21_REPRESENTATION_SUCCESS_OUTPUT_FAIL` at 2026-09-13T00:42:34Z. Independent U1000 adjudication agrees.

## Treatment
T17/T18 two-phase from Phase1G, uniform **mean** `λ_ans=1.0` over the answer span: **first answer token CE through the base**, **suffix+period+EOS CE stop-grad** through `language_head`. No T19 suffix weight 3.0. Fact-clause, 9:1, lr 3.75e-5, seeds 800001–800003.
Earned by T20/T20X: head-only CE recovered representation 2/3 and collapsed exact to 9/10/13; first-token BOS fell to 42/34/37.

Parent SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`.

## Terminal U1000 (Phase B)

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U1000 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 800001 | **114** | **97** | **27** | **53/64** | **11/16** | **61** | **53** | 1.28584 | 0.4265 | T21_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `281dec09af1475fb37c960dff51c7e60e15b9d348cf0ee618b2f27d62df8f97f` |
| 800002 | **108** | 83 | **29** | **49/64** | **9/16** | **60** | **48** | 1.28396 | 0.4220 | T21_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `116325bb94e772f77d00715323cacdded5f104d3ed1aa4c6b784cd55921922a4` |
| 800003 | **113** | 89 | **26** | **49/64** | **9/16** | **60** | **53** | 1.27897 | 0.4176 | T21_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `5d3828029bf065fb2704c73625dbae14453a883d896a69ada739379dc266cf11` |

U0 DEV CE **1.2040123894810677** bit-identical 3/3. Binding+early intact 3/3. Native reversals 33/25/25. ACQ16 exact+native pass **1/3** (800002 only). TEST `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

U750 SHA256: 800001 `0b30b6cc…d5cd84d8`; 800002 `7edd0d6b…80632f19`; 800003 `9abc33ed…bf0e7265`.
U750: 800001 111/**51**/10/exact 15/CE 1.280 gap 0.515; 800002 102/**46**/?/exact 7/CE 1.278 gap 0.508; 800003 111/**47**/?/exact 9/CE 1.274 gap 0.504.
Phase B repaired 800002 reversals 46→49 and 800003 47→49.

## Frozen-gate adjudication
**Did language hold?** **Yes, 3/3.** CE 1.279–1.286 ≤ 1.30 and ≤ U0+0.25. Gap 0.418–0.427 ≤ 0.5. Phase B repaired U750 gaps (0.504–0.515).

**Did representation hold ≥2/3?** **Yes. 3/3.** First time the two-phase recipe cleared representation on all three seeds.

**Did first-token-through-base + suffix-stop-grad recover T18 exact?** **No.** Exact **27 / 29 / 26** vs T18 **36 / 36 / 40** and T20 **9 / 10 / 13**. Partial recovery, not T18. Native FC 97 / 83 / 89 — only 800001 clears 96; none clear native reversals 48 or exact 80. TEST is **not** authorized.

## Interpretation
**Observed in Baby.** Through-base CE on the first answer token is necessary and sufficient to restore a generation channel T20 lost, and it does not repeat T19’s representation tax (here representation is **3/3**). It is not sufficient to match T18’s exact 36–40. Dose-response on exact: T20 head-only 9–13 → T21 first-token-through-base 26–29 → T18 full-span-through-base 36–40.

Closed by this result: “first-token-through-base + suffix-stop-grad recovers T18 exact while holding representation 2/3.” Do not relaunch T21. Do not parent T21/T20/T19/T18 OUTPUT_FAIL. Do not raise suffix weight. Do not retokenize.

## Next
T21X: read-only greedy fail-mode. Did first-token BOS recover toward T18X 76–84, and is leftover `other_then_eos` or T18X diverge?
