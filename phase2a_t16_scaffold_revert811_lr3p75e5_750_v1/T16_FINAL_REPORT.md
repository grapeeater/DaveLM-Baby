# PHASE 2A T16 SCAFFOLD REVERT 8–11 — FINAL REPORT

Status: **TERMINAL — study classification `T16_FAIL_NO_REPRESENTATION`.**
Per-seed: 750001 / 750002 / 750003 all `T16_FAIL_NO_REPRESENTATION`.
Frozen 2/3 rule: representation-class seeds = 0 ≠ ≥2. Do not lower the bar.
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

Watchdog PID 21376 launched 750001–750003 sequentially. Phase A→B (`after_u500_eval`, trainable 21,163,264) logged on all three seeds. `STUDY_COMPLETE T16_FAIL_NO_REPRESENTATION` at 2026-09-12T08:06:51Z. Independent U750 adjudication agrees.

## Treatment
Two-phase from Phase1G. Fact-clause pointer, 9:1, lr 3.75e-5, 750 updates, seeds 750001–750003.
- **Phase A (1–500):** freeze 0–3 + binding; train 4–11 + embed/ln/head (40,849,664).
- **After U500 eval:** copy Phase1G into blocks 8–11, freeze them, rebuild AdamW.
- **Phase B (501–750):** train 4–7 + embed/ln/head (21,163,264).

Tests whether 8–11 plasticity is a *training scaffold* that can be removed once 4–7 have had 500 joint updates. Do not parent T14/T15/730002.

Parent SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`.

## Terminal U750 (Phase B)

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U750 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 750001 | 100 | 89 | 0 | **43/64** | 8/16 | 55 | **45** | 1.24125 | 0.3676 | T16_FAIL_NO_REPRESENTATION | `7aae8df4a540db41800eeaef051422587ecbd7b217c8aa79d8b3e257afe453e9` |
| 750002 | 109 | 73 | 0 | **46/64** | 10/16 | 63 | **46** | 1.24134 | 0.3710 | T16_FAIL_NO_REPRESENTATION | `03517a7dc2b5b9a0dc1bd06deb01029c6e6a79cca2be42660aa8a7a0b016291d` |
| 750003 | 100 | 87 | 0 | **39/64** | 8/16 | 53 | **47** | 1.24473 | 0.3693 | T16_FAIL_NO_REPRESENTATION | `3b01b599d85bb3141bb5933061eade2e2b50651bb58307a5e609fe4f8b6737d6` |

U0 DEV CE **1.2040123894810677** bit-identical 3/3. Binding+early intact 3/3 (Phase B `early_intact` includes frozen 8–11). Exact 0. TEST `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

U500 (Phase A endpoint, before revert) SHA256: 750001 `eebe6a9a…2f467a17`; 750002 `64871c4c…24ee304a`; 750003 `1ebd1df5…40c1ba24`.
U500: 750001 98 ptr / **35** rev / 4 fam / CE 1.252 gap 0.418; 750002 101/ **40** /6 / CE 1.249 gap 0.427; 750003 97/ **36** /6 / CE 1.254 gap 0.418.

## Frozen-gate adjudication
**Did language hold?** **Yes, 3/3.** CE 1.241–1.245. Gap 0.368–0.371. Better than T14’s language-safe seeds (~1.276 / 0.47). Phase B *improved* language vs U500 (CE 1.249–1.254 → 1.241–1.245; gap 0.418–0.427 → 0.368–0.371).

**Did representation replicate ≥2/3?** **No. 0/3.** Every seed misses reversals (43 / 46 / 39). All three also miss template-disjoint (45 / 46 / 47). Pointer and name-disjoint pass 3/3; families pass 3/3. Native never cleared (best 89/128, exact 0). TEST is **not** authorized.

Phase B moved reversals **up** from U500 (35→43, 40→46, 36→39) but did not close 46→48. 750002 is the familiar language-safe near-miss ceiling. Do not parent T16.

## Interpretation
The A→B revert executed as frozen. Language half of T14X **replicated** under a trained two-phase. Representation half **did not**. Cutting the 8–11 scaffold at U500 is too early: T14’s only representation hit appeared at U750 (730002 U500 was 45 rev, U750 was 49). T16 Phase A ended at 35–40 rev.

Closed by this result: “500 joint + 250 revert/freeze 8–11” at this recipe. Do not relaunch T16. Do not parent T14/T15/T16/730002. Do not resume the scalar LR sweep.

## Next
Read-only DEV reversal-fail overlap (T16X) across T16 / T14 / T15 item_results — no TEST — then one Baby-native action. Do not launch another two-phase until that cheap diagnostic says whether the remaining misses are a stable hard set or an early-cut scaffold.
