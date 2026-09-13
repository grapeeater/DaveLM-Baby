# PHASE 2A T11 FACT-CLAUSE POINTER 4:1 750 — FINAL REPORT

Status: **TERMINAL — study classification `T11_LANGUAGE_REGRESSION` (3/3).**
T3 TEST was **not** loaded. T2-EVAL-TEST / FINAL / sacred untouched.

## Treatment
T9 fact-clause late-base pointer + native margin, binding frozen, blocks 0–3 frozen, **4:1 QA:language** (lang every 5th), **750 updates**. Parent Phase1G U6000 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`. Seeds 700001–700003. Stop seed if DEV CE > 1.30.

Watchdog PID 5896. STUDY_COMPLETE 2026-09-11T20:11:43Z.

## Terminal U750 (frozen study class)

| seed | ptr | native | exact | ptr rev | ptr fam | nd | td | DEV CE | gap | class | U750 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 700001 | 107 | 85 | 0 | 49/64 | 10/16 | 58 | 49 | 1.33923 | 0.8080 | T11_LANGUAGE_REGRESSION | `47b99ecbd4b2ff19446628ea08735a049dec50757f4e3baa89fe714b8965ec18` |
| 700002 | 106 | 85 | 0 | 45/64 | 9/16 | 60 | 46 | 1.32721 | 0.8155 | T11_LANGUAGE_REGRESSION | `354a40b27f0a3bc20881061a5d1a0d0fc46e63029e78fd6e02a37038755dc892` |
| 700003 | 112 | 76 | 0 | 48/64 | 10/16 | 61 | 51 | 1.34779 | 0.8358 | T11_LANGUAGE_REGRESSION | `e73e17af7e6f077d520bbe75314377aa598f9c78e7dfc042159a7b3ef4dbec32` |

U0 DEV CE 1.2040123894810677 (700001 bit-identical). Early+binding intact 3/3. Exact 0. TEST `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`).

## Language-safe? U500 already no

| seed | ptr | ptr rev | fam | nd | td | DEV CE | gap | class | U500 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| 700001 | 106 | 44 | 9 | 58 | 48 | 1.29901 | 0.6707 | T11_LANGUAGE_REGRESSION | `b8363465571ef947c5a977c651f31e8f4f42f8b4e1ca9118538940a19eef0724` |
| 700002 | 100 | 38 | 8 | 58 | 42 | 1.29621 | 0.6712 | T11_LANGUAGE_REGRESSION | `650e55bc49608738b4bc34b76a400c5a22f326481cf084fe61175c3cc7ada8d7` |
| 700003 | 95 | 34 | 5 | — | — | 1.29120 | 0.6915 | T11_LANGUAGE_REGRESSION | `aff85867f2bb693819a54d7845a8461abeec5e49fcbcbd05724d2ccb52418e53` |

Gap already > 0.5 by U300 on 700001 (CE 1.242, gap 0.526). Train CE fell to ~0.51–0.53 by U750 (T10 U750 train CE was ~0.73): more passes over the same 320-window rehearsal overfit language train.

## Frozen-gate adjudication
**Did 4:1 hold CE≤1.30 through U750? No.** All three CE 1.327–1.348, stop_code `language_ce_gt_1_30`. Gap failed earlier than T10 9:1.
**Did representation replicate ≥2/3 under that constraint? No.** Language-safe representation is 0/3. U750 representation *metrics* hit on 700001 and 700003 (2/3) but the language gate failed; do not credit. TEST is **not** authorized. Do not parent T11 U500/U750.

## Interpretation
Adding language CE on the same `starts_mod_320` cycle is a dead end: it widened the gap and raised DEV CE versus T10 9:1 at the same horizon. Fact-clause retrieval still appears around U750 when language is allowed to break. The remaining hole is language-safe representation at the takeoff horizon. Do not add more language mix. Do not parent 680001 or T10/T11 late checkpoints.

## Next
T12: same fact-clause late-base pointer, Phase1G parent, **9:1** (not 4:1), **750 updates**, **lr 2.5e-5** (half of 5e-5), seeds 710001–710003, stop if DEV CE > 1.30. Isolates step size at the representation takeoff horizon. Bundle `phase2a_t12_factclause_pointer_lr2p5e5_750_v1`.
