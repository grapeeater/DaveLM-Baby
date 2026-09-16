# Baby v0.10 v2R4 mechanism census

This is a data-only refinement of [`V010_V2R4_EMISSION_SOURCE.md`](V010_V2R4_EMISSION_SOURCE.md). It does **not** rewrite the frozen terminal Gate L/C/R adjudication.

Reproduce:

```text
python -B -m src.baby_v010.autopsy_v2r4 --out runs/v2r4_isolation_autopsy
```

Read `mechanism` and `mechanism_headlines` in `AUTOPSY.json`.

## Corrections to earlier readings

Two previous readings from this branch were wrong, or at least overstated.

1. **"2-pair is the first place a binding signal appears."** Same-surface novel 2-pair queried copy is 19/31 versus 1/2 chance. Exact two-sided binomial p = 0.281. Short 2-pair 18/28, p = 0.185. Three-pair 7/21 is exactly 1/3. Four-pair 15/44 versus 1/4, p = 0.222. None of these reject 1/K selection among copied payloads at 0.05. The 2-pair "signal" was an over-read of a noisy slice.

2. **"U16000 is a late jump in queried copy."** Interim v2R4 evals used `probe_limit=16`. Only U16000 scored the full panels. On the matched first-16 novel items, queried copy is 6/16 at U15500 and 5/16 at U16000. Inventory copy on that slice is already 1.0 at U12000. The 41/96 terminal queried count is a full-panel measurement, not a last-checkpoint miracle.

## A–F split after this census

| question | result | locked observation |
|---|---|---|
| **A** identify internally | **supported for continuation** | On train novel, competitor-copy rows still have gold remaining-value ranks all 1 in **51/52**. Queried-copy rows lock 41/41 including sep/EOS. Given the gold first token, she copies the queried span. |
| **B** select/bind queried item | **failed, including 2-pair** | Dominant error is a full competitor span. Queried-copy rates are not above 1/K. Greedy first token is some inventory first token in **95/96** train-novel rows. Value first tokens are unique in 96/96, so this is not prefix collision. |
| **C/D** copy/emit payload | **supported on train surface** | Inventory copy 93/96. TF value = free value. |
| **E** sep/EOS | **two glue problems** | Held-out exact remains OOD seps `{82–85}`. Held-out off-inventory 41/41 emissions contain a **train** separator; 34/41 are `token + train_sep`. All **22/22** unique induction immediate-EOS rows have last context token = that item's own separator. When the last token is not that separator: **0/64** full-induction immediate EOS. |
| **F** new surfaces | **copy recipe itself weakens** | Held-out inventory copy 55/96. At value_length=10: **0/12**. Query-before-body (`keyed_5`) is not worse on train (5/12 vs 36/84). |

## Developmental sequence (matched n=16)

| stage | what appears on the probe slice |
|---|---|
| U7000–U10000 short/primitive | 1-pair primitive keyed copy rises to saturation (~U10000–U11500) |
| U12000 full_foundation | same-surface novel inventory copy hits 16/16 |
| U12000–U16000 | queried copy oscillates ~4–9/16 and does **not** take over |
| U16000 full panel | 41 queried / 52 competitor / 3 off-inventory |

Copy-a-span is acquired. Query-conditioned selection is not. Extra updates after U12000 do not treat B.

## What this does not resolve

These still need the U16000 weights on Fan Diesel:

- query-swap follow vs stuck-on-old (does the first token track the queried key at all?)
- body-reorder first vs last (is the 4-pair first-slot tilt causal?)
- marker-swap vs sep-swap
- value-absent knockout

Do not interpret missing weights as a capability result. v2R5 remains unresolved.
