# P3 early residual bind: REGRESSION (futility)

Status: **REGRESSION**. Futility at 400. Selection did not move. L0 query-gen
cosine did not install the D3b splice. Authoritative Baby unchanged: v2R4
U16000 `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
P2 not launched. No TEST.

Protocol:
[`design/V010_SELECTION_REPAIR_P3_EARLY_BIND.md`](../design/V010_SELECTION_REPAIR_P3_EARLY_BIND.md)
Adjudication: `runs/selection_p3/ADJUDICATION_170001_400.json`

## Primary (gap ≥ 13, n=215)

| | parent | treatment 400 | control 400 |
|---|---:|---:|---:|
| hits | 74 | 73 | 74 |
| excess | 0 | −0.005 | 0 |
| L0 query-gen cosine | 0.040 | 0.052 | 0.041 |
| L0–2 query-track | 0.140 | 0.144 | — |

Treatment−control excess **−0.005**, bootstrap 95% CI **[−0.014, 0.0]**.

Futility fired: excess gain < +0.02 and L0 cosine gain +0.012 < +0.02.
`L_bind` did not fall by half (first50 ~1.2, last ~0.73).

## Retention

`primitive_induction` first-top1 0.297 → **0.203** treatment, **0.188**
control. Diet, not the aux. Language CE held. `rest_lock` / primitive keyed /
negative controls held. Official verdict REGRESSION because the parent drop
exceeds 0.05.

## What this says

D3b proved a **replace** of `h0[gen]` with `h0[query]` is sufficient.
P3's competitor-contrast InfoNCE at λ=0.25 did not create that replace
(cosine stayed ~0.04). Ranking query above competitor keys at L0 is a
different, weaker demand than the splice.

## What this does not license

Raising P3 λ, resuming to 800, P2, P1 λ. Next: a **direct** L0 copy
(`1 - cosine(h0[gen], sg(h0[query]))`) with more induction in **both** arms
so the known 400-update induction drop is not repeated as the only change.
