# P9 slot overwrite through gate-on: MECHANISM SUPPORTED, DOSE INSUFFICIENT

Status: **MECHANISM SUPPORTED, DOSE INSUFFICIENT**. First lineage result
where overwrite-head mass, L0 query cosine, and long-gap hits all move
versus parent **and** matched λ=0, with bootstrap CI excluding zero, while
`primitive_induction` holds. Frozen SUCCESS bars are not met. Authoritative
Baby unchanged: v2R4 U16000
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
P2 not launched. No TEST. Not promotable.

Protocol:
[`design/V010_SELECTION_REPAIR_P9_SLOT_GATEON.md`](../design/V010_SELECTION_REPAIR_P9_SLOT_GATEON.md)
Adjudication: `runs/selection_p9/ADJUDICATION_230001_800.json`

## Primary (gap ≥ 13, n=215)

| | parent | treatment 800 | control 800 |
|---|---:|---:|---:|
| hits | 74 | **84** (peak 87 at 600) | 74 |
| excess | 0 | **+0.047** | 0 |
| L0 query cosine | 0.042 | **0.228** | 0.044 |
| overwrite-head mass | 0.003 | **0.568** | 0.00007 |
| median gate[gen] | 0.018 | **0.289** | 0.026 |
| induction first-top1 | 0.297 | **0.297** | 0.297 |
| language DEV CE | 1.244 | 1.265 | 1.245 |

Treatment−control excess **+0.047**, bootstrap 95% CI **[+0.023, +0.073]**.
SUCCESS required excess ≥ 0.10 and T−C ≥ 0.07.

## Timeline (treatment)

| step | hits | cosine | mass | gate | note |
|---:|---:|---:|---:|---:|---|
| 0 | 74 | 0.042 | 0.003 | 0.018 | init |
| 200 | 74 | 0.047 | 0.154 | 0.024 | pointer on, gate off; passed mass≥0.05 |
| 400 | 84 | 0.175 | 0.327 | 0.271 | first gate-on window |
| 600 | 87 | 0.199 | 0.503 | 0.310 | peak hits |
| 800 | 84 | 0.228 | 0.568 | 0.289 | cosine still climbing; gate stalled |

Control stayed at 74 / cosine 0.044 / mass ~0 at every 200-step eval.

## Retention (treatment vs parent)

- primitive_induction first_top1 0.297 → **0.297**
- primitive_keyed first_top1 1.0 → 1.0
- short_keyed free_exact 0.828 → 0.797 (drop 0.031 < 0.05)
- rest_lock 0.977 → 0.958 (drop 0.019 < 0.05)
- value_absent / broken_context / broken_order all pass
- short-gap (0–1) hits 73 → 61 (not a frozen panel; watch in P10)

Query-dependent spread median 0.235 → **0.753**. same_first_token_rate
0.833 → 0.778 (still high).

## What this says

P1 attended at L10 and did not select. P9 writes query mass into the L0
generation residual and **does** select, but only +10/215. Gate[gen]
stalls near 0.29 against CE, so overwrite stays a 30% blend
(`cosine 0.23`, far from D3b replace). The remaining bottleneck is
**incomplete replace**, not "the pointer cannot learn."

## Licensed next

P10: same slot overwrite, Baby frozen, **CE does not update overwrite**
(pointer+gate aux only). Tests whether CE-through-overwrite, including
language-only steps, is what pins the gate. Not P9 λ. Not unfreezing.
