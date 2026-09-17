# P10 aux-only overwrite: REGRESSION (global gate-on)

Status: **REGRESSION**. Hard-stop on language DEV CE at 400. Removing CE
gradients into overwrite **did** open the gate (0.018→0.93) and raise L0
cosine (0.042→0.886), confirming P9's stall was CE fight. The write was
**not** gen-local: a content-based gate generalized across the residual
stream, collapsing induction/keyed/rest_lock. Long-gap hits stayed 74/215.
Authoritative Baby unchanged: v2R4 U16000
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
P2 not launched. No TEST. Not promotable.

Protocol:
[`design/V010_SELECTION_REPAIR_P10_AUX_ONLY.md`](../design/V010_SELECTION_REPAIR_P10_AUX_ONLY.md)
Adjudication: `runs/selection_p10/ADJUDICATION_240001_400.json`

## Primary (gap ≥ 13, n=215)

| | parent | treatment 400 | control 400 |
|---|---:|---:|---:|
| hits | 74 | 74 | 74 |
| L0 query cosine | 0.042 | **0.886** | 0.042 |
| overwrite-head mass | 0.003 | 0.371 | 0.003 |
| median gate | 0.018 | **0.934** | 0.018 |
| induction first-top1 | 0.297 | **0.047** | 0.297 |
| language DEV CE | 1.243 | **4.760** | 1.243 |

Treatment−control excess **0**, CI **[−0.026, +0.026]**.

## What this says

H1 confirmed: CE-through-overwrite pinned P9's gate at ~0.29. Aux-only
lets `(1-g)^2` win. H0 on selection: cosine≈0.89 is **not** D3b replace.
The pointer is source-independent and the gate is content-based, so
opening g globally rewrites every token toward the same slot mixture.
That inflates post-overwrite cosine and destroys Baby. Hits did not move.

## Licensed next

P11: same aux-only slot overwrite, **hard-mask the write to the
generation index only** (`len(input)-1`). Language / non-gen residuals
stay identity. Not P10 λ. Not unfreezing.
