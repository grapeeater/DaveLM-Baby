# P4 direct L0 copy: REGRESSION (futility)

Status: **REGRESSION**. Futility at 400. Direct `1 - cosine(h0[gen],
sg(h0[query]))` at λ=1.0 did not install the D3b splice. Authoritative Baby
unchanged: v2R4 U16000
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
P2 not launched. No TEST.

Protocol:
[`design/V010_SELECTION_REPAIR_P4_EARLY_COPY.md`](../design/V010_SELECTION_REPAIR_P4_EARLY_COPY.md)
Adjudication: `runs/selection_p4/ADJUDICATION_180001_400.json`

## Primary (gap ≥ 13, n=215)

| | parent | treatment 400 | control 400 |
|---|---:|---:|---:|
| hits | 74 | 75 | 74 |
| excess | 0 | +0.005 | 0 |
| L0 query-gen cosine | 0.040 | 0.067 | 0.044 |

Treatment−control excess **+0.005**, bootstrap 95% CI **[0.0, +0.014]**.
The lower bound is not > 0.

Futility fired: excess gain +0.005 < +0.02 and L0 cosine gain +0.027 < +0.05.
Train `L_copy` first50 **0.962** → last50 **0.933** (not halved). Runtime
logs: first batches had 6–9 copy specs, mean cosine ~0.03, loss ~0.96. The
aux was applied; the optimizer did not implement the replace.

## Retention

`primitive_induction` first-top1 0.297 → **0.188** treatment, **0.219**
control. More induction in both arms (5/batch vs P3's 3) did not prevent the
drop. Language CE held (1.243 → 1.248 / 1.247). `primitive_keyed` first-top1
stayed 1.0. `short_keyed` free exact 0.844 → 0.812 / 0.797 (within 0.05).
Negative controls held. Official verdict REGRESSION because the parent
induction drop exceeds 0.05.

L0–2 query-track stayed **0.144** (parent 0.140). Copy loss did not create
an L0 pointer.

## What this says

D3b proved a **replace** of `h0[gen]` with `h0[query]` is sufficient at
eval. P3's InfoNCE ranking and P4's direct cosine copy are both **weight
updates** that share L0 across positions. Neither produced the replace
(cosine 0.040 → 0.052 P3, → 0.067 P4). Matching two residuals after the
fact is a different demand than an attention OV copy at generation time.

## What this does not license

Raising P4 λ, resuming to 800, P3 λ, P2, P1 λ. Next: a **read-only L0
attention patch** (D4). If one-hot query attention at block 0 writes the
query residual and lifts gold, an L0-only pointer is licensed. If it does
not write, the OV path is dead and the next repair is architectural, not
another residual-matching loss.
