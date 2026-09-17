# P7 frozen-Baby overwrite-only: NULL (futility at 200)

Status: **NULL**. Futility at 200. Freeze-backbone **held induction**.
Bilinear L0 pointer still did not find the query (mass 0.004→0.002).
Authoritative Baby unchanged: v2R4 U16000
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
P2 not launched. No TEST.

Protocol:
[`design/V010_SELECTION_REPAIR_P7_OVERWRITE_ONLY.md`](../design/V010_SELECTION_REPAIR_P7_OVERWRITE_ONLY.md)
Adjudication: `runs/selection_p7/ADJUDICATION_210001_200.json`

## Primary (gap ≥ 13, n=215)

| | parent | treatment 200 | control 200 |
|---|---:|---:|---:|
| hits | 74 | 74 | 75 |
| overwrite-head mass | 0.0043 | 0.0016 | ~0 |
| induction first-top1 | 0.297 | **0.297** | **0.312** |

Treatment−control excess **−0.005**, CI **[−0.015, 0.0]**. Language CE held.

## What this says

Freezing Baby weights removes the P3–P6 induction regression. The remaining
failure is the **pointer**, not retention and not the D3b write (write is
off while gate≈0.02 and mass is not on the query).

Gen is a filler token ~50 steps after the query. Bilinear `Q(h0[gen])·K(h0[t])`
has to find that slot from a query-invariant gen residual (D1b). 200 steps at
1e-3 did not.

## Licensed next

A **local slot scorer** on `(h0[t], h0[t+1])` (unpaired key vs key+value),
same frozen Baby and identity overwrite. Not P7 λ. Not unfreezing Baby.
