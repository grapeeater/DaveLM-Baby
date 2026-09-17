# P8 local slot overwrite: NULL (futility at 200)

Status: **NULL**. Futility at 200 because overwrite-head mass **0.133** missed
the frozen 0.20 bar. The slot scorer **did move** vs control (0.133 vs
0.0006). Induction held. Authoritative Baby unchanged: v2R4 U16000
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
P2 not launched. No TEST.

Protocol:
[`design/V010_SELECTION_REPAIR_P8_SLOT_OVERWRITE.md`](../design/V010_SELECTION_REPAIR_P8_SLOT_OVERWRITE.md)
Adjudication: `runs/selection_p8/ADJUDICATION_220001_200.json`

## Primary (gap ≥ 13, n=215)

| | parent | treatment 200 | control 200 |
|---|---:|---:|---:|
| hits | 74 | 74 | 75 |
| overwrite-head mass | 0.0022 | **0.133** | 0.0006 |
| train L_ptr (step 150) | ~5.6 | **1.98** | 0 |
| induction first-top1 | 0.297 | **0.297** | **0.297** |

Treatment−control excess **−0.005**, CI **[−0.015, 0.0]**. Gate still ~0.03
(phase A). Cosine 0.042→0.049.

## What this says

Local `(h0[t], h0[t+1])` scoring is the first pointer class that leaves
uniform. It did not reach 0.20 in 200 updates, so P8 cannot continue. The
write is still off. Freeze-backbone still retains induction.

## Licensed next

P9: the same slot overwrite, Baby frozen, 200-update futility only if mass
< **0.05** (P8 already showed this class exceeds that), otherwise continue
to 800 so gate-on can fire.
