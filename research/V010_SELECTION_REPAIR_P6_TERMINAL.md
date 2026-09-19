# P6 Xavier overwrite: REGRESSION (futility at 200)

Status: **REGRESSION**. Futility at 200. Xavier Q/K made the pointer
Jacobian exist, but v2R4 `HIGH_LR=3.75e-5` did not move overwrite-head mass
off ~0.006. Authoritative Baby unchanged: v2R4 U16000
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
P2 not launched. No TEST.

Protocol:
[`design/V010_SELECTION_REPAIR_P6_OVERWRITE_XAVIER.md`](../design/V010_SELECTION_REPAIR_P6_OVERWRITE_XAVIER.md)
Adjudication: `runs/selection_p6/ADJUDICATION_200001_200.json`

## Primary (gap ≥ 13, n=215)

| | parent | treatment 200 | control 200 |
|---|---:|---:|---:|
| hits | 74 | 74 | 75 |
| overwrite-head mass | 0.0052 | 0.0065 | 0.0052 |
| L_ptr (train) | ~5.04 | ~4.80 | 0 |

Treatment−control excess **−0.005**, CI **[−0.015, 0.0]**.

## Retention

`primitive_induction` 0.297 → **0.203** treatment, **0.219** control. Same
diet drop as P3–P5. Official REGRESSION.

## What this says

The saddle is gone (mass is not frozen at 1/T). The step size on a new
640×640 Q/K at v2R4 later-layer LR cannot reach the 0.20 mass bar in 200
updates. Continued training of Baby weights keeps tripping induction even
when the aux does nothing.

## Licensed next

Freeze Baby weights. Train **only** the overwrite module at a module-scale
LR (1e-3). Not P6 λ. Not P5 zero Q/K.
