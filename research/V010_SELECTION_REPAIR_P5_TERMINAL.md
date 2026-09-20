# P5 identity overwrite: REGRESSION (futility at 200)

Status: **REGRESSION**. Futility at 200. Zero Q/K init made the overwrite
pointer's score Jacobian vanish, so the aux never engaged. Authoritative
Baby unchanged: v2R4 U16000
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
P2 not launched. No TEST.

Protocol:
[`design/V010_SELECTION_REPAIR_P5_OVERWRITE.md`](../design/V010_SELECTION_REPAIR_P5_OVERWRITE.md)
Adjudication: `runs/selection_p5/ADJUDICATION_190001_200.json`

## Primary (gap ≥ 13, n=215)

| | parent | treatment 200 | control 200 |
|---|---:|---:|---:|
| hits | 74 | 74 | 74 |
| overwrite-head mass | 0.0111 | 0.0111 | 0.0111 |
| median gate | 0.018 | 0.0185 | 0.0185 |
| L0 cosine | 0.042 | 0.042 | 0.042 |

Treatment−control excess **0**, bootstrap 95% CI **[0.0, 0.0]**. Eval JSON
for the two arms is identical. `L_ptr` stayed ~4.0 (uniform −log mass).

Futility fired: overwrite-head long-gap mass 0.011 < 0.20.

## Retention

`primitive_induction` first-top1 0.297 → **0.219** in **both** arms.
Language CE 1.244 → 1.249 both. Official REGRESSION on the parent drop.

## What this says

Identity overwrite with gate≈0 preserves the parent at step 0 (74 hits).
Zero-initialized Q and K make `scores = (xW_q)(xW_k)ᵀ = 0` and
`∂scores/∂W_q = 0` when `W_k = 0` (and symmetrically). The pointer cannot
learn. D4's architectural write is still licensed; this init is not.

Unit confirmation: `tests/test_residual_overwrite.py`
`test_zero_qk_pointer_grad_vanishes` vs `test_xavier_qk_pointer_grad_flows`.

## What this does not license

Raising P5 λ, resuming P5, P4 λ, P1 λ. Next: the same overwrite operator
with **Xavier Q/K** so the pointer Jacobian is nonzero at init.
