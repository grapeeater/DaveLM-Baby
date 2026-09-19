# P1 pointer-attention auxiliary: REGRESSION, pointer transferred, selection did not

Status: **REGRESSION** by the frozen retention gate (`primitive_induction`
first-top1 0.297 → 0.188). Authoritative Baby unchanged: v2R4 U16000
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
Protected TEST was not opened. Gate L/C/R were not moved. No checkpoint
promoted.

Protocol:
[`design/V010_SELECTION_REPAIR_P1_POINTER.md`](../design/V010_SELECTION_REPAIR_P1_POINTER.md)
Licensed by D1b valid B. D1 remains INVALID.

Adjudication: `runs/selection_p1/ADJUDICATION_150001_800.json`

## What happened

Both arms ran to the 800-update terminal. Futility did not fire: long-gap
query-tracking-head fraction on the frozen S2 diagnostic rose from 0.377 to
0.851 by update 400 (bar to continue was +0.10).

| metric (gap ≥ 13, n=215) | parent | treatment 800 | control 800 |
|---|---:|---:|---:|
| candidate accuracy | 0.344 | 0.340 | 0.344 |
| excess over chance | ~0 | −0.005 | ~0 |
| query-tracking-head fraction | 0.377 | **0.851** | 0.367 |
| median max-head query mass | 0.041 | **0.935** | 0.042 |

Treatment−control on the primary endpoint: **−0.005**, bootstrap 95% CI
[−0.025, +0.015] includes 0.

At gap ≥ 31 (n=159) treatment tracking is 0.811 and median query mass is 0.615.
The aux did not only fire on short-gap rows.

If the induction retention gate were ignored, the frozen mechanism rule would
return **MECHANISM SUPPORTED, DOSE INSUFFICIENT**. That is a counterfactual.
The official verdict is REGRESSION.

## Retention

| metric | parent | treatment | control | drop bar 0.05 |
|---|---:|---:|---:|---|
| language DEV CE | 1.243 | 1.236 | 1.234 | no (improved) |
| `primitive_induction` first-top1 | 0.297 | **0.188** | 0.219 | **yes** (both arms) |
| `primitive_keyed` first-top1 | 1.000 | 1.000 | 1.000 | no |
| `short_keyed` free exact | 0.844 | 0.828 | 0.828 | no |
| gold `rest_lock` | 0.977 | 0.977 | 0.977 | no |

Induction drop is in the **control** as well, so it is not uniquely the pointer
term. It still trips the frozen gate.

## Negative controls

`value_absent` free exact stays 0. `broken_context` free exact 0.016 ≤ 0.05.
`broken_order` first-top1 0.

## What this says about H1

H1 was: raising max-head query mass at gen_pos will write query identity into
the residual **and** raise long-gap candidate selection.

The first half happened on the frozen attention instrument (and only in the
λ=0.25 arm). The second half did not. Query-logit effect rose 0.78 → 1.03,
the S1/S2 pattern: amplitude without greedy selection.

So a gen-position pointer is **not sufficient** for keyed selection on this
parent. Either the attention mass is not mixing into the residual that the
unembed reads, or query identity in that residual is still unused by the
candidate decoder.

## What this does not license

Another λ, a T1 resume, sidecar enablement, TEST, or promoting
`checkpoint_16800.pt`. Next licensed read-only question: D2 query-presence on
the P1 treatment checkpoint — are long-gap twins still residual-identical
after the pointer write?
