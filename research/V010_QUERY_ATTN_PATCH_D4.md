# D4 L0 attention patch: OV_DEAD

Status: **OV_DEAD**. Replication held. Authoritative Baby unchanged: v2R4
U16000 `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
P2 not launched. No TEST. P3/P4 stay REGRESSION.

Protocol:
[`design/V010_QUERY_ATTN_PATCH_D4.md`](../design/V010_QUERY_ATTN_PATCH_D4.md)
Adjudication: `runs/query_attn_patch_d4/ADJUDICATION.json`

## Replication

| cell | hits / 215 |
|---|---:|
| as-is | **74** |
| Q0R residual replace | **127** |
| STEER on 141 misses | **1.00** |

## Patches (all L0 heads one-hot)

| cell | gold hits | Δg | median L0 cosine |
|---|---:|---:|---:|
| as-is | 74 | 0 | 0.040 |
| A0Q (query) | 90 | **+0.074** | 0.070 |
| A0C (competitor) | 68 | −0.028 | 0.049 |
| A0P (prev token) | 74 | 0.000 | 0.028 |

A0C spliced 80→92 (Δs **+0.056** < +0.10). Cosine write bar (A0Q ≥ 0.20 or
gain ≥ +0.10) missed. Gold lift bar (+0.10) missed. A0P did not lift.

## Grad census (not gated)

On one S2 long-gap row, P4 copy functional: `L_copy=0.89` vs `L_CE=0.20`,
but `‖g_copy‖/‖g_CE‖ = 0.077`. The matching loss is almost flat in
parameter space.

## What this says

Oracle L0 attention through the existing OV+residual+MLP path does **not**
implement the D3b replace. Attention can only **add** a projected value;
D3b **replaces** `h0[gen]` with `h0[query]`. That is why P1 (late pointer),
P3 (InfoNCE), and P4 (cosine copy) could not install the splice.

## Licensed next

Architectural identity overwrite after block 0 (`h[gen] ← (1-g)h[gen] +
g·read` with identity values), not another matching loss, not P1 λ, not P4 λ.
