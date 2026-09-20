# D3 residual splice: MIXED

Status: **MIXED**. Instrument valid. Negative control failed. No training
license. P2 not launched. Authoritative Baby unchanged: v2R4 U16000
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.

Protocol:
[`design/V010_QUERY_SPLICE_D3.md`](../design/V010_QUERY_SPLICE_D3.md)
Adjudication: `runs/query_splice_d3/ADJUDICATION.json`

## Frozen verdict

STEER recovered **141/141** long-gap as-is misses (bar 0.95). Hooks reached
the decoder.

`C11R` gold Δ = **+0.186** > +0.03. The L11 competitor-key replace was
supposed to leave gold flat. It did not. Official headline **MIXED**. Early
query lift (+0.247) and match-key L0 lift (+0.270) are **not** a licensed
EARLY_QUERY under the frozen rule.

## Long-gap candidate accuracy (n=215, chance 0.344, as-is 74/215)

| cell | hit | accuracy | Δ vs as-is | emit query |
|---|---:|---:|---:|---:|
| as-is | 74 | 0.344 | 0 | 0 |
| STEER | 215 | 1.000 | +0.656 | 0 |
| Q0R | 127 | 0.591 | **+0.247** | 0 |
| Q1R | 123 | 0.572 | +0.228 | 0 |
| Q2R | 125 | 0.581 | +0.237 | 0 |
| QE | 121 | 0.563 | +0.219 | 0 |
| K0R | 132 | 0.614 | **+0.270** | 0 |
| Q11R | 105 | 0.488 | +0.144 | 0 |
| Q11M | 91 | 0.423 | +0.079 | 0 |
| C11R | 114 | 0.530 | **+0.186** | 0 |
| K11R | 34 | 0.158 | **−0.186** | 0 |

No splice made Baby emit the query token.

## M4 pointer census (long-gap max-head query mass)

Parent: no layer has mass. Median argmax layer 5 with median mass ≤ 0.027.

P1 treatment SHA `5c8296ef…4fbafa`: median argmax layer **10**, L10 median
mass **0.808**, L11 0.020. The trained pointer is a late-layer head. One
block remains after it (L11 MLP). That is the P1 site-of-write.

## What MIXED still shows, without moving the gate

The software instrument works. P1 did not install an early query read. Late
match-key replace at L11 is anti-gold. L11 competitor replace is not a clean
"wrong key" control. Those facts license a **control-decomposition** (D3b),
not P2 and not a P1 λ bump.

## What this does not license

P2 as written, early-layer training, sidecar, TEST, promotion.
