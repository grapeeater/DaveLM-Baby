# D3b: L0 splice control decomposition

Protocol id: `V010_QUERY_SPLICE_D3B`
Status: **preregistration — frozen before any D3b forward**

Read-only. No optimizer. No TEST. D3 stays **MIXED**. P2 not launched.
Authoritative parent unchanged.

Licensed by D3 MIXED: the L0 query/match-key lifts were large, but the frozen
negative control sat at **L11**, the wrong layer for that effect.

## 1. Hypothesis

D3 Q0R long-gap gold 127/215 vs as-is 74/215. Three remaining explanations:

**H_QUERY_IDENTITY.** L0 `gen_pos` must hold the **query token's** residual.
A competitor-key residual at L0 will not raise gold; it may raise the
competitor's own value.

**H_KEY_SUBSPACE.** L0 `gen_pos` must leave the filler/prev-token attractor
and enter a **key-like** subspace. Query and competitor-key L0 splices both
raise **gold**, because later layers still read the real query in the
sequence.

**H_OVERWRITE.** Any L0 residual overwrite of `gen_pos` (including a
non-key filler position) raises gold.

## 2. Identity

Same parent, same S2 diagnostic, same `eligible_item` filter as D3.
Device `cuda`. Splices `sdpa`. No P1 splice.

Replication (else INVALID): as-is long-gap hits **74** and Q0R hits **127**,
matching D3 exactly.

## 3. Frozen cells (long-gap only gated)

| cell | source | op | layer |
|---|---|---|---|
| as_is | — | — | — |
| Q0R | query | replace | 0 |
| C0R | competitor_key | replace | 0 |
| R0R | filler position | replace | 0 |

Filler position: last index in `[0, gen)` that is not `query_position` and
not any rendered inventory-key start. Rows without a filler position are
dropped from **all** gated cells (shared support).

For C0R also score `spliced`: candidate argmax equals the source-order pair
index of the spliced key (`source_index` from `recover_rendered_pairs`).
As-is `spliced` on the same pair is the baseline for that metric.

## 4. Decision rules — fixed before seeing results

Shared-support long-gap rows only. `Δg(X) = gold(X) - gold(as_is)`.
`Δs(C0R) = spliced(C0R) - spliced(as_is)` on the competitor pair.

Lift **+0.10**. Negative **+0.03**.

| verdict | rule (replication held) |
|---|---|
| **QUERY_IDENTITY** | `Δg(Q0R) ≥ +0.10` and `Δg(C0R) ≤ +0.03` and `Δg(R0R) ≤ +0.03` |
| **KEY_SUBSPACE** | `Δg(Q0R) ≥ +0.10` and `Δg(C0R) ≥ +0.10` and `Δg(R0R) ≤ +0.03` and `Δs(C0R) < +0.10` |
| **SPLICED_IDENTITY** | `Δg(Q0R) ≥ +0.10` and `Δs(C0R) ≥ +0.10` and `Δg(C0R) ≤ +0.03` and `Δg(R0R) ≤ +0.03` |
| **OVERWRITE** | `Δg(R0R) ≥ +0.10` |
| **NONE** | `Δg(Q0R) < +0.10` |
| **MIXED** | anything else valid |
| **INVALID** | as-is hits ≠ 74 or Q0R hits ≠ 127 on the **full** D3 long-gap set (before filler drop), or filler-drop leaves n < 50 |

Precedence if OVERWRITE and another lift both fire: **OVERWRITE**.
If QUERY_IDENTITY and SPLICED_IDENTITY both fire: **SPLICED_IDENTITY**.

What this licenses (not run here):

- QUERY_IDENTITY or SPLICED_IDENTITY → early query pointer (max over layers
  **0–2 only**), matched λ=0 control. Not P2. Not P1-all-layers.
- KEY_SUBSPACE → same early query pointer still licensed (query V is
  key-like); do not train a competitor-key aux.
- OVERWRITE → residual overwrite is too blunt; next diagnostic must change
  attention Q at gen without replacing the whole residual.
- NONE → D3 L0 lift did not replicate.

## 5. Frozen

These rules are not to be changed after results are seen.
