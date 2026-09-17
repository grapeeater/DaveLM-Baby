# D3: Early vs late residual splice

Protocol id: `V010_QUERY_SPLICE_D3`
Status: **preregistration — frozen before any splice forward**

Read-only. No optimizer. No TEST. Gate L/C/R unchanged. Authoritative parent
unchanged (v2R4 U16000). D1 stays INVALID. D1b stays B. P1 stays REGRESSION.
P2 stays preregistered and is **not launched** by this protocol.

Licensed by D2 **MIXED** plus the already-frozen D2 layer table, which this
protocol treats as a site-of-write question rather than a license to train
another final-residual loss.

## 1. Why this, and not P2 yet

D2 showed P1 tracking transferred (0.81) while long-gap candidate argmax did
not. The same D2 identity table, already frozen, shows:

- Working short-gap twins differ **early**: gap≤1 block cosine L0 0.938, L1
  0.877, then reconverges toward L11 0.959 (parent D1b).
- `last_is_query` is strongest at L0 (0.368).
- P1 treatment long-gap twins stay identical through L7 (~0.9999) and only
  diverge at L8–L11 (0.9996 → 0.9973).
- Prev-token mass remains ~0.99 after P1.

P2 pulls the **final-norm residual** toward the query-token residual. That is
a late write, the P1 site, and the site the working circuit does *not* use as
its primary query-identity injection.

The missing split is causal and read-only: if query identity (or matching-key
identity) is spliced into `gen_pos` at an **early** block and the rest of the
forward is allowed to run, does long-gap selection recover? If only a **late**
splice recovers, P2 is licensed. If neither query splice recovers but a
matching-key splice does, the bottleneck is routing to the inventory key, not
query presence at unembed.

## 2. Hypothesis

**H_EARLY_QUERY.** Replacing or mixing `gen_pos` with the same item's
`query_pos` residual at block 0–2, then continuing the forward, raises
long-gap candidate accuracy by ≥ **+0.10** vs as-is. Later layers can finish
QUERY → KEY → VALUE if query identity arrives early.

**H_LATE_QUERY.** The same query splice works at block 11 and not at 0–2.
Then a final-residual bind (P2) is the licensed class.

**H_MATCH_KEY.** Query splices fail, but splicing the **matching inventory
key** residual into `gen_pos` raises gold accuracy. Then the copy path is
"unembed whatever the matched key residual predicts," and the missing operator
is routing to that key, not making `h[gen] ≈ h[query]`.

**H_NONE.** No query or match-key splice meets the lift bar. Then residual
identity at `gen_pos` is the wrong intervention class.

Matched negative control: splice a **competitor key** residual at block 11.
This must not raise **gold** accuracy.

Software instrument: `logit_steer` adds `10 * W_U[gold]` to the final-norm
residual at `gen_pos`. Among long-gap misses, gold accuracy must become ≥
**0.95**. If it does not, the hooks never reached the decoder and the run is
INVALID.

## 3. Identity

| item | value |
|---|---|
| parent | U16000 SHA `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` |
| frozen eval | S2 `runs/selection_s2/DIAGNOSTIC.json` (same primary rows as P1/P2) |
| P1 treatment (census only) | `checkpoint_16800.pt` SHA `5c8296ef759543dd608c18b8048ccce4bd4fcf8de7a6183b2d84e5f53d4fbafa` |
| device | `cuda` |
| attention for splices | `sdpa` |
| attention for layer census | `reference` |
| optimizer | none |
| train | false |

P1 checkpoint is used only for a per-layer query-mass histogram. It is not
spliced, not trained, and not promoted.

## 4. Frozen conditions

Primary rows: S2 diagnostic keyed items with `gap = gen - query_pos ≥ 13`.
Score is candidate-restricted argmax on `candidate_heads` (same as P1/P2).

Sources (same item, never twins):

| name | position |
|---|---|
| `query` | `query_position` |
| `match_key` | rendered inventory key whose `original_key` or displayed `key` equals `query_key` |
| `competitor_key` | first other rendered inventory key |

Operations:

- `replace`: `h[gen] ← h[src]` at the named `block_out` layer, then remaining
  blocks + final_norm + unembed.
- `mix`: `h[gen] ← 0.5 h[gen] + 0.5 h[src]` at that layer, then continue.
- `query_embed_add`: add `E[query_key]` to the residual at `gen_pos` **before**
  block 0 (token+position embeddings already applied). Last token unchanged.
- `logit_steer`: after `final_norm`, `h[gen] ← h[gen] + 10 * W_U[gold_head]`.

Required cells (no others gated):

| cell | source | op | site |
|---|---|---|---|
| Q0R | query | replace | block 0 |
| Q1R | query | replace | block 1 |
| Q2R | query | replace | block 2 |
| Q11R | query | replace | block 11 |
| Q0M | query | mix | block 0 |
| Q1M | query | mix | block 1 |
| Q11M | query | mix | block 11 |
| K0R | match_key | replace | block 0 |
| K11R | match_key | replace | block 11 |
| C11R | competitor_key | replace | block 11 |
| QE | query_embed_add | add | pre-block 0 |
| STEER | logit_steer | add | final_norm |

Also report as-is, short-gap (gap≤1) as-is, and whether greedy first token
equals the query token after each splice (copy-the-query failure mode).

M4 census: for parent and P1 treatment, on the same long-gap rows, per-layer
max-head attention mass on `query_pos` from `gen_pos`. Record the median
argmax layer (which layer holds the max mass).

## 5. Decision rules — fixed before seeing results

Instrument: among long-gap as-is **misses**, STEER gold accuracy ≥ **0.95**.
Else **INVALID**.

Let `Δ(cell) = accuracy(cell) - accuracy(as_is)` on long-gap rows that have
every required source position. Lift bar **+0.10**. Negative bar: `Δ(C11R)`
on **gold** accuracy ≤ **+0.03**.

`early_query` is the max of `{Δ(Q0R), Δ(Q1R), Δ(Q2R), Δ(Q0M), Δ(Q1M), Δ(QE)}`.
`late_query` is the max of `{Δ(Q11R), Δ(Q11M)}`.
`match_key` is the max of `{Δ(K0R), Δ(K11R)}`.

| verdict | rule (instrument valid, C11R gold Δ ≤ +0.03) |
|---|---|
| **EARLY_QUERY** | `early_query` ≥ +0.10 and `late_query` < +0.10 |
| **LATE_QUERY** | `late_query` ≥ +0.10 and `early_query` < +0.10 |
| **BOTH_QUERY** | both ≥ +0.10 |
| **MATCH_KEY** | `match_key` ≥ +0.10 and `early_query` < +0.10 and `late_query` < +0.10 |
| **NONE** | no lift bar met |
| **MIXED** | C11R gold Δ > +0.03, or otherwise not in the table |
| **INVALID** | STEER fails |

What each licenses (not run by this protocol):

- EARLY_QUERY or BOTH_QUERY → early-layer pointer / early residual write
  (not P2 as written; not another max-over-all-layers P1 λ).
- LATE_QUERY → P2 residual-bind as written is licensed.
- MATCH_KEY → matching-key pointer / key-routing aux, not query-residual
  matching at unembed.
- NONE → residual splice of query or matched key is the wrong class;
  next diagnostic must target value-copy routing with query-conditional
  **keys**, or architecture.

P2 is not launched here even if LATE_QUERY. A later protocol may launch it.

## 6. What this does not do

Train. Open TEST. Mutate S1/S2/D1b. Resume T1. Raise P1 λ. Enable the
binding sidecar. Promote a checkpoint. Re-score D1.

## 7. Frozen

These rules are not to be changed after results are seen.
