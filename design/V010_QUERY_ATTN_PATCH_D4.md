# D4: L0 attention one-hot patch

Protocol id: `V010_QUERY_ATTN_PATCH_D4`
Status: **preregistration — frozen before any D4 forward**

Read-only. No optimizer. No TEST. P3 and P4 stay **REGRESSION**. P2 not
launched. Authoritative parent unchanged.

Licensed by P4 REGRESSION plus D3b SPLICED_IDENTITY. Matching losses at L0
did not install the splice. This asks whether **forcing L0 attention** at
`gen` onto the query token is enough for OV to write query identity.

## 1. Hypothesis

**H_OV_WRITES.** If every L0 head at `gen_pos` is one-hot on the query
position, block-0 output at `gen` becomes query-like and later layers emit
gold, as in D3b Q0R.

**H_OV_DEAD.** The same patch leaves L0 cosine ~0.04 and gold at chance.
Then L0 OV cannot copy the query residual even with perfect attention, and
another residual-matching or all-layer pointer loss is not licensed.

**H_PREV.** One-hot on `gen-1` lifts gold as much as one-hot on query. Then
the patch is a generic overwrite, not a query write.

Matched identity control: one-hot on the **competitor key** start. Gold must
not rise; the competitor's own value may.

## 2. Why this class

Not P4 λ. Not P3 λ. Not P1 max-over-layers (gamed at L10). Not P2. D3b
replaced the residual after the whole block. D4 changes only L0 **attention
weights** before OV and MLP, which is the copy-head circuit P1 never pinned
to layer 0.

## 3. Identity

| item | value |
|---|---|
| parent | U16000 SHA `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` |
| frozen eval | S2 diagnostic, same `eligible_item` as D3/D3b |
| device | `cuda` |
| patch attention | `reference` (weights must be editable) |
| as-is / Q0R / steer | `sdpa`, same instruments as D3b |
| optimizer | none |

P1/P3/P4 checkpoints are not patched.

## 4. Frozen cells (long-gap eligible rows)

| cell | intervention |
|---|---|
| as_is | none |
| Q0R | D3b residual replace at block 0 (validity replication) |
| A0Q | L0 attn at `gen`: all heads one-hot **query** |
| A0C | L0 attn at `gen`: all heads one-hot **competitor key** |
| A0P | L0 attn at `gen`: all heads one-hot **gen-1** |
| STEER | `logit_steer` on as-is misses (validity) |

Also record median `cosine(h0[gen], h0[query])` after each cell that produces
a block-0 residual.

A0C also scores `spliced`: candidate argmax equals the competitor pair index.

## 5. Decision rules — fixed before seeing results

Replication (else **INVALID**): as-is hits **74**, Q0R hits **127**, STEER on
as-is long-gap misses ≥ **0.95**.

`Δg(X) = gold(X) - gold(as_is)` on the shared long-gap set.
`Δs(A0C) = spliced(A0C) - spliced(as_is)` on the competitor pair.
Lift **+0.10**. Negative **+0.03**. Cosine write bar: median L0 cosine under
A0Q ≥ **0.20** or gain vs as-is ≥ **+0.10**.

| verdict | rule |
|---|---|
| **POINTER_SUFFICIENT** | `Δg(A0Q) ≥ +0.10` and A0Q writes (cosine bar) and `Δs(A0C) ≥ +0.10` and `Δg(A0C) ≤ +0.03` and `Δg(A0P) ≤ +0.03` |
| **POINTER_SELECTS** | `Δg(A0Q) ≥ +0.10` and cosine bar missed and `Δg(A0P) ≤ +0.03` |
| **POINTER_WRITES_ONLY** | cosine bar hit and `Δg(A0Q) < +0.10` and `Δg(A0P) ≤ +0.03` |
| **OV_DEAD** | cosine bar missed and `Δg(A0Q) < +0.10` and `Δg(A0P) ≤ +0.03` |
| **PREV_TOKEN** | `Δg(A0P) ≥ +0.10` |
| **MIXED** | anything else valid |
| **INVALID** | replication failed |

Precedence: INVALID, then PREV_TOKEN, then POINTER_SUFFICIENT, then the
single remaining named class, else MIXED.

Census only (not gated): one-batch parent `‖grad L_copy‖ / ‖grad L_CE‖` on
an S2 long-gap row using the P4 copy functional.

## 6. What this licenses (not run here)

- POINTER_SUFFICIENT or POINTER_SELECTS → P5 **L0-only** pointer vs matched
  λ=0. Not P1-all-layers. Not P4 λ.
- POINTER_WRITES_ONLY → next diagnostic patches or skips L0 MLP at `gen`
  (attention wrote; MLP may wipe it).
- OV_DEAD → architectural write (explicit copy/replace after attn), not
  another matching loss.
- PREV_TOKEN → do not treat one-hot attention as query-specific.

## 7. Frozen

These rules are not to be changed after results are seen.
