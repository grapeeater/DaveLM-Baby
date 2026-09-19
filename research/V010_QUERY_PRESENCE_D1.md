# D1 Query Presence Localization: INVALID instrument, long-gap pattern is B-like

Status: **INVALID by the frozen positive-control bar**. No treatment authorized.
Authoritative Baby unchanged: v2R4 U16000
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
Protected TEST was not opened. Gate L/C/R were not moved. No optimizer step.

Protocol (hash recorded in the run receipt **before** scoring):
[`design/V010_QUERY_PRESENCE_D1.md`](../design/V010_QUERY_PRESENCE_D1.md)
`ba181fe1fc772ed109ff911fb0672b847c58ac645eed38515f5eac7eaa21ea35`

Adjudication: `runs/query_presence_d1/ADJUDICATION.json`
Summary: `runs/query_presence_d1/QUERY_PRESENCE.json`

## Freeze note

A git freeze commit was attempted before the parent forward and **failed**
(`user.email` / `user.name` unset; git config was not changed). The protocol
bytes were hashed as `ba181fe1…1ea35` and that hash is inside the run
preflight object. The run is therefore reconstructible even without that
commit. This is a provenance defect, not a license to move gates.

## Preflight (passed)

| check | result |
|---|---|
| parent SHA | match |
| S2 diagnostic SHA | match `5b63533f…3dc446` |
| twins differ only at `query_position` | true |
| protected flag | false |
| SDPA vs reference max-abs gen logit | **7.6e-6** (bar 1e-3) |
| `W_U` rank | **640 / 640**, injective |
| update | 16000 |

## Frozen adjudication

Positive control, gap ≤ 1, 362 directed pairs:

| metric | value | bar | pass |
|---|---:|---:|---|
| median full-vocab logit cosine | 0.9949 | ≤ 0.995 | yes |
| `flip_toward_donor` at `final` | **0.2431** | ≥ 0.25 | **no** |
| best `block_out` (layer 7) | 0.2431 | ≥ 0.25 | **no** |

`invalid_reasons = ["positive_flip"]`.
88/362 pairs flipped toward the donor at `final`; the bar required ≥ 0.25 × 362
= 90.5, i.e. **91**. Missed by three pairs.

**Verdict stays INVALID.** The bar is not moved. Long-gap numbers below are
descriptive. They are not a licensed B/A/COMPOSITION call.

## What the instrument actually measured

M2 attention is not broken. On gap ≤ 1 rows, some head puts **0.98** mass on
the previous token / query slot (they are the same position when gap = 1), and
`fraction_any_prev_tracking_head = 1.0`. The model has a near-hard prev-token
head.

On gap ≥ 31 that same prev-token mass stays **0.98**, now on filler. Query-slot
median max mass falls to **0.030**. Query-tracking-head fraction is 0.27, below
the 0.50 COMPOSITION bar.

`W_U` is injective, so full-vocab logit identity is residual identity.

| stratum | pairs | logit cosine | residual cosine | same argmax | `final` flip toward donor | `final` changed |
|---|---:|---:|---:|---:|---:|---:|
| last token = query | 14 | 0.942 | 0.838 | 0.286 | (in short pool) | — |
| prev token = query | 348 | 0.995 | 0.966 | 0.661 | (in short pool) | — |
| gap ≤ 1 (all) | 362 | 0.995 | 0.965 | 0.646 | **0.243** | 0.354 |
| query deeper | 598 | 0.99994 | 0.99957 | 0.960 | — | — |
| gap ≥ 13 | 460 | 0.99998 | 0.99979 | 0.978 | **0.0043** | 0.022 |
| gap ≥ 31 | 346 | 0.99998 | 0.99986 | **1.000** | **0.000** | **0.000** |

Identity skips (`max_abs < 1e-5`) were **zero** even at long gap. Patches were
actually applied. At gap ≥ 31, **0/346** patched forwards changed candidate
argmax.

Post-hoc, not a gate: among gap ≤ 1 pairs whose unpatched argmax was *not*
already the donor gold (278/362), flip rate is 0.317. That number was not
frozen and is not used to pass D1.

## Hypotheses, informally

If the positive-control bar had been met, the frozen long-gap rule would have
returned **B**: residual cosine 0.99979 ≥ 0.999, logit cosine 0.99998 ≥ 0.995,
every site's flip ≤ 0.05, query-tracking fraction 0.377 < 0.50.

That is **not** the official verdict.

Competing reading that remains alive because D1 is INVALID: the 0.243 short-gap
flip means even a *present* residual only sometimes overrides the salience
prior, so a slightly different patch site or a different flip definition might
have been the right positive control. That is a reason to design D1b, not a
reason to relabel this run.

What is hard to rescue: gap ≥ 31 is bit-for-bit query-invariant in greedy
choice after actually writing the other twin's gen residual. That is not a
weak A.

The prev-token head explains the locality cliff without a stretchable
transport pathway: at gap 1 it copies the query; at gap ≥ 2 it copies filler.

## What this does not authorize

- Treating D1 as B
- T1+800, S3, λ, binding sidecar, architecture change
- Opening TEST, promoting a checkpoint, moving Gate L/C/R

## Highest-information next action

Owner review. If a successor is authorized, freeze **D1b** *before* any
forward, with an explicit positive-control definition that cannot miss by
three pairs on a working prev-token circuit (for example: gap ≤ 1
`flip_toward_donor` among pairs with `before != donor_gold` ≥ 0.25, **and**
gap ≥ 31 `changed` ≤ 0.02 as a negative control). Do not lower this run's
0.25 bar after seeing 0.243.

## Local-only artifacts

Pair dumps stay local. Hashes:
[`V010_QUERY_PRESENCE_D1_SHA256SUMS.txt`](V010_QUERY_PRESENCE_D1_SHA256SUMS.txt).
Protected material opened: false.
