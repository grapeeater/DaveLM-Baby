# D1b: Query Presence Localization on a fresh diagnostic

Protocol id: `V010_QUERY_PRESENCE_D1B`
Status: **preregistration — frozen before diagnostic generation and before any
new parent forward**

Read-only. No optimizer. No TEST. Gate L/C/R unchanged. Authoritative parent
unchanged. D1 receipts are not rewritten and D1's INVALID verdict is not
reopened.

D1's positive-control bar was `flip_toward_donor` ≥ 0.25 on **all** gap ≤ 1
directed pairs, including pairs whose unpatched argmax was already the donor
gold (which cannot flip by definition). That bar missed by three pairs. D1b does
**not** lower 0.25 on those same pairs. It freezes a different instrument on
**new bodies**.

## 1. Why this is a new experiment

The D1 data have been seen. Applying an eligible-only flip rule to those 362
pairs would be post-hoc. D1b therefore:

- generates a new 144-body all-K diagnostic (seed `140200`)
- denies frozen panels, S1 diagnostic, S2 diagnostic, and D1's scored inputs
- re-runs M1–M3 on the hashed parent
- uses an eligible-flip positive control frozen here, before generation

## 2. Hypothesis (unchanged)

**H_B.** On fresh twins, when gap ≥ 13, gen-position residual is query-invariant
and patching does not flip greedy selection. Prev-token attention remains ~1;
query-slot mass does not.

**H_A.** Long-gap `flip_toward_donor` at `final` ≥ 0.20.

**H_C.** Residual invariant and ≥ 50% of long-gap rows have a query-tracking head.

## 3. Identity

| item | value |
|---|---|
| parent | `checkpoint_16000.pt` SHA `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` |
| diagnostic seed | `140200` |
| bodies | 48 per K ∈ {2,3,4} = 144, all-K twins |
| device | `cuda` / RX 9060 XT |
| attention | `reference` |
| trained | false |

## 4. Measurements

Same M1–M3 as D1 (`design/V010_QUERY_PRESENCE_D1.md` sections 4), including
identity-skip at max-abs < 1e-5.

Additional frozen statistics (not optional):

- `eligible_flip_toward_donor`: among directed pairs with `before != donor_gold`
- `n_eligible`
- `changed` at `final` for gap ≥ 31

## 5. Frozen decision rules

### Instrument validity (else INVALID)

1. Parent SHA match. Diagnostic SHA match vs D1b manifest. Protected false.
   Twins differ only at `query_position`. SDPA vs reference max-abs < 1e-3.
2. Gap ≤ 1 median final residual cosine ≤ **0.99** (twins are not identical).
3. Gap ≤ 1 `fraction_any_prev_tracking_head` ≥ **0.90**.
4. Gap ≤ 1 `eligible_flip_toward_donor` at `final` **or** best `block_out` ≥ **0.25**,
   with `n_eligible` ≥ **50**.

Rule 4 is the D1 positive control with pairs that cannot flip removed. It is
**not** a lowered 0.25. Raw (all-pair) flip is reported and is not a validity
gate.

### Long-gap adjudication (gap ≥ 13), valid instruments only

Precedence: **A**, then **COMPOSITION**, then **B**, then **MIXED**.

| verdict | rule |
|---|---|
| **A** | `flip_toward_donor` at `final` ≥ 0.20 |
| **COMPOSITION** | not A, query-tracking-head fraction ≥ 0.50, median final residual cosine ≥ 0.999 |
| **B** | median final residual cosine ≥ 0.999 **and** median full-vocab logit cosine ≥ 0.995 **and** `flip_toward_donor` at `final` and every `block_out`/`attn_out` ≤ 0.05 **and** gap ≥ 31 `changed` at `final` ≤ 0.05 |
| **MIXED** | anything else |

## 6. What this does not authorize

Training, T1 resume, S3, sidecar, moving D1's 0.25 bar, opening TEST, promoting
a checkpoint. A treatment protocol may be frozen **only after** a valid B, A, or
COMPOSITION verdict, as a separate preregistration.

## 7. Outputs

`runs/query_presence_d1b/QUERY_PRESENCE.json`, `ADJUDICATION.json`, hashed local
`ROWS_*.json`, `research/V010_QUERY_PRESENCE_D1B.md`.

## 8. Frozen

These rules are not to be changed after seeing D1b numbers.
