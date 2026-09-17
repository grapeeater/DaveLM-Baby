# D1: Query Presence Localization

Protocol id: `V010_QUERY_PRESENCE_D1`
Status: **preregistration — frozen before any diagnostic forward on the parent**

Read-only. No optimizer. No weight write. No TEST/FINAL/SACRED. Gate L/C/R
unchanged. Authoritative parent unchanged. S1/S2/T1 receipts are not rewritten.

This is not T1+800, not S3, and not a sidecar enablement. It answers whether, at
the generation position when query gap > 1, query identity is **present but
unused** (A) or **absent from the residual that the LM head can see, and from
the residual itself** (B).

Prior evidence: gap-stratified candidate-logit decomposition on
`ROWS_parent_u16000.json` (session 2026-09-16). That decomposition is **not**
this protocol's input; D1 re-forwards the hashed parent.

## 1. Hypothesis

**H_B (leading).** On frozen all-K twins that differ by only the query token,
when the query sits two or more tokens before generation, the residual at the
generation position is query-invariant. Patching that residual from twin *i*
into twin *j* is therefore a no-op. No head at any layer assigns above-baseline
mass to the query position from the generation position.

**H_A.** The residual at the generation position still differs across twins at
long gap, and replacing *j*'s gen-position residual with *i*'s flips candidate
argmax toward *i*'s gold. Then identity arrived and greedy did not use it.

**H_C (composition).** Heads attend to the distant query position, but the
written residual remains query-invariant. Attention looks; the OV path does not
write key identity.

## 2. Identity

| item | value |
|---|---|
| parent | `runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt` |
| parent SHA-256 | `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` |
| frozen diagnostic | `runs/selection_s2/DIAGNOSTIC.json` vs `runs/selection_s2/MANIFEST.json` |
| device | `cuda` / AMD Radeon RX 9060 XT |
| attention | `reference` backend for all scored forwards (SDPA vs reference max-abs logit check is a preflight, not a scored arm) |
| trained | false |

Default `forward` is not modified. Tracing uses forward hooks plus the existing
`set_attention_backend("reference")`. Weights are not updated.

## 3. Data

All 432 frozen diagnostic rows (144 bodies, all-K query twins). Twins of a body
must differ at `query_position` only; the runner refuses the file otherwise.

Location classes, computed from the input bytes, not from variant labels:

- `last_is_query` — `input[-1] == query_key`
- `prev_is_query` — `input[-2] == query_key` and not last
- `query_deeper` — otherwise (gap ≥ 2)

Gap is `len(input) - 1 - query_position`. Primary long-gap split is **gap ≥ 13**.
Positive-control split is **gap ≤ 1**.

## 4. Measurements (in this order)

### M1 — Full-vocab twin identity

For every directed pair *(donor i, recipient j)* of a body, cosine, L2, and
max-abs of the 1024-d gen-position logit vectors, and of the K-d candidate
subvector. Also cosine/L2 of the final-norm residual at gen-position, and of
each layer's block-out and attn-out at gen-position.

If `W_U` (`language_head.weight`) has rank `d_model` under singular values
> 1e-4, a query-invariant full-vocab logit vector implies a query-invariant
final residual (injective unembed). Rank is recorded, not a gate.

### M2 — Attention from the generation position

Reference-softmax weights, gen-position row, every layer and head. Reported
masses: query slot, previous token, self, queried body-key occurrence, all body
keys, uniform `1/(gen_pos+1)`. A head **tracks the query** if its query-slot
mass is ≥ 5× uniform.

Positive control: `prev_is_query` / gap ≤ 1 rows must show at least one
query-tracking or prev-token-tracking head in some layer, or M2 is invalid as
an instrument (not a long-gap conclusion).

### M3 — Causal residual patch

Sites, always at the generation position:

- `final` — after `final_norm`, before `language_head`
- `block_out` — output of transformer block *L*, L = 0..11
- `attn_out` — output of attention module in block *L*, L = 0..11

Intervention: run recipient *j* with that site's gen-position vector replaced
by donor *i*'s cached vector from an unpatched forward.

If the donor and recipient vectors differ by max-abs < 1e-5, the patch is the
identity: **do not spend a forward**; record `skip_identity=true`, `changed=false`,
`flip_toward_donor=false`.

Otherwise run the patched forward. Metrics per pair×site:

- `before` / `after` candidate argmax
- `changed` — after ≠ before
- `flip_toward_donor` — after == donor gold index AND before ≠ donor gold index
- `match_donor_after` — after == donor gold index

## 5. Frozen decision rules

Evaluate **after** M1–M3. Do not move these numbers.

### Instrument validity (or D1 is INVALID)

1. Preflight parent SHA matches. Diagnostic SHA matches the S2 manifest.
   Protected flag on the checkpoint is false. Twin inputs differ only at
   `query_position`.
2. SDPA vs reference max-abs gen logit < 1e-3 on 8 smoke rows (eval mode).
3. **Positive control (gap ≤ 1 directed pairs):**
   - median full-vocab logit cosine ≤ 0.995 (twins are not identical)
   - `flip_toward_donor` rate at `final` **or** at the best `block_out` layer ≥ 0.25

If (3) fails, stop. Do not interpret long-gap A/B.

### Long-gap adjudication (gap ≥ 13 directed pairs)

Precedence after validity: **A**, then **COMPOSITION**, then **B**, then **MIXED**.
A is a causal flip and overrides residual-invariance labels. COMPOSITION is
residual invariance plus query-tracking attention (heads look, OV does not write).

| verdict | rule |
|---|---|
| **A** | `flip_toward_donor` at `final` ≥ 0.20 |
| **COMPOSITION** | not A, fraction of long-gap rows with any query-tracking head ≥ 0.50, and median final residual cosine ≥ 0.999 |
| **B** | median final residual cosine ≥ 0.999 **and** median full-vocab logit cosine ≥ 0.995 **and** `flip_toward_donor` at `final` ≤ 0.05 **and** at every `block_out` / `attn_out` ≤ 0.05 |
| **MIXED** | anything else |

`H_B` is **supported** by verdict B after a valid instrument.
`H_A` is **supported** by verdict A.
`H_C` is **supported** by verdict COMPOSITION.

A MIXED verdict means D1 did not decide; it does not license a treatment.

## 6. What this does not authorize

- Training, λ changes, T1 resume, S3, binding-sidecar enablement
- Promoting a checkpoint, opening TEST, moving Gate L/C/R
- Treating a high query-tracking mass at gap ≤ 1 as evidence of long-gap transport

## 7. Outputs

- `runs/query_presence_d1/QUERY_PRESENCE.json` — summaries (git)
- `runs/query_presence_d1/ADJUDICATION.json` — frozen rules + verdict (git)
- `runs/query_presence_d1/ROWS_*.json` — local-only pair dumps, hashed
- `research/V010_QUERY_PRESENCE_D1.md` — written after the run

## 8. Frozen

These rules are not to be changed after seeing D1 numbers.
