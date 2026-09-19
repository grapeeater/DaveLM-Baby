# P11 decode-time gen-index diagnostic

Protocol id: `V010_SELECTION_REPAIR_P11_DECODE`
Status: **preregistration — frozen before any decode-time measurement**

Licensed by P11 **SUCCESS** on the pack-path inventory endpoint. That
endpoint does not set `gen_index` during `score_items` greedy decode or
`evaluate_panels`. This diagnostic applies the same gen-only overwrite at
the last prefix token on every forward, then measures greedy generation,
query-follow, and a pairing counterfactual on frozen S2 long-gap rows.

No training. No TEST. Does not reopen P11 gates.

## 1. Hypotheses

**H1 (greedy).** With decode-time gen-index, long-gap `free_exact` for the
P11 treatment overwrite exceeds parent/init overwrite by ≥ +0.10, and the
treatment−init bootstrap 95% CI lower bound is > 0.

**H2 (query follow).** Changing only the query, inventory/body fixed,
changes the greedy first token toward the value of the new query more
often than init overwrite (follow rate Δ ≥ +0.10).

**H3 (pairing consistency).** On multi-query long-gap bodies, the fraction
of query-pairs where greedy first tokens equal the two distinct gold
heads is higher for treatment than init (Δ ≥ +0.10). This is a
within-body QUERY→VALUE follow check that does not require splicing.

If H1 fails, P11 SUCCESS is pack-path logit ranking only, not greedy
generation. That does not un-SUCCESS P11; it blocks mission A/B/C.

## 2. Arms

- **treatment:** `runs/selection_p11/treatment_250001/checkpoint_16800.pt`
  overwrite weights, Baby U16000 frozen.
- **init:** fresh `LocalSlotOverwrite(gen_only=True)` on the same Baby
  (decode-time control).

Same S2 diagnostic rows. Same bootstrap seed `250300`.

## 3. Frozen bars

H1 pass: treatment free_exact excess vs chance is not required; required
is treatment − init free_exact ≥ 0.10 and CI lo > 0, n = long-gap 215.

H2/H3 are reported but do not alter P11 SUCCESS.

## 4. TEST / promotion

Closed. Not a promotion.
