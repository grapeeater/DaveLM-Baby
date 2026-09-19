# P11 first-step-only decode

Protocol id: `V010_SELECTION_REPAIR_P11_DECODE_FIRSTSTEP`
Status: **preregistration — frozen before measurement**

Licensed by P11_DECODE **H1_FAIL**. Decode-time overwrite at *every*
greedy step made first-token gold-rate 73→105 but free_exact 71→0:
later value tokens were also overwritten. This diagnostic applies
gen-only overwrite on **step 0 only**, then identity for the rest of
the span.

No training. No TEST. Does not reopen P11 gates.

## 1. Hypotheses

**H1.** First-step-only overwrite recovers long-gap `free_exact` so
treatment − init ≥ +0.10 with bootstrap CI lo > 0 (n=215).

**H2.** `first_correct` still ≥ init + 0.10 (regression check vs
P11_DECODE first-token lift).

**H3.** Multi-query pair-bind rate Δ vs init ≥ +0.10.

## 2. Arms / rows

Same as P11_DECODE: treatment checkpoint_16800 overwrite vs init
overwrite, S2 long-gap rows, bootstrap seed 250300.

## 3. Frozen bars

H1_PASS iff free_exact Δ ≥ 0.10 and CI lo > 0.
H2/H3 reported. Failure does not un-SUCCESS P11.

## 4. TEST / promotion

Closed. Not a promotion.
