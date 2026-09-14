# Post-T28 tokenizer diagnostic

Status: **COMPLETE — `TOKENIZER_ASSOCIATION_SUPPORTED`**.

This was a DEV-only, read-only analysis of 128 rows across T28 seeds
850001–850003 (384 seed-row observations). No protected material was loaded.

## Main result

Targets whose first answer token was shared with another DEV name were much less
likely to generate exactly: **9/192 (4.7%)**, versus **102/192 (53.1%)** for
first-token-unique targets. The exact-success odds ratio was **0.043**. The
first-token-correct-then-diverge rate was **43.2%** with a shared prefix versus
**21.4%** without one. This large association replicated within a population in
which every answer name was multi-token.

| Name-token length | Observations | Exact | First-correct then diverge |
|---:|---:|---:|---:|
| 2 | 144 | 34.7% | 52.8% |
| 3 | 192 | 30.7% | 4.7% |
| 4 | 48 | 4.2% | 81.3% |

Length is associated, but segmentation pattern matters more than length alone:
two-token Mel was 91.7% exact while two-token Wes was 0%; three-token Ross was
91.7% while three-token Omar/Opal were approximately 0%.

Only two first-token collision groups existed: Omar/Opal and Sal/Skye. Their
longest shared prefix was one token and their first identity-disambiguating token
was position 2. All other names were distinguished at position 1. Sal generated
`Salt.` in 36/48 observations; Skye generated `Sky.` in 28/48. These are therefore
replicated manifestations of the shared-prefix pattern, not isolated anecdotes.

Token/string frequency was mixed and confounded with identity. The rarest-token
quartile was only 1.0% exact, but zero-occurrence full strings split sharply:
Ross was 91.7% exact while Wes was 0% and Omar/Opal were near 0%. Skye used common
component tokens yet was only 4.2% exact. Frequency alone does not explain the
result.

Per-seed exact counts were 36, 35, and 40; first-token-correct-then-diverge counts
were 39, 40, and 45. The direction of the tokenizer association was stable across
all three checkpoints.

This establishes association, not tokenizer causality or a license to retokenize.

