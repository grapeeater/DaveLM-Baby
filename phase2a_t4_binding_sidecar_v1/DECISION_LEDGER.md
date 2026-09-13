# T4 DECISION — binding sidecar on Phase1G parent

T3 rebuilt study is terminal: `T3_LANGUAGE_REGRESSION` (3/3). Retrieval rose (97–102/128) but reversals/families/exact failed and DEV CE crossed 1.30. T3 TEST stays sealed. T3 U500 weights are **not** the parent (language regression).

Highest-information next step: train the dormant 984,321-param OrthoLocalizer + retrieval head with an explicit `BindingLayout` on the same disjoint T3 train/DEV, **base LM frozen**, parent = Phase1G U6000 `c5406f80…`.

Hypothesis: query-conditioned slot routing and answer-position splice can produce reversal-complete native selection without destroying Phase1G language. T3 already showed base-only pointer/margin cannot.

Not authorized: T3 TEST, T2-EVAL-TEST, FINAL, sacred, T3 descendant fine-tunes, ~100M scale.
