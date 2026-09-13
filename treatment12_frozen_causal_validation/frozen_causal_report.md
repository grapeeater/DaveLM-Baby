# Treatment-12 frozen causal validation report

Bounded audit. No training. Frozen checkpoint and frozen retention set only.

## Provenance

- retention pool sha256: `c341d7308b145bd3c633b62d56e01391f4b05635bfcf2edfb146bcd9f2de4e69`
- final T12 checkpoint sha256: `e378d3d85f3a53add4046aefc7cc42d643f59d12525859af6ea79dbfca266cd5`
- documents: 160, quartets: 40

## Baseline reproduction (Condition A)

passed: True
- condition: A
- n: 160
- answer_accuracy: 1.0
- predictions_equal_target_count: 160
- predictions_equal_distractor_count: 0
- target_gt_distractor_rate: 1.0
- mean_answer_margin: 13.184189174324274
- median_answer_margin: 12.976151943206787
- mean_candidate_mass: 0.9999769378453494
- complete_quartet_success_count: 40
- complete_quartet_success_rate: 1.0
- retrieval_argmax_matches_correct_row_rate: 1
- mean_correct_row_attention_weight: 0.9999392062425614
- mean_incorrect_row_attention_weight: 6.079539714311383e-05
- mean_retrieval_score_margin: 12.107150509953499

## Geometry census findings

- unique layout signatures: train=20, retention=20, retention-not-in-train=0
- retention has genuinely unseen layout geometry: False
- all absolute q/k/v/answer positions reused in retention: True
- relative-distance combinations reused in retention: True
- query_slot deterministic given layout: False; orientation deterministic given layout: False; target row deterministic given layout: False

## Intervention definitions

- **A**: normal frozen T12 forward pass
- **B**: uniform row routing alpha=[0.5,0.5]; original scores reported for diagnostics
- **C**: normal scores/alpha but zero Wo(retrieved) residual (answer_hidden only)
- **D**: alpha one-hot forced to the WRONG row (correct row used only to build the test)
- **E**: in-distribution q activation swap from matched same-orientation query twin

## Results table

| condition | answer acc | tgt>dist | mean margin | quartet succ | retr row acc | corr weight |
|---|---|---|---|---|---|---|
| A | 1.0000 | 1.0000 | 13.1842 | 40/40 | 1.0000 | 0.9999 |
| B | 0.5062 | 0.5062 | -0.0003 | 0/40 | 1.0000 | 0.5000 |
| C | 0.3875 | 0.4938 | -0.0003 | 0/40 | 1.0000 | 0.9999 |
| D | 0.0000 | 0.0000 | -13.1865 | 0/40 | 1.0000 | 0.0000 |

## Per-paired A->D change analysis

- A->D prediction changed: 160/160
- D predictions equal distractor: 160/160
- margin sign inverted A->D: 160/160
- A->B prediction changed: 79/160
- A->C prediction changed: 98/160

## Query-swap (E)

- pairs: 160
- routing-flip rate: 1.0000
- predicted-direction routing-flip rate: 1.0000
- answer-flip rate: 1.0000
- answer follows patched query's local mapping rate: 1.0000

## Preregistered interpretation logic

- **A**: Baseline must reproduce the authoritative 160/160 retention result (provenance check).
- **B**: Uniform alpha causing large degradation => learned row selection is causally important.
- **C**: Zero retrieved residual causing large degradation while retrieval stays accurate => the value-transfer residual is causally important beyond row scoring.
- **D**: Forced wrong row shifting answers strongly toward the distractor => row routing controls answer identity in the predicted direction.
- **E**: Query swap flipping row preference in the predicted direction => the query representation governs row selection.
- **general**: If an intervention does not behave as expected, report it plainly and identify which conclusion is weakened. No thresholds were tuned after outcomes. Emphasis is on effect sizes and predicted-direction paired changes.

## Causal question markers

- uniform_alpha: PASS
- zero_retrieval: PASS
- force_wrong_row: PASS
- query_swap: PASS

## Prohibited overclaims

- Transformers cannot bind without T12
- T11 proved native self-attention is incapable
- T12 solved general variable binding
- T12 proves Baby can reason
- T12's architecture is universally necessary
- T12 can search arbitrary documents
- Fixed/absolute positional embeddings caused T9/T11 failure
- Structural assistance is cheating

## Notes

- T12 remains structurally assisted (q/k/v/answer positions are grammar-supplied).
- No threshold was tuned after inspecting intervention outcomes.

## Strongest supported conclusion

Frozen causal interventions support that T12's query-conditioned row selection and retrieved-value transfer are causally responsible for its held-out strict-counterfactual binding behavior. Directly steering the learned retrieval pathway changes row selection and answer identity in the predicted direction: uniform row routing collapses answer accuracy from 1.0 to 0.506, zeroing the retrieved residual collapses it to 0.388 while retrieval-row accuracy stays 1.0, forcing the wrong row drives all 160 predictions to the distractor with mirrored margins (0 target accuracy), and patching the query representation from the matched query twin flips routing and answers in the predicted direction 160/160 times. T12 remains structurally assisted; no claim of general binding or architectural universality is made.

## Notes for prospective work

- No T13 design is proposed here. This audit bounds T12 evidence; any architectural implications should be evaluated as a separate preregistered treatment.
