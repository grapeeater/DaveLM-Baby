# Champion vs bounded localizer internal-mechanism diagnostic

## Scope and provenance

This was the prospectively authorized frozen inference diagnostic. Both checkpoints were run once on the same 320-document retention pool in `eval()`/`inference_mode()` with all parameters non-trainable. No optimizer was created, no update was taken, and no treatment or behavioral result was changed.

Champion checkpoint SHA-256: `0d5bf015a303ada49e6f3ab0ff3d574ea57ddcc18505cab29c2715267841e4b4`  
Bounded checkpoint SHA-256: `a4ac52d1f24ad253d0a9a475891bf326e4b429f23d872b981968d797521601f2`  
Retention pool SHA-256: `29dc566e90df7c9a592014d1d0cf60dd5da654d570096c96d0c7287bcd5f9072`

## Reproduction gate

The frozen inference exactly reproduced:

| Model | BOTH_DISTINCT | EXACTLY_ONE | SLOT_COLLAPSE | NEITHER | Answer exact |
|---|---:|---:|---:|---:|---:|
| Champion | 290/320 | 30/320 | 0/320 | 0/320 | 307/320 |
| Bounded | 276/320 | 12/320 | 32/320 | 0/320 | 292/320 |

The corrected bounded physical-order census reproduced later 24 / earlier 8. Matched transitions reproduced: BD→BD 254, EO→BD 22, EO→collapse 8, BD→EO 12, BD→collapse 24.

## Saved internal quantities

`INTERNAL_MEASUREMENTS.pt` contains per-document metadata plus candidate-position hidden states, norms, localization logits/probabilities/ranks, bounded `S`, `a`, `tanh(a)`, `R`, answer hidden states, and retrieved vectors. `RESULTS.json` contains aggregate and per-document scalar records. No retention tokens were decoded beyond the already-authorized model inputs.

## Collapse decomposition

For every one of the 32 bounded collapse documents, both slots preferred the collapsed true source `x` over the other true source `y`; the two direct slot-logit differences had the same positive sign in 32/32 cases. The shared difference `Delta_S=S_x-S_y` was positive in 32/32 and exceeded `|Delta_R|` in 32/32. Means:

| Quantity | Collapse (n=32) | BOTH_DISTINCT (n=276) |
|---|---:|---:|
| `|Delta_S|` mean | 0.3178 | 0.2204 |
| `|Delta_R|` mean | 0.0696 | 2.8423 |
| `|Delta_S|/(|Delta_R|+eps)` mean | 5.698 | 0.102 |

Thus collapse is directly characterized by shared ordering plus very small residual disagreement between the two true sources.

### Rho saturation

The maximum true-source `|R|/rho` was near 0.90 in 32/32 collapse, 270/276 BOTH_DISTINCT, and 12/12 EXACTLY_ONE; it was at least 0.99 in 32/32 collapse, 256/276 BOTH_DISTINCT, and 12/12 EXACTLY_ONE. Collapse means: max ratio 0.99965, `|Delta_R|` 0.0696. BOTH_DISTINCT means: max ratio 0.99520, `|Delta_R|` 2.8423. Individual residuals are therefore saturated in collapse, but saturation is common in successful documents; the discriminating feature is that both sources land on nearly the same saturated residual, leaving little slot disagreement.

## Physical order and champion ownership

The 24 later-source and 8 earlier-source collapse cases are exactly the same 24/8 split as champion ownership:

| Bounded collapse target | Previously champion slot 0 owned | Previously champion slot 1 owned |
|---|---:|---:|
| Earlier | 0 | 8 |
| Later | 24 | 0 |

The two factors are perfectly confounded in these 32 cases; this sample cannot separate a relational order effect from inherited slot ownership. Collapse layouts include p20_b8 (8), p16_b12 (8), p24_b10 (4), p32_b12 (4), p24_b8 (4), and p32_b8 (4); source separation is 12 throughout. Collapse coordinates are quartet-stable.

## Champion-success → bounded-collapse comparison

The 24 champion BOTH_DISTINCT documents that became bounded collapse had a champion scorer disagreement that was erased by the bounded parameterization. Champion score margins for the future collapse target minus the other source were, under slot 0, mean `+2.415` and, under slot 1, mean `−2.643`; neither independent scorer preferred the future target in both directions (0/24). By contrast, the bounded scorer made both slots prefer it in 24/24. This is direct evidence that useful independent scorer disagreement was removed.

## Champion EXACTLY_ONE → bounded outcomes

Of the 30 champion EXACTLY_ONE documents, 22 became BOTH_DISTINCT and 8 became SLOT_COLLAPSE. The old wandering scorer’s missed-source score was much lower in the 8 converted cases (mean −0.531) than in the 22 repaired cases (mean 4.374), while the opposite scorer recognized that source strongly (means 8.893 versus 4.513). This supports a scorer-geometry association, not a uniform repair.

## Representation shift

Final post-LayerNorm source representations changed substantially in absolute terms between independently trained models (typical same-document source cosine about 0.52; mean L2 distance about 17). However, transition-group means were similar for BD→BD (cos 0.527), BD→collapse (0.528), BD→EO (0.519), and EO→BD (0.516). EO→collapse had higher cosine 0.710 and lower L2 13.70. No failure-specific representation separation is established; the saved distributions point more strongly to scorer geometry than to a distinct source-representation collapse.

## Offline scorer-only cross-evaluation

Using saved hidden states without feeding hybrid scores through the model:

- Champion scorer on bounded hidden states: 306 BOTH_DISTINCT, 14 EXACTLY_ONE, 0 collapse.
- Bounded scorer on champion hidden states: 208 BOTH_DISTINCT, 12 EXACTLY_ONE, 100 collapse.

These hybrids are diagnostic only. The first strongly restores distinct localization, and the second induces collapse on otherwise champion representations, establishing that the bounded scorer itself is sufficient to create the assignment failure, while representation shift may modulate it.

## Seven bounded BOTH_DISTINCT answer errors

All seven are listed in `ANALYSIS.json`/`RESULTS.json`. Six selected the query-correct row but emitted the wrong answer; their target–distractor margins were negative (predictions were usually distractor token 583 or token 1002). One case selected the non-query row. The saved audit does not expose enough value-path decomposition to distinguish retrieval residual from answer-head competition for the six; those remain unresolved downstream errors rather than localization errors.

## Hypothesis adjudication

- **H1 shared-score dominance: SUPPORTED.** `|Delta_S|>|Delta_R|` and both-slot same-sign preference held 32/32 collapses.
- **H2 antisymmetric saturation/insufficient range: SUPPORTED as a contributor, not sufficient alone.** Collapse residuals were saturated and nearly equal across sources, but many successful cases also saturated.
- **H3 slot-ownership/scorer asymmetry: SUPPORTED.** Physical order and inherited champion ownership are perfectly confounded in collapse cases; hybrid tests show scorer-specific causality.
- **H4 representation shift: INSUFFICIENT as the primary explanation.** Representations moved globally, but no transition-specific discriminator was found.
- **H5 mixed mechanism: SUPPORTED.** Shared ordering, bounded residual disagreement, and inherited scorer ownership jointly characterize the collapse; downstream answer errors are a separate residual.

## Final diagnosis

**ESTABLISHED:** The bounded model both repaired wandering and converted failures into legitimate-source collapse. Collapse occurs when the shared score orders one source above the other and the bounded residual supplies little differential disagreement; both slots then choose the same source. The 24/32 later pattern is exactly confounded with champion slot-0 ownership. The bounded scorer induces collapse even on champion hidden states, while the champion scorer largely restores distinct localization on bounded hidden states.

**SUPPORTED:** The frozen bounded-shared-antisymmetric hypothesis, as instantiated with `rho=1.5919504166`, created an assignment-capacity bottleneck by constraining useful slot-specific disagreement. This is not evidence of a general architectural impossibility.

**UNRESOLVED:** A unique causal split between physical order and inherited slot ownership; whether tanh saturation versus shared-score ordering is the dominant parameter-level cause; and the exact value-retrieval/answer-head source of six conditional answer errors.

No treatment, rho change, sweep, continuation, or follow-up experiment was designed or run.
