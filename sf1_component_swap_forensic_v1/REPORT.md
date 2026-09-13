# SF1 component-swap forensic v1

This is a read-only 2x2x2 functional intervention at the representation immediately before final normalization. No optimizer, autograd, checkpoint mutation, locked panel, FINAL, or sacred material was used.

## Architecture and intervention

Pilot1 and SF1 are both `Treatment13Model` wrappers around `DaveLMV082`. The base path is token/position embeddings -> eight transformer blocks -> `ExplicitLayerNorm` -> an independent linear language head (weight plus bias). Input embeddings and output weights are untied by both object identity and storage checks.
The upstream source selects embeddings, positions, and blocks 0–7 from Pilot1 or SF1. The final-normalization source selects its affine weight/bias from Pilot1 or SF1. The output-head source selects its untied weight and bias from Pilot1 or SF1.

## Baseline reproduction

PPP native maximum absolute logit error: 0; SSS: 0; required tolerance: 1e-06.
Prior-artifact D3/D4/language reproduction: `{"PPP": {"language_ce_abs_error": 0.0, "new_language_ce": 3.3906604051589966, "prior_d3_max_abs_error": 0.0, "prior_d4_max_abs_error": 0.0, "prior_language_ce": 3.3906604051589966}, "SSS": {"language_ce_abs_error": 0.0, "new_language_ce": 4.65291690826416, "prior_d3_max_abs_error": 0.0, "prior_d4_max_abs_error": 0.0, "prior_language_ce": 4.65291690826416}}`.

## Factual training-item results

| condition | sequence correct | top-1 correct | exact answer+EOS | mean sequence margin | Owen-correct margin |
|---|---:|---:|---:|---:|---:|
| PPP | 9/16 | 0/16 | 0/16 | 0.0553 | -1.0477 |
| PPS | 9/16 | 0/16 | 0/16 | 0.0905 | -1.2337 |
| PSP | 9/16 | 0/16 | 0/16 | 0.0555 | -1.0532 |
| PSS | 9/16 | 0/16 | 0/16 | 0.0910 | -1.2378 |
| SPP | 12/16 | 12/16 | 12/16 | 0.3793 | -0.3448 |
| SPS | 12/16 | 12/16 | 12/16 | 0.3875 | -0.2768 |
| SSP | 12/16 | 12/16 | 12/16 | 0.3799 | -0.3422 |
| SSS | 12/16 | 12/16 | 12/16 | 0.3881 | -0.2730 |

## Unrelated-context D3 and language

| condition | four-name mass | entropy | true-next probability | aligned CE | PPL |
|---|---:|---:|---:|---:|---:|
| PPP | 0.000907 | 3.0032 | 0.1401 | 3.3907 | 29.69 |
| PPS | 0.001786 | 2.9343 | 0.1425 | 3.3887 | 29.63 |
| PSP | 0.000914 | 2.9970 | 0.1404 | 3.3913 | 29.70 |
| PSS | 0.001805 | 2.9268 | 0.1428 | 3.3899 | 29.66 |
| SPP | 0.036394 | 2.9685 | 0.0801 | 4.2769 | 72.01 |
| SPS | 0.067286 | 2.5038 | 0.0793 | 4.6416 | 103.71 |
| SSP | 0.036707 | 2.9562 | 0.0803 | 4.2845 | 72.57 |
| SSS | 0.067833 | 2.4907 | 0.0794 | 4.6529 | 104.89 |

## Classification

`MIXED_OR_AMBIGUOUS`

No single component label fits every endpoint:

- Factual training-item behavior is `UPSTREAM_DOMINANT`. All four P-upstream conditions remained at 9/16 sequence correctness, 0/16 full-vocabulary top-1, and 0/16 exact answer+EOS. All four S-upstream conditions reached 12/16 on all three measures, independent of final-norm or head source. SPP recovered 97.36% of the PPP-to-SSS mean sequence-margin shift.
- The unrelated-context name prior is a `DISTRIBUTED_INTERACTION`. S upstream with the P norm/head raised four-name mass from 0.000907 to 0.036394 (53.02% of the PPP-to-SSS shift). The S head alone on P upstream raised it only to 0.001786 (1.31%), while the S head paired with S upstream raised it further to 0.067286/0.067833. The factorial upstream main effect was 0.050702, head main effect 0.015947, and upstream-by-head interaction 0.015062.
- The language regression is `UPSTREAM_DOMINANT_WITH_MATERIAL_HEAD_INTERACTION`. SPP aligned CE was 4.2769 versus PPP 3.3907 and SSS 4.6529, recovering 70.21% of the regression. The S head alone on P upstream slightly improved CE to 3.3887, but with S upstream it increased CE from 4.2769 to 4.6416.
- Final normalization: `FINAL_NORM_NOT_MATERIAL_ON_TESTED_ENDPOINTS`. Norm-only substitutions changed factual mean sequence margin by about 0.00025, D3 four-name mass by about 0.000006, and aligned CE by about 0.00059 on P upstream; corresponding factorial norm effects remained very small.
- Owen Stage-B attribution is `MIXED_OR_AMBIGUOUS`. Changing only upstream PPP-to-SPP improved the mean Owen-correct sequence margin from -1.0477 to -0.3448. Adding the S head with S upstream improved it further to -0.2768, but all S-upstream hybrids remained 0/4 Owen-correct. The S output path is therefore not required for the increased Owen signal and is not established as its suppressor; the residual Alex preference was not localized by this intervention.

The factorial effect table and single-component recovery fractions are in `FACTORIAL_EFFECTS.json` and `CLASSIFICATION.json`.

## Limitations

- Factual results are restricted to the 16 SF1 training records and do not establish held-out relational generalization.
- Functional hybrids isolate endpoint components, but they do not identify the training-time causal sequence that produced those components.
- Nonlinear softmax probabilities can make main effects and interactions metric-dependent; logits, margins, probabilities, and CE are all preserved.
- No gradient/Adam diagnostic or component training was performed.

SF1_COMPONENT_SWAP_FORENSIC_COMPLETE
