# Frozen English context-sensitivity characterization

This report characterizes the three specified checkpoints on the identical sealed battery. Raw complete-family counts/proportions, reversal profiles, and likelihood-margin summaries are the headline evidence. No pass/fail gates were added.

The English scores use only the ordinary `base_model` causal-LM path. The separate synthetic reference uses the existing structurally assisted binding path. All candidate comparisons use the full leading-space word plus period; priors are reported without calibration. Exact ties count as incorrect.

These observations concern the frozen naming, having/association, and location constructions. They do not measure general English capability, establish conversational competence, isolate block-protection effects, establish synthetic-to-English transfer, or prove an architectural/capacity limit.

## Headline primary evidence

Possession and location are separate endpoints. Possession queries an object to recover a person; location queries a person to recover a place. Differences can reflect query direction and wording as well as relation content.

Reversal profile entries below count families with 0, 1, 2, 3, or 4 successful reversal pairs, in that order. Complete-family success requires all eight items correct, including both fact orders.

### Possession primary cloze

| Checkpoint | Complete families | Proportion | Reversal profile [0,1,2,3,4] | Both-correct reversal pairs | Mean family reversal rate | Median family mean margin | Median family minimum margin |
|---|---:|---:|---|---:|---:|---:|---:|
| Graduate | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | -5.8189e-06 | -0.314233 |
| Pilot 0 | 0/16 | 0.00% | [15, 1, 0, 0, 0] | 1/64 | 1.56% | -0.000407933 | -1.33935 |
| Pilot 1 | 0/16 | 0.00% | [15, 1, 0, 0, 0] | 1/64 | 1.56% | -0.00154696 | -1.49933 |

### Location primary cloze

| Checkpoint | Complete families | Proportion | Reversal profile [0,1,2,3,4] | Both-correct reversal pairs | Mean family reversal rate | Median family mean margin | Median family minimum margin |
|---|---:|---:|---|---:|---:|---:|---:|
| Graduate | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | 8.15835e-06 | -0.492647 |
| Pilot 0 | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | -0.000617499 | -2.84732 |
| Pilot 1 | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | -0.000467715 | -0.618548 |

## Separate diagnostic strata

Naming is diagnostic and does not gate later strata. QA is scored separately from cloze. Alternate frames measure robustness to those specified query frames; their vocabulary is not claimed unseen during TinyStories training. No lexical holdout or distractor condition was added.

### Naming diagnostic

| Checkpoint | Complete families | Proportion | Reversal profile [0,1,2,3,4] | Both-correct reversal pairs | Mean family reversal rate | Median family mean margin | Median family minimum margin |
|---|---:|---:|---|---:|---:|---:|---:|
| Graduate | 0/8 | 0.00% | [8, 0, 0, 0, 0] | 0/32 | 0.00% | -2.48918e-05 | -0.491985 |
| Pilot 0 | 0/8 | 0.00% | [8, 0, 0, 0, 0] | 0/32 | 0.00% | -0.000396289 | -1.70536 |
| Pilot 1 | 0/8 | 0.00% | [7, 0, 1, 0, 0] | 2/32 | 6.25% | -0.000374272 | -1.42328 |

### Possession alternate-frame diagnostic

| Checkpoint | Complete families | Proportion | Reversal profile [0,1,2,3,4] | Both-correct reversal pairs | Mean family reversal rate | Median family mean margin | Median family minimum margin |
|---|---:|---:|---|---:|---:|---:|---:|
| Graduate | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | -7.04855e-06 | -0.515251 |
| Pilot 0 | 0/16 | 0.00% | [15, 1, 0, 0, 0] | 1/64 | 1.56% | 0.00316291 | -0.955377 |
| Pilot 1 | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | 0.000238609 | -1.95772 |

### Location alternate-frame diagnostic

| Checkpoint | Complete families | Proportion | Reversal profile [0,1,2,3,4] | Both-correct reversal pairs | Mean family reversal rate | Median family mean margin | Median family minimum margin |
|---|---:|---:|---|---:|---:|---:|---:|
| Graduate | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | 5.99372e-06 | -0.579981 |
| Pilot 0 | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | 0.000488223 | -2.76166 |
| Pilot 1 | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | -2.69185e-05 | -0.674101 |

### Possession QA diagnostic

| Checkpoint | Complete families | Proportion | Reversal profile [0,1,2,3,4] | Both-correct reversal pairs | Mean family reversal rate | Median family mean margin | Median family minimum margin |
|---|---:|---:|---|---:|---:|---:|---:|
| Graduate | 0/16 | 0.00% | [14, 2, 0, 0, 0] | 2/64 | 3.12% | 2.58996e-06 | -0.0286806 |
| Pilot 0 | 0/16 | 0.00% | [15, 1, 0, 0, 0] | 1/64 | 1.56% | -0.00121346 | -1.90327 |
| Pilot 1 | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | -0.000625454 | -1.5002 |

### Location QA diagnostic

| Checkpoint | Complete families | Proportion | Reversal profile [0,1,2,3,4] | Both-correct reversal pairs | Mean family reversal rate | Median family mean margin | Median family minimum margin |
|---|---:|---:|---|---:|---:|---:|---:|
| Graduate | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | 8.82378e-07 | -0.534734 |
| Pilot 0 | 0/16 | 0.00% | [15, 1, 0, 0, 0] | 1/64 | 1.56% | -0.000196027 | -3.50495 |
| Pilot 1 | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | -9.23327e-05 | -3.79689 |

## Completion-boundary, item, and mass diagnostics

The word-only diagnostic excludes the period and retains both word tokens. Headline scores remain the full completed-answer scores. Any disagreement is explicitly retained as completion-boundary sensitivity; it is not used to choose a preferred scoring method after seeing results.

| Stratum | Checkpoint | Full item correct | Full ties | Word-only item correct | Word ties | Full-only correct / word-only correct | Complete families full / word | Median pair mass | Median pooled candidate LL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| possession/core | Graduate | 64/128 | 0 | 64/128 | 0 | 16 / 16 | 0 / 0 | 6.43939e-09 | -19.6443 |
| possession/core | Pilot 0 | 65/128 | 0 | 64/128 | 0 | 1 / 0 | 0 / 0 | 0.0010428 | -7.46597 |
| possession/core | Pilot 1 | 64/128 | 0 | 64/128 | 0 | 15 / 15 | 0 / 0 | 0.000262385 | -8.95522 |
| location/core | Graduate | 64/128 | 0 | 64/128 | 0 | 24 / 24 | 0 / 0 | 3.0641e-09 | -20.1913 |
| location/core | Pilot 0 | 64/128 | 0 | 64/128 | 0 | 0 / 0 | 0 / 0 | 0.00123735 | -7.76499 |
| location/core | Pilot 1 | 62/128 | 0 | 64/128 | 0 | 11 / 13 | 0 / 0 | 5.84024e-05 | -10.5209 |
| naming/core | Graduate | 32/64 | 0 | 32/64 | 0 | 16 / 16 | 0 / 0 | 9.81902e-09 | -19.1973 |
| naming/core | Pilot 0 | 32/64 | 0 | 33/64 | 0 | 5 / 6 | 0 / 0 | 0.00137488 | -7.13682 |
| naming/core | Pilot 1 | 32/64 | 0 | 33/64 | 0 | 2 / 3 | 0 / 0 | 0.000198991 | -9.19388 |
| possession/frame_holdout | Graduate | 64/128 | 0 | 64/128 | 0 | 12 / 12 | 0 / 0 | 9.35516e-09 | -19.2848 |
| possession/frame_holdout | Pilot 0 | 63/128 | 0 | 66/128 | 0 | 9 / 12 | 0 / 0 | 0.00859986 | -5.41006 |
| possession/frame_holdout | Pilot 1 | 64/128 | 0 | 65/128 | 0 | 6 / 7 | 0 / 0 | 0.00160954 | -7.57866 |
| location/frame_holdout | Graduate | 64/128 | 0 | 64/128 | 0 | 24 / 24 | 0 / 0 | 3.44319e-09 | -19.9943 |
| location/frame_holdout | Pilot 0 | 64/128 | 0 | 64/128 | 0 | 0 / 0 | 0 / 0 | 0.00195536 | -7.91097 |
| location/frame_holdout | Pilot 1 | 64/128 | 0 | 64/128 | 0 | 20 / 20 | 0 / 0 | 7.056e-05 | -10.3095 |
| possession/qa | Graduate | 59/128 | 0 | 63/128 | 0 | 31 / 35 | 0 / 0 | 5.11435e-09 | -19.7829 |
| possession/qa | Pilot 0 | 64/128 | 0 | 66/128 | 0 | 17 / 19 | 0 / 0 | 0.000287048 | -9.1009 |
| possession/qa | Pilot 1 | 64/128 | 0 | 64/128 | 0 | 4 / 4 | 0 / 0 | 0.00145115 | -7.65147 |
| location/qa | Graduate | 63/128 | 0 | 64/128 | 0 | 12 / 13 | 0 / 0 | 3.1968e-09 | -20.1102 |
| location/qa | Pilot 0 | 64/128 | 0 | 64/128 | 0 | 3 / 3 | 0 / 0 | 3.20407e-07 | -15.6771 |
| location/qa | Pilot 1 | 64/128 | 0 | 64/128 | 0 | 12 / 12 | 0 / 0 | 1.23278e-05 | -13.8458 |

Candidate-pair mass is the sum of probabilities of the two complete sequences. A preference within this pair does not establish that either answer would be freely generated. Exact candidate and per-token log-likelihoods are preserved in [RAW_SCORES.jsonl](RAW_SCORES.jsonl); distributions and word-only reversal/margin profiles are in [SUMMARY.json](SUMMARY.json).

### Word-only family and margin profiles

### Possession primary cloze — word-only

| Checkpoint | Complete families | Proportion | Reversal profile [0,1,2,3,4] | Both-correct reversal pairs | Mean family reversal rate | Median family mean margin | Median family minimum margin |
|---|---:|---:|---|---:|---:|---:|---:|
| Graduate | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | 2.2296e-06 | -0.327858 |
| Pilot 0 | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | -0.000516237 | -1.35977 |
| Pilot 1 | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | 0.000160654 | -1.63226 |

### Location primary cloze — word-only

| Checkpoint | Complete families | Proportion | Reversal profile [0,1,2,3,4] | Both-correct reversal pairs | Mean family reversal rate | Median family mean margin | Median family minimum margin |
|---|---:|---:|---|---:|---:|---:|---:|
| Graduate | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | -1.71439e-05 | -0.482481 |
| Pilot 0 | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | 4.34838e-05 | -1.79103 |
| Pilot 1 | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | -0.000494327 | -1.71875 |

### Naming diagnostic — word-only

| Checkpoint | Complete families | Proportion | Reversal profile [0,1,2,3,4] | Both-correct reversal pairs | Mean family reversal rate | Median family mean margin | Median family minimum margin |
|---|---:|---:|---|---:|---:|---:|---:|
| Graduate | 0/8 | 0.00% | [8, 0, 0, 0, 0] | 0/32 | 0.00% | 4.4033e-06 | -0.343203 |
| Pilot 0 | 0/8 | 0.00% | [7, 1, 0, 0, 0] | 1/32 | 3.12% | 0.000187067 | -1.51746 |
| Pilot 1 | 0/8 | 0.00% | [7, 1, 0, 0, 0] | 1/32 | 3.12% | 0.000505501 | -1.57233 |

### Possession alternate-frame diagnostic — word-only

| Checkpoint | Complete families | Proportion | Reversal profile [0,1,2,3,4] | Both-correct reversal pairs | Mean family reversal rate | Median family mean margin | Median family minimum margin |
|---|---:|---:|---|---:|---:|---:|---:|
| Graduate | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | 4.07357e-06 | -0.40261 |
| Pilot 0 | 0/16 | 0.00% | [14, 2, 0, 0, 0] | 2/64 | 3.12% | 0.00236419 | -1.06011 |
| Pilot 1 | 0/16 | 0.00% | [15, 1, 0, 0, 0] | 1/64 | 1.56% | 0.000618396 | -2.28209 |

### Location alternate-frame diagnostic — word-only

| Checkpoint | Complete families | Proportion | Reversal profile [0,1,2,3,4] | Both-correct reversal pairs | Mean family reversal rate | Median family mean margin | Median family minimum margin |
|---|---:|---:|---|---:|---:|---:|---:|
| Graduate | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | 3.89983e-07 | -0.422379 |
| Pilot 0 | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | 0.000138973 | -1.90485 |
| Pilot 1 | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | -0.000143419 | -1.82907 |

### Possession QA diagnostic — word-only

| Checkpoint | Complete families | Proportion | Reversal profile [0,1,2,3,4] | Both-correct reversal pairs | Mean family reversal rate | Median family mean margin | Median family minimum margin |
|---|---:|---:|---|---:|---:|---:|---:|
| Graduate | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | 7.00193e-06 | -0.311907 |
| Pilot 0 | 0/16 | 0.00% | [10, 6, 0, 0, 0] | 6/64 | 9.38% | -0.00129227 | -1.01949 |
| Pilot 1 | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | 0.00121985 | -1.45582 |

### Location QA diagnostic — word-only

| Checkpoint | Complete families | Proportion | Reversal profile [0,1,2,3,4] | Both-correct reversal pairs | Mean family reversal rate | Median family mean margin | Median family minimum margin |
|---|---:|---:|---|---:|---:|---:|---:|
| Graduate | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | -4.43729e-06 | -0.399529 |
| Pilot 0 | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | 0.000558186 | -1.45928 |
| Pilot 1 | 0/16 | 0.00% | [16, 0, 0, 0, 0] | 0/64 | 0.00% | -0.000300774 | -2.02837 |

## Query-only priors and matched contextual modulation

Candidate 0/1 refers to each family's fixed candidate ordering, not a universal word. Query-only priors change prefix length and position as well as removing facts. They are diagnostic; no prior was subtracted from a contextual margin.

| Stratum | Checkpoint | Prior preferences c0 / c1 / tie | Median prior d | Median absolute prior d | Context/prior preference agreement | Median answer-oriented reversal change | Positive changes |
|---|---|---:|---:|---:|---:|---:|---:|
| possession/core | Graduate | 12 / 20 / 0 | -0.155197 | 0.347789 | 128/128 | 4.64003e-05 | 32/64 |
| possession/core | Pilot 0 | 25 / 7 / 0 | 0.977462 | 1.28706 | 101/128 | -0.00780164 | 29/64 |
| possession/core | Pilot 1 | 12 / 20 / 0 | -0.762439 | 0.904029 | 108/128 | -0.0088517 | 29/64 |
| location/core | Graduate | 26 / 6 / 0 | 0.492773 | 0.492773 | 104/128 | 0.000144611 | 34/64 |
| location/core | Pilot 0 | 11 / 21 / 0 | -1.41247 | 2.95597 | 108/128 | -0.0030588 | 31/64 |
| location/core | Pilot 1 | 16 / 16 / 0 | -0.634715 | 1.79872 | 112/128 | 0.002481 | 32/64 |
| naming/core | Graduate | 12 / 4 / 0 | 0.426545 | 0.426545 | 48/64 | 0.000647743 | 16/32 |
| naming/core | Pilot 0 | 12 / 4 / 0 | 2.07754 | 2.07754 | 60/64 | 0.00474193 | 19/32 |
| naming/core | Pilot 1 | 10 / 6 / 0 | 0.795948 | 0.979986 | 60/64 | 0.00430524 | 16/32 |
| possession/frame_holdout | Graduate | 12 / 20 / 0 | -0.428653 | 0.438503 | 104/128 | 4.27815e-05 | 32/64 |
| possession/frame_holdout | Pilot 0 | 25 / 7 / 0 | 0.92185 | 1.06157 | 47/128 | 0.000158961 | 32/64 |
| possession/frame_holdout | Pilot 1 | 7 / 25 / 0 | -1.10463 | 2.28325 | 116/128 | -0.00443442 | 31/64 |
| location/frame_holdout | Graduate | 32 / 0 / 0 | 0.724974 | 0.724974 | 80/128 | -0.000360478 | 32/64 |
| location/frame_holdout | Pilot 0 | 6 / 26 / 0 | -2.40491 | 3.20829 | 128/128 | -0.00749402 | 28/64 |
| location/frame_holdout | Pilot 1 | 16 / 16 / 0 | -0.0684504 | 0.611225 | 128/128 | 0.00193102 | 32/64 |
| possession/qa | Graduate | 12 / 20 / 0 | -0.0651756 | 0.236531 | 91/128 | 4.84323e-05 | 32/64 |
| possession/qa | Pilot 0 | 6 / 26 / 0 | -0.482559 | 0.483253 | 88/128 | -0.0091712 | 30/64 |
| possession/qa | Pilot 1 | 14 / 18 / 0 | -0.0737596 | 0.645991 | 80/128 | -0.000245876 | 32/64 |
| location/qa | Graduate | 29 / 3 / 0 | 0.523598 | 0.523598 | 93/128 | 9.39867e-05 | 33/64 |
| location/qa | Pilot 0 | 16 / 16 / 0 | -0.282711 | 2.41886 | 122/128 | 0.00700114 | 33/64 |
| location/qa | Pilot 1 | 3 / 29 / 0 | -3.3968 | 3.3968 | 116/128 | -0.00891162 | 30/64 |

For a matched pair, the raw change is d(a=0)−d(a=1). The answer-oriented change multiplies it by +1 when candidate 0 is correct at a=0 and by −1 otherwise; equivalently it is margin(a=0)+margin(a=1). A positive change supports contextual modulation, and does not by itself establish successful reversal or selection. Both raw and oriented changes, including word-only versions, are preserved per pair in SUMMARY.json.

## Order, mention, and query diagnostics

Fact order 0 means the E0 fact is presented first. Mention rank 0 means the correct candidate appears earlier among the two factual candidate mentions. Queried fact rank 0 means the queried association is presented first. Query index identifies the queried identity in the frozen pair.

| Stratum | Checkpoint | Fact-order correctness 0 / 1 | Correct-candidate mention correctness early / late | Queried-fact correctness first / second | Query correctness 0 / 1 | Families complete both orders / only 0 / only 1 / neither |
|---|---|---|---|---|---|---|
| possession/core | Graduate | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 0 / 0 / 0 / 16 |
| possession/core | Pilot 0 | 33/64 / 32/64 | 30/64 / 35/64 | 30/64 / 35/64 | 32/64 / 33/64 | 0 / 0 / 0 / 16 |
| possession/core | Pilot 1 | 32/64 / 32/64 | 27/64 / 37/64 | 27/64 / 37/64 | 33/64 / 31/64 | 0 / 0 / 0 / 16 |
| location/core | Graduate | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 0 / 0 / 0 / 16 |
| location/core | Pilot 0 | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 0 / 0 / 0 / 16 |
| location/core | Pilot 1 | 31/64 / 31/64 | 32/64 / 30/64 | 32/64 / 30/64 | 31/64 / 31/64 | 0 / 0 / 0 / 16 |
| naming/core | Graduate | 16/32 / 16/32 | 16/32 / 16/32 | 16/32 / 16/32 | 16/32 / 16/32 | 0 / 0 / 0 / 8 |
| naming/core | Pilot 0 | 16/32 / 16/32 | 16/32 / 16/32 | 16/32 / 16/32 | 16/32 / 16/32 | 0 / 0 / 0 / 8 |
| naming/core | Pilot 1 | 16/32 / 16/32 | 14/32 / 18/32 | 14/32 / 18/32 | 16/32 / 16/32 | 0 / 0 / 0 / 8 |
| possession/frame_holdout | Graduate | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 0 / 0 / 0 / 16 |
| possession/frame_holdout | Pilot 0 | 31/64 / 32/64 | 25/64 / 38/64 | 25/64 / 38/64 | 31/64 / 32/64 | 0 / 0 / 0 / 16 |
| possession/frame_holdout | Pilot 1 | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 0 / 0 / 0 / 16 |
| location/frame_holdout | Graduate | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 0 / 0 / 0 / 16 |
| location/frame_holdout | Pilot 0 | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 0 / 0 / 0 / 16 |
| location/frame_holdout | Pilot 1 | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 0 / 0 / 0 / 16 |
| possession/qa | Graduate | 29/64 / 30/64 | 28/64 / 31/64 | 28/64 / 31/64 | 29/64 / 30/64 | 0 / 0 / 0 / 16 |
| possession/qa | Pilot 0 | 32/64 / 32/64 | 28/64 / 36/64 | 28/64 / 36/64 | 31/64 / 33/64 | 0 / 0 / 0 / 16 |
| possession/qa | Pilot 1 | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 0 / 0 / 0 / 16 |
| location/qa | Graduate | 31/64 / 32/64 | 32/64 / 31/64 | 32/64 / 31/64 | 32/64 / 31/64 | 0 / 0 / 0 / 16 |
| location/qa | Pilot 0 | 31/64 / 33/64 | 32/64 / 32/64 | 32/64 / 32/64 | 31/64 / 33/64 | 0 / 0 / 0 / 16 |
| location/qa | Pilot 1 | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 32/64 / 32/64 | 0 / 0 / 0 / 16 |

These order subsets are diagnostic, with correlated members. Their full and word-only margins are retained in SUMMARY.json; no separate significance tests or new gates are introduced.

## Paired checkpoint comparisons

All differences are second checkpoint minus first on identical families. 'Both / first only / second only / neither' classifies complete-family success. The comparison is behavioral: training trajectories, degree of TinyStories adaptation, and binding material differ.

| Stratum | First → second | CF proportion difference | Both / first only / second only / neither | Median paired family-mean margin difference |
|---|---|---:|---|---:|
| possession/core | Graduate → Pilot 0 | 0.00% | 0 / 0 / 0 / 16 | -0.000404177 |
| possession/core | Graduate → Pilot 1 | 0.00% | 0 / 0 / 0 / 16 | -0.00155332 |
| possession/core | Pilot 0 → Pilot 1 | 0.00% | 0 / 0 / 0 / 16 | -0.000515727 |
| location/core | Graduate → Pilot 0 | 0.00% | 0 / 0 / 0 / 16 | -0.000621421 |
| location/core | Graduate → Pilot 1 | 0.00% | 0 / 0 / 0 / 16 | -0.000548585 |
| location/core | Pilot 0 → Pilot 1 | 0.00% | 0 / 0 / 0 / 16 | -4.18412e-05 |
| naming/core | Graduate → Pilot 0 | 0.00% | 0 / 0 / 0 / 8 | -0.000239628 |
| naming/core | Graduate → Pilot 1 | 0.00% | 0 / 0 / 0 / 8 | -0.000389288 |
| naming/core | Pilot 0 → Pilot 1 | 0.00% | 0 / 0 / 0 / 8 | -0.000872793 |
| possession/frame_holdout | Graduate → Pilot 0 | 0.00% | 0 / 0 / 0 / 16 | 0.00322487 |
| possession/frame_holdout | Graduate → Pilot 1 | 0.00% | 0 / 0 / 0 / 16 | 0.000276492 |
| possession/frame_holdout | Pilot 0 → Pilot 1 | 0.00% | 0 / 0 / 0 / 16 | -0.00576583 |
| location/frame_holdout | Graduate → Pilot 0 | 0.00% | 0 / 0 / 0 / 16 | 0.000463793 |
| location/frame_holdout | Graduate → Pilot 1 | 0.00% | 0 / 0 / 0 / 16 | -1.82276e-05 |
| location/frame_holdout | Pilot 0 → Pilot 1 | 0.00% | 0 / 0 / 0 / 16 | -0.000459323 |
| possession/qa | Graduate → Pilot 0 | 0.00% | 0 / 0 / 0 / 16 | -0.00119782 |
| possession/qa | Graduate → Pilot 1 | 0.00% | 0 / 0 / 0 / 16 | -0.000640796 |
| possession/qa | Pilot 0 → Pilot 1 | 0.00% | 0 / 0 / 0 / 16 | -0.000118511 |
| location/qa | Graduate → Pilot 0 | 0.00% | 0 / 0 / 0 / 16 | -0.000190521 |
| location/qa | Graduate → Pilot 1 | 0.00% | 0 / 0 / 0 / 16 | -7.38703e-05 |
| location/qa | Pilot 0 → Pilot 1 | 0.00% | 0 / 0 / 0 / 16 | -0.00106107 |

Per-family reversal-rate differences and paired margin differences, including word-only versions, are in SUMMARY.json. There is no pooled cloze/QA score, primary possession/location aggregate, statistical general-winner label, causal protection claim, or automatic training-parent selection.

## Synthetic-binding development references

The two previously used nonsacred DEV panels are evaluated separately and identically across checkpoints. They are development references with prior research exposure, not an untouched confirmation exam. English and binding use distinct inference paths and are not combined into an endpoint.

| Checkpoint | Reference | Answer exact | BOTH_DISTINCT | Collapse | Complete quartets | Strict reversal both-correct | Queried row given BD | Answer given BD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Graduate | pilot0_dev | 80/80 | 80/80 | 0 | 20/20 | 40/40 | 80/80 | 80/80 |
| Graduate | pilot1_dev | 79/80 | 76/80 | 0 | 19/20 | 39/40 | 76/76 | 76/76 |
| Pilot 0 | pilot0_dev | 73/80 | 76/80 | 4 | 15/20 | 34/40 | 72/76 | 71/76 |
| Pilot 0 | pilot1_dev | 73/80 | 78/80 | 2 | 15/20 | 34/40 | 74/78 | 72/78 |
| Pilot 1 | pilot0_dev | 80/80 | 80/80 | 0 | 20/20 | 40/40 | 80/80 | 80/80 |
| Pilot 1 | pilot1_dev | 80/80 | 80/80 | 0 | 20/20 | 40/40 | 80/80 | 80/80 |

Exact binding rows and provenance are retained in BINDING_REFERENCE_RESULTS.json and the execution manifest. The redundant measure that the selected row belongs to either true source is not counted as independent evidence.

## Secondary finite-universe confidence intervals

These 95% exact hypergeometric intervals concern only the explicitly enumerated eligible lexical-family universe for the specified fixed frame: naming N=18, possession N=36, location N=24. They are secondary to the raw behavioral counts, reversal profiles, and margins. They are not uncertainty over Baby's general English capability, all vocabulary, all constructions, or training-seed variation. Each complete family is the unit; its eight items are not independent trials.

| Stratum | Checkpoint | Sample CF successes | Universe N | Secondary finite-universe interval |
|---|---|---:|---:|---:|
| possession/core | Graduate | 0/16 | 36 | [0.00%, 13.89%] |
| possession/core | Pilot 0 | 0/16 | 36 | [0.00%, 13.89%] |
| possession/core | Pilot 1 | 0/16 | 36 | [0.00%, 13.89%] |
| location/core | Graduate | 0/16 | 24 | [0.00%, 12.50%] |
| location/core | Pilot 0 | 0/16 | 24 | [0.00%, 12.50%] |
| location/core | Pilot 1 | 0/16 | 24 | [0.00%, 12.50%] |
| naming/core | Graduate | 0/8 | 18 | [0.00%, 27.78%] |
| naming/core | Pilot 0 | 0/8 | 18 | [0.00%, 27.78%] |
| naming/core | Pilot 1 | 0/8 | 18 | [0.00%, 27.78%] |
| possession/frame_holdout | Graduate | 0/16 | 36 | [0.00%, 13.89%] |
| possession/frame_holdout | Pilot 0 | 0/16 | 36 | [0.00%, 13.89%] |
| possession/frame_holdout | Pilot 1 | 0/16 | 36 | [0.00%, 13.89%] |
| location/frame_holdout | Graduate | 0/16 | 24 | [0.00%, 12.50%] |
| location/frame_holdout | Pilot 0 | 0/16 | 24 | [0.00%, 12.50%] |
| location/frame_holdout | Pilot 1 | 0/16 | 24 | [0.00%, 12.50%] |
| possession/qa | Graduate | 0/16 | 36 | [0.00%, 13.89%] |
| possession/qa | Pilot 0 | 0/16 | 36 | [0.00%, 13.89%] |
| possession/qa | Pilot 1 | 0/16 | 36 | [0.00%, 13.89%] |
| location/qa | Graduate | 0/16 | 24 | [0.00%, 12.50%] |
| location/qa | Pilot 0 | 0/16 | 24 | [0.00%, 12.50%] |
| location/qa | Pilot 1 | 0/16 | 24 | [0.00%, 12.50%] |

Intervals invert both hypergeometric tails at 0.025 using exact integer/rational probabilities. No p-value grid, independent-item binomial interval, or 1/256 family-chance baseline is used.

## Shortcut references

| Stratum | Rule | Item correct | Complete families | Reversal both-correct |
|---|---|---:|---:|---:|
| possession/core | fixed_candidate0 | 64/128 | 0/16 | 0/64 |
| possession/core | first_mentioned | 64/128 | 0/16 | 0/64 |
| possession/core | last_mentioned | 64/128 | 0/16 | 0/64 |
| location/core | fixed_candidate0 | 64/128 | 0/16 | 0/64 |
| location/core | first_mentioned | 64/128 | 0/16 | 32/64 |
| location/core | last_mentioned | 64/128 | 0/16 | 32/64 |
| naming/core | fixed_candidate0 | 32/64 | 0/8 | 0/32 |
| naming/core | first_mentioned | 32/64 | 0/8 | 16/32 |
| naming/core | last_mentioned | 32/64 | 0/8 | 16/32 |
| possession/frame_holdout | fixed_candidate0 | 64/128 | 0/16 | 0/64 |
| possession/frame_holdout | first_mentioned | 64/128 | 0/16 | 0/64 |
| possession/frame_holdout | last_mentioned | 64/128 | 0/16 | 0/64 |
| location/frame_holdout | fixed_candidate0 | 64/128 | 0/16 | 0/64 |
| location/frame_holdout | first_mentioned | 64/128 | 0/16 | 32/64 |
| location/frame_holdout | last_mentioned | 64/128 | 0/16 | 32/64 |
| possession/qa | fixed_candidate0 | 64/128 | 0/16 | 0/64 |
| possession/qa | first_mentioned | 64/128 | 0/16 | 0/64 |
| possession/qa | last_mentioned | 64/128 | 0/16 | 0/64 |
| location/qa | fixed_candidate0 | 64/128 | 0/16 | 0/64 |
| location/qa | first_mentioned | 64/128 | 0/16 | 32/64 |
| location/qa | last_mentioned | 64/128 | 0/16 | 32/64 |

First/last mention can produce successful reversal pairs in some constructions while failing every complete family. The battery does not exclude every shallow rule: success may reflect a narrow learned rule for these constructions.

## Interpretation and next-decision boundaries

For possession primary cloze: Graduate 0/16 complete families and 0/64 reversal pairs; Pilot 0 0/16 complete families and 1/64 reversal pairs; Pilot 1 0/16 complete families and 1/64 reversal pairs. These are the measured checkpoint behaviors under the frozen construction; no new pass threshold is inferred.

For location primary cloze: Graduate 0/16 complete families and 0/64 reversal pairs; Pilot 0 0/16 complete families and 0/64 reversal pairs; Pilot 1 0/16 complete families and 0/64 reversal pairs. These are the measured checkpoint behaviors under the frozen construction; no new pass threshold is inferred.

Use the following prospective interpretations when weighing these numerical profiles:

| Pattern | Supports | Does not support | Most useful next question |
|---|---|---|---|
| Graduate weak; both pilots stronger | Increased tested context sensitivity accompanies the language-trained checkpoint histories | English updates alone caused it; broad English competence | Robustness across reserved frames and additional combinations |
| Pilot 0 stronger than Pilot 1 | Better Pilot 0 behavior on these matched tasks | Freezing caused weaker English | Learning progress versus protection under matched training conditions |
| Pilot 1 stronger than Pilot 0 | Better contextual discrimination can coexist with worse TinyStories perplexity | Causal protection improvement or synthetic-binding transfer | Replication with matched rehearsal and controlled trajectories |
| Cloze stronger than QA | Context-sensitive continuation with additional question-format difficulty | Invalid cloze evidence or conversational competence | Which QA-format requirement limits accessible answers? |
| Perplexity improves but reversal remains weak | Distribution prediction improves without reliable tested counterfactual selection | Pure memorization or no semantic learning of any kind | Do priors, wording, or association selection explain errors? |
| All three weak | Failure on the frozen battery | Architectural impossibility, insufficient capacity, or inability to acquire binding | Frame accessibility and modulation without successful selection |
| Graduate unexpectedly strong | Tested ordinary-path behavior predates the language pilots | Pilot-acquired capability or broad language competence | Verify path isolation/scoring, alternate frames, later independent replication |
| Naming weak, relational cloze stronger | Relational success despite a weaker naming diagnostic | Contradiction or reason to discard cloze | Naming wording and corpus familiarity |
| Naming strong, relational cloze weak | A simpler identity dependency succeeds while these relations do not | Universal binding or a known architectural barrier | Relation wording, query direction, and role selection |

'Stronger' must be read through the displayed profiles, not an undisclosed cutoff. No mechanism was probed here. Missing attention patterns or the earlier T10/T11 failures cannot establish mathematical impossibility. No result changes synthetic graduation criteria, warrants replacing the model, or automatically authorizes a training curriculum.

The report deliberately retains disagreements across endpoints and frames. The next research decision should use the observed contextual modulation, successful selection, order sensitivity, and QA/frame differences to select a discriminating follow-up. No additional training, mechanistic experiment, or revised test design is executed by this aggregation.

## Reproducibility and artifacts

- [RAW_SCORES.jsonl](RAW_SCORES.jsonl): every frozen item/prior, checkpoint hash, and exact per-token/per-candidate score.
- [FAMILY_RESULTS.csv](FAMILY_RESULTS.csv): complete-family, reversal, item, and margin outcomes in both scoring modes.
- [SUMMARY.json](SUMMARY.json): all distributions, order/priors, signed reversal changes, paired comparisons, finite-universe intervals, and validation counts.
- [BINDING_REFERENCE_RESULTS.json](BINDING_REFERENCE_RESULTS.json): separately evaluated nonsacred synthetic panels.

Aggregation validated every input record against the sealed materialized exam, answer index, candidate tokens, token-score sums, correctness/tie flags, matched family structure, and required counts before emitting outputs. It loads no model and performs no inference.
