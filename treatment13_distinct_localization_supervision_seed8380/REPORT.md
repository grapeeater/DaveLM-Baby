# T13 distinct-localization supervision — final report

Pretreatment passed after the authorized sign correction: assignment log-probabilities now select the maximum (equivalently, minimum NLL). Permutation invariance, distinct-coverage superiority, finite/nonzero localizer gradients, forward anti-leakage, train-only lambda calibration, and zero-update checks all passed. Lambda was 1.0536573711078283 from initial train losses 7.465176582336426 / 7.085013389587402. No retention behavior influenced calibration.

Exactly one seed-8380 run was executed for exactly 1,000 AdamW updates on the existing T13 schedule (800 train documents, 320 retention documents, batch 32, 3e-4 LR, weight decay .05, clip 2.0). The only scientific change was permutation-invariant two-source localization supervision; no architecture or persistent parameters were added.

## Frozen retention results

| metric | T13 baseline | treatment |
|---|---:|---:|
| answer exact | 256/320 (80.00%) | **307/320 (95.94%)** |
| target > distractor | — | 309/320 (96.56%) |
| mean margin | 3.3802 | 11.2415 |
| median margin | 4.1112 | 11.8977 |
| candidate mass | 0.9958 | 0.9854 |
| complete quartets | 48/80 (60.00%) | **73/80 (91.25%)** |
| reversal both-correct | 107/160 (66.88%) | **153/160 (95.63%)** |

Localization categories:

- BOTH_DISTINCT: **290/320 (90.625%)**, answer 290/290 (100%).
- EXACTLY_ONE: 30/320 (9.375%), answer 17/30 (56.67%).
- SLOT_COLLAPSE: 0/320.
- NEITHER: 0/320.

Slot source-hit rates were 308/320 (96.25%) for slot 0 and 302/320 (94.375%) for slot 1. Mean summed true-source attention mass was 1.89595. On BOTH_DISTINCT examples, downstream selected-row correctness and query-row correctness were each 290/290 (100%).

Query-slot splits: slot 0 = 151/160 (94.375%), slot 1 = 156/160 (97.50%). Orientation 1 = 153/160 (95.625%), orientation 2 = 154/160 (96.25%). Unseen-layout accuracies ranged from 35/40 (87.5%) to 40/40 (100%); BOTH_DISTINCT rates ranged from 28/40 (70%) to 40/40 (100%).

Quartets: 72/80 were 4/4 BOTH_DISTINCT, 1 mixed, 7 were 4/4 EXACTLY_ONE. Coordinates were identical across four members in 79/80 quartets.

## Classification

**CLEAR_SUCCESS.** The intervention materially improved simultaneous distinct two-row localization and preserved/strengthened exact binding on unseen layouts. It does not establish universal binding or justify a follow-up automatically.

Retention was evaluated once with `model.eval()` and `torch.no_grad()`; retention optimizer steps were zero. No continuation, rescue, second seed, or further experiment was run.
