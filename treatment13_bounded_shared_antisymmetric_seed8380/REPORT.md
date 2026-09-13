# Bounded shared-plus-antisymmetric scorer — authoritative seed 8380

## Frozen execution

The treatment used the original T13 baseline/start checkpoint, not the champion final checkpoint, and the exact champion train pool, retention pool, schedule, seed, optimizer, and answer-CE plus permutation-invariant hard-min localization objective. The only scientific change was the bounded shared-plus-antisymmetric localizer scorer with fixed rho `1.5919504166`. Training completed exactly 1,000 optimizer updates. The frozen retention exam was run once under `model.eval()` and `torch.no_grad()` with zero optimizer updates during evaluation.

Starting checkpoint SHA-256: `cf53eef03adf874255cd54508fc7948dfd1e4926a57537f7e2f828f5f582adab`

Final checkpoint SHA-256: `a4ac52d1f24ad253d0a9a475891bf326e4b429f23d872b981968d797521601f2`

Train pool SHA-256: `40d78c450f2335aa155031a612d5b8e3311de2071cc5feb97fb430bc67f84df3`

Retention pool SHA-256: `29dc566e90df7c9a592014d1d0cf60dd5da654d570096c96d0c7287bcd5f9072`

Schedule SHA-256: `6157dce61d990b2d96ef91470a80a973bd5e96339d406bceb9fdc75e020cea42`

Optimizer updates: `1000`; retention optimizer updates: `0`; seed: `8380`.

## Frozen retention results

| Metric | Result |
|---|---:|
| Answer exact | 292/320 (91.25%) |
| BOTH_DISTINCT | 276/320 (86.25%) |
| EXACTLY_ONE | 12/320 (3.75%) |
| SLOT_COLLAPSE | 32/320 (10.00%) |
| NEITHER | 0/320 (0%) |
| Complete quartets | 64/80 (80.00%) |
| Reversal both-correct | 135/160 (84.375%) |
| Prediction changed under reversal | 138/160 (86.25%) |

By localization category: BOTH_DISTINCT answer `269/276` (97.464%); EXACTLY_ONE `7/12` (58.333%); SLOT_COLLAPSE `16/32` (50%). Derived from saved row weights, selected-row correctness conditional on BOTH_DISTINCT was `276/276`; query-row correctness was `275/276`. Slot-0 and slot-1 source-hit rates were `312/320` (97.5%) and `316/320` (98.75%).

### Layouts

| Layout | BOTH_DISTINCT | Answer exact |
|---|---:|---:|
| p16_b8 | 40/40 | 40/40 |
| p20_b8 | 32/40 | 36/40 |
| p24_b10 | 36/40 | 36/40 |
| p28_b10 | 36/40 | 38/40 |
| p32_b12 | 28/40 | 34/40 |
| p16_b12 | 32/40 | 36/40 |
| p24_b8 | 36/40 | 37/40 |
| p32_b8 | 36/40 | 35/40 |

### Quartet and localization pathology

Quartet patterns were 69/80 4/4 BOTH_DISTINCT, 8/80 mixed, and 3/80 4/4 EXACTLY_ONE; localization coordinates were identical across all 80 quartets in the saved audit. All 32 SLOT_COLLAPSE documents collapsed onto a true source (destinations: position 39 eight times, 55 eight times, and 47, 43, 31, 35 four times each). The 12 EXACTLY_ONE cases' non-source destinations were positions 19, 62, and 76 (four each).

## Comparison with champion

| Metric | Champion | Treatment | Delta |
|---|---:|---:|---:|
| BOTH_DISTINCT | 290/320 | 276/320 | -14 |
| Answer exact | 307/320 | 292/320 | -15 |
| Complete quartets | 73/80 | 64/80 | -9 |
| Reversal both-correct | 153/160 | 135/160 | -18 |
| SLOT_COLLAPSE | 0/320 | 32/320 | +32 |

## Classification

**FAILURE.** The treatment did not approach any frozen graduation regime: BOTH_DISTINCT, answer exact, complete quartets, reversal following, every-layout coverage, and SLOT_COLLAPSE gates all fail. The conditional downstream selected-row mechanism remains strong on BOTH_DISTINCT, but new slot collisions and reduced query-row/answer correctness show that the bounded scorer parameterization was harmful under this frozen regime.

No retention result was used to alter training. No continuation, rescue, second seed, rho change, loss addition, or follow-up experiment was performed.
