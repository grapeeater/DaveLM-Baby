# Source-recognition authoritative treatment — seed 8380

Exactly one 1,000-update run was completed from the frozen champion starting
checkpoint using the unchanged hard-min localization objective plus the frozen
source-set recognition auxiliary (`lambda_src_rec=0.20985975404004048`).

## Frozen retention results

| Metric | Result |
|---|---:|
| Answer exact | 307/320 (95.9375%) |
| BOTH_DISTINCT | 276/320 (86.25%) |
| EXACTLY_ONE | 44/320 (13.75%) |
| SLOT_COLLAPSE | 0/320 |
| NEITHER | 0/320 |
| Complete quartets | 72/80 (90.0%) |
| Reversal both-correct | 152/160 (95.0%) |

Conditional on BOTH_DISTINCT:

- answer correctness: 276/276 (100%)
- selected-row correctness: 276/276 (100%)
- query-row correctness: 276/276 (100%)

Per-layout BOTH_DISTINCT / answer exact:

- p16_b12: 40/40, 40/40
- p16_b8: 40/40, 40/40
- p20_b8: 28/40, 37/40
- p24_b10: 28/40, 36/40
- p24_b8: 32/40, 37/40
- p28_b10: 40/40, 40/40
- p32_b12: 36/40, 38/40
- p32_b8: 32/40, 39/40

## Classification

**FAILURE.** Compared with the champion, BOTH_DISTINCT fell from 290/320 to
276/320 and answer exact remained 307/320. The graduation BOTH_DISTINCT and
answer gates failed, although complete-quartet, reversal, slot-collapse, and
conditional downstream checks met their numerical thresholds. The auxiliary
did not improve the targeted localization bottleneck under this frozen regime.

## Integrity

- Starting checkpoint SHA256: `cf53eef03adf874255cd54508fc7948dfd1e4926a57537f7e2f828f5f582adab`
- Final checkpoint SHA256: `8f7fa0cc3f509eeb600def2f233f38022afbfe9ccb27e0a0ccdf813f25e8a7a0`
- Train pool SHA256: `40d78c450f2335aa155031a612d5b8e3311de2071cc5feb97fb430bc67f84df3`
- Retention pool SHA256: `29dc566e90df7c9a592014d1d0cf60dd5da654d570096c96d0c7287bcd5f9072`
- Schedule SHA256: `6157dce61d990b2d96ef91470a80a973bd5e96339d406bceb9fdc75e020cea42`
- Optimizer updates: exactly 1,000; retention optimizer updates: 0.
- Retention was run once under `model.eval()` and `torch.no_grad()`.
- Seed 8380 only; no continuation, rescue, or second run.
- Predecessor and preflight artifacts were preserved.
