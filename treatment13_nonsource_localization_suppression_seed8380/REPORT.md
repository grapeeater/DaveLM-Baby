# Nonsource-suppression authoritative treatment — seed 8380

Exactly one run was executed for 1,000 optimizer updates from the frozen T13
starting checkpoint. The only objective addition was the generic strongest
non-source attention penalty, with frozen `lambda_filler=2.9595848365588058`
and existing localization weight `1.0536573711078283`.

## Frozen retention result

| Metric | Result |
|---|---:|
| Answer exact | 286/320 (89.375%) |
| BOTH_DISTINCT | 254/320 (79.375%) |
| EXACTLY_ONE | 62/320 (19.375%) |
| SLOT_COLLAPSE | 4/320 (1.25%) |
| NEITHER | 0/320 |
| Complete quartets | 63/80 (78.75%) |
| Reversal both-correct | 143/160 (89.375%) |

Conditional on BOTH_DISTINCT, answer accuracy was 246/254 (96.85%); selected
localized-row correctness was 254/254. Query-row correctness was 246/254.
Thus downstream execution remained strong when localization succeeded, but
localization coverage regressed substantially and slot collapse appeared.

Per-layout BOTH_DISTINCT / answer exact: p16_b12 30/40, 38/40; p16_b8 28/40,
38/40; p20_b8 32/40, 34/40; p24_b10 40/40, 38/40; p24_b8 32/40, 34/40;
p28_b10 28/40, 34/40; p32_b12 32/40, 34/40; p32_b8 32/40, 36/40.

## Classification

**FAILURE.** None of the graduation gates passed: BOTH_DISTINCT 254/320
(required 310), answer exact 286/320 (required 312), complete quartets 63/80
(required 72), reversal both-correct 143/160 (required 152), and
SLOT_COLLAPSE was nonzero (4/320). The intervention did not improve the
residual localization bottleneck and damaged the previously collapse-free
mechanism.

## Integrity

- Starting checkpoint SHA256: `cf53eef03adf874255cd54508fc7948dfd1e4926a57537f7e2f828f5f582adab`
- Final checkpoint SHA256: `ad0fe0743c2335e4a2268aeb589c80faf7ff23dc82be4394b2150b179ceb7f4f`
- Training pool SHA256: `40d78c450f2335aa155031a612d5b8e3311de2071cc5feb97fb430bc67f84df3`
- Retention pool SHA256: `29dc566e90df7c9a592014d1d0cf60dd5da654d570096c96d0c7287bcd5f9072`
- Schedule SHA256: `6157dce61d990b2d96ef91470a80a973bd5e96339d406bceb9fdc75e020cea42`
- Optimizer updates: exactly 1,000; retention optimizer updates: 0.
- Retention ran once under `model.eval()` and `torch.no_grad()`.
- Seed 8380 only; no continuation, rescue, or second run.
- Original abort, provenance audit, and corrective preflight remain preserved.
- No architecture, corpus, tokenizer, or historical-artifact changes.
