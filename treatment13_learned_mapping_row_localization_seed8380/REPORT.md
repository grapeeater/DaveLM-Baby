# T13 — Learned Mapping-Row Localization

## Handoff and preflight

The inherited T13 preflight was already a clean `TREATMENT13_LOCALIZATION_PREFLIGHT_PASS`. The earlier abort was a smoke-test assertion bug: it compared `localization_attention.shape[:2]` with `(B,)`, although the intended tensor is `[B, n_candidates, 2]`. The corrected assertion checks batch and slot axes; the observed smoke shape was `[32, 188, 2]`. No optimizer step or backward pass occurred during preflight.

The authoritative one-map initialization checkpoint was loaded by parameter state only:
`C:\DaveLM-v0.9\experiments\minimal_contextual_binding\checkpoints\treatment_one_mapping\seed_8380\latest.pt`, SHA-256 `345984c52a06db5f988aaf4cd47963eea0e9d77489cee94dbb10816af2f5443e`.

T13 removed T12's explicit mapping-row source/value positions. It retained only the public query-key and answer positions. The learned localizer scores candidate pre-query positions for two row slots; the public grammar's fixed `+4` offset pairs each localized source with its row-local value. No row label, target, candidate identity, or localization label enters `forward()`.

## Training

- Fresh T13 wrapper initialized from the one-map checkpoint; fresh AdamW state.
- Architecture: unchanged 10,594,944-parameter DaveLM backbone plus T13 localizer/retrieval module (10,841,346 total; 246,402 new parameters).
- Seed 8380, batch 32, AdamW, learning rate 3e-4, weight decay 0.05, gradient clipping 2.0.
- Answer-only causal cross-entropy at the registered answer position; exactly 1,000 optimizer updates.
- Deterministic schedule: 200 training quartets, 8 quartets (32 documents) per step, 40 shuffled presentations per quartet. Schedule SHA-256: `6157dce61d990b2d96ef91470a80a973bd5e96339d406bceb9fdc75e020cea42`.
- No continuation, rescue, second seed, auxiliary localization supervision, or architecture change.

## Frozen retention exam

Retention contains 80 strict-counterfactual quartets / 320 documents across eight layout signatures that were absent from training. The final checkpoint is `checkpoints\learned_mapping_row_localization\seed_8380\latest.pt`, SHA-256 `cf53eef03adf874255cd54508fc7948dfd1e4926a57537f7e2f828f5f582adab`.

| metric | result |
|---|---:|
| first-answer exact | 256/320 (80.00%) |
| complete four-document quartet | 48/80 (60.00%) |
| target > distractor | 256/320 (80.00%) |
| mean target–distractor margin | 3.3802 |
| median target–distractor margin | 4.1112 |
| mean target probability | 0.7911 |
| mean target+distractor candidate mass | 0.9958 |
| reversal pairs following both orientations | 107/160 (66.88%) |
| prediction changed under reversal | 118/160 (73.75%) |

Query-slot breakdown: slot 0 = 135/160 (84.38%), slot 1 = 121/160 (75.63%). Orientation 1 and 2 were each 128/160 (80.00%). Every one of the eight unseen retention layout signatures scored 32/40 (80.00%).

Localization diagnostics on retention were uniform across cells: at least one of the two learned slot distributions placed its maximum on a true mapping source position in 320/320 cases (100%). Mean summed attention mass on the two true source positions was 1.3194 (this is summed over both slot columns, so it is bounded by 2). The learned localizer therefore found the mapping-row source positions on the exam; remaining errors are in query-conditioned selection/value-to-answer execution, not failure to inspect the rows.

The post-training read-only training-pool audit was 640/800 (80.00%) first-answer exact, with 120/200 complete quartets (60.00%), confirming the model learned the task but did not reach perfect train mastery.

## Conclusion

T13 is a partial but clear capability result: within this strict-counterfactual synthetic family, Baby learned a position-localization mechanism that operated on unseen retention layouts and produced 80% first-token answers without explicit mapping-row positions. This supports the claim that mapping-row localization can be learned from answer-only CE. It does not establish universal binding, unrestricted self-localization, or general reasoning, and the 20% retention answer error / 33% incomplete-quartet rate shows query-conditioned row selection and/or value transfer remains imperfect.

The preregistered bounded T13 run is complete. No T14, rescue, continuation, or other experiment was started.
