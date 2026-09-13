# SF3 prospective treatment decision

## Decision

SF3 manipulates one variable: the English-update learning rate after update 100. It replays SF2 exactly through
update 100, then linearly anneals the remaining 90 English learning rates from `5e-5` to zero. Binding updates
remain at `5e-5`. Every other scientific and executable choice is pinned to SF2.

The decisive Baby evidence is the residual autopsy. SF2 moved the Alex/Owen boundary monotonically toward Owen,
recovered all four Owen-correct records, and ended with one Alex-correct miss at a sequence margin of only
`-0.030` nats. All four cell midpoints were mildly Owen-shifted. Every record appeared twice in every English
batch, excluding stochastic recency and unequal exposure. Component swaps attributed most movement to upstream
state, with the head contributing only the final small tip. SF2 retained both binding pools, language CE, and D3
far better than SF1. This makes a late step-size test narrower than changing the objective, data, or scope.

The preserved constant-rate SF2 endpoint is the historical comparator. SF2 remains
`SF2_ACQUISITION_FAIL`; its 15/16 result is not reclassified.

## Ranked alternatives

1. **Late English LR annealing — selected.** Weak but strongest direct support: SF2 showed a continuing group
   boundary sweep and a near-tie residual under constant LR. Linear-to-zero needs no curve or floor parameter.
2. **Output-head freezing/constraint — rejected for SF3.** The head tipped the residual item, but upstream changes
   produced most of the boundary motion. A head constraint changes the representation/readout allocation more
   broadly and is a poorer first causal test.
3. **Pairwise or contrastive margin — rejected.** It changes the learning objective and candidate/full-vocabulary
   tradeoff without direct evidence that SF2 lacked candidate discrimination.
4. **Additional constant-rate CE — rejected.** More of the same could push the one residual item back, but the
   observed group sweep was still moving toward Owen and could worsen other Alex-correct records.
5. **Ordinary-language replay — rejected.** SF2 already met its language and D3 retention criteria. Replay changes
   the data mixture without targeting the residual boundary.
6. **Continued or modified parent KL — rejected.** Lambda 1.0 already delivered the intended retention. Changing
   KL strength would test a different retention/acquisition tradeoff, while continuing unchanged is not a new
   causal treatment.
7. **Staged/changing KL strength — rejected.** The residual is not evidence of a KL support failure, and stage
   timing/strength adds choices unsupported by Baby's frozen evidence.
8. **Architecture, tokenizer, norm, or attention changes — rejected.** Neither Baby's evidence nor the comparative
   background establishes a need for them.

Granite, Qwen, and LFM2 evidence is background only. It supports considering decay/replay/KD design families but
does not choose this schedule or override Baby's direct experiment.

## Predicted diagnostic signature

If the constant late LR caused boundary overshoot, SF3 should exactly reproduce SF2 at update 100, retain the
SF2 language/D3/binding protections, and reach 16/16, 8/8 reversal, 4/4 family, and 16/16 exact at update 200
without simply creating a new opposite-side miss. Intermediate update-125/150/175 TRAIN measurements describe
the boundary trajectory but cannot tune or stop the run.

If update 100 fails exact tensor reproduction, the comparison is invalid and training stops before the treatment
diverges. If SF3 still misses the fixed endpoint, annealing as specified did not solve the residual; that result
does not authorize another experiment in this task.
