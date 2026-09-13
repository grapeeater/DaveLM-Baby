# SF1 upstream localization forensic v1

This diagnostic spliced the residual stream between authoritative Pilot1 and SF1-update100 at the embedding output and after each attention and MLP residual sublayer. It used only the 16 SF1 training items, the frozen 256-position unrelated TinyStories D3 sample, and the authorized aligned first-128 TinyStories DEV records.

## Result

`DISTRIBUTED_OR_INTERACTIVE`

The effects are partially separable at inference time, but there is no clean single-depth decomposition. Alex/Owen identity bias changes sharply in the middle-to-late stack, while the final useful factual pattern, broad name-prior inflation, and language regression all depend strongly on the late path.

## Execution graph and controls

The ordinary path is token plus position embeddings, eight pre-normalized transformer blocks (attention residual, then MLP residual), final normalization, and an untied language head. The T13 localizer/retrieval modules exist in both wrappers but are inactive here because all three authorized endpoints call `base_model` directly.

Same-source manual execution reproduced native logits exactly at every boundary (maximum absolute error `0`; tolerance `1e-6`). Native aggregate summaries reproduced the prior component-swap forensic within `3.43e-6` for Pilot1 and `1.72e-6` for SF1. The initial `1e-6` cross-run aggregate check stopped on those harmless batching-order differences; that stop is preserved under `attempt_1_aggregate_tolerance_stop`, and finalization used a documented `1e-5` cross-run aggregate tolerance without repeating inference.

No optimizer was created, autograd was disabled, all parameters had `requires_grad=False`, no model state was saved, and checkpoint/tokenizer hashes were reverified after evaluation.

## Native endpoints

| endpoint | Pilot1 | SF1 |
|---|---:|---:|
| factual sequence correct | 9/16 | 12/16 |
| complete matched reversal pairs | 1/8 | 4/8 |
| Alex/Owen matched pairs | 1/4 | 0/4 |
| Mia/Nora matched pairs | 0/4 | 4/4 |
| mean factual sequence margin | 0.0553 | 0.3881 |
| Owen-correct margin | -1.0477 | -0.2730 |
| unrelated four-name mass | 0.000907 | 0.067833 |
| aligned language CE | 3.3907 | 4.6529 |

## Where each effect emerges

**Useful factual signal.** The demonstrated SF1 improvement is concentrated in Mia/Nora: all four Mia/Nora reversal pairs become correct only when the SF1 prefix includes the block-7 MLP residual. The overall mean margin crosses the descriptive 50% recovery point after block 6 MLP and reaches 98% after block 7 attention, but correctness stays at 8/16 until block 7 MLP changes it to 12/16. Alex/Owen remains 0/4 complete pairs even in native SF1.

**Alex/Owen suppression.** Pilot1 has a large Alex-default component (`1.1215`) and a small context-separation component (`0.0737`). SF1 reduces the Alex default to `0.3887` and raises context separation modestly to `0.1157`, but Owen remains wrong. In the cumulative SF1-prefix path, the default first flips toward Owen after block 4 attention, is strongest after block 4 MLP, and flips back toward Alex after block 5 MLP. This non-monotonic switching localizes identity-selection transformations, not successful relational selection.

**Broad name-prior inflation.** With the Pilot1 output path, four-name mass rises gradually from blocks 4-7, crosses 50% of the native Pilot1-to-SF1 change only after block 7 MLP, and has its largest increment there (`+0.014710`). The SF1 output path further amplifies the full-SF1-upstream value from `0.036394` to `0.067833`.

**Language regression.** Aligned CE stays near Pilot1 through the early stack, then worsens cumulatively through blocks 5-7. Its first majority-recovery point and largest increment are both after block 7 MLP (`+0.428046`); the SF1 output path raises the full-SF1-upstream CE from `4.2769` to `4.6529`.

## Interpretation

### Established

- The SF1 factual endpoint, unrelated-context name prior, and language regression are upstream effects with materially different internal depth profiles.
- Alex/Owen identity preference is transformed non-monotonically across blocks 4-7. Correct Alex/Owen reversal behavior never emerges on these 16 training items.
- Mia/Nora complete-pair correctness requires the block-7 MLP residual in the tested cumulative direction.
- Name-prior inflation and language damage are late, cumulative effects and are amplified by the SF1 output path.

### Supported

- The residual Alex/Owen failure is better described as a mixture of modest context-conditioned separation and much larger changing identity bias than as a total absence of contextual signal.
- Some inference-time separation exists: identity-default switching begins before the strongest broad name-prior and language-loss changes. The final useful factual pattern and both harms nevertheless overlap in the late path.

### Not established

- Patching depth does not reveal where learning occurred during optimization.
- These training-item interventions do not establish held-out factual generalization or a reusable ordinary-English binding mechanism.
- No architectural limit, treatment choice, or claim about locked panels follows from this diagnostic.

## Smallest justified next action

Perform one bounded read-only **token-position patch** at the already-localized transition sites—block 4 attention/MLP, block 5 attention/MLP, and block 7 attention/MLP—using only these same 16 training items and the same D3/language samples. Patch answer/cue positions separately from factual-name/object positions. That would test whether the mid-stack identity flips and late suppression enter through the answer-position state or through transformed context-token states. It is more discriminating than another whole-block swap and requires no treatment or locked-panel access.

SF1_UPSTREAM_LOCALIZATION_FORENSIC_COMPLETE
