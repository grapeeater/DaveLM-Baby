# Post-P7 integrity stop audit

Read-only evidence audit, 2026-09-05. No inference or training performed.

## Confirmed answer-key defect

144 of 288 frozen controlled items have an expected_subject that contradicts
the subject explicitly stated in the queried fact: near_distribution 48/96,
counterfactual 32/64, surface_form 32/64, distractor 32/64.

Example: near_distribution:f000:a0q0o1 says:

    Sam found the blue ball. Ben carried the blue book.
    Who found the blue ball?

The frozen answer key identifies Ben. The stated fact identifies Sam.

The builder changes subject identities when fact_order changes, while target
remains independent of fact_order. This swaps relationships rather than merely
reordering intact facts. The existing validators checked balanced metadata,
counts and tokenization, but did not check answer truth against serialized facts.

The previously reported item, reversal, complete-family and signed-margin
correctness summaries are therefore not valid measures of intended contextual
selection. Raw likelihoods remain observations of the actual prompts, subject
to their execution provenance; this audit does not rescore or rerun them.
The previous interpretation of these aggregates as evidence of P7 transfer
failure must be withdrawn. Broad contextual transfer remains unestablished.
The original corrected 12-item milestone is not invalidated by this finding.

## Confirmed lineage-record discrepancy

ARCHIVE_RECORD.json names language_compositional_p6/latest.pt as P7's parent
with hash da009100412fba2adcf58823aefd37cc9486ef7aff39672361ce6111773c6a3b.
The P7 training source instead names language_sentencebound_p5/latest.pt.
Read-only hashes confirm the recorded parent hash belongs to P5:

- P5: da009100412fba2adcf58823aefd37cc9486ef7aff39672361ce6111773c6a3b
- P6: e708f00bdab4c4325382f3ceb724ad377360daf8ff2f6c14278cc17e692a9930

The archival lineage path is inconsistent with its hash and training source.
No archive record or checkpoint was modified.

## Stop decision

The active goal explicitly requires stopping when provenance/evaluation
integrity is uncertain. No next parent or treatment is chosen. Frozen materials
and historical execution outputs remain preserved. Changing keys to follow
existing facts and rebuilding facts to follow intended order balancing are
different scientific repairs; neither is silently applied. A retrospective
rescore would not restore prospective status to an already exposed battery.

Next review question: authorize a separately versioned integrity repair and
fresh prospective battery, with semantic answer-key validation independent of
construction, before drawing treatment conclusions.
