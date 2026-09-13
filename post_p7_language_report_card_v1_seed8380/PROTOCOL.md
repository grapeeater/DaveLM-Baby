# Post-P7 language report card

Protocol v1.0, construction seed 8380. This is a prospective behavioral battery; no Baby checkpoint was loaded or evaluated during construction.

## Sections

- Near-distribution compositional transfer: 12 families × 8 counterbalanced items = 96 items. Familiar names, objects, colors, and active clauses are recombined; the article and paired fact structure differ from P7 training text.
- Counterfactual/reversal control: 8 families × 8 items = 64 items. Assignment, query, and fact order are crossed; both opposed assignments must be answered correctly.
- Surface-form transfer: 8 families × 8 items = 64 items. Passive facts and a paraphrased query frame are evaluated separately.
- Distractor/interference diagnostic: 8 families × 8 items = 64 items. An irrelevant third-entity sentence is inserted; this is diagnostic only.
- Naturalistic transfer: 24 fixed story-start prompts, generation-only, reported separately.

## Controlled scoring

Score each candidate completion as the full conditional log-likelihood of the leading-space name plus period. The item is correct iff the correct-minus-incorrect margin is strictly positive; ties are incorrect. Report item accuracy, ties, family complete success (all eight members), reversal success (both assignments at fixed query/order), and raw margins. Primary controlled endpoints are complete-family proportion and mean within-family reversal success, reported separately by section. No prior subtraction or calibration is used.

Naturalistic prompts use greedy generation with a maximum of 32 new tokens and stop at EOS or the first complete sentence. Human review is required for grammatical completeness, subject/reference consistency, relation appropriateness, repetition, contradiction, and truncation; reviewers must score every prompt using the frozen rubric and may not select examples after viewing outputs.

## Controls and interpretation

Every candidate is correct and incorrect equally often; assignment, query, and fact order are balanced within every family. Full prompt duplication is rejected. Exact-string non-overlap is checked against P7 TRAIN_TEXT and TinyStories TRAIN/DEV under case-folded whitespace normalization; this does not exclude semantic or near-duplicate contamination. Controlled success establishes only the measured constructions. Naturalistic failure does not erase controlled success, and controlled success does not establish broad English competence. No sacred binding material is used.
