# HUMAN_TEST_READY development protocol v2

This is a prospective, nonsacred development battery built outcome-blindly with construction seed 87010. It is not the sealed FINAL battery. It must be frozen before evaluating any future candidate.

## Battery

- 20 ordinary short generation prompts, greedy decoding only, max 32 tokens.
- 24 factual cloze items: 6 families x 4 assignment/query cells, with 12 matched reversal pairs. Candidate pairs are equal-token-length under the approved tokenizer.
- 12 elementary instruction items, balanced across yes/no, happy/sad, and hot/cold exact responses.
- 12 two-turn continuity items, balanced across six name pairs.

## Controlled scoring

For fact, instruction, and continuity items, score each full candidate token sequence conditionally on BOS plus the exact prompt. Sum only candidate-token log probabilities; exclude EOS; apply no length normalization or prior subtraction. A strictly positive correct-minus-incorrect margin is correct. A zero margin fails. Greedy exact success requires the exact correct candidate tokens followed immediately by EOS within 32 tokens.

## Generation rubric

Preserve every greedy response verbatim. Automatic failure fields are immediate EOS, any three-identical-token run, any repeated decoded trigram, and a duplicate normalized non-EOS response shared by more than three prompts. Two blinded reviewers independently mark grammatical completeness and prompt relevance using `HUMAN_REVIEW_RUBRIC.md`; a disagreement is not a pass.

## Advancement gates

A development candidate must pass all gates: aligned TinyStories DEV loss <= 3.25 on the fixed 128-record causal slice; fact candidate correctness >= 20/24, >= 10/12 strict reversal pairs, >= 4/6 complete families, and >= 18/24 greedy exact; instruction candidate correctness >= 10/12 and greedy exact >= 9/12; continuity candidate correctness >= 10/12 and greedy exact >= 9/12; generation non-immediate-EOS >= 16/20, automatic non-degenerate >= 16/20, reviewer-complete >= 14/20, reviewer-relevant >= 14/20, and both reviewer fields passing >= 12/20. Both nonsacred binding pools independently require answer >= 76/80, BOTH_DISTINCT >= 76/80, and zero collapse.

Loss is a safety diagnostic and cannot substitute for behavior. No candidate may access FINAL unless all development gates pass.
