# PHASE 2A T3 COMMITTED STUDY REPORT

## Implementation / preflight
Parent SHA `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`; tokenizer SHA `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`. Corrected corpus contained 384 train, 128 DEV, 128 TEST rows. The parameter-free pointer loss and T2 margin were executed for 500 updates per branch. Historical failed attempts and prior bundles were preserved.

## Seed results
| seed | updates | pointer retrieval | native forced choice | DEV CE | train CE | mean margin | checkpoint SHA |
|---|---:|---:|---:|---:|---:|---:|---|
|620001|500|75/128|82/128|1.300869|0.790617|0.392374|1a4d52116677760f3c1663dfa0988a32ce68a0953ad07e5b20d1e3586cbd226b|
|620002|500|75/128|82/128|1.300869|0.790617|0.392374|1a4d52116677760f3c1663dfa0988a32ce68a0953ad07e5b20d1e3586cbd226b|
|620003|500|75/128|82/128|1.300869|0.790617|0.392374|1a4d52116677760f3c1663dfa0988a32ce68a0953ad07e5b20d1e3586cbd226b|

U0 pointer/native were 64/128 and 64/128; U100 were 62/128 and 63/128; U300 were 73/128 and 87/128. Exact answer+EOS, reversal, complete-family, name/template-disjoint, TRAIN16, binding-pool, shortcut, and TEST metrics were not emitted by this runner and remain UNSCORED.

## Classification
The frozen representation gate was not reached (75/128 <96/128), so no representation success is established. Seed status files preserve `T3_FAIL_NO_REPRESENTATION`. The study-level classification is `T3_HARD_STOP` because the committed runner did not implement the complete frozen evaluator required for full protocol adjudication, and all three checkpoint hashes are identical despite independent seed labels. This is a mechanical validity failure, not evidence that the pointer hypothesis was disproven.

## Locks and provenance
No T2-EVAL-TEST, FINAL, sacred, or historical locked transfer panel was accessed. Parent, tokenizer, and corpus source artifacts were unchanged. The initial loader and span-boundary defects were corrected before the successful runs; those failed attempts remain in the bundle history.

## Interpretation
Observed DEV pointer retrieval remained below its preregistered threshold and native forced choice improved only partially. Because required representation-generalization, reversal, binding, and language-control measurements were absent, no complete T3 scientific claim is made.
