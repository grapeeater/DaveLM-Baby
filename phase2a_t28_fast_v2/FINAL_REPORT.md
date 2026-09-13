# T28 FAST V2 FINAL REPORT

## Study classification
T28_SUFFIX_MARGIN_VALID_COMPLETION; mechanism outcome remains OUTPUT_FAIL under inherited T24 gates.

## Scientific change
A vectorized suffix hard-negative hinge was added to the known-fast T24 teacher-forced forward path. Margin=1.0, lambda=0.5. No rollout or extra model forward was used. All other T24/T18 settings were held fixed.

## Runtime and integrity
The original T28 attempt is preserved as an engineering artifact. This v2 copied the untouched T24 runner and added only the suffix loss. Preflight reproduced the parent baseline (DEV CE 1.2040123895, binding intact, no test loaded). All three seeds reached update 1000 with finite losses/gradients. The runner emitted a missing-ledger warning after completion; this was repaired by creating the additive RUN_LEDGER.json only, with no checkpoint or metric changes.

## Per-seed terminal results
| seed | updates | DEV CE | PPL | gap | pointer | native FC | exact+EOS | reversals native | complete native families | TRAIN16 | binding | protected data |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
|850001|1000|1.253619|3.503|0.4211|112/128|88/128|36/128|24/64|3/16|PASS|PASS|LOCKED|
|850002|1000|1.259366|3.523|0.4218|108/128|82/128|35/128|—|—|PASS|PASS|LOCKED|
|850003|1000|1.256593|3.513|0.4216|116/128|82/128|40/128|—|—|PASS|PASS|LOCKED|

All terminal JSON and item-level evaluations remain under the corresponding run directories. T3_TEST, T2_EVAL_TEST, FINAL, and sacred material were not loaded.

## Interpretation
The rebuilt runner restored normal T24-class execution: each branch completed rather than hitting the prior pre-update resource stall. The suffix margin is therefore mechanically feasible. The available terminal evaluator reports strong pointer/forced-choice acquisition and preserved TRAIN16/language/binding, but exact native generation remains in the same failure regime as T24. The frozen T28 question is not supported as an exact-generation solution by these results. No locked panel was opened and no rescue was attempted.

## Provenance
Parent SHA-256: c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1
Seeds: 850001, 850002, 850003
Bundle: C:\\DaveLM-CADAVER\\phase2a_t28_fast_v2
