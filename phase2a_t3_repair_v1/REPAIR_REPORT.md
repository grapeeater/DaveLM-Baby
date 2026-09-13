# PHASE 2A T3 REPAIR — INFRASTRUCTURE BLOCKED

## 1. Hard-stop ledger
The prior committed study remains historical `T3_HARD_STOP`; it is not counted as a scientific replication.

## 2. Identical-seed root cause
The prior runner used the same deterministic QA ordering for every seed and did not provide a complete seed-specific schedule/RNG ledger. Terminal checkpoints were byte-identical. The corrected runner added seed-specific QA permutation, but the attempted correction was stopped before its first update and therefore generated no scientific result.

## 3. Evaluator/data closure failure
The sealed T3 package contains only `corpus.json` and `audits.json` (384 train, 128 DEV, 128 TEST rows). It does not contain the required TRAIN16 panel, either nonsacred binding pool, shortcut-audit grouping metadata, representation generalization splits, or a complete frozen evaluator. The T2 bundle also contains no binding-pool or TRAIN16 artifact that is uniquely authoritative for T3. These values cannot be reconstructed without a scientific choice.

## 4. Validation
The parent and tokenizer hashes verified. The corrected span adapter and seed-specific ordering logic compiled. No T2-EVAL-TEST, FINAL, sacred, or locked transfer material was accessed. No valid clean rerun was completed after the repair because the mandatory evaluator/data closure failed.

## 5. Adjudication
`PHASE2A_T3_INFRASTRUCTURE_BLOCKED`. No repaired three-seed scientific classification is issued. Existing v1 runs and their checkpoint hashes remain preserved historical artifacts and must not be treated as valid replications.
