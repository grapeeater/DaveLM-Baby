# SF12 FINAL CLASSIFICATION

**ANSWER_VOCAB_RETENTION_WEAKENED**

# WHAT CHANGED

The sole scientific change from SF11 was adding the frozen teacher-anchored answer-vocabulary excess term `R_name` to English updates. Curriculum, schedules, parents, optimizer, full KL, margin loss, parameter scope, binding rehearsal, evaluation, and gates remained fixed.

# SEALED PROVENANCE

- Receipt: `a1d1423a2997c6f830f2d49661a0bd700fab64ca4c56a073293bcfd9db45e7cc`
- Manifest: `140847565ebb512d42cd3a9aeb8e555e2e4600232349362fb36c343952d41123`
- Controller: `2b6bd08380cfae015756d9c35d94c298bf6857a93894ceddf363ca843e5b3cad`
- Historical/FINAL/sacred panels: locked and unscored

# PER-SEED RESULTS

| Seed | Stop | TRAIN16 c/e/r/f | Surface c/e | Order e (fact/copy) | Language CE | D3 | Binding | R_name probe | Status |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 87029 | 100 | 16/16/8/4 | 14/11 | 8 (5/3) | 3.6520 | 0.01322 | pilot0:80/80/c0; pilot1:80/80/c0 | 0.002195 | STOP_REGRESSION |
| 87030 | 100 | 16/16/8/4 | 14/12 | 5 (4/1) | 3.6173 | 0.01571 | pilot0:80/80/c0; pilot1:80/80/c0 | 0.002090 | STOP_REGRESSION |
| 87031 | 100 | 16/16/8/4 | 14/10 | 6 (4/2) | 3.6722 | 0.01600 | pilot0:80/80/c0; pilot1:80/80/c0 | 0.002302 | STOP_REGRESSION |

# CROSS-SEED RESULT

- Complete endpoint successes: 0/3
- Partial retention/widening successes: 0/3
- D3 failures: 3/3
- Retention regressions: 0/3

# DID THE NEW TERM ACTUALLY ENGAGE

- Seed 87029: mean R_name `0.003204`, active fraction `0.198`, mean student/teacher mass `0.005073` / `0.001663`.
- Seed 87030: mean R_name `0.003236`, active fraction `0.205`, mean student/teacher mass `0.005117` / `0.001663`.
- Seed 87031: mean R_name `0.003215`, active fraction `0.197`, mean student/teacher mass `0.005086` / `0.001663`.

# DAVE-CODED INTERPRETATION

The dedicated name-retention term did not stop the replicated D3 breach in at least two Babies.

# SCIENTIFIC INTERPRETATION

This weakens the proposition that answer-vocabulary under-attention on the existing KL positions is sufficient to explain or fix the D3 regression. Corpus coverage remains untested.

# ARTIFACTS

- Study: `C:\DaveLM-CADAVER\sf12_answer_vocab_retention_v1`
- Machine-readable results: `C:\DaveLM-CADAVER\sf12_answer_vocab_retention_v1_reporting_hotfix_v1\RESULTS.json`
- Output hashes: `C:\DaveLM-CADAVER\sf12_answer_vocab_retention_v1_reporting_hotfix_v1\OUTPUT_SHA256SUMS.txt`

# EXACTLY ONE NEXT ACTION

Prospectively review broader retention-position coverage as the next bounded hypothesis without changing this result.
