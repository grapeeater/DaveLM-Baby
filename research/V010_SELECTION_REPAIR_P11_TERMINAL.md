# P11 gen-only aux overwrite: SUCCESS (frozen inventory endpoint)

Status: **SUCCESS** under frozen P11 gates. Long-gap **inventory-restricted**
hits 74→**106**/215 vs matched control 74. CI **[+0.105, +0.194]**. L0
cosine 0.041→**0.951**. Gate[gen] **0.983**. Mass **0.576**. Language CE
and primitive_induction identical to parent. Authoritative Baby unchanged:
v2R4 U16000
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
P2 not launched. No TEST. Not a promotion of U16000.

Protocol:
[`design/V010_SELECTION_REPAIR_P11_GEN_ONLY.md`](../design/V010_SELECTION_REPAIR_P11_GEN_ONLY.md)
Adjudication: `runs/selection_p11/ADJUDICATION_250001_800.json`
Checkpoint: `runs/selection_p11/treatment_250001/checkpoint_16800.pt`
SHA-256: `369d95c5fdfafea6b270afc47e5a008d17415b7efda275f1aef522d2e4821157`

Replication `250002` also **SUCCESS**: 103/215 vs control 74, CI
**[+0.089, +0.183]**, cosine 0.976, mass 0.653. Checkpoint SHA-256
`ac49bf4138b3bda125cd989392e46ee70fbf3d80362dae8323c56fe2ace19629`.

## Primary (gap ≥ 13, n=215, inventory-restricted argmax)

| | parent | treatment 800 | control 800 |
|---|---:|---:|---:|
| hits | 74 | **106** | 74 |
| excess | 0 | **+0.149** | 0 |
| L0 query cosine | 0.041 | **0.951** | 0.041 |
| overwrite-head mass | 0.004 | **0.576** | 0.004 |
| median gate[gen] | 0.018 | **0.983** | 0.018 |
| induction first-top1 | 0.297 | **0.297** | 0.297 |
| language DEV CE | 1.243 | **1.243** | 1.243 |

Treatment−control excess **+0.149**, bootstrap 95% CI **[+0.105, +0.194]**.

Gap 31+: 54→**79**/159. Gap 13–30: 20→**27**/56. Multi-query bodies whose
inventory argmax varies: 2→**30**.

## Timeline (treatment)

| step | hits | cosine | mass | gate | CE |
|---:|---:|---:|---:|---:|---:|
| 0 | 74 | 0.041 | 0.004 | 0.018 | 1.243 |
| 200 | 74 | 0.045 | 0.144 | 0.018 | 1.243 |
| 400 | 102 | 0.834 | 0.399 | 0.948 | 1.243 |
| 600 | 104 | 0.927 | 0.492 | 0.975 | 1.243 |
| 800 | 106 | 0.951 | 0.576 | 0.983 | 1.243 |

## What this is not yet

Every-step decode-time overwrite (P11_DECODE) made first-token gold
73→105 but free_exact 71→**0** (later value tokens were overwritten).
First-step-only decode (P11_DECODE_FIRSTSTEP) recovered greedy
free_exact **102**/215 vs init 71, Δ **+0.144**, CI **[+0.093, +0.197]**.
Multi-query pair-bind 0→**0.157**. TEST closed. Not a promotion.


## Licensed next

Replication and first-step greedy are in. Remaining for mission-complete
(not P11 SUCCESS): stronger pair-bind, overwrite-on frozen-panel
retention, owner authorization before TEST, no promotion of U16000.
