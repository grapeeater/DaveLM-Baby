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

Frozen `free_exact` / greedy decode on the same long-gap rows stayed
**71/215**. `score_items` generation does not call `pack()`, so
`PACK_HOOK` never sets `gen_index` and gen-only overwrite is identity
during greedy. Isolation panels via `evaluate_panels` have the same hole,
which is why they match parent to floating-point identity. The SUCCESS
endpoint is real (diagnostic pack path). Mission-level greedy generation
and overwrite-on retention still need a decode-time gen-index diagnostic
and the preregistered replicate seed `250002`.

## Licensed next

1. Replication seed 250002 under the same frozen protocol.
2. Decode-time gen-index diagnostic for greedy / query-swap / binding
   counterfactuals (does not reopen P11 gates).
3. No TEST. No promotion until replicate + greedy/binding pass.
