# SF12 prospective preflight

Status: **PASS**

## Hypothesis and sole variable

SF12 tests whether a dedicated teacher-anchored retention signal for the four trained answer-token classes contains the replicated D3 increase while the frozen SF11 widening lesson remains learnable. The sole scientific change is `R_name`; the experiment does not alter or test retention-corpus coverage.

At each of the same 160 frozen KL positions used by SF11, the controller adds:

`mean ReLU(sum_N P_student - sum_N P_teacher - 0.001)`

with `N={314,536,925,512}` and weight `1.0`. It reuses the same distributions already computed for the authoritative full-vocabulary forward KL.

## Frozen design

- Parents: the three hash-verified SF8 low-dose update-200 successes.
- Continuation seeds: 87029, 87030, 87031.
- Curriculum and schedule: byte-identical to SF11; 32 widening plus 16 preservation items, 180 English and 20 binding updates.
- Optimizer, LR, scope, binding rehearsal, causal CE, forward KL, margin objective, evaluation panels, gates, and stopping logic: unchanged.
- Evaluations: update 0, 100, and 200 only.
- Locked historical transfer panels, FINAL, and sacred material: absent from the controller and unscored.

## Validation

- SF11 integrity chain and all external inputs verified.
- New parents and tokenizer verified; name token IDs matched exactly.
- Curriculum, schedule, KL pool, D3 selection, evaluator engine, and development panels are byte-identical to SF11.
- The augmented helper reproduced the authoritative KL value exactly on a deterministic mock and activated only on excess name mass.
- D3 occurs only in evaluation; `R_name` uses only the existing KL pool.
- Reporter classification passed outcome-blind synthetic cases.
- No checkpoint was loaded, no optimizer was created, and zero updates ran.

Warnings: the SF12 comparison to SF11 is historical rather than concurrent, and corpus-coverage/disjointness remains untested.
