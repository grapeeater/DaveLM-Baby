# Baby v0.10 Foundation v1R2 terminal adjudication

Date: 2026-09-15  
Protocol: `BABY_V010_FOUNDATION_V1R2`  
Seeds: `101001`, `101002`  
Parent checkpoint: none (fresh random initialization for both seeds)  
Protected material opened: false

## Decision

Foundation v1R2 **fails the frozen capability gates**. Baby v0.10 is not
foundationally ready, and no binding, QA, or instruction stage is authorized by
this result. The failure is preserved as a scientific result; thresholds and
evaluation panels were not changed after observing the result.

## Frozen gates and terminal observations

The preregistered requirements included language DEV CE <= 2.50; same-surface
novel first-token top-1 >= .90 and teacher-forced exact >= .80; held-out-surface
first-token top-1 >= .80 and free-running exact >= .70; unseen-length,
low-prior, changed-order, and distractor thresholds; broken-context free exact
<= .05; context-dependent lift; and two independent seeds.

| metric | seed 101001 | seed 101002 | gate |
|---|---:|---:|---:|
| language DEV CE | 2.4592 | 2.3648 | <= 2.50 |
| same-surface novel first top-1 | 0.0000 | 0.0104 | >= 0.90 |
| same-surface novel teacher-forced exact | 0.0000 | 0.0000 | >= 0.80 |
| held-out surface first top-1 | 0.0000 | 0.0000 | >= 0.80 |
| held-out surface free exact | 0.0000 | 0.0000 | >= 0.70 |
| unseen-length first top-1 | 0.0000 | 0.0000 | >= 0.80 |
| low-prior first top-1 | 0.0000 | 0.0313 | >= 0.80 |
| distractor first top-1 | 0.0000 | 0.0000 | >= 0.75 |
| broken-context free exact | 0.0000 | 0.0000 | <= 0.05 |
| novel distinct emitted tokens | 12 | 20 | >= 100 |

The negative-control result is not a pass: both intact and broken contexts
produce essentially no exact reproduction. The stored context-lift proxies
were small relative to the required behavioral effect (seed 101001 target-logit
difference 0.3402, margin difference 0.0318; seed 101002 target-logit
difference 0.4157, margin difference 0.4543; free-exact difference 0.0).

## Interpretation

Both runs were mechanically healthy: finite losses, valid checkpoints, fresh
initialization, and no protected-material access. Language modeling passed the
frozen language gate, while the structured objective did not produce a
context-dependent copy capability. The repeated low diversity of greedy output
and target ranks far from one are consistent with a shortcut/collapse toward a
small set of frequent structural tokens rather than identity transport.

This does **not** falsify the v0.9 T34 result or the architectural family. It
does falsify the v1R2 hypothesis that a fresh 60M model can acquire keyed
multi-token retrieval and induction from a mixed, high-entropy objective after
a short language warmup without an easier capability scaffold.

No checkpoint from either run will be used as a parent for the next scientific
iteration. The next iteration must start from fresh random initialization and
must preregister a staged objective that first establishes an elementary
contextual-copy primitive, then increases relational and surface difficulty.

## Source records

- `runs/foundation_v1r2_seed101001/metrics.jsonl`
- `runs/foundation_v1r2_seed101002/metrics.jsonl`
- `data_specs/V010_FOUNDATION_PROTOCOL_V1R2.json`
- `data_specs/V010_FOUNDATION_FREEZE.json`
- `design/V010_CAPABILITY_GATES.md`

