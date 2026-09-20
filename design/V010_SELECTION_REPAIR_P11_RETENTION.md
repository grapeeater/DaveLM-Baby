# P11 active-overwrite retention

Protocol id: `V010_SELECTION_REPAIR_P11_RETENTION`
Status: **preregistration — frozen before measurement**

Licensed by P11_DECODE_FIRSTSTEP **H1_PASS**. First-step-only decode-time
overwrite recovered long-gap greedy `free_exact` 71→102/215. Every-step
overwrite destroyed continuation (`free_exact` 0). P11 training-time
retention used `evaluate_panels` / `score_items` **without** `gen_index`,
so those receipts are overwrite-OFF (identity write). This diagnostic
measures retention while the useful intervention is actually active.

No training. No new mechanism. No TEST. Does not reopen P11 gates. Does
not promote U16000.

## 1. Question

Does first-step-only gen-only overwrite preserve frozen retention guards
on the same P11 terminal checkpoint that produces the long-gap greedy
lift?

## 2. Arms (no new training)

| arm | weights | runtime overwrite |
|---|---|---|
| `parent` | v2R4 U16000 | none |
| `p11_off` | P11 `checkpoint_16800.pt` | attached, `gen_index` never set (identity) |
| `p11_on` | same P11 checkpoint | first answer-token decision only; off after that |

Parent vs `p11_off` isolates training/checkpoint damage. `p11_off` vs
`p11_on` isolates activating the overwrite. Do not invent a matched
control training arm.

Language DEV CE has no answer-token decision: overwrite stays inert
(identity) on all three arms.

## 3. Identity

| item | value |
|---|---|
| parent | `runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt` SHA `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` |
| P11 treatment | `runs/selection_p11/treatment_250001/checkpoint_16800.pt` SHA `369d95c5fdfafea6b270afc47e5a008d17415b7efda275f1aef522d2e4821157` |
| long-gap rows | frozen S2 diagnostic, gap ≥ 13, n=215 |
| bootstrap | `250300` |
| overwrite | `LocalSlotOverwrite(gen_only=True)` from the P11 checkpoint |

Denied: TEST / FINAL / SACRED, training, threshold edits, promotion,
`main` merge, every-step overwrite.

## 4. Frozen bars (unchanged)

Retention drop bar **0.05** on:

- `primitive_induction.first_top1`
- `primitive_keyed.first_top1`
- `short_keyed.free_exact`
- diagnostic `rest_lock`

Negative controls (absolute, unchanged):

- `value_absent_same_surface_novel.free_exact` == 0
- `broken_context.free_exact` ≤ 0.05
- `broken_order.first_top1` ≤ 0.20

Language DEV CE hard-stop: arm − parent > **0.20** is a retention fail.

Long-gap benefit (same first-step H1 bar): `p11_on` − `parent` greedy
`free_exact` Δ ≥ **0.10** with bootstrap 95% CI lower bound > 0, n=215.

## 5. Adjudication

**PASS** iff long-gap benefit holds **and** every required retention
check passes on `p11_on` (and is reported on `p11_off`).

**REGRESSION** if any required retention gate fails.

If `p11_off` passes and `p11_on` fails: attribute the regression to the
deployed first-step overwrite.

If both fail: distinguish checkpoint/training damage from
overwrite-specific damage.

If benefit is absent but retention holds: **STOP_NO_BENEFIT**. That is
not a retention regression and does not reopen P11 SUCCESS.

A PASS licenses the next validation stage. It is **not** a promotion.

## 6. Still required after PASS

Robust query sensitivity, QUERY→KEY→VALUE binding, held-out/new-body
generalization, varied long-gap generalization, replication under the
frozen lineage, owner authorization before TEST.

## 7. TEST / promotion

Closed. Not a promotion of U16000.
