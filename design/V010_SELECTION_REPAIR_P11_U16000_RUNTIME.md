# P11 U16000 runtime overwrite retention

Protocol id: `V010_SELECTION_REPAIR_P11_U16000_RUNTIME`
Status: **preregistration — frozen before measurement**

Corrects the model-state conflation in the earlier three-arm retention
draft (`V010_SELECTION_REPAIR_P11_RETENTION`), which loaded the full
P11 checkpoint `model_state_dict`. P11 gen-only training kept Baby frozen
(`baby_weights_frozen: true`); the licensed 71→102/215 first-step decode
result used **authoritative U16000 Baby weights** plus the learned
overwrite only.

This protocol holds Baby weights fixed at U16000 in both arms. The only
difference is runtime activation of the learned P11 overwrite.

No training. No TEST. Does not promote U16000. Does not use the P11-trained
Baby checkpoint question (that remains a separate future experiment).

## 1. Question

On authoritative v2R4 U16000, does first-step-only activation of the
learned P11 overwrite (a) reproduce the frozen long-gap behavioral lift
and (b) preserve frozen retention guards?

## 2. Arms (same Baby weights)

| arm | Baby weights | overwrite module | runtime activation |
|---|---|---|---|
| `off` | v2R4 U16000 | learned P11 overwrite attached | `gen_index` never set (identity) |
| `on` | same U16000 | same learned overwrite | first answer-token only |

Both arms load `overwrite_state_dict` from
`runs/selection_p11/treatment_250001/checkpoint_16800.pt`.
Neither arm loads P11 `model_state_dict`.

Language DEV CE has no answer-token decision: overwrite stays inert on
both arms during CE scoring.

## 3. Identity

| item | value |
|---|---|
| parent Baby | `runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt` SHA `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` |
| learned overwrite | `overwrite_state_dict` inside P11 `checkpoint_16800.pt` SHA `369d95c5fdfafea6b270afc47e5a008d17415b7efda275f1aef522d2e4821157` |
| long-gap rows | frozen S2 diagnostic, gap ≥ 13, n=215 |
| bootstrap | `250300` |
| frozen firststep anchor | init 71/215, treatment 102/215 (`V010_SELECTION_REPAIR_P11_DECODE_FIRSTSTEP`) |

Denied: TEST / FINAL / SACRED, training, P11 `model_state_dict` load,
threshold edits, promotion, `main` merge, every-step overwrite.

## 4. Endpoints

### Long-gap behavioral (greedy, not teacher-forced)

Same population and scorer as `selection_p11_decode` first-step path:
greedy span decode, exact match on full target, overwrite at step 0 only
when `on`.

Benefit bar: `on` − `off` greedy `free_exact` Δ ≥ **0.10** with
bootstrap 95% CI lower bound > 0, n=215.

Record comparison to frozen firststep receipt (init 71, treatment 102) as
diagnostic only — OFF here uses trained overwrite inert, not init
overwrite.

### Retention (unchanged bars)

Drop bar **0.05** vs `off` on `on`:

- `primitive_induction.first_top1`
- `primitive_keyed.first_top1`
- `short_keyed.free_exact`
- diagnostic `rest_lock`

Negative controls on `on` (absolute):

- `value_absent_same_surface_novel.free_exact` == 0
- `broken_context.free_exact` ≤ 0.05
- `broken_order.first_top1` ≤ 0.20

Language DEV CE hard-stop: `on` − `off` > **0.20** is retention fail.

## 5. Adjudication

**PASS** iff long-gap benefit holds **and** every required retention check
passes on `on` (reported on `off`).

**REGRESSION** if any required retention gate fails on `on`.

**STOP_NO_BENEFIT** if retention holds but benefit absent.

**PREFLIGHT_BLOCKED** if required hashed artifacts missing.

**REPRODUCTION_MISMATCH** is not a separate verdict; record if `on` ≠ 102
or `off` ≠ frozen init 71 in diagnostics. Do not tune to recover 102.

A PASS licenses deeper validation. It is **not** a promotion.

## 6. TEST / promotion

Closed. Not a promotion of U16000.
