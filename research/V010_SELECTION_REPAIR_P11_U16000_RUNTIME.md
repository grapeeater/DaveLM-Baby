# P11 U16000 runtime overwrite retention: REGRESSION

Status: **REGRESSION**. Same-U16000 OFF-vs-ON was scored on Fan Diesel.
First-step-only activation of the learned P11 overwrite reproduced the
frozen long-gap lift (71→102/215) and then failed a mandatory
active-overwrite retention gate. Not a promotion. Not mission complete.

Protocol:
[`design/V010_SELECTION_REPAIR_P11_U16000_RUNTIME.md`](../design/V010_SELECTION_REPAIR_P11_U16000_RUNTIME.md)

Preflight: `runs/selection_p11_u16000_runtime/PREFLIGHT.json` = **PREFLIGHT_OK**  
Adjudication: `runs/selection_p11_u16000_runtime/ADJUDICATION.json` = **REGRESSION**  
Arms: `runs/selection_p11_u16000_runtime/OFF.json`, `ON.json`

## Model states (scored)

| state | path | SHA-256 | loaded? |
|---|---|---|---|
| U16000 Baby | `runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt` | `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` | both arms |
| learned overwrite | `overwrite_state_dict` in `runs/selection_p11/treatment_250001/checkpoint_16800.pt` | file `369d95c5fdfafea6b270afc47e5a008d17415b7efda275f1aef522d2e4821157` | both arms |
| P11 Baby bundle | same `.pt` `model_state_dict` | identical to U16000 (max abs 0; `baby_weights_frozen: true`) | **not loaded** |

Device: `cuda` (AMD Radeon RX 9060 XT). No CPU fallback. Complete=true.

## Long-gap (greedy, n=215, gap ≥ 13)

| | OFF | ON |
|---|---:|---:|
| free_exact | 71 | **102** |
| first_correct | 73 | **105** |
| ON − OFF | | **+0.144** |
| bootstrap 95% CI | | **[+0.093, +0.197]** |

Benefit bar held (`Δ ≥ 0.10` and CI lo > 0). Diagnostic firststep anchor
71/102 matched; that match is **not** required for validity. Pair-bind
was not an endpoint of this protocol.

## Mandatory active-overwrite retention (ON vs OFF)

| gate | OFF | ON | bar | result |
|---|---:|---:|---|---|
| primitive_induction.first_top1 | 0.296875 | 0.156250 | drop ≤ 0.05 | **FAIL** (drop 0.141) |
| primitive_keyed.first_top1 | 1.000000 | 0.984375 | drop ≤ 0.05 | PASS |
| short_keyed.free_exact | 0.843750 | 0.828125 | drop ≤ 0.05 | PASS |
| diagnostic rest_lock | 0.976852 | 0.979167 | drop ≤ 0.05 | PASS |
| language DEV CE | 1.243005 | 1.243005 | ON−OFF ≤ 0.20 | PASS (overwrite inert; no answer-token) |
| value_absent_same_surface_novel.free_exact | 0 | 0 | == 0 | PASS |
| broken_context.free_exact | 0 | 0 | ≤ 0.05 | PASS |
| broken_order.first_top1 | 0.015625 | 0.015625 | ≤ 0.20 | PASS |

ON was active on panels with a first answer-token decision
(primitive_induction first_top1 and free_exact both moved). Language DEV
CE is labeled inert.

## Verdict

**B / REGRESSION / STOP.** The runtime overwrite is causally sufficient
for the long-gap greedy lift and is not safe against frozen induction
retention when it is actually on. Do not deploy. Do not promote U16000.
Do not train. Do not invent P12. TEST / FINAL / SACRED stayed closed.

The three-arm draft `V010_SELECTION_REPAIR_P11_RETENTION` was not run.
