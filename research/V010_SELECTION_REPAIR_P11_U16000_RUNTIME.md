# P11 U16000 runtime overwrite retention: PREFLIGHT_BLOCKED

Status: **PREFLIGHT_BLOCKED**. The corrected primary protocol is frozen.
The U16000 OFF-vs-ON evaluation was **not scored** on this machine because
required local artifacts are absent. No PASS, REGRESSION, or promotion claim.

Protocol:
[`design/V010_SELECTION_REPAIR_P11_U16000_RUNTIME.md`](../design/V010_SELECTION_REPAIR_P11_U16000_RUNTIME.md)

Preflight: `runs/selection_p11_u16000_runtime/PREFLIGHT.json`  
Adjudication: `runs/selection_p11_u16000_runtime/ADJUDICATION.json`

## Why this protocol exists

The earlier three-arm retention draft (`V010_SELECTION_REPAIR_P11_RETENTION`)
loaded the full P11 checkpoint `model_state_dict`, conflating runtime overwrite
activation with the separate P11 checkpoint bundle. The licensed 71→102/215
first-step decode result used **authoritative U16000 Baby weights** plus the
learned overwrite only (`selection_p11_decode.load_arm`).

This protocol holds Baby at U16000 in both arms and changes only overwrite
activation. It refuses to load P11 `model_state_dict` if it differs from U16000.

## Model states (explicit)

| state | path | SHA-256 | loaded? |
|---|---|---|---|
| U16000 Baby | `runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt` | `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` | both arms |
| learned overwrite | `overwrite_state_dict` in `runs/selection_p11/treatment_250001/checkpoint_16800.pt` | file `369d95c5fdfafea6b270afc47e5a008d17415b7efda275f1aef522d2e4821157` | both arms |
| P11 Baby bundle | same `.pt` `model_state_dict` | — | **refused** if ≠ U16000 |

## Arms (not yet scored)

1. **OFF** — U16000 + learned overwrite attached, `gen_index` never set.
2. **ON** — same U16000 + same overwrite, first answer-token only.

Long-gap behavioral endpoint uses **greedy decode** (not teacher-forced), same
population as firststep decode (S2 diagnostic, gap ≥ 13, n=215).

Frozen firststep anchor (diagnostic only): init 71/215, treatment 102/215.
OFF here is **not** init overwrite; do not require OFF == 71 for validity.

## Missing local identities

| role | path |
|---|---|
| parent checkpoint | `runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt` |
| overwrite checkpoint | `runs/selection_p11/treatment_250001/checkpoint_16800.pt` |
| S2 diagnostic | `runs/selection_s2/DIAGNOSTIC.json` |
| language DEV stream | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\LANGUAGE_DEV_STREAM.u16` |

## Fan Diesel command (when artifacts present)

```text
python -m src.baby_v010.selection_p11_u16000_runtime run --device cuda
```

Do **not** use `selection_p11_retention run` for tonight's primary experiment;
that draft mixed P11 checkpoint Baby weights.

## What this is not

Not a PASS. Not a REGRESSION. Not a promotion. TEST / FINAL / SACRED closed.
The separate P11-trained-checkpoint question is explicitly deferred.
