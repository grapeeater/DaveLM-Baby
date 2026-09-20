# P11 active-overwrite retention: PREFLIGHT_BLOCKED

Status: **PREFLIGHT_BLOCKED**. The protocol is frozen. Hashed P11 receipts
were verified. The three-arm evaluation was **not scored** because the
required local artifacts are absent from this checkout. No PASS or
REGRESSION is claimed. No TEST. Not a promotion of U16000.

Protocol:
[`design/V010_SELECTION_REPAIR_P11_RETENTION.md`](../design/V010_SELECTION_REPAIR_P11_RETENTION.md)
Preflight: `runs/selection_p11_retention/PREFLIGHT.json`
Adjudication: `runs/selection_p11_retention/ADJUDICATION.json`

## Verified P11 receipts (tracked)

| receipt | frozen result |
|---|---|
| P11 gen-only `250001` | SUCCESS |
| P11_DECODE every-step | H1_FAIL (`free_exact` 0) |
| P11_DECODE_FIRSTSTEP | H1_PASS (`free_exact` 71→102/215) |
| parent SHA | `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` |
| P11 treatment SHA | `369d95c5fdfafea6b270afc47e5a008d17415b7efda275f1aef522d2e4821157` |
| TEST / FINAL / SACRED | closed |

## Required comparison (not yet scored)

1. Authoritative parent v2R4 U16000, overwrite absent.
2. P11 `checkpoint_16800.pt`, overwrite attached, `gen_index` never set.
3. Same P11 checkpoint, overwrite on **first answer-token only**.

Frozen bars unchanged: retention drop 0.05 on primitive_induction,
primitive_keyed, short_keyed, rest_lock; value_absent / broken_context /
broken_order negative controls; language DEV CE hard-stop +0.20;
long-gap greedy Δ ≥ 0.10 with CI lo > 0.

## Missing local identities

| role | path | expected SHA-256 |
|---|---|---|
| parent checkpoint | `runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt` | `94b3a9da…17827` |
| P11 treatment | `runs/selection_p11/treatment_250001/checkpoint_16800.pt` | `369d95c5…21157` |
| S2 diagnostic | `runs/selection_s2/DIAGNOSTIC.json` | `5b63533f…dc446` |
| language DEV stream | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\LANGUAGE_DEV_STREAM.u16` | (presence required) |

These files are gitignored / machine-local (Fan Diesel). Inventing scores
without them would be a protocol breach. Frozen panels matched after the
existing CRLF/LF identity check. Isolation panels are present.

## What this is not

Not a retention PASS. Not a REGRESSION. Not a promotion. P11 SUCCESS and
first-step H1_PASS are unchanged. The next measurement, on the machine
that holds the hashed checkpoints, is:

```text
python -m src.baby_v010.selection_p11_retention run
```

## Licensed next

Run the frozen retention scorer against the hashed U16000 and P11
`checkpoint_16800` weights. Until that returns PASS or REGRESSION, do not
open TEST and do not promote Baby.
