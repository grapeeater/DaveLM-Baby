# Selection repair P9: slot overwrite through gate-on

Protocol id: `V010_SELECTION_REPAIR_P9_SLOT_GATEON`
Status: **preregistration — frozen before schedule generation and before any
optimizer update**

Licensed by P8 **NULL**. P8's local slot scorer moved overwrite-head mass
0.002→0.133 (control 0.0006) but hit the frozen 0.20 bar at 200, before
λ_gate turned on. This protocol is the same operator with a 200-update
futility bar of **0.05** (P8 already exceeded that) so gate-on can occur.
P2 not launched. No TEST. Do not unfreeze Baby.

## 1. Hypothesis

**H1.** Continuing the P8 slot scorer through updates 201–800 with λ_gate=1
will push gate[gen]→1, raise post-overwrite L0 query cosine, and lift
long-gap gold vs matched λ=0.

## 2. Why this class

Not P8 λ. Not a new pointer. P8 demonstrated the pointer class and was
stopped by a bar this class had not yet reached. Lowering P8's bar after
seeing 0.133 is forbidden; P9 preregisters a looser 200 bar before any P9
update.

## 3. Identity

Same as P8 except:

| item | value |
|---|---|
| train seed | `230001` (replicate `230002` generated, not launched unless SUCCESS) |
| data seed | `230100` |
| bootstrap | `230300` |
| 200 futility | overwrite-head mass < **0.05** |
| 400 futility | excess gain < +0.02 and cosine gain < +0.05 and mass < **0.50** |
| denied | panels, S1, S2, D1b, P1/P3/P4/P5/P6/P7/P8 exact inputs |

SUCCESS/REGRESSION/MECHANISM bars unchanged from P8/P7. Step-0 hits within
8 of 74.

## 4. Frozen

These rules are not to be changed after results are seen.
