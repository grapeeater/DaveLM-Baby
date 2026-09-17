# Selection repair P7: frozen Baby, overwrite-only at 1e-3

Protocol id: `V010_SELECTION_REPAIR_P7_OVERWRITE_ONLY`
Status: **preregistration — frozen before schedule generation and before any
optimizer update**

Licensed by P6 **REGRESSION**. Xavier Q/K unstuck the P5 saddle, but
(1) overwrite mass stayed 0.006 at v2R4 `HIGH_LR`, and (2) λ=0 controls
keep dropping `primitive_induction` whenever Baby weights keep training.
P2 not launched. No TEST.

## 1. Hypothesis

**H1.** Training **only** the identity overwrite (Xavier Q/K, gate bias −4)
at **1e-3**, with Baby weights frozen, will raise overwrite-head query mass
and, after gate-on at 201, implement the D3b replace without the diet
induction drop.

Matched control: same frozen Baby, same overwrite init, **λ_ptr = λ_gate = 0**.

## 2. Why this class

Not P6 λ. Not unfreezing Baby. Two P6 facts license both changes at once:
the new module did not move at 3.75e-5, and the backbone update is the
retention failure shared by every λ=0 control since P3.

## 3. Identity

Same parent, S2 eval, diet, λ_ptr/λ_gate schedule, SUCCESS/REGRESSION bars
as P6, except:

| item | value |
|---|---|
| train seed | `210001` (replicate `210002` generated, not launched unless SUCCESS) |
| data seed | `210100` |
| bootstrap | `210300` |
| Baby weights | **frozen** |
| overwrite LR | **1e-3** |
| qk_init | xavier |
| denied | panels, S1, S2, D1b, P1/P3/P4/P5/P6 exact inputs |

Step-0 long-gap hits must stay within 8 of 74.

## 4. Frozen decision rules

Reuse P6/P5 bars, including 200-update mass futility < 0.20. These rules
are not to be changed after results are seen.
