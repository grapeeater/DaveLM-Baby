# Selection repair P6: identity overwrite with Xavier Q/K

Protocol id: `V010_SELECTION_REPAIR_P6_OVERWRITE_XAVIER`
Status: **preregistration — frozen before schedule generation and before any
optimizer update**

Licensed by P5 **REGRESSION**. P5's identity overwrite with **zero** Q/K
left `∂scores/∂W_q = 0` when `W_k = 0`. Mass stayed uniform 0.011. This
protocol is the same operator with **Xavier Q/K** so the pointer Jacobian
exists at step 0. Gate bias remains −4. P2 not launched. No TEST.

## 1. Hypothesis

**H1.** Xavier-initialized Q/K let `L_ptr` move overwrite-head mass onto the
query. After gate-on at 201, identity overwrite implements the D3b replace
and long-gap gold rises.

Matched control: same module, same Xavier init, **λ_ptr = λ_gate = 0**.

## 2. Why this class

Not P5 λ. Not P4 matching. The failed P5 functional was a saddle, not the
overwrite operator. Unit test: zero Q/K pointer grad is 0; Xavier is not.

## 3. Identity

Same parent, S2 eval, diet, λ schedule, futility, SUCCESS/REGRESSION bars as
P5, except:

| item | value |
|---|---|
| train seed | `200001` (replicate `200002` generated, not launched unless SUCCESS) |
| data seed | `200100` |
| bootstrap | `200300` |
| qk_init | **xavier** |
| denied | panels, S1, S2, D1b, P1/P3/P4/P5 exact inputs |

Step-0 long-gap hits must stay within 8 of 74 (Xavier + gate −4). Recorded.

## 4. Frozen

P5 decision rules are reused unchanged, including 200-update mass futility
< 0.20 and 400-update combined futility. These rules are not to be changed
after results are seen.
