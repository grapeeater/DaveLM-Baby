# Selection repair P5: identity residual overwrite after block 0

Protocol id: `V010_SELECTION_REPAIR_P5_OVERWRITE`
Status: **preregistration — frozen before schedule generation and before any
optimizer update**

Licensed by D4 **OV_DEAD** plus D3b **SPLICED_IDENTITY**. P4/P3/P1 stay
REGRESSION. P2 not launched. Authoritative parent unchanged. No TEST.

## 1. Hypothesis

**H1.** D3b's sufficient write is replace, not OV-add. An identity-value
gated overwrite inserted on the block-0 residual,

`h ← h + g ⊙ (read − h)`, `read = attn @ h` (V = identity),

with a dedicated Q/K and per-position gate, will implement Q0R at eval when
attn[gen, query] → 1 and g[gen] → 1, and will raise long-gap gold selection.

Matched control: same module, same data, **λ_ptr = λ_gate = 0**. Gate bias
init **−4** so step-0 is approximately the parent.

## 2. Why this class

D4 one-hot L0 attention through existing OV missed both bars (Δg +0.074,
cosine 0.070). P3/P4 matching losses could not implement replace
(`‖g_copy‖/‖g_CE‖ = 0.077`). P1 pointer was max-over-layers at L10. This is
a new forward operator, not a new matching loss on the old operator.

## 3. Identity

| item | value |
|---|---|
| parent | U16000 SHA `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` |
| frozen eval | S2 diagnostic |
| train seed | `190001` (replicate `190002` generated, not launched unless SUCCESS) |
| data seed | `190100` |
| bootstrap | `190300` |
| λ_ptr | treatment **1.0**, control **0.0** |
| λ_gate | treatment **1.0 from update 201**, control **0.0**; updates 1–200 λ_gate **0** in both arms |
| site | hook on `blocks[0]` output, before block 1 |
| values | identity (no W_V / W_O) |
| gate bias init | −4.0 |
| min gap | 2 |
| batch | 8 full keyed, 3 primitive keyed, 5 full induction |
| language | 0.20 |

`BabyVNextLM` weights load from the parent. Overwrite parameters are new.
Parent checkpoints remain loadable. Eval uses the hook.

## 4. Loss (frozen)

`L_CE` = answer-span CE (language updates: CE only).

Keyed gap ≥ 2:

`L_ptr = mean −log(attn[gen, query] + ε)`

`L_gate = mean (1 − g[gen])²`  (treatment, updates ≥ 201 only)

`L = L_CE + λ_ptr L_ptr + λ_gate L_gate`

Denied: panels, S1, S2, D1b, P1/P3/P4 exact inputs.

## 5. Decision rules — fixed before seeing results

Primary: S2 gap ≥ 13 candidate argmax. Parent 74/215.

| verdict | rule |
|---|---|
| **SUCCESS** | treatment long-gap excess ≥ **+0.10** and (T−C) ≥ **+0.07** with body-bootstrap 95% CI lo > 0; no retention regression; post-overwrite L0 query-gen cosine ≥ **0.40** or gain ≥ **+0.15** |
| **MECHANISM SUPPORTED, DOSE INSUFFICIENT** | overwrite-head long-gap median query mass ≥ **0.50** or cosine ≥ **0.20** / gain ≥ **+0.10**; selection bars missed; no retention regression |
| **FALSIFIED** | train L_ptr fell ≥ 50% first50→last50, and overwrite-head long-gap mass still < **0.20**, and excess < +0.02 |
| **NULL** | neither selection, cosine, nor overwrite-head mass moved |
| **REGRESSION** | any retention drop > 0.05 vs parent |

Futility at **200**: stop if overwrite-head long-gap median query mass < **0.20**.
Futility at **400**: stop if excess gain < **+0.02** and cosine gain < **+0.05**
and overwrite-head mass < **0.50**. If mass ≥ 0.50, continue to 800.

Retention and negative controls: same numeric bars as P4.

Hard stops: language DEV CE > parent + 0.20; non-finite; disk < 10 GiB; 2 h/arm.

Step 0 must remain within 8 hits of 74 on the long-gap primary (else INVALID
init). That check is recorded, not a license to retune after seeing later
results.

## 6. Frozen

These rules are not to be changed after results are seen.
