# Selection repair P4: direct L0 query-residual copy

Protocol id: `V010_SELECTION_REPAIR_P4_EARLY_COPY`
Status: **preregistration — frozen before schedule generation and before any
optimizer update**

Licensed by P3 **REGRESSION** plus D3b **SPLICED_IDENTITY**. P3 stays
REGRESSION. P2 not launched. Authoritative parent unchanged. No TEST.

## 1. Hypothesis

**H1.** D3b's sufficient write is **replace** `h0[gen] ← h0[query]`. P3
InfoNCE only required query ≳ competitor at L0 and left median cosine 0.04.
A direct copy loss `1 - cosine(h0[gen], sg(h0[query]))` on gap ≥ 2 will raise
that cosine and long-gap gold selection.

Matched control: same data, same CE, **λ = 0**.

Diet change vs P3 is in **both** arms: more induction, because P3's λ=0
control already dropped `primitive_induction` first-top1 0.297→0.188 in 400
updates. That is not a second treatment.

## 2. Why this class

Not P3 λ. Not P2 final-residual. Not P1 attention. The failed P3 functional
was contrastive ranking. The licensed functional is the D3b splice.

## 3. Identity

| item | value |
|---|---|
| parent | U16000 SHA `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` |
| frozen eval | S2 diagnostic |
| train seed | `180001` (replicate `180002` generated, not launched unless SUCCESS) |
| data seed | `180100` |
| bootstrap | `180300` |
| λ | treatment **1.0**, control **0.0** |
| bind site | block 0 output |
| min gap | 2 |
| batch | 8 full keyed, 3 primitive keyed, 5 full induction |
| language | 0.20 |

`BabyVNextLM.forward` unchanged. Aux is a block-0 hook.

## 4. Loss (frozen)

`L_CE` = answer-span CE.

For keyed gap ≥ 2: `L_copy = mean_i (1 - cosine(h0[gen], sg(h0[query])))`

`L = L_CE + λ L_copy`

No competitor negatives. Language / induction / gap ≤ 1: `L_copy = 0`.

Denied: panels, S1, S2, D1b, P1 exact inputs, P3 exact inputs.

## 5. Decision rules — fixed before seeing results

Primary: S2 gap ≥ 13 candidate argmax. Parent 74/215.

| verdict | rule |
|---|---|
| **SUCCESS** | treatment long-gap excess ≥ **+0.10** and (T−C) ≥ **+0.07** with body-bootstrap 95% CI lo > 0; no retention regression; L0 query-gen cosine ≥ **0.40** or gain vs parent ≥ **+0.15** |
| **MECHANISM SUPPORTED, DOSE INSUFFICIENT** | L0 cosine ≥ **0.20** or gain ≥ **+0.10**, selection bars missed; no retention regression |
| **FALSIFIED** | train `L_copy` fell ≥ 50% first50→last50, and L0 cosine still < **0.10**, and excess < +0.02 |
| **NULL** | neither selection nor L0 cosine moved |
| **REGRESSION** | any retention drop > 0.05 vs parent |

Futility at 400: stop if excess gain < **+0.02** and L0 cosine gain < **+0.05**.
If cosine moved, continue to 800.

Retention and negative controls: same numeric bars as P3.

Hard stops: language DEV CE > parent + 0.20; non-finite; disk < 10 GiB; 2 h/arm.

## 6. Frozen

These rules are not to be changed after results are seen.
