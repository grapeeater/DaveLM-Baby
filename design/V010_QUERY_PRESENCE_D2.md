# D2: Query presence on the P1 treatment checkpoint

Protocol id: `V010_QUERY_PRESENCE_D2`
Status: **preregistration — frozen before any forward on a P1 checkpoint**

Read-only. No optimizer. No TEST. Gate L/C/R unchanged. Authoritative parent
unchanged (still v2R4 U16000). D1 stays INVALID. D1b stays B and is not
rescored. P1 stays REGRESSION. This is not another λ, not a T1 resume, and not
a sidecar.

P1 showed that a gen-position pointer can be trained (long-gap query mass
0.04 → 0.93 on the S2 diagnostic) while greedy candidate selection stays at
chance. D2 asks whether that pointer **wrote query identity into the residual
the unembed sees**, or only moved attention weights.

## 1. Hypothesis

**H_WEIGHTS (leading).** On the frozen D1b twins, the P1 treatment checkpoint
is D1b-**COMPOSITION**: long-gap query-tracking ≥ 0.50 and the gen residual
stays query-invariant (cosine ≥ 0.999, patch does not meet A). The matched
λ=0 control remains D1b-**B**. Attention looks; the OV path does not write
key identity.

**H_RESIDUAL.** Treatment is D1b-**A**: patching twin *i*'s gen residual into
twin *j* flips candidate argmax toward *i*'s gold at gap ≥ 13. Then identity
arrived and unpatched greedy still did not use it (P1 selection stayed at
chance).

**H_STILL_B.** Treatment is still D1b-**B** on these twins. Then the P1 pointer
did not transfer off the S2 diagnostic.

## 2. Identity

| item | value |
|---|---|
| authoritative parent | U16000 SHA `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` (not scored here) |
| treatment ckpt | `runs/selection_p1/treatment_150001/checkpoint_16800.pt` SHA `5c8296ef759543dd608c18b8048ccce4bd4fcf8de7a6183b2d84e5f53d4fbafa` |
| control ckpt | `runs/selection_p1/control_150001/checkpoint_16800.pt` SHA `3f005e19da8db071a763eb66ef0cb6f7e59ca63cf55078eeec9fad2338cd2fc1` |
| diagnostic | frozen D1b `runs/query_presence_d1b/DIAGNOSTIC.json` SHA `b7260e6761bbedecacc6f26801b54141d6539b9ecfe76ff72c34c2e32ce76c57` |
| device | `cuda` / RX 9060 XT |
| attention | `reference` |
| optimizer | none |

D1b twins were denied from P1 training. This is a held-out presence set, not a
train-set rescore.

## 3. Measurements

Same M1–M3 as D1/D1b, independently on each checkpoint. D1b eligible-flip
statistics are required. D1b and P1 JSON receipts are read, never rewritten.

## 4. Frozen decision rules

### Per-arm instrument (else that arm is INVALID)

Exactly the D1b validity bars, with checkpoint SHA in place of parent SHA:

1. Checkpoint SHA matches the frozen P1 receipt. Diagnostic SHA matches the
   D1b manifest. Protected false. Twins query-only. SDPA vs reference max-abs
   < 1e-3. Update 16800.
2. Gap ≤ 1 median final residual cosine ≤ **0.99**.
3. Gap ≤ 1 prev-tracking-head fraction ≥ **0.90**.
4. Gap ≤ 1 eligible `flip_toward_donor` at `final` or best `block_out` ≥ **0.25**,
   `n_eligible` ≥ **50**.

D1b long-gap bars are applied unchanged: A, then COMPOSITION, then B, else MIXED.

### Headline (both arms valid; else INVALID)

Precedence: **RESIDUAL_WRITE**, then **WEIGHTS_ONLY**, then **STILL_B**, then
**MIXED**.

| headline | rule |
|---|---|
| **RESIDUAL_WRITE** | treatment verdict **A** |
| **WEIGHTS_ONLY** | treatment **COMPOSITION** and control **B** or **COMPOSITION** |
| **STILL_B** | treatment **B** and control **B** |
| **MIXED** | anything else valid |

Raw residual cosines, flip rates, and tracking fractions are reported for both
arms and are not optional. They do not move the bars.

## 5. What this does not authorize

Training, a new λ, T1 resume, sidecar, moving D1/D1b/P1 bars, opening TEST,
promoting either P1 checkpoint, or renaming P1's REGRESSION.

## 6. Outputs

`runs/query_presence_d2/treatment/QUERY_PRESENCE.json`,
`runs/query_presence_d2/control/QUERY_PRESENCE.json`,
`runs/query_presence_d2/ADJUDICATION.json`, local hashed `ROWS_*.json`,
`research/V010_QUERY_PRESENCE_D2.md`.

## 7. Frozen

These rules are not to be changed after seeing D2 numbers.
