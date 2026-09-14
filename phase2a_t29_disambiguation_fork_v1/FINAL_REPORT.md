# T29 FINAL REPORT

## Study classification
`T29_REPRESENTATION_SUCCESS_OUTPUT_FAIL`

Three seeds (860001–860003) completed U1000. Representation replicated 3/3. Native exact and native forced-choice missed the frozen bars on 3/3. TEST / FINAL / sacred remained sealed.

## Scientific change
Inherited T24 teacher-forced stack (Phase1G parent, full-span CE, pointer, candidate margin, 24-name unlikelihood, 9:1 language, two-phase 1000-update recipe). T28 suffix-vocab margin off. Added `L_fork`: hinge at the first TRAIN-identity unique-commit token, `lambda_fork=0.5`, `margin=1.0`. Fork rule frozen from 16 TRAIN names only. No tokenizer change. No constrained decoding. No generated-prefix rollout.

## Runtime
Smoke: 11.4 s/QA update on CUDA (RX 9060 XT); `L_fork` used existing answer logits. U0 preflight reproduced parent DEV CE `1.2040123894810677`; binding intact. Watchdog sequential 860001→860003, 2026-09-14T02:48:26Z to 2026-09-14T06:05:48Z.

## Per-seed terminal results (DEV n=128)

| seed | DEV CE | gap | pointer | FC | exact+EOS | ptr reversals | complete ptr fam | fork-token acc | shared exact | unique exact | diverge | post-fork complete | TRAIN16 | binding |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 860001 | 1.254286 | 0.422 | 110/128 | 88/128 | 35/128 | 52/64 | 11/16 | 26/64 | 2/64 | 33/64 | 38 | 2/26 | PASS | PASS |
| 860002 | 1.254771 | 0.419 | 111/128 | 87/128 | 35/128 | 49/64 | 9/16 | 35/64 | 2/64 | 33/64 | 54 | 2/35 | 14/16 (not acq16_pass) | PASS |
| 860003 | 1.255805 | 0.423 | 114/128 | 88/128 | 35/128 | 56/64 | 12/16 | 30/64 | 0/64 | 35/64 | 45 | 0/30 | PASS | PASS |

Language held (CE ≤ 1.30 and ≤ U0+0.25; gap ≤ 0.5). Name-disjoint and template-disjoint pointer ≥ 48/64. Native FC < 96 and exact < 80 on all seeds.

## Vs T18 / T24 / T28
T28 exact 36/35/40. T29 exact **35/35/35**. Shared-prefix exact rates 3.1%/3.1%/0% did not rise above T28 4.7%. Unique-prefix 51.6%/51.6%/54.7% vs T28 53.1% (no material unique-prefix win; no collapse). First-token-correct-then-diverge 38/54/45 vs T28 39/40/45: not a replicated fall.

## Interpretation
Baby did not learn to win the native DEV fork in a way that recovers exact generation. Fork-token accuracy is mixed versus U0 (0.266) and does not convert: post-fork free completion is 8%/6%/0% on DEV. TRAIN/ACQ16 can complete forks when the identity is in-train; that does not transfer to DEV shared-prefix names (Omar/Opal/Sal/Skye), which were correctly excluded from the training rule.

This specific disambiguation-fork training hypothesis is **closed**. Do not raise lambda/margin and rerun. Tokenizer lineage remains paused. TEST remains sealed. T30 is not authorized.

## Provenance
- Parent SHA-256: `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`
- Tokenizer v0_7 SHA-256: `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`
- TEST seal: `SEALED_UNOPENED` `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`
- Bundle: `phase2a_t29_disambiguation_fork_v1`
