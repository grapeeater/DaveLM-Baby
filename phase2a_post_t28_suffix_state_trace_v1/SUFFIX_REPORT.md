# Post-T28 suffix hidden-state trace

Status: **COMPLETE — `GOLD_PREFIX_DIRECT_DECODABILITY_WITH_FREE_PREFIX_COLLAPSE`**.

The frozen population was the 124 DEV seed-row cases classified
first-token-correct-then-diverge. It produced 419 suffix decision states in each
of the gold-prefix and free-running-prefix conditions. Exact-success and
wrong-first-token internal traces were not in the frozen protocol and remain
`NOT_RUN_PROTOCOL_NOT_FROZEN`.

Baby's final normalization was applied before the frozen LM readout at every
intermediate site. Layer 0 is the embedding residual; layer 1 follows block 0;
layer 12 follows block 11 and is the native final path.

| Site | Gold-prefix target top-1 | Free-prefix target top-1 | Gold mean rank | Free mean rank |
|---|---:|---:|---:|---:|
| after block 4 | 2.4% | 0.0% | 163.1 | 246.6 |
| after block 5 | 18.9% | 0.5% | 125.6 | 231.1 |
| after block 6 | 29.8% | 1.7% | 86.6 | 187.3 |
| after block 7 | 60.4% | 20.3% | 47.8 | 148.2 |
| after block 8 | 65.4% | 24.1% | 50.5 | 83.6 |
| after block 9 | 66.3% | 26.3% | 66.4 | 85.9 |
| after block 10 | 70.2% | 29.1% | 35.8 | 52.7 |
| after block 11/final | 66.6% | 26.5% | 27.1 | 41.8 |

The target is not directly decodable through the frozen final readout in early
layers. Under a gold answer prefix, direct decodability emerges strongly around
block 7 and remains high through the late stack. Under a divergent generated
prefix it is much weaker. This does not prove information is absent earlier, nor
that Baby causally uses any intermediate directly decodable state.

At suffix step 1, gold and generated prefixes are identical and results match
exactly (55.6% top-1). Once a wrong generated token enters the prefix, the gap
appears: at steps 2/3/4 gold-prefix top-1 is 61.3%/70.2%/100%, versus
22.6%/5.6%/14.9% free-running. Only 14/198 (7.1%) post-error states put the
correct target back at top-1, so limited token-level self-recovery exists but
errors usually compound.

The final gold/free top-1 rates replicated by seed: 64.9%/26.0%, 68.1%/27.4%,
and 66.7%/26.1%.

