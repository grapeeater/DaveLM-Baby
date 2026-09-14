# Tokenizer replacement decision

## Classification

**`TOKENIZER_REPLACEMENT_NOT_JUSTIFIED`**

## Why

The frozen material rule required DEV shared-first-token rate to fall by at least half (0.500 → ≤ 0.250), TRAIN collisions to fall as well, disambiguation not to worsen, and Phase1G compression to stay within +25% of v0_7.

- DEV shared-first-token rate stayed **0.500** on every serious candidate (Omar/Opal and Sal/Skye). Title-case upsampling made it **worse** (0.625).
- No DEV name became a single token.
- Group-conditional disambiguation stayed 1.50 or got worse (1.75).
- TRAIN collision reduction (0.750 → 0.375 on some vocabs) did **not** generalize to DEV. That is expected: TRAIN names were in the BPE corpus; DEV names were not, and we correctly refused to add them.
- Language bytes/token worsened from 2.201 to 3.45–4.00.

v0_7 remains the best tokenizer in the frozen ranking.

## What this does and does not mean

Option C showed that unused appended IDs do not fix greedy. This study shows that a **general** retrained BPE on authorized TRAIN text, at 1536/2048/4096 and with whitespace or title-case variants, does **not** remove the DEV shared-prefix geometry.

It does **not** prove that no tokenizer on earth could help. It does prove that the authorized, prospective, non-DEV-tuned replacement sweep failed the frozen gate.

Therefore:

- **Do not** recommend full 61.5M retrain
- **Do not** launch T29
- **Do not** overwrite v0_7
- **STOP** on tokenizer replacement as an autonomous next step

A future tokenizer study would need a different authorized corpus or a different frozen family, plus owner approval. Hand-adding Sal/Skye/Wes remains forbidden.

## Operator flags

- Full retrain recommended: **NO**
- Full retrain authorized: **NO**
- Fresh lineage required *if* one ever replaced the tokenizer: yes, but replacement is not justified
- Owner decision required: **YES** (what to pursue instead of tokenizer replacement)
- TEST / FINAL / sacred: **SEALED**
- T29: **NOT AUTHORIZED / NOT LAUNCHED**
