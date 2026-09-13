# T14X LATE-BASE FAILURE DECOMPOSITION — FINAL

Status: **COMPLETE, read-only.** TEST not loaded. No training. Did not parent T14 checkpoints.

Question: within trainable late base, do reversal gain and language cost occupy different modules?

## Drift vs Phase1G
Largest per-parameter RMS is always `embed.pos`, `language_head`, `embed.token` (T10 > T14 > T12/T13). That tracks language-token update volume, not reversal. Block RMS is smaller; drift rank does not localize the representation circuit.

## Gradient conflict (trainable late-base grads; pointer is parameter-free)
On language-safe 730002 U750:
- **Block 4 attn/mlp/norm aligned** with language (ptr-vs-lang cos +0.12 to +0.16).
- **Blocks 8–11 norms and 10–11 MLP conflict** (ptr-vs-lang cos −0.16 to −0.28).
- `language_head` has **no pointer gradient** (pointer uses `forward_hidden` only).

730001 near-miss shows the same late-block conflict pattern, weaker alignment in block 4.

## Phase1G reversion (730001 U750 near-miss; baseline 110 ptr / 46 rev / CE 1.276 / gap 0.481)

| patch | ptr | rev | fam | nd | td | CE | gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 110 | 46 | 11 | 62 | 48 | 1.276 | 0.481 |
| revert embed | 110 | 46 | 11 | 62 | 48 | 1.276 | 0.478 |
| revert language_head | 110 | 46 | 11 | 62 | 48 | 1.272 | 0.441 |
| revert attn 4–11 | 96 | 37 | 7 | 58 | 38 | 1.247 | 0.432 |
| revert mlp 4–11 | 85 | 27 | 3 | 51 | 34 | 1.222 | 0.355 |
| revert blocks 4–7 | 73 | 20 | 0 | 40 | 33 | 1.244 | 0.412 |
| revert blocks 8–11 | 107 | 43 | 9 | 61 | 46 | 1.232 | 0.370 |
| revert **block 4 only** | **67** | **20** | **0** | 34 | 33 | 1.267 | 0.468 |
| revert block 10 | 111 | 47 | 11 | 62 | 49 | 1.257 | 0.454 |

Block 4 is necessary for reversal and cheap for language. Blocks 8–11 cost language and carry little reversal.

## 730002 U750 (language-safe representation hit; baseline 112/49 / CE 1.276 / gap 0.469)

| patch | ptr | rev | fam | nd | td | CE | gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 112 | 49 | 10 | 60 | 52 | 1.276 | 0.469 |
| revert language_head | 112 | 49 | 10 | 60 | 52 | 1.273 | 0.428 |
| revert attn 4–11 | 100 | 36 | 6 | 54 | 46 | 1.248 | 0.422 |
| revert mlp 4–11 | 87 | 28 | 1 | 47 | 40 | 1.221 | 0.352 |
| **revert blocks 8–11** | **115** | **51** | **12** | 60 | **55** | **1.234** | **0.367** |

Restoring blocks 8–11 to Phase1G **improves both** language and representation on the only T14 success seed. That is a post-hoc patch, not a 2/3 trained recipe. Do not parent 730002 or the patched weights.

## Verdict
Relational gain and language cost **localize to different late-base modules**:
- Gain: blocks **4–7**, especially **block 4**
- Language cost: blocks **8–11** (and whole late MLP when taken together)

Targeted freezing is justified. Structural isolation (adapter/LoRA) is not required yet.

## Next
T15: Phase1G parent, fact-clause, 9:1, lr 3.75e-5, 750 updates, seeds 740001–740003. **Freeze binding + blocks 0–3 + blocks 8–11.** Train blocks 4–7 + embed + final_norm + language_head. Tests whether the 4–7 pathway acquires language-safe representation without the 8–11 language tax. Do not parent T14.
