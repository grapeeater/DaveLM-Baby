# Tokenizer repair results

Classification: `NATIVE_BEAM_DOES_NOT_CLOSE_SHARED_PREFIX`. Constrained 8-name rerank is `NON_NATIVE` and is **not** a Phase 2A success.

TEST loaded: **false**. Tokenizer changed: **false**. T29 launched: **false**.

Greedy exact reproduced the post-T28 panel: **111/384** (seeds 36/35/40 of 128). Shared-prefix **9/192 (4.7%)**, unique **102/192 (53.1%)**.

## A1 — inventory-free beam (width 4)

| Split | Greedy exact | Beam exact |
|---|---:|---:|
| All | 111/384 (28.9%) | 118/384 (30.7%) |
| Shared first-token DEV names | 9/192 (**4.7%**) | 8/192 (**4.2%**) |
| Unique first-token DEV names | 102/192 (53.1%) | 110/192 (57.3%) |

Beam moved a few Ava rows (14→22/48). Sal 6→5, Skye 2→3, Omar 1→0, Opal 0→0, Wes 0→0. Shared-prefix gap **did not close**. Free search still prefers Salt/Sky/Walt-class completions.

Per-seed exact greedy vs beam: 36→41, 35→37, 40→40.

## A2 — 8-name DEV inventory rerank (non-native)

| Split | Exact |
|---|---:|
| All | 237/384 (61.7%) |
| Shared | 101/192 (**52.6%**) |
| Unique | 136/192 (70.8%) |

Sal 40/48, Skye 29/48, Omar 22/48, Opal 10/48. This shows complete-string scoring among DEV names can recover much of the Sal/Skye collision **when the decoder is not free**. It remains multiple-choice over the DEV inventory.

**Wes remains 0/48** even here. Among the eight DEV strings, Wes never wins mean logprob. That is not greedy prefix-collapse onto Walt (Walt is outside the eight). Wes is a remaining scoring/representation failure under constrained native-name ranking.

Seeds: 83/76/78 of 128. For comparison, T28 in-row 2-way forced-choice was 88/82/82 of 128 (easier foil set).

## Decision

A1 is **not** a mouth repair for the 4.7% vs 53.1% gap. A2 is **not** native generation and does not fix Wes. v0_7 stays. No tokenizer candidate was built. No full retrain. Owner review is required before Option B/C/D training or any T29-class treatment.

Language and pointer were not re-run; T28 already recorded language/binding PASS and pointer 108–116/128 on these checkpoints.
