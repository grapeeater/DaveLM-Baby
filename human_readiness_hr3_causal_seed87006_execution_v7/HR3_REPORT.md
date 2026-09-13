# HR-3 block-3 causal scope-relaxation report

## Classification

**HR3_FAILED_MACHINE_READINESS_GATES_BINDING_PRESERVED**

HR-3 completed 500/500 updates from the verified Pilot 1 parent. It did not meet the frozen readiness development gates. Both nonsacred binding pools independently passed their preservation gates. No FINAL or sacred material was accessed.

## Provenance

- Bundle: `C:\DaveLM-CADAVER\human_readiness_hr3_block3_causal_seed87006_v7`
- Bundle receipt SHA-256: `7d147dd92af73d3d1979c5b012b2fbd8e258efad8ff9650f8c3de244be862df9`
- Run directory: `C:\DaveLM-CADAVER\human_readiness_hr3_causal_seed87006_execution_v7`
- Parent: Pilot 1, SHA-256 `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb`
- HR-3 checkpoint 500: `C:\DaveLM-CADAVER\human_readiness_hr3_causal_seed87006_execution_v7\checkpoint_500.pt`, SHA-256 `23726ce76ff789bbe05d44340952a1763b7231584b0c91ff8e15f5c64e6ef93d`
- Training: seed 87006, 500 updates (450 English / 50 binding), block 3 newly trainable on English; blocks 0-2 and T13 localization/retrieval protected.
- FINAL readiness battery: sealed and untouched.

## Aligned DEV trajectory

| checkpoint | loss | perplexity |
|---|---:|---:|
| update 0 | 3.3907 | 29.69 |
| update 100 | 3.0041 | 20.17 |
| update 250 | 2.9492 | 19.09 |
| update 500 | 2.8814 | 17.84 |

Pilot 1 aligned loss was 3.3907 (PPL 29.69). Corrected causal HR-1 update 500 was 2.8305 (PPL 16.95). HR-3 update 500 was 2.8814 (PPL 17.83), so the scope relaxation did not outperform aligned HR-1 on this fixed loss diagnostic.

## Readiness DEV comparison

| checkpoint | fact correct | reversals | complete families | fact greedy | instruction | continuity | generation non-EOS | automatic non-degenerate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pilot1_parent | 13/24 (54.2%) | 1/12 (8.3%) | 0/6 | 0/24 (0.0%) | 7/12 (58.3%) | 6/12 (50.0%) | 20/20 (100.0%) | 7/20 (35.0%) |
| HR1_causal_aligned_500 | 12/24 (50.0%) | 0/12 (0.0%) | 0/6 | 0/24 (0.0%) | 6/12 (50.0%) | 5/12 (41.7%) | 14/20 (70.0%) | 14/20 (70.0%) |
| HR3_block3_update_100 | 11/24 (45.8%) | 0/12 (0.0%) | 0/6 | 0/24 (0.0%) | 7/12 (58.3%) | 6/12 (50.0%) | 14/20 (70.0%) | 10/20 (50.0%) |
| HR3_block3_update_250 | 12/24 (50.0%) | 0/12 (0.0%) | 0/6 | 0/24 (0.0%) | 6/12 (50.0%) | 5/12 (41.7%) | 14/20 (70.0%) | 12/20 (60.0%) |
| HR3_block3_update_500 | 12/24 (50.0%) | 0/12 (0.0%) | 0/6 | 0/24 (0.0%) | 6/12 (50.0%) | 5/12 (41.7%) | 14/20 (70.0%) | 13/20 (65.0%) |

Facts are scored by positive correct-minus-distractor conditional log-likelihood margin; ties fail. Reversal success requires both members of a matched assignment pair to be correct. Greedy exact requires the candidate token sequence followed immediately by EOS.

## Binding preservation

| checkpoint | pool | answer exact | BOTH_DISTINCT | collapse | complete quartets | strict reversal | gate |
|---|---|---:|---:|---:|---:|---:|---|
| Pilot1_parent | pilot0 | 80/80 | 80/80 | 0 | 20/20 | 40/40 | PASS |
| Pilot1_parent | pilot1 | 80/80 | 80/80 | 0 | 20/20 | 40/40 | PASS |
| HR1_causal_aligned_500 | pilot0 | 80/80 | 80/80 | 0 | 20/20 | 40/40 | PASS |
| HR1_causal_aligned_500 | pilot1 | 80/80 | 80/80 | 0 | 20/20 | 40/40 | PASS |
| HR3_block3_update_100 | pilot0 | 80/80 | 80/80 | 0 | 20/20 | 40/40 | PASS |
| HR3_block3_update_100 | pilot1 | 80/80 | 80/80 | 0 | 20/20 | 40/40 | PASS |
| HR3_block3_update_250 | pilot0 | 80/80 | 80/80 | 0 | 20/20 | 40/40 | PASS |
| HR3_block3_update_250 | pilot1 | 80/80 | 80/80 | 0 | 20/20 | 40/40 | PASS |
| HR3_block3_update_500 | pilot0 | 80/80 | 80/80 | 0 | 20/20 | 40/40 | PASS |
| HR3_block3_update_500 | pilot1 | 80/80 | 80/80 | 0 | 20/20 | 40/40 | PASS |

Both HR-3 checkpoints evaluated at updates 100, 250, and 500 passed both pools (80/80 answer, 80/80 BOTH_DISTINCT, zero collapse).

## Frozen machine-gate outcome

- **Pilot1_parent**: `FAIL_MACHINE_OR_BINDING_GATES`; all machine gates: `False`; binding gates: `True`.
- **HR1_causal_aligned_500**: `FAIL_MACHINE_OR_BINDING_GATES`; all machine gates: `False`; binding gates: `True`.
- **HR3_block3_update_100**: `FAIL_MACHINE_OR_BINDING_GATES`; all machine gates: `False`; binding gates: `True`.
- **HR3_block3_update_250**: `FAIL_MACHINE_OR_BINDING_GATES`; all machine gates: `False`; binding gates: `True`.
- **HR3_block3_update_500**: `FAIL_MACHINE_OR_BINDING_GATES`; all machine gates: `False`; binding gates: `True`.

No checkpoint met the fact, reversal, complete-family, greedy, instruction, continuity, and generation criteria together. Human-review fields were left pending and did not affect this machine-gate classification.

## HR-3 update-500 raw generation diagnostic

Raw responses below are copied verbatim from the frozen greedy procedure; they are descriptive and not repaired or human-scored.

**generation:01** — prompt: `Say hello to a new neighbor.`

Response: ``

**generation:02** — prompt: `A small bird landed on the fence and`

Response: ` the bird found the birdd of the big big boy.`

**generation:03** — prompt: `Tell a short sentence about a turtle.`

Response: ``

**generation:04** — prompt: `User: My favorite color is green. What color do I like? Assistant:`

Response: ` "I's my, I's my, I's my, I's my, Lily.`

**generation:05** — prompt: `A boy gave his friend a paper star. His friend`

Response: `s and he was very happy.`

**generation:06** — prompt: `When the rain stopped, the children`

Response: `'t want to the box.`

**generation:07** — prompt: `Mara dropped her paintbrush, so`

Response: ` happy.`

**generation:08** — prompt: `The kitten climbed onto the chair and`

Response: ` them.`

**generation:09** — prompt: `The lamp went out during dinner. Everyone`

Response: ` was very happy.`

**generation:10** — prompt: `What should a child do after spilling water on the floor?`

Response: ``

**generation:11** — prompt: `What is one safe thing to do before crossing a street?`

Response: ``

**generation:12** — prompt: `Why might someone smile after receiving a letter?`

Response: ``

**generation:13** — prompt: `The child found a note under the pillow. The note said`

Response: `, "It's go, I we can be cark.`

**generation:14** — prompt: `After the cake was baked, the family`

Response: `.`

**generation:15** — prompt: `User: The bird is in the tree. Where is the bird? Assistant:`

Response: `.`

**generation:16** — prompt: `How can a person be kind to a new friend?`

Response: ``

**generation:17** — prompt: `A girl lost her mitten at the park. She`

Response: ` was her mommy and she wanted to her mommy.`

**generation:18** — prompt: `User: I feel tired today. Assistant:`

Response: ` "It's my, I we can have to the bie.`

**generation:19** — prompt: `User: I found a lost pencil. What should I do? Assistant:`

Response: ` "I's go, I we can have you.`

**generation:20** — prompt: `A dog heard a noise outside and`

Response: ` the big big boys.`

## Parameter drift from Pilot 1

- block0_2: 0.1090% relative L2 delta across 150 tensors.
- block3_7: 0.4141% relative L2 delta across 250 tensors.
- embeddings_positions: 0.2627% relative L2 delta across 2 tensors.
- final_norm: 0.3963% relative L2 delta across 2 tensors.
- language_head: 12.0085% relative L2 delta across 2 tensors.
- localizer_retrieval: 1.7524% relative L2 delta across 8 tensors.

The measured drift is a descriptive scope audit. It does not establish that block 3 caused or failed to cause any behavioral change.

## Interpretation

HR-3 lowered the correctly aligned TinyStories DEV loss relative to Pilot 1, but controlled contextual fact behavior did not improve: 12/24 items, 0/12 matched reversals, and 0 complete families at update 500. This does not support reliable use of short contextual facts, counterfactual selection, or HUMAN_TEST_READY readiness. Generation remained short and malformed, with 14/20 non-immediate-EOS responses and 13/20 automatically non-degenerate responses; several responses were fragments, punctuation-only, repetitive, or context-inappropriate. Instruction and continuity remained below their frozen gates.

The treatment preserves the established binding capability on both nonsacred pools. Failure is evidence about this bounded scope-only treatment and these diagnostics, not architectural impossibility or a general capacity ceiling. A future step would require a new prospective decision; this report makes no parent or curriculum selection.

## Artifact pointers

- Complete machine-readable comparison, including every controlled item, token-level candidate scores, margins, greedy IDs, raw generation text, and binding rows: `C:\DaveLM-CADAVER\human_readiness_hr3_causal_seed87006_execution_v7\NONSACRED_DEV_COMPARISON.json`
- Durable trainer metrics: `C:\DaveLM-CADAVER\human_readiness_hr3_causal_seed87006_execution_v7\UPDATE_METRICS.jsonl`
- Durable update trajectory: `C:\DaveLM-CADAVER\human_readiness_hr3_causal_seed87006_execution_v7\DEV_TRAJECTORY.json`
- Frozen v2 readiness battery: `C:\DaveLM-CADAVER\human_test_readiness_v2_seed87010` (not modified or reopened beyond its sealed manifest/receipt checks).
