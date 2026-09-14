# Current state

This file is a short pointer. The authoritative current state is
[`CANONICAL_STATE.md`](CANONICAL_STATE.md), updated 2026-09-14.

Phase1G U6000 is the preserved Phase 2A parent. T28 FAST V2 is terminal:
`T28_SUFFIX_MARGIN_VALID_COMPLETION / OUTPUT_FAIL`. Post-T28 tokenizer and
suffix-state diagnostics are complete. Decode-only mouth repair
(`tokenizer_repair_v1`) found that free beam does not close the shared-prefix
exact gap; constrained 8-name rerank is not native generation.

Post-T28 prefix rescue is complete:
`EARLY_TOKEN_ERROR_CAUSALLY_DOMINATES` (gold single-token oracle then free greedy
recovered 108/124 exact+EOS on the diverge population). Tokenizer lineage remains
paused. Activation patching was not run.

T3 TEST remains `SEALED_UNOPENED`. T29 is terminal:
`T29_REPRESENTATION_SUCCESS_OUTPUT_FAIL` (exact 35/35/35). The fork-objective
hypothesis is closed. Do not overwrite v0_7. Do not resume T24. Do not retune
T29 lambda/margin. T30 and T31 are terminal. T31 is T31_EXPLICIT_SOURCE_TO_TOKEN_BRIDGE_INSUFFICIENT; do not launch T32.


T31 terminal: explicit source-to-token copy bridge insufficient; exact 35/128 in both required seeds, seed 3 not required, T32 not authorized. TEST/FINAL/sacred locked.
