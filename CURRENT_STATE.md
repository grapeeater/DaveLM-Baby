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

## T32 terminal closure (2026-09-14)
T32 Causal Ordered Source-Memory Prefix is terminal. It used ordered source-token K/V memory in the final Transformer block with a zero-initialized 640-channel gate, parent-neutral at U0. Seeds 890001 and 890002 completed the frozen 1000-update schedule; seed 890003 was not required by the sealed 2-of-3 rule after two scientific failures. Terminal exact+EOS was 35/128 for both; shared-prefix exact was 0/64 and 2/64; pointer was 112/128 and 110/128; language and binding remained intact. Memory ON/OFF terminal ablation was effectively unchanged. Classification: T32_EXPLICIT_ORDERED_SOURCE_MEMORY_INSUFFICIENT. The Phase2A surgical treatment line is exhausted. Per the frozen failure branch, no T32b/T33 is authorized; the next owner-planned action is TARGETED_EXTERNAL_MODEL_ARCHAEOLOGY_EXACTLY_3_MODELS. T3 TEST, FINAL, and sacred remain sealed/unopened.
