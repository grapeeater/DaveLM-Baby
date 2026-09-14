# Current state

This file is a short pointer. The authoritative current state is
[`CANONICAL_STATE.md`](CANONICAL_STATE.md), updated 2026-09-13.

Phase1G U6000 is the preserved Phase 2A parent. T28 FAST V2 is terminal:
`T28_SUFFIX_MARGIN_VALID_COMPLETION / OUTPUT_FAIL`. Post-T28 tokenizer and
suffix-state diagnostics are complete. Decode-only mouth repair
(`tokenizer_repair_v1`) found that free beam does not close the shared-prefix
exact gap; constrained 8-name rerank is not native generation.

T3 TEST remains `SEALED_UNOPENED`. T29 is not authorized. Do not overwrite v0_7.
Do not resume T24. Do not launch a from-scratch tokenizer-replacement retrain
without owner review.
