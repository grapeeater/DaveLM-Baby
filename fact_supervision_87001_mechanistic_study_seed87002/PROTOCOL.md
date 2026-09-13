# Mechanistic localization study - pre-execution freeze

Scientific question: when a competing mapping is added (ONE->TWO), is usable query-conditioned correct-actor information absent from the answer-position hidden state, or present but overridden by competition/position?

Stage 1 (always): read-only representational census on the frozen L2 ONE/TWO items using the answer-position final_norm hidden state, a per-family actor direction built from matched ONE pairs (full preregistered set, no outcome selection), correct-side projection rates, matched ONE/TWO cosine similarity, and cosine alignment with the frozen token embeddings of the two names.

Stage 2 (conditional, gated BEFORE results): patch the TWO answer-position hidden with the matched ONE answer hidden and recompute the answer-position logits through the frozen language_head; report first-token correct-actor selection under the frozen first-token LL. Controls: sham (own hidden), wrong-actor (other query's ONE hidden), and both row orders. Stage 2 runs for a checkpoint only if TWO correct-side rate >= 0.90 AND frozen TWO behavioral accuracy <= 0.60; otherwise it is recorded as not triggered.

No training, no weight modification, no outcome-selected subsets, no new behavioral items, no trained probes, no thresholds beyond the preregistered gate.
