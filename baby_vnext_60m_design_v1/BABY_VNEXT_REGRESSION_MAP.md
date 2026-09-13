# Baby vNext regression laboratory map

The tokenizer, vocabulary, and context length remain unchanged. Frozen item identities and token sequences therefore remain meaningful. Loader and model-call adapters are still required because vNext has a new checkpoint schema and an explicit binding layout.

| Capability / evaluator | Classification | Migration requirement |
|---|---|---|
| T13 answer exact | REQUIRES SHAPE-ONLY MIGRATION | Load vNext schema; call explicit binding layout through legacy offset-four adapter. |
| T13 BOTH_DISTINCT / collapse | REQUIRES SHAPE-ONLY MIGRATION | Preserve two-slot definitions and source-row accounting. |
| T13 reversals / complete quartets | REQUIRES SHAPE-ONLY MIGRATION | Dataset and scoring remain valid; update loader/interface only. |
| Pilot0 nonsacred binding DEV | REQUIRES SHAPE-ONLY MIGRATION | Evaluate separately; do not average with Pilot1 pool. |
| Pilot1 nonsacred binding DEV | REQUIRES SHAPE-ONLY MIGRATION | Evaluate separately; retain existing gates. |
| Ordinary-English aligned CE/PPL | REQUIRES SHAPE-ONLY MIGRATION | Preserve token windows and causal alignment; replace checkpoint loader. |
| Ordinary-English greedy diagnostics | REQUIRES SHAPE-ONLY MIGRATION | Preserve BOS/EOS/max-token rules; replace loader. |
| TRAIN16 forced correctness | REQUIRES SHAPE-ONLY MIGRATION | Candidate scoring is unchanged because tokenizer is unchanged. |
| TRAIN16 exact / reversals / families | REQUIRES SHAPE-ONLY MIGRATION | Preserve exact candidate+EOS and grouping rules. |
| DEV_SURFACE and subgroups | REQUIRES SHAPE-ONLY MIGRATION | Same battery and gates after prospective authorization. |
| DEV_ORDER fact/copy/order cells | REQUIRES SHAPE-ONLY MIGRATION | Same battery and gates after prospective authorization. |
| D3 four-name mass | REQUIRES SHAPE-ONLY MIGRATION | Same 256 positions and four token IDs; retain its narrow interpretation. |
| Historical SF1 transfer/copy/competing panels | HISTORICAL-ONLY / LOCKED | No access unless separately authorized; never training data. |
| HUMAN_TEST_READY DEV | HISTORICAL-ONLY until a future readiness phase | Its contents/gates were built for descendants of Research Baby; prospective applicability must be declared before scoring. |
| FINAL / sacred material | HISTORICAL-ONLY / LOCKED | No vNext access under this design task. |

No gate is weakened by this map. From-scratch vNext must first acquire the prerequisites; a larger untrained model is not expected to pass descendant-retention gates at update zero.

