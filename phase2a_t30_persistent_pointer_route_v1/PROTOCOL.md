# T30 persistent pointer-route treatment

## Scientific question
Can Baby's existing query-to-fact-clause pointer state remain available across the answer span and improve native greedy generation?

## Selected intervention
The inherited T24/T28/T29 teacher-forced recipe is preserved. `T30_ROUTE.py` adds one new trainable vector `route_gate` of length 640, initialized to exactly zero. For each factual row, the existing cosine pointer scores the existing fact-clause spans from the final query state. Their softmax-weighted mean is added elementwise, through `route_gate`, to every hidden state that predicts the answer (`prompt_len-1` onward). The frozen existing language head then produces logits. No correct-answer label is used to construct the route, no rollout is added, and binding/localizer parameters remain frozen.

The neutral route state is exact zero. U0 preflight reproduced the parent DEV CE `1.2040123894810677` and binding integrity.

## Frozen execution

- Parent: Phase1G U6000 `c5406f...eefb1`, tokenizer v0_7 `e1c18b...b343b1`.
- Seeds: 870001, 870002, 870003; one sequential run per seed.
- Inherited 1000-update, 900 QA / 100 language, phase-A/phase-B schedule; no T3 TEST, T2 EVAL TEST, FINAL, or sacred access.
- Existing pointer, full answer CE, candidate margin, inventory unlikelihood, language rehearsal, optimizer, LR, scopes, and gates are unchanged. T29 fork and T28 suffix-margin losses are off.
- Frozen gates are the inherited representation/native/language/binding/TRAIN16 gates in `PROTOCOL.json`.

## Prospective interpretation
Native exact generation is the primary outcome. A replicated rise in exact output and fall in first-token-correct-then-diverge while representation, language, and binding remain green is supportive. Route-gate activation alone is not success; a representation-only result is output failure. This study ends after the three seed adjudications.
