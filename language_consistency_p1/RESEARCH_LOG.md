# Semantic-consistency treatment log

Parent: `language_continuation_p0_v1/latest.pt` (SHA-256 recorded in `RESULTS.json`).

The frozen English characterization and the continuation panel showed that lower perplexity did not prevent subject drift and repetition. This treatment trained only the ordinary base model for 500 updates on 819 short two-sentence examples generated from held-out combinations of names, colors, and objects. 205 combinations were held out; the 12-panel prompts were fixed before generation. T13 modules were frozen and no sealed artifact was touched.

All 12 held-out prompts produced a grammatical, subject-consistent second sentence, but the outputs immediately repeated the same template or restarted the prompt (for example, `Lily found a red ball. Lily looked at the red ball.Lily found a red`). Because the success is confined to a narrow synthetic template and visibly memorized continuation pattern, it does not meet the genuine natural-language criterion. It is evidence that the backbone can acquire a local context-to-sentence mapping, not evidence of broad coherence.

Next decision: require a mixed naturalistic/controlled evaluation with held-out frames and a repetition stop rule before treating any future improvement as genuine. Do not select this checkpoint as a general language parent based on its narrow template performance.
