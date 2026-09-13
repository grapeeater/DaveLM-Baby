# Mixed naturalistic/consistency treatment

Parent: `language_consistency_p1/latest.pt` (hash in `RESULTS.json`). The prior narrow consistency treatment produced grammatical but visibly templated outputs, while TinyStories continuation produced semantic drift. This run alternated nine TinyStories batches with one consistency batch for 500 updates at learning rate 1e-4. The eight held-out-style prompts were fixed before generation.

Result: local sentence form improved, but contextual reliability remained inadequate. Examples include `Alex found a green book. Nora played with the green book` (subject substitution) and repeated prompt restarts on most prompts. The treatment therefore failed the genuine-coherence criterion. It supports local pattern learning and does not support robust context-sensitive natural-language behavior.

Next step: do not continue adding narrow templates. A useful next experiment must use held-out multi-sentence natural stories or a carefully controlled anti-repetition/subject-consistency objective with independent evaluation; this checkpoint is not automatically selected as a future parent.
