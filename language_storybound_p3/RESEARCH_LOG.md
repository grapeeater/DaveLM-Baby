# Story-boundary treatment log

Parent: `language_continuation_p0_v1/latest.pt`; all hashes and settings are in `RESULTS.json`.

This 500-update, 1e-4 language-only treatment trained complete TinyStories starts with explicit BOS/EOS and masked padding. On the fixed eight-prompt panel it produced several grammatical, context-linked first sentences, but four were exact corpus substrings, one was malformed, and longer continuations still repeated or drifted. A follow-up lexical-transfer probe with novel names showed the same memorized-style phrasing and rabbit/ball semantic drift.

Conclusion: story-boundary training improved local sentence form but did not demonstrate non-memorized, context-appropriate natural language. The goal remains unmet. Further work should target broader held-out semantic composition and repetition control rather than more narrow template exposure.
