# T19X GENERATION FAIL-MODE — FINAL

Status: **COMPLETE, read-only.** TEST not loaded.

T19 leftover modes are the same family as T18X: `first_token_correct_then_diverge` 43/49/42 (T18X 40/41/44). First-token BOS 82/84/76. Samples still `Skye.`→`Sky.`, `Wes.`→`Walt.`, `Ava.`→`York.`. Suffix-weight 3.0 did not change the fail-mode split.

Next: T20 readout-only answer CE (stop-grad through base). Do not raise suffix weight. Do not retokenize.
