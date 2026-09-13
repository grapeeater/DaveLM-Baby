# Sentence-boundary treatment

Parent: `language_unlikelihood_p4/latest.pt`; settings and hashes are in `RESULTS.json`.

This 500-update language-only treatment trained on 147,521 naturally occurring TinyStories sentence segments with explicit BOS/EOS. A fixed five-prompt panel produced fluent short completions, but every completion was an exact substring of TinyStories and one was degenerate (`The dog found a red ball and a ball.`).

Conclusion: sentence-boundary training improved surface completion but did not demonstrate non-memorized contextual language. The objective remains unmet.
