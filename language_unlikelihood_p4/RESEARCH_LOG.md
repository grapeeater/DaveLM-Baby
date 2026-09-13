# Repetition-targeted treatment

Parent: `language_storybound_p3/latest.pt`; settings and hashes are in `RESULTS.json`.

A 500-update language-only treatment added a recent-token unlikelihood penalty (window 8, weight 0.1) to complete TinyStories starts. The fixed eight-prompt panel showed less direct repetition in some outputs, but semantic failures remained: “It had a ball that it had a ball,” “a big box with a big box,” and “It was a big, bird.” Two apparent successes were exact TinyStories phrases.

Conclusion: repetition is only part of the problem. Baby still lacks reliable context-sensitive semantic continuation. The target was not reached.
