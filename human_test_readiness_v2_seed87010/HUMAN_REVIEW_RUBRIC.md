# Frozen generation review rubric

For each raw response, two reviewers work independently without seeing checkpoint identity.

- `complete`: 1 only if the response contains a grammatical, completed English sentence ending in `.`, `?`, or `!`, without truncation or an unfinished trailing fragment.
- `relevant`: 1 only if the response answers the prompt or continues its stated event while preserving an appropriate prompted entity, relation, question answer, or social act. A response dominated by an unrelated newly introduced actor/event, contradiction, prompt copying, or generic template receives 0.
- `automatic_non_degenerate`: computed, not judged. It is 1 only if the response is non-EOS, has no three-identical-token run, no repeated decoded trigram, and is not a fourth-or-later duplicate normalized non-EOS response.

A reviewer disagreement is scored 0 for the corresponding advancement field. Raw text is never repaired before review.
