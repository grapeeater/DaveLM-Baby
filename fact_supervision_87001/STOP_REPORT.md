# Prospective construction stopped before freeze

The implementation has reached a scientific design conflict. It is NOT a frozen
training/evaluation package and must not be used for training or model selection.

The approved first-eight-object-families rule yields 16 naturalistic entries per
holdout but only 13 unique primary and
16 unique confirmation prompts. Across both holdouts,
32 entries have only 26 unique prompts; 3 prompts occur in both holdouts.
Different two-fact families can share one fact. Reducing each fact to a single-sentence
naturalistic prompt therefore loses the family distinction. This is a design conflict,
not an answer-key implementation error.

No entries were removed, replaced, resampled, or rewritten to pass. Review is needed
before changing the naturalistic selection rule, counts, or uniqueness requirement.

## Completed construction checks

- 108 semantic families: train 48, DEV 12, primary 24, confirmation 24; each half object/predicate.
- 1,728 base rows across factual/control arms: 768 train, 192 DEV, 384 primary, 384 confirmation.
- Independent validation passed all base answer keys, 8-member structures, assignment/query/order transformations, matched targets/cues, and exact balanced control target twins.
- 960 additional factual transfer items were rendered and independently parsed (QA, passive wording, distractor placement).
- All 1,824 factual prompts across base and transfer sections are unique.
- All candidate completions are four tokens. UTF-8 text round trips, prompt/candidate boundary concatenation, and completed context limits passed with the required tokenizer hash.
- TRAIN factual/control prompt token lengths match exactly; maximum absolute mismatch elsewhere is one token.
- No exact normalized prompt/completion hits in the 66 inventoried historical sources. This does not establish semantic or near-duplicate non-contamination.

Construction candidates, identifiers, final text, answers, token IDs, and full validation
details are preserved in construction_candidates.json and construction_check.json.
The independent validator imports no builder code. Control twins are intentionally
duplicated inputs with opposite balanced practice targets, not factual answer labels.

## Work not completed after the stop

The final schedules, mock test execution, complete runtime/training harness freeze,
and pretraining freeze receipt remain incomplete. The helper/test source is provisional,
not approved for execution against Baby. The accompanying erratum is additive only.

No checkpoint was loaded. No inference, training, backward pass, optimizer creation,
or sacred access occurred. Historical checkpoints, datasets, and results remain unchanged.

Mechanical startup corrections: explicit UTF-8 tokenizer JSON decoding; explicit
allowlist for the four nonsacred historical binding corpus files already listed in the
old audit. No scientific construction rule changed.
