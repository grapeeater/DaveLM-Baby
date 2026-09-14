# Tokenizer repair options

Evaluated before selecting an intervention. No option may open TEST or overwrite v0_7 / historical checkpoints.

## Option A — tokenizer-aware decoding / string scoring

**Keep v0_7 and existing T28/Phase1G weights.**

Subtypes:

- **A1 native free beam / delayed commitment** (inventory-free). Search the 1024-way distribution. Preserves the native-generation scientific question.
- **A2 constrained candidate scoring.** Mean logprob over answer strings. T28 native forced-choice is already this for the two in-row names (82–88/128). An 8-name DEV inventory rerank is stronger multiple-choice. **It is not native generation** and cannot declare Phase 2A solved.

Likelihood of attacking 4.7% vs 53.1%: A1 tests whether greedy local commits are the bottleneck; A2 will look better than greedy almost by construction if representation is intact.

Preservation: maximum. Compatibility: full. Compute: low. Contamination: none if A1 stays inventory-free.

## Option B — tokenizer vNext with better identity boundaries

Train a new BPE from **Phase1G/language-authorized TRAIN only**. DEV may evaluate a frozen design, not seed the vocab. Do not hand-add DEV answers.

A larger vocab or different merge set might split `Sal`/`Salt` and `Sky`/`Skye`, but Wes→Walt and Omar→York can remain if those strings are still cheaper completions. Existing embeddings/head are **not** transferable without a mapping. This is a new lineage. Full language retrain is an owner-scale decision.

## Option C — append-only vocabulary extension

Baby vNext uses **untied** `token_embedding` `[1024, 640]` and `language_head` `[640, 1024]` plus bias. Architecturally, old IDs can be preserved and new rows appended if `vocab_size` is raised and old rows are copied bit-identically.

Scientifically cleaner than silent retokenization **only if** new pieces are learned from authorized TRAIN/language data, not DEV identities. Adding `Sal`/`Skye` as special tokens from the DEV panel would contaminate the experiment.

Requires a new config/checkpoint lineage even if old rows are copied. U0 old-token language behavior must be verified before any training. Not implemented in this study.

## Option D — full retokenization / retrain

Destroys compatibility with the 61.5M brain. Preserve v0_7 + Phase1G + all Phase 2A artifacts; start a new provenance identity. Owner review required before launch. Not selected.

## Ranking (this study)

| Rank | Option | Why |
|---|---|---|
| 1 | A1 native free beam | Least destructive test of the demonstrated greedy prefix-collapse |
| 2 | A2 labeled constrained rerank | Diagnostic only; already partly measured by native FC |
| 3 | C append-only from TRAIN | Feasible, not yet justified; contamination risk if DEV-driven |
| 4 | B new tokenizer | Possible later; no full retrain without owner review |
| 5 | D full replace | Last resort |

**Selected and executed:** A1 (failed to close shared-prefix). A2 reported as non-native (helps Sal/Skye, not Wes). Tokenizer unchanged.
