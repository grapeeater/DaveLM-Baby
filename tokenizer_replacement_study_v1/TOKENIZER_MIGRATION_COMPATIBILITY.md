# Tokenizer migration compatibility

This documents what replacing v0_7 would imply. **No weights were migrated. No model was retrained.**

## Vocabulary and IDs

| Item | v0_7 | Any new candidate here |
|---|---|---|
| Vocab size | 1024 | 1536 / 2048 / 4096 |
| Token ID space | 0–1023 with frozen specials 0–4 | Same specials, **different** merge IDs |
| Historical streams | Phase1G/T3 `u16` IDs | Invalid; must retokenize from text into a **new** stream |
| v0_7 file | SHA-256 `e1c18bae…343b` | Must remain on disk untouched |

IDs are **not** compatible. Loading a T28 checkpoint into a 1536-way embedding would be silent corruption even if one padded the matrix.

## Weights that cannot transfer

Baby vNext `BabyVNextWithBinding`:

- `base_model.token_embedding.weight` `[1024, 640]` — ID-semantic; **no safe row copy**
- `base_model.language_head.weight` `[1024, 640]` and bias `[1024]` — untied but still ID-semantic; **no safe row copy**
- Transformer blocks, positions, final norm, localizer, `wq/wk/wv/wo` — not vocab-indexed, but they were trained in v0_7 token space. Hidden states for the same *string* would differ because the token sequence differs. Transferring the 61.5M body onto a new tokenizer is **not scientifically defensible** as “the same model.”

Option C showed that even **preserving** old IDs and copying old rows bit-identically did not change native generation. Replacing IDs is strictly more incompatible.

## Weights that could theoretically transfer

None of the vocab-tied matrices. Non-vocab tensors could be used only as an initialization *experiment* for a **new lineage**, not as a continuation of Phase1G/T28 identity. That experiment is **not authorized** here.

Embedding composition from v0_7 pieces (Option C’s init) is the one defensible composition method, and it already failed to change greedy. It does not apply to a shuffled BPE.

## Corpus retokenization

A new lineage would need **new** files, never in-place overwrite:

- Phase1G language TRAIN/DEV streams
- T3 `prompt_token_ids`, mention spans, decision positions, candidate IDs
- All Phase 2A schedules that index token positions

Historical artifacts stay v0_7 so T18–T28 remain reproducible.

## Fresh model lineage vs transplant

A clean **from-scratch** language parent on the new tokenizer is the only honest path if a tokenizer were ever justified. That is a new provenance identity, new parent hash, new Phase-1-class compute, then a new Phase 2A stack.

This study does **not** justify that path.

## Checkpoint compatibility

**None.** T28 `rolling_restart.pt` hashes (`d6f5be58…`, `f01066dd…`, `95e34fd5…`) and Phase1G parent `c5406f80…` remain v0_7-only.
