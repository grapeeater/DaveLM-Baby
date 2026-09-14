# Tokenizer append-only extension v1 (Option C)

Frozen before selection, initialization, adaptation, or DEV scoring.

This study tests whether Baby can gain native articulation on shared-prefix
collisions by **appending** TRAIN-derived tokens while preserving v0_7 IDs
0–1023 bit-identically. It is **not** T29, **not** a full tokenizer
replacement, and **not** a from-scratch retrain.

## Integrity

- Workspace: `C:\DaveLM-CADAVER`
- v0_7 path and SHA-256 unchanged; this study never writes that file
- Phase1G parent and T28 `rolling_restart.pt` files are read-only
- Historical `LANGUAGE_*_STREAM.u16` files are not overwritten
- `qa_dev.jsonl` is loaded only after the tokenizer candidate is frozen
- `qa_test.jsonl`, T3 TEST, T2-EVAL-TEST, FINAL, and sacred are not loaded
- T24 is not resumed; T29 is not launched; gates are not lowered

## Phase A

Inspect tokenizer, embeddings, untied `language_head`, checkpoint load,
and the Phase1G/T28 pipelines. Append-only is valid only if old rows can
be copied bit-identically and `strict` load of an expanded matrix is
replaced by an explicit copy of rows 0–1023.

## Phase B (TRAIN only)

Corpus: Phase1G `LANGUAGE_TRAIN_SOURCE.jsonl`.
Rule: title-case word surface forms (`[A-Z][a-z]{2,11}`), including a
leading space when present in the source, that v0_7 currently encodes as
two or more tokens. Rank by frequency, lexicographic tie-break, minimum
frequency 30, maximum 64 added tokens. No DEV names, no hand-adds.

## Phase C

New tokenizer identity `v0_8_append64` stored only under this study
directory. Expand embedding and head; copy old rows exactly. Initialize
new embedding/head-weight rows as the mean of the candidate's v0_7
pieces; new bias = mean piece bias minus 8.0 so U0 old-token argmax and
sliced 1024-way CE are preserved until adaptation.

U0 hard stop: old-token logits, v0_7 pointer, and sliced language CE must
match the parent checkpoint. Unexplained regression stops the study.

## Phase D

Train **only** new embedding/output rows for 100 AdamW updates at
3.75e-5 on a **new** retokenized Phase1G TRAIN stream. Do not escalate
scope if this fails.

## Phase E

DEV-only. Report shared-prefix vs unique-prefix exact+EOS (original four
shared names are descriptive), first-token-correct-then-diverge,
language, pointer, Wes family, and native free greedy (not constrained
rerank). Replicate on T28 seeds 850001–850003.

Success requires material shared-prefix exact improvement from 4.7%,
material drop in first-token-correct-then-diverge, no material unique-prefix
regression from 53.1%, language/representation retention, native free
generation improvement, replication, and sealed TEST.
