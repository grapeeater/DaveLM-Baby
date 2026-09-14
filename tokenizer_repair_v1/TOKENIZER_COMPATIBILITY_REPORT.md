# Tokenizer / checkpoint compatibility

## v0_7 preserved

DaveLM tokenizer v0_7 remains byte-identical. Path `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json` (also tracked under `DaveLM-v0.9/tokenizer/v0_7/`). SHA-256 `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`. Vocab **1024**. Byte-level BPE. This study does not write that file.

## Why weights cannot be swapped onto a new vocab

Baby vNext `BabyVNextLM`:

- `token_embedding`: `nn.Embedding(vocab_size=1024, d_model=640)` → 655,360 parameters
- `language_head`: `nn.Linear(640, 1024, bias=True)` → 655,360 + 1,024 parameters
- Untied: embedding and head do not share storage (`validate_design.py` checks distinct data pointers)

Token IDs in Phase1G language streams and all Phase 2A QA encodings are v0_7 IDs. A different BPE changes ID semantics even at the same width. Loading old rows onto a reshuffled 1024-piece vocab would be silent corruption.

## Append-only extension (not implemented)

Technically possible:

1. Copy old embedding rows 0–1023 and old `language_head` rows/bias 0–1023 bit-identically.
2. Initialize only new rows with a frozen seed.
3. Raise `BabyVNextConfig.vocab_size`.
4. Emit a **new** checkpoint identity; do not overwrite Phase1G or T28 files.
5. At U0, verify old-token language CE / generation prompts match the parent within numerical noise.

Not cleaner than A1 until TRAIN-only pieces are specified. DEV names as added special tokens are forbidden.

## Full replacement

Requires a new tokenizer version directory (for example `v0_8`, only if that slot is unused), a new language parent, and owner review. Historical v0_7 results stay reproducible only if v0_7 and old streams remain untouched.

## This study's actual compatibility

Decode-only on T28 FAST V2 checkpoints. Tokenizer changed: **no**. Checkpoint compatibility: **full**. Full retrain required: **no** for A1/A2; **yes** for B/D; **maybe later** for C.
