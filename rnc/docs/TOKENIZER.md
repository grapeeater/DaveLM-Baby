# RNC tokenizer: `rnc_bpe_8k`

Byte-level BPE, 8,192 ids, trained 2026-10-16 on 400,000 training-split TinyStories documents.
Fingerprint (SHA-256 of `tokenizer/rnc_bpe_8k/tokenizer.json`): `d37360ce034918fcc2dc9be5fee17903244001eb307a1154ac662a5ee10ff6d3`.
The fingerprint is pinned in `configs/rnc1_tinystories.json`, written into `data/prepared/*/META.json` and into
every checkpoint; a mismatch anywhere stops the program.

## Design

| Aspect | Decision | Reason (lessons from JAWN-BPE / Baby-J and the S3 namespace defect) |
|---|---|---|
| Algorithm | byte-level BPE (HF `tokenizers` 0.23.2), 256-byte alphabet, no `unk`, no byte-fallback needed | Every byte string round-trips exactly; arbitrary Unicode is representable. |
| Normaliser | none | A normaliser (e.g. NFC) breaks byte-exact round trips. |
| Pre-tokeniser | Llama-3-style regex: contractions; letter runs with one optional leading non-letter; digits in groups of ≤ 3; punctuation runs with trailing newlines; newline runs; whitespace split so trailing spaces attach to the next word; then ByteLevel | Keeps words, punctuation, numbers and whitespace in predictable pieces; avoids giant whitespace/digit tokens. |
| Training input | `train_from_iterator` on whole documents | The file-based trainer reads line by line and can never learn newline-adjacent merges. |
| Training data | training split only (never validation documents) | No held-out text leaks into the vocabulary. |
| Control tokens | ids 0–7: `<\|pad\|>` `<\|bos\|>` `<\|eos\|>` `<\|sep\|>` and four reserved; bytes at ids 8–263; merges from 264 | A fixed namespace; ids fit in uint16. |
| Control-token isolation | `encode()` runs with `encode_special_tokens=True`, so the literal text `<\|eos\|>` becomes ordinary bytes; only `encode_document()` inserts `<\|bos\|>`/`<\|eos\|>` | Text can never inject a control id (the Baby-J "special-string injection" lesson). |
| Single codec | `rnc.tokenizer.RNCTokenizer` is the only path between text and ids | The S3 raw-id defect came from calling the library directly with the wrong offset. |
| Determinism | same documents → byte-identical `tokenizer.json`; the file is re-serialised canonically (sorted keys) | Verified by `tests/test_tokenizer.py::test_training_is_deterministic`. |
| Document format | `<\|bos\|> text <\|eos\|>` | `<\|bos\|>` lets sampling start a fresh story; `<\|eos\|>` ends it. |

## Vocabulary size: measured, not assumed

Three candidates were trained on the same 400k documents (18 s each) and measured on held-out documents
(`reports/phase3_tokenizer/`):

| Vocab | tokens / word (5k docs) | bytes / token | ids never seen (20k docs, 4.1M tokens) | ids seen < 10 times | compression vs 8k |
|---:|---:|---:|---:|---:|---:|
| 8,192 | 1.1993 | 4.236 | 615 (7.5 %) | 1,635 (20 %) | — |
| 16,384 | 1.1893 | 4.272 | 3,384 (21 %) | 10,111 (62 %) | +0.84 % |
| 24,576 | 1.1878 | 4.277 | 9,921 (40 %) | 18,516 (75 %) | +0.96 % |

TinyStories deliberately uses a small vocabulary, so merges beyond ~8k are mostly rare names and fragments
(`" Pippin"`, `" Albertina"`, `" hairstylist"`). Doubling the vocabulary buys under 1 % compression and leaves most
of the new embedding rows almost untrained. 8,192 is the corpus-justified size; it also saves 6.3M parameters.
The ~16K candidate suggested earlier was treated as a candidate, not a target, and the measurements decided.

The 16k and 24k candidates are kept under `tokenizer/candidates/` for reference.

## Tests

`tests/test_tokenizer.py` (tiny fixture tokenizer) and `tests/test_real_tokenizer.py` (the pinned 8k tokenizer):
exact round trips on 23 edge-case strings (empty, whitespace runs, CRLF/CR/LF, tabs, NUL, BOM, NFD accents, Hangul
jamo, ZWJ emoji, CJK, Greek, Cyrillic, numbers, code-like identifiers, control-token literals), 300 random Unicode
strings, all 256 Latin-1 characters, random byte sequences; control tokens never produced from text; document
structure; batch = single encoding; common words are single pieces; deterministic training; fingerprint pinning.
