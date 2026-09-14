# Tokenizer candidate comparison

Option C remains `APPEND_ONLY_EXTENSION_VALID_COMPLETION / NO_EFFECT`. This Option B study is tokenizer geometry only. Baby was not retrained. TEST was not loaded. v0_7 was not overwritten.

Candidates were frozen in `CANDIDATE_FREEZE.json` before `qa_dev.jsonl` was opened. Construction used Phase1G `LANGUAGE_TRAIN_SOURCE.jsonl` plus T3 `qa_train.jsonl` prompts only.

## Ranking (frozen score, DEV)

| Rank | Candidate | Vocab | SHA-256 | DEV shared-first | TRAIN shared-first | DEV single-token | DEV tokens/name | Phase1G bytes/token | score_dev |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `v0_7_baseline` | 1024 | `e1c18bae…343b` | **0.500** | 0.750 | 0 | 2.75 | **2.201** | **2.864** |
| 2 | `bpe_bl_1536` | 1536 | `9c565458…7504` | 0.500 | 0.500 | 0 | 3.00 | 3.446 | 2.551 |
| 3 | `bpe_ws_2048` | 2048 | `3e551077…3c6a` | 0.500 | **0.375** | 0 | 2.62 | 3.835 | 2.510 |
| 4 | `bpe_bl_2048` | 2048 | `151a57b1…c2b1` | 0.500 | 0.500 | 0 | 3.00 | 3.652 | 2.504 |
| 5 | `bpe_bl_2048_prefix_space` | 2048 | `77d04805…4a82` | 0.500 | 0.500 | 0 | 3.00 | 3.653 | 2.504 |
| 6 | `bpe_bl_4096` | 4096 | `eba2b803…3cce` | 0.500 | 0.375 | 0 | 2.62 | 4.000 | 2.472 |
| 7 | `bpe_bl_2048_titlecase_x4` | 2048 | `28683e61…cf82` | **0.625** | 0.688 | 0 | 3.12 | 3.603 | 2.126 |

Best *new* candidate by frozen score: `bpe_bl_1536`. It does **not** beat v0_7.

## Material rule (frozen)

Need DEV shared-first rate ≤ half of v0_7 (≤ 0.250), disambiguation not worse, TRAIN shared rate also down, language bytes/token within +25% of 2.201 (≤ 2.751).

**No candidate meets it.** Every new tokenizer keeps DEV shared-first at 0.500 (Omar/Opal and Sal/Skye) except title-case upsample, which **worsens** it to 0.625 by pulling Wes into the `O`/`W` space-split group. All new tokenizers miss the language-compression bound (3.45–4.00 vs 2.20).

TRAIN-side collision *does* fall for some candidates (0.750 → 0.375) because TRAIN identity names occur in `qa_train.jsonl` prompts. That improvement does **not** appear on held-out DEV names.

## Classification

`TOKENIZER_REPLACEMENT_NOT_JUSTIFIED`

Full hashes are in `CANDIDATE_FREEZE.json` and `SHA256SUMS.txt`.
