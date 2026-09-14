# Tokenizer replacement candidate protocol (Option B)

Frozen **before** candidate training and **before** DEV scoring.

Study: `tokenizer_replacement_study_v1`. This is a **tokenizer-design** study.
It is **not** T29, **not** a 61.5M retrain, and **not** a license to mutate v0_7.

Option C remains terminal evidence: `APPEND_ONLY_EXTENSION_VALID_COMPLETION / NO_EFFECT`.

## Integrity

- Workspace `C:\DaveLM-CADAVER`
- v0_7 path `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json`
- v0_7 SHA-256 `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`
- Tokenizer construction corpus (TRAIN only):
  - Phase1G `LANGUAGE_TRAIN_SOURCE.jsonl` SHA-256 `807a89417924a9b1cfa1e0a2a952d97400684db32a138fe11943d93e130312c8`
  - Phase2A `phase2a_t3_rebuilt_study_v1/data/qa_train.jsonl` prompt strings only
- DEV (`qa_dev.jsonl`) is loaded **only after** every candidate file is written and hashed
- Never loaded: `qa_test.jsonl`, T2-EVAL-TEST, FINAL, sacred
- No hand-added DEV identities. No post-hoc retuning after DEV scores.

## Frozen candidate set

All new candidates use special tokens `<pad> <unk> <bos> <eos> <doc>` in that ID order, matching v0_7’s specials. Seed for any trainer shuffle: **none** (deterministic iterator order).

| ID | Family | Vocab | Pretokenizer | Other |
|---|---|---:|---|---|
| `v0_7_baseline` | existing byte-level BPE | 1024 | ByteLevel `add_prefix_space=false` | **not rebuilt**; v0_7 file read-only |
| `bpe_bl_1536` | same family, larger vocab | 1536 | ByteLevel `add_prefix_space=false` | `min_frequency=2` |
| `bpe_bl_2048` | same family, larger vocab | 2048 | ByteLevel `add_prefix_space=false` | `min_frequency=2` |
| `bpe_bl_4096` | same family, larger vocab | 4096 | ByteLevel `add_prefix_space=false` | `min_frequency=2` |
| `bpe_bl_2048_prefix_space` | byte-level BPE | 2048 | ByteLevel `add_prefix_space=true` | leading-space identity of words |
| `bpe_bl_2048_titlecase_x4` | byte-level BPE | 2048 | ByteLevel `add_prefix_space=false` | TRAIN title-case words `[A-Z][a-z]+` appended 4× as extra documents (general name-span upsampling; not DEV names) |
| `bpe_ws_2048` | whitespace-then-byte BPE | 2048 | `WhitespaceSplit` then ByteLevel `add_prefix_space=false` | word-first merges |

## Evaluation surfaces (frozen)

Identity inventory encoding uses the T3 answer convention: leading space, optional period.

- name span: `" " + name`
- answer span: `" " + name + "."`

DEV inventory = unique `correct_name` values in `qa_dev.jsonl` **after freeze**.
TRAIN inventory = unique `correct_name` values in `qa_train.jsonl` (construction-time, not a DEV leak).

Descriptive-only strings after freeze: Sal, Salt, Skye, Sky, Wes, Walt, Omar, Opal.
These do **not** enter the ranking formula.

## Frozen tokenizer-quality score

Higher is better. Computed independently on DEV and TRAIN inventories; the ranking key is `score_dev` with `score_train` reported for generalization.

```
score = 3.0 * (1 - shared_first_token_rate)
      + 1.5 * single_token_name_rate
      + 1.0 / mean_tokens_per_name
      + 1.0 / mean_first_disambiguating_position
      - 0.5 * max(0, (lang_bytes_per_token / v0_7_lang_bytes_per_token) - 1)
```

Language compression is Phase1G TRAIN source UTF-8 bytes / token count (same documents for every candidate).

A candidate is **materially better** than v0_7 only if **all** hold:

1. DEV shared-first-token rate falls by at least **half** relative to v0_7 (or reaches 0)
2. DEV mean first-disambiguating position does not increase
3. TRAIN shared-first-token rate also falls (generality)
4. Phase1G bytes/token is not worse than v0_7 by more than **25%**
5. No protected data used

## Decision labels (frozen)

- `TOKENIZER_REPLACEMENT_STRONGLY_JUSTIFIED`: material by the rule above, and DEV single-token rate ≥ 50%
- `TOKENIZER_REPLACEMENT_PROMISING`: material by the rule, single-token rate < 50%
- `TOKENIZER_REPLACEMENT_WEAK`: some collision improvement but fails the material rule
- `TOKENIZER_REPLACEMENT_NOT_JUSTIFIED`: no material shared-prefix collision reduction

If none materially reduce shared-prefix geometry: **STOP**. Do not recommend full retrain.

If one does: write a migration plan and **do not execute it**.

## Forbidden

T29, T24 resume, overwriting v0_7, loading TEST/FINAL/sacred, training Baby, hand-adding DEV names, lowering Phase 2A gates.
