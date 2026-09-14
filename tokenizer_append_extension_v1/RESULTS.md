# Tokenizer append-only extension v1 — results

Classification: **`APPEND_ONLY_COMPATIBLE_SCIENTIFIC_EFFICACY_FAIL`**.

Option C was executed. T29 was not launched. v0_7 was not overwritten. TEST / FINAL / sacred were not loaded. Gates were not lowered.

## Compatibility (Phase A)

Append-only extension is **architecturally valid**.

- Untied `token_embedding` `[1024, 640]` and `language_head` `[1024, 640]` + bias
- In-memory `add_tokens` assigned new IDs at 1024+ and preserved unrelated encodings
- Old embedding/head rows copied **bit-identically** (`torch.equal`, max abs 0)
- U0 old-token logits matched (max abs 0)
- After 100 new-row updates, old rows were still bit-identical (AdamW decay on the full tensor was undone by restoring rows 0–1023 after each step)

Historical `BABY_VNEXT_CONFIG.json`, v0_7, Phase1G streams, and T28 checkpoints were not overwritten.

## Selection (Phase B) — TRAIN only

Corpus: Phase1G `LANGUAGE_TRAIN_SOURCE.jsonl` SHA-256 `807a89417924a9b1cfa1e0a2a952d97400684db32a138fe11943d93e130312c8`.

Rule (frozen before DEV): title-case word surface forms `[A-Z][a-z]{2,11}` (leading space when present) that v0_7 encodes as ≥2 pieces; frequency ≥ 30; top 64; lexicographic ties.

- Qualifying candidates: 470
- Added: **64**
- DEV used for token selection: **NO**
- None of Ava, Mel, Omar, Opal, Ross, Sal, Skye, Wes appear in the added 64

Added surfaces are TinyStories-frequent forms (` She`, ` Lily`, `Once`, ` Timmy`, ` Ben`, ` Tom`, …). That is what the frozen TRAIN-frequency rule selected.

New tokenizer `v0_8_append64` SHA-256 `2b6ff009a11f26cbd12041c19cab46fb86fdb1d362e3b107f2ce187b1b5107d9` lives only under this study directory. v0_7 remains `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`.

## Initialization and adaptation

New embedding/head-weight rows = mean of v0_7 piece rows. New bias = mean piece bias − 8.0. Trainable scope: new rows only, 100 AdamW updates at 3.75e-5 on a **new** retokenized Phase1G TRAIN stream (8,001,794 tokens; 282,399 new-token occurrences). Scope was not escalated.

## Native greedy (DEV, three T28 seeds)

T28 greedy baseline: overall **111/384 (28.9%)**, shared **9/192 (4.7%)**, unique **102/192 (53.1%)**.

After extension + adaptation, vs **v0_8 gold** (and identically vs v0_7 gold):

| Seed | Overall exact | Shared exact | Unique exact | Shared diverge | Wes |
|---:|---:|---:|---:|---:|---:|
| 850001 | 36/128 | 3/64 | 33/64 | 26/64 | 0/16 |
| 850002 | 35/128 | 2/64 | 33/64 | 27/64 | 0/16 |
| 850003 | 40/128 | 4/64 | 36/64 | 30/64 | 0/16 |
| **Total** | **111/384 (28.9%)** | **9/192 (4.7%)** | **102/192 (53.1%)** | **83/192 (43.2%)** | **0/48** |

Greedy **never emitted an ID ≥ 1024** (max generated id 1019). Native free generation did not change.

Descriptive collision families (not optimization targets): Sal 3/16, 3/16, 4/16-class pattern remains; Skye/Omar/Opal/Wes remain at or near 0 on these seeds’ 16-row slices.

## Language and representation

Authoritative Phase1G DEV-stream CE (1280 frozen windows) after adaptation, sliced to 1024 logits, **matches T28 terminals exactly**:

| Seed | T28 DEV CE | Expanded sliced CE | Expanded full-vocab CE |
|---:|---:|---:|---:|
| 850001 | 1.2536188177764416 | 1.2536188177764416 | 1.2536194317042828 |
| 850002 | 1.2593662537634374 | 1.2593662537634374 | 1.2593668535351754 |
| 850003 | 1.2565930142998696 | 1.2565930142998696 | 1.256593645364046 |

Pointer on unmodified v0_7 DEV encodings: **112 / 108 / 116**, identical to T28. Representation path unchanged.

`RUN_RESULTS.json` also records a U0 sliced-CE identity check that accidentally indexed DEV window starts into the TRAIN stream (values ~1.03). That check still proves expanded vs original identity; it is **not** the Phase1G language number. Use `LANGUAGE_DEV_CE.json` for language status.

## Success standard

1. Shared-prefix exact material gain from 4.7% — **FAIL** (still 4.7%)
2. First-token-correct-then-diverge material drop — **FAIL** (still 43.2% shared)
3. Unique-prefix no material regression from 53.1% — **held** (still 53.1%)
4. Language retention — **PASS**
5. Relational representation intact — **PASS**
6. Native free generation improved — **FAIL** (byte-identical panel counts; no new-token emissions)
7. Replication — the null result replicates on 850001–850003
8. TEST sealed — **PASS**

## If append-only fails

- **Why:** Compatibility and initialization succeeded. New-row adaptation ran. The frozen TRAIN title-case frequency rule did not add DEV identity pieces and the expanded head never won greedy. Failure mode: **scientific efficacy of this selection + new-row-only budget**, not compatibility.
- **Option B** (new tokenizer lineage from TRAIN BPE, not ID-preserving append) remains a justified owner decision, not an auto-launch.
- **Option D** (full retokenization/retrain) is **not** shown necessary yet; B is the remaining tokenizer-lineage question.
- **Do not launch T29.** Wait for owner approval.

## Protected data

TEST loaded: **false**. T24 not resumed. T29 not launched. v0_7 not mutated.
