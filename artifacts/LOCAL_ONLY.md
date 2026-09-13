# Local-only artifacts (not in git)

These files stay on this machine (or future LFS/release assets). Git records hashes and paths only.

## Phase1G parent `best.pt`

| | |
|---|---|
| Path | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001\checkpoints\best.pt` |
| SHA-256 | `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1` |

The entire `*_run_seed610001` tree is gitignored.

## T22 pause (seed 810001 only)

Run: `C:\DaveLM-CADAVER\phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1_run_seed810001`  
Source of hashes: `phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1\T22_PAUSE.md` (after process stop; sizes stable).

| file | bytes | SHA-256 |
|---|---:|---|
| `rolling_restart.pt` (resume, `completed=550`, phase A) | 573,021,453 | `ff1000a8a0b117c7f9f28e866e9cf1d4920cceed59031c8c3bf69df8a10047b7` |
| `checkpoints\checkpoint_0500.pt` | 246,141,795 | `0ba71d0967fce73300170270b3593c7818f1e6deba8eb75059d2e9863dc8cfd7` |
| `evaluation_0500.json` | (in run dir) | `c737dd662b7ec1a5c531f088f80627f4e6a8bf584cded960cb6130cdf2328096` |
| `STATUS.json` | (in run dir) | `a9c3a85415df98815707614f75b15c95bea66ec63d8bf0a805eca5af28e69477` |

810002 / 810003 run dirs do not exist. Do not create them while paused.

## Tokenizer

Git **does** track `DaveLM-v0.9/tokenizer/` (nested CADAVER copy).

| | |
|---|---|
| Path | `C:\DaveLM-CADAVER\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json` |
| SHA-256 | `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b` |
| Runners often still point at | `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json` (external, identical role) |

## Protected TEST bytes

**Not in git.** Do not open.

| | |
|---|---|
| Path | `C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1\data\qa_test.jsonl` |
| SHA-256 | `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb` |
| In git | `phase2a_t3_rebuilt_study_v1\data\TEST_SEAL.json` (`SEALED_UNOPENED`) |

Any other `qa_test.jsonl` is also local-only / gitignored.

## Phase1G `LANGUAGE_TRAIN_SOURCE.jsonl` (local-only)

Gitignored. The live file is under `data/` (not the bundle root).

| | |
|---|---|
| Canonical path | `C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\data\LANGUAGE_TRAIN_SOURCE.jsonl` |
| Bytes | 58,353,985 |
| SHA-256 | `807a89417924a9b1cfa1e0a2a952d97400684db32a138fe11943d93e130312c8` |
| Purpose | JSONL of 20,990 train documents written by `build_phase1g_bundle.py`. Tokenized into `data/LANGUAGE_TRAIN_STREAM.u16` (still in the bundle; used by T3–T22 language CE). |
| Relationship | Cited as `data.train_source` / `data.train_source_sha256` in `baby_vnext_phase1g_language_v1/PHASE1_LANGUAGE_CONFIG.json`. Head 9,000 docs match the frozen Phase-1 stream prefix. Regenerate only via the sealed Phase1G build; do not invent a second source. |

## HR `ENGLISH_TRAIN.jsonl` (26 byte-identical copies)

Verified 2026-09-12: **all 26 files share one SHA-256**. They are copies of one regenerated HR English train set, not 26 independent corpora.

| | |
|---|---|
| Canonical hash | `31617e7062ee0246b0a2e8a05d74bb568ee15ecdad43c92b18d0bc6f2a982f43` |
| Bytes (each) | 30,855,743 |
| Variants | **none** (1 hash × 26 paths) |
| Purpose | Deterministic sentence segments from the frozen Pilot 1 TinyStories train corpus (1–96 content tokens), used as the 9:1 English arm of HR1/HR2/HR3 (450 batches × 64 in `ENGLISH_SCHEDULE.json`). Materialized by HR build scripts from `language_pilot_1_early_block_protection_seed8380/language_train.jsonl`. |
| Relationship | Closed 10.6M human-readiness lineage only. Not used by Phase 2A / T22. `ENGLISH_DEV.jsonl` and schedules remain in those experiment dirs. |

Canonical local path (first HR1 tree):  
`C:\DaveLM-CADAVER\human_readiness_hr1_seed87004\ENGLISH_TRAIN.jsonl`

All 26 paths (same bytes, same hash):

```
human_readiness_hr1_causal_aligned_seed87006_v1/data/ENGLISH_TRAIN.jsonl
human_readiness_hr1_fullbase_causal_seed87007_v1/data/ENGLISH_TRAIN.jsonl
human_readiness_hr1_seed87004/ENGLISH_TRAIN.jsonl
human_readiness_hr1_seed87004_v2/ENGLISH_TRAIN.jsonl
human_readiness_hr1_seed87004_v3/ENGLISH_TRAIN.jsonl
human_readiness_hr1_seed87004_v4/ENGLISH_TRAIN.jsonl
human_readiness_hr1_seed87004_v5/ENGLISH_TRAIN.jsonl
human_readiness_hr1_seed87004_v6/ENGLISH_TRAIN.jsonl
human_readiness_hr1_seed87004_v7/ENGLISH_TRAIN.jsonl
human_readiness_hr1_seed87004_v8/ENGLISH_TRAIN.jsonl
human_readiness_hr2_seed87005/ENGLISH_TRAIN.jsonl
human_readiness_hr2_seed87005_v2/ENGLISH_TRAIN.jsonl
human_readiness_hr2_seed87005_v3/ENGLISH_TRAIN.jsonl
human_readiness_hr2_seed87005_v4/ENGLISH_TRAIN.jsonl
human_readiness_hr2_seed87005_v5/ENGLISH_TRAIN.jsonl
human_readiness_hr2_seed87005_v7/ENGLISH_TRAIN.jsonl
human_readiness_hr3_block3_causal_seed87006_v1/data/ENGLISH_TRAIN.jsonl
human_readiness_hr3_block3_causal_seed87006_v1_build_attempt1_failed/data/ENGLISH_TRAIN.jsonl
human_readiness_hr3_block3_causal_seed87006_v1_build_attempt2_failed/data/ENGLISH_TRAIN.jsonl
human_readiness_hr3_block3_causal_seed87006_v1_build_attempt3_failed/data/ENGLISH_TRAIN.jsonl
human_readiness_hr3_block3_causal_seed87006_v2/data/ENGLISH_TRAIN.jsonl
human_readiness_hr3_block3_causal_seed87006_v3/data/ENGLISH_TRAIN.jsonl
human_readiness_hr3_block3_causal_seed87006_v4/data/ENGLISH_TRAIN.jsonl
human_readiness_hr3_block3_causal_seed87006_v5/data/ENGLISH_TRAIN.jsonl
human_readiness_hr3_block3_causal_seed87006_v6/data/ENGLISH_TRAIN.jsonl
human_readiness_hr3_block3_causal_seed87006_v7/data/ENGLISH_TRAIN.jsonl
```

## Other local-only (see `.gitignore`)

`sf2_runtime/`, `archive/`, all `*_run_seed*` trees, `*.pt` / `*.safetensors` / `*.bin`, `t5_t8_structural_census.json`, logs, `.aider*`, `.config/`, `mistral_baby_history.json`, nested `DaveLM-v0.9/**` except `tokenizer/`, `**/ENGLISH_TRAIN.jsonl`, Phase1G `LANGUAGE_TRAIN_SOURCE.jsonl`.
