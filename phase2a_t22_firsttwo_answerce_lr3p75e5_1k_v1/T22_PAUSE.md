# T22 PAUSED — physically quiescent

Do **not** resume until the user explicitly says CONTINUE.
Do **not** launch 810002 / 810003 / T22X / T23.
TEST remains `SEALED_UNOPENED`. `qa_test.jsonl` was not loaded.

## Exact pause point
- Seed **810001** only. Phase **A**.
- Proven watchdog resume pair: `rolling_restart.pt` **`completed=550`**, **`phase=A`**, includes `model_state_dict` + `optimizer`.
- `STATUS.json`: `TRAINING_RUNNING`, update **550**, phase **A**.
- Last scheduled eval on disk: **U500** (`evaluation_0500.json`, `checkpoint_0500.pt`). Next scheduled eval would be U750 (not written).
- `training_metrics.jsonl` last line is update **580** (a few steps after the U550 persist, before stop). Resume from `completed=550` will redo 551+.

U0 DEV CE **1.2040123894810677** (bit-identical). Parent SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`.

## Paths
- Bundle: `C:\DaveLM-CADAVER\phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1`
- Run: `C:\DaveLM-CADAVER\phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1_run_seed810001`
- Resume file: `...\rolling_restart.pt` (573,021,453 bytes)
- Last eval ckpt: `...\checkpoints\checkpoint_0500.pt` (246,141,795 bytes)
- Last eval json: `...\evaluation_0500.json`
- Status: `...\STATUS.json`
- U0 eval: `...\evaluation_0000.json`

## SHA256
| file | SHA256 |
|---|---|
| `rolling_restart.pt` (resume) | `ff1000a8a0b117c7f9f28e866e9cf1d4920cceed59031c8c3bf69df8a10047b7` |
| `checkpoints\checkpoint_0500.pt` | `0ba71d0967fce73300170270b3593c7818f1e6deba8eb75059d2e9863dc8cfd7` |
| `evaluation_0500.json` | `c737dd662b7ec1a5c531f088f80627f4e6a8bf584cded960cb6130cdf2328096` |
| `STATUS.json` | `a9c3a85415df98815707614f75b15c95bea66ec63d8bf0a805eca5af28e69477` |

Hashes taken after sizes were stable; `rolling_restart` SHA re-checked after process stop (unchanged).

## PID kill confirmation
Stopped **T22 only**. T14 leftover PIDs 2644 / 19768 / 14592 were **not** touched.

| PID | role | result |
|---:|---|---|
| 23396 | T22 WATCHDOG (inner) | terminated |
| 14200 | T22 WATCHDOG (venv parent) | already gone after 23396 |
| 27036 | T22_RUNNER (inner) 810001 | terminated |
| 22592 | T22_RUNNER (venv parent) | terminated |
| 14032 | T22 LAUNCH_WATCHDOG.cmd | terminated |

Post-stop scan: **no** process command line matches `T22_RUNNER.py` or `t22_firsttwo`.

## 810002 / 810003
Never started. Directories do not exist:
- `phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1_run_seed810002` — absent
- `phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1_run_seed810003` — absent

No T22X / T23 / successor study was launched.

## TEST
`phase2a_t3_rebuilt_study_v1\data\TEST_SEAL.json` status **`SEALED_UNOPENED`**.
`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`. Seal file read only; `qa_test.jsonl` not opened.

## Resume (DO NOT RUN until CONTINUE)
Frozen recipe is unchanged. Watchdog resume path: if `810001` lacks `FINAL_STATUS.json` and `rolling_restart.pt` exists, launch with `--resume`.

```
cmd /c start "" /min C:\DaveLM-CADAVER\phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1\LAUNCH_WATCHDOG.cmd
```

That would resume **810001 from `rolling_restart` `completed=550` Phase A**, then after 810001 `FINAL_STATUS` would launch **810002 then 810003**. Do **not** run this command until the user says CONTINUE. Phase A→B still armed for after U750 eval (`after_u750_eval`).
