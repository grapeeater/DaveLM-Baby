# T24 PAUSED — physically quiescent

Do **not** resume until the user explicitly says CONTINUE.
Do **not** launch T24X / T25 / any competing treatment.
Do **not** adjudicate T24 as a scientific result in this pause.
TEST remains `SEALED_UNOPENED`. `qa_test.jsonl` was not loaded.

## Exact pause point
Watchdog **34608** wrote `STUDY_COMPLETE T24_REPRESENTATION_SUCCESS_OUTPUT_FAIL` at **2026-09-13T12:52:23Z** before this pause order was executed. All three seeds have `FINAL_STATUS.json` at **update 1000**, **phase B**. This is a physical completion of the frozen 1000-update protocol, **not** a scientific close-out.

Last live seed before watchdog exit:

- Seed **830003**. Phase **B**. `rolling_restart.pt` **`completed=1000`**, **`phase=B`**, includes `model_state_dict` + `optimizer`.
- `STATUS.json`: `COMPLETE`, seed 830003, update **1000**.
- Last scheduled eval on disk: **U1000** (`evaluation_1000.json`, `checkpoint_1000.pt`) for 830001 / 830002 / 830003.
- `training_metrics.jsonl` last line on 830003 is update **1000**.

U0 DEV CE **1.2040123894810677** (bit-identical on 830001 at launch). Parent SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`.

## What has / has not completed

| seed | FINAL_STATUS | last eval | rolling_restart | notes |
|---|---|---|---|---|
| 830001 | present, update 1000 | U1000 | present, 415,491,817 bytes | finished 2026-09-13T10:41:23Z |
| 830002 | present, update 1000 | U1000 | present, 415,491,817 bytes | finished 2026-09-13T11:46:48Z |
| 830003 | present, update 1000 | U1000 | present, 415,491,817 bytes, `completed=1000` phase B | finished 2026-09-13T12:52:23Z |

No T24X / T25 / successor study was launched. No `T24_FINAL_REPORT.md` was written.

## Paths
- Bundle: `C:\DaveLM-CADAVER\phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1`
- Runs: `C:\DaveLM-CADAVER\phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1_run_seed830001` / `830002` / `830003`
- Resume file (last live seed): `...\phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1_run_seed830003\rolling_restart.pt` (415,491,817 bytes)
- Last eval ckpt (830003): `...\checkpoints\checkpoint_1000.pt` (246,141,795 bytes)
- Last eval json (830003): `...\evaluation_1000.json`
- Status (830003): `...\STATUS.json`
- Watchdog log: `C:\DaveLM-CADAVER\phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1_watchdog.log`

## SHA256 (sizes stable at t0 and t+3s; 830003 rolling re-checked after leftover cmd stop)

| file | SHA256 |
|---|---|
| 830003 `rolling_restart.pt` (resume artifact) | `e00f9b625d31c9c5615d3c639422362b585c92427828c254dc639704b9a186c9` |
| 830003 `checkpoints\checkpoint_1000.pt` | `13ed3e09170022b8209094f07a1710442af8ff6bb1d2df9f6a2eeb907608b858` |
| 830003 `evaluation_1000.json` | `01a04da314b34f563ecc0a20a512c949015e0a9d1fdf83eff76ffc848dd29b4a` |
| 830003 `STATUS.json` | `faa88a4ddf5cfc640b4cb1512f09df54cb31466cff79abc8667be036efca60d9` |
| 830003 `FINAL_STATUS.json` | `88284e9b74c822338ff1e5a0cb360fe7b230cce1b985806689d24637554d4b95` |
| 830001 `rolling_restart.pt` | `1ed68db5430827f42d339358c62f1c34afc76a0a00ef3d0185f807f94b34efc0` |
| 830001 `checkpoints\checkpoint_1000.pt` | `bf8da7492d7e8276d5fd18c1a0249560ee29522864643f0affde3aa26cec15da` |
| 830002 `rolling_restart.pt` | `4d6b35dd144837f1047aee945deeb5cb8788b3a6c480ef83bd9f8e38ee6798e7` |
| 830002 `checkpoints\checkpoint_1000.pt` | `7bd1b8d27ee3d77e9cb5482e5189fcb7869133df494ba656dee69bdea30a6525` |

## PID confirmation
Watchdog **34608** and its venv parent **5808** were **already exited** after `STUDY_COMPLETE` (Get-Process: not found). T24 runners were already gone. Stopped **T24 leftover launcher only**. T22 leftover cmd 4604 and T23 leftover cmd 36356 were **not** touched. T14 leftover PIDs were **not** touched.

| PID | role | result |
|---:|---|---|
| 34608 | T24 WATCHDOG (inner) | already gone after STUDY_COMPLETE |
| 5808 | T24 WATCHDOG (venv parent) | already gone |
| 33560 / 8736 | last T24_RUNNER 830003 | already gone |
| 1760 | T24 `LAUNCH_WATCHDOG.cmd` leftover (`cmd /K`) | terminated this pause |

Post-stop scan: **no** process command line matches `T24_RUNNER.py` or `t24_inventory`.

## TEST
`phase2a_t3_rebuilt_study_v1\data\TEST_SEAL.json` status **`SEALED_UNOPENED`**.
`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`. Seal file read only; `qa_test.jsonl` hashed for seal check only, not scored or printed.

## Resume (DO NOT RUN until CONTINUE)
All three seeds already have `FINAL_STATUS.json`. The frozen watchdog will treat them as complete and will **not** train if launched.

```
cmd /c start "" /min C:\DaveLM-CADAVER\phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1\LAUNCH_WATCHDOG.cmd
```

Do **not** run this command. If CONTINUE were later issued against an incomplete seed, resume would be `--resume` from that seed’s `rolling_restart.pt`. 830003’s restart is `completed=1000` Phase B SHA256 `e00f9b62…b9a186c9`.
