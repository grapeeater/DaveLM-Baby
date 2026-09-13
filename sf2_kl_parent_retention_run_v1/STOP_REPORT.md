# STOP — SF2 preflight FAIL (environmental), training did not begin

Status: **PREFLIGHT FAIL — HARD STOP BEFORE TRAINING.**
Date: 2026-09-07.

## Summary

The SF2 KL-to-parent retention run could **not** pass its first frozen preflight gate: the pinned runtime
requires **Python 3.12.14** with **torch 2.12.0+rocm7.14.0** and **tokenizers 0.23.1** (`FROZEN_PROTOCOL_DRAFT.md`
§2; the SF2 harness is, per correction M10, the authoritative SF1 `ENGINE.py` reused byte-identical, which asserts
`platform.python_version() == '3.12.14'` at `ENGINE.py:110`). The only interpreter on this machine that provides
the required torch build is Python **3.12.0** (venv `C:\DaveLM\.venv`). **No Python 3.12.14 interpreter exists
anywhere on this machine** (exhaustive search: python.org installs, all `C:\DaveLM-*` trees, uv/conda/micromamba/
StabilityMatrix/Documents/Desktop/.local managed toolchains). Therefore the byte-identical harness aborts at its
environment assertion before any model is loaded and before any optimizer exists.

Per the mission's HARD STOP rule ("If ANY required hash, provenance check, ... or other frozen preflight
requirement fails, STOP BEFORE TRAINING. Report the failure. Do not repair it by changing the scientific
protocol, λ, data, gates, tolerances, or treatment"), this run **stops here**. No scientific protocol value, λ,
data, gate, tolerance, or harness line was changed to work around the environment gap. No training, optimizer
creation, autograd, checkpoint load, locked-panel scoring, or FINAL/sacred access occurred.

## What was verified (all PASS) before the environment gate

| Check | Result |
|---|---|
| Review-package `SHA256SUMS.txt` (REVIEW.md, REQUIRED_CORRECTIONS.md, FROZEN_PROTOCOL_DRAFT.md, PROVENANCE.md) | PASS (all 4 match) |
| Pilot1 parent `...\pilot_run\checkpoints\seed_8380\latest.pt` = `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb` | PASS |
| Tokenizer `C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json` = `e1c18bae...` | PASS |
| SF1 `TRAIN.json` = `08318d68...`, `PROTOCOL.json` = `5c236478...`, `SCHEDULE.json` = `91dad351...` | PASS |
| SF1 `data\ENGLISH_DEV.jsonl` = `2053a803...` | PASS |
| Pilot1 `language_train.jsonl` = `450f78bd...`, `language_dev.jsonl` = `deff4fc7...` | PASS |
| SF1 PROTOCOL `external_hashes` (11 v0_7/v0_8/v0_8_2 module files under `C:\DaveLM-v0.9`) | PASS (all match) |
| SF1 bundle `SHA256SUMS.txt` full manifest | PASS |

## What failed (environment gate)

Authoritative-harness environment gate, executed read-only with `--mode preload`
(no model load, no optimizer) under the only torch-2.12.0+rocm7.14.0 runtime available:

```
C:\DaveLM\.venv\Scripts\python.exe  ENGINE.py --bundle ... --mode preload
  File "...\ENGINE.py", line 110, in main
    assert platform.python_version()=='3.12.14' and tokenizers.__version__=='0.23.1'
AssertionError
```

Gate detail: torch `2.12.0+rocm7.14.0` PASS; `torch.cuda.is_available()` PASS (AMD Radeon RX 9060 XT);
`PYTHONHASHSEED=87011` PASS; `tokenizers 0.23.1` PASS; **`platform.python_version()=='3.12.14'` FAIL
(actual `3.12.0`)**. Raw capture: `run\env_gate_preload_attempt.txt`.

## Consequences and required next step

- Baseline reproduction (M9: acquisition 9/16, aligned CE 3.3907, D3 mass 0.000907, binding PASS, KL==0) was
  **not reached** because the harness cannot start under the available interpreter.
- Implementation/freeze of the SF2 harness, retention pool, and D3 diagnostic (mission steps 2-3) were **not
  executed**, deliberately: freezing an unvalidatable harness would entrench an untested artifact; the frozen
  protocol requires that these be frozen and validated in the pinned runtime before any checkpoint load. This is
  a non-scientific, environmental blocker, not a protocol change.
- **Required next step:** provide/install a **Python 3.12.14** interpreter with **torch 2.12.0+rocm7.14.0**,
  **tokenizers 0.23.1**, CUDA available, and `PYTHONHASHSEED=87011` (the exact runtime recorded in the SF1 run
  provenance and the SF2 frozen protocol). Then re-run the SF2 flow from step 1 (implementation of M1-M10,
  freeze+hash, preflight, update-0 baseline reproduction) exactly as frozen. Do not alter the protocol to
  accommodate Python 3.12.0.
