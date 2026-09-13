# Provenance — SF2 run v1 (STOP before training)

## Authoring context

- **Date:** 2026-09-07.
- **Mode:** read-only verification and environment gating only. No training, no optimizer, no autograd, no weight
  mutation, no checkpoint load, no locked/held-out panel, no FINAL/sacred access, no gate/tolerance/protocol
  change. The only writes are the files in this directory.
- **Authority:** `C:\DaveLM-CADAVER\sf2_kl_treatment_preflight_review_v1\` (REVIEW.md, REQUIRED_CORRECTIONS.md,
  FROZEN_PROTOCOL_DRAFT.md, PROVENANCE.md, SHA256SUMS.txt) — integrity verified, all four file hashes match.
- The referenced "DeepSeek V4 Flash independent review" artifacts are not present in the repo (see the preflight
  review PROVENANCE); no such content was reconstructed or relied on here.

## Environment discovery (decisive)

| Runtime | Interpreter | torch | tokenizers | CUDA |
|---|---|---|---|---|
| `C:\DaveLM\.venv\Scripts\python.exe` | Python 3.12.0 | 2.12.0+rocm7.14.0 | 0.23.1 | True (AMD Radeon RX 9060 XT) |
| `C:\Users\jdman\AppData\Local\Programs\Python\Python312\python.exe` | Python 3.12.0 | 2.9.1+rocmsdk20260116 | - | True |

Exhaustive search for any Python 3.12.14 on the machine (python.org installs, all `C:\DaveLM*` trees, uv, conda,
micromamba, pyenv, scoop, StabilityMatrix, Documents, Desktop, `.local`, `.cursor`, Temp, WinSxS scans) found
**none**. The SF2 frozen protocol (§2) and the byte-identical SF1 harness it must reuse (M10) require
`platform.python_version() == '3.12.14'`, asserted at `ENGINE.py:110`.

## Verifications performed and recorded

- All eight referenced artifact SHA-256 checks PASS (parent, tokenizer, TRAIN, PROTOCOL, SCHEDULE, ENGLISH_DEV,
  language_train, language_dev).
- SF1 PROTOCOL `external_hashes` (11 module files) PASS.
- SF1 bundle `SHA256SUMS.txt` full manifest PASS.
- Environment gate demonstration: `python -B ENGINE.py --bundle <SF1 bundle> --mode preload` under
  `C:\DaveLM\.venv` exited 1 with `AssertionError` at `ENGINE.py:110`
  (captured at `run\env_gate_preload_attempt.txt`). Configured gates that passed before the failure: torch
  version assert, CUDA availability, `PYTHONHASHSEED=87011`.

## Files written (this directory only)

- `STOP_REPORT.md` — full stop report (see also above).
- `PREFLIGHT_RESULT.json` — structured gate-by-gate result.
- `run\env_gate_preload_attempt.txt` — raw capture of the authoritative-harness environment-gate failure.
- `PROVENANCE.md` — this file.
- `SHA256SUMS.txt` — hashes of the above.

## Integrity statement

No pre-existing artifact anywhere was modified. No checkpoint, dataset, gate, or code was changed. No training
occurred. The SF2 harness implementation, retention-pool materialization, and executable freeze (mission steps
2-3) were deliberately NOT executed because the pinned runtime is unavailable; freezing an unvalidatable harness
would be worse than stopping, and the frozen protocol forbids working around the failure by changing the
environment expectation or the harness.
