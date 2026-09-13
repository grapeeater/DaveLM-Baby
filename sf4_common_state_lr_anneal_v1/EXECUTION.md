# Frozen execution order

Use `C:\DaveLM-CADAVER\sf2_runtime\sf2venv\Scripts\python.exe` in a fresh process for every command below.
Set `PYTHONHASHSEED=87011` and `PYTHONDONTWRITEBYTECODE=1`. Working directory is this package.

1. Run BUILD_PREFLIGHT.py once to verify inputs/mechanics and seal the prospective payload.
2. Run CONTROLLER.py --arm common_a --mode preload. This follows the real integrity/data dependency path.
3. Run CONTROLLER.py --arm common_a --mode train, then common_b --mode train in independent processes.
4. Run BRANCH_FREEZE.py. Any exact state, metric, baseline or safety mismatch stops the study.
5. Run CONTROLLER.py --arm control --mode branch-preflight, then treatment with the same mode.
6. Run CONTROLLER.py --arm control --mode train, then treatment --mode train, each independently.
7. Run FINALIZE.py only if both endpoints complete. It classifies before any eligible-arm transfer access.

If interrupted, use --resume for the interrupted arm only. It restores the latest committed rolling restart.
Never rerun a committed optimizer update. If a scheduled evaluation is missing, evaluate the committed state
without redoing the update. Corrupt state/provenance, safety failure or failed mandatory equality means stop.
No second treatment, extra seed, gate relaxation or rescue is authorized by this package.
