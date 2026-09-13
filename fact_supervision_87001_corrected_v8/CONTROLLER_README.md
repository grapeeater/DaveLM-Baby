# v5 entrypoint

After human approval, execute one arm in a fresh process with the pinned runtime:

`$env:PYTHONHASHSEED=87002; python CONTROLLER.py --bundle C:\DaveLM-CADAVER\fact_supervision_87001_corrected_v5 --arm factual --seed 87002 --parent C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt --out C:\DaveLM-CADAVER\fact_supervision_87001_corrected_v5\run_factual --mode train`

Use `--arm control` and a distinct output directory for the matched control process. `--mode mock-0-update` exercises the same top-level controller without loading the real checkpoint, creating a real optimizer, or updating weights. The schedule is read from `MATERIALIZED_SCHEDULE_SEED87002.json`; no regeneration is permitted. Confirmation and sacred paths are rejected.
