"""Run rebuilt T3 seeds 620001–620003 sequentially. TEST remains sealed."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PY = Path(r"C:\DaveLM-CADAVER\sf2_runtime\sf2venv\Scripts\python.exe")
RUNNER = Path(r"C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1\T3_RUNNER.py")
BUNDLE = Path(r"C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1")
ROOT = Path(r"C:\DaveLM-CADAVER")
SEEDS = (620001, 620002, 620003)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    ledger_path = BUNDLE / "RUN_LEDGER.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["status"] = "TRAINING_LAUNCHED"
    ledger["preflight_dev_ce"] = 1.2040123894810677
    ledger["preflight_pointer_u0"] = 64
    ledger["preflight_native_u0"] = 62
    write_json(ledger_path, ledger)
    results = {}
    for seed in SEEDS:
        out = ROOT / f"phase2a_t3_rebuilt_study_v1_run_seed{seed}"
        log = ROOT / f"phase2a_t3_rebuilt_study_v1_train_seed{seed}.log"
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [str(PY), "-u", str(RUNNER), "--mode", "train", "--seed", str(seed),
               "--bundle", str(BUNDLE), "--out", str(out)]
        write_json(ledger_path, {
            **json.loads(ledger_path.read_text(encoding="utf-8")),
            "status": f"TRAINING_SEED_{seed}",
            "active_seed": seed,
        })
        print(f"LAUNCH seed={seed} out={out}", flush=True)
        with open(log, "w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        status_path = out / "FINAL_STATUS.json"
        if proc.returncode != 0 or not status_path.exists():
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["status"] = f"TRAINING_FAILED_SEED_{seed}"
            ledger["failed_returncode"] = proc.returncode
            write_json(ledger_path, ledger)
            raise SystemExit(f"seed {seed} failed rc={proc.returncode}; see {log}")
        results[str(seed)] = json.loads(status_path.read_text(encoding="utf-8"))
        print(json.dumps(results[str(seed)], indent=2), flush=True)

    classes = [results[str(s)]["classification"] for s in SEEDS]
    n_full = sum(c == "T3_FULL_SUCCESS" for c in classes)
    n_rep = sum(c in ("T3_FULL_SUCCESS", "T3_REPRESENTATION_SUCCESS_OUTPUT_FAIL") for c in classes)
    if any(c == "T3_HARD_STOP" for c in classes):
        study = "T3_HARD_STOP"
    elif n_full >= 2:
        study = "T3_FULL_SUCCESS"
    elif n_rep >= 2:
        study = "T3_REPRESENTATION_SUCCESS_OUTPUT_FAIL"
    elif classes.count("T3_SHORTCUT_FAILURE") >= 2:
        study = "T3_SHORTCUT_FAILURE"
    elif classes.count("T3_LANGUAGE_REGRESSION") >= 2:
        study = "T3_LANGUAGE_REGRESSION"
    else:
        study = "T3_FAIL_NO_REPRESENTATION"
    study_rec = {
        "study": "BABY_VNEXT_PHASE2A_T3_REBUILT_EXPLICIT_REPRESENTATION_FORMING_V1",
        "study_classification": study,
        "seed_classifications": classes,
        "runs": results,
        "test_opened": False,
        "next": "If DEV representation AND native gates passed on a seed, score TEST once. Otherwise do not open TEST. Binding training remains a subsequent treatment only after this rebuilt T3 classification is on disk.",
    }
    write_json(BUNDLE / "STUDY_STATUS.json", study_rec)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["status"] = "STUDY_COMPLETE"
    ledger["study_classification"] = study
    ledger["runs"] = results
    write_json(ledger_path, ledger)
    report = BUNDLE / "T3_REBUILT_FINAL_REPORT.md"
    lines = [
        "# PHASE 2A T3 REBUILT STUDY — FINAL REPORT",
        "",
        f"Study classification: **{study}**",
        "",
        "## Seeds",
        "",
        "| seed | update | class | pointer | native | exact | DEV CE | binding |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for s in SEEDS:
        r = results[str(s)]
        lines.append(
            f"|{s}|{r['update']}|{r['classification']}|{r['dev_pointer']}|{r['dev_native']}|{r['dev_exact']}|{r['dev_ce']:.6f}|{r['binding_intact']}|"
        )
    lines += [
        "",
        "TEST was not opened.",
        "Parent SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`.",
        "This classification supersedes historical v1 HARD_STOP, repair INFRASTRUCTURE_BLOCKED, and name-overlapping ctxbind runs as the scientific T3 result.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(study_rec, indent=2), flush=True)


if __name__ == "__main__":
    main()
