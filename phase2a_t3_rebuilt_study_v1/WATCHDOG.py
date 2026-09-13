"""Detached sequential T3 study watchdog.

Does not touch a live T3_RUNNER/RUN_STUDY. If those processes die, resumes
620001 from rolling_restart if needed, then launches 620002 and 620003.
Never opens TEST. Never trains binding.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PY = Path(r"C:\DaveLM-CADAVER\sf2_runtime\sf2venv\Scripts\python.exe")
RUNNER = Path(r"C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1\T3_RUNNER.py")
BUNDLE = Path(r"C:\DaveLM-CADAVER\phase2a_t3_rebuilt_study_v1")
ROOT = Path(r"C:\DaveLM-CADAVER")
SEEDS = (620001, 620002, 620003)
HEARTBEAT = BUNDLE / "WATCHDOG_HEARTBEAT.json"
LOG = ROOT / "phase2a_t3_rebuilt_study_v1_watchdog.log"
POLL = 20


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def python_cmds() -> list[str]:
    try:
        raw = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "ForEach-Object { $_.ProcessId.ToString() + '|' + $_.CommandLine }"],
            text=True, errors="replace",
        )
    except subprocess.CalledProcessError:
        return []
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


def live_cmds() -> list[str]:
    return [c for c in python_cmds() if "T3_RUNNER.py" in c or "RUN_STUDY.py" in c]


def seed_live(seed: int) -> bool:
    needle = f"--seed {seed}"
    return any("T3_RUNNER.py" in c and needle in c for c in live_cmds())


def study_owner_live() -> bool:
    return any("RUN_STUDY.py" in c or "T3_RUNNER.py" in c for c in live_cmds())


def out_dir(seed: int) -> Path:
    return ROOT / f"phase2a_t3_rebuilt_study_v1_run_seed{seed}"


def heartbeat(extra: dict) -> None:
    payload = {"utc": now(), "pid": os.getpid(), **extra}
    write_json(HEARTBEAT, payload)


def latest_update(seed: int):
    metrics = out_dir(seed) / "training_metrics.jsonl"
    if not metrics.is_file():
        return None
    last = None
    with open(metrics, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def complete(seed: int) -> bool:
    return (out_dir(seed) / "FINAL_STATUS.json").is_file()


def run_seed(seed: int, resume: bool) -> None:
    out = out_dir(seed)
    log_path = ROOT / f"phase2a_t3_rebuilt_study_v1_train_seed{seed}.log"
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [str(PY), "-u", str(RUNNER), "--mode", "train", "--seed", str(seed),
           "--bundle", str(BUNDLE), "--out", str(out)]
    if resume:
        cmd.append("--resume")
    log(f"WATCHDOG_LAUNCH seed={seed} resume={resume} cmd={' '.join(cmd)}")
    mode = "a" if resume else "w"
    with open(log_path, mode, encoding="utf-8") as f:
        f.write(f"\n--- watchdog launch {now()} resume={resume} ---\n")
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if complete(seed):
        if proc.returncode != 0:
            log(f"WATCHDOG_WARN seed={seed} rc={proc.returncode} but FINAL_STATUS present; continuing")
        log(f"WATCHDOG_DONE seed={seed}")
        return
    raise SystemExit(f"seed {seed} failed rc={proc.returncode}")


def classify_study(results: dict) -> str:
    classes = [results[str(s)]["classification"] for s in SEEDS]
    n_full = sum(c == "T3_FULL_SUCCESS" for c in classes)
    n_rep = sum(c in ("T3_FULL_SUCCESS", "T3_REPRESENTATION_SUCCESS_OUTPUT_FAIL") for c in classes)
    if any(c == "T3_HARD_STOP" for c in classes):
        return "T3_HARD_STOP"
    if n_full >= 2:
        return "T3_FULL_SUCCESS"
    if n_rep >= 2:
        return "T3_REPRESENTATION_SUCCESS_OUTPUT_FAIL"
    if classes.count("T3_SHORTCUT_FAILURE") >= 2:
        return "T3_SHORTCUT_FAILURE"
    if classes.count("T3_LANGUAGE_REGRESSION") >= 2:
        return "T3_LANGUAGE_REGRESSION"
    return "T3_FAIL_NO_REPRESENTATION"


def write_study(results: dict) -> None:
    study = classify_study(results)
    study_rec = {
        "study": "BABY_VNEXT_PHASE2A_T3_REBUILT_EXPLICIT_REPRESENTATION_FORMING_V1",
        "study_classification": study,
        "seed_classifications": [results[str(s)]["classification"] for s in SEEDS],
        "runs": results,
        "test_opened": False,
        "watchdog_pid": os.getpid(),
        "next": "TEST stays sealed unless DEV representation AND native gates passed. Binding training is not authorized yet.",
    }
    write_json(BUNDLE / "STUDY_STATUS.json", study_rec)
    ledger = json.loads((BUNDLE / "RUN_LEDGER.json").read_text(encoding="utf-8"))
    ledger["status"] = "STUDY_COMPLETE"
    ledger["study_classification"] = study
    ledger["runs"] = results
    ledger["test_opened"] = False
    write_json(BUNDLE / "RUN_LEDGER.json", ledger)
    log(f"STUDY_COMPLETE {study}")


def main() -> None:
    log(f"WATCHDOG_START pid={os.getpid()}")
    while True:
        live = live_cmds()
        lu = latest_update(620001)
        hb = {
            "live_cmds": live,
            "seed620001_complete": complete(620001),
            "seed620002_complete": complete(620002),
            "seed620003_complete": complete(620003),
            "seed620001_latest": lu,
            "study_owner_live": study_owner_live(),
        }
        heartbeat(hb)
        if all(complete(s) for s in SEEDS):
            results = {str(s): json.loads((out_dir(s) / "FINAL_STATUS.json").read_text(encoding="utf-8"))
                       for s in SEEDS}
            write_study(results)
            return
        if study_owner_live():
            log(f"owner live; 620001 latest={lu}; sleeping {POLL}s")
            time.sleep(POLL)
            continue
        # No owner: take over remaining work.
        pending = None
        for seed in SEEDS:
            if complete(seed):
                continue
            pending = seed
            break
        if pending is None:
            continue
        od = out_dir(pending)
        restart = od / "rolling_restart.pt"
        if od.exists() and any(od.iterdir()):
            if restart.is_file() and not complete(pending):
                log(f"resuming incomplete seed {pending}")
                run_seed(pending, resume=True)
            else:
                raise SystemExit(f"incomplete seed {pending} with no rolling_restart.pt")
        else:
            run_seed(pending, resume=False)


if __name__ == "__main__":
    main()
