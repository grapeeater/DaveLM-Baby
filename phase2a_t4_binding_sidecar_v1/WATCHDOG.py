"""Detached T4 sequential watchdog. Never loads T3 TEST."""
from __future__ import annotations

import json, os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

PY = Path(r"C:\DaveLM-CADAVER\sf2_runtime\sf2venv\Scripts\python.exe")
RUNNER = Path(r"C:\DaveLM-CADAVER\phase2a_t4_binding_sidecar_v1\T4_RUNNER.py")
BUNDLE = Path(r"C:\DaveLM-CADAVER\phase2a_t4_binding_sidecar_v1")
ROOT = Path(r"C:\DaveLM-CADAVER")
SEEDS = (630001, 630002, 630003)
HEARTBEAT = BUNDLE / "WATCHDOG_HEARTBEAT.json"
LOG = ROOT / "phase2a_t4_binding_sidecar_v1_watchdog.log"
POLL = 20


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def live_cmds():
    try:
        raw = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "ForEach-Object { $_.ProcessId.ToString() + '|' + $_.CommandLine }"],
            text=True, errors="replace")
    except subprocess.CalledProcessError:
        return []
    return [ln.strip() for ln in raw.splitlines() if "T4_RUNNER.py" in ln or "t4_binding" in ln.lower() and "WATCHDOG" in ln]


def out_dir(seed):
    return ROOT / f"phase2a_t4_binding_sidecar_v1_run_seed{seed}"


def complete(seed):
    return (out_dir(seed) / "FINAL_STATUS.json").is_file()


def owner_live():
    return any("T4_RUNNER.py" in c for c in live_cmds())


def run_seed(seed, resume):
    out = out_dir(seed)
    log_path = ROOT / f"phase2a_t4_binding_sidecar_v1_train_seed{seed}.log"
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [str(PY), "-u", str(RUNNER), "--mode", "train", "--seed", str(seed),
           "--bundle", str(BUNDLE), "--out", str(out)]
    if resume:
        cmd.append("--resume")
    log(f"WATCHDOG_LAUNCH seed={seed} resume={resume}")
    mode = "a" if resume else "w"
    with open(log_path, mode, encoding="utf-8") as f:
        f.write(f"\n--- watchdog {now()} resume={resume} ---\n")
        proc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if complete(seed):
        if proc.returncode != 0:
            log(f"WATCHDOG_WARN seed={seed} rc={proc.returncode} FINAL_STATUS present")
        log(f"WATCHDOG_DONE seed={seed}")
        return
    raise SystemExit(f"seed {seed} failed rc={proc.returncode}")


def classify_study(results):
    classes = [results[str(s)]["classification"] for s in SEEDS]
    if any(c == "T4_HARD_STOP" for c in classes):
        return "T4_HARD_STOP"
    if classes.count("T4_BINDING_SUCCESS") >= 2:
        return "T4_BINDING_SUCCESS"
    if sum(c in ("T4_BINDING_SUCCESS", "T4_ROUTING_SUCCESS_OUTPUT_FAIL") for c in classes) >= 2:
        return "T4_ROUTING_SUCCESS_OUTPUT_FAIL"
    if classes.count("T4_LANGUAGE_REGRESSION") >= 2:
        return "T4_LANGUAGE_REGRESSION"
    return "T4_FAIL_NO_ROUTING"


def main():
    log(f"WATCHDOG_START pid={os.getpid()}")
    write_json(BUNDLE / "RUN_LEDGER.json", {
        "status": "WATCHDOG_STARTED", "study": "BABY_VNEXT_PHASE2A_T4_BINDING_SIDECAR_V1",
        "seeds": list(SEEDS), "runs": {}, "test_opened": False, "t3_test_loaded": False,
        "parent_sha256": "c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1",
    })
    while True:
        write_json(HEARTBEAT, {
            "utc": now(), "pid": os.getpid(),
            "complete": {str(s): complete(s) for s in SEEDS},
            "owner_live": owner_live(),
        })
        if all(complete(s) for s in SEEDS):
            results = {str(s): json.loads((out_dir(s) / "FINAL_STATUS.json").read_text(encoding="utf-8")) for s in SEEDS}
            study = classify_study(results)
            write_json(BUNDLE / "STUDY_STATUS.json", {
                "study": "BABY_VNEXT_PHASE2A_T4_BINDING_SIDECAR_V1",
                "study_classification": study, "runs": results, "test_opened": False,
            })
            ledger = json.loads((BUNDLE / "RUN_LEDGER.json").read_text(encoding="utf-8"))
            ledger["status"] = "STUDY_COMPLETE"
            ledger["study_classification"] = study
            ledger["runs"] = results
            write_json(BUNDLE / "RUN_LEDGER.json", ledger)
            log(f"STUDY_COMPLETE {study}")
            return
        if owner_live():
            time.sleep(POLL)
            continue
        pending = next((s for s in SEEDS if not complete(s)), None)
        if pending is None:
            continue
        od = out_dir(pending)
        if od.exists() and any(od.iterdir()) and (od / "rolling_restart.pt").is_file():
            run_seed(pending, True)
        elif od.exists() and any(od.iterdir()) and not complete(pending):
            raise SystemExit(f"incomplete seed {pending} without rolling_restart")
        else:
            run_seed(pending, False)


if __name__ == "__main__":
    main()
