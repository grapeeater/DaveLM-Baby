"""Launch the frozen S2 matched comparison. Control first, then treatment."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(args: list[str]) -> None:
    subprocess.check_call([PY, "-u", "-B", *args], cwd=ROOT)


def main() -> None:
    out = ROOT / "runs/selection_s2"
    if not (out / "MANIFEST.json").exists():
        run(["-m", "src.baby_v010.selection_s2", "generate"])
    run(["scripts/preflight_selection_s2.py"])
    run(["-m", "src.baby_v010.selection_s2", "run", "--arm", "control", "--seed", "120001", "--until", "400"])
    run(["-m", "src.baby_v010.selection_s2", "run", "--arm", "treatment", "--seed", "120001", "--until", "400"])
    receipt_paths = sorted((out / "treatment_120001").glob("RECEIPT_*.json"))
    if not receipt_paths:
        raise SystemExit("treatment receipt missing")
    receipt = json.loads(receipt_paths[-1].read_text(encoding="utf-8"))
    step = int(receipt["last_step"])
    if step not in (200, 400):
        # still adjudicate the last full eval if it exists
        evals = sorted((out / "treatment_120001").glob("eval_*.json"))
        step = int(evals[-1].stem.split("_")[1])
        if step == 0:
            raise SystemExit("no post-baseline eval")
    run(["scripts/adjudicate_selection_s2.py", "--seed", "120001", "--step", str(step)])
    print(json.dumps({"status": "s2_driver_complete", "treatment_reason": receipt["reason"], "step": step}), flush=True)


if __name__ == "__main__":
    main()
