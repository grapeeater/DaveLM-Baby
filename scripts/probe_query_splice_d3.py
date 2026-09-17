"""Run frozen D3 residual splice on the hashed parent.

Usage:
    python -B scripts/probe_query_splice_d3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baby_v010.query_splice_d3 import run  # noqa: E402


def main() -> None:
    report = run(device="cuda")
    decision = report["decision"]
    print(
        json.dumps(
            {
                "verdict": decision["verdict"],
                "early_query": decision["early_query"],
                "late_query": decision["late_query"],
                "match_key": decision["match_key"],
                "c11r_gold": decision["c11r_gold"],
                "licenses_next": decision["licenses_next"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
