"""Run frozen D3b L0 control decomposition.

Usage:
    python -B scripts/probe_query_splice_d3b.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baby_v010.query_splice_d3b import run  # noqa: E402


def main() -> None:
    report = run(device="cuda")
    print(json.dumps(report["decision"]), flush=True)


if __name__ == "__main__":
    main()
