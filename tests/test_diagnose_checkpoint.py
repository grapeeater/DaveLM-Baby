from __future__ import annotations

import json
from pathlib import Path

from src.baby_v010.diagnose_checkpoint import run_probe
from src.baby_v010.v2r4_provenance import TERMINAL_CHECKPOINT_SHA256

ROOT = Path(__file__).resolve().parents[1]
PANELS = ROOT / "data" / "generated" / "foundation_v2" / "panels.json"


def test_diagnose_checkpoint_skips_without_weights(tmp_path: Path):
    missing = tmp_path / "checkpoint_16000.pt"
    out = tmp_path / "probe"
    receipt = run_probe(missing, PANELS, out, "cpu", TERMINAL_CHECKPOINT_SHA256)
    assert receipt["status"] == "CHECKPOINT_ABSENT"
    assert receipt["trained"] is False
    assert receipt["protected_material_opened"] is False
    written = json.loads((out / "PROBE.json").read_text(encoding="utf-8"))
    assert written["status"] == "CHECKPOINT_ABSENT"
    assert "Do not launch training" in written["next"]
