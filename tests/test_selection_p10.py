from __future__ import annotations

from pathlib import Path

from src.baby_v010.selection_p10 import CE_TO_OVERWRITE, P9_LICENSE_VERDICT, TRAIN_SEED, assert_p9_licenses_p10


def test_p10_identity() -> None:
    assert TRAIN_SEED == 240001
    assert CE_TO_OVERWRITE is False
    adj = assert_p9_licenses_p10()
    assert adj["verdict"] == P9_LICENSE_VERDICT


def test_p10_source_isolates_ce() -> None:
    src = Path("src/baby_v010/selection_p10.py").read_text(encoding="utf-8")
    assert "hidden.detach() if not CE_TO_OVERWRITE" in src
    assert "if loss.requires_grad:" in src
    assert "with torch.no_grad():" in src
