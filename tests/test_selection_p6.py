from __future__ import annotations

from src.baby_v010.selection_p6 import TRAIN_SEED, assert_p5_licenses_p6


def test_p6_identity() -> None:
    assert TRAIN_SEED == 200001
    adj = assert_p5_licenses_p6()
    assert adj["verdict"] == "REGRESSION"
