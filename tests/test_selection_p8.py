from __future__ import annotations

from src.baby_v010.selection_p8 import TRAIN_SEED, assert_p7_licenses_p8


def test_p8_identity() -> None:
    assert TRAIN_SEED == 220001
    adj = assert_p7_licenses_p8()
    assert adj["verdict"] == "NULL"
