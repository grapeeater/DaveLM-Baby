from __future__ import annotations

from src.baby_v010.selection_p7 import OVERWRITE_LR, TRAIN_SEED, assert_p6_licenses_p7


def test_p7_identity() -> None:
    assert TRAIN_SEED == 210001
    assert OVERWRITE_LR == 1e-3
    adj = assert_p6_licenses_p7()
    assert adj["verdict"] == "REGRESSION"
