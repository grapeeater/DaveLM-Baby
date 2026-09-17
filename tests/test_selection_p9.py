from __future__ import annotations

from src.baby_v010.selection_p9 import FUTILITY_MASS_200, TRAIN_SEED, assert_p8_licenses_p9


def test_p9_identity() -> None:
    assert TRAIN_SEED == 230001
    assert FUTILITY_MASS_200 == 0.05
    adj = assert_p8_licenses_p9()
    assert adj["verdict"] == "NULL"
