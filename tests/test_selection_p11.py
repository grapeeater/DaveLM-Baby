from __future__ import annotations

from src.baby_v010.selection_p11 import CE_TO_OVERWRITE, GEN_ONLY, P10_LICENSE_VERDICT, TRAIN_SEED, assert_p10_licenses_p11


def test_p11_identity() -> None:
    assert TRAIN_SEED == 250001
    assert CE_TO_OVERWRITE is False
    assert GEN_ONLY is True
    adj = assert_p10_licenses_p11()
    assert adj["verdict"] == P10_LICENSE_VERDICT
