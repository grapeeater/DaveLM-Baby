import json

import pytest

from rnc.config import load_config


def test_real_config_loads(real_cfg):
    assert real_cfg.model.vocab_size == real_cfg.tokenizer.vocab_size == 8192
    assert len(real_cfg.tokenizer.expected_sha256) == 64, "tokenizer fingerprint must be pinned"
    assert real_cfg.model.head_dim == 64
    assert real_cfg.train.precision in ("bf16", "fp32")


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"model": {"d_model": 64, "typo_key": 1}}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(p)


def test_vocab_mismatch_rejected(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"model": {"vocab_size": 100}, "tokenizer": {"vocab_size": 200}}), encoding="utf-8")
    with pytest.raises(ValueError, match="vocab_size"):
        load_config(p)


def test_head_dim_divisibility(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"model": {"d_model": 100, "n_heads": 3}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p)
