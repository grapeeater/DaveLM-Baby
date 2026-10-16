"""Checks on the real, pinned RNC tokenizer (not the tiny fixture)."""
from __future__ import annotations

import random

import pytest

from rnc.tokenizer import FIRST_BYTE_ID, RNCTokenizer
from tests.test_tokenizer import EDGE_CASES


@pytest.fixture(scope="module")
def tok(real_cfg):
    return RNCTokenizer(real_cfg.resolve(real_cfg.tokenizer.path), real_cfg.tokenizer.expected_sha256)


def test_pinned_and_sized(tok, real_cfg):
    assert tok.vocab_size == real_cfg.model.vocab_size
    assert tok.sha256 == real_cfg.tokenizer.expected_sha256
    assert tok.config["source"]["train_split_only"] is True


@pytest.mark.parametrize("text", EDGE_CASES)
def test_round_trip(tok, text):
    ids = tok.encode(text)
    assert all(i >= FIRST_BYTE_ID for i in ids)
    assert tok.decode(ids) == text


def test_random_round_trip(tok):
    rng = random.Random(99)
    for _ in range(500):
        s = "".join(chr(rng.choice([rng.randint(0x20, 0x7E), rng.randint(0xA0, 0x2FF), rng.randint(0x4E00, 0x4F00), 0x1F600 + rng.randint(0, 60), 10, 13, 9])) for _ in range(rng.randint(1, 60)))
        assert tok.decode(tok.encode(s)) == s


def test_english_is_compact(tok):
    text = "Once upon a time, there was a little girl named Lily. She loved to play with her dog in the park."
    ids = tok.encode(text)
    words = text.split()
    assert len(ids) <= len(words) + 6, tok.pieces(text)  # punctuation may add a few pieces, nothing else
    for w in [" the", " little", " girl", " park", " happy", " Once", " time"]:
        assert len(tok.encode(w)) == 1, (w, tok.pieces(w))


def test_numbers_split_in_small_groups(tok):
    assert tok.decode(tok.encode("1234567")) == "1234567"
    assert len(tok.encode("1234567")) >= 3  # groups of at most three digits
