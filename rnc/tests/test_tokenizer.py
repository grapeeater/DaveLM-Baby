"""Tokenizer correctness: exact round trips, control-token isolation, structure, determinism."""
from __future__ import annotations

import hashlib
import random

import pytest

from rnc.tokenizer import BYTE_ALPHABET_SIZE, CONTROL_TOKENS, FIRST_BYTE_ID, RNCTokenizer, train_tokenizer

EDGE_CASES = [
    "",
    "a",
    " ",
    "   ",
    "Hello, world!",
    "Once upon a time, there was a little girl named Lily.",
    "It costs $3.14, or 1,234,567 dollars; call 555-0199 at 09:45.",
    "naïve café résumé — “quotes” ‘single’ … ©2026",
    "日本語のテキストと한국어 그리고 Ελληνικά and русский",
    "emoji: 👩‍👩‍👧‍👦 🇺🇸 👍🏽 ☃️",
    "NFD: e\u0301 a\u030a  jamo: \u1100\u1161\u11a8",
    "line1\nline2\r\nline3\rline4",
    "tabs\tand\t\tmore\n\n\n\nblank lines",
    "trailing spaces   \nleading    spaces",
    " " * 64 + "x",
    "x" * 1000,
    "\x00 NUL and \x01 control \x7f DEL",
    "\ufeffBOM at start",
    "mixed 123abc456 7890123 00012",
    "<|eos|> literal control strings <|bos|><|pad|> in text <|reserved_0|>",
    "<|endoftext|>",
    "snake_case_name and CamelCaseName and kebab-case and path/to/file.txt",
    "C:\\Windows\\path with \"quotes\" and 'apostrophes' don't won't it's I'd we're they've",
]


def test_vocab_layout(tiny_tokenizer: RNCTokenizer):
    tok = tiny_tokenizer
    assert tok.vocab_size == 512
    for i, name in enumerate(CONTROL_TOKENS):
        assert tok.id_to_piece(i) == name
    byte_ids = {tok._tok.token_to_id(sym) for sym in __import__("tokenizers").pre_tokenizers.ByteLevel.alphabet()}
    assert byte_ids == set(range(FIRST_BYTE_ID, FIRST_BYTE_ID + BYTE_ALPHABET_SIZE))
    assert tok.pad_id == 0 and tok.bos_id == 1 and tok.eos_id == 2


@pytest.mark.parametrize("text", EDGE_CASES)
def test_round_trip_edge_cases(tiny_tokenizer: RNCTokenizer, text):
    ids = tiny_tokenizer.encode(text)
    assert all(FIRST_BYTE_ID <= i < tiny_tokenizer.vocab_size for i in ids), "text must never yield control ids"
    assert tiny_tokenizer.decode(ids) == text


def test_round_trip_random_unicode(tiny_tokenizer: RNCTokenizer):
    rng = random.Random(1234)
    for _ in range(300):
        n = rng.randint(1, 40)
        chars = []
        for _ in range(n):
            cp = rng.choice([rng.randint(0x20, 0x7E), rng.randint(0xA0, 0x24F), rng.randint(0x370, 0x3FF),
                             rng.randint(0x400, 0x4FF), rng.randint(0x3040, 0x30FF), rng.randint(0x4E00, 0x9FFF),
                             rng.randint(0x1F300, 0x1F64F), rng.randint(0x0, 0x1F), 0x0A, 0x0D, 0x09, 0x20])
            if 0xD800 <= cp <= 0xDFFF:
                cp = 0x41
            chars.append(chr(cp))
        text = "".join(chars)
        assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text


def test_round_trip_all_latin1_and_random_bytes(tiny_tokenizer: RNCTokenizer):
    text = "".join(chr(i) for i in range(256))
    assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text
    rng = random.Random(5)
    for _ in range(50):
        raw = bytes(rng.randint(0, 255) for _ in range(rng.randint(1, 64)))
        text = raw.decode("utf-8", errors="replace")
        assert tiny_tokenizer.decode(tiny_tokenizer.encode(text)) == text


def test_control_tokens_never_from_text(tiny_tokenizer: RNCTokenizer):
    for name in CONTROL_TOKENS:
        ids = tiny_tokenizer.encode(name)
        assert min(ids) >= FIRST_BYTE_ID
        assert tiny_tokenizer.decode(ids) == name


def test_document_encoding_structure(tiny_tokenizer: RNCTokenizer):
    text = "Tom saw the big dog. It was 7 days."
    ids = tiny_tokenizer.encode_document(text)
    assert ids[0] == tiny_tokenizer.bos_id and ids[-1] == tiny_tokenizer.eos_id
    assert all(i >= FIRST_BYTE_ID for i in ids[1:-1])
    assert tiny_tokenizer.decode(ids) == text
    assert tiny_tokenizer.decode(ids, skip_control=False) == f"<|bos|>{text}<|eos|>"


def test_batch_encode_matches_single(tiny_tokenizer: RNCTokenizer):
    texts = EDGE_CASES[:12]
    assert tiny_tokenizer.encode_batch(texts) == [tiny_tokenizer.encode(t) for t in texts]


def test_common_words_are_single_tokens(tiny_tokenizer: RNCTokenizer):
    # The synthetic corpus repeats these constantly; a sane BPE must merge them into single pieces.
    for w in [" the", " dog", " saw", " Lily", "."]:
        assert len(tiny_tokenizer.encode(w)) == 1, (w, tiny_tokenizer.pieces(w))


def test_training_is_deterministic(tmp_path, tiny_cfg):
    from rnc.data import count_documents, tokenizer_training_documents

    n = count_documents(tiny_cfg)
    docs = list(tokenizer_training_documents(tiny_cfg, n))
    a = train_tokenizer(docs, 512, tmp_path / "a", {})
    b = train_tokenizer(docs, 512, tmp_path / "b", {})
    assert a["tokenizer_json_sha256"] == b["tokenizer_json_sha256"]
    assert (tmp_path / "a" / "tokenizer.json").read_bytes() == (tmp_path / "b" / "tokenizer.json").read_bytes()
    assert a["tokenizer_json_sha256"] == hashlib.sha256((tmp_path / "a" / "tokenizer.json").read_bytes()).hexdigest()


def test_fingerprint_pin(tiny_tokenizer: RNCTokenizer):
    RNCTokenizer(tiny_tokenizer.dir, expected_sha256=tiny_tokenizer.sha256)
    with pytest.raises(RuntimeError, match="fingerprint"):
        RNCTokenizer(tiny_tokenizer.dir, expected_sha256="0" * 64)
