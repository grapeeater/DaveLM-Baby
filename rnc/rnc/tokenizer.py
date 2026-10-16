"""RNC tokenizer: byte-level BPE with a fixed control-token namespace.

Design (lessons from JAWN-BPE / Baby-J and the S3 namespace defect, see docs/TOKENIZER.md):

* Byte-level alphabet, no normaliser -> every byte string round-trips exactly, any Unicode included.
* Llama-3-style pre-tokenisation regex: contractions, letter runs with one leading punctuation/space, digits in
  groups of up to three, punctuation runs, newline runs, and whitespace handled so trailing spaces attach to the
  following word.
* Trained with ``train_from_iterator`` on whole documents (the line-by-line file trainer can never learn
  newline-adjacent merges).
* Control tokens occupy ids 0..7 and are NEVER produced from text: ``encode`` runs with
  ``encode_special_tokens=True`` so the literal string "<|eos|>" in a document becomes ordinary bytes. Control ids
  are inserted only by :meth:`RNCTokenizer.encode_document`. There is exactly one codec; nothing else may call the
  underlying library directly.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

CONTROL_TOKENS = [
    "<|pad|>",        # 0
    "<|bos|>",        # 1 document start
    "<|eos|>",        # 2 document end
    "<|sep|>",        # 3 reserved for future dialogue turn separation
    "<|reserved_0|>", # 4
    "<|reserved_1|>", # 5
    "<|reserved_2|>", # 6
    "<|reserved_3|>", # 7
]
PAD_ID, BOS_ID, EOS_ID, SEP_ID = 0, 1, 2, 3
BYTE_ALPHABET_SIZE = 256
FIRST_BYTE_ID = len(CONTROL_TOKENS)

PRETOKENIZER_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


def _build_untrained(vocab_size: int) -> tuple[Tokenizer, trainers.BpeTrainer]:
    tok = Tokenizer(models.BPE(unk_token=None, byte_fallback=False, fuse_unk=False))
    tok.normalizer = None
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PRETOKENIZER_REGEX), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=False,
        special_tokens=CONTROL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        max_token_length=32,
    )
    return tok, trainer


def train_tokenizer(documents: Iterable[str], vocab_size: int, out_dir: str | Path, source_note: dict) -> dict:
    """Train a byte-level BPE on an iterable of whole documents and write tokenizer.json + TOKENIZER_CONFIG.json.

    Returns the config dict (including the SHA-256 fingerprint of tokenizer.json).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok, trainer = _build_untrained(vocab_size)
    tok.train_from_iterator(documents, trainer=trainer)
    if tok.get_vocab_size() != vocab_size:
        raise RuntimeError(f"trained vocab size {tok.get_vocab_size()} != requested {vocab_size}")
    # Control tokens must hold ids 0..7 and the 256 byte symbols the next 256 ids.
    vocab = tok.get_vocab()
    for i, t in enumerate(CONTROL_TOKENS):
        if vocab.get(t) != i:
            raise RuntimeError(f"control token {t} got id {vocab.get(t)}, expected {i}")
    for sym in pre_tokenizers.ByteLevel.alphabet():
        if not (FIRST_BYTE_ID <= vocab[sym] < FIRST_BYTE_ID + BYTE_ALPHABET_SIZE):
            raise RuntimeError(f"byte symbol {sym!r} outside the byte block: id {vocab[sym]}")
    path = out_dir / "tokenizer.json"
    tok.save(str(path))
    # Re-save canonically (sorted, no pretty gaps) so the fingerprint is byte-stable across runs.
    obj = json.loads(path.read_text(encoding="utf-8"))
    canonical = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    path.write_text(canonical, encoding="utf-8")
    cfg = {
        "name": f"rnc_bpe_{vocab_size}",
        "algorithm": "byte-level BPE (HF tokenizers)",
        "vocab_size": vocab_size,
        "control_tokens": CONTROL_TOKENS,
        "first_byte_id": FIRST_BYTE_ID,
        "n_merges": vocab_size - len(CONTROL_TOKENS) - BYTE_ALPHABET_SIZE,
        "pretokenizer_regex": PRETOKENIZER_REGEX,
        "normalizer": None,
        "max_token_length": 32,
        "min_frequency": 2,
        "tokenizer_json_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": source_note,
    }
    (out_dir / "TOKENIZER_CONFIG.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return cfg


class RNCTokenizer:
    """The one and only codec between text and RNC token ids."""

    def __init__(self, path: str | Path, expected_sha256: str = ""):
        self.dir = Path(path)
        self.json_path = self.dir / "tokenizer.json"
        self.sha256 = hashlib.sha256(self.json_path.read_bytes()).hexdigest()
        if expected_sha256 and expected_sha256 != self.sha256:
            raise RuntimeError(f"tokenizer fingerprint mismatch: {self.sha256} != pinned {expected_sha256}")
        self.config = json.loads((self.dir / "TOKENIZER_CONFIG.json").read_text(encoding="utf-8"))
        self._tok = Tokenizer.from_file(str(self.json_path))
        self._tok.encode_special_tokens = True  # literal control strings in text are just bytes
        self.vocab_size = self._tok.get_vocab_size()
        if self.vocab_size != self.config["vocab_size"]:
            raise RuntimeError("tokenizer.json vocab size does not match TOKENIZER_CONFIG.json")
        self.pad_id, self.bos_id, self.eos_id, self.sep_id = PAD_ID, BOS_ID, EOS_ID, SEP_ID

    # text -> ids (never emits control ids)
    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch(texts, add_special_tokens=False)]

    def encode_document(self, text: str) -> list[int]:
        return [self.bos_id] + self.encode(text) + [self.eos_id]

    # ids -> text
    def decode(self, ids: Iterable[int], skip_control: bool = True) -> str:
        ids = list(ids)
        if skip_control:
            ids = [i for i in ids if i >= FIRST_BYTE_ID]
            return self._tok.decode(ids, skip_special_tokens=False)
        out = []
        run: list[int] = []
        for i in ids:
            if i < FIRST_BYTE_ID:
                if run:
                    out.append(self._tok.decode(run, skip_special_tokens=False))
                    run = []
                out.append(CONTROL_TOKENS[i])
            else:
                run.append(i)
        if run:
            out.append(self._tok.decode(run, skip_special_tokens=False))
        return "".join(out)

    def id_to_piece(self, i: int) -> str:
        return self._tok.id_to_token(i)

    def pieces(self, text: str) -> list[str]:
        """Human-readable pieces (byte-level symbols decoded individually) for inspection."""
        return [self.decode([i]) for i in self.encode(text)]


def iter_documents(path: str | Path, separator: str, min_chars: int = 1) -> Iterator[str]:
    """Stream documents from a text file split on ``separator`` lines; strips surrounding whitespace."""
    buf: list[str] = []
    with open(path, encoding="utf-8", errors="strict", newline="") as f:
        for line in f:
            if line.strip() == separator:
                doc = "".join(buf).strip()
                if len(doc) >= min_chars:
                    yield doc
                buf = []
            else:
                buf.append(line)
    doc = "".join(buf).strip()
    if len(doc) >= min_chars:
        yield doc


def corpus_statistics(tok: RNCTokenizer, documents: Iterable[str]) -> dict:
    """Tokens per byte / per word and single-token word rate — used to justify the vocabulary size."""
    n_docs = n_bytes = n_tokens = n_words = n_single = 0
    for d in documents:
        ids = tok.encode(d)
        n_docs += 1
        n_bytes += len(d.encode("utf-8"))
        n_tokens += len(ids)
        words = d.split()
        n_words += len(words)
        for w in words[:50]:
            if len(tok.encode(" " + w)) == 1:
                n_single += 1
    return {
        "documents": n_docs,
        "bytes": n_bytes,
        "tokens": n_tokens,
        "bytes_per_token": round(n_bytes / max(1, n_tokens), 4),
        "tokens_per_word": round(n_tokens / max(1, n_words), 4),
        "single_token_word_rate_first50": round(n_single / max(1, min(n_words, 50 * n_docs)), 4),
    }
