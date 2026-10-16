"""Dataset preparation and batching.

Preparation turns the raw corpus into two flat uint16 token streams (train.bin, val.bin) plus META.json:

* documents are streamed in file order and split into train / validation by a seeded random draw, so the split is
  reproducible from (corpus, split_seed, validation_fraction) alone and never depends on the tokenizer;
* exact duplicate documents are dropped (first occurrence wins), so no validation document has a byte-identical
  copy in training;
* each document is encoded as ``<|bos|> text <|eos|>`` and the streams are concatenated.

Batching samples random windows of ``context_length + 1`` tokens. Window positions are a pure function of
(seed, step, micro-step), so a resumed run draws exactly the batches an uninterrupted run would have drawn.
Validation uses a fixed, contiguous set of windows from val.bin so every evaluation measures the same tokens.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import RNCConfig
from .tokenizer import RNCTokenizer, iter_documents

DTYPE = np.uint16


class SplitAssigner:
    """Sequential, seeded train/validation assignment by document index."""

    def __init__(self, seed: int, fraction: float, max_validation: int):
        self.rng = random.Random(seed)
        self.fraction = fraction
        self.max_validation = max_validation
        self.n_val = 0
        self.expect = 0

    def is_validation(self, index: int) -> bool:
        if index != self.expect:
            raise RuntimeError("SplitAssigner must be called sequentially from index 0")
        self.expect += 1
        draw = self.rng.random()
        if draw < self.fraction and self.n_val < self.max_validation:
            self.n_val += 1
            return True
        return False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def count_documents(cfg: RNCConfig) -> int:
    corpus = cfg.resolve(cfg.data.corpus_path)
    return sum(1 for _ in iter_documents(corpus, cfg.data.document_separator, cfg.data.min_doc_chars))


def tokenizer_training_documents(cfg: RNCConfig, n_docs_total: int):
    """Deterministic sample of TRAIN-split documents for BPE training (never validation documents)."""
    corpus = cfg.resolve(cfg.data.corpus_path)
    want = cfg.data.tokenizer_train_docs
    est_train = max(1, int(n_docs_total * (1 - cfg.data.validation_fraction)))
    stride = max(1, est_train // want)
    assigner = SplitAssigner(cfg.data.split_seed, cfg.data.validation_fraction, cfg.data.max_validation_docs)
    taken = 0
    train_index = 0
    for i, doc in enumerate(iter_documents(corpus, cfg.data.document_separator, cfg.data.min_doc_chars)):
        if assigner.is_validation(i):
            continue
        if train_index % stride == 0 and taken < want:
            taken += 1
            yield doc
        train_index += 1


def prepare(cfg: RNCConfig, tok: RNCTokenizer, log=print) -> dict:
    corpus = cfg.resolve(cfg.data.corpus_path)
    out = cfg.resolve(cfg.data.prepared_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    corpus_sha = sha256_file(corpus)
    if cfg.data.corpus_sha256 and corpus_sha != cfg.data.corpus_sha256:
        raise RuntimeError(f"corpus hash {corpus_sha} != pinned {cfg.data.corpus_sha256}")
    log(f"corpus {corpus} sha256={corpus_sha}")
    assigner = SplitAssigner(cfg.data.split_seed, cfg.data.validation_fraction, cfg.data.max_validation_docs)
    seen: set[bytes] = set()
    stats = {"train": {"docs": 0, "tokens": 0, "bytes": 0}, "val": {"docs": 0, "tokens": 0, "bytes": 0},
             "duplicates_dropped": 0, "too_short_or_empty_skipped": 0, "max_doc_tokens": 0}
    files = {k: open(out / f"{k}.bin", "wb") for k in ("train", "val")}
    batch: dict[str, list[str]] = {"train": [], "val": []}

    def flush(split: str) -> None:
        docs = batch[split]
        if not docs:
            return
        for ids in tok.encode_batch(docs):
            arr = np.asarray([tok.bos_id] + ids + [tok.eos_id], dtype=DTYPE)
            if int(arr.max()) >= tok.vocab_size:
                raise RuntimeError("token id outside vocabulary")
            files[split].write(arr.tobytes())
            stats[split]["tokens"] += len(arr)
            stats["max_doc_tokens"] = max(stats["max_doc_tokens"], len(arr))
        batch[split] = []

    for i, doc in enumerate(iter_documents(corpus, cfg.data.document_separator, cfg.data.min_doc_chars)):
        split = "val" if assigner.is_validation(i) else "train"
        digest = hashlib.blake2b(doc.encode("utf-8"), digest_size=16).digest()
        if digest in seen:
            stats["duplicates_dropped"] += 1
            continue
        seen.add(digest)
        stats[split]["docs"] += 1
        stats[split]["bytes"] += len(doc.encode("utf-8"))
        batch[split].append(doc)
        if len(batch[split]) >= 2000:
            flush(split)
        if (i + 1) % 200000 == 0:
            log(f"  {i + 1} documents, {stats['train']['tokens'] + stats['val']['tokens']} tokens, {time.time() - t0:.0f}s")
    for split in ("train", "val"):
        flush(split)
        files[split].close()
    meta = {
        "corpus_path": str(corpus),
        "corpus_sha256": corpus_sha,
        "document_separator": cfg.data.document_separator,
        "split_seed": cfg.data.split_seed,
        "validation_fraction": cfg.data.validation_fraction,
        "max_validation_docs": cfg.data.max_validation_docs,
        "min_doc_chars": cfg.data.min_doc_chars,
        "tokenizer_sha256": tok.sha256,
        "vocab_size": tok.vocab_size,
        "dtype": "uint16",
        "stats": stats,
        "val_bytes_per_token": round(stats["val"]["bytes"] / max(1, stats["val"]["tokens"]), 5),
        "train_bytes_per_token": round(stats["train"]["bytes"] / max(1, stats["train"]["tokens"]), 5),
        "files": {k: {"path": str(out / f"{k}.bin"), "sha256": sha256_file(out / f"{k}.bin"),
                      "bytes": (out / f"{k}.bin").stat().st_size} for k in ("train", "val")},
        "seconds": round(time.time() - t0),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "META.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"prepared: train {stats['train']['docs']} docs / {stats['train']['tokens']} tokens; "
        f"val {stats['val']['docs']} docs / {stats['val']['tokens']} tokens; dropped dups {stats['duplicates_dropped']}")
    return meta


def validate_prepared(cfg: RNCConfig, tok: RNCTokenizer) -> dict:
    """Integrity checks on the prepared streams: hashes, id range, bos/eos structure, tokenizer pin."""
    out = cfg.resolve(cfg.data.prepared_dir)
    meta = json.loads((out / "META.json").read_text(encoding="utf-8"))
    if meta["tokenizer_sha256"] != tok.sha256:
        raise RuntimeError("prepared data was produced with a different tokenizer")
    report = {}
    for k in ("train", "val"):
        p = Path(meta["files"][k]["path"])
        if sha256_file(p) != meta["files"][k]["sha256"]:
            raise RuntimeError(f"{k}.bin hash mismatch")
        arr = np.memmap(p, dtype=DTYPE, mode="r")
        n = len(arr)
        if n != meta["stats"][k]["tokens"]:
            raise RuntimeError(f"{k}.bin length mismatch")
        if int(arr.max()) >= tok.vocab_size:
            raise RuntimeError(f"{k}.bin has an id outside the vocabulary")
        bos = int(np.count_nonzero(arr == tok.bos_id))
        eos = int(np.count_nonzero(arr == tok.eos_id))
        if bos != eos or bos != meta["stats"][k]["docs"] or arr[0] != tok.bos_id or arr[-1] != tok.eos_id:
            raise RuntimeError(f"{k}.bin document structure is inconsistent (bos={bos}, eos={eos})")
        report[k] = {"tokens": n, "docs": bos}
    return report


class TokenStream:
    def __init__(self, path: Path):
        self.arr = np.memmap(path, dtype=DTYPE, mode="r")
        self.n = len(self.arr)

    def _windows(self, starts: np.ndarray, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = starts[:, None] + np.arange(T + 1)[None, :]
        chunk = torch.from_numpy(self.arr[idx].astype(np.int64))
        return chunk[:, :-1], chunk[:, 1:]

    def random_batch(self, batch_size: int, T: int, seed: int, step: int, micro: int):
        g = torch.Generator().manual_seed((seed * 1_000_003 + step) * 7919 + micro)
        starts = torch.randint(0, self.n - T - 1, (batch_size,), generator=g).numpy()
        return self._windows(starts, T)

    def fixed_batch(self, batch_index: int, batch_size: int, T: int):
        """Contiguous, non-overlapping windows: batch i covers tokens [i*B*(T+1), (i+1)*B*(T+1))."""
        start = batch_index * batch_size * (T + 1)
        if start + batch_size * (T + 1) > self.n:
            raise ValueError("validation stream too short for the requested fixed batches")
        starts = start + np.arange(batch_size) * (T + 1)
        return self._windows(starts, T)
