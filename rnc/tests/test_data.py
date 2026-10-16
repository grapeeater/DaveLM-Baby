"""Data pipeline: deterministic split, held-out disjointness, stream integrity, batch determinism."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rnc.data import SplitAssigner, TokenStream, validate_prepared
from rnc.tokenizer import iter_documents


def test_split_assigner_deterministic_and_sequential():
    a = SplitAssigner(3, 0.1, 500)
    b = SplitAssigner(3, 0.1, 500)
    seq_a = [a.is_validation(i) for i in range(2000)]
    seq_b = [b.is_validation(i) for i in range(2000)]
    assert seq_a == seq_b
    assert 150 < sum(seq_a) < 250
    with pytest.raises(RuntimeError):
        a.is_validation(5)


def test_prepared_meta_and_validation(tiny_cfg, tiny_tokenizer, tiny_prepared):
    meta = tiny_prepared
    rep = validate_prepared(tiny_cfg, tiny_tokenizer)
    assert rep["train"]["docs"] == meta["stats"]["train"]["docs"]
    assert rep["val"]["docs"] == meta["stats"]["val"]["docs"] > 100
    assert meta["stats"]["train"]["docs"] + meta["stats"]["val"]["docs"] + meta["stats"]["duplicates_dropped"] == 3000


def test_validation_documents_absent_from_training(tiny_cfg, tiny_tokenizer, tiny_prepared):
    """Reconstruct both splits from the raw text and check no validation document has a copy in training."""
    assigner = SplitAssigner(tiny_cfg.data.split_seed, tiny_cfg.data.validation_fraction, tiny_cfg.data.max_validation_docs)
    seen = set()
    train_digests, val_docs = set(), []
    for i, d in enumerate(iter_documents(tiny_cfg.resolve(tiny_cfg.data.corpus_path), tiny_cfg.data.document_separator, tiny_cfg.data.min_doc_chars)):
        is_val = assigner.is_validation(i)
        h = hashlib.blake2b(d.encode(), digest_size=16).digest()
        if h in seen:
            continue
        seen.add(h)
        (val_docs.append(d) if is_val else train_digests.add(h))
    assert val_docs
    for d in val_docs:
        assert hashlib.blake2b(d.encode(), digest_size=16).digest() not in train_digests
    # and the val stream really decodes to those documents, in order
    meta = tiny_prepared
    arr = np.memmap(meta["files"]["val"]["path"], dtype=np.uint16, mode="r")
    ids = arr.tolist()
    docs = []
    cur = []
    for t in ids:
        if t == tiny_tokenizer.bos_id:
            cur = []
        elif t == tiny_tokenizer.eos_id:
            docs.append(tiny_tokenizer.decode(cur))
        else:
            cur.append(t)
    assert docs == val_docs


def test_random_batches_deterministic_and_shifted(tiny_prepared):
    s = TokenStream(Path(tiny_prepared["files"]["train"]["path"]))
    x1, y1 = s.random_batch(4, 16, seed=11, step=3, micro=0)
    x2, y2 = s.random_batch(4, 16, seed=11, step=3, micro=0)
    x3, _ = s.random_batch(4, 16, seed=11, step=4, micro=0)
    x4, _ = s.random_batch(4, 16, seed=11, step=3, micro=1)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert not torch.equal(x1, x3) and not torch.equal(x1, x4)
    assert torch.equal(x1[:, 1:], y1[:, :-1])
    assert x1.dtype == torch.int64 and x1.shape == (4, 16)


def test_fixed_batches_contiguous_non_overlapping(tiny_prepared):
    s = TokenStream(Path(tiny_prepared["files"]["val"]["path"]))
    x0, y0 = s.fixed_batch(0, 2, 8)
    x1, _ = s.fixed_batch(1, 2, 8)
    raw = torch.from_numpy(s.arr[: 4 * 9].astype(np.int64))
    assert torch.equal(x0[0], raw[0:8]) and torch.equal(y0[0], raw[1:9])
    assert torch.equal(x0[1], raw[9:17])
    assert torch.equal(x1[0], raw[18:26])
    with pytest.raises(ValueError):
        s.fixed_batch(10**6, 2, 8)


def test_corrupted_stream_detected(tiny_cfg, tiny_tokenizer, tiny_prepared, tmp_path):
    meta_path = tiny_cfg.resolve(tiny_cfg.data.prepared_dir) / "META.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bad = tmp_path / "val.bin"
    data = bytearray(Path(meta["files"]["val"]["path"]).read_bytes())
    data[100] ^= 0xFF
    bad.write_bytes(bytes(data))
    meta2 = dict(meta)
    meta2["files"] = {"train": meta["files"]["train"], "val": dict(meta["files"]["val"], path=str(bad))}
    (tmp_path / "META.json").write_text(json.dumps(meta2), encoding="utf-8")
    import copy

    cfg2 = copy.deepcopy(tiny_cfg)
    cfg2.data.prepared_dir = str(tmp_path)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_prepared(cfg2, tiny_tokenizer)
