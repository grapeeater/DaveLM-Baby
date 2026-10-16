"""Rare-token audit for candidate tokenizers: how much of each vocabulary is actually used on held-out text?

Usage: python tools/vocab_audit.py -c configs/rnc1_tinystories.json -n 20000 tokenizer/candidates/rnc_bpe_8192 ...
Prints, per tokenizer: ids never seen, ids seen < 10 times, tokens per word, and a sample of the highest-numbered
(latest-merged) pieces so the human can see what the extra vocabulary buys.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rnc.config import load_config  # noqa: E402
from rnc.data import SplitAssigner  # noqa: E402
from rnc.tokenizer import FIRST_BYTE_ID, RNCTokenizer, iter_documents  # noqa: E402


def held_out_docs(cfg, n):
    a = SplitAssigner(cfg.data.split_seed, cfg.data.validation_fraction, cfg.data.max_validation_docs)
    out = []
    for i, d in enumerate(iter_documents(cfg.resolve(cfg.data.corpus_path), cfg.data.document_separator, cfg.data.min_doc_chars)):
        if a.is_validation(i):
            out.append(d)
            if len(out) >= n:
                break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-n", type=int, default=20000)
    p.add_argument("tokenizers", nargs="+")
    p.add_argument("--out", default="")
    a = p.parse_args()
    cfg = load_config(a.config)
    docs = held_out_docs(cfg, a.n)
    n_words = sum(len(d.split()) for d in docs)
    n_bytes = sum(len(d.encode("utf-8")) for d in docs)
    results = []
    for tdir in a.tokenizers:
        tok = RNCTokenizer(tdir)
        counts = Counter()
        n_tokens = 0
        for ids in tok.encode_batch(docs):
            counts.update(ids)
            n_tokens += len(ids)
        V = tok.vocab_size
        text_ids = range(FIRST_BYTE_ID, V)
        never = sum(1 for i in text_ids if counts[i] == 0)
        rare = sum(1 for i in text_ids if counts[i] < 10)
        top_tail = [tok.decode([i]) for i in range(V - 12, V)]
        least_used_merges = sorted(((counts[i], tok.decode([i])) for i in range(FIRST_BYTE_ID + 256, V)))[:12]
        r = {"tokenizer": tdir, "vocab_size": V, "held_out_docs": len(docs), "tokens": n_tokens,
             "tokens_per_word": round(n_tokens / n_words, 4), "bytes_per_token": round(n_bytes / n_tokens, 4),
             "ids_never_seen": never, "ids_seen_lt_10": rare, "frac_seen_lt_10": round(rare / (V - FIRST_BYTE_ID), 4),
             "last_12_merges": top_tail, "least_used_12_merges": least_used_merges}
        results.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
