"""Shared fixtures: a tiny synthetic English-like corpus, a tiny tokenizer, prepared streams and a tiny config.

Everything lives under a per-session temporary workspace so the tests never touch real runs or historical files.
GPU tests run on the GPU (RNC is GPU-first); they are skipped only if no GPU is visible, and that skip is itself
reported as a failure of the health check by run_health_checks.ps1.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rnc import gpu  # noqa: E402

gpu.configure_environment()

from rnc.config import load_config  # noqa: E402
from rnc.data import prepare, validate_prepared  # noqa: E402
from rnc.tokenizer import RNCTokenizer, train_tokenizer  # noqa: E402

ADJ = ["big", "small", "red", "blue", "happy", "sad", "quick", "quiet", "little", "old"]
NOUN = ["dog", "cat", "bird", "girl", "boy", "tree", "car", "ball", "fish", "frog"]
VERB = ["saw", "liked", "found", "chased", "wanted", "hugged", "watched", "lost"]
NAMES = ["Lily", "Tom", "Max", "Mia", "Sam", "Ben"]


def make_corpus(path: Path, n_docs: int, seed: int = 7) -> None:
    rng = random.Random(seed)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for _ in range(n_docs):
            n_sent = rng.randint(2, 5)
            sents = []
            for _ in range(n_sent):
                sents.append(f"{rng.choice(NAMES)} {rng.choice(VERB)} the {rng.choice(ADJ)} {rng.choice(NOUN)}.")
            if rng.random() < 0.3:
                sents.append(f'"{rng.choice(ADJ).capitalize()}!" said {rng.choice(NAMES)}. It was {rng.randint(1, 999)} days.')
            f.write(" ".join(sents) + "\n<|endoftext|>\n")


@pytest.fixture(scope="session")
def tiny_workspace(tmp_path_factory):
    ws = tmp_path_factory.mktemp("rnc_tiny_ws")
    (ws / "data" / "raw").mkdir(parents=True)
    make_corpus(ws / "data" / "raw" / "tiny.txt", 3000)
    cfg = {
        "workspace": str(ws),
        "model": {"vocab_size": 512, "d_model": 64, "n_layers": 2, "n_heads": 4, "d_mlp": 128, "context_length": 64},
        "tokenizer": {"path": "tokenizer/tiny", "vocab_size": 512, "expected_sha256": ""},
        "data": {"corpus_path": "data/raw/tiny.txt", "corpus_sha256": "", "prepared_dir": "data/prepared/tiny",
                 "validation_fraction": 0.1, "max_validation_docs": 500, "split_seed": 3, "min_doc_chars": 5,
                 "tokenizer_train_docs": 2000},
        "train": {"run_name": "tiny_run", "out_dir": "runs", "seed": 11, "precision": "fp32", "attention_backend": "auto",
                  "micro_batch_size": 8, "grad_accum_steps": 2, "total_steps": 12, "warmup_steps": 2, "lr": 3e-3,
                  "log_interval": 1, "eval_interval": 6, "eval_batches": 4, "checkpoint_interval": 6,
                  "sample_interval": 6, "sample_prompts": ["Lily saw the"], "sample_max_new_tokens": 12},
    }
    (ws / "configs").mkdir()
    (ws / "configs" / "tiny.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return ws


@pytest.fixture(scope="session")
def tiny_cfg(tiny_workspace):
    return load_config(tiny_workspace / "configs" / "tiny.json")


@pytest.fixture(scope="session")
def tiny_tokenizer(tiny_cfg):
    from rnc.data import count_documents, tokenizer_training_documents

    n = count_documents(tiny_cfg)
    docs = list(tokenizer_training_documents(tiny_cfg, n))
    out = tiny_cfg.resolve(tiny_cfg.tokenizer.path)
    train_tokenizer(docs, tiny_cfg.tokenizer.vocab_size, out, {"synthetic": True, "docs": len(docs)})
    return RNCTokenizer(out)


@pytest.fixture(scope="session")
def tiny_prepared(tiny_cfg, tiny_tokenizer):
    meta = prepare(tiny_cfg, tiny_tokenizer, log=lambda *a: None)
    validate_prepared(tiny_cfg, tiny_tokenizer)
    return meta


@pytest.fixture(scope="session")
def device():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("GPU not available (RNC health checks require the ROCm GPU)")
    return torch.device("cuda")


@pytest.fixture(autouse=True)
def _synchronize_gpu_after_each_test():
    """A process that exits with HIP work in flight hangs forever on this machine (observed again 2026-10-16 when a
    failing test left kernels queued). Drain the GPU after every test so a failure can never wedge the runner."""
    yield
    gpu.synchronize_quietly()


def pytest_sessionfinish(session, exitstatus):
    gpu.synchronize_quietly()


@pytest.fixture(scope="session")
def real_cfg():
    return load_config(ROOT / "configs" / "rnc1_tinystories.json")
