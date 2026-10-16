"""One authoritative configuration for an RNC run.

A run is fully described by a single JSON file (see ``configs/``). Every key is validated against the dataclasses
below; unknown keys are an error so that a typo can never silently fall back to a default.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_mlp: int = 2048
    context_length: int = 512
    rope_base: float = 10000.0
    qk_norm: bool = True
    tie_embeddings: bool = True
    norm_eps: float = 1e-6
    init_std: float = 0.02

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self.d_model // self.n_heads


@dataclass
class TokenizerConfig:
    path: str = "tokenizer/rnc_bpe_8k"
    vocab_size: int = 8192
    expected_sha256: str = ""  # sha256 of tokenizer.json once frozen; empty = not pinned yet


@dataclass
class DataConfig:
    corpus_path: str = "data/raw/TinyStories-train.txt"
    corpus_sha256: str = ""
    document_separator: str = "<|endoftext|>"
    prepared_dir: str = "data/prepared/tinystories_8k"
    validation_fraction: float = 0.01
    max_validation_docs: int = 20000
    split_seed: int = 20261016
    min_doc_chars: int = 20
    tokenizer_train_docs: int = 400000  # documents sampled (deterministically) from the TRAIN split for BPE training


@dataclass
class TrainConfig:
    run_name: str = "rnc1_tinystories_shakedown_001"
    out_dir: str = "runs"
    seed: int = 1337
    device: str = "cuda"
    precision: str = "bf16"  # "bf16" (autocast, fp32 master weights) or "fp32"
    attention_backend: str = "auto"  # "auto" (SDPA, fused kernels when available) or "math"
    micro_batch_size: int = 16
    grad_accum_steps: int = 4
    total_steps: int = 1000
    warmup_steps: int = 100
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    log_interval: int = 10
    eval_interval: int = 250
    eval_batches: int = 50
    checkpoint_interval: int = 250  # every checkpoint is kept; they are evidence
    sample_interval: int = 250
    sample_prompts: list[str] = field(default_factory=lambda: [
        "Once upon a time, there was a little",
        "The dog was very",
        "One day, Lily went to the park and",
    ])
    sample_max_new_tokens: int = 120
    sample_temperature: float = 0.8
    sample_top_k: int = 50
    max_wall_minutes: float = 0.0  # 0 = no wall-clock limit; otherwise stop cleanly (with a checkpoint) after this


@dataclass
class RNCConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    workspace: str = ""  # absolute path of the rnc/ workspace; relative paths above resolve against it

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else Path(self.workspace) / p

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls, d: dict[str, Any], where: str):
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown config keys in {where}: {sorted(unknown)}")
    return cls(**d)


def load_config(path: str | Path) -> RNCConfig:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"model", "tokenizer", "data", "train", "workspace"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    cfg = RNCConfig(
        model=_build(ModelConfig, raw.get("model", {}), "model"),
        tokenizer=_build(TokenizerConfig, raw.get("tokenizer", {}), "tokenizer"),
        data=_build(DataConfig, raw.get("data", {}), "data"),
        train=_build(TrainConfig, raw.get("train", {}), "train"),
        workspace=raw.get("workspace", "") or str(path.resolve().parent.parent),
    )
    if cfg.model.vocab_size != cfg.tokenizer.vocab_size:
        raise ValueError("model.vocab_size must equal tokenizer.vocab_size")
    if cfg.train.precision not in ("bf16", "fp32"):
        raise ValueError("train.precision must be bf16 or fp32")
    if cfg.train.attention_backend not in ("auto", "math"):
        raise ValueError("train.attention_backend must be auto or math")
    _ = cfg.model.head_dim
    return cfg


def save_config(cfg: RNCConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
