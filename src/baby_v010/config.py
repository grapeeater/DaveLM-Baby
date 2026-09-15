from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class BindingConfig:
    enabled: bool = True
    num_slots: int = 2
    retrieval_dim: int = 128
    localizer: str = "orthogonal_two_slot"
    localizer_initialization: str = "normal_std_d_model_neg_half"
    layout_interface: str = "explicit_key_value_positions_v1"

    def validate(self, d_model: int) -> None:
        if self.num_slots != 2:
            raise ValueError("The preserved orthogonal localizer has exactly two slots")
        if self.retrieval_dim <= 0 or self.retrieval_dim > d_model:
            raise ValueError("retrieval_dim must be in [1, d_model]")
        if self.localizer != "orthogonal_two_slot":
            raise ValueError("unsupported localizer")
        if self.localizer_initialization != "normal_std_d_model_neg_half":
            raise ValueError("unsupported localizer initialization")


@dataclass(frozen=True)
class BabyVNextConfig:
    architecture: str = "baby_vnext_capacity_successor_v1"
    initialization: str = "pytorch_module_defaults_v1"
    vocab_size: int = 1024
    context_length: int = 256
    d_model: int = 640
    n_heads: int = 10
    n_layers: int = 12
    d_mlp: int = 2560
    activation: str = "relu"
    embedding_dropout: float = 0.05
    attention_dropout: float = 0.05
    residual_dropout: float = 0.05
    norm_type: str = "explicit_layer_norm"
    norm_epsilon: float = 1e-5
    norm_placement: str = "pre_norm"
    position_method: str = "learned_absolute"
    tie_embeddings: bool = False
    attention_backend: str = "sdpa"
    qkv_bias: bool = False
    attention_output_bias: bool = True
    mlp_bias: bool = True
    lm_head_bias: bool = True
    binding: BindingConfig = field(default_factory=BindingConfig)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def validate(self) -> None:
        if self.vocab_size <= 0 or self.context_length <= 0:
            raise ValueError("vocab_size and context_length must be positive")
        if self.d_model <= 0 or self.n_layers <= 0 or self.d_mlp <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must divide evenly across n_heads")
        if self.activation != "relu":
            raise ValueError("capacity-successor candidate preserves ReLU")
        if self.initialization != "pytorch_module_defaults_v1":
            raise ValueError("unsupported initialization policy")
        for value in (self.embedding_dropout, self.attention_dropout, self.residual_dropout):
            if not 0.0 <= value < 1.0:
                raise ValueError("dropout values must be in [0, 1)")
        if self.norm_type != "explicit_layer_norm" or self.norm_placement != "pre_norm":
            raise ValueError("candidate preserves the validated explicit pre-norm path")
        if self.position_method != "learned_absolute":
            raise ValueError("candidate preserves learned absolute positions")
        if self.tie_embeddings:
            raise ValueError("candidate preserves the untied output head")
        if self.attention_backend not in {"sdpa", "reference"}:
            raise ValueError("attention_backend must be sdpa or reference")
        self.binding.validate(self.d_model)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BabyVNextConfig":
        raw = dict(raw)
        raw["binding"] = BindingConfig(**raw.get("binding", {}))
        config = cls(**raw)
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> "BabyVNextConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def candidate_config() -> BabyVNextConfig:
    config = BabyVNextConfig()
    config.validate()
    return config
