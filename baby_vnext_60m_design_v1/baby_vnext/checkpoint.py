from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from .binding import BabyVNextWithBinding
from .config import BabyVNextConfig


SCHEMA_VERSION = "baby_vnext_checkpoint_v1"


def _state_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_initialized_checkpoint(
    path: str | Path,
    model: BabyVNextWithBinding,
    *,
    tokenizer_sha256: str,
    initialization_seed: int,
) -> dict[str, Any]:
    path = Path(path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "architecture": model.config.architecture,
        "config": model.config.to_dict(),
        "config_sha256": model.config.sha256(),
        "tokenizer_sha256": tokenizer_sha256,
        "initialization_seed": int(initialization_seed),
        "optimizer_updates": 0,
        "model_state_dict": model.state_dict(),
        "model_state_sha256": _state_digest(model),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def load_initialized_checkpoint(
    path: str | Path, *, expected_tokenizer_sha256: str
) -> tuple[BabyVNextWithBinding, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
    config = BabyVNextConfig.from_dict(payload["config"])
    if payload.get("config_sha256") != config.sha256():
        raise ValueError("checkpoint config hash mismatch")
    if payload.get("tokenizer_sha256") != expected_tokenizer_sha256:
        raise ValueError("checkpoint tokenizer hash mismatch")
    if payload.get("optimizer_updates") != 0:
        raise ValueError("validation checkpoint unexpectedly records training")
    model = BabyVNextWithBinding(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if _state_digest(model) != payload.get("model_state_sha256"):
        raise ValueError("checkpoint model-state digest mismatch")
    return model, payload


__all__ = [
    "SCHEMA_VERSION",
    "_state_digest",
    "save_initialized_checkpoint",
    "load_initialized_checkpoint",
]

