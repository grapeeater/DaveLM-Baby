from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .binding import BabyVNextWithBinding
from .config import BabyVNextConfig


def analytical_counts(config: BabyVNextConfig) -> dict[str, int]:
    d = config.d_model
    hidden = config.d_mlp
    vocab = config.vocab_size
    context = config.context_length
    attention = 4 * d * d + d
    mlp = 2 * d * hidden + hidden + d
    norms = 4 * d
    block = attention + mlp + norms
    token_embedding = vocab * d
    position_embedding = context * d
    final_norm = 2 * d
    lm_head = d * vocab + vocab
    base = token_embedding + position_embedding + config.n_layers * block + final_norm + lm_head
    localizer = 2 * d + 1
    retrieval = 2 * d * config.binding.retrieval_dim + 2 * d * d
    binding = localizer + retrieval
    return OrderedDict(
        token_embedding=token_embedding,
        position_embedding=position_embedding,
        attention_per_block=attention,
        mlp_per_block=mlp,
        norms_per_block=norms,
        block_total=block,
        all_blocks=config.n_layers * block,
        final_norm=final_norm,
        lm_head=lm_head,
        base_total=base,
        binding_localizer=localizer,
        binding_retrieval=retrieval,
        binding_total=binding,
        trained_total=base + binding,
    )


def physical_counts(model: BabyVNextWithBinding) -> dict[str, Any]:
    base = sum(parameter.numel() for parameter in model.base_model.parameters())
    binding = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("base_model.")
    )
    return {
        "base_total": base,
        "binding_total": binding,
        "trained_total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_total": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }

