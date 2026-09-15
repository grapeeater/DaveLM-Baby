from __future__ import annotations

import json
from pathlib import Path

import torch

from src.baby_v010.config import BabyVNextConfig
from src.baby_v010.data import LANG_TRAIN, build_banks, build_panels, read_u16
from src.baby_v010.model import BabyVNextLM


def test_model_forward_backward() -> None:
    config = BabyVNextConfig.load(Path("configs/foundation_v1.json"))
    model = BabyVNextLM(config)
    x = torch.randint(0, config.vocab_size, (2, 32))
    z = model(x)
    assert z.shape == (2, 32, config.vocab_size)
    loss = z.float().square().mean()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_generator_surface_separation() -> None:
    banks = build_banks(read_u16(LANG_TRAIN))
    panels = build_panels(banks, 101000)
    assert panels["all_intact"]
    assert all(len(item["input"]) + len(item["target"]) <= 255 for item in panels["all_intact"])
    assert {item["surface"] for item in panels["heldout_surface"]} == {"heldout"}
    assert all(item["target_span"] for item in panels["all_intact"])
