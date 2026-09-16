"""Baby v0.10 foundation model and capability pipeline."""

from .config import BabyVNextConfig

__all__ = ["BabyVNextConfig", "BabyVNextLM"]


def __getattr__(name: str):
    if name == "BabyVNextLM":
        from .model import BabyVNextLM

        return BabyVNextLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
