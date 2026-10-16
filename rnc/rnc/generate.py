"""Load a checkpoint and continue text. Also the local chat/continuation REPL.

RNC-1 is a plain next-token model trained on short stories; "chat" here means you type a beginning and the model
continues it. Nothing is injected, filtered or hard-coded on the way in or out.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from . import gpu
from .checkpoint import load_checkpoint
from .config import RNCConfig, load_config
from .model import RNCModel
from .tokenizer import RNCTokenizer


def load_model_for_inference(checkpoint: Path, device) -> tuple[RNCModel, RNCTokenizer, RNCConfig]:
    checkpoint = Path(checkpoint)
    state = load_checkpoint(checkpoint, map_location=device)
    cfg_dict = state["config"]
    ckpt_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
    run_cfg_path = ckpt_dir.parent / "CONFIG.json"
    if not run_cfg_path.exists():
        raise SystemExit(f"CONFIG.json not found next to the run at {run_cfg_path}")
    cfg = load_config(run_cfg_path)
    if cfg.to_dict()["model"] != cfg_dict["model"]:
        raise SystemExit("checkpoint model config differs from run CONFIG.json")
    tok = RNCTokenizer(cfg.resolve(cfg.tokenizer.path), cfg.tokenizer.expected_sha256)
    if state["extra"].get("tokenizer_sha256") != tok.sha256:
        raise SystemExit("tokenizer fingerprint differs from the checkpoint")
    model = RNCModel(cfg.model).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, tok, cfg


@torch.no_grad()
def complete(model: RNCModel, tok: RNCTokenizer, prompt: str, device, max_new_tokens=150, temperature=0.8,
             top_k=50, seed: int | None = None) -> str:
    g = None
    if seed is not None:
        g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.tensor([[tok.bos_id] + tok.encode(prompt)], device=device)
    out = model.generate(ids, max_new_tokens, temperature=temperature, top_k=top_k, eos_id=tok.eos_id, generator=g)
    return tok.decode(out[0, ids.shape[1]:].tolist())


def generate_cli(checkpoint: Path, prompts: list[str], max_new_tokens: int, temperature: float, top_k: int, seed: int | None):
    device = gpu.require_gpu("cuda")
    try:
        model, tok, cfg = load_model_for_inference(checkpoint, device)
        results = []
        for p in prompts:
            text = complete(model, tok, p, device, max_new_tokens, temperature, top_k, seed)
            results.append({"prompt": p, "completion": text})
            print(f"--- prompt: {p!r}\n{p}{text}\n")
        return results
    finally:
        gpu.synchronize_quietly()


def chat_cli(checkpoint: Path, max_new_tokens: int, temperature: float, top_k: int):
    device = gpu.require_gpu("cuda")
    try:
        model, tok, cfg = load_model_for_inference(checkpoint, device)
        print(f"RNC-1 continuation REPL. {model.num_parameters():,} parameters. Type a beginning; the model continues it."
              " Empty line = start a fresh story. Ctrl-C to quit.")
        while True:
            try:
                prompt = input("you> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            text = complete(model, tok, prompt, device, max_new_tokens, temperature, top_k)
            print(f"rnc> {prompt}{text}\n")
    finally:
        gpu.synchronize_quietly()
