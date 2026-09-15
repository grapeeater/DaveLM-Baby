from __future__ import annotations

"""Minimal read-only interactive inference for the Baby v0.10 U6000 checkpoint.

This module intentionally does not import the training or evaluation pipelines.
It verifies the requested checkpoint and frozen v0.10 configuration, loads the
model in eval/inference mode, and records the human conversation separately.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import torch

from .config import BabyVNextConfig
from .model import BabyVNextLM


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "runs" / "diagnostic_v2r4_seed106001_8000" / "checkpoint_06000.pt"
CONFIG = ROOT / "configs" / "foundation_v1.json"
TOKENIZER = ROOT / "data" / "tokenizer" / "davelm_tokenizer_v0_7.json"
EXPECTED_CHECKPOINT_SHA256 = "75d2c761f34a5716d3f35c2118b5d3446f8b63decf2fe5781b45888c164037b3"
EXPECTED_TOKENIZER_SHA256 = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
PROTOCOL = "BABY_V010_FOUNDATION_V2R4"
U6000 = 6000
BOS = 2
EOS = 3
TEMPERATURE = 0.8
TOP_P = 0.9
MAX_NEW_TOKENS = 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def device_from_arg(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_verified_model(
    checkpoint_path: Path, config_path: Path, device: torch.device
) -> BabyVNextLM:
    observed_checkpoint_sha256 = sha256(checkpoint_path)
    if observed_checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "checkpoint SHA-256 mismatch: "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, got {observed_checkpoint_sha256}"
        )
    config = BabyVNextConfig.load(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("protocol") != PROTOCOL:
        raise RuntimeError(f"unexpected checkpoint protocol: {checkpoint.get('protocol')!r}")
    if checkpoint.get("update") != U6000:
        raise RuntimeError(f"expected U6000 checkpoint, got update {checkpoint.get('update')!r}")
    if checkpoint.get("parent_checkpoint") is not None:
        raise RuntimeError("checkpoint is not a fresh-initialization handoff")
    if checkpoint.get("protected_material_opened"):
        raise RuntimeError("checkpoint provenance reports protected material access")
    if checkpoint.get("config") != config.to_dict():
        raise RuntimeError("checkpoint configuration does not match foundation_v1.json")
    model = BabyVNextLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def top_p_sample(logits: torch.Tensor) -> int:
    logits = logits.float() / TEMPERATURE
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probabilities = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probabilities, dim=-1)
    remove = cumulative > TOP_P
    remove[0] = False
    probabilities = probabilities.masked_fill(remove, 0.0)
    probabilities = probabilities / probabilities.sum()
    selected = torch.multinomial(probabilities, num_samples=1)
    return int(sorted_indices[selected].item())


@torch.inference_mode()
def generate(model: BabyVNextLM, tokenizer, prompt: str, device: torch.device) -> str:
    prompt_ids = [BOS, *tokenizer.encode(prompt).ids]
    generated: list[int] = []
    for _ in range(MAX_NEW_TOKENS):
        context = prompt_ids + generated
        context = context[-model.config.context_length :]
        tokens = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        next_token = top_p_sample(model(tokens)[0, -1])
        if next_token == EOS:
            break
        generated.append(next_token)
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_chat(
    model: BabyVNextLM,
    tokenizer,
    device: torch.device,
    transcript_path: Path,
) -> None:
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[tuple[str, str]] = []
    with transcript_path.open("w", encoding="utf-8") as transcript:
        transcript.write("Baby v0.10 U6000 read-only chat transcript\n")
        transcript.write(f"checkpoint_sha256={EXPECTED_CHECKPOINT_SHA256}\n")
        transcript.write(f"tokenizer_sha256={EXPECTED_TOKENIZER_SHA256}\n")
        transcript.write(
            f"temperature={TEMPERATURE} top_p={TOP_P} max_new_tokens={MAX_NEW_TOKENS}\n\n"
        )
        print("Baby v0.10 U6000 is ready (read-only inference).", flush=True)
        print(f"Transcript: {transcript_path}", flush=True)
        print("Commands: /reset clears chat history; /exit closes the session.", flush=True)
        while True:
            try:
                user_text = input("\nYou> ")
            except (EOFError, KeyboardInterrupt):
                print("\nSession closed.", flush=True)
                break
            if user_text.strip().lower() == "/exit":
                print("Session closed.", flush=True)
                break
            if user_text.strip().lower() == "/reset":
                history.clear()
                transcript.write("[history reset]\n")
                transcript.flush()
                print("History cleared.", flush=True)
                continue
            if not user_text.strip():
                continue
            dialogue = "".join(
                f"Human: {human}\nBaby: {baby}\n" for human, baby in history
            )
            reply = generate(model, tokenizer, dialogue + f"Human: {user_text}\nBaby:", device)
            history.append((user_text, reply))
            transcript.write(f"Human: {user_text}\nBaby: {reply}\n\n")
            transcript.flush()
            print(f"Baby> {reply}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--transcript", type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    tokenizer_sha256 = sha256(args.tokenizer)
    if tokenizer_sha256 != EXPECTED_TOKENIZER_SHA256:
        raise RuntimeError(
            "tokenizer SHA-256 mismatch: "
            f"expected {EXPECTED_TOKENIZER_SHA256}, got {tokenizer_sha256}"
        )
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    device = device_from_arg(args.device)
    model = load_verified_model(args.checkpoint, args.config, device)
    transcript = args.transcript or (
        ROOT
        / "chat_transcripts"
        / f"BABY_U6000_CHAT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                "tokenizer": str(args.tokenizer),
                "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
                "config": str(args.config),
                "protocol": PROTOCOL,
                "update": U6000,
                "device": str(device),
                "sampling": {
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "max_new_tokens": MAX_NEW_TOKENS,
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    run_chat(model, tokenizer, device, transcript)


if __name__ == "__main__":
    main()
