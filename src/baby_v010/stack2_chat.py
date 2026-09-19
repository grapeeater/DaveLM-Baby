from __future__ import annotations

"""Interactive inference-only chat for stack2 s2a experimental Baby.

Thin wrapper around the verified language-bridge decode path used by
``selection_stack2.py`` and ``play_usable_chat()``. No training, no gate
changes, no authority/promotion updates, TEST/FINAL/SACRED stay closed.
"""

import argparse
from pathlib import Path

from .data import BOS
from .data_language_bridge import encode_ids, load_tokenizer
from .selection_language_bridge import greedy_decode_until_stop, load_experimental_baby
from .selection_p11_u16000_runtime import resolve_device

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    ROOT / "runs" / "actual_baby" / "stack2" / "s2a_protect40_combine_322001" / "checkpoint_00200.pt"
)

# Query stems used by usable-chat / dialogue eval (not fact sentences).
_QUERY_PREFIXES = (
    "what ",
    "which ",
    "who ",
    "how about ",
    "how is ",
    "tell ",
    "name ",
    "please ",
    "do you ",
    "can you ",
    "give ",
    "talk ",
    "stop ",
    "do not ",
)


def _looks_like_query(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True
    lower = stripped.lower()
    if lower.endswith(" then?") or lower.endswith(" then"):
        return True
    return any(lower.startswith(prefix) for prefix in _QUERY_PREFIXES)


def split_facts_and_human(user_text: str) -> tuple[str, str | None]:
    """Split a typed line the way play_usable_chat separates facts vs Human.

    Official first-turn prompt is ``{facts}\\nHuman: {query}\\nBaby:``.
    Stuffing facts inside ``Human:`` is off-protocol and collapses to a
    color prior (green.).
    """
    text = user_text.strip()
    if not text:
        return "", None
    if "\nHuman:" in text:
        prefix, _, rest = text.partition("\nHuman:")
        human = rest.split("\nBaby:", 1)[0].strip()
        return prefix.strip(), human or None
    last_break = max(text.rfind(". "), text.rfind("! "))
    if last_break >= 0:
        facts = text[: last_break + 1].strip()
        tail = text[last_break + 2 :].strip()
        if _looks_like_query(tail):
            return facts, tail
    if _looks_like_query(text):
        return "", text
    return text, None


def usable_chat_prompt(transcript: str, user_text: str) -> tuple[str, bool]:
    """Same construction as ``play_usable_chat``: facts live in ``transcript``."""
    extra_facts, human = split_facts_and_human(user_text)
    prefix = transcript.strip()
    if extra_facts:
        prefix = f"{prefix} {extra_facts}".strip() if prefix else extra_facts
    if human is None:
        return prefix, False
    if prefix:
        return f"{prefix}\nHuman: {human}\nBaby:", True
    return f"Human: {human}\nBaby:", True


def decode_user_turn(model, tokenizer, device, transcript: str, user_text: str) -> tuple[str, str]:
    prompt, should_generate = usable_chat_prompt(transcript, user_text)
    if not should_generate:
        return prompt, ""
    prompt_ids = [BOS, *encode_ids(tokenizer, prompt)]
    emitted, _stopped = greedy_decode_until_stop(
        model, prompt_ids, device, tokenizer, max_new=16
    )
    decoded = tokenizer.decode(emitted, skip_special_tokens=True)
    return prompt + decoded, decoded


def run_interactive_chat(model, tokenizer, device) -> None:
    transcript = ""
    print("s2a stack2 Baby ready (inference-only). Commands: /reset, /exit", flush=True)
    while True:
        try:
            user_text = input("\nYou> ")
        except (EOFError, KeyboardInterrupt):
            print("\nSession closed.", flush=True)
            break
        cmd = user_text.strip().lower()
        if cmd == "/exit":
            print("Session closed.", flush=True)
            break
        if cmd == "/reset":
            transcript = ""
            print("History cleared.", flush=True)
            continue
        if not user_text.strip():
            continue
        transcript, decoded = decode_user_turn(model, tokenizer, device, transcript, user_text)
        if decoded:
            print(f"Baby> {decoded}", flush=True)
        else:
            print("Context noted. Ask a question when ready.", flush=True)


def run_smoke(model, tokenizer, device) -> int:
    """Inference-only: in-domain lines must not all collapse to green."""
    cases = (
        ("The cat is blue. What color is the cat?", "blue", "green"),
        ("The dog is small in size. What size is the dog?", "small", "green"),
        ("fox looks white.\nHuman: Which color is the fox?", "white", "green"),
    )
    failed = 0
    for prompt, expect, banned in cases:
        _transcript, decoded = decode_user_turn(model, tokenizer, device, "", prompt)
        text = decoded.strip().lower()
        first = text.split()[0].strip(".,!?") if text else ""
        ok = expect in first and banned not in first
        print(f"smoke prompt={prompt!r} decoded={decoded!r} first={first!r} ok={ok}", flush=True)
        if not ok:
            failed += 1
    if failed:
        print(f"SMOKE FAIL {failed}/{len(cases)}", flush=True)
        return 1
    print(f"SMOKE PASS {len(cases)}/{len(cases)}", flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="inference-only in-domain prompts; no interactive session",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)
    tokenizer = load_tokenizer()
    model, _config, ckpt = load_experimental_baby(args.checkpoint, device)
    from .selection_stack2_r2 import RelAssistRuntime, WhoPropRuntime

    WhoPropRuntime(model, tokenizer).install().enabled = True
    RelAssistRuntime(model, tokenizer).install().enabled = True
    print(
        f"checkpoint={args.checkpoint} update={ckpt.get('update')} "
        f"protocol={ckpt.get('protocol')} device={device} routers=who_prop+rel_assist",
        flush=True,
    )
    if args.smoke:
        raise SystemExit(run_smoke(model, tokenizer, device))
    run_interactive_chat(model, tokenizer, device)


if __name__ == "__main__":
    main()
