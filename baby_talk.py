import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

# ---- DaveLM / Baby paths ----
ROOT = Path(r"C:\DaveLM-CADAVER")
ARCH = ROOT / "baby_vnext_60m_design_v1"
CHECKPOINT = ROOT / "baby_vnext_phase1g_language_v1_run_seed610001" / "checkpoints" / "best.pt"
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")

sys.path.insert(0, str(ARCH))

from baby_vnext.config import BabyVNextConfig
from baby_vnext.binding import BabyVNextWithBinding

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[Baby] Device: {device}")

# ---- Load trained checkpoint ----
print(f"[Baby] Loading: {CHECKPOINT}")
payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

print(f"[Baby] Checkpoint update: {payload.get('completed_update')}")
print(f"[Baby] Schema: {payload.get('schema')}")

# The trained checkpoint is model-only, so recover the exact architecture
# config from the original sealed initialization checkpoint.
INIT_CHECKPOINT = (
    ROOT
    / "baby_vnext_phase1g_language_v1"
    / "initialization"
    / "seed_610001_initialized.pt"
)

init_payload = torch.load(
    INIT_CHECKPOINT,
    map_location="cpu",
    weights_only=False,
)

config = BabyVNextConfig.from_dict(init_payload["config"])

model = BabyVNextWithBinding(config)
model.load_state_dict(payload["model_state_dict"], strict=True)
model.to(device)
model.eval()

tokenizer = Tokenizer.from_file(str(TOKENIZER))

print("[Baby] Model loaded successfully.")
print("[Baby] Type /quit to leave.")
print("[Baby] This is completion, NOT instruction-tuned chat.\n")


@torch.inference_mode()
def generate(prompt: str, max_new_tokens: int = 80) -> str:
    ids = tokenizer.encode(prompt).ids

    if not ids:
        return ""

    generated = list(ids)

    # Baby's context window is 256 tokens.
    for _ in range(max_new_tokens):
        context = generated[-256:]
        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=device,
        )

        logits, _ = model(x)

        # Greedy decoding for the first experiment.
        next_id = int(torch.argmax(logits[0, -1], dim=-1).item())
        generated.append(next_id)

        # Stop at EOS if the tokenizer defines one.
        token = tokenizer.id_to_token(next_id)
        if token in ("<eos>", "<|endoftext|>"):
            break

    # Show ONLY Baby's continuation separately.
    continuation_ids = generated[len(ids):]

    try:
        continuation = tokenizer.decode(
            continuation_ids,
            skip_special_tokens=True,
        )
    except TypeError:
        continuation = tokenizer.decode(continuation_ids)

    return continuation


while True:
    try:
        prompt = input("DAVE > ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[Baby] bye")
        break

    if prompt.lower() in {"/quit", "/exit", "quit", "exit"}:
        break

    if not prompt:
        continue

    try:
        answer = generate(prompt)
        print(f"BABY > {answer}\n")
    except Exception as exc:
        print(f"\n[Baby ERROR] {type(exc).__name__}: {exc}\n")