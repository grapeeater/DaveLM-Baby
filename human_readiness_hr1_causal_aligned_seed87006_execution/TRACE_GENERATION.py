import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(r"C:\DaveLM-CADAVER")
BUNDLE = ROOT / "human_readiness_hr1_causal_aligned_seed87006_v1"
RUN = ROOT / "human_readiness_hr1_causal_aligned_seed87006_execution"
TOK_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
DEVICE = torch.device("cuda")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, r"C:\DaveLM-v0.9")
from treatment13_model import Treatment13Model

spec = importlib.util.spec_from_file_location(
    "pinned_binding", BUNDLE / "sources" / "PINNED_PILOT1_BINDING_IMPLEMENTATION.py"
)
pinned = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pinned)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    raw = torch.load(path, map_location=DEVICE, weights_only=False)
    state = dict(raw["model_state_dict"])
    vals = [state.pop("localizer." + k) for k in ("u", "q", "bs", "ba")]
    state.pop("localizer.scorer.weight", None)
    state.pop("localizer.scorer.bias", None)
    model = Treatment13Model().to(DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and set(missing) == {
        "localizer.scorer.weight",
        "localizer.scorer.bias",
    }
    model.localizer = pinned.OrthoLocalizer(*vals).to(DEVICE)
    return model.eval()


def trace(model, tok, prompt, temperature=1.0, top_k=None):
    prompt_ids = tok.encode(prompt).ids
    seq = [2] + prompt_ids
    rows = []
    with torch.no_grad():
        for step in range(32):
            logits = model.base_model(
                torch.tensor([seq[-256:]], dtype=torch.long, device=DEVICE)
            )[0, -1]
            probs = (logits / temperature).softmax(-1)
            if top_k is not None:
                vals, ids = torch.topk(probs, top_k)
                clipped = torch.zeros_like(probs)
                clipped[ids] = vals
                probs = clipped / clipped.sum()
            top = torch.topk(probs, 2)
            nxt = int(top.indices[0])
            rows.append(
                {
                    "step": step + 1,
                    "token_id": nxt,
                    "token": tok.decode([nxt], skip_special_tokens=True),
                    "top1_probability": float(top.values[0]),
                    "top1_margin": float(top.values[0] - top.values[1]),
                    "entropy": float(
                        -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum()
                    ),
                }
            )
            seq.append(nxt)
            if nxt == 3:
                break
    return {
        "prompt": prompt,
        "mode": {"temperature": temperature, "top_k": top_k},
        "decoded": tok.decode(seq, skip_special_tokens=True),
        "rows": rows,
        "length": len(rows),
        "immediate_eos": len(rows) == 1 and rows[0]["token_id"] == 3,
    }


def main():
    tok = Tokenizer.from_file(str(TOK_PATH))
    checkpoints = {
        "Pilot1": ROOT
        / "language_pilot_1_early_block_protection_seed8380"
        / "pilot_run"
        / "checkpoints"
        / "seed_8380"
        / "latest.pt",
        "HR1_causal_aligned_500": RUN / "run_factual" / "checkpoint_500.pt",
    }
    prompts = [
        "The dog ran to the park because",
        "Mia found a red ball and",
        "The small cat sat on the mat. It",
        "A boy opened the box and",
        "Tell me one thing about a sunny day.",
    ]
    result = {"tokenizer_sha256": sha(TOK_PATH), "checkpoints": {}}
    for name, path in checkpoints.items():
        model = load(path)
        result["checkpoints"][name] = {
            "checkpoint_sha256": sha(path),
            "greedy": [trace(model, tok, p) for p in prompts],
            "temperature07_top20": [
                trace(model, tok, p, temperature=0.7, top_k=20) for p in prompts
            ],
        }
    (RUN / "GENERATION_TRACE.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("GENERATION_TRACE_COMPLETE")


if __name__ == "__main__":
    main()
