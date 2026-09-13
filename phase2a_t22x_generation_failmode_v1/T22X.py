"""Read-only T22 generation fail-mode vs T18X / T21X.

Does not train. Does not load T3 TEST. Does not parent T22 checkpoints.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(r"C:\DaveLM-CADAVER")
OUT = ROOT / "phase2a_t22x_generation_failmode_v1"
T22 = ROOT / "phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1"
T3_DATA = ROOT / "phase2a_t3_rebuilt_study_v1" / "data"
sys.path.insert(0, str(T22))
sys.path.insert(0, str(ROOT / "baby_vnext_60m_design_v1"))

from T22_RUNNER import ARCH, CTX, EOS, TOK_PATH, TOK_SHA, load_jsonl, sha256  # noqa: E402

SEEDS = (810001, 810002, 810003)
SCHEMA = "baby_vnext_phase2a_t22_firsttwo_answerce_pointer_lr3p75e5_1k_model_v1"
SAMPLE_N = 8


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def classify_gen(new, target, wrongs):
    if new == target:
        return "exact"
    if not new or new[0] == EOS:
        return "immediate_eos"
    if new and new[0] == target[0]:
        return "first_token_correct_then_diverge" if new != target else "exact"
    for w in wrongs:
        if new == w:
            return "wrong_candidate"
        if new and new[0] == w[0]:
            return "wrong_candidate_first_token"
    if EOS in new:
        return "other_then_eos"
    return "unterminated_other"


def load_ckpt(path, device):
    from baby_vnext.config import BabyVNextConfig
    from baby_vnext.binding import BabyVNextWithBinding
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise RuntimeError(f"unexpected schema {payload.get('schema')}")
    model = BabyVNextWithBinding(BabyVNextConfig.load(ARCH / "BABY_VNEXT_CONFIG.json"))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


@torch.no_grad()
def greedy(model, row, device):
    prompt = [2] + [int(x) for x in row["prompt_token_ids"]]
    target = [int(t) for t in row["candidate_token_ids"][int(row["correct_index"])]] + [EOS]
    wrongs = [[int(t) for t in c] + [EOS] for i, c in enumerate(row["candidate_token_ids"]) if i != int(row["correct_index"])]
    gen = list(prompt)
    new = []
    for _ in range(max(len(target) + 4, 16)):
        logits, _ = model(torch.tensor([gen[-CTX:]], device=device))
        t = int(logits[0, -1].argmax())
        gen.append(t)
        new.append(t)
        if t == EOS:
            break
    return new, target, wrongs


def main():
    if sha256(Path(TOK_PATH)) != TOK_SHA:
        raise RuntimeError("tokenizer hash mismatch")
    seal = json.loads((T3_DATA / "TEST_SEAL.json").read_text(encoding="utf-8"))
    if seal["status"] != "SEALED_UNOPENED":
        raise RuntimeError("TEST seal unexpected")
    if sha256(T3_DATA / "qa_test.jsonl") != seal["sha256"]:
        raise RuntimeError("TEST hash mismatch")
    tok = Tokenizer.from_file(str(TOK_PATH))
    dev = {r["id"]: r for r in load_jsonl(T3_DATA / "qa_dev.jsonl")}
    device = torch.device("cuda")
    report = {"test_loaded": False, "seeds": {}}
    for seed in SEEDS:
        run = ROOT / f"phase2a_t22_firsttwo_answerce_lr3p75e5_1k_v1_run_seed{seed}"
        items = json.loads((run / "item_results_1000.json").read_text(encoding="utf-8"))["dev"]
        model = load_ckpt(run / "checkpoints" / "checkpoint_1000.pt", device)
        modes = Counter()
        first = Counter()
        misses = []
        for item in items:
            src = dev[item["id"]]
            new, target, wrongs = greedy(model, src, device)
            mode = classify_gen(new, target, wrongs)
            modes[mode] += 1
            if new and new[0] == target[0]:
                first["target_bos"] += 1
            else:
                first["not_target_bos"] += 1
            if (not item["exact_ok"]) and item["pointer_ok"] and len(misses) < SAMPLE_N:
                misses.append({
                    "id": item["id"], "correct_name": src["correct_name"], "mode": mode,
                    "target_txt": tok.decode(target), "gen_txt": tok.decode(new),
                })
        del model
        torch.cuda.empty_cache()
        report["seeds"][str(seed)] = {
            "exact_ok": sum(1 for r in items if r["exact_ok"]),
            "generation_modes": dict(modes),
            "first_token": dict(first),
            "miss_samples": misses,
        }
        print(seed, dict(modes), dict(first), flush=True)
    write_json(OUT / "FAILMODES.json", report)
    write_json(OUT / "STATUS.json", {"status": "COMPLETE", "test_loaded": False})
    print("WROTE", OUT / "FAILMODES.json", flush=True)


if __name__ == "__main__":
    main()
