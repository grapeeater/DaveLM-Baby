import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(r"C:\DaveLM-CADAVER")
BUNDLE = ROOT / "human_readiness_hr1_causal_aligned_seed87006_v1"
RUN = ROOT / "human_readiness_hr1_causal_aligned_seed87006_execution"
DEV_ITEMS = ROOT / "human_test_readiness_v1" / "DEV_ITEMS.jsonl"
TOK_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
DEVICE = torch.device("cuda")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, r"C:\DaveLM-v0.9")
from treatment13_model import Treatment13Model

PINNED_PATH = BUNDLE / "sources" / "PINNED_PILOT1_BINDING_IMPLEMENTATION.py"
spec = importlib.util.spec_from_file_location("pinned_binding", PINNED_PATH)
pinned = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pinned)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_model(path):
    raw = torch.load(path, map_location=DEVICE, weights_only=False)
    state = dict(raw["model_state_dict"])
    vals = [state.pop("localizer." + k) for k in ("u", "q", "bs", "ba")]
    state.pop("localizer.scorer.weight", None)
    state.pop("localizer.scorer.bias", None)
    model = Treatment13Model().to(DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected
    assert set(missing) == {"localizer.scorer.weight", "localizer.scorer.bias"}
    model.localizer = pinned.OrthoLocalizer(*vals).to(DEVICE)
    return model.eval()


def aligned_dev_loss(model):
    rows = [
        json.loads(line)
        for line in (BUNDLE / "data" / "ENGLISH_DEV.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    vals = []
    with torch.no_grad():
        for j in range(0, min(len(rows), 128), 32):
            chunk = rows[j : j + 32]
            width = max(len(r["token_ids"]) for r in chunk) + 2
            x = torch.zeros((len(chunk), width), dtype=torch.long, device=DEVICE)
            y = torch.full((len(chunk), width), -100, dtype=torch.long, device=DEVICE)
            for i, row in enumerate(chunk):
                z = [2] + row["token_ids"] + [3]
                x[i, : len(z) - 1] = torch.tensor(z[:-1], device=DEVICE)
                y[i, : len(z) - 1] = torch.tensor(z[1:], device=DEVICE)
            vals.append(
                float(
                    F.cross_entropy(
                        model.base_model(x).reshape(-1, 1024),
                        y.reshape(-1),
                        ignore_index=-100,
                    )
                )
            )
    loss = sum(vals) / len(vals)
    return {"loss": loss, "perplexity": math.exp(loss), "records": min(len(rows), 128)}


def candidate_score(model, prompt_ids, candidate_ids):
    full = [2] + prompt_ids + candidate_ids
    with torch.no_grad():
        logits = model.base_model(torch.tensor([full], dtype=torch.long, device=DEVICE))[0]
        lp = logits.log_softmax(-1)
        start = 1 + len(prompt_ids)
        positions = torch.arange(
            start - 1, start - 1 + len(candidate_ids), device=DEVICE
        )
        targets = torch.tensor(candidate_ids, dtype=torch.long, device=DEVICE)
        return float(lp[positions, targets].sum())


def greedy(model, prompt_ids):
    seq = [2] + prompt_ids
    with torch.no_grad():
        for _ in range(32):
            inp = torch.tensor([seq[-256:]], dtype=torch.long, device=DEVICE)
            nxt = int(model.base_model(inp)[0, -1].argmax())
            seq.append(nxt)
            if nxt == 3:
                break
    return seq


def score_items(model, tok, items):
    rows = []
    for row in items:
        if row["task"] != "fact":
            continue
        prompt_ids = tok.encode(row["prompt"]).ids
        candidate_ids = [tok.encode(c).ids for c in row["candidates"]]
        scores = [candidate_score(model, prompt_ids, ids) for ids in candidate_ids]
        margin = scores[0] - scores[1]
        predicted = 0 if margin > 0 else 1 if margin < 0 else None
        response = greedy(model, prompt_ids)
        generated = response[len([2] + prompt_ids) :]
        expected = candidate_ids[row["correct"]] + [3]
        rows.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "candidates": row["candidates"],
                "candidate_token_ids": candidate_ids,
                "correct_index": row["correct"],
                "candidate_log_scores": scores,
                "margin_candidate0_minus_candidate1": margin,
                "predicted_index": predicted,
                "correct": predicted == row["correct"],
                "tie": predicted is None,
                "candidate_pair_mass": [
                    math.exp(scores[k] - max(scores)) / sum(math.exp(s - max(scores)) for s in scores)
                    for k in range(2)
                ],
                "greedy_response_ids": generated,
                "greedy_exact_correct_then_eos": generated == expected,
            }
        )
    return rows


def generation_items(model, tok, items):
    rows = []
    for row in items:
        if row["task"] == "fact":
            continue
        prompt_ids = tok.encode(row["prompt"]).ids
        seq = greedy(model, prompt_ids)
        response_ids = seq[len([2] + prompt_ids) :]
        rows.append(
            {
                "id": row["id"],
                "task": row["task"],
                "prompt": row["prompt"],
                "prompt_ids": prompt_ids,
                "generated_ids": seq,
                "response_ids": response_ids,
                "raw_decoded_full": tok.decode(seq, skip_special_tokens=True),
                "raw_decoded_response": tok.decode(response_ids, skip_special_tokens=True),
                "immediate_eos": response_ids == [3],
                "generated_length_including_eos": len(response_ids),
            }
        )
    return rows


def fact_summary(rows):
    correct = sum(r["correct"] for r in rows)
    ties = sum(r["tie"] for r in rows)
    pairs = []
    ordered = sorted(rows, key=lambda r: int(r["id"][4:]))
    for i in range(0, len(ordered), 2):
        a, b = ordered[i], ordered[i + 1]
        pairs.append(
            {
                "ids": [a["id"], b["id"]],
                "both_correct": bool(a["correct"] and b["correct"]),
                "margins": [
                    a["margin_candidate0_minus_candidate1"],
                    b["margin_candidate0_minus_candidate1"],
                ],
            }
        )
    margins = [r["margin_candidate0_minus_candidate1"] for r in rows]
    greedy = sum(r["greedy_exact_correct_then_eos"] for r in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "ties": ties,
        "reversal_pairs": len(pairs),
        "successful_reversal_pairs": sum(p["both_correct"] for p in pairs),
        "complete_family_count": sum(p["both_correct"] for p in pairs),
        "greedy_exact_correct_then_eos": greedy,
        "mean_margin": sum(margins) / len(margins),
        "min_margin": min(margins),
        "max_margin": max(margins),
        "reversal_profiles": pairs,
    }


def summarize_binding(result):
    return {
        "documents": result["documents"],
        "overall": result["overall"],
        "complete_quartets": result["complete_quartets"],
        "reversal_pairs": result["reversal_pairs"],
        "reversal_both_correct": result["reversal_both_correct"],
        "reversal_rate": result["reversal_rate"],
    }


def evaluate_checkpoint(name, path, items, tok):
    model = load_model(path)
    facts = score_items(model, tok, items)
    generations = generation_items(model, tok, items)
    binding = {}
    for pool_name in ("pilot0", "pilot1"):
        data = json.loads(
            (BUNDLE / "data" / ("binding_dev_" + pool_name + ".json")).read_text(encoding="utf-8")
        )
        docs = [d for q in data["quartets"] for d in q["docs"]]
        binding[pool_name] = summarize_binding(pinned.binding_eval(model, docs, DEVICE))
    return {
        "name": name,
        "checkpoint": str(path),
        "checkpoint_sha256": sha(path),
        "aligned_dev_loss": aligned_dev_loss(model),
        "fact_summary": fact_summary(facts),
        "fact_items": facts,
        "generation_items": generations,
        "generation_summary": {
            "n": len(generations),
            "immediate_eos": sum(r["immediate_eos"] for r in generations),
            "immediate_eos_rate": sum(r["immediate_eos"] for r in generations) / len(generations),
            "mean_length": sum(r["generated_length_including_eos"] for r in generations) / len(generations),
        },
        "binding": binding,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tok = Tokenizer.from_file(str(TOK_PATH))
    items = [
        json.loads(line)
        for line in DEV_ITEMS.read_text(encoding="utf-8").splitlines()
    ]
    checkpoints = [
        (
            "Pilot1",
            ROOT
            / "language_pilot_1_early_block_protection_seed8380"
            / "pilot_run"
            / "checkpoints"
            / "seed_8380"
            / "latest.pt",
        ),
        ("HR1_causal_aligned_100", RUN / "run_factual" / "checkpoint_100.pt"),
        ("HR1_causal_aligned_500", RUN / "run_factual" / "checkpoint_500.pt"),
        (
            "fact_supervision_v8_factual_500",
            ROOT
            / "fact_supervision_87001_corrected_v8"
            / "run_factual"
            / "checkpoint_500.pt",
        ),
        (
            "fact_supervision_v8_control_500",
            ROOT
            / "fact_supervision_87001_corrected_v8"
            / "run_control"
            / "checkpoint_500.pt",
        ),
    ]
    out = {
        "status": "HR1_CAUSAL_ALIGNED_DEV_EVALUATION_COMPLETE",
        "battery": str(DEV_ITEMS),
        "battery_scope": "nonsacred DEV only; final readiness battery not accessed",
        "tokenizer_sha256": sha(TOK_PATH),
        "checkpoints": [evaluate_checkpoint(n, p, items, tok) for n, p in checkpoints],
    }
    (RUN / "DEV_EVALUATION.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
