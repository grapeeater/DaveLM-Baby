"""DEV-only decode comparison on frozen T28 checkpoints. Never loads TEST."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(r"C:\DaveLM-CADAVER")
ARCH = ROOT / "baby_vnext_60m_design_v1"
sys.path.insert(0, str(ARCH))
DEV_PATH = ROOT / "phase2a_t3_rebuilt_study_v1" / "data" / "qa_dev.jsonl"
TOK_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
TOK_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
SEEDS = (850001, 850002, 850003)
EXPECTED_CKPT = {
    850001: "d6f5be589c3905f76265431fa377c74245e972d9b28e8cee2c085d8f219e2684",
    850002: "f01066dd90aa3ff65b6affc7caad942486d6bfcc740733b24a440615ecf4f172",
    850003: "95e34fd52bc1b836a726816a2d8ae259e392620733146c2941aa53e00deaf597",
}
EOS, PERIOD, CTX, BOS = 3, 18, 256, 2
BEAM = 4
MAX_NEW = 12
OUT = ROOT / "tokenizer_repair_v1"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x]


def logits(model, ids: list[int], device: torch.device) -> torch.Tensor:
    out = model(torch.tensor([ids[-CTX:]], device=device))
    z = out[0] if isinstance(out, tuple) else out
    return z[0, -1]


@torch.no_grad()
def greedy(model, prompt: list[int], device: torch.device) -> list[int]:
    running, generated = list(prompt), []
    for _ in range(MAX_NEW):
        nxt = int(logits(model, running, device).argmax())
        running.append(nxt)
        generated.append(nxt)
        if nxt == EOS:
            break
    return generated


@torch.no_grad()
def free_beam(model, prompt: list[int], device: torch.device) -> list[int]:
    beams = [(0.0, list(prompt), [])]
    finished: list[tuple[float, list[int]]] = []
    for _ in range(MAX_NEW):
        nxt = []
        for lp, ids, gen in beams:
            if gen and gen[-1] == EOS:
                finished.append((lp / len(gen), gen))
                continue
            logp = F.log_softmax(logits(model, ids, device), dim=-1)
            vals, idx = torch.topk(logp, BEAM)
            for v, t in zip(vals.tolist(), idx.tolist()):
                t = int(t)
                nxt.append((lp + float(v), ids + [t], gen + [t]))
        nxt.sort(key=lambda x: x[0], reverse=True)
        beams = nxt[:BEAM]
        if not beams:
            break
    for lp, ids, gen in beams:
        if gen:
            finished.append((lp / len(gen), gen))
    if not finished:
        return greedy(model, prompt, device)
    finished.sort(key=lambda x: x[0], reverse=True)
    return finished[0][1]


@torch.no_grad()
def mean_logprob(model, prompt: list[int], cont: list[int], device: torch.device) -> float:
    x = torch.tensor([prompt + cont], device=device)
    out = model(x)
    z = out[0] if isinstance(out, tuple) else out
    lp = F.log_softmax(z[0, len(prompt) - 1:len(prompt) - 1 + len(cont)], dim=-1)
    idx = torch.tensor(cont, device=device)
    return float(lp[torch.arange(len(cont), device=device), idx].mean().cpu())


def classify(gen: list[int], target: list[int]) -> tuple[bool, bool, str]:
    exact = gen == target
    first = bool(gen) and bool(target) and gen[0] == target[0]
    if exact:
        return True, first, "exact"
    if first:
        return False, True, "first_token_correct_then_diverge"
    if gen and gen[-1] == EOS:
        return False, False, "other_then_eos"
    return False, False, "other"


def load_model(seed: int, device: torch.device):
    from baby_vnext.config import BabyVNextConfig
    from baby_vnext.binding import BabyVNextWithBinding
    ckpt = ROOT / f"phase2a_t28_fast_v2_run_seed{seed}" / "rolling_restart.pt"
    got = sha(ckpt)
    if got != EXPECTED_CKPT[seed]:
        raise RuntimeError(f"checkpoint hash mismatch {seed}: {got}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = BabyVNextWithBinding(BabyVNextConfig.load(ARCH / "BABY_VNEXT_CONFIG.json"))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model


def summarize(rows: list[dict], key: str) -> dict:
    n = len(rows)
    exact = sum(r[key]["exact"] for r in rows)
    diverge = sum(r[key]["mode"] == "first_token_correct_then_diverge" for r in rows)
    shared = [r for r in rows if r["shared_prefix"]]
    unique = [r for r in rows if not r["shared_prefix"]]
    return {
        "n": n,
        "exact": exact,
        "exact_rate": exact / n,
        "diverge": diverge,
        "diverge_rate": diverge / n,
        "shared_exact": sum(r[key]["exact"] for r in shared),
        "shared_n": len(shared),
        "shared_exact_rate": sum(r[key]["exact"] for r in shared) / len(shared),
        "unique_exact": sum(r[key]["exact"] for r in unique),
        "unique_n": len(unique),
        "unique_exact_rate": sum(r[key]["exact"] for r in unique) / len(unique),
        "by_name": {
            name: {
                "n": sum(r["name"] == name for r in rows),
                "exact": sum(r["name"] == name and r[key]["exact"] for r in rows),
            }
            for name in sorted({r["name"] for r in rows})
        },
    }


def main() -> None:
    if sha(TOK_PATH) != TOK_SHA:
        raise RuntimeError("v0_7 hash mismatch; refusing to run")
    _ = Tokenizer.from_file(str(TOK_PATH))
    rows_in = load_jsonl(DEV_PATH)
    first_ids = defaultdict(set)
    for r in rows_in:
        name = r["correct_name"]
        ids = list(map(int, r["candidate_token_ids"][int(r["correct_index"])]))
        first_ids[ids[0]].add(name)
    name_seqs = {}
    for r in rows_in:
        name = r["correct_name"]
        if name not in name_seqs:
            name_seqs[name] = list(map(int, r["candidate_token_ids"][int(r["correct_index"])]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []
    by_seed = {}
    for seed in SEEDS:
        model = load_model(seed, device)
        seed_rows = []
        for r in rows_in:
            prompt = [BOS] + list(map(int, r["prompt_token_ids"]))
            target = list(map(int, r["candidate_token_ids"][int(r["correct_index"])])) + [EOS]
            name = r["correct_name"]
            shared = len(first_ids[target[0]]) > 1
            g = greedy(model, prompt, device)
            b = free_beam(model, prompt, device)
            scores = {n: mean_logprob(model, prompt, seq, device) for n, seq in name_seqs.items()}
            pick = max(scores, key=scores.get)
            g_ex, g_ft, g_mode = classify(g, target)
            b_ex, b_ft, b_mode = classify(b, target)
            rec = {
                "seed": seed,
                "id": r["id"],
                "name": name,
                "shared_prefix": shared,
                "greedy": {"ids": g, "exact": g_ex, "first_token_correct": g_ft, "mode": g_mode},
                "beam": {"ids": b, "exact": b_ex, "first_token_correct": b_ft, "mode": b_mode},
                "constrained_8name": {
                    "pick": pick,
                    "exact": pick == name,
                    "mode": "exact" if pick == name else "wrong_candidate",
                    "label": "NON_NATIVE_INVENTORY_RERANK",
                },
            }
            seed_rows.append(rec)
            all_rows.append(rec)
        by_seed[str(seed)] = {
            "greedy": summarize(seed_rows, "greedy"),
            "beam": summarize(seed_rows, "beam"),
            "constrained_8name": summarize(seed_rows, "constrained_8name"),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {
        "status": "COMPLETE",
        "classification": "DECODE_COMPARISON_COMPLETE",
        "test_loaded": False,
        "tokenizer_changed": False,
        "beam_width": BEAM,
        "greedy": summarize(all_rows, "greedy"),
        "beam": summarize(all_rows, "beam"),
        "constrained_8name": summarize(all_rows, "constrained_8name"),
        "by_seed": by_seed,
        "interpretation_rule": "constrained_8name is not native generation and cannot close Phase 2A.",
    }
    (OUT / "DECODE_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "DECODE_ITEMS.json").write_text(json.dumps(all_rows) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("greedy", "beam", "constrained_8name")}, indent=2))


if __name__ == "__main__":
    main()
