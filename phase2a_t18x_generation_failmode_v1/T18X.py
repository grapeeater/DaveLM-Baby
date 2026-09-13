"""Read-only T18 generation fail-mode vs T17X continuation baseline.

Does not train. Does not load T3 TEST. Does not write T18 runners/checkpoints.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(r"C:\DaveLM-CADAVER")
OUT = ROOT / "phase2a_t18x_generation_failmode_v1"
T18 = ROOT / "phase2a_t18_t17recipe_answerce_lr3p75e5_1k_v1"
T3_DATA = ROOT / "phase2a_t3_rebuilt_study_v1" / "data"
sys.path.insert(0, str(T18))
sys.path.insert(0, str(ROOT / "baby_vnext_60m_design_v1"))

from T18_RUNNER import (  # noqa: E402
    ARCH, CTX, EOS, TOK_PATH, TOK_SHA, load_jsonl, sha256,
)

SEEDS = (770001, 770002, 770003)
SCHEMA = "baby_vnext_phase2a_t18_t17recipe_answerce_pointer_lr3p75e5_1k_model_v1"
SAMPLE_N = 12


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def confusion(rows: list) -> dict:
    c = Counter()
    pn_e_templates = Counter()
    for r in rows:
        key = ("P" if r["pointer_ok"] else "p") + ("N" if r["native_ok"] else "n") + ("E" if r["exact_ok"] else "e")
        c[key] += 1
        if r["pointer_ok"] and r["native_ok"] and not r["exact_ok"]:
            pn_e_templates[r.get("template", "?")] += 1
    return {
        "cells": dict(c),
        "n": len(rows),
        "exact_ok": sum(1 for r in rows if r["exact_ok"]),
        "pointer_and_native_not_exact": sum(1 for r in rows if r["pointer_ok"] and r["native_ok"] and not r["exact_ok"]),
        "pn_not_exact_templates": dict(pn_e_templates),
    }


def classify_gen(new, target, wrongs):
    if new == target:
        return "exact"
    if not new:
        return "empty"
    if new == [EOS] or new[0] == EOS:
        return "immediate_eos"
    if new == target[:-1]:
        return "missing_eos"
    if new and new[0] == target[0]:
        if len(new) < len(target) and target[:len(new)] == new:
            return "correct_prefix"
        if len(new) > len(target) and new[:len(target)] == target:
            return "target_then_extra"
        return "first_token_correct_then_diverge"
    for w in wrongs:
        if new == w:
            return "wrong_candidate"
        if new and new[0] == w[0]:
            return "wrong_candidate_first_token"
    if EOS in new:
        return "other_then_eos"
    return "unterminated_other"


def load_ckpt(path: Path, device):
    from baby_vnext.config import BabyVNextConfig
    from baby_vnext.binding import BabyVNextWithBinding
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise RuntimeError(f"unexpected schema {payload.get('schema')}")
    model = BabyVNextWithBinding(BabyVNextConfig.load(ARCH / "BABY_VNEXT_CONFIG.json"))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def greedy(model, row, device):
    prompt = [2] + [int(x) for x in row["prompt_token_ids"]]
    target = [int(t) for t in row["candidate_token_ids"][int(row["correct_index"])]] + [EOS]
    wrongs = []
    ci = int(row["correct_index"])
    for i, cand in enumerate(row["candidate_token_ids"]):
        if i != ci:
            wrongs.append([int(t) for t in cand] + [EOS])
    gen = list(prompt)
    new = []
    for _ in range(max(len(target) + 4, 16)):
        inp = torch.tensor([gen[-CTX:]], device=device)
        logits, _ = model(inp)
        t = int(logits[0, -1].argmax())
        gen.append(t)
        new.append(t)
        if t == EOS:
            break
    return new, target, wrongs


def main():
    seal = json.loads((T3_DATA / "TEST_SEAL.json").read_text(encoding="utf-8"))
    if seal["status"] != "SEALED_UNOPENED":
        raise RuntimeError("TEST seal unexpected")
    if sha256(T3_DATA / "qa_test.jsonl") != seal["sha256"]:
        raise RuntimeError("TEST hash mismatch")
    if sha256(TOK_PATH) != TOK_SHA:
        raise RuntimeError("tokenizer hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")

    tok = Tokenizer.from_file(str(TOK_PATH))
    dev = load_jsonl(T3_DATA / "qa_dev.jsonl")
    by_id = {r["id"]: r for r in dev}
    device = torch.device("cuda")
    report = {"test_loaded": False, "test_seal": seal["status"], "seeds": {}}

    for seed in SEEDS:
        run = ROOT / f"phase2a_t18_t17recipe_answerce_lr3p75e5_1k_v1_run_seed{seed}"
        items = json.loads((run / "item_results_1000.json").read_text(encoding="utf-8"))["dev"]
        conf = confusion(items)
        model = load_ckpt(run / "checkpoints" / "checkpoint_1000.pt", device)
        modes = Counter()
        first_is_target = 0
        first_is_wrong = 0
        first_other = 0
        hits, misses = [], []
        for item in items:
            src = by_id[item["id"]]
            new, target, wrongs = greedy(model, src, device)
            mode = classify_gen(new, target, wrongs)
            modes[mode] += 1
            if new and new[0] == target[0]:
                first_is_target += 1
            elif new and any(new[0] == w[0] for w in wrongs):
                first_is_wrong += 1
            else:
                first_other += 1
            rec = {
                "id": item["id"],
                "template": src["template"],
                "correct_name": src["correct_name"],
                "pointer_ok": item["pointer_ok"],
                "native_ok": item["native_ok"],
                "exact_ok": item["exact_ok"],
                "mode": mode,
                "first_is_target_bos": bool(new and new[0] == target[0]),
                "target_txt": tok.decode(target),
                "gen_txt": tok.decode(new),
            }
            if item["exact_ok"] and len(hits) < SAMPLE_N:
                hits.append(rec)
            if (not item["exact_ok"]) and item["pointer_ok"] and item["native_ok"] and len(misses) < SAMPLE_N:
                misses.append(rec)
        del model
        torch.cuda.empty_cache()
        report["seeds"][str(seed)] = {
            "confusion_u1000": conf,
            "generation_modes": dict(modes),
            "first_token": {
                "target_bos": first_is_target,
                "wrong_candidate_bos": first_is_wrong,
                "other": first_other,
            },
            "exact_hit_samples": hits,
            "pne_miss_samples": misses,
        }
        print(seed, conf["cells"], dict(modes),
              "first", first_is_target, first_is_wrong, first_other, flush=True)

    write_json(OUT / "FAILMODES.json", report)
    write_json(OUT / "STATUS.json", {"status": "COMPLETE", "test_loaded": False})
    print("WROTE", OUT / "FAILMODES.json", flush=True)


if __name__ == "__main__":
    main()
