"""Read-only SF1 readout/selection forensic audit.

This file is deliberately an isolated diagnostic.  It only reads the
authoritative Pilot-1/SF1 checkpoints, SF1 TRAIN records, and the first 128
records of the authorized aligned TinyStories DEV source.  It never creates an
optimizer, enables gradients, writes a checkpoint, or opens any locked panel.

Run ``--select`` first.  That command writes and hashes the deterministic D3
position sample and returns before either checkpoint is loaded.  Run ``--audit``
afterwards to perform the read-only comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

# The project runtime is pinned outside this repository.  Insert its package
# directory before importing torch/tokenizers; no packages are installed or
# changed by this audit.
SITE = r"C:\DaveLM\.venv\Lib\site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from tokenizers import Tokenizer  # type: ignore


ROOT = Path(r"C:\DaveLM-CADAVER")
OUT = ROOT / "sf1_readout_selection_forensic_v1"
BUNDLE = ROOT / "single_fact_acquisition_sf1_seed87011"
RUN = ROOT / "single_fact_acquisition_sf1_seed87011_run"
PARENT = ROOT / "language_pilot_1_early_block_protection_seed8380" / "pilot_run" / "checkpoints" / "seed_8380" / "latest.pt"
CHILD = RUN / "checkpoint_100.pt"
TOKENIZER_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
DEV = BUNDLE / "data" / "ENGLISH_DEV.jsonl"
TRAIN = BUNDLE / "TRAIN.json"
PROTOCOL = BUNDLE / "PROTOCOL.json"
SCOPE = RUN / "PARAMETER_SCOPE.json"
RESTART = RUN / "restart.pt"
MASKING = BUNDLE / "sources" / "PINNED_MASKING.py"
RUNTIME_SOURCE = BUNDLE / "sources" / "hr3_block3_runtime.py"
MODEL_SOURCE = BUNDLE / "sources" / "treatment13_model.py"
CONFIG_SOURCE = BUNDLE / "sources" / "treatment13_config.py"
BINDING_SOURCE = BUNDLE / "sources" / "PINNED_PILOT1_BINDING_IMPLEMENTATION.py"

PARENT_SHA = "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb"
CHILD_SHA = "550ce4306b450c2bdb64e57314e45ec7db5bb2ee68fc0978ed9d3617f4fc1e7e"
TOKENIZER_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
DEV_SHA = "2053a8036d7d0f02ba56ab28f739df95a1743760d0f8c980e112d6667b38104e"

SPECIAL = {0, 1, 2, 3, 4}
NAME_ORDER = ("Alex", "Owen", "Mia", "Nora")
NAME_LOWER = {n.lower() for n in NAME_ORDER}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dev_first_128() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with DEV.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == 128:
                break
    assert len(rows) == 128
    return rows


def names_and_candidates(train_rows: list[dict[str, Any]]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    by_name: dict[str, list[int]] = {}
    by_candidate: dict[str, list[int]] = {}
    for row in train_rows:
        for text, ids in zip(row["candidates"], row["candidate_token_ids"]):
            clean = text.strip().rstrip(".")
            if clean not in by_name:
                by_name[clean] = list(ids)
                by_candidate[text] = list(ids)
            assert by_name[clean] == list(ids), f"inconsistent candidate tokenization for {clean}"
    assert set(NAME_ORDER).issubset(by_name)
    return by_name, by_candidate


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.casefold()).strip()


def select_positions() -> None:
    """Select/hash D3 positions without importing/loading a model."""
    assert sha256(TOKENIZER_PATH) == TOKENIZER_SHA
    assert sha256(DEV) == DEV_SHA
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    train_rows = read_json(TRAIN)
    name_ids, _ = names_and_candidates(train_rows)
    first_name_ids = {ids[0] for ids in name_ids.values()}
    rows = load_dev_first_128()
    candidates: list[dict[str, Any]] = []
    bad = re.compile(r"\?|\banswer\b|\bwho\b|\bthe person who\b|\bfound by\b|\bcarried by\b|\bcopy\b", re.I)
    for row in rows:
        text = row["text"]
        lower = text.casefold()
        if any(n in lower.split() for n in NAME_LOWER) or bad.search(text):
            continue
        ids = list(row["token_ids"])
        for i in range(len(ids) - 1):
            prefix = ids[: i + 1]
            if len(prefix) + 1 > 255:
                continue
            candidates.append({
                "position_id": f"{row['id']}:{i}",
                "row_id": row["id"],
                "row_text": text,
                "token_index": i,
                "prefix_token_ids": prefix,
                "target_token_id": ids[i + 1],
                "prefix_text": tok.decode(prefix),
                "target_text": tok.decode([ids[i + 1]]),
            })
    candidates.sort(key=lambda x: (x["row_id"], x["token_index"]))
    assert len(candidates) >= 256
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    n = len(candidates)
    for k in range(256):
        idx = int(math.floor((k + 0.5) * n / 256.0))
        if idx >= n:
            idx = n - 1
        item = candidates[idx]
        assert item["position_id"] not in seen
        seen.add(item["position_id"])
        selected.append(item)
    # Controls are selected solely from deterministic token frequencies in the
    # same 128 authorized DEV records, with no model outputs involved.
    freq = Counter(t for row in rows for t in row["token_ids"] if t not in SPECIAL and t not in first_name_ids)
    common = sorted(freq, key=lambda t: (-freq[t], t))[:4]
    rare = sorted(freq, key=lambda t: (freq[t], t))[:4]
    controls = []
    for group, ids in (("common", common), ("rare", rare)):
        for token_id in ids:
            controls.append({"group": group, "token_id": int(token_id), "frequency": int(freq[token_id]), "text": tok.decode([int(token_id)])})
    assert len(controls) == 8 and len({x["token_id"] for x in controls}) == 8
    selection = {
        "rule": {
            "source": str(DEV),
            "records": "first 128 JSONL records in file order",
            "filters": "exclude case-insensitive Alex/Owen/Mia/Nora contexts and obvious QA/copy cues",
            "ordering": "row_id then token_index",
            "selection": "index floor((k+0.5)*N/256), k=0..255",
            "controls": "frequency over same records, common descending frequency/id, rare ascending frequency/id, excluding specials and first name tokens",
            "before_model_loading": True,
        },
        "positions": selected,
        "controls": controls,
        "name_tokenizations": {k: v for k, v in sorted(name_ids.items())},
    }
    payload = OUT / "D3_SELECTION.json"
    write_json(payload, selection)
    receipt = {
        "selection_sha256": sha256(payload),
        "position_count": len(selected),
        "control_count": len(controls),
        "source_sha256": DEV_SHA,
        "tokenizer_sha256": TOKENIZER_SHA,
        "selection_seed": "deterministic index rule; no RNG",
        "model_loaded": False,
        "optimizer_created": False,
    }
    write_json(OUT / "D3_SELECTION_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2))


def ranks_from_logits(logits: torch.Tensor, ids: list[int]) -> dict[int, int]:
    # Ties receive the same competition rank; candidate rows are not tied in
    # practice but this is deterministic and does not depend on arg-sort order.
    return {int(i): int((logits > logits[int(i)]).sum().item()) + 1 for i in ids}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None, "p95": None, "p99": None}
    return {
        "n": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p10": percentile(values, 10), "p25": percentile(values, 25), "p50": percentile(values, 50),
        "p75": percentile(values, 75), "p90": percentile(values, 90), "p95": percentile(values, 95), "p99": percentile(values, 99),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


def model_runtime_module():
    # Import only after D3 selection has been materialized.  This module loads
    # architecture code but no checkpoint and performs no stochastic operation.
    source_dir = BUNDLE / "sources"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    import hr3_block3_runtime as rt  # type: ignore
    return rt


def load_base_model(checkpoint: Path, device: torch.device, rt):
    # rt.load_model is the pinned loader.  We score only model.base_model; the
    # specialized binding path is never called by this audit.
    return rt.load_model(checkpoint, device, BUNDLE)


def causal_batch(rows: list[dict[str, Any]], device: torch.device):
    width = max(len(r["token_ids"]) for r in rows) + 2
    x = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    y = torch.full((len(rows), width), -100, dtype=torch.long, device=device)
    lengths: list[int] = []
    for j, row in enumerate(rows):
        z = [2] + list(row["token_ids"]) + [3]
        x[j, : len(z) - 1] = torch.tensor(z[:-1], dtype=torch.long, device=device)
        y[j, : len(z) - 1] = torch.tensor(z[1:], dtype=torch.long, device=device)
        lengths.append(len(z) - 1)
    return x, y, lengths


def score_d3(model, selection: dict[str, Any], tok: Tokenizer, device: torch.device) -> list[dict[str, Any]]:
    ids = [x["token_id"] for x in selection["controls"]]
    name_ids = [int(v[0]) for v in selection["name_tokenizations"].values()]
    all_ids = list(dict.fromkeys(name_ids + ids))
    rows: list[dict[str, Any]] = []
    model.base_model.eval()
    with torch.inference_mode():
        for item in selection["positions"]:
            x = torch.tensor([[2] + item["prefix_token_ids"]], dtype=torch.long, device=device)
            logits = model.base_model(x)[0, -1].float()
            logp = logits.log_softmax(-1)
            p = logp.exp()
            ranks = ranks_from_logits(logits, all_ids + [3, item["target_token_id"]])
            name_measure = {}
            for name, seq in selection["name_tokenizations"].items():
                tid = int(seq[0])
                name_measure[name] = {"token_id": tid, "text": tok.decode([tid]), "logit": float(logits[tid]), "probability": float(p[tid]), "rank": ranks[tid]}
            control_measure = []
            for c in selection["controls"]:
                tid = int(c["token_id"])
                control_measure.append({**c, "logit": float(logits[tid]), "probability": float(p[tid]), "rank": ranks[tid]})
            entropy = float(-(p * logp).sum())
            rows.append({
                **{k: item[k] for k in ("position_id", "row_id", "token_index", "target_token_id", "target_text", "prefix_text")},
                "names": name_measure,
                "controls": control_measure,
                "combined_name_probability": float(sum(p[int(seq[0])] for seq in selection["name_tokenizations"].values())),
                "eos": {"logit": float(logits[3]), "probability": float(p[3]), "rank": ranks[3]},
                "true_next": {"token_id": int(item["target_token_id"]), "probability": float(p[int(item["target_token_id"])]), "rank": ranks[int(item["target_token_id"])]},
                "entropy": entropy,
                "full_vocab_top1": int(torch.argmax(logits).item()),
            })
    return rows


def first_position_info(logits: torch.Tensor, token_id: int) -> dict[str, Any]:
    logits = logits.float()
    logp = logits.log_softmax(-1)
    p = logp.exp()
    return {"token_id": int(token_id), "logit": float(logits[token_id]), "probability": float(p[token_id]), "rank": int((logits > logits[token_id]).sum().item()) + 1}


def score_d4(model, train_rows: list[dict[str, Any]], tok: Tokenizer, device: torch.device) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    model.base_model.eval()
    with torch.inference_mode():
        for row in train_rows:
            prompt_ids = list(row["prompt_token_ids"])
            candidates = [list(x) for x in row["candidate_token_ids"]]
            first_logits = None
            candidate_scores: list[float] = []
            token_scores: list[list[float]] = []
            token_info: list[list[dict[str, Any]]] = []
            for cand in candidates:
                full = [2] + prompt_ids + cand
                logits = model.base_model(torch.tensor([full], dtype=torch.long, device=device))[0].float()
                lp = logits.log_softmax(-1)
                start = len(prompt_ids)
                vals = [float(lp[start + j, tid]) for j, tid in enumerate(cand)]
                candidate_scores.append(sum(vals))
                token_scores.append(vals)
                if first_logits is None:
                    first_logits = logits[start]
                token_info.append([first_position_info(logits[start + j], tid) for j, tid in enumerate(cand)])
            assert first_logits is not None
            # For a prompt of length m, answer-position logits are at the last
            # prompt position (index m in [BOS]+prompt+[EOS]).
            answer_position_logits = model.base_model(torch.tensor([[2] + prompt_ids], dtype=torch.long, device=device))[0, -1].float()
            answer_lp = answer_position_logits.log_softmax(-1)
            ans_p = answer_lp.exp()
            eos_info = first_position_info(answer_position_logits, 3)
            correct = int(row["correct_index"])
            wrong = 1 - correct
            out.append({
                "id": row["id"], "pair_id": row["pair_id"], "family_id": row["family_id"],
                "assignment": row["assignment"], "actor": row["actor"], "object": row["object"], "predicate": row["predicate"],
                "correct_index": correct, "candidates": row["candidates"], "candidate_token_ids": candidates,
                "candidate_scores": candidate_scores, "candidate_token_logprobs": token_scores, "candidate_token_info": token_info,
                "correct_minus_distractor": candidate_scores[correct] - candidate_scores[wrong],
                "correct_candidate_mass": float(ans_p[candidates[correct][0]]),
                "distractor_candidate_mass": float(ans_p[candidates[wrong][0]]),
                "candidate_pair_mass": float(ans_p[candidates[0][0]] + ans_p[candidates[1][0]]),
                "candidate_first_position": [first_position_info(answer_position_logits, c[0]) for c in candidates],
                "candidate_first_full_vocab_top1": int(torch.argmax(answer_position_logits).item()),
                "eos": eos_info,
                "full_vocab_top1_text": tok.decode([int(torch.argmax(answer_position_logits).item())]),
            })
    return out


def score_language(model, dev_rows: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    model.base_model.eval()
    with torch.inference_mode():
        for start in range(0, 128, 32):
            rows = dev_rows[start : start + 32]
            x, y, lengths = causal_batch(rows, device)
            logits = model.base_model(x).float()
            lp = logits.log_softmax(-1)
            for i, row in enumerate(rows):
                z = [2] + list(row["token_ids"]) + [3]
                for pos in range(lengths[i]):
                    target = int(z[pos + 1])
                    ce = float(-lp[i, pos, target])
                    out.append({"row_id": row["id"], "row_text": row["text"], "token_index": pos, "input_token_id": int(z[pos]), "target_token_id": target, "target_text": str(row["text"]) if pos >= len(row["token_ids"]) else "", "ce": ce})
    return out


def map_group(key: str) -> str:
    if key.startswith("base_model.blocks."):
        m = re.match(r"base_model\.blocks\.(\d+)\.", key)
        assert m
        i = int(m.group(1))
        if ".attention." in key:
            return f"block_{i}.attention"
        if ".feed_forward." in key or ".mlp." in key:
            return f"block_{i}.mlp"
        return f"block_{i}.other"
    if key.startswith("base_model.token_embedding"):
        return "embeddings"
    if key.startswith("base_model.position_embedding"):
        return "positional"
    if key.startswith("base_model.final_norm"):
        return "final_norm"
    if key.startswith("base_model.language_head"):
        return "output_head"
    if key.startswith(("localizer.", "wq.", "wk.", "wv.", "wo.")):
        return "localizer_retrieval"
    return "other"


def parameter_deltas(parent_state: dict[str, torch.Tensor], child_state: dict[str, torch.Tensor]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert set(parent_state) == set(child_state)
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in sorted(parent_state):
        p = parent_state[key].detach().cpu().float()
        c = child_state[key].detach().cpu().float()
        assert p.shape == c.shape, key
        d = c - p
        pn = float(torch.linalg.vector_norm(p))
        cn = float(torch.linalg.vector_norm(c))
        dn = float(torch.linalg.vector_norm(d))
        dot = float((p.reshape(-1) * c.reshape(-1)).sum())
        denom = pn * cn
        row = {"name": key, "tensor_kind": "buffer" if key.endswith(".mask") else "parameter", "group": map_group(key), "numel": int(p.numel()), "delta_l2": dn, "relative_delta": dn / max(pn, 1e-12), "rms_delta": float(torch.sqrt(torch.mean(d * d))), "max_abs_delta": float(torch.max(torch.abs(d))), "parent_norm": pn, "child_norm": cn, "cosine": dot / denom if denom else None}
        rows.append(row)
        grouped[row["group"]].append(row)
    aggregates: dict[str, Any] = {}
    for group, vals in sorted(grouped.items()):
        pn2 = sum(v["parent_norm"] ** 2 for v in vals)
        cn2 = sum(v["child_norm"] ** 2 for v in vals)
        dn2 = sum(v["delta_l2"] ** 2 for v in vals)
        dot = sum(v["cosine"] * v["parent_norm"] * v["child_norm"] for v in vals if v["cosine"] is not None)
        aggregates[group] = {"tensor_count": len(vals), "numel": sum(v["numel"] for v in vals), "delta_l2": math.sqrt(dn2), "relative_delta": math.sqrt(dn2) / max(math.sqrt(pn2), 1e-12), "rms_delta": math.sqrt(sum(v["rms_delta"] ** 2 * v["numel"] for v in vals) / max(sum(v["numel"] for v in vals), 1)), "max_abs_delta": max(v["max_abs_delta"] for v in vals), "parent_norm": math.sqrt(pn2), "child_norm": math.sqrt(cn2), "cosine": dot / max(math.sqrt(pn2 * cn2), 1e-12), "zero_drift_tensors": sum(v["delta_l2"] == 0.0 for v in vals)}
    return rows, aggregates


def output_head_audit(parent_state: dict[str, torch.Tensor], child_state: dict[str, torch.Tensor], name_ids: dict[str, list[int]], tok: Tokenizer, dev_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pk = "base_model.language_head.weight"
    assert pk in parent_state and child_state[pk].ndim == 2
    pw = parent_state[pk].cpu().float(); cw = child_state[pk].cpu().float(); dw = cw - pw
    norms = torch.linalg.vector_norm(dw, dim=1).tolist()
    freq = Counter(t for row in dev_rows for t in row["token_ids"] if t not in SPECIAL)
    candidates = []
    for name, ids in name_ids.items():
        for pos, tid in enumerate(ids):
            candidates.append({"label": name, "position": pos, "token_id": int(tid)})
    common = sorted([t for t in freq if t not in {int(x[0]) for x in name_ids.values()}], key=lambda t: (-freq[t], t))[:4]
    rare = sorted([t for t in freq if t not in {int(x[0]) for x in name_ids.values()}], key=lambda t: (freq[t], t))[:4]
    for group, ids in (("common_dev", common), ("rare_dev", rare)):
        for tid in ids:
            candidates.append({"label": group, "position": None, "token_id": int(tid), "frequency": int(freq[tid])})
    def row_info(x: dict[str, Any]) -> dict[str, Any]:
        tid = int(x["token_id"]); p = pw[tid]; c = cw[tid]; d = dw[tid]; pn = float(torch.linalg.vector_norm(p)); cn = float(torch.linalg.vector_norm(c)); dn = float(torch.linalg.vector_norm(d)); dot = float((p*c).sum())
        return {**x, "text": tok.decode([tid]), "delta_l2": dn, "relative_delta": dn / max(pn, 1e-12), "parent_norm": pn, "child_norm": cn, "cosine": dot / max(pn*cn, 1e-12), "percentile_delta_l2": 100.0 * sum(v <= dn for v in norms) / len(norms), "percentile_relative_delta": 100.0 * sum((float(torch.linalg.vector_norm(dw[i])) / max(float(torch.linalg.vector_norm(pw[i])), 1e-12)) <= (dn / max(pn, 1e-12)) for i in range(pw.shape[0])) / pw.shape[0]}
    return {"shape": list(pw.shape), "all_row_delta_l2_summary": summarize([float(x) for x in norms]), "all_row_delta_l2_percentiles": {f"p{q}": percentile([float(x) for x in norms], q) for q in (0,1,5,10,25,50,75,90,95,99,100)}, "rows": [row_info(x) for x in candidates]}


def architecture_report(rt) -> dict[str, Any]:
    m = rt.Treatment13Model()
    params = dict(m.named_parameters())
    base = m.base_model
    emb = params["base_model.token_embedding.weight"].numel()
    pos = params["base_model.position_embedding.weight"].numel()
    head = params["base_model.language_head.weight"].numel() + params["base_model.language_head.bias"].numel()
    total = sum(p.numel() for p in m.parameters())
    return {"model_parameter_count": total, "base_model_parameter_count": sum(p.numel() for p in base.parameters()), "block_count": len(base.blocks), "embedding_shape": list(base.token_embedding.weight.shape), "position_shape": list(base.position_embedding.weight.shape), "language_head_shape": list(base.language_head.weight.shape), "vocab_size": int(base.language_head.weight.shape[0]), "embedding_parameter_count": emb, "positional_parameter_count": pos, "output_head_parameter_count": head, "retrieval_dim": 64, "tied_embedding_output": bool(base.language_head.weight.data_ptr() == base.token_embedding.weight.data_ptr()), "model_class": type(m).__name__, "base_class": type(base).__name__}


def score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(field: str, subset: list[dict[str, Any]] | None = None):
        xs = [float(r[field]) for r in (subset if subset is not None else rows)]
        return statistics.fmean(xs) if xs else None
    return {"n": len(rows), "mean_correct_minus_distractor": mean("correct_minus_distractor"), "positive_margin": sum(r["correct_minus_distractor"] > 0 for r in rows), "zero_margin": sum(r["correct_minus_distractor"] == 0 for r in rows), "negative_margin": sum(r["correct_minus_distractor"] < 0 for r in rows), "mean_candidate_pair_mass": mean("candidate_pair_mass"), "mean_correct_candidate_mass": mean("correct_candidate_mass"), "mean_distractor_candidate_mass": mean("distractor_candidate_mass"), "full_vocab_top1_correct": sum(r["candidate_first_full_vocab_top1"] == r["candidate_token_ids"][r["correct_index"]][0] for r in rows), "mean_eos_probability": mean("eos_probability") if rows and "eos_probability" in rows[0] else statistics.fmean(float(r["eos"]["probability"]) for r in rows), "by_actor_pair": {pair: {"n": sum(set(["Alex", "Owen"]) == set([r["candidates"][0].strip().rstrip('.'), r["candidates"][1].strip().rstrip('.')]) and r["actor"] in set(pair.split('/')) for r in rows)} for pair in ("Alex/Owen", "Mia/Nora")}}


def aggregate_d3(rows: list[dict[str, Any]], selection: dict[str, Any]) -> dict[str, Any]:
    names = list(selection["name_tokenizations"])
    controls = [c["token_id"] for c in selection["controls"]]
    result: dict[str, Any] = {"n": len(rows), "name_deltas": {}, "control_deltas": {}, "combined_name_probability": summarize([r["combined_name_probability"] for r in rows]), "eos_probability": summarize([r["eos"]["probability"] for r in rows]), "entropy": summarize([r["entropy"] for r in rows]), "true_next_probability": summarize([r["true_next"]["probability"] for r in rows])}
    for name in names:
        result["name_deltas"][name] = {"logit": summarize([r["names"][name]["logit"] for r in rows]), "probability": summarize([r["names"][name]["probability"] for r in rows]), "rank": summarize([float(r["names"][name]["rank"]) for r in rows])}
    for tid in controls:
        vals = [r["controls"][next(i for i, c in enumerate(r["controls"]) if c["token_id"] == tid)]["logit"] for r in rows]
        probs = [r["controls"][next(i for i, c in enumerate(r["controls"]) if c["token_id"] == tid)]["probability"] for r in rows]
        result["control_deltas"][str(tid)] = {"logit": summarize(vals), "probability": summarize(probs)}
    return result


def d4_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {"all": rows, "Mia/Nora": [r for r in rows if set(x.strip().rstrip('.') for x in r["candidates"]) == {"Mia", "Nora"}], "Alex/Owen": [r for r in rows if set(x.strip().rstrip('.') for x in r["candidates"]) == {"Alex", "Owen"}], "Alex-correct": [r for r in rows if r["actor"] == "Alex"], "Owen-correct": [r for r in rows if r["actor"] == "Owen"]}
    result = {}
    for g, xs in groups.items():
        result[g] = {"n": len(xs), "margin": summarize([r["correct_minus_distractor"] for r in xs]), "positive_margin": sum(r["correct_minus_distractor"] > 0 for r in xs), "correct_first_token_logit": statistics.fmean(r["candidate_first_position"][r["correct_index"]]["logit"] for r in xs) if xs else None, "correct_first_token_probability": statistics.fmean(r["candidate_first_position"][r["correct_index"]]["probability"] for r in xs) if xs else None, "candidate_pair_mass": statistics.fmean(r["candidate_pair_mass"] for r in xs) if xs else None, "full_vocab_top1_correct": sum(r["candidate_first_full_vocab_top1"] == r["candidate_token_ids"][r["correct_index"]][0] for r in xs)}
    return result


def d4_compare(parent_rows: list[dict[str, Any]], child_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pm = {r["id"]: r for r in parent_rows}; cm = {r["id"]: r for r in child_rows}
    pairs = []
    for rid in sorted(pm):
        p, c = pm[rid], cm[rid]
        pairs.append({"id": rid, "pair_id": p["pair_id"], "actor": p["actor"], "parent_correct_minus_distractor": p["correct_minus_distractor"], "child_correct_minus_distractor": c["correct_minus_distractor"], "margin_delta": c["correct_minus_distractor"] - p["correct_minus_distractor"], "parent_correct_first_logit": p["candidate_first_position"][p["correct_index"]]["logit"], "child_correct_first_logit": c["candidate_first_position"][c["correct_index"]]["logit"], "correct_first_logit_delta": c["candidate_first_position"][c["correct_index"]]["logit"] - p["candidate_first_position"][p["correct_index"]]["logit"], "parent_correct": p["correct_minus_distractor"] > 0, "child_correct": c["correct_minus_distractor"] > 0, "child_full_vocab_top1_correct": c["candidate_first_full_vocab_top1"] == c["candidate_token_ids"][c["correct_index"]][0]})
    ow = [x for x in pairs if x["actor"] == "Owen"]
    mean_delta = statistics.fmean(x["correct_first_logit_delta"] for x in ow) if ow else 0.0
    mean_prob_delta = statistics.fmean(cm[x["id"]]["candidate_first_position"][cm[x["id"]]["correct_index"]]["probability"] - pm[x["id"]]["candidate_first_position"][pm[x["id"]]["correct_index"]]["probability"] for x in ow) if ow else 0.0
    failures = sum(not x["child_full_vocab_top1_correct"] for x in ow)
    if mean_delta > 0 and mean_prob_delta > 0 and failures >= math.ceil(len(ow) / 2):
        cls = "STAGE_B_SIGNAL_INCREASED_BUT_SUPPRESSED"
    elif mean_delta <= 0 and mean_prob_delta <= 0:
        cls = "NO_CLEAR_STAGE_B_SIGNAL"
    else:
        cls = "MIXED/AMBIGUOUS"
    return {"rows": pairs, "Owen-correct": {"n": len(ow), "mean_correct_first_logit_delta": mean_delta, "mean_correct_first_probability_delta": mean_prob_delta, "full_vocab_top1_failures": failures}, "classification_rule": "increased-and-suppressed iff mean Owen-correct first-token logit and probability deltas are both >0 while at least half remain non-top1; no-clear iff both means <=0; otherwise mixed", "classification": cls}


def d3_delta(parent: list[dict[str, Any]], child: list[dict[str, Any]], selection: dict[str, Any]) -> dict[str, Any]:
    pm = {r["position_id"]: r for r in parent}; cm = {r["position_id"]: r for r in child}
    names = list(selection["name_tokenizations"])
    result: dict[str, Any] = {"n": len(parent), "name": {}, "combined_name_probability_delta": summarize([cm[k]["combined_name_probability"] - pm[k]["combined_name_probability"] for k in pm])}
    for name in names:
        result["name"][name] = {"logit_delta": summarize([cm[k]["names"][name]["logit"] - pm[k]["names"][name]["logit"] for k in pm]), "probability_delta": summarize([cm[k]["names"][name]["probability"] - pm[k]["names"][name]["probability"] for k in pm]), "rank_delta": summarize([float(cm[k]["names"][name]["rank"] - pm[k]["names"][name]["rank"]) for k in pm])}
    for group in ("common", "rare"):
        tids = [c["token_id"] for c in selection["controls"] if c["group"] == group]
        result[group] = {str(tid): {"logit_delta": summarize([cm[k]["controls"][next(i for i, c in enumerate(cm[k]["controls"]) if c["token_id"] == tid)]["logit"] - pm[k]["controls"][next(i for i, c in enumerate(pm[k]["controls"]) if c["token_id"] == tid)]["logit"] for k in pm]), "probability_delta": summarize([cm[k]["controls"][next(i for i, c in enumerate(cm[k]["controls"]) if c["token_id"] == tid)]["probability"] - pm[k]["controls"][next(i for i, c in enumerate(pm[k]["controls"]) if c["token_id"] == tid)]["probability"] for k in pm])} for tid in tids}
    return result


def language_compare(parent_rows: list[dict[str, Any]], child_rows: list[dict[str, Any]], d3p: list[dict[str, Any]], d3c: list[dict[str, Any]]) -> dict[str, Any]:
    pm = {(r["row_id"], r["token_index"]): r for r in parent_rows}; cm = {(r["row_id"], r["token_index"]): r for r in child_rows}
    rows = []
    for key in sorted(pm):
        p, c = pm[key], cm[key]
        rows.append({**p, "child_ce": c["ce"], "delta_ce": c["ce"] - p["ce"]})
    d3_keys = {(r["row_id"], r["token_index"]): r for r in d3p}
    d3c_map = {(r["row_id"], r["token_index"]): r for r in d3c}
    paired = []
    for key, pr in d3_keys.items():
        if key in d3c_map and key in pm:
            paired.append({"key": list(key), "ce_delta": cm[key]["ce"] - pm[key]["ce"], "name_mass_delta": d3c_map[key]["combined_name_probability"] - pr["combined_name_probability"]})
    xs = [float(r["delta_ce"]) for r in rows]
    corr = None
    if len(paired) > 1:
        a = [float(r["ce_delta"]) for r in paired]; b = [float(r["name_mass_delta"]) for r in paired]
        ma, mb = statistics.fmean(a), statistics.fmean(b); da = math.sqrt(sum((v-ma)**2 for v in a)); db = math.sqrt(sum((v-mb)**2 for v in b)); corr = sum((x-ma)*(y-mb) for x,y in zip(a,b)) / (da*db) if da and db else None
    return {"n_tokens": len(rows), "delta_ce": summarize(xs), "fraction_improved": sum(x < 0 for x in xs)/len(xs), "fraction_worsened": sum(x > 0 for x in xs)/len(xs), "fraction_equal": sum(x == 0 for x in xs)/len(xs), "largest_increases": sorted(rows, key=lambda r: r["delta_ce"], reverse=True)[:20], "largest_improvements": sorted(rows, key=lambda r: r["delta_ce"])[:20], "d3_name_mass_ce_delta_correlation": corr, "d3_join_count": len(paired), "d3_joined": paired}


def collect_provenance() -> dict[str, Any]:
    paths = [PARENT, CHILD, TOKENIZER_PATH, DEV, TRAIN, PROTOCOL, SCOPE, RESTART, MASKING, RUNTIME_SOURCE, MODEL_SOURCE, CONFIG_SOURCE, BINDING_SOURCE, BUNDLE / "SCHEDULE.json", BUNDLE / "PREFLIGHT.json", RUN / "OUTPUT_RECEIPT.json"]
    return {str(p): {"exists": p.exists(), "sha256": sha256(p) if p.exists() else None, "size": p.stat().st_size if p.exists() else None} for p in paths}


def make_report(results: dict[str, Any]) -> str:
    arch = results["architecture"]
    lines = ["# SF1 readout-selection forensic audit v1", "", "This is a read-only audit of Pilot1 versus the preserved SF1 update100 checkpoint. No optimizer, gradient, weight mutation, locked panel, readiness FINAL, or sacred material was used.", "", "## ESTABLISHED", "", f"- Pilot1 and SF1 update100 hashes verified: `{results['hashes']['parent']}`, `{results['hashes']['child']}`; tokenizer `{results['hashes']['tokenizer']}`.", f"- Model has {arch['model_parameter_count']:,} parameters, {arch['block_count']} blocks, vocabulary {arch['vocab_size']}, and an untied output head (`tied_embedding_output={arch['tied_embedding_output']}`).", "- The SF1 protocol froze blocks 0–3 and specialized localization/retrieval only during English updates; binding updates restored the full T13 scope. The endpoint delta therefore cannot isolate English-only drift, and nonzero cumulative drift in those groups is not itself a freeze violation.", "- D3 used 256 deterministic positions from the first 128 authorized aligned DEV records; selection was written and hashed before checkpoint loading.", "", "### Parameter delta census", ""]
    for g, a in results["parameter_aggregates"].items():
        lines.append(f"- `{g}`: n={a['numel']:,}, relative delta={a['relative_delta']:.6g}, delta L2={a['delta_l2']:.6g}, RMS={a['rms_delta']:.6g}, max abs={a['max_abs_delta']:.6g}, zero-drift tensors={a['zero_drift_tensors']}/{a['tensor_count']}.")
    lines += ["", "### D3 global-versus-contextual prior audit", "", "The complete raw D3 table is in `D3_PARENT.jsonl`, `D3_SF1.jsonl`; aggregate summaries and parent→child deltas are in `D3_SUMMARY.json`. Name/control changes are descriptive logits on unrelated DEV contexts, not a causal claim about context use.", "", "### D4 SF1 training-item readout", "", "The 16 allowed SF1 TRAIN records were scored through `base_model` only. Raw records are in `D4_PARENT.jsonl` and `D4_SF1.jsonl`; grouped summaries and Owen-correct signal classification are in `D4_SUMMARY.json`.", "", "### Language-loss decomposition", "", "The aligned objective was recomputed at the same token positions for both checkpoints. Raw per-token values are in `LANGUAGE_LOSS.jsonl`; summary/quantiles and largest changes are in `LANGUAGE_LOSS_SUMMARY.json`.", "", "## SUPPORTED", "", f"- D3 aggregate deltas: `{results['d3_delta']['name']}`; interpret alongside common/rare controls in `D3_SUMMARY.json` rather than calling the effect global by default.", f"- D4 descriptive classification: `{results['d4_compare']['classification']}` under the predeclared rule recorded in `D4_SUMMARY.json`.", f"- Language regression profile: mean delta CE {results['language']['delta_ce']['mean']:.6g}, median {results['language']['delta_ce']['median']:.6g}, improved {results['language']['fraction_improved']:.3f}, worsened {results['language']['fraction_worsened']:.3f}; correlation with D3 name-mass delta was {results['language']['d3_name_mass_ce_delta_correlation']}.", "", "## NOT ESTABLISHED", "", "- These measurements do not establish relational generalization, held-out transfer, general English competence, conversational ability, architectural impossibility, or a causal explanation for SF1 behavior.", "- D4 training-item logits cannot establish genuine relational generalization; they only describe evidence on the 16 training records.", "- Endpoint parameter deltas cannot determine which update type caused each change, because the preserved endpoint includes binding updates after English updates.", "- Exact-string and token-level comparisons here do not prove absence of semantic or near-duplicate data contamination beyond the authorized sources; no locked SF1 transfer/copy panels were opened.", "", "## Provenance and safeguards", "", "- Authorized inputs, source hashes, and output hashes are in `PROVENANCE.json`, `OUTPUT_SHA256SUMS.txt`, and `RECEIPT.json`.", "- The sealed readiness FINAL and all sacred material remained untouched.", "", "SF1_READOUT_SELECTION_FORENSIC_COMPLETE"]
    return "\n".join(lines) + "\n"


def audit() -> None:
    # Verify selection receipt before any model loading.
    sel_path = OUT / "D3_SELECTION.json"; rec_path = OUT / "D3_SELECTION_RECEIPT.json"
    assert sel_path.is_file() and rec_path.is_file(), "run --select first"
    selection = read_json(sel_path); selrec = read_json(rec_path)
    assert sha256(sel_path) == selrec["selection_sha256"] and selrec["model_loaded"] is False
    assert sha256(PARENT) == PARENT_SHA and sha256(CHILD) == CHILD_SHA and sha256(TOKENIZER_PATH) == TOKENIZER_SHA and sha256(DEV) == DEV_SHA
    assert OUT != BUNDLE and not (OUT / "FINAL").exists()
    train_rows = read_json(TRAIN); assert len(train_rows) == 16
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    name_ids, _ = names_and_candidates(train_rows)
    for name, ids in name_ids.items():
        assert len(ids) == 4
        assert tok.encode(" " + name + ".").ids == ids
    dev_rows = load_dev_first_128()
    # Architecture and state census happen before inference and never mutate state.
    parent_raw = torch.load(PARENT, map_location="cpu", weights_only=False); child_raw = torch.load(CHILD, map_location="cpu", weights_only=False)
    parent_state = parent_raw["model_state_dict"]; child_state = child_raw["model_state_dict"]
    drows, pagg = parameter_deltas(parent_state, child_state)
    write_jsonl(OUT / "PARAMETER_DELTAS.jsonl", drows)
    write_json(OUT / "PARAMETER_AGGREGATES.json", pagg)
    arch_rt = model_runtime_module(); arch = architecture_report(arch_rt)
    write_json(OUT / "ARCHITECTURE.json", arch)
    write_json(OUT / "OUTPUT_HEAD_ROW_AUDIT.json", output_head_audit(parent_state, child_state, name_ids, tok, dev_rows))
    # Read-only optimizer/scope metadata, not optimizer construction.
    restart = torch.load(RESTART, map_location="cpu", weights_only=False)
    opt = restart["optimizer"]; opt_groups = [{k: jsonable(v) for k, v in g.items() if k != "params"} | {"param_count_ids": len(g.get("params", []))} for g in opt["param_groups"]]
    write_json(OUT / "OPTIMIZER_METADATA.json", {"param_groups": opt_groups, "state_entries": len(opt["state"]), "completed_update": restart["completed"], "scope_at_checkpoint": restart["scope"], "provenance": restart["provenance"]})
    write_json(OUT / "PARAMETER_SCOPE.json", read_json(SCOPE))
    # Load each model only after selection is sealed.  Ordinary base_model path
    # is used; no specialized forward, optimizer, or gradient is invoked.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "pinned SF1 audit expects the authorized GPU runtime"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    rt = arch_rt
    pm = load_base_model(PARENT, device, rt)
    d3p = score_d3(pm, selection, tok, device); d4p = score_d4(pm, train_rows, tok, device); langp = score_language(pm, dev_rows, device)
    del pm; torch.cuda.empty_cache()
    cm = load_base_model(CHILD, device, rt)
    d3c = score_d3(cm, selection, tok, device); d4c = score_d4(cm, train_rows, tok, device); langc = score_language(cm, dev_rows, device)
    del cm; torch.cuda.empty_cache()
    write_jsonl(OUT / "D3_PARENT.jsonl", d3p); write_jsonl(OUT / "D3_SF1.jsonl", d3c)
    write_json(OUT / "D3_SUMMARY.json", {"parent": aggregate_d3(d3p, selection), "sf1": aggregate_d3(d3c, selection), "delta": d3_delta(d3p, d3c, selection)})
    write_jsonl(OUT / "D4_PARENT.jsonl", d4p); write_jsonl(OUT / "D4_SF1.jsonl", d4c)
    d4cmp = d4_compare(d4p, d4c); write_json(OUT / "D4_SUMMARY.json", {"parent": d4_summary(d4p), "sf1": d4_summary(d4c), "comparison": d4cmp})
    lang = language_compare(langp, langc, d3p, d3c); write_jsonl(OUT / "LANGUAGE_LOSS.jsonl", lang["d3_joined"] if False else [{**r, "child_ce": r["child_ce"], "delta_ce": r["delta_ce"]} for r in [{**p, "child_ce": c["ce"], "delta_ce": c["ce"]-p["ce"]} for p,c in zip(langp, langc)]])
    write_json(OUT / "LANGUAGE_LOSS_SUMMARY.json", lang)
    prov = collect_provenance(); write_json(OUT / "PROVENANCE.json", {"inputs": prov, "authorized_scope": "Pilot1, SF1 checkpoint100, SF1 TRAIN.json, first128 ENGLISH_DEV.jsonl only; no locked panels", "runtime": {"python": sys.version, "torch": torch.__version__, "tokenizers": __import__('tokenizers').__version__, "device": str(device), "cuda_available": bool(torch.cuda.is_available())}, "forbidden_access": ["SF1 HELDOUT", "SF1 ALTERNATE", "SF1 COPY", "SF1 COMPETING", "readiness FINAL", "sacred graduation material"]})
    # Summary used for report is kept compact and separate from raw files.
    result = {"hashes": {"parent": sha256(PARENT), "child": sha256(CHILD), "tokenizer": sha256(TOKENIZER_PATH), "dev": sha256(DEV), "selection": sha256(sel_path)}, "architecture": arch, "parameter_aggregates": pagg, "d3_delta": d3_delta(d3p, d3c, selection), "d4_compare": d4cmp, "language": lang}
    write_json(OUT / "RESULT_SUMMARY.json", result)
    (OUT / "REPORT.md").write_text(make_report(result), encoding="utf-8", newline="\n")
    # The output manifest is deliberately non-circular: it excludes itself and
    # the receipt, then records their hashes separately in RECEIPT.json.
    files = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in {"OUTPUT_SHA256SUMS.txt", "RECEIPT.json", "RECEIPT.sha256", "FINAL_HASHES.txt"}:
            files.append((sha256(p), str(p.relative_to(OUT)).replace("\\", "/")))
    (OUT / "OUTPUT_SHA256SUMS.txt").write_text("\n".join(f"{h}  {rel}" for h, rel in files) + "\n", encoding="utf-8", newline="\n")
    receipt = {"status": "SF1_READOUT_SELECTION_FORENSIC_COMPLETE", "payload_manifest_sha256": sha256(OUT / "OUTPUT_SHA256SUMS.txt"), "excluded_from_manifest": ["OUTPUT_SHA256SUMS.txt", "RECEIPT.json"], "checkpoint_hashes": {"parent": PARENT_SHA, "sf1_update100": CHILD_SHA}, "selection_sha256": sha256(sel_path), "no_optimizer": True, "no_weight_mutation": True, "no_locked_panel": True}
    write_json(OUT / "RECEIPT.json", receipt)
    (OUT / "RECEIPT.sha256").write_text(sha256(OUT / "RECEIPT.json") + "  RECEIPT.json\n", encoding="utf-8", newline="\n")
    # Add the manifest/receipt hashes to a separate final checksum line so all
    # created artifacts are auditable without introducing a circular manifest.
    (OUT / "FINAL_HASHES.txt").write_text(json.dumps({"manifest_sha256": sha256(OUT / "OUTPUT_SHA256SUMS.txt"), "receipt_sha256": sha256(OUT / "RECEIPT.json"), "detached_receipt_file_sha256": sha256(OUT / "RECEIPT.sha256")}, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    # Final manifest is recomputed to include FINAL_HASHES but still excludes
    # the manifest/receipt pair.  The receipt's manifest hash now matches the
    # physically final manifest.
    files = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name not in {"OUTPUT_SHA256SUMS.txt", "RECEIPT.json", "RECEIPT.sha256", "FINAL_HASHES.txt"}:
            files.append((sha256(p), str(p.relative_to(OUT)).replace("\\", "/")))
    (OUT / "OUTPUT_SHA256SUMS.txt").write_text("\n".join(f"{h}  {rel}" for h, rel in files) + "\n", encoding="utf-8", newline="\n")
    receipt["payload_manifest_sha256"] = sha256(OUT / "OUTPUT_SHA256SUMS.txt")
    write_json(OUT / "RECEIPT.json", receipt)
    (OUT / "RECEIPT.sha256").write_text(sha256(OUT / "RECEIPT.json") + "  RECEIPT.json\n", encoding="utf-8", newline="\n")
    # Final hash file is outside the payload manifest by design; refresh it.
    (OUT / "FINAL_HASHES.txt").write_text(json.dumps({"manifest_sha256": sha256(OUT / "OUTPUT_SHA256SUMS.txt"), "receipt_sha256": sha256(OUT / "RECEIPT.json"), "detached_receipt_file_sha256": sha256(OUT / "RECEIPT.sha256")}, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"status": receipt["status"], "output": str(OUT), "manifest_sha256": receipt["payload_manifest_sha256"], "receipt_sha256": sha256(OUT / "RECEIPT.json")}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.select == args.audit:
        ap.error("choose exactly one of --select or --audit")
    if args.select:
        select_positions()
    else:
        audit()


if __name__ == "__main__":
    main()
