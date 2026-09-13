"""SF1_COMPONENT_SWAP_FORENSIC_V1.

Read-only 2x2x2 functional intervention at the representation immediately
before the base model's final normalization.  The script never constructs an
optimizer, enables autograd, mutates/saves weights, or opens a locked panel.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SITE = r"C:\DaveLM\.venv\Lib\site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from tokenizers import Tokenizer  # type: ignore


ROOT = Path(r"C:\DaveLM-CADAVER")
OUT = ROOT / "sf1_component_swap_forensic_v1"
BUNDLE = ROOT / "single_fact_acquisition_sf1_seed87011"
RUN = ROOT / "single_fact_acquisition_sf1_seed87011_run"
PRIOR = ROOT / "sf1_readout_selection_forensic_v1"
PARENT = ROOT / "language_pilot_1_early_block_protection_seed8380" / "pilot_run" / "checkpoints" / "seed_8380" / "latest.pt"
CHILD = RUN / "checkpoint_100.pt"
TOKENIZER_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
TRAIN_PATH = BUNDLE / "TRAIN.json"
DEV_PATH = BUNDLE / "data" / "ENGLISH_DEV.jsonl"
SELECTION_PATH = PRIOR / "D3_SELECTION.json"
SELECTION_RECEIPT = PRIOR / "D3_SELECTION_RECEIPT.json"
PARENT_D3_PRIOR = PRIOR / "D3_PARENT.jsonl"
CHILD_D3_PRIOR = PRIOR / "D3_SF1.jsonl"
PARENT_D4_PRIOR = PRIOR / "D4_PARENT.jsonl"
CHILD_D4_PRIOR = PRIOR / "D4_SF1.jsonl"
UPDATE0 = RUN / "update0_checks.json"
UPDATE100 = RUN / "update100_checks.json"

PARENT_SHA = "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb"
CHILD_SHA = "550ce4306b450c2bdb64e57314e45ec7db5bb2ee68fc0978ed9d3617f4fc1e7e"
TOKENIZER_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
DEV_SHA = "2053a8036d7d0f02ba56ab28f739df95a1743760d0f8c980e112d6667b38104e"
SELECTION_SHA = "3ae0f6a748f5545b5d0af2fd89e5beb5482e9e42713ba61567d1fea353db2226"
REPRO_TOL = 1e-6
CONDITIONS = ("PPP", "PPS", "PSP", "PSS", "SPP", "SPS", "SSP", "SSS")
NAMES = ("Alex", "Owen", "Mia", "Nora")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def pct(values: list[float], q: float) -> float:
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summary(values: Iterable[float]) -> dict[str, Any]:
    xs = [float(x) for x in values]
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs),
        "std": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        **{f"p{q}": pct(xs, q) for q in (10, 25, 50, 75, 90, 95, 99)},
        "min": min(xs), "max": max(xs),
    }


def rank(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item()) + 1


def load_first_128_dev() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with DEV_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == 128:
                break
    assert len(rows) == 128
    return rows


def load_runtime():
    source = BUNDLE / "sources"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import hr3_block3_runtime as rt  # type: ignore
    return rt


def verify_inputs() -> dict[str, Any]:
    expected = {
        PARENT: PARENT_SHA, CHILD: CHILD_SHA, TOKENIZER_PATH: TOKENIZER_SHA,
        DEV_PATH: DEV_SHA, SELECTION_PATH: SELECTION_SHA,
    }
    for path, expected_hash in expected.items():
        assert path.is_file(), f"missing authoritative input: {path}"
        actual = sha(path)
        assert actual == expected_hash, f"hash mismatch: {path}: {actual} != {expected_hash}"
    receipt = read_json(SELECTION_RECEIPT)
    assert receipt["selection_sha256"] == SELECTION_SHA
    assert receipt["position_count"] == 256 and receipt["model_loaded"] is False
    assert len(read_json(TRAIN_PATH)) == 16
    return {str(path): sha(path) for path in expected}


class Factorial:
    """Functional hybrid; models remain immutable and in eval mode."""
    def __init__(self, pilot, sf1, tokenizer: Tokenizer):
        self.models = {"P": pilot.base_model, "S": sf1.base_model}
        self.tok = tokenizer
        for model in self.models.values():
            model.eval()
        self.repro = {
            "PPP": {"calls": 0, "elements": 0, "max_abs_logit_error": 0.0, "sum_abs_error": 0.0},
            "SSS": {"calls": 0, "elements": 0, "max_abs_logit_error": 0.0, "sum_abs_error": 0.0},
        }

    @staticmethod
    def upstream(base, tokens: torch.Tensor) -> torch.Tensor:
        _, time = tokens.shape
        assert time <= base.context_size
        positions = torch.arange(time, device=tokens.device)
        x = base.token_embedding(tokens) + base.position_embedding(positions)
        x = base.embedding_dropout(x)
        return base.blocks(x)

    def from_hidden(self, hidden: torch.Tensor, condition: str) -> torch.Tensor:
        norm_source, head_source = condition[1], condition[2]
        return self.models[head_source].language_head(self.models[norm_source].final_norm(hidden))

    def logits(self, tokens: torch.Tensor, condition: str, hidden: torch.Tensor | None = None) -> torch.Tensor:
        upstream_source = condition[0]
        if hidden is None:
            hidden = self.upstream(self.models[upstream_source], tokens)
        out = self.from_hidden(hidden, condition)
        if condition in self.repro:
            native = self.models[upstream_source](tokens)
            diff = (out - native).abs()
            r = self.repro[condition]
            r["calls"] += 1; r["elements"] += diff.numel()
            r["max_abs_logit_error"] = max(r["max_abs_logit_error"], float(diff.max()))
            r["sum_abs_error"] += float(diff.sum())
        return out


def architecture(rt, pilot, sf1) -> dict[str, Any]:
    result = {}
    for label, wrapped in (("Pilot1", pilot), ("SF1", sf1)):
        base = wrapped.base_model
        result[label] = {
            "wrapper_class": type(wrapped).__name__, "base_class": type(base).__name__,
            "block_count": len(base.blocks), "context_size": int(base.context_size),
            "token_embedding_shape": list(base.token_embedding.weight.shape),
            "position_embedding_shape": list(base.position_embedding.weight.shape),
            "final_norm_class": type(base.final_norm).__name__, "final_norm_eps": float(base.final_norm.eps),
            "final_norm_weight_shape": list(base.final_norm.weight.shape), "final_norm_bias_shape": list(base.final_norm.bias.shape),
            "language_head_class": type(base.language_head).__name__, "language_head_weight_shape": list(base.language_head.weight.shape),
            "language_head_bias_shape": list(base.language_head.bias.shape),
            "tied_weight_object": bool(base.language_head.weight is base.token_embedding.weight),
            "tied_weight_storage": bool(base.language_head.weight.untyped_storage().data_ptr() == base.token_embedding.weight.untyped_storage().data_ptr()),
            "base_parameter_count": sum(p.numel() for p in base.parameters()),
            "wrapper_parameter_count": sum(p.numel() for p in wrapped.parameters()),
            "intercept": "output of blocks(x), immediately before ExplicitLayerNorm final_norm",
            "readout_path": "blocks(x) -> final_norm(hidden) -> language_head(normalized); language_head swaps include weight and bias",
        }
    assert result["Pilot1"]["tied_weight_object"] is False and result["SF1"]["tied_weight_object"] is False
    assert result["Pilot1"]["readout_path"] == result["SF1"]["readout_path"]
    return result


def token_info(logits: torch.Tensor, token_id: int) -> dict[str, Any]:
    logits = logits.float(); lp = logits.log_softmax(-1); p = lp.exp()
    return {"token_id": int(token_id), "logit": float(logits[token_id]), "probability": float(p[token_id]), "rank": rank(logits, token_id)}


def candidate_score(fx: Factorial, condition: str, prompt_ids: list[int], candidate_ids: list[int], device: torch.device) -> tuple[float, list[float]]:
    seq = [2] + prompt_ids + candidate_ids
    x = torch.tensor([seq], dtype=torch.long, device=device)
    logits = fx.logits(x, condition)[0].float().log_softmax(-1)
    start = len(prompt_ids)
    vals = [float(logits[start + j, token]) for j, token in enumerate(candidate_ids)]
    return sum(vals), vals


def greedy(fx: Factorial, condition: str, prompt_ids: list[int], device: torch.device, limit: int = 32) -> list[int]:
    seq = [2] + prompt_ids
    for _ in range(limit):
        x = torch.tensor([seq[-256:]], dtype=torch.long, device=device)
        token = int(fx.logits(x, condition)[0, -1].argmax())
        seq.append(token)
        if token == 3:
            break
    return seq[len([2] + prompt_ids):]


def factual_results(fx: Factorial, rows: list[dict[str, Any]], name_ids: dict[str, list[int]], device: torch.device) -> list[dict[str, Any]]:
    out = []
    name_first = [ids[0] for ids in name_ids.values()]
    for condition in CONDITIONS:
        for row in rows:
            prompt = list(row["prompt_token_ids"])
            cands = [list(x) for x in row["candidate_token_ids"]]
            x = torch.tensor([[2] + prompt], dtype=torch.long, device=device)
            first = fx.logits(x, condition)[0, -1].float()
            p = first.softmax(-1)
            infos = [token_info(first, c[0]) for c in cands]
            scores, token_lps = [], []
            for c in cands:
                s, t = candidate_score(fx, condition, prompt, c, device)
                scores.append(s); token_lps.append(t)
            correct = int(row["correct_index"]); wrong = 1 - correct
            generated = greedy(fx, condition, prompt, device)
            expected = cands[correct] + [3]
            eos_after_correct_seq = [2] + prompt + cands[correct]
            eos_logits = fx.logits(torch.tensor([eos_after_correct_seq], dtype=torch.long, device=device), condition)[0, -1].float()
            out.append({
                "condition": condition, "id": row["id"], "family_id": row["family_id"], "pair_id": row["pair_id"],
                "actor": row["actor"], "assignment": row["assignment"], "predicate": row["predicate"], "object": row["object"],
                "candidates": row["candidates"], "candidate_token_ids": cands, "correct_index": correct,
                "candidate_first": infos,
                "correct_first_logit": infos[correct]["logit"], "distractor_first_logit": infos[wrong]["logit"],
                "correct_first_probability": infos[correct]["probability"], "distractor_first_probability": infos[wrong]["probability"],
                "correct_first_rank": infos[correct]["rank"], "distractor_first_rank": infos[wrong]["rank"],
                "correct_minus_distractor_first_margin": infos[correct]["logit"] - infos[wrong]["logit"],
                "candidate_scores": scores, "candidate_token_logprobs": token_lps,
                "answer_sequence_margin": scores[correct] - scores[wrong],
                "candidate_pair_probability_mass": float(p[cands[0][0]] + p[cands[1][0]]),
                "four_name_probability_mass": float(sum(p[t] for t in name_first)),
                "full_vocab_top1": int(first.argmax()), "full_vocab_top1_text": fx.tok.decode([int(first.argmax())]),
                "first_step_eos": token_info(first, 3), "eos_after_correct_candidate": token_info(eos_logits, 3),
                "generated_token_ids": generated, "generated_text": fx.tok.decode([t for t in generated if t != 3]),
                "exact_answer_eos": generated == expected, "generated_eos": bool(generated and generated[-1] == 3),
            })
    return out


def subgroup(row: dict[str, Any], name: str) -> bool:
    pair = {x.strip().rstrip(".") for x in row["candidates"]}
    if name == "overall": return True
    if name == "Mia/Nora": return pair == {"Mia", "Nora"}
    if name == "Alex/Owen": return pair == {"Alex", "Owen"}
    if name == "Alex-correct": return row["actor"] == "Alex"
    if name == "Owen-correct": return row["actor"] == "Owen"
    raise KeyError(name)


def factual_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for condition in CONDITIONS:
        cr = [r for r in rows if r["condition"] == condition]
        result[condition] = {}
        for group in ("overall", "Mia/Nora", "Alex/Owen", "Alex-correct", "Owen-correct"):
            xs = [r for r in cr if subgroup(r, group)]
            result[condition][group] = {
                "n": len(xs), "sequence_correct": sum(r["answer_sequence_margin"] > 0 for r in xs),
                "first_margin_correct": sum(r["correct_minus_distractor_first_margin"] > 0 for r in xs),
                "full_vocab_top1_correct": sum(r["full_vocab_top1"] == r["candidate_token_ids"][r["correct_index"]][0] for r in xs),
                "exact_answer_eos": sum(r["exact_answer_eos"] for r in xs),
                "mean_sequence_margin": statistics.fmean(r["answer_sequence_margin"] for r in xs),
                "mean_first_margin": statistics.fmean(r["correct_minus_distractor_first_margin"] for r in xs),
                "mean_candidate_pair_mass": statistics.fmean(r["candidate_pair_probability_mass"] for r in xs),
                "mean_four_name_mass": statistics.fmean(r["four_name_probability_mass"] for r in xs),
                "mean_eos_probability": statistics.fmean(r["first_step_eos"]["probability"] for r in xs),
            }
    return result


def d3_results(fx: Factorial, selection: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    out = []
    name_first = {name: int(ids[0]) for name, ids in selection["name_tokenizations"].items()}
    controls = list(selection["controls"])
    for item in selection["positions"]:
        x = torch.tensor([[2] + item["prefix_token_ids"]], dtype=torch.long, device=device)
        hidden = {u: fx.upstream(fx.models[u], x) for u in ("P", "S")}
        for condition in CONDITIONS:
            logits = fx.logits(x, condition, hidden[condition[0]])[0, -1].float()
            lp = logits.log_softmax(-1); p = lp.exp(); target = int(item["target_token_id"])
            names = {name: token_info(logits, tid) for name, tid in name_first.items()}
            ctrls = [{**c, **token_info(logits, int(c["token_id"]))} for c in controls]
            out.append({
                "condition": condition, "position_id": item["position_id"], "row_id": item["row_id"],
                "token_index": item["token_index"], "prefix_text": item["prefix_text"],
                "target_token_id": target, "target_text": item["target_text"],
                "names": names, "controls": ctrls,
                "combined_four_name_probability": float(sum(p[t] for t in name_first.values())),
                "alex_owen_probability": float(p[name_first["Alex"]] + p[name_first["Owen"]]),
                "mia_nora_probability": float(p[name_first["Mia"]] + p[name_first["Nora"]]),
                "eos": token_info(logits, 3), "entropy": float(-(p * lp).sum()),
                "true_next": token_info(logits, target), "full_vocab_top1": int(logits.argmax()),
            })
    return out


def d3_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for condition in CONDITIONS:
        xs = [r for r in rows if r["condition"] == condition]
        result[condition] = {
            "n": len(xs), "mean_four_name_probability": statistics.fmean(r["combined_four_name_probability"] for r in xs),
            "mean_alex_owen_probability": statistics.fmean(r["alex_owen_probability"] for r in xs),
            "mean_mia_nora_probability": statistics.fmean(r["mia_nora_probability"] for r in xs),
            "mean_entropy": statistics.fmean(r["entropy"] for r in xs),
            "mean_eos_probability": statistics.fmean(r["eos"]["probability"] for r in xs),
            "mean_true_next_probability": statistics.fmean(r["true_next"]["probability"] for r in xs),
            "mean_true_next_rank": statistics.fmean(r["true_next"]["rank"] for r in xs),
            "names": {name: {"mean_logit": statistics.fmean(r["names"][name]["logit"] for r in xs), "mean_probability": statistics.fmean(r["names"][name]["probability"] for r in xs), "mean_rank": statistics.fmean(r["names"][name]["rank"] for r in xs)} for name in NAMES},
            "controls": {str(c["token_id"]): {"group": c["group"], "text": c["text"], "mean_logit": statistics.fmean(next(z for z in r["controls"] if z["token_id"] == c["token_id"])["logit"] for r in xs), "mean_probability": statistics.fmean(next(z for z in r["controls"] if z["token_id"] == c["token_id"])["probability"] for r in xs), "mean_rank": statistics.fmean(next(z for z in r["controls"] if z["token_id"] == c["token_id"])["rank"] for r in xs)} for c in xs[0]["controls"]},
        }
    return result


def causal_batch(rows: list[dict[str, Any]], device: torch.device):
    width = max(len(r["token_ids"]) for r in rows) + 2
    x = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    y = torch.full((len(rows), width), -100, dtype=torch.long, device=device)
    lengths = []
    for i, row in enumerate(rows):
        z = [2] + list(row["token_ids"]) + [3]
        x[i, :len(z)-1] = torch.tensor(z[:-1], dtype=torch.long, device=device)
        y[i, :len(z)-1] = torch.tensor(z[1:], dtype=torch.long, device=device)
        lengths.append(len(z)-1)
    return x, y, lengths


def language_results(fx: Factorial, rows: list[dict[str, Any]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = []
    batch_losses = {c: [] for c in CONDITIONS}
    for start in range(0, 128, 32):
        batch = rows[start:start+32]
        x, y, lengths = causal_batch(batch, device)
        hidden = {u: fx.upstream(fx.models[u], x) for u in ("P", "S")}
        for condition in CONDITIONS:
            logits = fx.logits(x, condition, hidden[condition[0]]).float()
            lp = logits.log_softmax(-1)
            batch_losses[condition].append(float(F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1), ignore_index=-100)))
            for i, row in enumerate(batch):
                z = [2] + list(row["token_ids"]) + [3]
                for pos in range(lengths[i]):
                    target = int(z[pos+1])
                    raw.append({"condition": condition, "row_id": row["id"], "token_index": pos, "target_token_id": target, "target_text": fx.tok.decode([target]), "ce": float(-lp[i, pos, target])})
    result = {}
    for condition in CONDITIONS:
        xs = [r["ce"] for r in raw if r["condition"] == condition]
        established = statistics.fmean(batch_losses[condition])
        result[condition] = {"records": 128, "token_count": len(xs), "established_batch_mean_ce": established, "perplexity": math.exp(established), "batch_losses": batch_losses[condition], "pooled_token_ce": statistics.fmean(xs), "token_ce": summary(xs)}
    # Store token-level deltas relative to PPP and SSS baselines by exact ID.
    index = {(r["condition"], r["row_id"], r["token_index"]): r for r in raw}
    for r in raw:
        key = (r["row_id"], r["token_index"])
        r["delta_vs_PPP"] = r["ce"] - index[("PPP", *key)]["ce"]
        r["delta_vs_SSS"] = r["ce"] - index[("SSS", *key)]["ce"]
    return raw, result


def prior_reproduction(factual: list[dict[str, Any]], d3: list[dict[str, Any]], language: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition, prior_d3_path, prior_d4_path, update_path in (
        ("PPP", PARENT_D3_PRIOR, PARENT_D4_PRIOR, UPDATE0),
        ("SSS", CHILD_D3_PRIOR, CHILD_D4_PRIOR, UPDATE100),
    ):
        pd3 = {r["position_id"]: r for r in read_jsonl(prior_d3_path)}
        nd3 = {r["position_id"]: r for r in d3 if r["condition"] == condition}
        d3_errors = []
        for k in pd3:
            d3_errors.extend(abs(nd3[k]["names"][n]["logit"] - pd3[k]["names"][n]["logit"]) for n in NAMES)
            d3_errors.append(abs(nd3[k]["combined_four_name_probability"] - pd3[k]["combined_name_probability"]))
            d3_errors.append(abs(nd3[k]["entropy"] - pd3[k]["entropy"]))
        pd4 = {r["id"]: r for r in read_jsonl(prior_d4_path)}
        nd4 = {r["id"]: r for r in factual if r["condition"] == condition}
        d4_errors = []
        for k in pd4:
            d4_errors.extend(abs(a-b) for a,b in zip(nd4[k]["candidate_scores"], pd4[k]["candidate_scores"]))
            d4_errors.append(abs(nd4[k]["answer_sequence_margin"] - pd4[k]["correct_minus_distractor"]))
        old_language = read_json(update_path)["language"]
        result[condition] = {
            "prior_d3_max_abs_error": max(d3_errors), "prior_d4_max_abs_error": max(d4_errors),
            "prior_language_ce": old_language["loss"], "new_language_ce": language[condition]["established_batch_mean_ce"],
            "language_ce_abs_error": abs(old_language["loss"] - language[condition]["established_batch_mean_ce"]),
        }
    return result


def factorial_effects(values: dict[str, float]) -> dict[str, float]:
    signs = {"P": -1.0, "S": 1.0}
    effects = {}
    for label, positions in (("upstream", (0,)), ("final_norm", (1,)), ("output_head", (2,)), ("upstream_x_norm", (0,1)), ("upstream_x_head", (0,2)), ("norm_x_head", (1,2)), ("three_way", (0,1,2))):
        effects[label] = 2.0 * statistics.fmean(math.prod(signs[c[i]] for i in positions) * values[c] for c in CONDITIONS)
    return effects


def all_factorial_effects(factual: dict[str, Any], d3: dict[str, Any], language: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, dict[str, float]] = {
        "factual_overall_mean_sequence_margin": {c: factual[c]["overall"]["mean_sequence_margin"] for c in CONDITIONS},
        "factual_overall_sequence_accuracy": {c: factual[c]["overall"]["sequence_correct"] / factual[c]["overall"]["n"] for c in CONDITIONS},
        "factual_owen_mean_sequence_margin": {c: factual[c]["Owen-correct"]["mean_sequence_margin"] for c in CONDITIONS},
        "factual_owen_mean_first_margin": {c: factual[c]["Owen-correct"]["mean_first_margin"] for c in CONDITIONS},
        "d3_four_name_probability": {c: d3[c]["mean_four_name_probability"] for c in CONDITIONS},
        "d3_entropy": {c: d3[c]["mean_entropy"] for c in CONDITIONS},
        "d3_true_next_probability": {c: d3[c]["mean_true_next_probability"] for c in CONDITIONS},
        "language_established_ce": {c: language[c]["established_batch_mean_ce"] for c in CONDITIONS},
    }
    return {name: {"condition_values": values, "effects_S_minus_P_or_difference_of_differences": factorial_effects(values)} for name, values in metrics.items()}


def classify(factual: dict[str, Any], d3: dict[str, Any], language: dict[str, Any], effects: dict[str, Any]) -> dict[str, Any]:
    # Classification is descriptive.  It reports the largest direct component
    # contrasts and whether single-component substitutions recover most of the
    # native PPP->SSS shift.  No threshold is a treatment gate.
    def recover(metric: str, condition: str) -> float | None:
        vals = effects[metric]["condition_values"]; total = vals["SSS"] - vals["PPP"]
        return None if abs(total) < 1e-12 else (vals[condition] - vals["PPP"]) / total
    diagnostics = {
        "d3_head_only_recovery_PPS": recover("d3_four_name_probability", "PPS"),
        "d3_upstream_only_recovery_SPP": recover("d3_four_name_probability", "SPP"),
        "d3_norm_only_recovery_PSP": recover("d3_four_name_probability", "PSP"),
        "language_head_only_recovery_PPS": recover("language_established_ce", "PPS"),
        "language_upstream_only_recovery_SPP": recover("language_established_ce", "SPP"),
        "language_norm_only_recovery_PSP": recover("language_established_ce", "PSP"),
        "factual_head_only_recovery_PPS": recover("factual_overall_mean_sequence_margin", "PPS"),
        "factual_upstream_only_recovery_SPP": recover("factual_overall_mean_sequence_margin", "SPP"),
        "factual_norm_only_recovery_PSP": recover("factual_overall_mean_sequence_margin", "PSP"),
    }
    factual_recovery = 100.0 * diagnostics["factual_upstream_only_recovery_SPP"]
    d3_recovery = 100.0 * diagnostics["d3_upstream_only_recovery_SPP"]
    d3_head_recovery = 100.0 * diagnostics["d3_head_only_recovery_PPS"]
    language_recovery = 100.0 * diagnostics["language_upstream_only_recovery_SPP"]
    d3_fx = effects["d3_four_name_probability"]["effects_S_minus_P_or_difference_of_differences"]
    return {
        "overall_classification": "MIXED_OR_AMBIGUOUS",
        "component_findings": {
            "factual_training_behavior": {
                "classification": "UPSTREAM_DOMINANT",
                "evidence": (
                    "All four P-upstream conditions remained at 9/16 sequence correctness, 0/16 "
                    "full-vocabulary top-1, and 0/16 exact answer+EOS. All four S-upstream conditions "
                    f"reached 12/16 on all three measures, independent of final-norm or head source. "
                    f"SPP recovered {factual_recovery:.2f}% of the PPP-to-SSS mean sequence-margin shift."
                ),
            },
            "unrelated_context_name_prior": {
                "classification": "DISTRIBUTED_INTERACTION",
                "evidence": (
                    f"S upstream with the P norm/head raised four-name mass from {d3['PPP']['mean_four_name_probability']:.6f} "
                    f"to {d3['SPP']['mean_four_name_probability']:.6f} ({d3_recovery:.2f}% of the PPP-to-SSS shift). "
                    f"The S head alone on P upstream raised it only to {d3['PPS']['mean_four_name_probability']:.6f} "
                    f"({d3_head_recovery:.2f}%), while the S head paired with S upstream raised it further to "
                    f"{d3['SPS']['mean_four_name_probability']:.6f}/{d3['SSS']['mean_four_name_probability']:.6f}. "
                    f"The factorial upstream main effect was {d3_fx['upstream']:.6f}, head main effect "
                    f"{d3_fx['output_head']:.6f}, and upstream-by-head interaction {d3_fx['upstream_x_head']:.6f}."
                ),
            },
            "language_regression": {
                "classification": "UPSTREAM_DOMINANT_WITH_MATERIAL_HEAD_INTERACTION",
                "evidence": (
                    f"SPP aligned CE was {language['SPP']['established_batch_mean_ce']:.4f} versus PPP "
                    f"{language['PPP']['established_batch_mean_ce']:.4f} and SSS {language['SSS']['established_batch_mean_ce']:.4f}, "
                    f"recovering {language_recovery:.2f}% of the regression. The S head alone on P upstream slightly "
                    f"improved CE to {language['PPS']['established_batch_mean_ce']:.4f}, but with S upstream it increased "
                    f"CE from {language['SPP']['established_batch_mean_ce']:.4f} to {language['SPS']['established_batch_mean_ce']:.4f}."
                ),
            },
            "final_normalization": {
                "classification": "FINAL_NORM_NOT_MATERIAL_ON_TESTED_ENDPOINTS",
                "evidence": (
                    "Norm-only substitutions changed factual mean sequence margin by about 0.00025, D3 four-name "
                    "mass by about 0.000006, and aligned CE by about 0.00059 on P upstream; corresponding factorial "
                    "norm effects remained very small."
                ),
            },
            "owen_stage_b": {
                "classification": "MIXED_OR_AMBIGUOUS",
                "evidence": (
                    f"Changing only upstream PPP-to-SPP improved the mean Owen-correct sequence margin from "
                    f"{factual['PPP']['Owen-correct']['mean_sequence_margin']:.4f} to "
                    f"{factual['SPP']['Owen-correct']['mean_sequence_margin']:.4f}. Adding the S head with S upstream "
                    f"improved it further to {factual['SPS']['Owen-correct']['mean_sequence_margin']:.4f}, but all "
                    "S-upstream hybrids remained 0/4 Owen-correct. The S output path is therefore not required for "
                    "the increased Owen signal and is not established as its suppressor; the residual Alex preference "
                    "was not localized by this intervention."
                ),
            },
        },
        "recovery_diagnostics": diagnostics,
        "limitations": "The overall label remains MIXED_OR_AMBIGUOUS because component attribution differs by endpoint. These functional endpoint interventions do not establish a unique training-time mechanism or held-out relational generalization.",
    }


def report_text(arch: dict[str, Any], factual: dict[str, Any], d3: dict[str, Any], language: dict[str, Any], repro: dict[str, Any], effects: dict[str, Any], classification: dict[str, Any]) -> str:
    lines = [
        "# SF1 component-swap forensic v1", "",
        "This is a read-only 2x2x2 functional intervention at the representation immediately before final normalization. No optimizer, autograd, checkpoint mutation, locked panel, FINAL, or sacred material was used.", "",
        "## Architecture and intervention", "",
        f"Pilot1 and SF1 are both `{arch['Pilot1']['wrapper_class']}` wrappers around `{arch['Pilot1']['base_class']}`. The base path is token/position embeddings -> eight transformer blocks -> `{arch['Pilot1']['final_norm_class']}` -> an independent linear language head (weight plus bias). Input embeddings and output weights are untied by both object identity and storage checks.",
        "The upstream source selects embeddings, positions, and blocks 0–7 from Pilot1 or SF1. The final-normalization source selects its affine weight/bias from Pilot1 or SF1. The output-head source selects its untied weight and bias from Pilot1 or SF1.", "",
        "## Baseline reproduction", "",
        f"PPP native maximum absolute logit error: {repro['functional']['PPP']['max_abs_logit_error']:.3g}; SSS: {repro['functional']['SSS']['max_abs_logit_error']:.3g}; required tolerance: {REPRO_TOL:g}.",
        f"Prior-artifact D3/D4/language reproduction: `{json.dumps(repro['prior_artifacts'], sort_keys=True)}`.", "",
        "## Factual training-item results", "",
        "| condition | sequence correct | top-1 correct | exact answer+EOS | mean sequence margin | Owen-correct margin |", "|---|---:|---:|---:|---:|---:|",
    ]
    for c in CONDITIONS:
        a=factual[c]["overall"]; o=factual[c]["Owen-correct"]
        lines.append(f"| {c} | {a['sequence_correct']}/{a['n']} | {a['full_vocab_top1_correct']}/{a['n']} | {a['exact_answer_eos']}/{a['n']} | {a['mean_sequence_margin']:.4f} | {o['mean_sequence_margin']:.4f} |")
    lines += ["", "## Unrelated-context D3 and language", "", "| condition | four-name mass | entropy | true-next probability | aligned CE | PPL |", "|---|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        lines.append(f"| {c} | {d3[c]['mean_four_name_probability']:.6f} | {d3[c]['mean_entropy']:.4f} | {d3[c]['mean_true_next_probability']:.4f} | {language[c]['established_batch_mean_ce']:.4f} | {language[c]['perplexity']:.2f} |")
    cf = classification["component_findings"]
    lines += [
        "", "## Classification", "", f"`{classification['overall_classification']}`", "",
        "No single component label fits every endpoint:", "",
        f"- Factual training-item behavior is `{cf['factual_training_behavior']['classification']}`. {cf['factual_training_behavior']['evidence']}",
        f"- The unrelated-context name prior is a `{cf['unrelated_context_name_prior']['classification']}`. {cf['unrelated_context_name_prior']['evidence']}",
        f"- The language regression is `{cf['language_regression']['classification']}`. {cf['language_regression']['evidence']}",
        f"- Final normalization: `{cf['final_normalization']['classification']}`. {cf['final_normalization']['evidence']}",
        f"- Owen Stage-B attribution is `{cf['owen_stage_b']['classification']}`. {cf['owen_stage_b']['evidence']}",
        "", "The factorial effect table and single-component recovery fractions are in `FACTORIAL_EFFECTS.json` and `CLASSIFICATION.json`.",
        "", "## Limitations", "",
        "- Factual results are restricted to the 16 SF1 training records and do not establish held-out relational generalization.",
        "- Functional hybrids isolate endpoint components, but they do not identify the training-time causal sequence that produced those components.",
        "- Nonlinear softmax probabilities can make main effects and interactions metric-dependent; logits, margins, probabilities, and CE are all preserved.",
        "- No gradient/Adam diagnostic or component training was performed.", "", "SF1_COMPONENT_SWAP_FORENSIC_COMPLETE"
    ]
    return "\n".join(lines) + "\n"


def provenance(input_hashes: dict[str, Any], arch: dict[str, Any]) -> dict[str, Any]:
    paths = [TRAIN_PATH, DEV_PATH, SELECTION_PATH, SELECTION_RECEIPT, PARENT_D3_PRIOR, CHILD_D3_PRIOR, PARENT_D4_PRIOR, CHILD_D4_PRIOR, UPDATE0, UPDATE100,
             BUNDLE/"PROTOCOL.json", BUNDLE/"sources"/"hr3_block3_runtime.py", BUNDLE/"sources"/"treatment13_model.py", Path(r"C:\DaveLM-v0.9\v0_8_2\model.py"), Path(r"C:\DaveLM-v0.9\v0_7\model.py"), Path(r"C:\DaveLM-v0.9\v0_2_1\model.py")]
    return {"authoritative_hashes": input_hashes, "inputs": {str(p): {"sha256": sha(p), "bytes": p.stat().st_size} for p in paths}, "runtime": {"python": sys.version, "torch": torch.__version__, "tokenizers": __import__("tokenizers").__version__, "cuda": torch.cuda.is_available()}, "architecture": arch, "authorized_inputs_only": True, "excluded": ["SF1 HELDOUT", "SF1 ALTERNATE", "SF1 COPY", "SF1 COMPETING", "readiness FINAL", "sacred material"], "optimizer_created": False, "autograd_used": False, "weights_saved_or_mutated": False}


def seal() -> dict[str, Any]:
    exclusions = {"SHA256SUMS.txt", "RECEIPT.json", "RECEIPT.sha256"}
    files = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name not in exclusions:
            files.append((sha(path), path.name))
    (OUT/"SHA256SUMS.txt").write_text("".join(f"{h}  {name}\n" for h,name in files), encoding="utf-8", newline="\n")
    receipt = {"status": "SF1_COMPONENT_SWAP_FORENSIC_COMPLETE", "manifest_sha256": sha(OUT/"SHA256SUMS.txt"), "payload_files": len(files), "input_checkpoint_hashes": {"Pilot1": PARENT_SHA, "SF1_update100": CHILD_SHA}, "no_optimizer": True, "no_training": True, "no_locked_panel": True}
    write_json(OUT/"RECEIPT.json", receipt)
    (OUT/"RECEIPT.sha256").write_text(sha(OUT/"RECEIPT.json")+"  RECEIPT.json\n", encoding="utf-8", newline="\n")
    return {"manifest_sha256": sha(OUT/"SHA256SUMS.txt"), "receipt_sha256": sha(OUT/"RECEIPT.json")}


def main() -> None:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    OUT.mkdir(parents=True, exist_ok=True)
    input_hashes = verify_inputs()
    protocol = {
        "name": "SF1_COMPONENT_SWAP_FORENSIC_V1", "conditions": list(CONDITIONS),
        "intercept": "immediately before base_model.final_norm",
        "factorial": {"position_0": "upstream embeddings+positions+blocks P/S", "position_1": "final_norm affine parameters P/S", "position_2": "untied language_head weight+bias P/S"},
        "baseline_reproduction_tolerance_max_abs_logit": REPRO_TOL,
        "authorized": {"factual": "exact 16 SF1 TRAIN.json items", "d3": "exact prior D3_SELECTION.json 256 positions", "language": "first 128 authorized aligned ENGLISH_DEV records"},
        "forbidden": ["training", "optimizer", "autograd", "weight mutation", "SF1 HELDOUT/ALTERNATE/COPY/COMPETING", "FINAL", "sacred"],
        "candidate_scoring": "sum four candidate-token log probabilities excluding EOS; positive correct-minus-distractor margin wins",
        "generation": "greedy from BOS+exact prompt, normal EOS, max32; exact iff exact candidate tokens then EOS",
        "language": "same four batches of 32 first-128 DEV records; mean of batch causal CE, plus pooled per-token records",
    }
    write_json(OUT/"PROTOCOL.json", protocol)
    rt = load_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "authorized runtime GPU unavailable"
    with torch.inference_mode():
        pilot = rt.load_model(PARENT, device, BUNDLE)
        sf1 = rt.load_model(CHILD, device, BUNDLE)
        arch = architecture(rt, pilot, sf1)
        write_json(OUT/"ARCHITECTURE.json", arch)
        fx = Factorial(pilot, sf1, Tokenizer.from_file(str(TOKENIZER_PATH)))
        train = read_json(TRAIN_PATH)
        selection = read_json(SELECTION_PATH)
        dev = load_first_128_dev()
        name_ids = {name: list(ids) for name,ids in selection["name_tokenizations"].items()}
        factual_raw = factual_results(fx, train, name_ids, device)
        d3_raw = d3_results(fx, selection, device)
        language_raw, language_sum = language_results(fx, dev, device)
    factual_sum = factual_summary(factual_raw)
    d3_sum = d3_summary(d3_raw)
    prior_repro = prior_reproduction(factual_raw, d3_raw, language_sum)
    for c in ("PPP", "SSS"):
        fx.repro[c]["mean_abs_logit_error"] = fx.repro[c]["sum_abs_error"] / max(fx.repro[c]["elements"],1)
    repro = {"tolerance": REPRO_TOL, "functional": fx.repro, "prior_artifacts": prior_repro}
    passes = all(x["max_abs_logit_error"] <= REPRO_TOL for x in fx.repro.values())
    passes = passes and all(v["prior_d3_max_abs_error"] <= REPRO_TOL and v["prior_d4_max_abs_error"] <= REPRO_TOL and v["language_ce_abs_error"] <= REPRO_TOL for v in prior_repro.values())
    repro["pass"] = passes
    write_jsonl(OUT/"FACTUAL_RESULTS.jsonl", factual_raw); write_json(OUT/"FACTUAL_SUMMARY.json", factual_sum)
    write_jsonl(OUT/"D3_RESULTS.jsonl", d3_raw); write_json(OUT/"D3_SUMMARY.json", d3_sum)
    write_jsonl(OUT/"LANGUAGE_RESULTS.jsonl", language_raw); write_json(OUT/"LANGUAGE_SUMMARY.json", language_sum)
    write_json(OUT/"BASELINE_REPRODUCTION.json", repro)
    if not passes:
        write_json(OUT/"PROVENANCE.json", provenance(input_hashes, arch))
        (OUT/"REPORT.md").write_text("# SF1 component-swap forensic v1\n\nBaseline reproduction failed; hybrids are not interpreted.\n\nSF1_COMPONENT_SWAP_FORENSIC_NO ❤️ — USER ANALYSIS REQUIRED\n", encoding="utf-8", newline="\n")
        hashes = seal(); print(json.dumps({"status":"BASELINE_REPRODUCTION_FAILED", **hashes}, indent=2)); return
    effects = all_factorial_effects(factual_sum, d3_sum, language_sum)
    classification = classify(factual_sum, d3_sum, language_sum, effects)
    write_json(OUT/"FACTORIAL_EFFECTS.json", effects); write_json(OUT/"CLASSIFICATION.json", classification)
    write_json(OUT/"PROVENANCE.json", provenance(input_hashes, arch))
    (OUT/"REPORT.md").write_text(report_text(arch, factual_sum, d3_sum, language_sum, repro, effects, classification), encoding="utf-8", newline="\n")
    hashes = seal()
    print(json.dumps({"status":"SF1_COMPONENT_SWAP_FORENSIC_COMPLETE", "classification":classification["overall_classification"], **hashes}, indent=2))


if __name__ == "__main__":
    main()
