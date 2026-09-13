"""SF1_UPSTREAM_LOCALIZATION_FORENSIC_V1.

Read-only cumulative activation/state splicing between authoritative Pilot1 and
SF1-update100 base-model paths.  This file intentionally has no optimizer,
backward pass, checkpoint save, or access to locked evaluation panels.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

SITE = r"C:\DaveLM\.venv\Lib\site-packages"
if SITE not in sys.path:
    sys.path.insert(0, SITE)

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from tokenizers import Tokenizer  # type: ignore


ROOT = Path(r"C:\DaveLM-CADAVER")
OUT = ROOT / "sf1_upstream_localization_forensic_v1"
BUNDLE = ROOT / "single_fact_acquisition_sf1_seed87011"
RUN = ROOT / "single_fact_acquisition_sf1_seed87011_run"
PRIOR = ROOT / "sf1_readout_selection_forensic_v1"
COMPONENT = ROOT / "sf1_component_swap_forensic_v1"
PARENT = ROOT / "language_pilot_1_early_block_protection_seed8380" / "pilot_run" / "checkpoints" / "seed_8380" / "latest.pt"
CHILD = RUN / "checkpoint_100.pt"
TOKENIZER_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
TRAIN_PATH = BUNDLE / "TRAIN.json"
DEV_PATH = BUNDLE / "data" / "ENGLISH_DEV.jsonl"
SELECTION_PATH = PRIOR / "D3_SELECTION.json"

PARENT_SHA = "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb"
CHILD_SHA = "550ce4306b450c2bdb64e57314e45ec7db5bb2ee68fc0978ed9d3617f4fc1e7e"
TOKENIZER_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
DEV_SHA = "2053a8036d7d0f02ba56ab28f739df95a1743760d0f8c980e112d6667b38104e"
SELECTION_SHA = "3ae0f6a748f5545b5d0af2fd89e5beb5482e9e42713ba61567d1fea353db2226"
REPRO_TOL = 1e-6
NAMES = ("Alex", "Owen", "Mia", "Nora")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_first_128_dev() -> list[dict[str, Any]]:
    rows = []
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
    sys.path.insert(0, str(source))
    import hr3_block3_runtime as rt  # type: ignore
    return rt


def verify_inputs() -> dict[str, str]:
    expected = {
        PARENT: PARENT_SHA,
        CHILD: CHILD_SHA,
        TOKENIZER_PATH: TOKENIZER_SHA,
        DEV_PATH: DEV_SHA,
        SELECTION_PATH: SELECTION_SHA,
    }
    for path, wanted in expected.items():
        assert path.is_file(), f"missing authoritative input: {path}"
        got = sha(path)
        assert got == wanted, f"hash mismatch {path}: {got} != {wanted}"
    train = read_json(TRAIN_PATH)
    selection = read_json(SELECTION_PATH)
    assert len(train) == 16
    assert len(selection["positions"]) == 256
    forbidden_names = {"HELDOUT.json", "ALTERNATE.json", "COPY.json", "COMPETING.json"}
    assert not any(p.name in forbidden_names for p in [TRAIN_PATH, DEV_PATH, SELECTION_PATH])
    return {str(p): sha(p) for p in expected} | {str(TRAIN_PATH): sha(TRAIN_PATH)}


class PathPatcher:
    """Exact manual base path with a single residual-stream splice boundary."""

    def __init__(self, pilot, sf1):
        self.wrappers = {"P": pilot, "S": sf1}
        self.models = {"P": pilot.base_model, "S": sf1.base_model}
        for wrapper in self.wrappers.values():
            wrapper.eval()
            for param in wrapper.parameters():
                param.requires_grad_(False)
        self.boundaries: list[dict[str, Any]] = [{"name": "embedding_output", "stage": -1, "block": None, "sublayer": "embedding"}]
        for i in range(8):
            self.boundaries.append({"name": f"after_block_{i}_attention_residual", "stage": 2 * i, "block": i, "sublayer": "attention_residual"})
            self.boundaries.append({"name": f"after_block_{i}_mlp_residual", "stage": 2 * i + 1, "block": i, "sublayer": "mlp_residual"})

    @staticmethod
    def embed(base, tokens: torch.Tensor) -> torch.Tensor:
        _, time = tokens.shape
        assert time <= base.context_size
        positions = torch.arange(time, device=tokens.device)
        return base.embedding_dropout(base.token_embedding(tokens) + base.position_embedding(positions))

    @staticmethod
    def apply_stage(base, x: torch.Tensor, stage: int) -> torch.Tensor:
        block = base.blocks[stage // 2]
        if stage % 2 == 0:
            return x + block.attention_residual_dropout(block.attention(block.norm1(x)))
        return x + block.feed_forward_residual_dropout(block.feed_forward(block.norm2(x)))

    def prefix(self, source: str, tokens: torch.Tensor, boundary_stage: int) -> torch.Tensor:
        base = self.models[source]
        x = self.embed(base, tokens)
        for stage in range(boundary_stage + 1):
            x = self.apply_stage(base, x, stage)
        return x

    def suffix(self, source: str, x: torch.Tensor, boundary_stage: int) -> torch.Tensor:
        base = self.models[source]
        for stage in range(boundary_stage + 1, 16):
            x = self.apply_stage(base, x, stage)
        return x

    def output(self, source: str, x: torch.Tensor) -> torch.Tensor:
        base = self.models[source]
        return base.language_head(base.final_norm(x))

    def splice(self, tokens: torch.Tensor, boundary_stage: int, prefix: str, suffix: str, output: str) -> torch.Tensor:
        return self.output(output, self.suffix(suffix, self.prefix(prefix, tokens, boundary_stage), boundary_stage))

    def native(self, source: str, tokens: torch.Tensor) -> torch.Tensor:
        return self.models[source](tokens)


def architecture(patcher: PathPatcher) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source, wrapper in patcher.wrappers.items():
        base = wrapper.base_model
        result[source] = {
            "wrapper": type(wrapper).__name__,
            "base": type(base).__name__,
            "blocks": len(base.blocks),
            "context_size": int(base.context_size),
            "embedding_shape": list(base.token_embedding.weight.shape),
            "position_shape": list(base.position_embedding.weight.shape),
            "final_norm": type(base.final_norm).__name__,
            "language_head": type(base.language_head).__name__,
            "head_shape": list(base.language_head.weight.shape),
            "input_output_tied_object": base.language_head.weight is base.token_embedding.weight,
            "input_output_tied_storage": base.language_head.weight.untyped_storage().data_ptr() == base.token_embedding.weight.untyped_storage().data_ptr(),
            "t13_auxiliary_parameters": [n for n, _ in wrapper.named_parameters() if not n.startswith("base_model.")],
        }
    assert result["P"]["input_output_tied_object"] is False
    assert result["S"]["input_output_tied_object"] is False
    result["ordinary_language_execution_graph"] = [
        "token_embedding + position_embedding", "embedding_dropout",
        *[f"block_{i}.norm1->attention->residual; norm2->MLP->residual" for i in range(8)],
        "final_norm", "untied language_head",
    ]
    result["t13_role"] = (
        "Treatment13Model registers a final_norm hook and invokes localizer/wq/wk/wv/wo only in its specialized "
        "forward(input_ids,qpos,anspos). All authorized factual, D3, and language endpoints call base_model directly; "
        "T13 localization/retrieval parameters are registered but inactive in these interventions."
    )
    return result


def condition_specs(patcher: PathPatcher) -> list[dict[str, Any]]:
    specs = [
        {"id": "native_P", "kind": "native", "source": "P", "boundary": None},
        {"id": "native_S", "kind": "native", "source": "S", "boundary": None},
    ]
    for boundary in patcher.boundaries:
        for prefix, suffix in (("P", "S"), ("S", "P")):
            for output in ("P", "S"):
                specs.append({
                    "id": f"{boundary['name']}__{prefix}_to_{suffix}__{output}_output",
                    "kind": "splice", "boundary": boundary["name"], "stage": boundary["stage"],
                    "prefix": prefix, "suffix": suffix, "output": output,
                })
    return specs


def make_logits_fn(patcher: PathPatcher, spec: dict[str, Any]) -> Callable[[torch.Tensor], torch.Tensor]:
    if spec["kind"] == "native":
        return lambda x: patcher.native(spec["source"], x)
    return lambda x: patcher.splice(x, int(spec["stage"]), spec["prefix"], spec["suffix"], spec["output"])


def rank(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item()) + 1


def eval_variable_sequences(sequences: list[list[int]], fn: Callable[[torch.Tensor], torch.Tensor], device: torch.device) -> list[torch.Tensor]:
    groups: dict[int, list[int]] = defaultdict(list)
    for i, seq in enumerate(sequences):
        groups[len(seq)].append(i)
    out: list[torch.Tensor | None] = [None] * len(sequences)
    for _, indexes in sorted(groups.items()):
        x = torch.tensor([sequences[i] for i in indexes], dtype=torch.long, device=device)
        logits = fn(x).float().cpu()
        for j, original in enumerate(indexes):
            out[original] = logits[j]
    return [x for x in out if x is not None]


def factual_metrics(spec: dict[str, Any], fn, rows: list[dict[str, Any]], name_ids: dict[str, list[int]], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt_seqs = [[2] + list(r["prompt_token_ids"]) for r in rows]
    prompt_logits = eval_variable_sequences(prompt_seqs, fn, device)
    candidate_seqs, candidate_keys = [], []
    eos_seqs, eos_keys = [], []
    for i, row in enumerate(rows):
        for c, ids in enumerate(row["candidate_token_ids"]):
            candidate_seqs.append([2] + list(row["prompt_token_ids"]) + list(ids))
            candidate_keys.append((i, c))
        correct = list(row["candidate_token_ids"][int(row["correct_index"])])
        eos_seqs.append([2] + list(row["prompt_token_ids"]) + correct)
        eos_keys.append(i)
    cand_logits = eval_variable_sequences(candidate_seqs, fn, device)
    eos_logits = eval_variable_sequences(eos_seqs, fn, device)
    cand_map = {k: v for k, v in zip(candidate_keys, cand_logits)}
    eos_map = {k: v for k, v in zip(eos_keys, eos_logits)}
    name_first = [v[0] for v in name_ids.values()]
    raw = []
    for i, row in enumerate(rows):
        prompt_len = len(row["prompt_token_ids"])
        first = prompt_logits[i][-1]
        p = first.softmax(-1)
        scores, token_lps = [], []
        for c, ids in enumerate(row["candidate_token_ids"]):
            lp = cand_map[(i, c)].log_softmax(-1)
            vals = [float(lp[prompt_len + j, int(t)]) for j, t in enumerate(ids)]
            scores.append(sum(vals)); token_lps.append(vals)
        correct = int(row["correct_index"]); wrong = 1 - correct
        c0 = int(row["candidate_token_ids"][0][0]); c1 = int(row["candidate_token_ids"][1][0])
        expected_ids = list(row["candidate_token_ids"][correct])
        teacher_argmax = [int(cand_map[(i, correct)][prompt_len + j].argmax()) for j in range(len(expected_ids))]
        eos_argmax = int(eos_map[i][-1].argmax())
        raw.append({
            "condition": spec["id"], "boundary": spec.get("boundary"), "id": row["id"], "family_id": row["family_id"],
            "actor": row["actor"], "candidates": row["candidates"], "correct_index": correct,
            "correct_first_logit": float(first[int(row["candidate_token_ids"][correct][0])]),
            "distractor_first_logit": float(first[int(row["candidate_token_ids"][wrong][0])]),
            "correct_first_probability": float(p[int(row["candidate_token_ids"][correct][0])]),
            "distractor_first_probability": float(p[int(row["candidate_token_ids"][wrong][0])]),
            "correct_first_rank": rank(first, int(row["candidate_token_ids"][correct][0])),
            "distractor_first_rank": rank(first, int(row["candidate_token_ids"][wrong][0])),
            "first_margin": float(first[int(row["candidate_token_ids"][correct][0])] - first[int(row["candidate_token_ids"][wrong][0])]),
            "candidate_scores": scores, "candidate_token_logprobs": token_lps,
            "sequence_margin": scores[correct] - scores[wrong],
            "candidate_pair_mass": float(p[c0] + p[c1]), "four_name_mass": float(sum(p[t] for t in name_first)),
            "full_vocab_top1": int(first.argmax()), "eos_probability": float(p[3]), "eos_rank": rank(first, 3),
            "teacher_forced_exact_answer_eos": teacher_argmax == expected_ids and eos_argmax == 3,
        })
    groups = {
        "overall": lambda r: True,
        "Mia/Nora": lambda r: set(x.strip().rstrip(".") for x in r["candidates"]) == {"Mia", "Nora"},
        "Alex/Owen": lambda r: set(x.strip().rstrip(".") for x in r["candidates"]) == {"Alex", "Owen"},
        "Alex-correct": lambda r: r["actor"] == "Alex",
        "Owen-correct": lambda r: r["actor"] == "Owen",
    }
    summary = {}
    for name, pred in groups.items():
        xs = [r for r in raw if pred(r)]
        summary[name] = {
            "n": len(xs), "sequence_correct": sum(r["sequence_margin"] > 0 for r in xs),
            "first_margin_correct": sum(r["first_margin"] > 0 for r in xs),
            "full_vocab_top1_correct": sum(r["full_vocab_top1"] == int(rows[[z["id"] for z in rows].index(r["id"])]["candidate_token_ids"][r["correct_index"]][0]) for r in xs),
            "teacher_forced_exact_answer_eos": sum(r["teacher_forced_exact_answer_eos"] for r in xs),
            "mean_sequence_margin": statistics.fmean(r["sequence_margin"] for r in xs),
            "mean_first_margin": statistics.fmean(r["first_margin"] for r in xs),
            "mean_candidate_pair_mass": statistics.fmean(r["candidate_pair_mass"] for r in xs),
            "mean_four_name_mass": statistics.fmean(r["four_name_mass"] for r in xs),
        }
    return raw, summary


def d3_metrics(spec: dict[str, Any], fn, selection: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positions = selection["positions"]
    seqs = [[2] + list(r["prefix_token_ids"]) for r in positions]
    logits_all = eval_variable_sequences(seqs, fn, device)
    name_first = {n: int(ids[0]) for n, ids in selection["name_tokenizations"].items()}
    controls = list(selection["controls"])
    raw = []
    for item, all_logits in zip(positions, logits_all):
        logits = all_logits[-1]; lp = logits.log_softmax(-1); p = lp.exp(); target = int(item["target_token_id"])
        names = {n: {"logit": float(logits[t]), "probability": float(p[t]), "rank": rank(logits, t)} for n, t in name_first.items()}
        control_values = [{**c, "logit": float(logits[int(c["token_id"])]), "probability": float(p[int(c["token_id"])]), "rank": rank(logits, int(c["token_id"]))} for c in controls]
        raw.append({
            "condition": spec["id"], "boundary": spec.get("boundary"), "position_id": item["position_id"],
            "names": names, "controls": control_values,
            "four_name_mass": float(sum(p[t] for t in name_first.values())),
            "alex_owen_mass": float(p[name_first["Alex"]] + p[name_first["Owen"]]),
            "mia_nora_mass": float(p[name_first["Mia"]] + p[name_first["Nora"]]),
            "entropy": float(-(p * lp).sum()), "eos_probability": float(p[3]), "eos_rank": rank(logits, 3),
            "true_next_probability": float(p[target]), "true_next_rank": rank(logits, target), "full_vocab_top1": int(logits.argmax()),
        })
    summary = {
        "n": len(raw), "mean_four_name_mass": statistics.fmean(r["four_name_mass"] for r in raw),
        "mean_alex_owen_mass": statistics.fmean(r["alex_owen_mass"] for r in raw),
        "mean_mia_nora_mass": statistics.fmean(r["mia_nora_mass"] for r in raw),
        "mean_entropy": statistics.fmean(r["entropy"] for r in raw),
        "mean_eos_probability": statistics.fmean(r["eos_probability"] for r in raw),
        "mean_true_next_probability": statistics.fmean(r["true_next_probability"] for r in raw),
        "mean_true_next_rank": statistics.fmean(r["true_next_rank"] for r in raw),
        "names": {n: {"mean_logit": statistics.fmean(r["names"][n]["logit"] for r in raw), "mean_probability": statistics.fmean(r["names"][n]["probability"] for r in raw), "mean_rank": statistics.fmean(r["names"][n]["rank"] for r in raw)} for n in NAMES},
    }
    return raw, summary


def causal_batch(rows: list[dict[str, Any]], device: torch.device):
    width = max(len(r["token_ids"]) for r in rows) + 2
    x = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    y = torch.full((len(rows), width), -100, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        z = [2] + list(row["token_ids"]) + [3]
        x[i, :len(z)-1] = torch.tensor(z[:-1], dtype=torch.long, device=device)
        y[i, :len(z)-1] = torch.tensor(z[1:], dtype=torch.long, device=device)
    return x, y


def language_metrics(spec: dict[str, Any], fn, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    batch_losses, total_loss, count = [], 0.0, 0
    for start in range(0, 128, 32):
        x, y = causal_batch(rows[start:start+32], device)
        logits = fn(x).float()
        loss_sum = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1), ignore_index=-100, reduction="sum")
        n = int((y != -100).sum())
        batch_losses.append(float(loss_sum / n)); total_loss += float(loss_sum); count += n
    ce = statistics.fmean(batch_losses)
    return {"records": 128, "token_count": count, "established_batch_mean_ce": ce, "perplexity": math.exp(ce), "pooled_token_ce": total_loss / count, "batch_losses": batch_losses}


def validate_manual_graph(patcher: PathPatcher, device: torch.device) -> dict[str, Any]:
    probe = torch.tensor([[2, 10, 20, 30, 3], [2, 11, 21, 31, 3]], dtype=torch.long, device=device)
    result = {"tolerance": REPRO_TOL, "by_source_boundary": {}}
    for source in ("P", "S"):
        native = patcher.native(source, probe)
        for boundary in patcher.boundaries:
            manual = patcher.splice(probe, boundary["stage"], source, source, source)
            err = float((manual - native).abs().max())
            result["by_source_boundary"][f"{source}:{boundary['name']}"] = err
    result["max_abs_logit_error"] = max(result["by_source_boundary"].values())
    result["pass"] = result["max_abs_logit_error"] <= REPRO_TOL
    return result


def prior_baseline_reproduction(factual: dict[str, Any], d3: dict[str, Any], language: dict[str, Any]) -> dict[str, Any]:
    pf = read_json(COMPONENT / "FACTUAL_SUMMARY.json")
    pd = read_json(COMPONENT / "D3_SUMMARY.json")
    pl = read_json(COMPONENT / "LANGUAGE_SUMMARY.json")
    mapping = {"native_P": "PPP", "native_S": "SSS"}
    checks = {}
    for native, old in mapping.items():
        errors = {
            "factual_mean_sequence_margin": abs(factual[native]["overall"]["mean_sequence_margin"] - pf[old]["overall"]["mean_sequence_margin"]),
            "factual_owen_margin": abs(factual[native]["Owen-correct"]["mean_sequence_margin"] - pf[old]["Owen-correct"]["mean_sequence_margin"]),
            "d3_four_name_mass": abs(d3[native]["mean_four_name_mass"] - pd[old]["mean_four_name_probability"]),
            "d3_entropy": abs(d3[native]["mean_entropy"] - pd[old]["mean_entropy"]),
            "language_ce": abs(language[native]["established_batch_mean_ce"] - pl[old]["established_batch_mean_ce"]),
        }
        checks[native] = {"errors": errors, "max_abs_error": max(errors.values())}
    return {"tolerance": REPRO_TOL, "checks": checks, "pass": all(x["max_abs_error"] <= REPRO_TOL for x in checks.values())}


def curve_analysis(specs: list[dict[str, Any]], factual: dict[str, Any], d3: dict[str, Any], language: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "factual_overall_sequence_margin": lambda c: factual[c]["overall"]["mean_sequence_margin"],
        "factual_owen_sequence_margin": lambda c: factual[c]["Owen-correct"]["mean_sequence_margin"],
        "factual_alex_sequence_margin": lambda c: factual[c]["Alex-correct"]["mean_sequence_margin"],
        "d3_four_name_mass": lambda c: d3[c]["mean_four_name_mass"],
        "d3_alex_probability": lambda c: d3[c]["names"]["Alex"]["mean_probability"],
        "d3_owen_probability": lambda c: d3[c]["names"]["Owen"]["mean_probability"],
        "language_ce": lambda c: language[c]["established_batch_mean_ce"],
    }
    by_id = {s["id"]: s for s in specs}
    result: dict[str, Any] = {}
    for metric, get in metrics.items():
        p, s = get("native_P"), get("native_S")
        total = s - p
        entry: dict[str, Any] = {"native_P": p, "native_S": s, "native_change": total, "curves": {}}
        for direction, output in (("S_to_P", "P"), ("S_to_P", "S"), ("P_to_S", "P"), ("P_to_S", "S")):
            points = []
            prior_value = p if direction == "S_to_P" else s
            for boundary in [x for x in specs if x.get("kind") == "splice" and x.get("prefix") + "_to_" + x.get("suffix") == direction and x.get("output") == output]:
                value = get(boundary["id"])
                recovery = None if abs(total) < 1e-12 else (value - p) / total
                points.append({"boundary": boundary["boundary"], "stage": boundary["stage"], "value": value, "recovery_of_native_P_to_S_change": recovery, "increment_from_previous_boundary": value - prior_value})
                prior_value = value
            largest = max(points, key=lambda x: abs(x["increment_from_previous_boundary"]))
            majority = next((x["boundary"] for x in points if x["recovery_of_native_P_to_S_change"] is not None and x["recovery_of_native_P_to_S_change"] >= 0.5), None)
            entry["curves"][f"{direction}_{output}_output"] = {"points": points, "largest_absolute_increment": largest, "first_50pct_recovery_landmark": majority}
        result[metric] = entry
    return result


def classify(curves: dict[str, Any], factual: dict[str, Any]) -> dict[str, Any]:
    key_curve = "S_to_P_P_output"
    landmarks = {name: data["curves"][key_curve]["first_50pct_recovery_landmark"] for name, data in curves.items()}
    jumps = {name: data["curves"][key_curve]["largest_absolute_increment"]["boundary"] for name, data in curves.items()}
    effect_keys = ["factual_overall_sequence_margin", "factual_owen_sequence_margin", "d3_four_name_mass", "language_ce"]
    distinct_jumps = {jumps[k] for k in effect_keys}
    if len(distinct_jumps) >= 3:
        label = "SEPARABLE_BY_DEPTH"
    elif len(distinct_jumps) == 1:
        label = "COEMERGENT_BY_DEPTH"
    else:
        label = "DISTRIBUTED_OR_INTERACTIVE"
    return {
        "label": label,
        "basis": "Descriptive inference-time S-prefix/P-suffix curves with the Pilot1 output path; largest incremental transitions and 50% recovery landmarks are reported, not used as treatment gates.",
        "first_50pct_recovery_landmarks": landmarks,
        "largest_increment_boundaries": jumps,
        "factual_accuracy_curve": [{"boundary": s["boundary"], "sequence_correct": factual[s["id"]]["overall"]["sequence_correct"]} for s in []],
        "caution": "A splice boundary localizes where an inference-time state becomes sufficient in the complementary network. It does not show where learning occurred during optimization.",
    }


def report_text(arch: dict[str, Any], baseline: dict[str, Any], factual: dict[str, Any], d3: dict[str, Any], language: dict[str, Any], curves: dict[str, Any], classification: dict[str, Any]) -> str:
    key = "S_to_P_P_output"
    lines = [
        "# SF1 upstream localization forensic v1", "",
        "This diagnostic splices the residual stream between authoritative Pilot1 and SF1-update100 at the embedding output and after each attention and MLP residual sublayer. It is read-only and uses only the 16 SF1 training items, the frozen 256-position unrelated TinyStories D3 sample, and the authorized aligned first-128 TinyStories DEV records.", "",
        "## Execution graph", "",
        "The ordinary path is token plus position embeddings, eight pre-normalized transformer blocks (attention residual followed by MLP residual), final normalization, and an untied language head. The T13 wrapper's localization/retrieval route is present in both checkpoints but is invoked only by the specialized binding forward. It is inactive because every endpoint here calls `base_model` directly.", "",
        "## Integrity and baseline reproduction", "",
        f"Manual same-source execution reproduced native logits with maximum absolute error `{baseline['manual_graph']['max_abs_logit_error']:.3g}` (tolerance `{REPRO_TOL:g}`). Native endpoint summaries reproduced the prior component-swap forensic with maximum errors Pilot1 `{baseline['prior_endpoint']['checks']['native_P']['max_abs_error']:.3g}` and SF1 `{baseline['prior_endpoint']['checks']['native_S']['max_abs_error']:.3g}`.", "",
        "No optimizer was created, autograd was disabled, every parameter had `requires_grad=False`, no model state was saved, and both checkpoint file hashes were reverified after evaluation.", "",
        "## Native endpoints", "",
        "| endpoint | Pilot1 | SF1 |", "|---|---:|---:|",
        f"| factual sequence correct | {factual['native_P']['overall']['sequence_correct']}/16 | {factual['native_S']['overall']['sequence_correct']}/16 |",
        f"| factual mean sequence margin | {factual['native_P']['overall']['mean_sequence_margin']:.4f} | {factual['native_S']['overall']['mean_sequence_margin']:.4f} |",
        f"| Owen-correct mean sequence margin | {factual['native_P']['Owen-correct']['mean_sequence_margin']:.4f} | {factual['native_S']['Owen-correct']['mean_sequence_margin']:.4f} |",
        f"| unrelated four-name mass | {d3['native_P']['mean_four_name_mass']:.6f} | {d3['native_S']['mean_four_name_mass']:.6f} |",
        f"| aligned language CE | {language['native_P']['established_batch_mean_ce']:.4f} | {language['native_S']['established_batch_mean_ce']:.4f} |",
        "", "## Depth localization", "",
        "The table uses the cumulative `SF1 prefix -> Pilot1 suffix -> Pilot1 output` path. The 50% point is a descriptive landmark relative to the complete native Pilot1-to-SF1 shift; it is not a success gate.", "",
        "| endpoint | first 50% recovery | largest incremental transition |", "|---|---|---|",
    ]
    for metric in ("factual_overall_sequence_margin", "factual_owen_sequence_margin", "d3_four_name_mass", "d3_alex_probability", "d3_owen_probability", "language_ce"):
        c = curves[metric]["curves"][key]
        lines.append(f"| {metric} | {c['first_50pct_recovery_landmark'] or 'none'} | {c['largest_absolute_increment']['boundary']} ({c['largest_absolute_increment']['increment_from_previous_boundary']:+.6f}) |")
    lines += ["", f"Overall descriptive classification: `{classification['label']}`.", "", "The full bidirectional curves, both Pilot1 and SF1 output paths, raw factual rows, and raw D3 rows are preserved in the machine-readable artifacts.", "", "## Interpretation", ""]
    lines += [
        "- Useful factual signal is localized by the earliest cumulative boundary reported above, with the largest transition reported separately. Training-item evidence remains training-context evidence and does not establish relational generalization.",
        "- Alex/Owen suppression is tracked by separate Alex-correct and Owen-correct margins. A higher Owen margin without Owen correctness is evidence of increased signal that remains suppressed, not successful selection.",
        "- Broad name-prior pollution and aligned language damage are compared on exactly the prior authorized samples, so their depth profiles can be compared without changing evaluation material.",
        "- Divergent boundaries support inference-time separability. Coincident boundaries support coemergence only at the tested resolution. Gradual or direction-dependent curves support distributed interaction.",
        "", "## Established", "",
        f"- Native Pilot1 and SF1 endpoint behavior reproduced within tolerance.",
        f"- The inference-time localization label is `{classification['label']}` on these authorized endpoints.",
        "- T13 localization/retrieval modules do not participate in the ordinary-language forward path tested here.",
        "", "## Not established", "",
        "- These interventions do not locate where parameter learning occurred during training.",
        "- They do not establish held-out factual generalization, architectural incapacity, or a treatment choice.",
        "- They do not expose or score any locked SF1, FINAL, or sacred panel.",
        "", "## Smallest justified next action", "",
        "Use the localized boundary pattern to choose one further read-only diagnostic only if it discriminates the residual Alex/Owen suppression from the language/name-prior effects. No treatment is recommended or executed by this forensic.",
        "", "SF1_UPSTREAM_LOCALIZATION_FORENSIC_COMPLETE",
    ]
    return "\n".join(lines) + "\n"


def seal() -> dict[str, str]:
    excluded = {"SHA256SUMS.txt"}
    entries = [(sha(p), p.name) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in excluded]
    (OUT / "SHA256SUMS.txt").write_text("".join(f"{h}  {name}\n" for h, name in entries), encoding="utf-8", newline="\n")
    return {"manifest_sha256": sha(OUT / "SHA256SUMS.txt")}


def main() -> None:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    OUT.mkdir(parents=True, exist_ok=False) if not OUT.exists() else None
    input_hashes = verify_inputs()
    protocol = {
        "name": "SF1_UPSTREAM_LOCALIZATION_FORENSIC_V1",
        "authorized_inputs": {"factual": "exact 16 SF1 TRAIN items", "D3": "exact frozen prior 256 positions", "language": "authorized aligned first 128 TinyStories DEV records"},
        "boundaries": ["embedding_output"] + [x for i in range(8) for x in (f"after_block_{i}_attention_residual", f"after_block_{i}_mlp_residual")],
        "directions": ["Pilot1 prefix -> SF1 suffix", "SF1 prefix -> Pilot1 suffix"],
        "output_paths": ["Pilot1 final_norm+head", "SF1 final_norm+head"],
        "baseline_controls": ["native Pilot1", "native SF1", "same-source manual path at every boundary", "prior-forensic endpoint reproduction"],
        "candidate_scoring": "sum established four-token candidate conditional log probabilities; positive correct-minus-distractor margin is correct",
        "language_scoring": "aligned causal next-token CE on same four batches of 32 authorized DEV records",
        "forbidden": ["optimizer", "autograd", "training", "weight mutation", "locked SF1 panels", "FINAL", "sacred material", "treatment implementation"],
        "interpretive_rule": "inference-time causal sufficiency/localization only; never infer training-time site from splice depth",
    }
    write_json(OUT / "PROTOCOL.json", protocol)
    rt = load_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "authorized compatible GPU unavailable"
    with torch.inference_mode():
        pilot = rt.load_model(PARENT, device, BUNDLE)
        sf1 = rt.load_model(CHILD, device, BUNDLE)
        patcher = PathPatcher(pilot, sf1)
        arch = architecture(patcher)
        manual = validate_manual_graph(patcher, device)
        if not manual["pass"]:
            raise RuntimeError(f"manual graph reproduction failed: {manual}")
        specs = condition_specs(patcher)
        train = read_json(TRAIN_PATH)
        selection = read_json(SELECTION_PATH)
        dev = load_first_128_dev()
        name_ids = {name: list(ids) for name, ids in selection["name_tokenizations"].items()}
        factual_raw, d3_raw = [], []
        factual_sum, d3_sum, language_sum = {}, {}, {}
        localization_rows = []
        for index, spec in enumerate(specs, 1):
            fn = make_logits_fn(patcher, spec)
            fr, fs = factual_metrics(spec, fn, train, name_ids, device)
            dr, ds = d3_metrics(spec, fn, selection, device)
            ls = language_metrics(spec, fn, dev, device)
            factual_raw.extend(fr); d3_raw.extend(dr)
            factual_sum[spec["id"]] = fs; d3_sum[spec["id"]] = ds; language_sum[spec["id"]] = ls
            localization_rows.extend([
                {"condition": spec, "endpoint": "factual", "metrics": fs},
                {"condition": spec, "endpoint": "D3_unrelated_context", "metrics": ds},
                {"condition": spec, "endpoint": "aligned_language", "metrics": ls},
            ])
            print(f"[{index}/{len(specs)}] {spec['id']}", flush=True)
    prior = prior_baseline_reproduction(factual_sum, d3_sum, language_sum)
    baseline = {"manual_graph": manual, "prior_endpoint": prior, "pass": manual["pass"] and prior["pass"]}
    write_json(OUT / "BASELINE_REPRODUCTION.json", baseline)
    write_jsonl(OUT / "FACTUAL_ITEM_RESULTS.jsonl", factual_raw)
    write_jsonl(OUT / "D3_POSITION_RESULTS.jsonl", d3_raw)
    write_jsonl(OUT / "LOCALIZATION_RESULTS.jsonl", localization_rows)
    if not baseline["pass"]:
        write_json(OUT / "SUMMARY.json", {"status": "BASELINE_REPRODUCTION_FAILED"})
        write_json(OUT / "PROVENANCE.json", {"inputs": input_hashes, "architecture": arch})
        (OUT / "REPORT.md").write_text("# SF1 upstream localization forensic v1\n\nBaseline reproduction failed; hybrid results are not interpreted.\n\nSF1_READOUT_SELECTION_FORENSIC_NO ❤️ — USER ANALYSIS REQUIRED\n", encoding="utf-8", newline="\n")
        print(json.dumps(seal(), indent=2)); return
    curves = curve_analysis(specs, factual_sum, d3_sum, language_sum)
    classification = classify(curves, factual_sum)
    summary_obj = {"status": "SF1_UPSTREAM_LOCALIZATION_FORENSIC_COMPLETE", "classification": classification, "curves": curves, "native": {"factual": {"Pilot1": factual_sum["native_P"], "SF1": factual_sum["native_S"]}, "D3": {"Pilot1": d3_sum["native_P"], "SF1": d3_sum["native_S"]}, "language": {"Pilot1": language_sum["native_P"], "SF1": language_sum["native_S"]}}}
    write_json(OUT / "SUMMARY.json", summary_obj)
    post_hashes = {str(PARENT): sha(PARENT), str(CHILD): sha(CHILD), str(TOKENIZER_PATH): sha(TOKENIZER_PATH)}
    assert post_hashes[str(PARENT)] == PARENT_SHA and post_hashes[str(CHILD)] == CHILD_SHA and post_hashes[str(TOKENIZER_PATH)] == TOKENIZER_SHA
    provenance = {
        "input_hashes_before": input_hashes, "identity_hashes_after": post_hashes,
        "source_hashes": {str(p): sha(p) for p in [BUNDLE / "sources" / "treatment13_model.py", BUNDLE / "sources" / "hr3_block3_runtime.py", Path(r"C:\DaveLM-v0.9\v0_7\model.py"), Path(r"C:\DaveLM-v0.9\v0_8_2\model.py"), COMPONENT / "component_swap_forensic.py"]},
        "runtime": {"python": sys.version, "torch": torch.__version__, "tokenizers": __import__("tokenizers").__version__, "device": str(device)},
        "architecture": arch, "condition_count": len(specs), "boundary_count": len(patcher.boundaries),
        "optimizer_created": False, "autograd_used": False, "all_parameters_requires_grad_false": all(not p.requires_grad for w in patcher.wrappers.values() for p in w.parameters()),
        "checkpoint_or_dataset_mutation": False, "locked_or_sacred_access": False,
        "excluded_panels": ["SF1 held-out combinations", "alternate cues", "copy controls", "competing-name controls", "readiness FINAL", "sacred binding exam"],
    }
    write_json(OUT / "PROVENANCE.json", provenance)
    (OUT / "REPORT.md").write_text(report_text(arch, baseline, factual_sum, d3_sum, language_sum, curves, classification), encoding="utf-8", newline="\n")
    hashes = seal()
    print(json.dumps({"status": summary_obj["status"], "classification": classification["label"], **hashes}, indent=2))


if __name__ == "__main__":
    main()
