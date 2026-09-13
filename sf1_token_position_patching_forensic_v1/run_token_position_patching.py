"""Execute the frozen SF1 token-position patching forensic read-only."""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

SITE_PACKAGES = r"C:\DaveLM\.venv\Lib\site-packages"
if SITE_PACKAGES not in sys.path:
    sys.path.insert(0, SITE_PACKAGES)

import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
from tokenizers import Tokenizer  # type: ignore

ROOT = Path(r"C:\DaveLM-CADAVER")
OUT = ROOT / "sf1_token_position_patching_forensic_v1"
BUNDLE = ROOT / "single_fact_acquisition_sf1_seed87011"
PRIOR = ROOT / "sf1_readout_selection_forensic_v1"
UPSTREAM = ROOT / "sf1_upstream_localization_forensic_v1"
COMPONENT = ROOT / "sf1_component_swap_forensic_v1"
PARENT = ROOT / "language_pilot_1_early_block_protection_seed8380" / "pilot_run" / "checkpoints" / "seed_8380" / "latest.pt"
CHILD = ROOT / "single_fact_acquisition_sf1_seed87011_run" / "checkpoint_100.pt"
TOKENIZER_PATH = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
TRAIN_PATH = BUNDLE / "TRAIN.json"
DEV_PATH = BUNDLE / "data" / "ENGLISH_DEV.jsonl"
SELECTION_PATH = PRIOR / "D3_SELECTION.json"

PARENT_SHA = "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb"
CHILD_SHA = "550ce4306b450c2bdb64e57314e45ec7db5bb2ee68fc0978ed9d3617f4fc1e7e"
TOKENIZER_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
PROTOCOL_SHA = "e9bffb53616776cc2e9ad2695d893dee90152929ba55fd0ea6ff064a25805795"
INDEX_SHA = "7a530e2582692f1f7a7cdfd2f3ddacbaa265215a8d31ad11cec2ce8d14400730"
SAME_RUN_TOL = 1e-6
AGG_TOL = 1e-5
NAMES = ("Alex", "Owen", "Mia", "Nora")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")


def verify_preinference() -> dict[str, str]:
    assert sha(OUT / "PROTOCOL.json") == PROTOCOL_SHA
    assert sha(OUT / "TOKEN_INDEX_MAP.json") == INDEX_SHA
    receipt = read_json(OUT / "PREINFERENCE_RECEIPT.json")
    assert receipt["protocol_sha256"] == PROTOCOL_SHA and receipt["token_index_map_sha256"] == INDEX_SHA
    expected = {PARENT: PARENT_SHA, CHILD: CHILD_SHA, TOKENIZER_PATH: TOKENIZER_SHA}
    for path, wanted in expected.items():
        assert path.is_file() and sha(path) == wanted, path
    assert sha(TRAIN_PATH) == receipt["train_sha256"]
    assert sha(SELECTION_PATH) == "3ae0f6a748f5545b5d0af2fd89e5beb5482e9e42713ba61567d1fea353db2226"
    assert sha(DEV_PATH) == "2053a8036d7d0f02ba56ab28f739df95a1743760d0f8c980e112d6667b38104e"
    return {str(p): sha(p) for p in [PARENT, CHILD, TOKENIZER_PATH, TRAIN_PATH, SELECTION_PATH, DEV_PATH, OUT / "PROTOCOL.json", OUT / "TOKEN_INDEX_MAP.json"]}


def load_runtime():
    source = BUNDLE / "sources"
    sys.path.insert(0, str(source))
    import hr3_block3_runtime as rt  # type: ignore
    return rt


def load_first_128_dev():
    rows = []
    with DEV_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
            if len(rows) == 128: break
    assert len(rows) == 128
    return rows


class PatchEngine:
    def __init__(self, pilot, sf1):
        self.wrappers = {"P": pilot, "S": sf1}
        self.models = {"P": pilot.base_model, "S": sf1.base_model}
        for wrapper in self.wrappers.values():
            wrapper.eval()
            for param in wrapper.parameters(): param.requires_grad_(False)

    @staticmethod
    def embed(base, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return base.embedding_dropout(base.token_embedding(tokens) + base.position_embedding(positions))

    @staticmethod
    def stage(base, x, index):
        block = base.blocks[index // 2]
        if index % 2 == 0:
            return x + block.attention_residual_dropout(block.attention(block.norm1(x)))
        return x + block.feed_forward_residual_dropout(block.feed_forward(block.norm2(x)))

    def prefix(self, source, tokens, through_stage):
        x = self.embed(self.models[source], tokens)
        for stage in range(through_stage + 1): x = self.stage(self.models[source], x, stage)
        return x

    def suffix(self, source, x, after_stage):
        for stage in range(after_stage + 1, 16): x = self.stage(self.models[source], x, stage)
        return x

    def output(self, source, x):
        base = self.models[source]
        return base.language_head(base.final_norm(x))

    def native(self, source, tokens):
        return self.models[source](tokens)

    def patch(self, tokens, stage, recipient, donor, output, mask):
        assert mask.shape == tokens.shape and mask.dtype == torch.bool
        recipient_state = self.prefix(recipient, tokens, stage)
        donor_state = self.prefix(donor, tokens, stage)
        mixed = torch.where(mask.unsqueeze(-1), donor_state, recipient_state)
        return self.output(output, self.suffix(recipient, mixed, stage))


def condition_specs(protocol):
    factual, general = [], []
    for site in protocol["sites"]:
        for group in protocol["factual_position_groups"]:
            for recipient, donor, natural_output, alternate_output in (("P", "S", "P", "S"), ("S", "P", "S", "P")):
                for output in (natural_output, alternate_output):
                    factual.append({"id": f"{site['name']}__{donor}_into_{recipient}__{group}__{output}_output", "site": site["name"], "stage": site["stage"], "recipient": recipient, "donor": donor, "group": group, "output": output, "natural_output": output == recipient})
        for group in protocol["D3_language_position_groups"]:
            for recipient, donor, natural_output, alternate_output in (("P", "S", "P", "S"), ("S", "P", "S", "P")):
                for output in (natural_output, alternate_output):
                    general.append({"id": f"{site['name']}__{donor}_into_{recipient}__{group}__{output}_output", "site": site["name"], "stage": site["stage"], "recipient": recipient, "donor": donor, "group": group, "output": output, "natural_output": output == recipient})
    return factual, general


def native_spec(source):
    return {"id": f"native_{source}", "kind": "native", "source": source}


def factual_mask(meta, group, length, device):
    mask = torch.zeros(length, dtype=torch.bool, device=device)
    p = int(meta["prompt_token_count"])
    static = meta["model_positions_with_bos"]
    if group in static:
        positions = static[group]
    elif group == "answer_generation_path":
        positions = range(p, length)
    elif group == "all_positions":
        positions = range(length)
    else:
        raise KeyError(group)
    for pos in positions:
        if 0 <= int(pos) < length: mask[int(pos)] = True
    return mask


def general_mask(lengths, width, group, device):
    mask = torch.zeros((len(lengths), width), dtype=torch.bool, device=device)
    for i, length in enumerate(lengths):
        if group == "final_predictive": mask[i, length - 1] = True
        elif group == "all_prior_context": mask[i, :max(0, length - 1)] = True
        elif group == "all_positions": mask[i, :length] = True
        else: raise KeyError(group)
    return mask


def rank(logits, token_id):
    return int((logits > logits[token_id]).sum().item()) + 1


def variable_forward(payloads, fn, device):
    groups = collections.defaultdict(list)
    for i, payload in enumerate(payloads): groups[len(payload["tokens"])].append(i)
    out = [None] * len(payloads)
    for length, indexes in sorted(groups.items()):
        x = torch.tensor([payloads[i]["tokens"] for i in indexes], dtype=torch.long, device=device)
        masks = torch.stack([payloads[i]["mask"](length, device) for i in indexes])
        logits = fn(x, masks).float().cpu()
        for j, original in enumerate(indexes): out[original] = logits[j]
    return out


def factual_fn(engine, spec, group):
    if spec.get("kind") == "native": return lambda x, m: engine.native(spec["source"], x)
    return lambda x, m: engine.patch(x, spec["stage"], spec["recipient"], spec["donor"], spec["output"], m)


def factual_eval(engine, spec, group, rows, maps, device):
    fn = factual_fn(engine, spec, group)
    payloads = []
    for row in rows:
        meta = maps[row["id"]]
        payloads.append({"tokens": [2] + row["prompt_token_ids"], "mask": lambda n, d, m=meta, g=group: factual_mask(m, g, n, d)})
    prompt_logits = variable_forward(payloads, fn, device)
    cpayloads, keys = [], []
    for i, row in enumerate(rows):
        meta = maps[row["id"]]
        for c, candidate in enumerate(row["candidate_token_ids"]):
            cpayloads.append({"tokens": [2] + row["prompt_token_ids"] + candidate, "mask": lambda n, d, m=meta, g=group: factual_mask(m, g, n, d)})
            keys.append((i, c))
    candidate_logits = variable_forward(cpayloads, fn, device)
    cmap = {k: v for k, v in zip(keys, candidate_logits)}
    generated = greedy_batch(engine, spec, group, rows, maps, device)
    name_ids = {name: read_json(SELECTION_PATH)["name_tokenizations"][name][0] for name in NAMES}
    raw = []
    for i, row in enumerate(rows):
        first = prompt_logits[i][-1]; p = first.softmax(-1); prompt_len = len(row["prompt_token_ids"])
        scores = []
        for c, candidate in enumerate(row["candidate_token_ids"]):
            lp = cmap[(i, c)].log_softmax(-1)
            scores.append(sum(float(lp[prompt_len + j, int(t)]) for j, t in enumerate(candidate)))
        correct = int(row["correct_index"]); wrong = 1 - correct
        correct_id = int(row["candidate_token_ids"][correct][0]); wrong_id = int(row["candidate_token_ids"][wrong][0])
        expected = list(row["candidate_token_ids"][correct]) + [3]
        raw.append({
            "condition": spec["id"], "site": spec.get("site"), "group": group, "id": row["id"], "pair_id": row["id"].rsplit(":", 1)[0], "actor": row["actor"], "candidates": row["candidates"], "correct_index": correct,
            "correct_first_logit": float(first[correct_id]), "distractor_first_logit": float(first[wrong_id]), "first_margin": float(first[correct_id] - first[wrong_id]),
            "correct_first_probability": float(p[correct_id]), "distractor_first_probability": float(p[wrong_id]), "correct_first_rank": rank(first, correct_id), "distractor_first_rank": rank(first, wrong_id),
            "candidate_scores": scores, "sequence_margin": scores[correct] - scores[wrong], "candidate_mass": float(p[correct_id] + p[wrong_id]), "four_name_mass": float(sum(p[int(t)] for t in name_ids.values())),
            "full_vocab_top1": int(first.argmax()), "eos_probability": float(p[3]), "eos_rank": rank(first, 3),
            "generated_token_ids": generated[i], "exact_answer_eos": generated[i] == expected,
        })
    return raw, factual_summary(raw)


def greedy_batch(engine, spec, group, rows, maps, device, limit=32):
    generated = [[] for _ in rows]
    active = list(range(len(rows)))
    for _ in range(limit):
        if not active: break
        length_groups = collections.defaultdict(list)
        for i in active: length_groups[1 + len(rows[i]["prompt_token_ids"]) + len(generated[i])].append(i)
        next_active = []
        for _, indexes in sorted(length_groups.items()):
            seqs = [[2] + rows[i]["prompt_token_ids"] + generated[i] for i in indexes]
            x = torch.tensor(seqs, dtype=torch.long, device=device)
            masks = torch.stack([factual_mask(maps[rows[i]["id"]], group, x.shape[1], device) for i in indexes])
            if spec.get("kind") == "native": logits = engine.native(spec["source"], x)
            else: logits = engine.patch(x, spec["stage"], spec["recipient"], spec["donor"], spec["output"], masks)
            tokens = logits[:, -1].argmax(-1).tolist()
            for i, token in zip(indexes, tokens):
                generated[i].append(int(token))
                if token != 3: next_active.append(i)
        active = next_active
    return generated


def factual_summary(rows):
    def is_group(r, group):
        pair = {x.strip().rstrip(".") for x in r["candidates"]}
        return group == "overall" or (group == "Alex/Owen" and pair == {"Alex", "Owen"}) or (group == "Mia/Nora" and pair == {"Mia", "Nora"}) or (group == "Alex-correct" and r["actor"] == "Alex") or (group == "Owen-correct" and r["actor"] == "Owen")
    result = {}
    for group in ("overall", "Alex/Owen", "Mia/Nora", "Alex-correct", "Owen-correct"):
        xs = [r for r in rows if is_group(r, group)]
        result[group] = {"n": len(xs), "sequence_correct": sum(r["sequence_margin"] > 0 for r in xs), "first_correct": sum(r["first_margin"] > 0 for r in xs), "exact_answer_eos": sum(r["exact_answer_eos"] for r in xs), "mean_sequence_margin": statistics.fmean(r["sequence_margin"] for r in xs), "mean_first_margin": statistics.fmean(r["first_margin"] for r in xs), "mean_candidate_mass": statistics.fmean(r["candidate_mass"] for r in xs), "mean_four_name_mass": statistics.fmean(r["four_name_mass"] for r in xs)}
    pairs = collections.defaultdict(list)
    for row in rows: pairs[row["pair_id"]].append(row)
    pair_rows = []
    for pair_id, xs in sorted(pairs.items()):
        actors = {x["actor"] for x in xs}; label = "Alex/Owen" if actors == {"Alex", "Owen"} else "Mia/Nora"
        pair_rows.append({"pair_id": pair_id, "group": label, "both_correct": all(x["sequence_margin"] > 0 for x in xs), "signed_margin_sum": sum(x["sequence_margin"] for x in xs)})
    alex = result["Alex-correct"]["mean_sequence_margin"]; owen = result["Owen-correct"]["mean_sequence_margin"]
    result["pair_structure"] = {"complete": sum(x["both_correct"] for x in pair_rows), "total": 8, "Alex/Owen_complete": sum(x["both_correct"] for x in pair_rows if x["group"] == "Alex/Owen"), "Mia/Nora_complete": sum(x["both_correct"] for x in pair_rows if x["group"] == "Mia/Nora"), "alex_over_owen_default_component": (alex - owen) / 2, "alex_owen_contextual_separation_component": (alex + owen) / 2, "pairs": pair_rows}
    return result


def general_fn(engine, spec):
    if spec.get("kind") == "native": return lambda x, m: engine.native(spec["source"], x)
    return lambda x, m: engine.patch(x, spec["stage"], spec["recipient"], spec["donor"], spec["output"], m)


def d3_eval(engine, spec, group, selection, device):
    fn = general_fn(engine, spec); positions = selection["positions"]
    payloads = []
    for item in positions:
        tokens = [2] + item["prefix_token_ids"]
        payloads.append({"tokens": tokens, "mask": lambda n, d, g=group: general_mask([n], n, g, d)[0]})
    logits_all = variable_forward(payloads, fn, device)
    name_ids = {n: int(ids[0]) for n, ids in selection["name_tokenizations"].items()}
    raw = []
    for item, all_logits in zip(positions, logits_all):
        logits = all_logits[-1]; lp = logits.log_softmax(-1); p = lp.exp(); target = int(item["target_token_id"])
        names = {n: {"logit": float(logits[t]), "probability": float(p[t]), "rank": rank(logits, t)} for n, t in name_ids.items()}
        raw.append({"condition": spec["id"], "site": spec.get("site"), "group": group, "position_id": item["position_id"], "names": names, "four_name_mass": float(sum(p[t] for t in name_ids.values())), "entropy": float(-(p * lp).sum()), "eos_probability": float(p[3]), "eos_rank": rank(logits, 3), "true_next_probability": float(p[target]), "true_next_rank": rank(logits, target)})
    summary = {"n": 256, "mean_four_name_mass": statistics.fmean(r["four_name_mass"] for r in raw), "mean_entropy": statistics.fmean(r["entropy"] for r in raw), "mean_eos_probability": statistics.fmean(r["eos_probability"] for r in raw), "mean_true_next_probability": statistics.fmean(r["true_next_probability"] for r in raw), "mean_true_next_rank": statistics.fmean(r["true_next_rank"] for r in raw), "names": {n: {"mean_logit": statistics.fmean(r["names"][n]["logit"] for r in raw), "mean_probability": statistics.fmean(r["names"][n]["probability"] for r in raw), "mean_rank": statistics.fmean(r["names"][n]["rank"] for r in raw)} for n in NAMES}}
    return raw, summary


def causal_batch(rows, device):
    width = max(len(r["token_ids"]) for r in rows) + 1
    x = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    y = torch.full((len(rows), width), -100, dtype=torch.long, device=device)
    lengths = []
    for i, row in enumerate(rows):
        z = [2] + row["token_ids"] + [3]; inp = z[:-1]; target = z[1:]
        x[i, :len(inp)] = torch.tensor(inp, device=device); y[i, :len(target)] = torch.tensor(target, device=device); lengths.append(len(inp))
    return x, y, lengths


def language_eval(engine, spec, group, rows, device):
    fn = general_fn(engine, spec); batch_losses, total, count = [], 0.0, 0
    for start in range(0, 128, 32):
        x, y, lengths = causal_batch(rows[start:start+32], device)
        mask = general_mask(lengths, x.shape[1], group, device)
        logits = fn(x, mask).float()
        loss_sum = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1), ignore_index=-100, reduction="sum")
        n = int((y != -100).sum()); loss = float(loss_sum / n)
        batch_losses.append(loss); total += float(loss_sum); count += n
    ce = statistics.fmean(batch_losses)
    return {"records": 128, "token_count": count, "established_batch_mean_ce": ce, "perplexity": math.exp(ce), "pooled_token_ce": total / count, "batch_losses": batch_losses}


def same_source_integrity(engine, protocol, rows, maps, selection, dev, device):
    checks = []; max_error = 0.0
    frow = rows[0]; meta = maps[frow["id"]]; ftokens = torch.tensor([[2] + frow["prompt_token_ids"]], device=device)
    drow = selection["positions"][0]; dtokens = torch.tensor([[2] + drow["prefix_token_ids"]], device=device)
    lx, _, lengths = causal_batch(dev[:2], device)
    for site in protocol["sites"]:
        for source in ("P", "S"):
            native_f = engine.native(source, ftokens)
            for group in protocol["factual_position_groups"]:
                mask = factual_mask(meta, group, ftokens.shape[1], device).unsqueeze(0)
                patched = engine.patch(ftokens, site["stage"], source, source, source, mask)
                err = float((patched - native_f).abs().max()); max_error = max(max_error, err); checks.append({"site": site["name"], "source": source, "panel": "factual", "group": group, "max_abs_error": err})
            native_d = engine.native(source, dtokens)
            for group in protocol["D3_language_position_groups"]:
                mask = general_mask([dtokens.shape[1]], dtokens.shape[1], group, device)
                patched = engine.patch(dtokens, site["stage"], source, source, source, mask)
                err = float((patched - native_d).abs().max()); max_error = max(max_error, err); checks.append({"site": site["name"], "source": source, "panel": "D3", "group": group, "max_abs_error": err})
                lmask = general_mask(lengths, lx.shape[1], group, device)
                native_l = engine.native(source, lx); patched_l = engine.patch(lx, site["stage"], source, source, source, lmask)
                err = float((patched_l - native_l).abs().max()); max_error = max(max_error, err); checks.append({"site": site["name"], "source": source, "panel": "language", "group": group, "max_abs_error": err})
    return {"tolerance": SAME_RUN_TOL, "max_abs_logit_error": max_error, "pass": max_error <= SAME_RUN_TOL, "checks": checks}


def prior_native_reproduction(factual, d3, language):
    up = read_json(UPSTREAM / "SUMMARY.json")["native"]
    errors = {}
    for new, old in (("native_P", "Pilot1"), ("native_S", "SF1")):
        errors[new] = {
            "factual_margin": abs(factual[new]["overall"]["mean_sequence_margin"] - up["factual"][old]["overall"]["mean_sequence_margin"]),
            "factual_owen": abs(factual[new]["Owen-correct"]["mean_sequence_margin"] - up["factual"][old]["Owen-correct"]["mean_sequence_margin"]),
            "d3_mass": abs(d3[new]["mean_four_name_mass"] - up["D3"][old]["mean_four_name_mass"]),
            "language_ce": abs(language[new]["established_batch_mean_ce"] - up["language"][old]["established_batch_mean_ce"]),
        }
        errors[new]["max_abs_error"] = max(errors[new].values())
    return {"tolerance": AGG_TOL, "errors": errors, "pass": all(x["max_abs_error"] <= AGG_TOL for x in errors.values())}


def all_positions_reproduction(factual, d3, language, specs):
    rows = read_jsonl(UPSTREAM / "LOCALIZATION_RESULTS.jsonl")
    upstream = {(r["condition"]["id"], r["endpoint"]): r["metrics"] for r in rows}
    checks = []
    for spec in specs:
        if spec["group"] != "all_positions": continue
        direction = f"{spec['donor']}_to_{spec['recipient']}"
        old_id = f"{spec['site']}__{direction}__{spec['output']}_output"
        old_f = upstream[(old_id, "factual")]; old_d = upstream[(old_id, "D3_unrelated_context")]; old_l = upstream[(old_id, "aligned_language")]
        err = max(abs(factual[spec["id"]]["overall"]["mean_sequence_margin"] - old_f["overall"]["mean_sequence_margin"]), abs(d3[spec["id"]]["mean_four_name_mass"] - old_d["mean_four_name_mass"]), abs(language[spec["id"]]["established_batch_mean_ce"] - old_l["established_batch_mean_ce"]))
        checks.append({"condition": spec["id"], "upstream_condition": old_id, "max_aggregate_error": err})
    return {"tolerance": AGG_TOL, "max_aggregate_error": max(x["max_aggregate_error"] for x in checks), "pass": all(x["max_aggregate_error"] <= AGG_TOL for x in checks), "checks": checks}


def recovery(value, recipient, donor):
    den = donor - recipient
    return None if abs(den) < 1e-12 else (value - recipient) / den


def summarize_positions(factual, d3, language, factual_specs, general_specs):
    result = {"factual": [], "D3": [], "language": []}
    p_f, s_f = factual["native_P"], factual["native_S"]
    for spec in factual_specs:
        x = factual[spec["id"]]; recipient = p_f if spec["recipient"] == "P" else s_f; donor = s_f if spec["donor"] == "S" else p_f
        result["factual"].append({"condition": spec, "sequence_correct": x["overall"]["sequence_correct"], "exact_answer_eos": x["overall"]["exact_answer_eos"], "complete_pairs": x["pair_structure"]["complete"], "AO_complete_pairs": x["pair_structure"]["Alex/Owen_complete"], "MN_complete_pairs": x["pair_structure"]["Mia/Nora_complete"], "mean_margin": x["overall"]["mean_sequence_margin"], "owen_margin": x["Owen-correct"]["mean_sequence_margin"], "alex_default": x["pair_structure"]["alex_over_owen_default_component"], "context_separation": x["pair_structure"]["alex_owen_contextual_separation_component"], "margin_recovery": recovery(x["overall"]["mean_sequence_margin"], recipient["overall"]["mean_sequence_margin"], donor["overall"]["mean_sequence_margin"])})
    p_d, s_d = d3["native_P"], d3["native_S"]; p_l, s_l = language["native_P"], language["native_S"]
    for spec in general_specs:
        x = d3[spec["id"]]; recipient = p_d if spec["recipient"] == "P" else s_d; donor = s_d if spec["donor"] == "S" else p_d
        result["D3"].append({"condition": spec, "four_name_mass": x["mean_four_name_mass"], "mass_recovery": recovery(x["mean_four_name_mass"], recipient["mean_four_name_mass"], donor["mean_four_name_mass"]), "name_probabilities": {n: x["names"][n]["mean_probability"] for n in NAMES}})
        y = language[spec["id"]]; recipient_l = p_l if spec["recipient"] == "P" else s_l; donor_l = s_l if spec["donor"] == "S" else p_l
        result["language"].append({"condition": spec, "ce": y["established_batch_mean_ce"], "delta_from_Pilot1": y["established_batch_mean_ce"] - p_l["established_batch_mean_ce"], "delta_from_SF1": y["established_batch_mean_ce"] - s_l["established_batch_mean_ce"], "ce_recovery": recovery(y["established_batch_mean_ce"], recipient_l["established_batch_mean_ce"], donor_l["established_batch_mean_ce"])})
    return result


def main() -> None:
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    input_hashes = verify_preinference()
    protocol = read_json(OUT / "PROTOCOL.json"); index = read_json(OUT / "TOKEN_INDEX_MAP.json")
    maps = {x["id"]: x for x in index["records"]}; rows = read_json(TRAIN_PATH); selection = read_json(SELECTION_PATH); dev = load_first_128_dev()
    factual_specs, general_specs = condition_specs(protocol)
    rt = load_runtime(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); assert device.type == "cuda"
    with torch.inference_mode():
        pilot = rt.load_model(PARENT, device, BUNDLE); sf1 = rt.load_model(CHILD, device, BUNDLE); engine = PatchEngine(pilot, sf1)
        integrity = same_source_integrity(engine, protocol, rows, maps, selection, dev, device)
        if not integrity["pass"]: raise RuntimeError("same-source patch integrity failed")
        factual_raw, d3_raw, position_rows = [], [], []
        factual, d3, language = {}, {}, {}
        for source in ("P", "S"):
            spec = native_spec(source); group = "all_positions"
            fr, fs = factual_eval(engine, spec, group, rows, maps, device); dr, ds = d3_eval(engine, spec, group, selection, device); ls = language_eval(engine, spec, group, dev, device)
            factual_raw += fr; d3_raw += dr; factual[spec["id"]] = fs; d3[spec["id"]] = ds; language[spec["id"]] = ls
            position_rows += [{"endpoint": "factual", "condition": spec, "metrics": fs}, {"endpoint": "D3", "condition": spec, "metrics": ds}, {"endpoint": "language", "condition": spec, "metrics": ls}]
            print(spec["id"], flush=True)
        native_repro = prior_native_reproduction(factual, d3, language)
        if not native_repro["pass"]: raise RuntimeError(f"native reproduction failed: {native_repro}")
        for i, spec in enumerate(factual_specs, 1):
            fr, fs = factual_eval(engine, spec, spec["group"], rows, maps, device); factual_raw += fr; factual[spec["id"]] = fs; position_rows.append({"endpoint": "factual", "condition": spec, "metrics": fs})
            if i % 20 == 0: print(f"factual {i}/{len(factual_specs)}", flush=True)
        for i, spec in enumerate(general_specs, 1):
            dr, ds = d3_eval(engine, spec, spec["group"], selection, device); ls = language_eval(engine, spec, spec["group"], dev, device)
            d3_raw += dr; d3[spec["id"]] = ds; language[spec["id"]] = ls
            position_rows += [{"endpoint": "D3", "condition": spec, "metrics": ds}, {"endpoint": "language", "condition": spec, "metrics": ls}]
            if i % 12 == 0: print(f"general {i}/{len(general_specs)}", flush=True)
    allpos = all_positions_reproduction(factual, d3, language, [s for s in factual_specs if s["group"] == "all_positions"])
    baseline = {"same_source": integrity, "native_prior": native_repro, "all_positions_upstream_forensic": allpos, "pass": integrity["pass"] and native_repro["pass"] and allpos["pass"]}
    write_json(OUT / "BASELINE_REPRODUCTION.json", baseline); write_jsonl(OUT / "FACTUAL_ITEM_RESULTS.jsonl", factual_raw); write_jsonl(OUT / "D3_POSITION_RESULTS.jsonl", d3_raw); write_jsonl(OUT / "POSITION_PATCH_RESULTS.jsonl", position_rows)
    if not baseline["pass"]:
        write_json(OUT / "SUMMARY.json", {"status": "BASELINE_OR_SPLICE_INTEGRITY_FAILED"})
        (OUT / "REPORT.md").write_text("# SF1 token-position patching forensic v1\n\nBaseline or splice integrity failed. Hybrid results are not interpreted.\n", encoding="utf-8", newline="\n")
    else:
        compact = summarize_positions(factual, d3, language, factual_specs, general_specs)
        write_json(OUT / "SUMMARY.json", {"status": "COMPLETE_PENDING_INTERPRETATION", "native": {"factual": {"Pilot1": factual["native_P"], "SF1": factual["native_S"]}, "D3": {"Pilot1": d3["native_P"], "SF1": d3["native_S"]}, "language": {"Pilot1": language["native_P"], "SF1": language["native_S"]}}, "position_effects": compact})
        (OUT / "REPORT.md").write_text("# SF1 token-position patching forensic v1\n\nMachine-readable diagnostic complete; interpretation finalization pending.\n", encoding="utf-8", newline="\n")
    post = {str(PARENT): sha(PARENT), str(CHILD): sha(CHILD), str(TOKENIZER_PATH): sha(TOKENIZER_PATH)}
    assert post[str(PARENT)] == PARENT_SHA and post[str(CHILD)] == CHILD_SHA and post[str(TOKENIZER_PATH)] == TOKENIZER_SHA
    write_json(OUT / "PROVENANCE.json", {"inputs_before": input_hashes, "identities_after": post, "runtime": {"python": sys.version, "torch": torch.__version__, "tokenizers": __import__("tokenizers").__version__, "device": str(device)}, "condition_counts": {"factual_hybrids": len(factual_specs), "general_hybrids": len(general_specs), "native": 2}, "optimizer_created": False, "autograd_used": False, "all_parameters_requires_grad_false": all(not p.requires_grad for w in engine.wrappers.values() for p in w.parameters()), "weight_or_dataset_mutation": False, "locked_final_sacred_access": False})
    print(json.dumps({"baseline_pass": baseline["pass"], "factual_conditions": len(factual), "general_conditions": len(d3)}, indent=2))


if __name__ == "__main__":
    main()
