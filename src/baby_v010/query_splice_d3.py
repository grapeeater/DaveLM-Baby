"""D3: early vs late residual splice on the hashed parent.

Protocol: `design/V010_QUERY_SPLICE_D3.md`. Read-only. No optimizer.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from .query_presence import N_LAYERS, median
from .selection_s1 import PARENT, PARENT_SHA, digest, pack, write

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/query_splice_d3"
PROTOCOL = ROOT / "design/V010_QUERY_SPLICE_D3.md"
S2_DIAGNOSTIC = ROOT / "runs/selection_s2/DIAGNOSTIC.json"
P1_TREAT = ROOT / "runs/selection_p1/treatment_150001/checkpoint_16800.pt"
P1_TREAT_SHA = "5c8296ef759543dd608c18b8048ccce4bd4fcf8de7a6183b2d84e5f53d4fbafa"
D2_ADJUDICATION = ROOT / "runs/query_presence_d2/ADJUDICATION.json"

STEER_SCALE = 10.0
STEER_MIN = 0.95
LIFT = 0.10
NEG_GOLD_MAX = 0.03
LONG_GAP = 13
DEBUG_LOG = ROOT / "debug-145edc.log"
SESSION = "145edc"


def posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    payload = {
        "sessionId": SESSION,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with DEBUG_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    # #endregion


def splice_sources(item: dict) -> dict[str, int | None]:
    from .isolation_transforms import recover_rendered_pairs

    rendered = recover_rendered_pairs(item)
    query_key = int(item["query_key"])
    match_key = None
    competitor_key = None
    for row in rendered:
        start = int(row["start"])
        is_match = int(row["original_key"]) == query_key or int(row["key"]) == query_key
        if is_match and match_key is None:
            match_key = start
        elif (not is_match) and competitor_key is None:
            competitor_key = start
    return {
        "query": int(item["query_position"]),
        "match_key": match_key,
        "competitor_key": competitor_key,
    }


def eligible_item(item: dict) -> bool:
    if item.get("kind") != "keyed":
        return False
    if "query_position" not in item:
        return False
    sources = splice_sources(item)
    gen = len(item["input"]) - 1
    if sources["match_key"] is None or sources["competitor_key"] is None:
        return False
    if sources["match_key"] == sources["query"]:
        return False
    if not (0 <= sources["query"] <= gen):
        return False
    if not (0 <= sources["match_key"] <= gen):
        return False
    if not (0 <= sources["competitor_key"] <= gen):
        return False
    return True


def candidate_argmax(logits_row, heads: list[int]) -> int:
    import torch

    values = logits_row[heads]
    return int(torch.argmax(values).item())


def capture_blocks(model, tokens):
    captured = [None] * len(model.blocks)
    handles = []
    for layer, block in enumerate(model.blocks):
        def hook(_module, _inputs, output, layer=layer):
            captured[layer] = output.detach()
            return output

        handles.append(block.register_forward_hook(hook))
    try:
        logits = model(tokens)
    finally:
        for handle in handles:
            handle.remove()
    return logits, captured


def splice_block(model, tokens, gen_pos: int, layer: int, vec, mix: bool):
    import torch

    def hook(_module, _inputs, output):
        cloned = output.clone()
        current = cloned[:, gen_pos]
        source = vec.to(cloned.dtype)
        cloned[:, gen_pos] = 0.5 * current + 0.5 * source if mix else source
        return cloned

    handle = model.blocks[layer].register_forward_hook(hook)
    try:
        return model(tokens)
    finally:
        handle.remove()
        del torch


def embed_add(model, tokens, gen_pos: int, query_key: int):
    import torch

    query_vec = model.token_embedding.weight[query_key].detach()

    def pre_hook(_module, inputs):
        hidden = inputs[0].clone()
        hidden[:, gen_pos] = hidden[:, gen_pos] + query_vec.to(hidden.dtype)
        return (hidden,)

    handle = model.blocks[0].register_forward_pre_hook(pre_hook)
    try:
        return model(tokens)
    finally:
        handle.remove()
        del torch


def logit_steer(model, tokens, gen_pos: int, gold_token: int, scale: float = STEER_SCALE):
    hidden = model.forward_hidden(tokens).clone()
    hidden[:, gen_pos] = hidden[:, gen_pos] + scale * model.language_head.weight[gold_token]
    return model.language_head(hidden)


def load_ckpt(path, device: str):
    import torch

    from .config import BabyVNextConfig
    from .model import BabyVNextLM

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("protected_material_opened"):
        raise RuntimeError("refusing a checkpoint that recorded protected access")
    config = BabyVNextConfig.from_dict(blob["config"])
    model = BabyVNextLM(config)
    model.load_state_dict(blob["model_state_dict"])
    model.to(device)
    model.eval()
    return model, blob


def layer_query_masses(model, items, device: str) -> dict:
    import torch

    model.set_attention_backend("reference")
    buckets = defaultdict(list)
    argmax_layers = []
    with torch.no_grad():
        for item in items:
            captured: list = []
            model.set_attention_capture(captured)
            tokens = torch.tensor([item["input"]], dtype=torch.long, device=device)
            model(tokens)
            model.set_attention_capture(None)
            if len(captured) != N_LAYERS:
                raise RuntimeError("expected one attention map per layer")
            gen = len(item["input"]) - 1
            query = int(item["query_position"])
            masses = [float(w[0, :, gen, query].max().item()) for w in captured]
            best = max(range(len(masses)), key=lambda i: masses[i])
            argmax_layers.append(best)
            for layer, mass in enumerate(masses):
                buckets[layer].append(mass)
    model.set_attention_backend("sdpa")
    return {
        "n": len(items),
        "median_argmax_layer": median([float(x) for x in argmax_layers]),
        "mean_argmax_layer": (sum(argmax_layers) / len(argmax_layers)) if argmax_layers else float("nan"),
        "per_layer_median_max_head_query_mass": [median(buckets[i]) for i in range(N_LAYERS)],
    }


def summarize_rows(rows: list[dict], field: str) -> dict:
    if not rows:
        return {"n": 0, "hit": 0, "accuracy": float("nan"), "chance": float("nan"), "excess": float("nan")}
    hit = sum(int(row[field]) for row in rows)
    chance = sum(1.0 / row["K"] for row in rows) / len(rows)
    acc = hit / len(rows)
    return {
        "n": len(rows),
        "hit": hit,
        "accuracy": acc,
        "chance": chance,
        "excess": acc - chance,
        "emit_query_rate": sum(int(row["emit_query"]) for row in rows) / len(rows),
        "competitor_rate": sum(int(row["pick_competitor"]) for row in rows) / len(rows),
    }


def delta(cell: dict, baseline: dict) -> float:
    return float(cell["accuracy"] - baseline["accuracy"])


def decide(cells: dict, steer_miss_acc: float, n_miss: int) -> dict:
    if n_miss < 20 or steer_miss_acc < STEER_MIN:
        return {
            "verdict": "INVALID",
            "instrument_ok": False,
            "negative_ok": False,
            "early_query": None,
            "late_query": None,
            "match_key": None,
            "c11r_gold": None,
        }
    as_is = cells["as_is_long"]
    early = max(
        delta(cells[name], as_is)
        for name in ("Q0R", "Q1R", "Q2R", "Q0M", "Q1M", "QE")
    )
    late = max(delta(cells[name], as_is) for name in ("Q11R", "Q11M"))
    match = max(delta(cells[name], as_is) for name in ("K0R", "K11R"))
    c11 = delta(cells["C11R"], as_is)
    negative_ok = c11 <= NEG_GOLD_MAX
    if not negative_ok:
        verdict = "MIXED"
    elif early >= LIFT and late < LIFT:
        verdict = "EARLY_QUERY"
    elif late >= LIFT and early < LIFT:
        verdict = "LATE_QUERY"
    elif early >= LIFT and late >= LIFT:
        verdict = "BOTH_QUERY"
    elif match >= LIFT and early < LIFT and late < LIFT:
        verdict = "MATCH_KEY"
    elif early < LIFT and late < LIFT and match < LIFT:
        verdict = "NONE"
    else:
        verdict = "MIXED"
    return {
        "verdict": verdict,
        "instrument_ok": True,
        "negative_ok": negative_ok,
        "early_query": early,
        "late_query": late,
        "match_key": match,
        "c11r_gold": c11,
    }


def licenses_next(verdict: str) -> str:
    if verdict in {"EARLY_QUERY", "BOTH_QUERY"}:
        return "early_layer_pointer_or_early_residual_write_not_p2_as_written"
    if verdict == "LATE_QUERY":
        return "p2_residual_bind_as_written"
    if verdict == "MATCH_KEY":
        return "matching_key_pointer_not_query_residual_at_unembed"
    if verdict == "NONE":
        return "query_or_key_residual_splice_wrong_class"
    return "no_training_license"


def write_manifest() -> dict:
    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    d2 = json.loads(D2_ADJUDICATION.read_text(encoding="utf-8"))
    if d2.get("headline") != "MIXED":
        raise RuntimeError("D3 is licensed only by D2 MIXED")
    files = [
        PROTOCOL,
        Path(__file__),
        S2_DIAGNOSTIC,
        D2_ADJUDICATION,
        PARENT,
        ROOT / "src/baby_v010/model.py",
        ROOT / "src/baby_v010/query_presence.py",
        ROOT / "src/baby_v010/selection_s1.py",
    ]
    payload = {
        "protocol": "V010_QUERY_SPLICE_D3",
        "parent_sha256": PARENT_SHA,
        "licensed_by": "V010_QUERY_PRESENCE_D2",
        "d2_headline": d2["headline"],
        "p2_launched": False,
        "protected_material_opened": False,
        "files": {posix(path): digest(path) for path in files},
        "p1_treatment_sha256": P1_TREAT_SHA,
    }
    write(OUT / "MANIFEST.json", payload)
    return payload


def score_item(logits_row, item: dict, sources: dict) -> dict:
    import torch

    heads = [int(h) for h in item["candidate_heads"]]
    gold_index = int(item["query_index"])
    pred = candidate_argmax(logits_row, heads)
    gen_logit_token = int(torch.argmax(logits_row).item())
    competitor_index = None
    for index, head in enumerate(heads):
        if index != gold_index:
            competitor_index = index
            break
    return {
        "gold": pred == gold_index,
        "emit_query": gen_logit_token == int(item["query_key"]),
        "pick_competitor": competitor_index is not None and pred == competitor_index,
        "K": int(item["pair_count"]),
        "pred": pred,
    }


def run(device: str = "cuda") -> dict:
    import torch

    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    if (OUT / "QUERY_SPLICE.json").exists():
        raise RuntimeError("refusing to overwrite D3 results")
    write_manifest()
    items = [row for row in json.loads(S2_DIAGNOSTIC.read_text(encoding="utf-8")) if eligible_item(row)]
    model, blob = load_ckpt(PARENT, device)
    if int(blob.get("update", -1)) != 16000:
        raise RuntimeError("parent update is not 16000")
    model.set_attention_backend("sdpa")
    debug_log("H_E", "query_splice_d3.py:run", "d3_start", {"n_items": len(items)})

    cells = defaultdict(list)
    steer_miss = []
    with torch.no_grad():
        for index, item in enumerate(items):
            sources = splice_sources(item)
            gen = len(item["input"]) - 1
            gap = gen - int(item["query_position"])
            gold_token = int(item["candidate_heads"][int(item["query_index"])])
            tokens = torch.tensor([item["input"]], dtype=torch.long, device=device)
            logits, blocks = capture_blocks(model, tokens)
            as_is = score_item(logits[0, gen], item, sources)
            as_is.update(gap=gap, body_id=item["body_id"], query_index=item["query_index"])
            cells["as_is"].append(as_is)

            steered = logit_steer(model, tokens, gen, gold_token)
            steer = score_item(steered[0, gen], item, sources)
            steer.update(gap=gap)
            cells["STEER"].append(steer)
            if gap >= LONG_GAP and not as_is["gold"]:
                steer_miss.append(int(steer["gold"]))

            conditions = (
                ("Q0R", "query", 0, False),
                ("Q1R", "query", 1, False),
                ("Q2R", "query", 2, False),
                ("Q11R", "query", 11, False),
                ("Q0M", "query", 0, True),
                ("Q1M", "query", 1, True),
                ("Q11M", "query", 11, True),
                ("K0R", "match_key", 0, False),
                ("K11R", "match_key", 11, False),
                ("C11R", "competitor_key", 11, False),
            )
            for name, source_name, layer, mix in conditions:
                src = int(sources[source_name])
                vec = blocks[layer][0, src]
                out = splice_block(model, tokens, gen, layer, vec, mix)
                scored = score_item(out[0, gen], item, sources)
                scored.update(gap=gap)
                cells[name].append(scored)

            added = embed_add(model, tokens, gen, int(item["query_key"]))
            qe = score_item(added[0, gen], item, sources)
            qe.update(gap=gap)
            cells["QE"].append(qe)

            if index % 40 == 0:
                print(json.dumps({"item": index, "n": len(items), "gap": gap}), flush=True)

    def long_of(name: str) -> dict:
        return summarize_rows([row for row in cells[name] if row["gap"] >= LONG_GAP], "gold")

    def short_of(name: str) -> dict:
        return summarize_rows([row for row in cells[name] if row["gap"] <= 1], "gold")

    summary = {"as_is_long": long_of("as_is"), "as_is_short": short_of("as_is")}
    for name in ("STEER", "Q0R", "Q1R", "Q2R", "Q11R", "Q0M", "Q1M", "Q11M", "K0R", "K11R", "C11R", "QE"):
        summary[name] = long_of(name)
        summary[f"{name}_short"] = short_of(name)

    n_miss = len(steer_miss)
    steer_miss_acc = (sum(steer_miss) / n_miss) if n_miss else 0.0
    decision = decide(summary, steer_miss_acc, n_miss)
    decision["steer_miss_accuracy"] = steer_miss_acc
    decision["steer_n_miss"] = n_miss
    decision["licenses_next"] = licenses_next(decision["verdict"])

    long_items = [item for item in items if (len(item["input"]) - 1 - int(item["query_position"])) >= LONG_GAP]
    parent_census = layer_query_masses(model, long_items, device)
    p1_census = None
    if P1_TREAT.exists() and digest(P1_TREAT) == P1_TREAT_SHA:
        p1_model, p1_blob = load_ckpt(P1_TREAT, device)
        if int(p1_blob.get("update", -1)) != 16800:
            raise RuntimeError("P1 treatment update is not 16800")
        p1_census = layer_query_masses(p1_model, long_items, device)

    debug_log(
        "H_A",
        "query_splice_d3.py:run",
        "d3_long_gap",
        {
            "as_is": summary["as_is_long"],
            "K11R": summary["K11R"],
            "Q0R": summary["Q0R"],
            "Q11R": summary["Q11R"],
            "QE": summary["QE"],
            "C11R": summary["C11R"],
            "verdict": decision["verdict"],
        },
    )
    debug_log(
        "H_E",
        "query_splice_d3.py:run",
        "d3_pointer_census",
        {"parent": parent_census, "p1": p1_census},
    )

    report = {
        "protocol": "V010_QUERY_SPLICE_D3",
        "parent_sha256": PARENT_SHA,
        "n_items": len(items),
        "n_long": summary["as_is_long"]["n"],
        "summary": summary,
        "decision": decision,
        "parent_layer_census": parent_census,
        "p1_layer_census": p1_census,
        "trained": False,
        "p2_launched": False,
        "protected_material_opened": False,
        "gates_changed": False,
        "authoritative_parent_unchanged": True,
    }
    write(OUT / "QUERY_SPLICE.json", report)
    write(OUT / "ADJUDICATION.json", {
        "protocol": "V010_QUERY_SPLICE_D3",
        **decision,
        "parent_sha256": PARENT_SHA,
        "primary": {
            "as_is": summary["as_is_long"],
            "Q0R": summary["Q0R"],
            "Q1R": summary["Q1R"],
            "Q2R": summary["Q2R"],
            "Q11R": summary["Q11R"],
            "Q0M": summary["Q0M"],
            "Q1M": summary["Q1M"],
            "Q11M": summary["Q11M"],
            "K0R": summary["K0R"],
            "K11R": summary["K11R"],
            "C11R": summary["C11R"],
            "QE": summary["QE"],
            "STEER": summary["STEER"],
        },
        "parent_median_argmax_layer": parent_census["median_argmax_layer"],
        "p1_median_argmax_layer": None if p1_census is None else p1_census["median_argmax_layer"],
        "trained": False,
        "p2_launched": False,
        "protected_material_opened": False,
        "gates_changed": False,
        "authoritative_parent_unchanged": True,
    })
    print(json.dumps({"verdict": decision["verdict"], "licenses_next": decision["licenses_next"]}), flush=True)
    return report
