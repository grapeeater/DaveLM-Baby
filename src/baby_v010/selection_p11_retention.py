"""P11 active-overwrite retention: parent vs overwrite OFF vs first-step ON.

Protocol: `design/V010_SELECTION_REPAIR_P11_RETENTION.md`.
No training. No TEST. Does not promote U16000.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from .residual_overwrite import LocalSlotOverwrite, attach_overwrite
from .selection_p1 import bootstrap_delta
from .selection_p2 import RETENTION_DROP, negative_controls, retention_failures
from .selection_p11 import GATE_BIAS, LANGUAGE_CE_HARD, OUT as P11_OUT, TRAIN_SEED
from .selection_p11_decode import BOOTSTRAP_SEED, MIN_GAP, SUCCESS_DELTA, long_items
from .selection_s1 import PANEL, PARENT, PARENT_SHA, digest, write
from .train_v2r4 import DEV_STREAM, device_from_arg
from .v2r4_provenance import (
    FROZEN_PANELS_SHA256,
    TERMINAL_CHECKPOINT_SHA256,
    identify_frozen_file,
    require_frozen_file,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/selection_p11_retention"
PROTOCOL = ROOT / "design/V010_SELECTION_REPAIR_P11_RETENTION.md"
S2_DIAGNOSTIC = ROOT / "runs/selection_s2/DIAGNOSTIC.json"
ISOLATION = ROOT / "runs/v2r4_isolation_panels/ISOLATION_PANELS.json"
TREATMENT_CKPT = P11_OUT / f"treatment_{TRAIN_SEED}" / "checkpoint_16800.pt"
TREATMENT_SHA = "369d95c5fdfafea6b270afc47e5a008d17415b7efda275f1aef522d2e4821157"
S2_DIAGNOSTIC_SHA = "5b63533f7eb7feef65a5254deae735b43e2904bff30deeebcf81bb32913dc446"
P11_FIRSTSTEP = ROOT / "runs/selection_p11_decode_firststep/ADJUDICATION.json"
SKIP_PANELS = ("all_intact", "novel", "induction")


def posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def resolve_device(requested: str = "auto"):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device_from_arg(requested)


def required_identities() -> list[dict]:
    return [
        {
            "path": posix(PARENT),
            "role": "authoritative_parent",
            "sha256": PARENT_SHA,
            "exists": PARENT.exists(),
        },
        {
            "path": posix(TREATMENT_CKPT),
            "role": "p11_treatment_checkpoint",
            "sha256": TREATMENT_SHA,
            "exists": TREATMENT_CKPT.exists(),
        },
        {
            "path": posix(S2_DIAGNOSTIC),
            "role": "s2_long_gap_and_rest_lock",
            "sha256": S2_DIAGNOSTIC_SHA,
            "exists": S2_DIAGNOSTIC.exists(),
        },
        {
            "path": posix(PANEL),
            "role": "frozen_retention_panels",
            "sha256": FROZEN_PANELS_SHA256,
            "exists": PANEL.exists(),
        },
        {
            "path": str(DEV_STREAM),
            "role": "language_dev_stream",
            "sha256": None,
            "exists": DEV_STREAM.exists(),
        },
        {
            "path": posix(ISOLATION),
            "role": "isolation_negative_controls",
            "sha256": None,
            "exists": ISOLATION.exists(),
        },
    ]


def verify_p11_receipts() -> dict:
    p11 = json.loads((P11_OUT / f"ADJUDICATION_{TRAIN_SEED}_800.json").read_text(encoding="utf-8"))
    firststep = json.loads(P11_FIRSTSTEP.read_text(encoding="utf-8"))
    decode = json.loads((ROOT / "runs/selection_p11_decode/ADJUDICATION.json").read_text(encoding="utf-8"))
    if p11.get("verdict") != "SUCCESS":
        raise RuntimeError("P11 gen-only is not SUCCESS")
    if p11.get("parent_sha256") != PARENT_SHA:
        raise RuntimeError("P11 parent hash mismatch")
    if p11.get("protected_material_opened"):
        raise RuntimeError("P11 opened protected material")
    if firststep.get("h1_verdict") != "H1_PASS":
        raise RuntimeError("P11 first-step decode is not H1_PASS")
    if firststep.get("primary", {}).get("treatment_free_exact") != 102:
        raise RuntimeError("first-step endpoint is not the frozen 102/215")
    if decode.get("h1_verdict") != "H1_FAIL":
        raise RuntimeError("every-step decode is not the frozen H1_FAIL")
    return {
        "p11_verdict": p11["verdict"],
        "firststep_h1": firststep["h1_verdict"],
        "decode_h1": decode["h1_verdict"],
        "firststep_free_exact": firststep["primary"]["treatment_free_exact"],
        "firststep_init_free_exact": firststep["primary"]["init_free_exact"],
        "protected_material_opened": False,
    }


def hash_if_exists(path: Path, expected: str | None, *, newline_normalized: bool = False) -> dict:
    info = {"path": str(path), "exists": path.exists(), "sha256": None, "match": None, "match_mode": None}
    if not path.exists():
        return info
    if expected is None:
        info["sha256"] = digest(path)
        info["match"] = True
        info["match_mode"] = "unhashed_presence"
        return info
    if newline_normalized:
        identity = identify_frozen_file(path, expected)
        info["sha256"] = identity["working_tree_sha256"]
        info["match"] = identity["match"]
        info["match_mode"] = identity["match_mode"]
        return info
    info["sha256"] = digest(path)
    info["match"] = info["sha256"] == expected
    info["match_mode"] = "exact" if info["match"] else "mismatch"
    return info


def freeze() -> dict:
    if (OUT / "MANIFEST.json").exists():
        raise RuntimeError("P11 retention already frozen")
    receipts = verify_p11_receipts()
    require_frozen_file(PANEL, FROZEN_PANELS_SHA256, "panels")
    files = [
        PROTOCOL,
        Path(__file__),
        ROOT / "src/baby_v010/evaluate.py",
        ROOT / "src/baby_v010/residual_overwrite.py",
        PANEL,
        P11_OUT / f"ADJUDICATION_{TRAIN_SEED}_800.json",
        P11_FIRSTSTEP,
        ROOT / "runs/selection_p11_decode/ADJUDICATION.json",
        ROOT / "src/baby_v010/selection_p2.py",
    ]
    manifest = {
        "protocol": "V010_SELECTION_REPAIR_P11_RETENTION",
        "licensed_by": "V010_SELECTION_REPAIR_P11_DECODE_FIRSTSTEP",
        "parent_sha256": PARENT_SHA,
        "treatment_checkpoint_sha256": TREATMENT_SHA,
        "s2_diagnostic_sha256": S2_DIAGNOSTIC_SHA,
        "retention_drop": RETENTION_DROP,
        "language_ce_hard": LANGUAGE_CE_HARD,
        "long_gap_success_delta": SUCCESS_DELTA,
        "min_gap": MIN_GAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "first_answer_only": True,
        "every_step_overwrite": False,
        "training": False,
        "p11_receipts": receipts,
        "required": required_identities(),
        "files": {posix(path): digest(path) for path in files},
        "protected_material_opened": False,
        "authoritative_parent_unchanged": True,
        "p11_gates_reopened": False,
    }
    write(OUT / "MANIFEST.json", manifest)
    return manifest


def preflight() -> dict:
    receipts = verify_p11_receipts()
    identities = []
    missing = []
    mismatch = []
    for spec in required_identities():
        path = ROOT / spec["path"] if not spec["path"].startswith("/") and ":" not in spec["path"] else Path(spec["path"])
        if spec["role"] == "language_dev_stream":
            path = DEV_STREAM
        elif spec["role"] == "isolation_negative_controls":
            path = ISOLATION
        elif spec["role"] == "authoritative_parent":
            path = PARENT
        elif spec["role"] == "p11_treatment_checkpoint":
            path = TREATMENT_CKPT
        elif spec["role"] == "s2_long_gap_and_rest_lock":
            path = S2_DIAGNOSTIC
        elif spec["role"] == "frozen_retention_panels":
            path = PANEL
        info = hash_if_exists(
            path,
            spec["sha256"],
            newline_normalized=spec["role"] == "frozen_retention_panels",
        )
        info["role"] = spec["role"]
        identities.append(info)
        if not info["exists"]:
            missing.append(spec["role"])
        elif info["match"] is False:
            mismatch.append(spec["role"])
    if PARENT.exists() and digest(PARENT) != TERMINAL_CHECKPOINT_SHA256:
        mismatch.append("authoritative_parent")
    status = "PREFLIGHT_OK" if not missing and not mismatch else "PREFLIGHT_BLOCKED"
    report = {
        "protocol": "V010_SELECTION_REPAIR_P11_RETENTION",
        "status": status,
        "p11_receipts": receipts,
        "identities": identities,
        "missing": missing,
        "mismatch": mismatch,
        "training": False,
        "protected_material_opened": False,
        "device_cuda": _cuda_available(),
    }
    write(OUT / "PREFLIGHT.json", report)
    return report


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def load_parent_model(device):
    import torch

    from .config import BabyVNextConfig
    from .model import BabyVNextLM

    if digest(PARENT) != PARENT_SHA:
        raise RuntimeError("parent mismatch")
    ckpt = torch.load(PARENT, map_location="cpu", weights_only=False)
    if int(ckpt["update"]) != 16000:
        raise RuntimeError("parent is not U16000")
    if ckpt.get("protected_material_opened"):
        raise RuntimeError("parent opened protected material")
    config = BabyVNextConfig.from_dict(ckpt["config"])
    model = BabyVNextLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.set_attention_backend("sdpa")
    model.eval()
    return model, config, ckpt


def load_p11(device):
    import torch

    if digest(TREATMENT_CKPT) != TREATMENT_SHA:
        raise RuntimeError("P11 treatment checkpoint hash mismatch")
    ckpt = torch.load(TREATMENT_CKPT, map_location="cpu", weights_only=False)
    if ckpt.get("parent_checkpoint_sha256") != PARENT_SHA:
        raise RuntimeError("treatment checkpoint parent mismatch")
    if ckpt.get("protected_material_opened"):
        raise RuntimeError("P11 checkpoint opened protected material")
    model, config, parent_ckpt = load_parent_model(device)
    model.load_state_dict(ckpt["model_state_dict"])
    overwrite = LocalSlotOverwrite(config.d_model, gate_bias=GATE_BIAS, gen_only=True).to(device)
    overwrite.load_state_dict(ckpt["overwrite_state_dict"])
    overwrite.eval()
    delta = 0.0
    for key, tensor in parent_ckpt["model_state_dict"].items():
        delta = max(delta, float((tensor - ckpt["model_state_dict"][key]).abs().max().item()))
    return model, overwrite, ckpt, delta


def greedy_decode(model, item: dict, device, n_tokens: int, overwrite=None, first_answer_only: bool = False) -> list[int]:
    import torch

    generated = [int(t) for t in item["input"]]
    emitted: list[int] = []
    with torch.no_grad():
        for step in range(n_tokens):
            if overwrite is not None and first_answer_only and step == 0:
                overwrite.gen_index = torch.tensor([len(generated) - 1], device=device, dtype=torch.long)
            elif overwrite is not None:
                overwrite.gen_index = None
            tokens = torch.tensor([generated[-256:]], dtype=torch.long, device=device)
            logits = model(tokens)
            token = int(logits[0, -1].argmax().item())
            emitted.append(token)
            generated.append(token)
    return emitted


def score_long_gap(model, items: list[dict], device, overwrite=None, first_answer_only: bool = False) -> dict:
    rows = []
    for item in items:
        target = [int(t) for t in item["target"]]
        emitted = greedy_decode(
            model,
            item,
            device,
            len(target),
            overwrite=overwrite,
            first_answer_only=first_answer_only,
        )
        pred_head = emitted[0] if emitted else None
        rows.append(
            {
                "body_id": item["body_id"],
                "query_index": item["query_index"],
                "K": item["pair_count"],
                "free_exact": emitted == target,
                "first_correct": pred_head == target[0],
                "inventory_correct": emitted == target,
            }
        )
    n = len(rows)
    free = sum(int(row["free_exact"]) for row in rows)
    first = sum(int(row["first_correct"]) for row in rows)
    return {
        "n": n,
        "free_exact": free,
        "free_accuracy": free / n if n else 0.0,
        "first_correct": first,
        "first_accuracy": first / n if n else 0.0,
        "rows": rows,
    }


def rest_lock_mean(rows: list[dict], items: list[dict]) -> float:
    locks = []
    for row, item in zip(rows, items):
        span = len(item["target_span"])
        ranks = row["all_ranks"]
        locks.append(all(rank == 1 for rank in ranks[1:span]))
    return float(statistics.mean(locks)) if locks else float("nan")


def measure_arm(model, device, *, overwrite=None, first_answer_only: bool = False) -> dict:
    import torch

    from .data import read_u16
    from .evaluate import evaluate_panels, language_ce, score_items

    diagnostic_items = json.loads(S2_DIAGNOSTIC.read_text(encoding="utf-8"))
    long_gap_items = long_items(diagnostic_items)
    long_gap = score_long_gap(
        model,
        long_gap_items,
        device,
        overwrite=overwrite,
        first_answer_only=first_answer_only,
    )
    scored = []
    for start in range(0, len(diagnostic_items), 16):
        scored.extend(
            score_items(
                model,
                diagnostic_items[start : start + 16],
                device,
                overwrite=overwrite,
                first_answer_only=first_answer_only,
            )
        )
    panels = json.loads(PANEL.read_text(encoding="utf-8"))
    panels = {name: rows for name, rows in panels.items() if name not in SKIP_PANELS}
    frozen = evaluate_panels(
        model,
        panels,
        device,
        overwrite=overwrite,
        first_answer_only=first_answer_only,
    )
    isolation = evaluate_panels(
        model,
        json.loads(ISOLATION.read_text(encoding="utf-8")),
        device,
        overwrite=overwrite,
        first_answer_only=first_answer_only,
    )
    stream = torch.tensor(read_u16(DEV_STREAM), dtype=torch.long)
    if overwrite is not None:
        overwrite.gen_index = None
    ce = language_ce(model, stream, list(range(0, 32 * 256, 256)), device, limit=32)
    report = {
        "first_answer_only": first_answer_only,
        "overwrite_attached": overwrite is not None,
        "summary": {"rest_lock": rest_lock_mean(scored, diagnostic_items)},
        "frozen": {"summaries": frozen["summaries"]},
        "isolation": {"summaries": isolation["summaries"]},
        "language_dev_ce": ce,
        "long_gap": {key: value for key, value in long_gap.items() if key != "rows"},
        "long_gap_rows": long_gap["rows"],
        "protected_material_opened": False,
    }
    return report


def language_ce_fail(parent_ce: float, arm_ce: float) -> bool:
    return arm_ce - parent_ce > LANGUAGE_CE_HARD


def arm_retention(parent: dict, arm: dict) -> dict:
    regressions = retention_failures(parent, arm)
    controls = negative_controls(arm)
    ce_fail = language_ce_fail(parent["language_dev_ce"], arm["language_dev_ce"])
    if ce_fail:
        regressions = [
            *regressions,
            {
                "panel": "language_dev",
                "metric": "ce",
                "parent": parent["language_dev_ce"],
                "treatment": arm["language_dev_ce"],
                "bar_drop": LANGUAGE_CE_HARD,
            },
        ]
    return {
        "regressions": regressions,
        "negative_controls": controls,
        "language_ce_hard_fail": ce_fail,
        "pass": not regressions and all(row["pass"] for row in controls.values()),
    }


def decide_verdict(*, benefit_ok: bool, off_ok: bool, on_ok: bool) -> tuple[str, str]:
    if on_ok and benefit_ok:
        return "PASS", "none"
    if not on_ok:
        if off_ok:
            return "REGRESSION", "overwrite_on_first_step"
        return "REGRESSION", "p11_checkpoint_or_training"
    return "STOP_NO_BENEFIT", "none"


def slim_arm(report: dict) -> dict:
    return {
        "first_answer_only": report["first_answer_only"],
        "overwrite_attached": report["overwrite_attached"],
        "summary": report["summary"],
        "frozen": report["frozen"],
        "isolation": {"summaries": report["isolation"]["summaries"]},
        "language_dev_ce": report["language_dev_ce"],
        "long_gap": report["long_gap"],
        "protected_material_opened": False,
    }


def adjudicate(parent: dict, off: dict, on: dict) -> dict:
    off_ret = arm_retention(parent, off)
    on_ret = arm_retention(parent, on)
    on_flags = [
        {"body_id": row["body_id"], "K": row["K"], "inventory_correct": row["free_exact"]}
        for row in on["long_gap_rows"]
    ]
    parent_flags = [
        {"body_id": row["body_id"], "K": row["K"], "inventory_correct": row["free_exact"]}
        for row in parent["long_gap_rows"]
    ]
    off_flags = [
        {"body_id": row["body_id"], "K": row["K"], "inventory_correct": row["free_exact"]}
        for row in off["long_gap_rows"]
    ]
    ci_on_parent = bootstrap_delta(on_flags, parent_flags, seed=BOOTSTRAP_SEED)
    ci_on_off = bootstrap_delta(on_flags, off_flags, seed=BOOTSTRAP_SEED)
    free_delta = on["long_gap"]["free_accuracy"] - parent["long_gap"]["free_accuracy"]
    benefit_ok = free_delta >= SUCCESS_DELTA and ci_on_parent[0] > 0
    verdict, cause = decide_verdict(benefit_ok=benefit_ok, off_ok=off_ret["pass"], on_ok=on_ret["pass"])
    return {
        "protocol": "V010_SELECTION_REPAIR_P11_RETENTION",
        "licensed_by": "V010_SELECTION_REPAIR_P11_DECODE_FIRSTSTEP",
        "verdict": verdict,
        "cause": cause,
        "parent_sha256": PARENT_SHA,
        "treatment_checkpoint": posix(TREATMENT_CKPT),
        "treatment_checkpoint_sha256": TREATMENT_SHA,
        "n_long": parent["long_gap"]["n"],
        "long_gap": {
            "parent_free_exact": parent["long_gap"]["free_exact"],
            "off_free_exact": off["long_gap"]["free_exact"],
            "on_free_exact": on["long_gap"]["free_exact"],
            "on_minus_parent": free_delta,
            "on_minus_off": on["long_gap"]["free_accuracy"] - off["long_gap"]["free_accuracy"],
            "bootstrap_ci95_on_minus_parent": list(ci_on_parent),
            "bootstrap_ci95_on_minus_off": list(ci_on_off),
            "benefit_ok": benefit_ok,
        },
        "retention": {
            "drop_bar": RETENTION_DROP,
            "language_ce_hard": LANGUAGE_CE_HARD,
            "p11_off": off_ret,
            "p11_on": on_ret,
        },
        "benefit_ok": benefit_ok,
        "p11_off_retention_pass": off_ret["pass"],
        "p11_on_retention_pass": on_ret["pass"],
        "promoted": False,
        "training": False,
        "protected_material_opened": False,
        "authoritative_parent_unchanged": True,
        "p11_gates_reopened": False,
        "still_required": [
            "robust query sensitivity",
            "QUERY->KEY->VALUE binding",
            "held-out/new-body generalization",
            "varied long-gap generalization",
            "replication under the frozen lineage",
            "owner authorization before TEST",
        ],
    }


def run(device_name: str = "auto") -> dict:
    gate = preflight()
    if gate["status"] != "PREFLIGHT_OK":
        decision = {
            "protocol": "V010_SELECTION_REPAIR_P11_RETENTION",
            "verdict": "PREFLIGHT_BLOCKED",
            "cause": "missing_or_mismatched_hashed_artifacts",
            "preflight": gate,
            "promoted": False,
            "training": False,
            "protected_material_opened": False,
            "authoritative_parent_unchanged": True,
            "p11_gates_reopened": False,
        }
        write(OUT / "ADJUDICATION.json", decision)
        print(json.dumps({"verdict": "PREFLIGHT_BLOCKED", "missing": gate["missing"], "mismatch": gate["mismatch"]}), flush=True)
        return decision

    device = resolve_device(device_name)
    parent_model, _config, _parent_ckpt = load_parent_model(device)
    parent = measure_arm(parent_model, device, overwrite=None, first_answer_only=False)
    parent["baby_weight_max_abs_delta"] = 0.0
    del parent_model

    p11_model, overwrite, p11_ckpt, delta = load_p11(device)
    handle, _ = attach_overwrite(p11_model, overwrite)
    try:
        off = measure_arm(p11_model, device, overwrite=overwrite, first_answer_only=False)
        on = measure_arm(p11_model, device, overwrite=overwrite, first_answer_only=True)
    finally:
        handle.remove()
        overwrite.gen_index = None
    off["baby_weight_max_abs_delta"] = delta
    on["baby_weight_max_abs_delta"] = delta
    off["p11_update"] = p11_ckpt.get("update")
    on["p11_update"] = p11_ckpt.get("update")

    decision = adjudicate(parent, off, on)
    decision["baby_weight_max_abs_delta"] = delta
    decision["device"] = str(device)
    write(OUT / "PARENT.json", slim_arm(parent))
    write(OUT / "OFF.json", slim_arm(off))
    write(OUT / "ON.json", slim_arm(on))
    write(OUT / "ADJUDICATION.json", decision)
    print(json.dumps({"verdict": decision["verdict"], "long_gap": decision["long_gap"], "cause": decision["cause"]}), flush=True)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "preflight", "run"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.action == "freeze":
        print(json.dumps(freeze()), flush=True)
    elif args.action == "preflight":
        print(json.dumps(preflight()), flush=True)
    else:
        run(args.device)


if __name__ == "__main__":
    main()
