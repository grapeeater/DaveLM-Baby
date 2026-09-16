from __future__ import annotations

"""Read-only v2R4 checkpoint isolation probe.

This CLI never trains, never weakens gates, and never rewrites frozen panels.
If the local checkpoint is absent — expected on GitHub — it writes a skip
receipt and exits 0 so the command stays valid until Fan Diesel provides weights.
"""

import argparse
import hashlib
import json
from pathlib import Path

from .autopsy_v2r4 import load_metrics, terminal_record
from .config import BabyVNextConfig
from .isolation_classify import (
    classify_payload_origin,
    compact_query_swap_row,
    inventory_first_token_snapshot,
    summarize_scored,
)
from .isolation_transforms import assert_frozen_panels, build_isolation_panels
from .v2r4_provenance import (
    TERMINAL_CHECKPOINT_SHA256,
    TERMINAL_METRICS_SHA256,
    TERMINAL_UPDATE,
    require_frozen_file,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANELS = ROOT / "data" / "generated" / "foundation_v2" / "panels.json"
DEFAULT_CHECKPOINT = ROOT / "runs" / "structured_v2r4_seed106001_from6000_terminal" / "checkpoint_16000.pt"
DEFAULT_METRICS = ROOT / "runs" / "structured_v2r4_seed106001_from6000_terminal" / "metrics.jsonl"
CONFIG_PATH = ROOT / "configs" / "foundation_v1.json"
SCORE_PANEL_NAMES = (
    "query_swap_same_surface_novel",
    "query_swap_short_keyed",
    "marker_swap_keep_sep_same_surface_novel",
    "sep_swap_keep_markers_same_surface_novel",
    "value_absent_same_surface_novel",
    "value_absent_heldout_surface",
    "value_absent_broken_context",
    "same_surface_novel_pair_count_2",
    "same_surface_novel_pair_count_3",
    "same_surface_novel_pair_count_4",
    "short_keyed_pair_count_1",
    "short_keyed_pair_count_2",
    "body_reorder_query_first_same_surface_novel",
    "body_reorder_query_last_same_surface_novel",
    "body_reorder_query_first_heldout_surface",
)
PARENT_PANEL_FOR = {
    "query_swap_same_surface_novel": "same_surface_novel",
    "query_swap_short_keyed": "short_keyed",
    "marker_swap_keep_sep_same_surface_novel": "same_surface_novel",
    "sep_swap_keep_markers_same_surface_novel": "same_surface_novel",
    "value_absent_same_surface_novel": "same_surface_novel",
    "value_absent_heldout_surface": "heldout_surface",
    "value_absent_broken_context": "broken_context",
    "same_surface_novel_pair_count_2": "same_surface_novel",
    "same_surface_novel_pair_count_3": "same_surface_novel",
    "same_surface_novel_pair_count_4": "same_surface_novel",
    "short_keyed_pair_count_1": "short_keyed",
    "short_keyed_pair_count_2": "short_keyed",
    "body_reorder_query_first_same_surface_novel": "same_surface_novel",
    "body_reorder_query_last_same_surface_novel": "same_surface_novel",
    "body_reorder_query_first_heldout_surface": "heldout_surface",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def skip_receipt(out: Path, checkpoint: Path, reason: str) -> dict:
    receipt = {
        "status": "CHECKPOINT_ABSENT",
        "reason": reason,
        "checkpoint": str(checkpoint),
        "protected_material_opened": False,
        "trained": False,
        "optimizer_updated": False,
        "v2r5_status": "absent_in_github_unresolved_pending",
        "next": (
            "On Fan Diesel, pass the verified v2R4 U16000 checkpoint and rerun. "
            "Do not launch training. Do not interpret this skip as a v2R5 result."
        ),
    }
    write_json(out / "PROBE.json", receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


def device_from_arg(value: str):
    import torch

    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path, device) -> tuple[object, dict]:
    import torch

    from .model import BabyVNextLM

    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if blob.get("protected_material_opened"):
        raise RuntimeError("refusing a checkpoint that opened protected material")
    if "optimizer_state_dict" in blob:
        # Present on disk is fine; we never load it and never step it.
        blob = {key: value for key, value in blob.items() if key != "optimizer_state_dict"}
    config_raw = blob.get("config")
    config = BabyVNextConfig.from_dict(config_raw) if isinstance(config_raw, dict) else BabyVNextConfig.load(CONFIG_PATH)
    model = BabyVNextLM(config).to(device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    return model, blob


def item_join_key(item: dict, *, original: bool = False) -> tuple:
    if original and item.get("original_query_key") is not None:
        return (tuple(item["source"]), int(item["original_query_key"]), tuple(item["original_target_span"]))
    return (tuple(item["source"]), int(item["query_key"]), tuple(item["target_span"]))


def annotate_inventory_first_tokens(model, items: list[dict], rows: list[dict], device) -> None:
    """Attach inventory first-token ranks from a context-only forward. Read-only."""
    import torch

    if not items:
        return
    contexts = [[int(tok) for tok in item["input"]] for item in items]
    max_len = max(len(context) for context in contexts)
    x = torch.zeros((len(items), max_len), dtype=torch.long, device=device)
    lengths: list[int] = []
    for i, context in enumerate(contexts):
        x[i, : len(context)] = torch.tensor(context, dtype=torch.long, device=device)
        lengths.append(len(context))
    with torch.no_grad():
        logits = model(x)
    mismatches = 0
    for i, (item, row, length) in enumerate(zip(items, rows, lengths)):
        step = logits[i, length - 1]
        vector = step.detach().float().cpu().tolist()
        greedy = int(step.argmax().item())
        emitted = [int(tok) for tok in row.get("emitted", [])]
        if emitted and greedy != emitted[0]:
            mismatches += 1
        snapshot = inventory_first_token_snapshot(item, vector, greedy_token=emitted[0] if emitted else greedy)
        snapshot["context_greedy_token"] = greedy
        snapshot["emitted_matches_context_greedy"] = bool(emitted) and greedy == emitted[0]
        row["inventory_first_tokens"] = snapshot
    if mismatches:
        for row in rows:
            row.setdefault("inventory_first_tokens", {})
            row["inventory_first_tokens"]["context_greedy_mismatch_count"] = mismatches


def parent_matched_summary(parent_items: list[dict], parent_rows: list[dict], isolation_items: list[dict], *, original: bool) -> dict:
    index = {item_join_key(item): (item, row) for item, row in zip(parent_items, parent_rows)}
    matched_items: list[dict] = []
    matched_rows: list[dict] = []
    missing = 0
    for item in isolation_items:
        key = item_join_key(item, original=original)
        found = index.get(key)
        if found is None:
            missing += 1
            continue
        matched_items.append(found[0])
        matched_rows.append(found[1])
    summary = summarize_scored(matched_items, matched_rows) if matched_items else {"n": 0}
    summary["matched"] = len(matched_items)
    summary["missing"] = missing
    return summary


def run_probe(checkpoint: Path, panels_path: Path, out: Path, device_name: str, expect_sha256: str | None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if not checkpoint.exists():
        return skip_receipt(out, checkpoint, "checkpoint file is not present in this workspace")
    digest = sha256_file(checkpoint)
    expected = expect_sha256 or TERMINAL_CHECKPOINT_SHA256
    if digest != expected:
        receipt = {
            "status": "CHECKPOINT_HASH_MISMATCH",
            "checkpoint": str(checkpoint),
            "actual_sha256": digest,
            "expected_sha256": expected,
            "protected_material_opened": False,
            "trained": False,
            "optimizer_updated": False,
        }
        write_json(out / "PROBE.json", receipt)
        raise SystemExit(json.dumps(receipt, indent=2))
    panels_id = assert_frozen_panels(panels_path)
    panels = json.loads(panels_path.read_text(encoding="utf-8"))
    isolation = build_isolation_panels(panels)
    after = assert_frozen_panels(panels_path)
    if after["working_tree_sha256"] != panels_id["working_tree_sha256"]:
        raise RuntimeError("frozen panels.json changed while building isolation copies")
    write_json(out / "ISOLATION_META.json", isolation["meta"])
    device = device_from_arg(device_name)
    model, blob = load_model(checkpoint, device)
    if int(blob.get("update", -1)) != TERMINAL_UPDATE:
        raise RuntimeError(f"checkpoint update {blob.get('update')} is not U{TERMINAL_UPDATE}")
    from .evaluate import score_items

    scored: dict[str, dict] = {}
    row_dumps: dict[str, list[dict]] = {}
    for name in SCORE_PANEL_NAMES:
        items = isolation[name]
        rows: list[dict] = []
        for start in range(0, len(items), 16):
            chunk_items = items[start : start + 16]
            chunk_rows = score_items(model, chunk_items, device)
            annotate_inventory_first_tokens(model, chunk_items, chunk_rows, device)
            rows.extend(chunk_rows)
        scored[name] = summarize_scored(items, rows)
        if name.startswith("query_swap_") or name.startswith("value_absent_") or name.startswith("body_reorder_"):
            row_dumps[name] = [
                compact_query_swap_row(item, row, classify_payload_origin(item, row), row.get("inventory_first_tokens"))
                for item, row in zip(items, rows)
            ]
    parent_matched = {}
    if DEFAULT_METRICS.exists():
        metrics_id = require_frozen_file(DEFAULT_METRICS, TERMINAL_METRICS_SHA256, "metrics")
        term = terminal_record(load_metrics(DEFAULT_METRICS))
        for name in SCORE_PANEL_NAMES:
            parent_name = PARENT_PANEL_FOR[name]
            original = isolation[name] and isolation[name][0].get("isolation_transform") == "query_swap"
            parent_matched[name] = parent_matched_summary(
                panels[parent_name],
                term["rows"][parent_name],
                isolation[name],
                original=original,
            )
        parent_matched["metrics_identity"] = metrics_id
    report = {
        "status": "V2R4_ISOLATION_PROBE",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "panels_sha256": panels_id["frozen_sha256"],
        "panels_identity": panels_id,
        "update": blob.get("update"),
        "seed": blob.get("seed"),
        "protocol": blob.get("protocol"),
        "device": str(device),
        "protected_material_opened": False,
        "trained": False,
        "optimizer_updated": False,
        "v2r5_status": "unresolved_pending_not_opened_by_this_probe",
        "gates_weakened": False,
        "summaries": scored,
        "parent_matched": parent_matched,
        "meta": isolation["meta"],
    }
    write_json(out / "PROBE.json", report)
    write_json(out / "QUERY_SWAP_ROWS.json", {key: value for key, value in row_dumps.items() if key.startswith("query_swap_")})
    write_json(out / "ISOLATION_ROWS.json", row_dumps)
    print(json.dumps({"status": report["status"], "summaries": {key: value["n"] for key, value in scored.items()}}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only v2R4 isolation checkpoint probe")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--panels", type=Path, default=DEFAULT_PANELS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cpu")
    parser.add_argument("--expect-sha256", default=TERMINAL_CHECKPOINT_SHA256)
    args = parser.parse_args()
    if not args.panels.exists():
        raise SystemExit(f"panels missing: {args.panels}")
    run_probe(args.checkpoint, args.panels, args.out, args.device, args.expect_sha256)


if __name__ == "__main__":
    main()
