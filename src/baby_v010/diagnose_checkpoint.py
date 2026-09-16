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

from .autopsy_v2r4 import classify_item_row
from .config import BabyVNextConfig
from .isolation_transforms import assert_frozen_panels, build_isolation_panels
from .v2r4_provenance import (
    TERMINAL_CHECKPOINT_SHA256,
    TERMINAL_UPDATE,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANELS = ROOT / "data" / "generated" / "foundation_v2" / "panels.json"
DEFAULT_CHECKPOINT = ROOT / "runs" / "structured_v2r4_seed106001_from6000_terminal" / "checkpoint_16000.pt"
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
    config_raw = blob.get("config")
    config = BabyVNextConfig.from_dict(config_raw) if isinstance(config_raw, dict) else BabyVNextConfig.load(CONFIG_PATH)
    model = BabyVNextLM(config).to(device)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    return model, blob


def summarize_scored(items: list[dict], rows: list[dict]) -> dict:
    classified = [classify_item_row(item, row) for item, row in zip(items, rows)]
    follow = 0
    stuck_old = 0
    for item, row, info in zip(items, rows, classified):
        original = item.get("original_target_span")
        emitted = [int(tok) for tok in row.get("emitted", [])]
        if original and emitted[: len(original)] == list(original):
            stuck_old += 1
        if item.get("isolation_transform") == "query_swap" and info["value_ok"]:
            follow += 1
    n = len(classified)
    return {
        "n": n,
        "first_ok": sum(info["first_ok"] for info in classified),
        "tf_value": sum(info["tf_value"] for info in classified),
        "value_ok": sum(info["value_ok"] for info in classified),
        "free_exact": sum(info["free_exact"] for info in classified),
        "tf_exact": sum(info["tf_exact"] for info in classified),
        "fail": {key: sum(info["fail"] == key for info in classified) for key in sorted({info["fail"] for info in classified})},
        "query_swap_follow_new_value": follow if any(item.get("isolation_transform") == "query_swap" for item in items) else None,
        "emitted_original_value_span": stuck_old if any(item.get("original_target_span") for item in items) else None,
        "chance_1_over_k": (
            1.0 / int(items[0]["pair_count"])
            if items and items[0].get("kind") == "keyed" and items[0].get("pair_count")
            else None
        ),
        "value_ok_rate": (sum(info["value_ok"] for info in classified) / n) if n else None,
        "free_exact_rate": (sum(info["free_exact"] for info in classified) / n) if n else None,
    }


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
    for name in SCORE_PANEL_NAMES:
        items = isolation[name]
        rows = []
        for start in range(0, len(items), 16):
            rows.extend(score_items(model, items[start : start + 16], device))
        scored[name] = summarize_scored(items, rows)
    report = {
        "status": "V2R4_ISOLATION_PROBE",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "panels_sha256": panels_id["frozen_sha256"],
        "panels_identity": panels_id,
        "update": blob.get("update"),
        "seed": blob.get("seed"),
        "protocol": blob.get("protocol"),
        "protected_material_opened": False,
        "trained": False,
        "v2r5_status": "absent_in_github_unresolved_pending",
        "gates_weakened": False,
        "summaries": scored,
        "meta": isolation["meta"],
    }
    write_json(out / "PROBE.json", report)
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
