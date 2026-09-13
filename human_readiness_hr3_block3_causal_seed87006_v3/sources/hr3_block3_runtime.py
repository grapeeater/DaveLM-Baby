"""Shared runtime for the frozen HR-3 block-3 relaxation treatment.

The module deliberately contains no dataset sampling policy.  The trainer only
consumes the materialized schedules frozen with the treatment bundle.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SOURCE_DIR = Path(__file__).resolve().parent
V09 = Path(r"C:\DaveLM-v0.9")
if str(SOURCE_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SOURCE_DIR))
if str(V09) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(V09))
from treatment13_model import Treatment13Model


PARENT_SHA256 = "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb"
TOKENIZER_SHA256 = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "rb") as f:
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_integrity(bundle: Path) -> dict[str, Any]:
    detached = (bundle / "FREEZE_RECEIPT.sha256").read_text(encoding="utf-8").strip().split()[0]
    receipt_path = bundle / "FREEZE_RECEIPT.json"
    assert sha(receipt_path) == detached, "detached receipt checksum mismatch"
    receipt = read_json(receipt_path)
    assert receipt["status"] == "HR3_BLOCK3_PREFLIGHT_PASS", receipt["status"]
    assert sha(bundle / "MANIFEST.json") == receipt["manifest_sha256"]
    assert sha(bundle / "SHA256SUMS.txt") == receipt["sha256sums_sha256"]
    listed: set[str] = set()
    for line in (bundle / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = bundle / relative
        assert target.is_file(), f"missing frozen payload: {relative}"
        assert sha(target) == expected, f"payload checksum mismatch: {relative}"
        listed.add(relative)
    assert "FREEZE_RECEIPT.json" not in listed and "SHA256SUMS.txt" not in listed
    return receipt


def configure_runtime(seed: int) -> None:
    assert torch.__version__ == "2.12.0+rocm7.14.0", torch.__version__
    assert torch.cuda.is_available(), "frozen HR-3 protocol requires compatible GPU runtime"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def pinned_binding(bundle: Path):
    source = bundle / "sources" / "PINNED_PILOT1_BINDING_IMPLEMENTATION.py"
    spec = importlib.util.spec_from_file_location("hr3_pinned_binding", source)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def load_model(checkpoint: Path, device: torch.device, bundle: Path):
    raw = torch.load(checkpoint, map_location=device, weights_only=False)
    state = dict(raw["model_state_dict"])
    required = ["u", "q", "bs", "ba"]
    localizer_values = [state.pop("localizer." + key) for key in required]
    state.pop("localizer.scorer.weight", None)
    state.pop("localizer.scorer.bias", None)
    model = Treatment13Model().to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert set(missing) == {"localizer.scorer.weight", "localizer.scorer.bias"}
    assert not unexpected
    model.localizer = pinned_binding(bundle).OrthoLocalizer(*localizer_values).to(device)
    return model


def set_scope_block3(model, binding: bool) -> set[str]:
    """Frozen HR-3 scope: English adapts exactly blocks 3--7; binding is full."""
    for parameter in model.parameters():
        parameter.requires_grad_(binding)
    if not binding:
        for parameter in model.base_model.parameters():
            parameter.requires_grad_(True)
        # The sole experimental delta from corrected aligned HR-1 is this range(3).
        for index in range(3):
            for parameter in model.base_model.blocks[index].parameters():
                parameter.requires_grad_(False)
        for parameter in model.localizer.parameters():
            parameter.requires_grad_(False)
        for module in (model.wq, model.wk, model.wv, model.wo):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def aligned_tensors(rows: list[dict[str, Any]], device: torch.device):
    width = max(len(row["token_ids"]) for row in rows) + 2
    x = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    y = torch.full((len(rows), width), -100, dtype=torch.long, device=device)
    for i, row in enumerate(rows):
        z = [2] + row["token_ids"] + [3]
        # At tensor position j, x[j] predicts y[j], the next token in z.
        x[i, : len(z) - 1] = torch.tensor(z[:-1], dtype=torch.long, device=device)
        y[i, : len(z) - 1] = torch.tensor(z[1:], dtype=torch.long, device=device)
    return x, y


def aligned_dev_loss(model, rows: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for start in range(0, min(len(rows), 128), 32):
            x, y = aligned_tensors(rows[start : start + 32], device)
            logits = model.base_model(x)
            values.append(float(F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1), ignore_index=-100)))
    loss = sum(values) / len(values)
    return {"records": min(len(rows), 128), "loss": loss, "perplexity": math.exp(loss), "batch_losses": values}


def flatten_quartets(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return rehearsal quartets in their frozen canonical index ordering."""
    return list(data["quartets"])


def binding_docs_for_batch(quartets: list[dict[str, Any]], quartet_indices: list[int], expected_document_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Resolve literal schedule indices as the authoritative predecessor did."""
    docs: list[dict[str, Any]] = []
    for quartet_index in quartet_indices:
        assert isinstance(quartet_index, int) and 0 <= quartet_index < len(quartets), f"unknown frozen quartet index: {quartet_index}"
        docs.extend(quartets[quartet_index]["docs"])
    if expected_document_ids is not None:
        assert [doc["doc_id"] for doc in docs] == expected_document_ids, "binding schedule document identities disagree with canonical quartet indexing"
    return docs


def binding_loss(model, docs: list[dict[str, Any]], device: torch.device, pinned) -> tuple[torch.Tensor, dict[str, float]]:
    x = torch.tensor([d["full_document_token_ids"] for d in docs], dtype=torch.long, device=device)
    q = torch.tensor([d["qdp"] for d in docs], dtype=torch.long, device=device)
    answer_position = torch.tensor([d["answer_causal_position"] for d in docs], dtype=torch.long, device=device)
    target = torch.tensor([d["target_value_token"] for d in docs], dtype=torch.long, device=device)
    logits, extras = model(x, q, answer_position)
    rows = torch.arange(len(docs), device=device)
    answer = F.cross_entropy(logits[rows, answer_position], target)
    localization = pinned.loc_loss(extras["localization_attention"], docs)
    total = answer + pinned.LAM * localization
    return total, {"answer_ce": float(answer.detach()), "localization_hard_min": float(localization.detach()), "lambda": float(pinned.LAM)}


def binding_summary(result: dict[str, Any]) -> dict[str, Any]:
    overall = result["overall"]
    selected = [r["selected_row_correct"] for r in result["rows"] if r["localization_category"] == "BOTH_DISTINCT"]
    queried = [r["selected_query_row_correct"] for r in result["rows"] if r["localization_category"] == "BOTH_DISTINCT"]
    return {
        "answer_exact": overall["answer_exact"],
        "both_distinct": overall["BD"],
        "slot_collapse": overall["collapse"],
        "complete_quartets": result["complete_quartets"],
        "strict_reversal_both_correct": result["reversal_both_correct"],
        "strict_reversal_pairs": result["reversal_pairs"],
        "queried_row_selection_conditional_on_bd": (sum(bool(x) for x in queried) / len(queried)) if queried else None,
        "downstream_answer_correct_conditional_on_bd": (sum(r["answer_correct"] for r in result["rows"] if r["localization_category"] == "BOTH_DISTINCT") / overall["BD"]) if overall["BD"] else None,
    }


def binding_gate(summary: dict[str, Any]) -> bool:
    return summary["answer_exact"] >= 76 and summary["both_distinct"] >= 76 and summary["slot_collapse"] == 0


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng_cpu"])
    torch.cuda.set_rng_state_all(state["torch_rng_cuda"])


def capture_rng() -> dict[str, Any]:
    return {"python_rng": random.getstate(), "torch_rng_cpu": torch.get_rng_state(), "torch_rng_cuda": torch.cuda.get_rng_state_all()}


def candidate_log_probability(model, prompt_ids: list[int], candidate_ids: list[int], device: torch.device) -> tuple[float, list[float]]:
    full = [2] + prompt_ids + candidate_ids
    with torch.no_grad():
        logits = model.base_model(torch.tensor([full], dtype=torch.long, device=device))[0]
        lps = logits.log_softmax(-1)
        start = len(prompt_ids)
        token_lps = [float(lps[start + j, token]) for j, token in enumerate(candidate_ids)]
    return sum(token_lps), token_lps


def greedy_ids(model, prompt_ids: list[int], device: torch.device, limit: int = 32) -> list[int]:
    sequence = [2] + prompt_ids
    with torch.no_grad():
        for _ in range(limit):
            x = torch.tensor([sequence[-256:]], dtype=torch.long, device=device)
            token = int(model.base_model(x)[0, -1].argmax())
            sequence.append(token)
            if token == 3:
                break
    return sequence
