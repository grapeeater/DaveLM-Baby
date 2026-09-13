"""Executable HR-3 trainer; consumes only frozen materialized artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from hr3_block3_runtime import (
    PARENT_SHA256,
    aligned_dev_loss,
    aligned_tensors,
    atomic_json,
    atomic_torch_save,
    binding_docs_for_batch,
    binding_gate,
    binding_loss,
    binding_summary,
    capture_rng,
    configure_runtime,
    flatten_quartets,
    load_model,
    pinned_binding,
    read_json,
    restore_rng,
    set_scope_block3,
    sha,
    verify_integrity,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_preservation(model, bundle: Path, protocol: dict, device: torch.device, pinned) -> dict:
    out = {}
    model.eval()
    with torch.no_grad():
        for label, rel in protocol["data"]["binding_dev_pools"].items():
            pool = read_json(bundle / rel)
            docs = [doc for quartet in pool["quartets"] for doc in quartet["docs"]]
            result = pinned.binding_eval(model, docs, device)
            summary = binding_summary(result)
            out[label] = {"summary": summary, "gate_pass": binding_gate(summary), "raw": result}
    return out


def append_metric(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def assert_inactive_grads_none(model) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None, f"frozen parameter acquired gradient: {name}"


def save_restart(model, optimizer, update: int, protocol: dict, bundle: Path, output: Path, dev: dict) -> None:
    state = {
        "completed_update": update,
        "parent_sha256": protocol["parent_sha256"],
        "protocol_sha256": sha(bundle / "HR3_PROTOCOL.json"),
        "schedule_sha256": sha(bundle / protocol["data"]["english_schedule"]),
        "binding_schedule_sha256": sha(bundle / protocol["data"]["binding_schedule"]),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "dev": dev,
        **capture_rng(),
    }
    atomic_torch_save(state, output / "restart.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=87006)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preload-check", action="store_true", help="verify the real pre-parent-load path and stop before model construction")
    args = parser.parse_args()

    receipt = verify_integrity(args.bundle)
    protocol = read_json(args.bundle / "HR3_PROTOCOL.json")
    assert args.seed == protocol["training"]["seed"] == 87006
    assert protocol["parent_sha256"] == PARENT_SHA256
    configure_runtime(args.seed)
    device = torch.device("cuda")
    parent = Path(protocol["parent_path"])
    assert sha(parent) == PARENT_SHA256, "parent checkpoint identity mismatch"

    for rel, expected in protocol["frozen_payload_hashes"].items():
        assert sha(args.bundle / rel) == expected, f"frozen artifact mismatch: {rel}"
    train = load_jsonl(args.bundle / protocol["data"]["english_train"])
    dev = load_jsonl(args.bundle / protocol["data"]["english_dev"])
    english_schedule = read_json(args.bundle / protocol["data"]["english_schedule"])["batches"]
    binding_schedule = read_json(args.bundle / protocol["data"]["binding_schedule"])["batches"]
    rehearsal = read_json(args.bundle / protocol["data"]["binding_rehearsal"])
    records = {row["id"]: row for row in train}
    quartets = flatten_quartets(rehearsal)
    assert len(english_schedule) == 450 and len(binding_schedule) == 50

    if args.preload_check:
        print(json.dumps({"status": "HR3_PRELOAD_CHECK_PASS", "english_batches": len(english_schedule), "binding_batches": len(binding_schedule), "train_records": len(train), "binding_quartets": len(quartets), "checkpoint_loaded": False, "optimizer_created": False}, sort_keys=True))
        return

    args.output.mkdir(parents=True, exist_ok=True)
    model = load_model(parent, device, args.bundle)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.05,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
    pinned = pinned_binding(args.bundle)
    start = 1
    dev_results: dict[str, dict] = {}
    metric_path = args.output / "UPDATE_METRICS.jsonl"

    if args.resume:
        restart_path = args.output / "restart.pt"
        assert restart_path.exists(), "resume requested without rolling restart state"
        restart = torch.load(restart_path, map_location=device, weights_only=False)
        assert restart["parent_sha256"] == PARENT_SHA256
        accepted_protocol_hashes = {sha(args.bundle / "HR3_PROTOCOL.json")}
        completion = protocol.get("mechanical_completion", {})
        predecessor_hash = completion.get("accepted_predecessor_protocol_sha256")
        if predecessor_hash:
            accepted_protocol_hashes.add(predecessor_hash)
        accepted_protocol_hashes.update(completion.get("accepted_predecessor_protocol_sha256s", []))
        assert restart["protocol_sha256"] in accepted_protocol_hashes
        assert restart["schedule_sha256"] == sha(args.bundle / protocol["data"]["english_schedule"])
        assert restart["binding_schedule_sha256"] == sha(args.bundle / protocol["data"]["binding_schedule"])
        model.load_state_dict(restart["model_state"])
        optimizer.load_state_dict(restart["optimizer_state"])
        restore_rng(restart)
        start = int(restart["completed_update"]) + 1
        dev_results = restart.get("dev", {})
    else:
        assert not (args.output / "restart.pt").exists(), "output already contains a restart state; use --resume"
        dev_results["0"] = {"aligned_tinystories": aligned_dev_loss(model, dev, device), "checkpoint": "immutable Pilot 1 parent"}
        atomic_json(dev_results, args.output / "DEV_TRAJECTORY.json")

    for update in range(start, 501):
        is_binding = update % 10 == 0
        optimizer.zero_grad(set_to_none=True)
        active = set_scope_block3(model, is_binding)
        model.train()
        if is_binding:
            schedule_record = binding_schedule[update // 10 - 1]
            docs = binding_docs_for_batch(quartets, schedule_record["quartet_ids"], schedule_record["documents"])
            assert len(docs) == 32
            loss, pieces = binding_loss(model, docs, device, pinned)
            record = {"update": update, "scope": "binding", "quartet_ids": schedule_record["quartet_ids"], **pieces}
        else:
            english_index = update - 1 - ((update - 1) // 10)
            schedule_record = english_schedule[english_index]
            batch_ids = schedule_record["record_ids"]
            rows = [records[record_id] for record_id in batch_ids]
            assert len(rows) == 64 and [row["id"] for row in rows] == batch_ids
            x, y = aligned_tensors(rows, device)
            logits = model.base_model(x)
            loss = F.cross_entropy(logits.reshape(-1, 1024), y.reshape(-1), ignore_index=-100)
            record = {"update": update, "scope": "english", "english_batch_index": english_index, "record_ids": batch_ids}
        loss.backward()
        assert_inactive_grads_none(model)
        gradient_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 2.0)
        optimizer.step()
        record.update({"loss": float(loss.detach()), "gradient_norm": float(gradient_norm), "active_parameter_count": len(active)})
        append_metric(metric_path, record)

        if update in (100, 250, 500):
            model.eval()
            preservation = evaluate_preservation(model, args.bundle, protocol, device, pinned)
            dev_results[str(update)] = {"aligned_tinystories": aligned_dev_loss(model, dev, device), "binding": preservation}
            atomic_json(dev_results, args.output / "DEV_TRAJECTORY.json")
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "seed": args.seed,
                "updates": update,
                "parent_sha256": PARENT_SHA256,
                "scope": protocol["training"]["english_scope"],
                "protocol_sha256": sha(args.bundle / "HR3_PROTOCOL.json"),
            }
            atomic_torch_save(checkpoint, args.output / f"checkpoint_{update}.pt")
            if not all(row["gate_pass"] for row in preservation.values()):
                save_restart(model, optimizer, update, protocol, args.bundle, args.output, dev_results)
                atomic_json({"status": "STOP_BINDING_GATE_FAILURE", "update": update, "preservation": preservation}, args.output / "RUN_STATUS.json")
                raise RuntimeError(f"frozen binding preservation gate failed at update {update}")
            model.train()

        save_restart(model, optimizer, update, protocol, args.bundle, args.output, dev_results)

    final = args.output / "checkpoint_500.pt"
    assert final.exists() and sha(final)
    atomic_json(
        {
            "status": "HR3_TRAINING_COMPLETE_PENDING_FROZEN_DEV_EVALUATION",
            "completed_updates": 500,
            "checkpoint_500": str(final),
            "checkpoint_500_sha256": sha(final),
            "parent_sha256": PARENT_SHA256,
            "bundle_receipt_sha256": sha(args.bundle / "FREEZE_RECEIPT.json"),
        },
        args.output / "RUN_STATUS.json",
    )


if __name__ == "__main__":
    main()
