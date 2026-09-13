from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parent
EXCLUDED = {"SHA256SUMS.txt", "FREEZE_RECEIPT.json", "FREEZE_RECEIPT.sha256"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    validation = json.loads((ROOT / "VALIDATION_RESULTS.json").read_text(encoding="utf-8"))
    facts = json.loads((ROOT / "BABY_VNEXT_FACTS.json").read_text(encoding="utf-8"))
    counts = json.loads((ROOT / "PARAMETER_COUNTS.json").read_text(encoding="utf-8"))
    if validation["status"] != "PASS" or validation["optimizer_steps"] != 0:
        raise RuntimeError("validation is not a zero-update PASS")
    if facts["training_performed"] or counts["physical"]["trained_total"] != 61520385:
        raise RuntimeError("candidate facts/counts do not match the frozen decision")
    payload = sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file() and path.name not in EXCLUDED
    )
    lines = []
    for path in payload:
        relative = path.relative_to(ROOT).as_posix()
        lines.append(f"{sha256(path)}  {relative}")
    manifest_path = ROOT / "SHA256SUMS.txt"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    manifest_sha256 = sha256(manifest_path)
    receipt = {
        "status": "BABY_VNEXT_60M_CANDIDATE_FROZEN",
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "bundle": str(ROOT),
        "design_type": "CLEANER_CAPACITY_SUCCESSOR",
        "trained_parameters": 61520385,
        "base_parameters": 60536064,
        "binding_parameters": 984321,
        "config_sha256": validation["tests"]["config_roundtrip"]["config_sha256"],
        "tokenizer_sha256": validation["tokenizer"]["sha256"],
        "initialized_checkpoint_sha256": validation["tests"]["checkpoint_roundtrip"]["checkpoint_sha256"],
        "initialized_model_state_sha256": validation["tests"]["checkpoint_roundtrip"]["model_state_sha256"],
        "manifest_sha256": manifest_sha256,
        "manifest_payload_files": len(lines),
        "validation_status": validation["status"],
        "optimizer_created": False,
        "optimizer_updates": 0,
        "real_data_training": False,
        "locked_or_sacred_material_accessed": False,
    }
    receipt_path = ROOT / "FREEZE_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    receipt_sha256 = sha256(receipt_path)
    detached_path = ROOT / "FREEZE_RECEIPT.sha256"
    detached_path.write_text(
        f"{receipt_sha256}  FREEZE_RECEIPT.json\n", encoding="ascii", newline="\n"
    )
    for path in [*payload, manifest_path, receipt_path, detached_path]:
        os.chmod(path, path.stat().st_mode & ~stat.S_IWRITE)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_sha256": receipt_sha256,
                "manifest_sha256": manifest_sha256,
                "payload_files": len(lines),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
