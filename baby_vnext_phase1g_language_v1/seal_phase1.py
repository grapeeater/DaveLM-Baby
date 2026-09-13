from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parent
EXCLUDED = {"SHA256SUMS.txt", "FREEZE_RECEIPT.json", "FREEZE_RECEIPT.sha256"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if any((ROOT / name).exists() for name in EXCLUDED):
        raise RuntimeError("refusing to overwrite an existing seal")
    payloads = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and path.name not in EXCLUDED and "run_primary" not in path.parts
    )
    lines = [f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}" for path in payloads]
    manifest = ROOT / "SHA256SUMS.txt"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    manifest_hash = sha256(manifest)
    config_hash = sha256(ROOT / "PHASE1_LANGUAGE_CONFIG.json")
    receipt = {
        "status": "BABY_VNEXT_PHASE1_LANGUAGE_READY_TO_TRAIN",
        "study": "BABY_VNEXT_PHASE1_LANGUAGE_V1",
        "manifest_sha256": manifest_hash,
        "payload_files": len(payloads),
        "phase1_config_sha256": config_hash,
        "initialized_checkpoint_sha256": sha256(ROOT / "initialization" / "seed_610001_initialized.pt"),
        "training_schedule_sha256": sha256(ROOT / "data" / "TRAIN_WINDOW_STARTS.u32"),
        "architecture_receipt_sha256": "776859cb1bf671634ae2d7e8dfab7f7cb8e517ce802d30d3cb3733b6b734b368",
        "tokenizer_sha256": "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b",
        "primary_seed": 610001,
        "optimizer_created_during_preflight": False,
        "optimizer_steps": 0,
        "training_performed": False,
        "integrity_design": "FREEZE_RECEIPT.sha256 -> FREEZE_RECEIPT.json -> SHA256SUMS.txt -> immutable payload",
    }
    receipt_path = ROOT / "FREEZE_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    detached = sha256(receipt_path)
    (ROOT / "FREEZE_RECEIPT.sha256").write_text(
        f"{detached}  FREEZE_RECEIPT.json\n", encoding="ascii", newline="\n"
    )
    for path in payloads + [manifest, receipt_path, ROOT / "FREEZE_RECEIPT.sha256"]:
        os.chmod(path, stat.S_IREAD)
    print(json.dumps({"receipt_sha256": detached, "manifest_sha256": manifest_hash, "payload_files": len(payloads)}, indent=2))


if __name__ == "__main__":
    main()
