from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat


ROOT = Path(__file__).resolve().parent
EXCLUDED = {"SHA256SUMS.txt", "FREEZE_RECEIPT.json", "FREEZE_RECEIPT.sha256"}
EXCLUDED_DIRS = {"__pycache__"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if any((ROOT / name).exists() for name in EXCLUDED):
        raise RuntimeError("refusing to overwrite an existing seal")
    payloads = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name in EXCLUDED:
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        payloads.append(path)
    lines = [f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}" for p in payloads]
    manifest = ROOT / "SHA256SUMS.txt"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    manifest_hash = sha256(manifest)
    receipt = {
        "status": "BABY_VNEXT_PHASE2A_T2_CTXBIND_FROZEN",
        "study": "BABY_VNEXT_PHASE2A_T2_CTXBIND_V1",
        "manifest_sha256": manifest_hash,
        "payload_files": len(payloads),
        "config_sha256": sha256(ROOT / "PHASE2A_T2_CONFIG.json"),
        "parent_sha256": json.loads((ROOT / "PHASE2A_T2_CONFIG.json").read_text(encoding="utf-8"))["parent"]["sha256"],
        "primary_seed": 610005,
        "test_status": "SEALED_HASHED_UNOPENED",
        "optimizer_created_during_preflight": False,
        "optimizer_steps": 0,
        "training_performed": False,
    }
    receipt_path = ROOT / "FREEZE_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    detached = sha256(receipt_path)
    (ROOT / "FREEZE_RECEIPT.sha256").write_text(f"{detached}  FREEZE_RECEIPT.json\n",
                                                encoding="ascii", newline="\n")
    for path in payloads + [manifest, receipt_path, ROOT / "FREEZE_RECEIPT.sha256"]:
        os.chmod(path, stat.S_IREAD)
    print(json.dumps({"receipt_sha256": detached, "manifest_sha256": manifest_hash,
                      "payload_files": len(payloads)}, indent=2))


if __name__ == "__main__":
    main()
