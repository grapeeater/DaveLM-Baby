from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    receipt_path = ROOT / "FREEZE_RECEIPT.json"
    detached = (ROOT / "FREEZE_RECEIPT.sha256").read_text(encoding="ascii").split()[0]
    if sha256(receipt_path) != detached:
        raise RuntimeError("detached receipt hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["status"] != "BABY_VNEXT_PHASE1_LANGUAGE_READY_TO_TRAIN":
        raise RuntimeError("receipt status mismatch")
    manifest = ROOT / "SHA256SUMS.txt"
    if sha256(manifest) != receipt["manifest_sha256"]:
        raise RuntimeError("manifest hash mismatch")
    checked = 0
    writable = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"payload mismatch: {relative}")
        if path.stat().st_mode & 0o222:
            writable.append(relative)
        checked += 1
    print(json.dumps({
        "status": "PASS",
        "receipt_sha256": detached,
        "manifest_sha256": receipt["manifest_sha256"],
        "payload_files_checked": checked,
        "writable_payloads": writable,
    }, indent=2))


if __name__ == "__main__":
    main()
