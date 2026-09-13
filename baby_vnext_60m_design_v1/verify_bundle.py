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
    detached = (ROOT / "FREEZE_RECEIPT.sha256").read_text(encoding="ascii").strip().split()[0]
    receipt_path = ROOT / "FREEZE_RECEIPT.json"
    if sha256(receipt_path) != detached:
        raise RuntimeError("detached receipt hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / "SHA256SUMS.txt"
    if sha256(manifest_path) != receipt["manifest_sha256"]:
        raise RuntimeError("manifest hash mismatch")
    checked = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / Path(relative)
        if not path.is_file():
            raise RuntimeError(f"missing payload: {relative}")
        if sha256(path) != expected:
            raise RuntimeError(f"payload hash mismatch: {relative}")
        checked += 1
    facts = json.loads((ROOT / "BABY_VNEXT_FACTS.json").read_text(encoding="utf-8"))
    validation = json.loads((ROOT / "VALIDATION_RESULTS.json").read_text(encoding="utf-8"))
    if facts["training_performed"] or validation["optimizer_steps"] != 0:
        raise RuntimeError("bundle does not represent a zero-training freeze")
    print(
        json.dumps(
            {
                "status": "BABY_VNEXT_60M_CANDIDATE_INTEGRITY_PASS",
                "receipt_sha256": detached,
                "manifest_sha256": receipt["manifest_sha256"],
                "payload_files_checked": checked,
                "trained_parameters": facts["selected_candidate"]["trained_parameters"],
                "training_performed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

