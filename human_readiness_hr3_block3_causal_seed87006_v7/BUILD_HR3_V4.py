"""Mechanical persistence completion for HR-3 v3.

v3 reached its first in-memory optimizer step but could not commit restart state
because Windows rejected fsync on a read-only descriptor.  v4 pins the unique
read/write-descriptor fix and copies no bytecode caches into the sealed bundle.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(r"C:\DaveLM-CADAVER")
SOURCE = ROOT / "human_readiness_hr3_block3_causal_seed87006_v3"
OUT = ROOT / "human_readiness_hr3_block3_causal_seed87006_v4"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.replace("\r\n", "\n"), encoding="utf-8", newline="\n")


def jwrite(path: Path, value: Any) -> None:
    write(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def verify(path: Path) -> dict:
    expected = (path / "FREEZE_RECEIPT.sha256").read_text(encoding="utf-8").strip().split()[0]
    assert sha(path / "FREEZE_RECEIPT.json") == expected
    result = json.loads((path / "FREEZE_RECEIPT.json").read_text(encoding="utf-8"))
    assert result["status"] == "HR3_BLOCK3_PREFLIGHT_PASS"
    assert sha(path / "MANIFEST.json") == result["manifest_sha256"]
    assert sha(path / "SHA256SUMS.txt") == result["sha256sums_sha256"]
    for line in (path / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            assert sha(path / rel) == digest
    return result


def listed(path: Path) -> list[str]:
    return [line.split("  ", 1)[1] for line in (path / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def payload(path: Path) -> list[dict[str, Any]]:
    excluded = {"SHA256SUMS.txt", "FREEZE_RECEIPT.json", "FREEZE_RECEIPT.sha256"}
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.name not in excluded)
    return [{"path": p.relative_to(path).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size} for p in files]


def readonly(path: Path) -> None:
    for file in path.rglob("*"):
        if file.is_file():
            file.chmod(file.stat().st_mode & ~stat.S_IWRITE)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite v4: {OUT}")
    prior = verify(SOURCE)
    OUT.mkdir(parents=True)
    skip = {"PREFLIGHT.json", "HR3_PROTOCOL.json", "MECHANICAL_COMPLETION.json", "MECHANICAL_COMPLETION.md"}
    for rel in listed(SOURCE):
        if rel in skip or rel.startswith("sources/__pycache__/"):
            continue
        origin, target = SOURCE / rel, OUT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        target.chmod(target.stat().st_mode | stat.S_IWRITE)

    # Pin the sole corrected mechanical implementation; all scientific source
    # semantics remain unchanged.
    shutil.copy2(ROOT / "hr3_block3_runtime.py", OUT / "sources" / "hr3_block3_runtime.py")
    shutil.copy2(ROOT / "hr3_block3_train.py", OUT / "sources" / "TRAIN_HR3.py")
    shutil.copy2(ROOT / "hr3_block3_evaluate.py", OUT / "sources" / "EVALUATE_HR3.py")
    shutil.copy2(ROOT / "validate_hr3_block3_preflight.py", OUT / "sources" / "PREFLIGHT_VALIDATOR.py")
    protocol = json.loads((SOURCE / "HR3_PROTOCOL.json").read_text(encoding="utf-8"))
    protocol["version"] = "HR-3-block3-causal-v4"
    protocol["mechanical_completion"] = {"predecessor": str(SOURCE), "predecessor_receipt_sha256": sha(SOURCE / "FREEZE_RECEIPT.json"), "reason": "rolling restart fsync used a read-only Windows descriptor and failed after uncommitted update 1", "correction": "open temporary restart file r+b, flush, fsync, then atomic replace", "scientific_specification_changed": False}
    protocol["frozen_payload_hashes"] = {f.relative_to(OUT).as_posix(): sha(f) for f in sorted(list((OUT / "data").rglob("*")) + list((OUT / "sources").rglob("*"))) if f.is_file()}
    jwrite(OUT / "HR3_PROTOCOL.json", protocol)
    jwrite(OUT / "MECHANICAL_COMPLETION.json", {"status": "MECHANICAL_PERSISTENCE_COMPLETION_ONLY", "predecessor_bundle": str(SOURCE), "predecessor_receipt_sha256": sha(SOURCE / "FREEZE_RECEIPT.json"), "uncommitted_execution_record": str(ROOT / "human_readiness_hr3_block3_causal_seed87006_execution_attempt1_uncommitted"), "correction_source": str(ROOT / "hr3_block3_runtime.py"), "scientific_values_changed": False, "checkpoint_loaded_during_preflight": False, "optimizer_created_during_preflight": False, "optimizer_updates_during_preflight": 0, "final_accessed": False, "sacred_accessed": False})
    write(OUT / "MECHANICAL_COMPLETION.md", "# HR-3 v4 persistence completion\n\nThe v3 execution attempt performed update 1 in memory but failed before restart-state commit because Windows rejected `fsync` on a read-only descriptor. The attempt is preserved separately and update 1 is uncommitted by the frozen restart rule. v4 changes only the atomic write handle to `r+b`, flushes it, then fsyncs before atomic replacement. It also excludes all bytecode cache files from v4 payload. No scientific treatment component changed.\n")
    shutil.copy2(Path(__file__), OUT / "BUILD_HR3_V4.py")
    subprocess.run([sys.executable, str(OUT / "sources" / "PREFLIGHT_VALIDATOR.py"), "--bundle", str(OUT)], check=True)
    jwrite(OUT / "MANIFEST.json", {"bundle": OUT.name, "payload": payload(OUT)})
    values = payload(OUT)
    write(OUT / "SHA256SUMS.txt", "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in values))
    receipt = {"status": "HR3_BLOCK3_PREFLIGHT_PASS", "bundle": str(OUT), "predecessor_bundle": str(SOURCE), "predecessor_receipt_sha256": sha(SOURCE / "FREEZE_RECEIPT.json"), "mechanical_completion_only": True, "manifest_sha256": sha(OUT / "MANIFEST.json"), "sha256sums_sha256": sha(OUT / "SHA256SUMS.txt"), "protocol_sha256": sha(OUT / "HR3_PROTOCOL.json"), "parent_sha256": protocol["parent_sha256"], "tokenizer_sha256": protocol["tokenizer_sha256"], "readiness_dev_receipt_sha256": protocol["readiness_dev"]["receipt_sha256"], "checkpoint_loaded": False, "optimizer_created": False, "optimizer_updates": 0, "final_accessed": False, "sacred_accessed": False}
    jwrite(OUT / "FREEZE_RECEIPT.json", receipt)
    write(OUT / "FREEZE_RECEIPT.sha256", sha(OUT / "FREEZE_RECEIPT.json") + "  FREEZE_RECEIPT.json\n")
    readonly(OUT)
    print(json.dumps({"status": receipt["status"], "bundle": str(OUT), "receipt_sha256": sha(OUT / "FREEZE_RECEIPT.json")}, indent=2))


if __name__ == "__main__":
    main()
