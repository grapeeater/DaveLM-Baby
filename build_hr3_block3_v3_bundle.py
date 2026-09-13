"""Mechanical immutability completion for HR-3 v2.

v2's post-seal import check wrote a Python bytecode cache beside a sealed source.
This builder creates a clean v3 from v2's listed payload only and freezes the
required no-bytecode launch policy.  No scientific treatment value changes.
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
SOURCE = ROOT / "human_readiness_hr3_block3_causal_seed87006_v2"
OUT = ROOT / "human_readiness_hr3_block3_causal_seed87006_v3"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.replace("\r\n", "\n"), encoding="utf-8", newline="\n")


def jwrite(path: Path, value: Any) -> None:
    write(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def verify(path: Path) -> dict:
    detached = (path / "FREEZE_RECEIPT.sha256").read_text(encoding="utf-8").strip().split()[0]
    assert sha(path / "FREEZE_RECEIPT.json") == detached
    receipt = json.loads((path / "FREEZE_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "HR3_BLOCK3_PREFLIGHT_PASS"
    assert sha(path / "MANIFEST.json") == receipt["manifest_sha256"]
    assert sha(path / "SHA256SUMS.txt") == receipt["sha256sums_sha256"]
    for line in (path / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            assert sha(path / rel) == digest
    return receipt


def listed_payloads(path: Path) -> list[str]:
    return [line.split("  ", 1)[1] for line in (path / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def manifest(path: Path) -> list[dict[str, Any]]:
    excluded = {"SHA256SUMS.txt", "FREEZE_RECEIPT.json", "FREEZE_RECEIPT.sha256"}
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.name not in excluded)
    return [{"path": p.relative_to(path).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size} for p in files]


def make_readonly(path: Path) -> None:
    for file in path.rglob("*"):
        if file.is_file():
            file.chmod(file.stat().st_mode & ~stat.S_IWRITE)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite v3: {OUT}")
    predecessor = verify(SOURCE)
    OUT.mkdir(parents=True)
    # Only files named in v2's checksum manifest are copied.  The unmanifested
    # bytecode file remains evidence in v2 and cannot enter v3.
    for rel in listed_payloads(SOURCE):
        if rel in {"PREFLIGHT.json", "HR3_PROTOCOL.json", "MECHANICAL_COMPLETION.json", "MECHANICAL_COMPLETION.md"}:
            continue
        origin, target = SOURCE / rel, OUT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        target.chmod(target.stat().st_mode | stat.S_IWRITE)

    # Retain the original scientific protocol exactly, adding only an
    # executable immutability instruction and predecessor provenance.
    protocol = json.loads((SOURCE / "HR3_PROTOCOL.json").read_text(encoding="utf-8"))
    protocol["version"] = "HR-3-block3-causal-v3"
    protocol["mechanical_completion"] = {
        "predecessor": str(SOURCE),
        "predecessor_receipt_sha256": sha(SOURCE / "FREEZE_RECEIPT.json"),
        "reason": "v2 post-seal module import created an unmanifested __pycache__ artifact.",
        "correction": "All HR-3 execution commands must use Python -B and PYTHONDONTWRITEBYTECODE=1; v3 is copied only from v2's checksum-listed payload.",
        "scientific_specification_changed": False,
    }
    protocol["training"]["runtime"]["python_bytecode"] = "PYTHONDONTWRITEBYTECODE=1 and python -B are required for all frozen-bundle execution, validation, training, and evaluation commands"
    protocol["frozen_payload_hashes"] = {f.relative_to(OUT).as_posix(): sha(f) for f in sorted(list((OUT / "data").rglob("*")) + list((OUT / "sources").rglob("*"))) if f.is_file()}
    jwrite(OUT / "HR3_PROTOCOL.json", protocol)
    jwrite(OUT / "MECHANICAL_COMPLETION.json", {"status": "MECHANICAL_IMMUTABILITY_COMPLETION_ONLY", "predecessor_bundle": str(SOURCE), "predecessor_receipt_sha256": sha(SOURCE / "FREEZE_RECEIPT.json"), "scientific_values_changed": False, "checkpoint_loaded": False, "optimizer_created": False, "optimizer_updates": 0, "final_accessed": False, "sacred_accessed": False})
    write(OUT / "MECHANICAL_COMPLETION.md", "# HR-3 v3 mechanical immutability completion\n\nv2's final post-seal import check successfully reached the parent-loading boundary but caused Python to create an unmanifested bytecode cache under the sealed bundle. v3 copies only v2 checksum-listed payload files and requires `PYTHONDONTWRITEBYTECODE=1` plus Python `-B` on every frozen-bundle command. This is an execution-environment fix only; all scientific values are unchanged.\n")
    shutil.copy2(Path(__file__), OUT / "BUILD_HR3_V3.py")

    subprocess.run([sys.executable, str(OUT / "sources" / "PREFLIGHT_VALIDATOR.py"), "--bundle", str(OUT)], check=True)
    jwrite(OUT / "MANIFEST.json", {"bundle": OUT.name, "payload": manifest(OUT)})
    values = manifest(OUT)
    write(OUT / "SHA256SUMS.txt", "".join(f"{x['sha256']}  {x['path']}\n" for x in values))
    receipt = {"status": "HR3_BLOCK3_PREFLIGHT_PASS", "bundle": str(OUT), "predecessor_bundle": str(SOURCE), "predecessor_receipt_sha256": sha(SOURCE / "FREEZE_RECEIPT.json"), "mechanical_completion_only": True, "manifest_sha256": sha(OUT / "MANIFEST.json"), "sha256sums_sha256": sha(OUT / "SHA256SUMS.txt"), "protocol_sha256": sha(OUT / "HR3_PROTOCOL.json"), "parent_sha256": protocol["parent_sha256"], "tokenizer_sha256": protocol["tokenizer_sha256"], "readiness_dev_receipt_sha256": protocol["readiness_dev"]["receipt_sha256"], "checkpoint_loaded": False, "optimizer_created": False, "optimizer_updates": 0, "final_accessed": False, "sacred_accessed": False}
    jwrite(OUT / "FREEZE_RECEIPT.json", receipt)
    write(OUT / "FREEZE_RECEIPT.sha256", sha(OUT / "FREEZE_RECEIPT.json") + "  FREEZE_RECEIPT.json\n")
    make_readonly(OUT)
    print(json.dumps({"status": receipt["status"], "bundle": str(OUT), "receipt_sha256": sha(OUT / "FREEZE_RECEIPT.json")}, indent=2))


if __name__ == "__main__":
    main()
