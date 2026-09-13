"""Mechanical dependency-completion bundle for sealed HR-3 v1.

It adds the uniquely authoritative Treatment-13 configuration module required by
the copied T13 model import.  It makes no scientific change.
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
SOURCE = ROOT / "human_readiness_hr3_block3_causal_seed87006_v1"
OUT = ROOT / "human_readiness_hr3_block3_causal_seed87006_v2"
AUTH_CONFIG = ROOT / "treatment13_config.py"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text(path: Path, body: str) -> None:
    path.write_text(body.replace("\r\n", "\n"), encoding="utf-8", newline="\n")


def jwrite(path: Path, value: Any) -> None:
    text(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def payload(path: Path) -> list[dict[str, Any]]:
    excluded = {"SHA256SUMS.txt", "FREEZE_RECEIPT.json", "FREEZE_RECEIPT.sha256"}
    return [{"path": f.relative_to(path).as_posix(), "sha256": sha(f), "bytes": f.stat().st_size} for f in sorted(p for p in path.rglob("*") if p.is_file() and p.name not in excluded)]


def verify(path: Path) -> dict:
    expected = (path / "FREEZE_RECEIPT.sha256").read_text(encoding="utf-8").strip().split()[0]
    assert sha(path / "FREEZE_RECEIPT.json") == expected
    receipt = json.loads((path / "FREEZE_RECEIPT.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "HR3_BLOCK3_PREFLIGHT_PASS"
    assert sha(path / "MANIFEST.json") == receipt["manifest_sha256"]
    assert sha(path / "SHA256SUMS.txt") == receipt["sha256sums_sha256"]
    for line in (path / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split("  ", 1)
            assert sha(path / rel) == digest
    return receipt


def readonly(path: Path) -> None:
    for file in path.rglob("*"):
        if file.is_file():
            file.chmod(file.stat().st_mode & ~stat.S_IWRITE)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing v2 bundle: {OUT}")
    prior = verify(SOURCE)
    assert AUTH_CONFIG.exists()
    # Copy only v1 payload, excluding its integrity chain.  The v2 chain is new.
    excluded = {"SHA256SUMS.txt", "FREEZE_RECEIPT.json", "FREEZE_RECEIPT.sha256", "MANIFEST.json", "PREFLIGHT.json"}
    shutil.copytree(SOURCE, OUT, ignore=lambda _dir, names: {name for name in names if name in excluded})
    for file in OUT.rglob("*"):
        if file.is_file():
            file.chmod(file.stat().st_mode | stat.S_IWRITE)

    # Pin the only missing import dependency from its authoritative repository
    # path; the exact bytes are recorded in both protocol and completion audit.
    shutil.copy2(AUTH_CONFIG, OUT / "sources" / "treatment13_config.py")
    shutil.copy2(ROOT / "hr3_block3_train.py", OUT / "sources" / "TRAIN_HR3.py")
    shutil.copy2(ROOT / "hr3_block3_runtime.py", OUT / "sources" / "hr3_block3_runtime.py")
    shutil.copy2(ROOT / "hr3_block3_evaluate.py", OUT / "sources" / "EVALUATE_HR3.py")
    shutil.copy2(ROOT / "validate_hr3_block3_preflight.py", OUT / "sources" / "PREFLIGHT_VALIDATOR.py")
    protocol_path = OUT / "HR3_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["version"] = "HR-3-block3-causal-v2"
    protocol["mechanical_completion"] = {
        "predecessor": str(SOURCE),
        "predecessor_receipt_sha256": sha(SOURCE / "FREEZE_RECEIPT.json"),
        "reason": "T13 model imports treatment13_config.py. The v1 bundle omitted that authoritative configuration module, causing import failure before model/checkpoint load.",
        "authoritative_source": str(AUTH_CONFIG),
        "authoritative_source_sha256": sha(AUTH_CONFIG),
        "scientific_specification_changed": False,
    }
    protocol["frozen_payload_hashes"] = {f.relative_to(OUT).as_posix(): sha(f) for f in sorted(list((OUT / "data").rglob("*")) + list((OUT / "sources").rglob("*"))) if f.is_file()}
    jwrite(protocol_path, protocol)
    completion = {
        "status": "MECHANICAL_COMPLETION_ONLY",
        "predecessor_bundle": str(SOURCE),
        "predecessor_receipt_sha256": sha(SOURCE / "FREEZE_RECEIPT.json"),
        "failure": "ModuleNotFoundError: treatment13_config before checkpoint loading",
        "correction": "pin exact authoritative C:\\DaveLM-CADAVER\\treatment13_config.py as sources/treatment13_config.py",
        "source_sha256": sha(AUTH_CONFIG),
        "scientific_values_changed": False,
        "checkpoint_loaded": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "final_accessed": False,
        "sacred_accessed": False,
    }
    jwrite(OUT / "MECHANICAL_COMPLETION.json", completion)
    text(OUT / "MECHANICAL_COMPLETION.md", "# HR-3 v2 mechanical completion\n\nHR-3 v1 stopped before checkpoint loading because its copied `treatment13_model.py` imports `treatment13_config.py`, which was absent from that bundle. v2 pins the exact authoritative configuration source and adds the pre-parent-load entrypoint check. No scientific payload, parent, dataset, literal schedule, optimizer, scope, objective, gate, or sealed-material rule changed.\n")
    shutil.copy2(Path(__file__), OUT / "BUILD_HR3_V2.py")

    # Independent static/data validation remains the same; it never loads a
    # checkpoint or creates an optimizer.
    subprocess.run([sys.executable, str(OUT / "sources" / "PREFLIGHT_VALIDATOR.py"), "--bundle", str(OUT)], check=True)
    manifest = {"bundle": OUT.name, "payload": payload(OUT)}
    jwrite(OUT / "MANIFEST.json", manifest)
    sums = payload(OUT)
    text(OUT / "SHA256SUMS.txt", "".join(f"{x['sha256']}  {x['path']}\n" for x in sums))
    receipt = {
        "status": "HR3_BLOCK3_PREFLIGHT_PASS",
        "bundle": str(OUT),
        "predecessor_bundle": str(SOURCE),
        "predecessor_receipt_sha256": sha(SOURCE / "FREEZE_RECEIPT.json"),
        "mechanical_completion_only": True,
        "manifest_sha256": sha(OUT / "MANIFEST.json"),
        "sha256sums_sha256": sha(OUT / "SHA256SUMS.txt"),
        "protocol_sha256": sha(OUT / "HR3_PROTOCOL.json"),
        "parent_sha256": protocol["parent_sha256"],
        "tokenizer_sha256": protocol["tokenizer_sha256"],
        "readiness_dev_receipt_sha256": protocol["readiness_dev"]["receipt_sha256"],
        "checkpoint_loaded": False,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "final_accessed": False,
        "sacred_accessed": False,
    }
    jwrite(OUT / "FREEZE_RECEIPT.json", receipt)
    text(OUT / "FREEZE_RECEIPT.sha256", sha(OUT / "FREEZE_RECEIPT.json") + "  FREEZE_RECEIPT.json\n")
    readonly(OUT)
    print(json.dumps({"status": receipt["status"], "bundle": str(OUT), "receipt_sha256": sha(OUT / "FREEZE_RECEIPT.json")}, indent=2))


if __name__ == "__main__":
    main()
