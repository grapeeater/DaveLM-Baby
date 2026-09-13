"""Mechanical predecessor-protocol-chain completion for HR-3 v6."""

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
SOURCE = ROOT / "human_readiness_hr3_block3_causal_seed87006_v6"
OUT = ROOT / "human_readiness_hr3_block3_causal_seed87006_v7"
V4 = ROOT / "human_readiness_hr3_block3_causal_seed87006_v4"
V5 = ROOT / "human_readiness_hr3_block3_causal_seed87006_v5"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.replace("\r\n", "\n"), encoding="utf-8", newline="\n")


def jwrite(path: Path, obj: Any) -> None:
    write(path, json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


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


def listed(path: Path) -> list[str]:
    return [line.split("  ", 1)[1] for line in (path / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def payload(path: Path) -> list[dict[str, Any]]:
    excluded = {"SHA256SUMS.txt", "FREEZE_RECEIPT.json", "FREEZE_RECEIPT.sha256"}
    return [{"path": p.relative_to(path).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(x for x in path.rglob("*") if x.is_file() and x.name not in excluded)]


def readonly(path: Path) -> None:
    for p in path.rglob("*"):
        if p.is_file():
            p.chmod(p.stat().st_mode & ~stat.S_IWRITE)


def main() -> None:
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite v7: {OUT}")
    predecessor = verify(SOURCE)
    v4_hash = sha(V4 / "HR3_PROTOCOL.json")
    v5_hash = sha(V5 / "HR3_PROTOCOL.json")
    v6_hash = sha(SOURCE / "HR3_PROTOCOL.json")
    OUT.mkdir(parents=True)
    skip = {"PREFLIGHT.json", "HR3_PROTOCOL.json", "MECHANICAL_COMPLETION.json", "MECHANICAL_COMPLETION.md"}
    for rel in listed(SOURCE):
        if rel in skip or rel.startswith("sources/__pycache__/"):
            continue
        src, dst = SOURCE / rel, OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        dst.chmod(dst.stat().st_mode | stat.S_IWRITE)
    for source_name, target_name in [("hr3_block3_runtime.py", "hr3_block3_runtime.py"), ("hr3_block3_train.py", "TRAIN_HR3.py"), ("hr3_block3_evaluate.py", "EVALUATE_HR3.py"), ("validate_hr3_block3_preflight.py", "PREFLIGHT_VALIDATOR.py")]:
        shutil.copy2(ROOT / source_name, OUT / "sources" / target_name)
    protocol = json.loads((SOURCE / "HR3_PROTOCOL.json").read_text(encoding="utf-8"))
    protocol["version"] = "HR-3-block3-causal-v7"
    protocol["mechanical_completion"] = {
        "predecessor": str(SOURCE),
        "predecessor_receipt_sha256": sha(SOURCE / "FREEZE_RECEIPT.json"),
        "reason": "The only valid committed restart state is update 297 written under the v4 protocol hash; v5 and v6 resume attempts stopped before a new commit.",
        "correction": "Accept exactly the finite predecessor protocol-hash chain v6, v5, and v4 for restart provenance; all current payload/data/schedule/parent checks remain mandatory.",
        "accepted_predecessor_protocol_sha256s": [v6_hash, v5_hash, v4_hash],
        "scientific_specification_changed": False,
    }
    protocol["frozen_payload_hashes"] = {f.relative_to(OUT).as_posix(): sha(f) for f in sorted(list((OUT / "data").rglob("*")) + list((OUT / "sources").rglob("*"))) if f.is_file()}
    jwrite(OUT / "HR3_PROTOCOL.json", protocol)
    jwrite(OUT / "MECHANICAL_COMPLETION.json", {"status": "MECHANICAL_PREDECESSOR_PROTOCOL_CHAIN_COMPLETION_ONLY", "predecessor_bundle": str(SOURCE), "predecessor_receipt_sha256": sha(SOURCE / "FREEZE_RECEIPT.json"), "valid_restart_state": str(ROOT / "human_readiness_hr3_block3_causal_seed87006_execution_v4"), "valid_restart_completed_update": 297, "accepted_protocol_hashes": [v6_hash, v5_hash, v4_hash], "scientific_values_changed": False, "checkpoint_loaded_during_preflight": False, "optimizer_created_during_preflight": False, "optimizer_updates_during_preflight": 0, "final_accessed": False, "sacred_accessed": False})
    write(OUT / "MECHANICAL_COMPLETION.md", "# HR-3 v7 predecessor protocol-chain completion\n\nThe only valid committed restart is update 297 under the v4 protocol hash. v5 and v6 stopped before any new commit while correcting RNG restoration. v7 accepts exactly the finite v4/v5/v6 predecessor hashes, verifies every current payload and schedule identity, and resumes at update 298. This is mechanical provenance wiring only.\n")
    shutil.copy2(Path(__file__), OUT / "BUILD_HR3_V7.py")
    subprocess.run([sys.executable, str(OUT / "sources" / "PREFLIGHT_VALIDATOR.py"), "--bundle", str(OUT)], check=True)
    jwrite(OUT / "MANIFEST.json", {"bundle": OUT.name, "payload": payload(OUT)})
    values = payload(OUT)
    write(OUT / "SHA256SUMS.txt", "".join(f"{item['sha256']}  {item['path']}\n" for item in values))
    receipt = {"status": "HR3_BLOCK3_PREFLIGHT_PASS", "bundle": str(OUT), "predecessor_bundle": str(SOURCE), "predecessor_receipt_sha256": sha(SOURCE / "FREEZE_RECEIPT.json"), "mechanical_completion_only": True, "manifest_sha256": sha(OUT / "MANIFEST.json"), "sha256sums_sha256": sha(OUT / "SHA256SUMS.txt"), "protocol_sha256": sha(OUT / "HR3_PROTOCOL.json"), "parent_sha256": protocol["parent_sha256"], "tokenizer_sha256": protocol["tokenizer_sha256"], "readiness_dev_receipt_sha256": protocol["readiness_dev"]["receipt_sha256"], "accepted_predecessor_protocol_sha256s": [v6_hash, v5_hash, v4_hash], "checkpoint_loaded": False, "optimizer_created": False, "optimizer_updates": 0, "final_accessed": False, "sacred_accessed": False}
    jwrite(OUT / "FREEZE_RECEIPT.json", receipt)
    write(OUT / "FREEZE_RECEIPT.sha256", sha(OUT / "FREEZE_RECEIPT.json") + "  FREEZE_RECEIPT.json\n")
    readonly(OUT)
    print(json.dumps({"status": receipt["status"], "bundle": str(OUT), "receipt_sha256": sha(OUT / "FREEZE_RECEIPT.json")}, indent=2))


if __name__ == "__main__":
    main()
