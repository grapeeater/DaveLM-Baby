from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "configs" / "foundation_v1.json",
    ROOT / "data_specs" / "V010_FOUNDATION_PROTOCOL.json",
    ROOT / "design" / "V010_ARCHITECTURE.md",
    ROOT / "design" / "V010_CURRICULUM.md",
    ROOT / "design" / "V010_CAPABILITY_GATES.md",
    ROOT / "src" / "baby_v010" / "model.py",
    ROOT / "src" / "baby_v010" / "config.py",
    ROOT / "src" / "baby_v010" / "data.py",
    ROOT / "src" / "baby_v010" / "audit.py",
    ROOT / "src" / "baby_v010" / "evaluate.py",
    ROOT / "src" / "baby_v010" / "train.py",
    ROOT / "data" / "generated" / "foundation_v1" / "panels.json",
    ROOT / "data" / "generated" / "foundation_v1" / "AUDIT_RECHECK.json",
    ROOT / "data" / "generated" / "foundation_v1" / "MANIFEST.json",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    missing = [str(path) for path in FILES if not path.exists()]
    if missing:
        raise SystemExit(f"missing freeze inputs: {missing}")
    record = {
        "protocol": "BABY_V010_FOUNDATION_V1",
        "lineage": "Baby v0.10",
        "frozen_at": "2026-09-15",
        "parent_checkpoint": None,
        "protected_material_opened": False,
        "files": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in FILES},
    }
    out = ROOT / "data_specs" / "V010_FOUNDATION_FREEZE.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
