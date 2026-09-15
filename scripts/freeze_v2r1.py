from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    files = [
        "configs/foundation_v1.json",
        "data_specs/V010_FOUNDATION_PROTOCOL_V2R1.json",
        "design/V010_ARCHITECTURE.md",
        "design/V010_CURRICULUM_V2R1.md",
        "design/V010_CAPABILITY_GATES.md",
        "src/baby_v010/model.py",
        "src/baby_v010/config.py",
        "src/baby_v010/data.py",
        "src/baby_v010/data_v2.py",
        "src/baby_v010/evaluate.py",
        "src/baby_v010/train_v2r1.py",
        "data/generated/foundation_v2/panels.json",
        "data/generated/foundation_v2/AUDIT.json",
        "data/generated/foundation_v2/MANIFEST.json",
    ]
    receipt = {
        "protocol": "BABY_V010_FOUNDATION_V2R1",
        "lineage": "Baby v0.10",
        "parent_checkpoint": None,
        "protected_material_opened": False,
        "files": {name: digest(ROOT / name) for name in files},
    }
    out = ROOT / "data_specs" / "V010_FOUNDATION_V2R1_FREEZE.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
