"""Write the R7 integration pack once, then stop. Do not eval here."""

import hashlib
import json
from pathlib import Path

from src.baby_v010.selection_stack2_r7 import build_r7_integration_pack, build_r7_reuse_pack

OUT = Path("runs/actual_baby/stack2/r7")
OUT.mkdir(parents=True, exist_ok=True)

pack = build_r7_integration_pack()
reuse = build_r7_reuse_pack()
pack_path = OUT / "INTEGRATION_PACK.json"
reuse_path = OUT / "REUSE_PACK.json"
pack_path.write_text(json.dumps(pack, indent=2) + "\n")
reuse_path.write_text(json.dumps(reuse, indent=2) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = {
    "frozen_before_eval": True,
    "form_a_used": False,
    "protected_opened": False,
    "n_integration": len(pack),
    "n_reuse_scripts": len(reuse),
    "n_reuse_turns": sum(len(s["turns"]) for s in reuse),
    "families": sorted({item["family"] for item in pack}),
    "entities": sorted({item["entity"] for item in pack if item.get("entity")}),
    "integration_path": str(pack_path),
    "integration_sha256": sha256(pack_path),
    "reuse_path": str(reuse_path),
    "reuse_sha256": sha256(reuse_path),
    "scoring": "entity-first: inverse WHO/HAS/BESIDE fail if first bridge entity != gold entity",
    "note": "Pack is frozen. Later failures must be diagnosed with separate canaries. Do not edit these JSON files after eval.",
}
(OUT / "INTEGRATION_PACK_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
