"""Eval the frozen R7 pack. Never rewrite the pack files."""

import hashlib
import json
from pathlib import Path

from src.baby_v010.selection_language_bridge import load_experimental_baby
from src.baby_v010.selection_p11_u16000_runtime import resolve_device
from src.baby_v010.selection_stack2_r3 import (
    S5B3_PARENT,
    S5B3_PARENT_SHA,
    PropMatchRuntime,
    load_prop_match_head,
    verify_parent,
)
from src.baby_v010.selection_stack2_r7 import score_r7_items, score_r7_reuse

OUT = Path("runs/actual_baby/stack2/r7")
HEAD = Path("runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt")
PACK = OUT / "INTEGRATION_PACK.json"
REUSE = OUT / "REUSE_PACK.json"
MANIFEST = json.loads((OUT / "INTEGRATION_PACK_MANIFEST.json").read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


got_pack = sha256(PACK)
got_reuse = sha256(REUSE)
if got_pack != MANIFEST["integration_sha256"]:
    raise SystemExit(f"pack hash changed: {got_pack}")
if got_reuse != MANIFEST["reuse_sha256"]:
    raise SystemExit(f"reuse hash changed: {got_reuse}")

device = resolve_device("cuda")
sha = verify_parent(S5B3_PARENT, S5B3_PARENT_SHA, label="r7pack")
from src.baby_v010.data_language_bridge import load_tokenizer

tok = load_tokenizer()
model, config, ckpt = load_experimental_baby(S5B3_PARENT, device)
assert not ckpt.get("protected_material_opened")
head = load_prop_match_head(HEAD, config.d_model, device)
rt = PropMatchRuntime(model, head, tok).install()
rt.enabled = True
pack = json.loads(PACK.read_text())
reuse = json.loads(REUSE.read_text())
integ = score_r7_items(model, tok, device, pack)
reuse_scored = score_r7_reuse(model, tok, device, reuse)
rt.uninstall()
out = {
    "parent": sha,
    "pack_sha256": got_pack,
    "reuse_sha256": got_reuse,
    "frozen_before_eval": True,
    "form_a_used": False,
    "relassist": False,
    "whoprop": False,
    "integration": {k: integ[k] for k in ("n", "overall", "entity_lead", "by_family")},
    "integration_miss": integ["miss"],
    "reuse": {k: reuse_scored[k] for k in ("n", "ok", "reuse", "stop")},
    "reuse_miss": reuse_scored["miss"],
    "rows": integ["rows"],
}
(OUT / "INTEGRATION_EVAL.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: out[k] for k in out if k != "rows"}, indent=2))
