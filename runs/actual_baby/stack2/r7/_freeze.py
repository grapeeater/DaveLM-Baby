"""R7 freeze hashes + official who/has/beside readout. Does not retrain."""

import hashlib
import json
from pathlib import Path

from src.baby_v010.data_language_bridge import build_s3_panels, load_tokenizer
from src.baby_v010.selection_language_bridge import load_experimental_baby
from src.baby_v010.selection_p11_u16000_runtime import resolve_device
from src.baby_v010.selection_s1 import digest
from src.baby_v010.selection_stack2_r3 import (
    S5B3_PARENT,
    S5B3_PARENT_SHA,
    PropMatchRuntime,
    load_prop_match_head,
    score_who_with_prop_match,
    verify_parent,
)
from src.baby_v010.selection_stack2_s5 import CHEAP_OPS, run_beside_decode, run_has_decode, run_who_sentence_decode, summarize_bind

OUT = Path("runs/actual_baby/stack2/r7")
HEAD = Path("runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt")
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
U16000 = Path("U16000/checkpoint_16000.pt")

files = {
    "backbone": S5B3_PARENT,
    "prop_match_head": HEAD,
    "tokenizer": TOKENIZER,
    "u16000": U16000,
    "selection_stack2_r2.py": Path("src/baby_v010/selection_stack2_r2.py"),
    "selection_stack2_r3.py": Path("src/baby_v010/selection_stack2_r3.py"),
    "selection_stack2_r6.py": Path("src/baby_v010/selection_stack2_r6.py"),
    "selection_stack2_r7.py": Path("src/baby_v010/selection_stack2_r7.py"),
    "stack2_chat.py": Path("src/baby_v010/stack2_chat.py"),
    "test_selection_stack2_r3.py": Path("tests/test_selection_stack2_r3.py"),
    "integration_pack": OUT / "INTEGRATION_PACK.json",
    "reuse_pack": OUT / "REUSE_PACK.json",
}

device = resolve_device("cuda")
parent = verify_parent(S5B3_PARENT, S5B3_PARENT_SHA, label="r7freeze")
tok = load_tokenizer()
model, config, ckpt = load_experimental_baby(S5B3_PARENT, device)
assert not ckpt.get("protected_material_opened")
head = load_prop_match_head(HEAD, config.d_model, device)
rt = PropMatchRuntime(model, head, tok).install()
rt.enabled = True
official = build_s3_panels(tok, n=32)["who_bind_2e_heldout"]
who2 = score_who_with_prop_match(model, head, tok, device, official)
who_sent = summarize_bind(run_who_sentence_decode(model, tok, device, include=CHEAP_OPS))
has = summarize_bind(run_has_decode(model, tok, device, include=CHEAP_OPS))
bes = summarize_bind(run_beside_decode(model, tok, device, include=CHEAP_OPS))
rt.uninstall()

import subprocess

git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
out = {
    "git_head": git,
    "git_commit_created": False,
    "relassist_default": False,
    "whoprop_default": False,
    "runtime_default": "r3",
    "protected_opened": False,
    "form_a_used": False,
    "test_final_sacred": "sealed_not_opened",
    "backbone_sha256": parent,
    "head_sha256": digest(HEAD),
    "tokenizer_sha256": digest(TOKENIZER),
    "u16000_sha256": digest(U16000) if U16000.exists() else None,
    "file_sha256": {name: digest(path) for name, path in files.items() if path.exists()},
    "official_who_2e": {k: who2[k] for k in who2 if k != "rows"},
    "who_sent": who_sent,
    "has": has,
    "beside": bes,
    "dirty": [line for line in status.splitlines() if line.strip()],
}
(OUT / "R7_FREEZE.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
print(json.dumps({k: out[k] for k in out if k != "dirty"}, indent=2, default=str))
