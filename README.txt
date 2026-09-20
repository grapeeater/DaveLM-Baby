DaveLM v1.0 — Self-Contained Graduate (inference-only)
=========================================================
Git commit: 526b749eb8eca74e82e29f925048b51fd24d9ae7
Frozen: 2026-09-19

This tree runs the stack2 s5b3 + R3 PropMatchHead graduate with NO dependency
on C:\DaveLM-v0.9, C:\DaveLM-v0.10, or C:\DaveLM-CADAVER paths.

Runtime defaults: --runtime r3, RelAssist OFF, WhoProp OFF.

HOW TO RUN
----------
1. Python 3.12+ with torch and tokenizers (see requirements.txt).
   Known working venv (if still present): C:\DaveLM-CADAVER\sf2_runtime\sf2venv
2. From this directory:
     .\run_chat.ps1
   Or:
     $env:PYTHONPATH = "C:\DaveLM-v1.0\src"
     python -m baby_v010.stack2_chat --runtime r3 --device cuda
3. Smoke test:
     .\run_chat.ps1 --smoke
4. Verify canary (5 mechanism probes, not Form A/B):
     python scripts\verify_canary.py

LAYOUT
------
src/baby_v010/          Full baby_v010 package (inference + eval helpers)
runs/actual_baby/stack2/s5b3_compose_lock_326191/checkpoint_00025.pt  (backbone)
runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt       (PropMatchHead)
tokenizer/v0_7/davelm_tokenizer.json                                   (frozen tokenizer)
runs/actual_baby/final_exam_v1/form_b/                                 (Form B evidence)
scripts/verify_canary.py
VERIFY_CANARY.json      Last canary run output
GRADUATE_SHA256SUMS.txt Frozen artifact hashes
MANIFEST.txt            File inventory

PATH CHANGES (launcher/config only — frozen runtime bytes unchanged)
--------------------------------------------------------------------
data_language_bridge.py: TOKENIZER_PATH -> ROOT/tokenizer/v0_7/davelm_tokenizer.json
  (was hardcoded C:\DaveLM-v0.9\...; tokenizer bytes identical, hash verified)
selection_stack2_r3.py, selection_stack2_r7.py: COPIED VERBATIM (hashes match)
stack2_chat.py: DEFAULT_CHECKPOINT/HEAD use ROOT-relative paths (unchanged logic)

ARCHIVE
-------
C:\DaveLM_ARCHIVE\ holds graduation archive, git bundle, docs, graduate copies.
