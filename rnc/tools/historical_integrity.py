"""Verify that the historical Baby artefacts are byte-identical to their frozen records.

Run before and after any RNC work that touches the machine. Read-only. Exit 0 = every artefact present and matching.
The expected hashes come from the frozen records in the MRCN-Alpha laboratory (GRADUATE_SHA256SUMS.txt,
DAVELM_ECOSYSTEM.md, checkpoints/v1.0_pristine/SHA256SUMS.txt) and from C:\\DaveLM_ARCHIVE\\ARCHIVE_SHA256SUMS.txt.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

LAB = Path(r"C:\DaveLM_ARCHIVE\MRCN-Alpha_LAB_20261016")  # the MRCN-Alpha laboratory, relocated 2026-10-16
ARCHIVE = Path(r"C:\DaveLM_ARCHIVE")
V1 = Path(r"C:\DaveLM-v1.0")  # restored 2026-10-16 from the verified archive copies

# (label, candidate paths in priority order, sha256)
ARTEFACTS = [
    ("S1 118M graduate (MRCN-Alpha canonical)",
     [LAB / r"rebuild\baby_reincarnation_118m_001\runs\S1v3j_single_token_copy_anneal_from_s1v3h_001\S1_single_token_copy_graduated.pt"],
     "200e02188063885560edb4de0d6e0f05054fa75c236c0264d4a2c461ec51771b"),
    ("61M v1.0 backbone checkpoint_00025.pt (pristine copy)",
     [LAB / r"checkpoints\v1.0_pristine\checkpoint_00025.pt"],
     "0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd"),
    ("61M v1.0 backbone checkpoint_00025.pt (runtime path)",
     [LAB / r"runs\actual_baby\stack2\s5b3_compose_lock_326191\checkpoint_00025.pt"],
     "0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd"),
    ("61M v1.0 backbone (archive museum copy)",
     [ARCHIVE / r"DaveLM_v1.0_GRADUATE\weights\checkpoint_00025.pt"],
     "0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd"),
    ("61M v1.0 PropMatchHead",
     [LAB / r"checkpoints\v1.0_pristine\prop_match_head_00050.pt"],
     "8ea3dbfd4825058eea74626b07621ea8c0155a960fccf1530c9123f61f7d62d9"),
    ("DaveLM tokenizer v0.7",
     [LAB / r"tokenizer\v0_7\davelm_tokenizer.json"],
     "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"),
    # Frozen Form B (archive museum copies carry the frozen LF hashes).
    ("Form B RAW.json (archive, frozen)",
     [ARCHIVE / r"DaveLM_v1.0_GRADUATE\form_b\FINAL_EXAM_FORM_B_RAW.json"],
     "67aa03a48d41a411547153df1b507f890cc96e5ad8d4625e804f417a24452a96"),
    ("Form B RAW.txt (archive, frozen)",
     [ARCHIVE / r"DaveLM_v1.0_GRADUATE\form_b\FINAL_EXAM_FORM_B_RAW.txt"],
     "85372e2bfcae8cf9b2aa99a0d3049356b7ea4825617bd734c8c2012217bae7df"),
    ("Form B MANIFEST.json (archive, frozen)",
     [ARCHIVE / r"DaveLM_v1.0_GRADUATE\form_b\FINAL_EXAM_FORM_B_MANIFEST.json"],
     "2fa1184c7837dee4f8aadf0557a42288f5b7fcae0f2d8198646ad53f61852a66"),
    # The DaveLM-Training working copies of Form B are CRLF checkouts of the same bytes (identical after LF
    # normalisation, verified 2026-10-16); their CRLF hashes are pinned here so any later change is detected.
    ("Form B RAW.json (Training working copy, CRLF checkout)",
     [LAB / r"runs\actual_baby\final_exam_v1\form_b\FINAL_EXAM_FORM_B_RAW.json"],
     "1e31c9166dd3"),
    ("Form B RAW.txt (Training working copy, CRLF checkout)",
     [LAB / r"runs\actual_baby\final_exam_v1\form_b\FINAL_EXAM_FORM_B_RAW.txt"],
     "423854a4642d"),
    ("Form B MANIFEST.json (Training working copy, CRLF checkout)",
     [LAB / r"runs\actual_baby\final_exam_v1\form_b\FINAL_EXAM_FORM_B_MANIFEST.json"],
     "cf931f3339db"),
    ("selection_stack2_r3.py (archive, frozen)",
     [ARCHIVE / r"DaveLM_v1.0_GRADUATE\source\baby_v010\selection_stack2_r3.py"],
     "e7bd0b3006c22289182a4f5b9093b3c84606040f09ebfd595b747023aa852b33"),
    ("selection_stack2_r3.py (Training working copy; NS1 open-bind overlay per LEDGER.md, 97,583 B)",
     [LAB / r"src\baby_v010\selection_stack2_r3.py"],
     "SIZE:97583"),
    ("selection_stack2_r7.py (frozen runtime)",
     [LAB / r"src\baby_v010\selection_stack2_r7.py"],
     "7520622c24f54c7d3aac7ed790403cff2e84b153a4593a149278ce45f3f7f1c0"),
    ("Restored C:\\DaveLM-v1.0 backbone", [V1 / r"runs\actual_baby\stack2\s5b3_compose_lock_326191\checkpoint_00025.pt"],
     "0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd"),
    ("Restored C:\\DaveLM-v1.0 PropMatchHead", [V1 / r"runs\actual_baby\stack2\r3\r3b_329011\prop_match_head_00050.pt"],
     "8ea3dbfd4825058eea74626b07621ea8c0155a960fccf1530c9123f61f7d62d9"),
    ("Restored C:\\DaveLM-v1.0 tokenizer v0.7", [V1 / r"tokenizer\v0_7\davelm_tokenizer.json"],
     "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"),
    ("Restored C:\\DaveLM-v1.0 selection_stack2_r3.py", [V1 / r"src\baby_v010\selection_stack2_r3.py"],
     "e7bd0b3006c22289182a4f5b9093b3c84606040f09ebfd595b747023aa852b33"),
    ("Restored C:\\DaveLM-v1.0 Form B RAW.json", [V1 / r"runs\actual_baby\final_exam_v1\form_b\FINAL_EXAM_FORM_B_RAW.json"],
     "67aa03a48d41a411547153df1b507f890cc96e5ad8d4625e804f417a24452a96"),
    ("Oct-11 RESTORE: Baby-Cloud-Lab unique evidence sample (latest_healthy.pt)",
     [ARCHIVE / r"POST_S1_TO_B2_20261011_COPY01\RESTORE\VERIFIED_TREE\drive_C\Baby-Cloud-Lab\runs\BABY_FIRST_WORDS_002_CLOUD_001\returned_evidence\run\checkpoints\latest_healthy.pt"],
     None),
    ("S1v3j u4700 scheduled checkpoint (graduation window)",
     [LAB / r"rebuild\baby_reincarnation_118m_001\runs\S1v3j_single_token_copy_anneal_from_s1v3h_001\scheduled_checkpoints\u4700\latest_resumable.pt"],
     None),  # no frozen record found; presence + size recorded only
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main(out: str | None = None) -> int:
    rows, ok = [], True
    for label, paths, expected in ARTEFACTS:
        found = next((p for p in paths if p.exists()), None)
        if found is None:
            rows.append({"artefact": label, "status": "MISSING", "paths_tried": [str(p) for p in paths]})
            ok = False
            continue
        actual = sha256(found)
        if expected is None:
            status = "MATCH"
        elif expected.startswith("SIZE:"):
            status = "MATCH" if found.stat().st_size == int(expected[5:]) else "MISMATCH"
        else:
            status = "MATCH" if actual.startswith(expected) else "MISMATCH"
        if status == "MISMATCH":
            ok = False
        rows.append({"artefact": label, "path": str(found), "bytes": found.stat().st_size, "sha256": actual,
                     "expected": expected, "status": status if expected else "PRESENT (no frozen hash)"})
        print(f"{rows[-1]['status']:<26} {label}")
    report = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "all_ok": ok, "artefacts": rows}
    if out:
        Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("ALL OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
