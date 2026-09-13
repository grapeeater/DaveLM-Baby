import argparse, hashlib, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDED_FILES = {"PRETRAIN_SHA256SUMS.txt", "FREEZE_RECEIPT.json", "FREEZE_RECEIPT.sha256",
                  "SHA256SUMS.txt", "RESULTS.jsonl", "FINAL_REPORT.md"}
EXCLUDED_DIRS = {"run", "preflight_baseline", "__pycache__"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", required=True)
    a = ap.parse_args()
    rows = []
    for path in sorted(ROOT.rglob("*"), key=lambda x: x.relative_to(ROOT).as_posix()):
        rel = path.relative_to(ROOT)
        if not path.is_file() or path.name in EXCLUDED_FILES or any(x in EXCLUDED_DIRS for x in rel.parts):
            continue
        rows.append(f"{sha(path)}  {rel.as_posix()}")
    manifest = ROOT / "PRETRAIN_SHA256SUMS.txt"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    receipt = {"status": a.status, "run_id": "sf3_prospective_treatment_v1",
               "manifest_sha256": sha(manifest), "payload_file_count": len(rows),
               "integrity_chain": "FREEZE_RECEIPT.sha256 -> FREEZE_RECEIPT.json -> PRETRAIN_SHA256SUMS.txt -> payload",
               "scientific_change_from_sf2": "post-update-100 English learning-rate sequence only"}
    rp = ROOT / "FREEZE_RECEIPT.json"
    rp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
    (ROOT / "FREEZE_RECEIPT.sha256").write_text(sha(rp) + "  FREEZE_RECEIPT.json\n",
                                                  encoding="utf-8", newline="\n")
    print(json.dumps({"receipt_sha256": sha(rp), **receipt}, indent=2))


if __name__ == "__main__":
    main()
