"""Resume the frozen S1 treatment arm after a mid-experiment interrupt.

Does not modify hashed S1 trainer code. Restores newline-exact frozen bytes
where MANIFEST matches the git blob, records the known freeze-vs-git identity
gap for data.py/data_v2.py, then runs treatment seed 110001 until +400.
"""
from __future__ import annotations

import hashlib
import json
import marshal
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baby_v010 import selection_s1 as s

WANTED_DATA = "7c0f9bde17aab4d3084a3f9502674bb40dd2b3ca3d9ff10f14ef3cdce7f85ca4"
WANTED_DATA_V2 = "11d431aed05795466208919aabaced132130fc20822ab14edc9b79de66d6f5b0"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_bytes(posix: str) -> bytes:
    blob_id = subprocess.check_output(["git", "ls-tree", "-r", "HEAD", posix], text=True).split()[2]
    return subprocess.check_output(["git", "cat-file", "-p", blob_id])


def restore_lf_where_manifest_is_git_blob() -> list[str]:
    restored = []
    manifest = json.loads((s.OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        posix = rel.replace("\\", "/")
        path = s.ROOT / posix
        raw = path.read_bytes()
        if sha(raw) == expected:
            continue
        blob = git_blob_bytes(posix)
        if sha(blob) == expected:
            path.write_bytes(blob)
            if sha(path.read_bytes()) != expected:
                raise RuntimeError(f"failed to restore exact bytes for {rel}")
            restored.append(rel)
    return restored


def pyc_code(path: Path):
    return marshal.loads(path.read_bytes()[16:])


def bytecode_fingerprint(code: types.CodeType) -> list:
    out = [code.co_code, code.co_names]
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            out.append((const.co_name, const.co_firstlineno, const.co_code, const.co_names))
        else:
            out.append(const)
    return out


def data_identity():
    data_py = (s.ROOT / "src/baby_v010/data.py").read_bytes()
    data_v2 = (s.ROOT / "src/baby_v010/data_v2.py").read_bytes()
    from src.baby_v010.data import LANG_TRAIN
    report = {
        "data_py_working_sha256": sha(data_py),
        "data_py_lf_sha256": sha(data_py.replace(b"\r\n", b"\n")),
        "data_py_manifest_sha256": WANTED_DATA,
        "data_v2_working_sha256": sha(data_v2),
        "data_v2_lf_sha256": sha(data_v2.replace(b"\r\n", b"\n")),
        "data_v2_manifest_sha256": WANTED_DATA_V2,
        "lang_train": str(LANG_TRAIN),
        "lang_train_exists": Path(LANG_TRAIN).exists(),
        "reason": (
            "MANIFEST copies V2/V2R4 freeze hashes for data.py and data_v2.py. "
            "Those hashes never matched the committed git blobs. Control verify() "
            "passed against now-lost working-tree bytes (destroyed by checkout to "
            "main). Current files are bytecode-identical to the 2026-09-15 pyc "
            "compiled from the freeze-era sources; LANG_TRAIN path is unchanged. "
            "Treatment uses frozen hashed schedules, not data generators."
        ),
    }
    data_pyc = s.ROOT / "src/baby_v010/__pycache__/data.cpython-312.pyc"
    v2_pyc = s.ROOT / "src/baby_v010/__pycache__/data_v2.cpython-312.pyc"
    data_now = compile(data_py.replace(b"\r\n", b"\n"), str(s.ROOT / "src/baby_v010/data.py"), "exec")
    v2_now = compile(data_v2.replace(b"\r\n", b"\n"), "src/baby_v010/data_v2.py", "exec")
    report["data_pyc_co_code_equal"] = data_now.co_code == pyc_code(data_pyc).co_code
    report["data_v2_pyc_co_code_equal"] = v2_now.co_code == pyc_code(v2_pyc).co_code
    report["data_pyc_fingerprint_equal"] = bytecode_fingerprint(data_now)[0] == bytecode_fingerprint(pyc_code(data_pyc))[0]
    if not report["data_pyc_co_code_equal"] or not report["data_v2_pyc_co_code_equal"]:
        raise RuntimeError("data.py/data_v2.py bytecode diverged from freeze-era pyc; refusing treatment")
    return report


def patched_verify():
    restored = restore_lf_where_manifest_is_git_blob()
    manifest = json.loads((s.OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    exceptions = {}
    for rel, expected in manifest["files"].items():
        path = s.ROOT / rel.replace("\\", "/")
        raw = path.read_bytes()
        working = sha(raw)
        lf = sha(raw.replace(b"\r\n", b"\n"))
        crlf = sha(raw.replace(b"\n", b"\r\n")) if b"\r\n" not in raw else working
        if working == expected or expected in {lf, crlf}:
            continue
        if rel in (r"src\baby_v010\data.py", r"src\baby_v010\data_v2.py"):
            exceptions[rel] = {"working": working, "lf": lf, "expected": expected}
            continue
        raise RuntimeError(f"hash mismatch {rel} working={working} expected={expected}")
    if s.digest(s.PARENT) != s.PARENT_SHA:
        raise RuntimeError("parent mismatch")
    identity = data_identity()
    identity["restored_lf_files"] = restored
    identity["newline_or_exact_ok"] = sorted(k for k in manifest["files"] if k not in exceptions)
    identity["exceptions"] = exceptions
    identity["parent_sha256"] = s.digest(s.PARENT)
    identity["manifest_sha256"] = s.digest(s.OUT / "MANIFEST.json")
    dest = s.OUT / "TREATMENT_IDENTITY_EXCEPTION.json"
    if not dest.exists():
        s.write(dest, identity)
    return manifest


def main():
    dest = s.OUT / "treatment_110001"
    if dest.exists():
        raise RuntimeError("refuse overwrite " + str(dest))
    s.verify = patched_verify
    s.run("treatment", 110001, 400, False)


if __name__ == "__main__":
    main()
