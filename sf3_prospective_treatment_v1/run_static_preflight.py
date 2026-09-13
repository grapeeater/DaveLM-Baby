import ast, collections, hashlib, json, os, platform, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "frozen_base"
sys.path.insert(0, str(ROOT / "sources"))
import torch
import tokenizers
from tokenizers import Tokenizer
import hr3_block3_runtime as rt
from PINNED_MASKING import prepare_example, pad_batch, set_scope


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_manifest(bundle):
    rec = read(bundle / "RECEIPT.json")
    assert sha(bundle / "RECEIPT.json") == (bundle / "RECEIPT.sha256").read_text().split()[0]
    assert sha(bundle / "SHA256SUMS.txt") == rec["manifest_sha256"]
    checked = 0
    for line in (bundle / "SHA256SUMS.txt").read_text().splitlines():
        h, n = line.split("  ", 1)
        assert sha(bundle / n) == h, n
        checked += 1
    return checked


def main():
    p = read(ROOT / "PROTOCOL.json")
    checks = {}
    checks["copied_sf1_manifest_files"] = verify_manifest(BASE)
    assert sha(p["parent"]["path"]) == p["parent"]["sha256"]
    assert sha(p["tokenizer"]["path"]) == p["tokenizer"]["sha256"]
    assert sha(p["sf2_update100"]["path"]) == p["sf2_update100"]["sha256"]
    assert sha(p["d3"]["selection_manifest"]) == p["d3"]["selection_sha256"]
    assert sha(ROOT / "KL_POOL.json") == p["kl"]["pool_sha256"]
    assert sha(ROOT / "KL_POOL_MANIFEST.json") == p["kl"]["pool_manifest_sha256"]
    checks["external_identities"] = "PASS"

    assert platform.python_version() == p["runtime"]["python"]
    assert torch.__version__ == p["runtime"]["torch"]
    assert tokenizers.__version__ == p["runtime"]["tokenizers"]
    assert torch.cuda.is_available()
    checks["runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
                         "tokenizers": tokenizers.__version__, "device": torch.cuda.get_device_name(0)}

    train = read(BASE / "TRAIN.json")
    schedule = read(BASE / "SCHEDULE.json")
    assert len(train) == 16 and len(schedule) == 200
    assert [u["update"] for u in schedule] == list(range(1, 201))
    assert sum(u["kind"] == "english" for u in schedule) == 180
    assert sum(u["kind"] == "binding" for u in schedule) == 20
    for cycle in range(20):
        assert [u["kind"] for u in schedule[cycle*10:(cycle+1)*10]] == ["english"]*9 + ["binding"]
    ids = [r["id"] for r in train]
    assert len(ids) == len(set(ids))
    for u in schedule:
        if u["kind"] == "english":
            assert collections.Counter(u["ids"]) == collections.Counter({x: 2 for x in ids})
    pool = read(BASE / "data" / "binding_rehearsal.json")["quartets"]
    assert len(pool) == 80
    for u in schedule:
        if u["kind"] == "binding":
            docs = rt.binding_docs_for_batch(pool, u["quartets"], u["documents"])
            assert len(docs) == 32 and len(u["quartets"]) == 8
    checks["schedule"] = {"updates": 200, "english": 180, "binding": 20,
                          "english_rows_per_update": 32, "binding_docs_per_update": 32}

    index = {r["id"]: r for r in train}
    tokenizer = Tokenizer.from_file(p["tokenizer"]["path"])
    supervised = set()
    max_len = 0
    for u in schedule:
        if u["kind"] != "english":
            continue
        batch = [index[i] for i in u["ids"]]
        xs, ys = pad_batch(batch, u["pad"])
        assert xs.shape == ys.shape == (32, u["pad"])
        for row, x, y in zip(batch, xs, ys):
            z = [2] + row["prompt_token_ids"] + row["candidate_token_ids"][row["correct_index"]] + [3]
            assert x[:len(z)-1].tolist() == z[:-1]
            start = len(row["prompt_token_ids"])
            assert all(int(v) == -100 for v in y[:start])
            assert y[start:len(z)-1].tolist() == z[start+1:]
            assert all(int(v) == -100 for v in y[len(z)-1:])
            supervised.add(sum(int(v) != -100 for v in y.tolist()))
            max_len = max(max_len, len(z) - 1)
    assert supervised == {5}
    assert max_len <= 256
    checks["masking"] = {"supervised_tokens_per_row": 5, "max_model_input_length": max_len,
                          "padding_ignored": True, "next_token_aligned": True}

    from SF3_ENGINE import english_lr
    post = [english_lr(i, "english") for i in range(91, 181)]
    assert english_lr(90, "english") == 5e-5
    assert english_lr(180, "binding") == 5e-5
    assert post[0] == 5e-5 and post[-1] == 0.0
    assert all(a > b for a, b in zip(post, post[1:]))
    checks["lr"] = {"post100_english_count": 90, "first": post[0], "last": post[-1],
                    "sum": sum(post), "binding": 5e-5}

    sf2_sources = Path(r"C:\DaveLM-CADAVER\sf2_kl_parent_retention_run_v2\sources")
    source_rows = []
    for q in sorted((ROOT / "sources").glob("*.py")):
        ref = sf2_sources / q.name
        assert ref.exists() and sha(q) == sha(ref)
        source_rows.append({"name": q.name, "sha256": sha(q), "matches_sf2": True})
    checks["pinned_sources"] = source_rows

    engine = (ROOT / "SF3_ENGINE.py").read_text(encoding="utf-8")
    ast.parse(engine)
    prohibited = ["TODO", "NotImplementedError", "pass  # placeholder"]
    assert not any(x in engine for x in prohibited)
    assert "SF3_STOP_UPDATE100_REPLAY_MISMATCH" in engine
    assert "reproduce_sf2_update100" in engine
    assert "for label in [\"HELDOUT\", \"ALTERNATE\", \"COPY\", \"COMPETING\"]" in engine
    assert engine.index("if not acquire(s)") < engine.index("for label in [\"HELDOUT\"")
    assert 'read(b / "FINAL' not in engine and "read(b / 'FINAL" not in engine
    assert not re.search(r"Path\([^\n]*(FINAL|sacred)", engine, re.IGNORECASE)
    checks["controller_static"] = {"syntax": "PASS", "placeholders": 0,
                                    "update100_replay_gate": "PRESENT",
                                    "transfer_gate_order": "PASS", "final_or_sacred_path_refs": 0}

    # The actual scope function is exercised on the uninitialized architecture only; no checkpoint is loaded.
    from treatment13_model import Treatment13Model
    model = Treatment13Model()
    for binding in (False, True):
        set_scope(model, binding)
        for n, par in model.named_parameters():
            expected = binding or (n.startswith("base_model.") and not any(
                n.startswith(f"base_model.blocks.{i}.") for i in range(4)))
            assert par.requires_grad == expected
    checks["scope"] = "PASS on uninitialized architecture; no checkpoint loaded"

    out = {"status": "STATIC_PREFLIGHT_PASS", "checks": checks,
           "checkpoint_loaded": False, "optimizer_created": False,
           "training": False, "locked_panels_scored": False,
           "final_or_sacred_access": False}
    (ROOT / "STATIC_PREFLIGHT.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
