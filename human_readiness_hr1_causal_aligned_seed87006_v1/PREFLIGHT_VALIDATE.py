import hashlib
import importlib.util
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = Path(r"C:\\DaveLM-CADAVER")
V8 = ROOT / "human_readiness_hr1_seed87004_v8"
P1 = ROOT / "language_pilot_1_early_block_protection_seed8380"
TOK_PATH = Path(r"C:\\DaveLM-v0.9\\tokenizer\\v0_7\\davelm_tokenizer.json")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_alignment():
    p = R / "sources" / "CAUSAL_ALIGNMENT.py"
    spec = importlib.util.spec_from_file_location("causal_alignment", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main():
    proto = read_json(R / "HR1_CAUSAL_ALIGNED_PROTOCOL.json")
    out = {"status": "CAUSAL_ALIGNMENT_PREFLIGHT_PASS_NO_TRAINING", "checks": {}}

    # Hash and byte-for-byte inheritance checks.
    expected = {
        "data/ENGLISH_TRAIN.jsonl": "31617e7062ee0246b0a2e8a05d74bb568ee15ecdad43c92b18d0bc6f2a982f43",
        "data/ENGLISH_DEV.jsonl": "2053a8036d7d0f02ba56ab28f739df95a1743760d0f8c980e112d6667b38104e",
        "data/ENGLISH_SCHEDULE.json": "3651905febb0ea46409bbfdc6709476ae068badc0773e2f5c7d962950a6864c0",
        "data/BINDING_SCHEDULE.json": "9f582ec2e5a7d7e7b2de9bebf28d36e46b5f2cf67aa95356755eafb24647797d",
        "data/binding_rehearsal.json": "a47be70021468f6dc3d6bccc2646c993be96a0cd8ea8d0703972440bd9dc3558",
        "data/binding_dev_pilot0.json": "30bfbcbe1b11d3d2d427a631562f43b24dbb97f6e595164516d40553792edc4e",
        "data/binding_dev_pilot1.json": "3b6774b2cadeb7818d59c8a4f4f0b69b2cd85744efce91af562d9bd6f2a54ea1",
        "data/PILOT1_MATERIAL_MANIFEST.json": sha(P1 / "PILOT1_MATERIAL_MANIFEST.json"),
        "data/PILOT1_TRAINING_SPEC.json": sha(P1 / "pilot_run" / "TRAINING_SPEC.json"),
        "data/CORPUS_MANIFEST.json": sha(ROOT / "language_pilot_0_tinystories_seed8380" / "CORPUS_MANIFEST.json"),
    }
    for rel, h in expected.items():
        p = R / rel
        assert p.is_file(), rel
        assert sha(p) == h, (rel, sha(p), h)
        out["checks"][rel] = {"sha256": h, "exists": True}
    inherited = {
        "data/ENGLISH_TRAIN.jsonl": V8 / "ENGLISH_TRAIN.jsonl",
        "data/ENGLISH_DEV.jsonl": V8 / "ENGLISH_DEV.jsonl",
        "data/ENGLISH_SCHEDULE.json": V8 / "ENGLISH_SCHEDULE.json",
        "data/BINDING_SCHEDULE.json": V8 / "BINDING_SCHEDULE.json",
        "data/binding_rehearsal.json": P1 / "binding_rehearsal.json",
    }
    for rel, src in inherited.items():
        assert (R / rel).read_bytes() == src.read_bytes(), rel
    out["checks"]["byte_for_byte_inheritance"] = True

    # Parent and tokenizer identity are checked by bytes only; no checkpoint load.
    assert sha(Path(proto["parent_path"])) == proto["parent_sha256"] == "2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb"
    assert sha(TOK_PATH) == proto["tokenizer_sha256"] == "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
    out["checks"]["parent"] = {"path": proto["parent_path"], "sha256": proto["parent_sha256"], "loaded": False}
    out["checks"]["tokenizer"] = {"path": str(TOK_PATH), "sha256": proto["tokenizer_sha256"]}

    # Materialized records and schedules.
    train = [json.loads(x) for x in (R / "data" / "ENGLISH_TRAIN.jsonl").read_text(encoding="utf-8").splitlines()]
    dev = [json.loads(x) for x in (R / "data" / "ENGLISH_DEV.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train) > 0 and len(dev) > 0
    train_ids, dev_ids = {r["id"] for r in train}, {r["id"] for r in dev}
    assert len(train_ids) == len(train) and len(dev_ids) == len(dev) and train_ids.isdisjoint(dev_ids)
    es = read_json(R / "data" / "ENGLISH_SCHEDULE.json")["batches"]
    assert len(es) == 450 and all(len(b["record_ids"]) == 64 for b in es)
    records = {r["id"]: r for r in train}
    assert all(all(i in records for i in b["record_ids"]) for b in es)
    bs = read_json(R / "data" / "BINDING_SCHEDULE.json")["batches"]
    bind = read_json(R / "data" / "binding_rehearsal.json")["quartets"]
    assert len(bind) == 80 and all(len(q["docs"]) == 4 for q in bind)
    assert len(bs) == 50 and [b["update"] for b in bs] == list(range(10, 501, 10))
    assert all(len(b["quartet_ids"]) == 8 and len(b["documents"]) == 32 for b in bs)
    assert all(isinstance(q, int) and 0 <= q < len(bind) for b in bs for q in b["quartet_ids"])
    for b in bs:
        expected_docs = [d["doc_id"] for q in b["quartet_ids"] for d in bind[q]["docs"]]
        assert b["documents"] == expected_docs
    assert all(len({d["doc_id"] for q in bind for d in q["docs"]}) == 320 for _ in [0])
    out["checks"]["datasets_and_schedules"] = {"train_records": len(train), "dev_records": len(dev), "english_batches": 450, "binding_quartets": 80, "binding_documents": 320, "binding_batches": 50}

    # Tokenizer-only alignment sanity test.
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOK_PATH))
    text = "A cat sat."
    ids = tok.encode(text).ids
    assert ids == [37, 268, 265, 269, 265, 18]
    assert tok.decode(ids) == text
    align = load_alignment()
    x, y = align.sequence_pair(ids, len(ids) + 2)
    align.validate_pair(ids, x, y)
    transitions = list(zip(x[: len(ids) + 1], y[: len(ids) + 1]))
    assert transitions == [(2, 37), (37, 268), (268, 265), (265, 269), (269, 265), (265, 18), (18, 3)]
    assert y[len(ids) + 1] == -100 and x[len(ids) + 1] == 0
    assert all(a != b for a, b in transitions)
    old_y = [-100] * len(x)
    z = [2] + ids + [3]
    old_y[1 : len(z)] = z[1:]
    try:
        align.validate_pair(ids, x, old_y)
    except AssertionError:
        negative_old_shift = True
    else:
        negative_old_shift = False
    assert negative_old_shift
    out["checks"]["alignment_sanity"] = {"text": text, "token_ids": ids, "x": x, "y": y, "transitions": transitions, "eos_only_final": True, "padding_ignored": True, "old_shift_rejected": True}

    # Source-level root-cause and corrected-source checks. No source is imported or executed.
    old_hr1 = (V8 / "HR1_TRAIN_REAL.py").read_text(encoding="utf-8")
    old_hr2 = (ROOT / "human_readiness_hr2_seed87005_v7" / "HR2_TRAIN.py").read_text(encoding="utf-8")
    corrected = (R / "sources" / "CAUSAL_ALIGNED_TRAIN_REAL.py").read_text(encoding="utf-8")
    old_pattern = "y[i,1:len(z)]"
    new_pattern = "y[i,:len(z)-1]"
    assert old_hr1.count(old_pattern) >= 2 and old_hr2.count(old_pattern) >= 2
    assert corrected.count(new_pattern) >= 2 and old_pattern not in corrected
    assert corrected.count("F.cross_entropy(m.base_model(x).reshape(-1,1024),y.reshape(-1),ignore_index=-100)") >= 2
    out["checks"]["source_alignment"] = {"hr1_shifted_occurrences": old_hr1.count(old_pattern), "hr2_shifted_occurrences": old_hr2.count(old_pattern), "corrected_aligned_occurrences": corrected.count(new_pattern), "corrected_shifted_occurrences": corrected.count(old_pattern), "train_dev_objective_expression_count": corrected.count("F.cross_entropy(m.base_model(x).reshape(-1,1024),y.reshape(-1),ignore_index=-100)")}

    # Static source/provenance references and sealed-material lock.
    trainer_text = (R / "sources" / "CAUSAL_ALIGNED_TRAIN_REAL.py").read_text(encoding="utf-8")
    assert "HR1_CAUSAL_ALIGNED_PROTOCOL.json" in trainer_text
    assert "R/'data'/'ENGLISH_TRAIN.jsonl'" in trainer_text
    assert "R/'data'/'ENGLISH_DEV.jsonl'" in trainer_text
    assert "R/'data'/'ENGLISH_SCHEDULE.json'" in trainer_text
    assert "R/'data'/'BINDING_SCHEDULE.json'" in trainer_text
    assert "R/proto['data']['binding_rehearsal']" in trainer_text
    assert "R/'HR1_PROTOCOL.json'" not in trainer_text
    assert "R/'ENGLISH_TRAIN.jsonl'" not in trainer_text
    assert "R/'ENGLISH_DEV.jsonl'" not in trainer_text
    assert "R/'ENGLISH_SCHEDULE.json'" not in trainer_text
    assert "R/'BINDING_SCHEDULE.json'" not in trainer_text
    source_text = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in R.rglob("*.py") if p.name != "PREFLIGHT_VALIDATE.py")
    assert "FINAL_ITEMS" not in source_text
    out["checks"]["source_wiring"] = {"bundle_protocol_and_data_paths": True, "stale_hr1_paths_rejected": True}
    out["checks"]["sealed_material"] = {"final_battery_accessed": False, "sacred_material_accessed": False}

    # Runtime is recorded from this known-good interpreter; no model or optimizer is instantiated.
    import torch
    out["runtime"] = {"python": sys.version, "torch": torch.__version__, "tokenizers": __import__("tokenizers").__version__, "cuda_available": bool(torch.cuda.is_available()), "checkpoint_loaded": False, "optimizer_created": False, "optimizer_updates": 0}
    (R / "checks" / "ALIGNMENT_SANITY.json").write_text(json.dumps(out["checks"]["alignment_sanity"], indent=2), encoding="utf-8")
    (R / "checks" / "OBJECTIVE_EQUIVALENCE.json").write_text(json.dumps({"status": "PASS", "training_and_dev_use_same_aligned_expression": True, "source_check": out["checks"]["source_alignment"], "reference_module": "sources/CAUSAL_ALIGNMENT.py"}, indent=2), encoding="utf-8")
    (R / "PREFLIGHT.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
