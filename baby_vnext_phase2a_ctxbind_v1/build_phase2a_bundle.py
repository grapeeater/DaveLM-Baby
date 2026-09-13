from __future__ import annotations

import hashlib
import json
import random
import struct
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
ARCH = Path(r"C:\DaveLM-CADAVER\baby_vnext_60m_design_v1")
P1 = Path(r"C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1")
P1RUN = Path(r"C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001")
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")

sys.path.insert(0, str(ROOT))
import qa_lib
from qa_lib import ALL_NAMES, CANDIDATES, LEVELS, OBJECTS, PHRASES, GENERATORS, family_id

P1_TRAIN_STREAM_SHA = "5f36b283cebe00d88445383166257fbb9831b759657a28c745f3d888f9626a41"
PARENT_SHA = "c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1"
PARENT_DIGEST = "0101cec9e3220c1de1a0d8abc4e241e2fad32127b7358bc5a2944a34c5a5b953"
TOKENIZER_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"

PANEL_SEED = 61000201
TRAIN_SCHED_SEED = 61000202
REHEAR_SCHED_SEED = 61000203
MAX_UPDATES = 500
EFFECTIVE_BATCH = 64
LANG_CONTEXT = 256
QA_BATCH = 32
QA_UPDATES = 450
LANG_UPDATES = 50


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _pick_blocks(n_names, k, count, lo, hi, seed):
    for attempt in range(800):
        rng = random.Random(seed + attempt * 7919)
        deg = [0] * n_names
        blocks = []
        ok = True
        for _ in range(count):
            chosen = None
            for _t in range(600):
                cand = rng.sample(range(n_names), k)
                if all(deg[c] < hi for c in cand):
                    chosen = cand
                    break
            if chosen is None:
                ok = False
                break
            for c in chosen:
                deg[c] += 1
            blocks.append(chosen)
        if ok and all(lo <= d <= hi for d in deg):
            return blocks, attempt
    raise RuntimeError(f"unable to build blocks k={k} count={count} lo={lo} hi={hi}")


def _groups_for(kind, partition):
    """Return a list of name-tuples, one per family."""
    n = len(ALL_NAMES)
    if partition == "dev":
        if kind in ("A", "B", "D", "F"):
            edges = [(i, (i + 1) % n) for i in range(n)]
            edges = [e for i, e in enumerate(edges) if i not in (0, 6)]
            return [tuple(ALL_NAMES[i] for i in e) for e in edges]
        if kind in ("C", "E", "G"):
            blocks, _ = _pick_blocks(n, 4, 10, 3, 4, PANEL_SEED + 100)
        elif kind == "H":
            blocks, _ = _pick_blocks(n, 3, 10, 2, 3, PANEL_SEED + 200)
        elif kind == "AUD":
            blocks, _ = _pick_blocks(n, 4, 30, 8, 12, PANEL_SEED + 300)
        else:
            raise RuntimeError(kind)
    else:  # test
        if kind in ("A", "B", "D", "F"):
            edges = [(i, (i + 1) % n) for i in (0, 3, 5, 7, 10)]
            return [tuple(ALL_NAMES[i] for i in e) for e in edges]
        if kind in ("C", "E", "G"):
            blocks, _ = _pick_blocks(n, 4, 5, 1, 2, PANEL_SEED + 400)
        elif kind == "H":
            blocks, _ = _pick_blocks(n, 3, 5, 1, 2, PANEL_SEED + 500)
        elif kind == "AUD":
            blocks, _ = _pick_blocks(n, 4, 15, 4, 6, PANEL_SEED + 600)
        else:
            raise RuntimeError(kind)
    return [tuple(ALL_NAMES[i] for i in b) for b in blocks]


def _family_seed(kind, partition, name_group):
    h = hashlib.sha256(f"{kind}|{partition}|{json.dumps(name_group)}".encode("utf-8")).hexdigest()
    return int(h[:14], 16)


def _items_for(kind, seed):
    rng = random.Random(seed)
    if kind == "D":
        return list(rng.sample(PHRASES, 1))
    if kind in ("H", "C", "E"):
        return list(rng.sample(PHRASES, 3))
    if kind in ("G", "AUD"):
        return list(rng.sample(PHRASES, 4))
    return list(rng.sample(PHRASES, 2))


def _build_family(kind, name_group, partition, tokenizer):
    seed = _family_seed(kind, partition, name_group)
    items = _items_for(kind, seed)
    fid = family_id(kind, list(name_group), items)
    eps = GENERATORS[kind](fid, list(name_group), items, partition, tokenizer)
    for ep in eps:
        assert ep.prompt_len + ep.answer_len + 2 <= 200
    return eps, fid


def _item_json(ep):
    return {
        "level": ep.kind, "kind": ep.kind, "family_id": ep.family_id,
        "partition": ep.partition, "correct_index": ep.correct_index,
        "is_audit": bool(getattr(ep, "is_audit", False)),
        "prompt_token_ids": ep.prompt_token_ids,
        "candidate_token_ids": ep.candidate_token_ids,
        "candidate_names": ep.candidate_names,
        "correct_name": ep.correct_name,
        "prompt_text": ep.prompt_text,
    }


def main():
    import torch
    sys.path.insert(0, str(ARCH))
    from baby_vnext.config import BabyVNextConfig
    from baby_vnext.binding import BabyVNextWithBinding
    from baby_vnext.checkpoint import _state_digest

    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    if tokenizer.get_vocab_size() != 1024:
        raise RuntimeError("vocab")

    # ---- parent verification ------------------------------------------------
    if sha256(P1RUN / "checkpoints" / "best.pt") != PARENT_SHA:
        raise RuntimeError("parent sha mismatch")
    cfg = BabyVNextConfig.load(ARCH / "BABY_VNEXT_CONFIG.json")
    model = BabyVNextWithBinding(cfg)
    payload = torch.load(P1RUN / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    assert payload["schema"] == "baby_vnext_phase1_model_v1"
    assert payload["completed_update"] == 6000
    assert payload["provenance_hashes"]["train_stream_sha256"] == P1_TRAIN_STREAM_SHA
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if _state_digest(model) != PARENT_DIGEST:
        raise RuntimeError("parent digest mismatch")
    base = sum(v.numel() for k, v in model.named_parameters() if k.startswith("base_model."))
    bind = sum(v.numel() for k, v in model.named_parameters() if not k.startswith("base_model."))
    assert (base, bind) == (60_536_064, 984_321), (base, bind)
    del model, payload

    # ---- dev/test/audit panels ----------------------------------------------
    dev_rows, test_rows = [], []
    dev_aud, test_aud = [], []
    fams = {"dev": set(), "test": set()}
    plan = [("dev", lvl, 10) for lvl in LEVELS] + [("test", lvl, 5) for lvl in LEVELS]
    for partition, lvl, nfam in plan:
        groups = _groups_for(lvl, partition)
        if len(groups) < nfam:
            raise RuntimeError(f"not enough groups {partition} {lvl}: {len(groups)}")
        for g in groups[:nfam]:
            eps, fid = _build_family(lvl, g, partition, tokenizer)
            assert fid not in fams[partition] and (fid not in fams["dev"] or partition == "dev")
            fams[partition].add(fid)
            rows = dev_rows if partition == "dev" else test_rows
            rows.extend(_item_json(e) for e in eps)
    for partition, nfam in (("dev", 30), ("test", 15)):
        groups = _groups_for("AUD", partition)
        for g in groups[:nfam]:
            eps, fid = _build_family("AUD", g, partition, tokenizer)
            fams[partition].add(fid)
            target = dev_aud if partition == "dev" else test_aud
            target.extend(_item_json(e) for e in eps)

    # ---- training families ----------------------------------------------------
    train = {lvl: [] for lvl in LEVELS}
    train_target = {lvl: 150 for lvl in LEVELS}
    for lvl in LEVELS:
        got = 0
        rng = random.Random(PANEL_SEED + 10_000 + ord(lvl[0]))
        while got < train_target[lvl]:
            names = rng.sample(ALL_NAMES, CANDIDATES[lvl])
            items = _items_for(lvl, rng.randrange(1 << 30))
            fid = family_id(lvl, names, items)
            if fid in fams["dev"] or fid in fams["test"]:
                continue
            fams["dev"].add(fid)  # mark used (train occupies its own set)
            eps = GENERATORS[lvl](fid, names, items, "train", tokenizer)
            train[lvl].extend(_item_json(e) for e in eps)
            got += 1

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(ROOT / "data" / "qa_dev.jsonl", dev_rows)
    write_jsonl(ROOT / "data" / "qa_test.jsonl", test_rows)
    write_jsonl(ROOT / "data" / "qa_audit_dev.jsonl", dev_aud)
    write_jsonl(ROOT / "data" / "qa_audit_test.jsonl", test_aud)

    # exact text-set disjointness: drop any train item whose full token text is
    # identical to a dev/test/audit item text
    def full_tok(r):
        return tuple(r["prompt_token_ids"] + r["candidate_token_ids"][r["correct_index"]])

    eval_text = set()
    for r in dev_rows + test_rows + dev_aud + test_aud:
        eval_text.add(full_tok(r))
    for lvl in LEVELS:
        keep = [r for r in train[lvl] if full_tok(r) not in eval_text]
        train[lvl] = keep
        if len(keep) < train_target[lvl]:
            raise RuntimeError(f"train text filter emptied {lvl}: {len(keep)}")

    # stage pools over the train corpus (single global list, A..H order)
    order = LEVELS
    global_train = []
    for lvl in order:
        global_train.extend(train[lvl])
    train_id = {}
    for i, r in enumerate(global_train):
        train_id[id(r)] = i
    pool = {s: [] for s in ("s1", "s2", "s3new", "s3old")}
    for lvl in LEVELS:
        ids = [train_id[id(r)] for r in train[lvl]]
        if lvl in ("A", "B"):
            pool["s1"].extend(ids)
        if lvl in ("A", "B", "C", "D", "E"):
            pool["s2"].extend(ids)
            pool["s3old"].extend(ids)
        if lvl in ("F", "G", "H"):
            pool["s3new"].extend(ids)
    write_jsonl(ROOT / "data" / "qa_train.jsonl", global_train)

    # ---- 500-update schedule ---------------------------------------------------
    kinds = []
    for u in range(1, MAX_UPDATES + 1):
        if u % 10 == 0:
            kinds.append("lang")
        elif u <= 100:
            kinds.append("qa_s1")
        elif u <= 250:
            kinds.append("qa_s2")
        else:
            kinds.append("qa_s3")
    rng = random.Random(TRAIN_SCHED_SEED)
    rows = []
    for u in range(1, MAX_UPDATES + 1):
        k = kinds[u - 1]
        if k == "lang":
            continue
        if k == "qa_s1":
            src = "s1"
        elif k == "qa_s2":
            src = "s2"
        else:
            src = None
        slots = []
        for _ in range(QA_BATCH):
            if src is None:
                slots.append(rng.choice(pool["s3new"]) if rng.random() < 0.8 else rng.choice(pool["s3old"]))
            else:
                slots.append(rng.choice(pool[src]))
        rows.append(slots)
    assert len(rows) == QA_UPDATES
    with open(ROOT / "data" / "qa_schedule.bin", "wb") as f:
        for row in rows:
            f.write(struct.pack(f"<{QA_BATCH}I", *row))
    with open(ROOT / "data" / "update_kinds.json", "w", encoding="utf-8") as f:
        json.dump({"kinds": kinds}, f)

    # ---- rehearsal windows over the Phase1G train stream -----------------------
    stream_bytes = (P1 / "data" / "LANGUAGE_TRAIN_STREAM.u16").read_bytes()
    if hashlib.sha256(stream_bytes).hexdigest() != P1_TRAIN_STREAM_SHA:
        raise RuntimeError("rehearsal stream hash mismatch")
    n_tok = len(stream_bytes) // 2
    g = torch.Generator().manual_seed(REHEAR_SCHED_SEED)
    with open(ROOT / "data" / "rehearsal_windows.bin", "wb") as f:
        for _ in range(LANG_UPDATES):
            starts = torch.randint(0, n_tok - (LANG_CONTEXT + 1), (EFFECTIVE_BATCH,), generator=g)
            f.write(struct.pack(f"<{EFFECTIVE_BATCH}I", *[int(x) for x in starts.tolist()]))

    # ---- audits ----------------------------------------------------------------
    from collections import Counter

    def acc(rows):
        cc, cand = Counter(), Counter()
        for r in rows:
            cc[r["correct_name"]] += 1
            for n in r["candidate_names"]:
                cand[n] += 1
        return cc, cand

    dev_stats = {}
    for lvl in LEVELS:
        rows = [r for r in dev_rows if r["level"] == lvl]
        cc, _ = acc(rows)
        vals = [v for v in cc.values() if v > 0]
        assert len(rows) == 80, (lvl, len(rows))
        assert max(vals) / min(vals) <= 3.0, (lvl, cc)
        for r in rows:
            assert len(r["candidate_token_ids"]) == CANDIDATES[lvl]
        dev_stats[lvl] = {"n": len(rows), "ratio": round(max(vals) / min(vals), 3), "correct": dict(cc)}
    for lvl in LEVELS:
        rows = [r for r in test_rows if r["level"] == lvl]
        assert len(rows) == 40, (lvl, len(rows))
        for r in rows:
            assert len(r["candidate_token_ids"]) == CANDIDATES[lvl]
    assert len(dev_aud) == 120 and len(test_aud) == 60
    ccA, _ = acc(dev_aud)
    valsA = [v for v in ccA.values() if v > 0]
    assert max(valsA) / min(valsA) <= 2.0
    for r in dev_aud + test_aud:
        assert len(r["candidate_token_ids"]) == 4 and r["is_audit"] is True

    # contamination: no train text equals a locked DEV story or a dev/test text
    dev_src = Path(r"C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\language_dev.jsonl")
    dev_docs = set()
    for line in open(dev_src, encoding="utf-8"):
        dev_docs.add(tuple(json.loads(line)["token_ids"]))

    def full_tok(r):
        return tuple(r["prompt_token_ids"] + r["candidate_token_ids"][r["correct_index"]])

    tr_set = {full_tok(r) for r in global_train}
    assert not (tr_set & dev_docs), "train vs locked DEV overlap"
    dv_set = {full_tok(r) for r in dev_rows}
    te_set = {full_tok(r) for r in test_rows}
    assert not (dv_set & tr_set), "dev vs train text overlap"
    assert not (te_set & tr_set), "test vs train text overlap"
    assert not (dv_set & te_set), "dev vs test text overlap"
    for r in dev_rows + test_rows + dev_aud + test_aud:
        for c in r["candidate_token_ids"]:
            assert c and all(0 <= x < 1024 for x in c)

    L_Q = 0
    for r in dev_rows + test_rows + dev_aud + test_aud:
        L_Q = max(L_Q, 1 + len(r["prompt_token_ids"]) + len(r["candidate_token_ids"][r["correct_index"]]))
    for r in global_train:
        L_Q = max(L_Q, 1 + len(r["prompt_token_ids"]) + len(r["candidate_token_ids"][r["correct_index"]]))

    stats = {
        "train_items": len(global_train),
        "per_level_train": {l: len(train[l]) for l in LEVELS},
        "dev_stats": dev_stats,
        "qa_dev": len(dev_rows), "qa_test": len(test_rows),
        "qa_audit_dev": len(dev_aud), "qa_audit_test": len(test_aud),
        "audit_dev_correct": dict(ccA),
        "pool_sizes": {k: len(v) for k, v in pool.items()},
        "L_Q": L_Q,
        "rehearsal_stream_tokens": n_tok,
        "kind_counts": {k: kinds.count(k) for k in set(kinds)},
    }
    (ROOT / "data" / "PANEL_STATS.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    # ---- config / provenance ---------------------------------------------------
    def fsha(name):
        return sha256(ROOT / "data" / name)

    cfg = {
        "study": "BABY_VNEXT_PHASE2A_CTXBIND_V1",
        "status": "PROSPECTIVE_READY_TO_TRAIN",
        "parent": {
            "checkpoint": str(P1RUN / "checkpoints" / "best.pt"),
            "sha256": PARENT_SHA,
            "model_state_digest": PARENT_DIGEST,
            "completed_update": 6000,
            "eval_dev_ce": 1.2040123894810677,
        },
        "architecture": {
            "bundle": str(ARCH),
            "config_sha256": "8d7e1cbc604c9bf9f1942d4afc01817b20af17d8805a984f7ae57b1254f957b4",
            "trainable_base": 60536064,
            "frozen_binding": 984321,
        },
        "tokenizer": {"path": str(TOKENIZER), "sha256": TOKENIZER_SHA},
        "seeds": {"primary": 610002, "panel": PANEL_SEED, "train_schedule": TRAIN_SCHED_SEED,
                  "rehearsal": REHEAR_SCHED_SEED},
        "data": {
            "qa_train": {"file": "data/qa_train.jsonl", "sha256": fsha("qa_train.jsonl"), "items": len(global_train)},
            "qa_dev": {"file": "data/qa_dev.jsonl", "sha256": fsha("qa_dev.jsonl")},
            "qa_test": {"file": "data/qa_test.jsonl", "sha256": fsha("qa_test.jsonl")},
            "qa_audit_dev": {"file": "data/qa_audit_dev.jsonl", "sha256": fsha("qa_audit_dev.jsonl")},
            "qa_audit_test": {"file": "data/qa_audit_test.jsonl", "sha256": fsha("qa_audit_test.jsonl")},
            "qa_schedule": {"file": "data/qa_schedule.bin", "sha256": fsha("qa_schedule.bin")},
            "update_kinds": {"file": "data/update_kinds.json", "sha256": fsha("update_kinds.json")},
            "rehearsal_windows": {"file": "data/rehearsal_windows.bin", "sha256": fsha("rehearsal_windows.bin")},
            "panel_stats": {"file": "data/PANEL_STATS.json", "sha256": fsha("PANEL_STATS.json")},
            "L_Q": L_Q,
            "qa_batch": QA_BATCH,
            "qa_updates": QA_UPDATES,
            "lang_updates": LANG_UPDATES,
        },
        "language_rehearsal": {
            "stream": str(P1 / "data" / "LANGUAGE_TRAIN_STREAM.u16"),
            "stream_sha256": P1_TRAIN_STREAM_SHA,
            "context": LANG_CONTEXT,
            "effective_batch": EFFECTIVE_BATCH,
        },
        "language_eval": {
            "dev_stream": str(P1 / "data" / "LANGUAGE_DEV_STREAM.u16"),
            "dev_stream_sha256": "4e9834d43ec80a0eea5e1a5a59c284f8f76f5f84bffcf979e4686661d71b6017",
            "selection": str(P1 / "data" / "EVAL_WINDOW_STARTS.u32"),
            "selection_sha256": "744904653ed11605025318cd4e5579ef6a33bd516575834086daa24b61a76681",
            "prompts": str(P1 / "data" / "GENERATION_PROMPTS.json"),
            "prompts_sha256": "6bcc77244efc5dad70ef905d275aeed6495b16c76decc4ae2e1684aaaa83cf51",
            "dev_windows": 1280,
            "train_windows": 320,
        },
        "optimizer": {"type": "AdamW", "lr": 5e-5, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.05, "gradient_clip": 2.0},
        "schedule": {"max_updates": MAX_UPDATES, "cadence": "9 qa : 1 language",
                     "qa_batch": QA_BATCH, "lang_batch": EFFECTIVE_BATCH, "lang_context": LANG_CONTEXT},
        "checkpointing": {"rolling_interval": 50, "eval_points": [0, 100, 250, 500]},
        "scope": "all base_model parameters trainable; all binding/localizer parameters frozen",
        "gates": {
            "u100": {"lang_dev_ce_max": 1.30, "ab_bar": 0.55, "rel": 0.05},
            "u250": {"lang_dev_ce_max": 1.30, "abcde_abs": 0.46, "abcde_rel": 0.06,
                     "aud_abs": 0.35, "aud_rel": 0.10},
            "final": {"lang_dev_ce_max": 1.30, "dev_core_min": 0.70, "core_ab_min": 0.85,
                      "per_level_min": 0.50, "test_min": 0.60, "audit_min": 0.60,
                      "monotonic_tol": 0.02,
                      "gen_non_imm_eos": 10, "gen_no_triple": 10, "gen_no_repeat_trigram": 6},
        },
        "panel_structure": {
            "ab_count": 160, "abcde_count": 400, "audit_dev": 120, "audit_test": 60,
            "abcde_chance": 0.40, "ab_chance": 0.50, "aud_chance": 0.25,
        },
        "locks": {
            "historical_transfer": "LOCKED_UNSCORED",
            "final_and_sacred": "LOCKED_UNACCESSED",
            "synthetic_binding_panels": "LOCKED_UNACCESSED",
        },
    }
    (ROOT / "PHASE2A_CTXBIND_CONFIG.json").write_text(
        json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    provenance = {
        "study": cfg["study"],
        "parent": cfg["parent"],
        "architecture_config_sha256": cfg["architecture"]["config_sha256"],
        "tokenizer_sha256": TOKENIZER_SHA,
        "phase1g_reference": {
            "bundle": str(P1),
            "config_sha256": "5b338905ed08c1b803233381c6494ec28fdb50ccb758215d01e9694c062e054d",
            "final_status_sha256": "81c3fb4bc55760de63abe348b04bef5891ed51e7c5a22207c3577e7daabaf0c1",
        },
        "data_files": {k: v["sha256"] for k, v in cfg["data"].items() if isinstance(v, dict)},
        "corpus_audit": "see data/PANEL_STATS.json",
        "optimizer_created": False,
        "optimizer_updates": 0,
        "training_performed": False,
        "forbidden_material_accessed": False,
    }
    (ROOT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PHASE2A_BUILD_PASS", **stats,
                      "config_sha256": sha256(ROOT / "PHASE2A_CTXBIND_CONFIG.json"),
                      "provenance_sha256": sha256(ROOT / "PROVENANCE.json")}, indent=2))


if __name__ == "__main__":
    main()
