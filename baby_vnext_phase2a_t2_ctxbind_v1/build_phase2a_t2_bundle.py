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
V1 = Path(r"C:\DaveLM-CADAVER\baby_vnext_phase2a_ctxbind_v1")
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")

sys.path.insert(0, str(ROOT))
from qa_lib import (ALL_NAMES, CANDIDATES, LEVELS, OBJECTS, PHRASES, GENERATORS,
                    family_id, gen_A, gen_B)

P1_TRAIN_STREAM_SHA = "5f36b283cebe00d88445383166257fbb9831b759657a28c745f3d888f9626a41"
PARENT_SHA = "c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1"
PARENT_DIGEST = "0101cec9e3220c1de1a0d8abc4e241e2fad32127b7358bc5a2944a34c5a5b953"
TOKENIZER_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"

PRIMARY_SEED = 610005
PANEL_SEED = 61000211
TRAIN_SCHED_SEED = 61000212
REHEAR_SCHED_SEED = 61000213
TRAIN_FAMILY_SEED = 61000214
RESERVED_SEED = 610006

MAX_UPDATES = 500
EFFECTIVE_BATCH = 64
LANG_CONTEXT = 256
QA_BATCH = 32
QA_UPDATES = 450
LANG_UPDATES = 50
M_MARGIN = 1.0

NOVEL_NAMES = ["Kai", "Pia", "Ned", "Rex", "Ted", "Ella"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _pick_blocks(n_names, k, count, lo, hi, seed):
    for attempt in range(4000):
        rng = random.Random(seed + attempt * 7919)
        deg = [0] * n_names
        blocks = []
        ok = True
        for b in range(count):
            remaining_blocks = count - b - 1
            chosen = None
            for _t in range(800):
                cand = rng.sample(range(n_names), k)
                nd = list(deg)
                for c in cand:
                    nd[c] += 1
                if any(nd[c] > hi for c in cand):
                    continue
                # feasibility: every name can still reach lo; total deficit fits
                if any(nd[i] + remaining_blocks < lo for i in range(n_names)):
                    continue
                deficit = sum(max(0, lo - nd[i]) for i in range(n_names))
                if deficit > remaining_blocks * k:
                    continue
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
    raise RuntimeError(f"block construction failed k={k} count={count} lo={lo} hi={hi}")


def _groups_for(kind, partition):
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
    else:
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


SALT = "PHASE2A_T2"


def _family_seed(kind, partition, name_group):
    h = hashlib.sha256(f"{SALT}|{kind}|{partition}|{json.dumps(name_group)}".encode()).hexdigest()
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


def _build_family(kind, name_group, partition, tokenizer, forbidden):
    base = _family_seed(kind, partition, name_group)
    for variant in range(64):
        seed = base + variant * 1_000_003
        items = _items_for(kind, seed)
        fid = family_id(kind, list(name_group), items)
        eps = GENERATORS[kind](fid, list(name_group), items, partition, tokenizer)
        texts = {tuple(e.prompt_token_ids + e.candidate_token_ids[e.correct_index]) for e in eps}
        if not (texts & forbidden):
            return eps, fid
    raise RuntimeError(f"unable to build collision-free family {kind} {name_group}")


def _novel_family(kind, name_group, partition, tokenizer, round_idx, forbidden):
    base = _family_seed("NOV" + kind + str(round_idx), partition, name_group)
    for variant in range(64):
        seed = base + variant * 1_000_003
        items = _items_for(kind, seed)
        fid = family_id("NOV" + kind, list(name_group), items)
        eps = (gen_A if kind == "A" else gen_B)(fid, list(name_group), items, partition, tokenizer)
        eps = [e for e in eps if e.style.get("order") == 0]
        assert len(eps) == 4
        for e in eps:
            e.kind = "NOV"
        texts = {tuple(e.prompt_token_ids + e.candidate_token_ids[e.correct_index]) for e in eps}
        if not (texts & forbidden):
            return eps, fid
    raise RuntimeError(f"unable to build collision-free novel family {kind} {name_group}")


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
        "style": ep.style,
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
    assert (base, bind) == (60_536_064, 984_321)
    del model, payload

    # ---- panels --------------------------------------------------------------
    # texts of v1 inspectable panels (v1 TEST remains locked/unopened)
    v1text = set()
    for name in ("qa_dev.jsonl", "qa_audit_dev.jsonl"):
        with open(V1 / "data" / name, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                v1text.add(tuple(r["prompt_token_ids"] + r["candidate_token_ids"][r["correct_index"]]))
    dev_rows, test_rows = [], []
    dev_aud, test_aud = [], []
    nov_dev, nov_test = [], []
    fams = {"dev": set(), "test": set()}
    for partition, lvl, nfam in [(p, l, (10 if p == "dev" else 5)) for p in ("dev", "test") for l in LEVELS]:
        groups = _groups_for(lvl, partition)
        for g in groups[:nfam]:
            eps, fid = _build_family(lvl, g, partition, tokenizer, v1text)
            assert fid not in fams["dev"] and fid not in fams["test"]
            fams[partition].add(fid)
            (dev_rows if partition == "dev" else test_rows).extend(_item_json(e) for e in eps)
    for partition, nfam in (("dev", 30), ("test", 15)):
        groups = _groups_for("AUD", partition)
        for g in groups[:nfam]:
            eps, fid = _build_family("AUD", g, partition, tokenizer, v1text)
            fams[partition].add(fid)
            (dev_aud if partition == "dev" else test_aud).extend(_item_json(e) for e in eps)
    # novel-name slices (2-candidate, exactly 4 episodes/family)
    pairs = [(0, 1), (2, 3), (4, 5), (0, 2), (1, 4), (3, 5), (0, 3), (1, 5), (2, 4),
             (0, 4), (1, 2), (3, 4), (2, 5), (0, 5), (1, 3)]
    for partition, nfam in (("dev", 30), ("test", 15)):
        for i in range(nfam):
            pi = i % len(pairs)
            kind = "A" if i % 2 == 0 else "B"
            ng = (NOVEL_NAMES[pairs[pi][0]], NOVEL_NAMES[pairs[pi][1]])
            eps, fid = _novel_family(kind, ng, partition, tokenizer, i // len(pairs), v1text)
            assert fid not in fams["dev"] and fid not in fams["test"]
            fams[partition].add(fid)
            (nov_dev if partition == "dev" else nov_test).extend(_item_json(e) for e in eps)

    # ---- train corpus (fresh families, 12 standard names only) ---------------
    train = {lvl: [] for lvl in LEVELS}
    for lvl in LEVELS:
        got = 0
        rng = random.Random(TRAIN_FAMILY_SEED + ord(lvl[0]))
        while got < 150:
            names = rng.sample(ALL_NAMES, CANDIDATES[lvl])
            items = _items_for(lvl, rng.randrange(1 << 30))
            fid = family_id(lvl, names, items)
            if fid in fams["dev"] or fid in fams["test"]:
                continue
            fams["dev"].add(fid)
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
    write_jsonl(ROOT / "data" / "qa_novel_dev.jsonl", nov_dev)
    write_jsonl(ROOT / "data" / "qa_novel_test.jsonl", nov_test)

    # exact-text disjointness vs eval panels
    def full_tok(r):
        return tuple(r["prompt_token_ids"] + r["candidate_token_ids"][r["correct_index"]])

    eval_text = set()
    for r in dev_rows + test_rows + dev_aud + test_aud + nov_dev + nov_test:
        eval_text.add(full_tok(r))
    for lvl in LEVELS:
        train[lvl] = [r for r in train[lvl]
                      if full_tok(r) not in eval_text and full_tok(r) not in v1text]
        if len(train[lvl]) < 150:
            raise RuntimeError(f"train filter emptied {lvl}")

    global_train = []
    for lvl in LEVELS:
        global_train.extend(train[lvl])
    train_id = {id(r): i for i, r in enumerate(global_train)}
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

    # ---- schedule ------------------------------------------------------------
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
        slots = []
        for _ in range(QA_BATCH):
            if k == "qa_s3":
                slots.append(rng.choice(pool["s3new"]) if rng.random() < 0.8 else rng.choice(pool["s3old"]))
            else:
                slots.append(rng.choice(pool["s1" if k == "qa_s1" else "s2"]))
        rows.append(slots)
    assert len(rows) == QA_UPDATES
    with open(ROOT / "data" / "qa_schedule.bin", "wb") as f:
        for row in rows:
            f.write(struct.pack(f"<{QA_BATCH}I", *row))
    with open(ROOT / "data" / "update_kinds.json", "w", encoding="utf-8") as f:
        json.dump({"kinds": kinds}, f)

    # ---- rehearsal windows ---------------------------------------------------
    stream_bytes = (P1 / "data" / "LANGUAGE_TRAIN_STREAM.u16").read_bytes()
    if hashlib.sha256(stream_bytes).hexdigest() != P1_TRAIN_STREAM_SHA:
        raise RuntimeError("rehearsal stream hash mismatch")
    n_tok = len(stream_bytes) // 2
    g = torch.Generator().manual_seed(REHEAR_SCHED_SEED)
    with open(ROOT / "data" / "rehearsal_windows.bin", "wb") as f:
        for _ in range(LANG_UPDATES):
            starts = torch.randint(0, n_tok - (LANG_CONTEXT + 1), (EFFECTIVE_BATCH,), generator=g)
            f.write(struct.pack(f"<{EFFECTIVE_BATCH}I", *[int(x) for x in starts.tolist()]))

    # ---- audits --------------------------------------------------------------
    from collections import Counter

    def acc(rows_):
        cc = Counter()
        for r in rows_:
            cc[r["correct_name"]] += 1
        return cc

    for lvl in LEVELS:
        rows_ = [r for r in dev_rows if r["level"] == lvl]
        cc = acc(rows_)
        vals = [v for v in cc.values() if v > 0]
        assert len(rows_) == 80 and max(vals) / min(vals) <= 3.0, (lvl, len(rows_))
        for r in rows_:
            assert len(r["candidate_token_ids"]) == CANDIDATES[lvl]
    for lvl in LEVELS:
        rows_ = [r for r in test_rows if r["level"] == lvl]
        assert len(rows_) == 40
        for r in rows_:
            assert len(r["candidate_token_ids"]) == CANDIDATES[lvl]
    assert len(dev_aud) == 120 and len(test_aud) == 60
    assert len(nov_dev) == 120 and len(nov_test) == 60
    for r in nov_dev + nov_test:
        assert len(r["candidate_token_ids"]) == 2
        assert all(n in NOVEL_NAMES for n in r["candidate_names"])
    for r in global_train:
        assert all(n in ALL_NAMES for n in r["candidate_names"])
    # no exact-text overlap across train/dev/test/audit/novel
    sets = {"train": {full_tok(r) for r in global_train},
            "dev": {full_tok(r) for r in dev_rows},
            "test": {full_tok(r) for r in test_rows},
            "aud": {full_tok(r) for r in dev_aud + test_aud},
            "nov": {full_tok(r) for r in nov_dev + nov_test}}
    keys = list(sets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not (sets[keys[i]] & sets[keys[j]]), (keys[i], keys[j])
    # disjoint from v1 panels (inspectable ones only; v1 TEST remains locked)
    assert not (sets["train"] & v1text) and not (sets["dev"] & v1text)
    assert not (sets["test"] & v1text) and not (sets["nov"] & v1text)

    L_Q = 0
    for r in dev_rows + test_rows + dev_aud + test_aud + nov_dev + nov_test + global_train:
        L_Q = max(L_Q, 1 + len(r["prompt_token_ids"]) + len(r["candidate_token_ids"][r["correct_index"]]))

    stats = {
        "train_items": len(global_train),
        "per_level_train": {l: len(train[l]) for l in LEVELS},
        "qa_dev": len(dev_rows), "qa_test": len(test_rows),
        "qa_audit_dev": len(dev_aud), "qa_audit_test": len(test_aud),
        "novel_dev": len(nov_dev), "novel_test": len(nov_test),
        "pool_sizes": {k: len(v) for k, v in pool.items()},
        "L_Q": L_Q, "M": M_MARGIN,
        "rehearsal_stream_tokens": n_tok,
        "kind_counts": {k: kinds.count(k) for k in set(kinds)},
    }
    (ROOT / "data" / "PANEL_STATS.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    # ---- config / provenance -------------------------------------------------
    def fsha(name):
        return sha256(ROOT / "data" / name)

    cfg_out = {
        "study": "BABY_VNEXT_PHASE2A_T2_CTXBIND_V1",
        "status": "PROSPECTIVE_READY_TO_TRAIN",
        "objective": {
            "type": "length_normalized_hardest_distractor_pairwise_margin_hinge",
            "M": M_MARGIN,
            "candidate_score": "mean per-token log-likelihood of canonical ' Name.' including trailing period, excluding EOS",
            "qa_loss": "mean over items of max(0, M - (s_correct - max_distractor s))",
            "no_prompt_ce": True, "no_continuation_ce": True, "no_eos_supervision": True,
            "no_kl": True, "no_localization_term": True,
        },
        "parent": {
            "checkpoint": str(P1RUN / "checkpoints" / "best.pt"),
            "sha256": PARENT_SHA, "model_state_digest": PARENT_DIGEST,
            "completed_update": 6000, "eval_dev_ce": 1.2040123894810677,
        },
        "architecture": {"bundle": str(ARCH),
                         "config_sha256": "8d7e1cbc604c9bf9f1942d4afc01817b20af17d8805a984f7ae57b1254f957b4",
                         "trainable_base": 60536064, "frozen_binding": 984321},
        "tokenizer": {"path": str(TOKENIZER), "sha256": TOKENIZER_SHA},
        "seeds": {"primary": PRIMARY_SEED, "panel": PANEL_SEED, "train_schedule": TRAIN_SCHED_SEED,
                  "rehearsal": REHEAR_SCHED_SEED, "train_family": TRAIN_FAMILY_SEED,
                  "reserved_replication": RESERVED_SEED},
        "data": {
            "qa_train": {"file": "data/qa_train.jsonl", "sha256": fsha("qa_train.jsonl"), "items": len(global_train)},
            "qa_dev": {"file": "data/qa_dev.jsonl", "sha256": fsha("qa_dev.jsonl")},
            "qa_test": {"file": "data/qa_test.jsonl", "sha256": fsha("qa_test.jsonl")},
            "qa_audit_dev": {"file": "data/qa_audit_dev.jsonl", "sha256": fsha("qa_audit_dev.jsonl")},
            "qa_audit_test": {"file": "data/qa_audit_test.jsonl", "sha256": fsha("qa_audit_test.jsonl")},
            "qa_novel_dev": {"file": "data/qa_novel_dev.jsonl", "sha256": fsha("qa_novel_dev.jsonl")},
            "qa_novel_test": {"file": "data/qa_novel_test.jsonl", "sha256": fsha("qa_novel_test.jsonl")},
            "qa_schedule": {"file": "data/qa_schedule.bin", "sha256": fsha("qa_schedule.bin")},
            "update_kinds": {"file": "data/update_kinds.json", "sha256": fsha("update_kinds.json")},
            "rehearsal_windows": {"file": "data/rehearsal_windows.bin", "sha256": fsha("rehearsal_windows.bin")},
            "panel_stats": {"file": "data/PANEL_STATS.json", "sha256": fsha("PANEL_STATS.json")},
            "L_Q": L_Q, "qa_batch": QA_BATCH, "qa_updates": QA_UPDATES, "lang_updates": LANG_UPDATES,
        },
        "language_rehearsal": {"stream": str(P1 / "data" / "LANGUAGE_TRAIN_STREAM.u16"),
                               "stream_sha256": P1_TRAIN_STREAM_SHA,
                               "context": LANG_CONTEXT, "effective_batch": EFFECTIVE_BATCH},
        "language_eval": {"dev_stream": str(P1 / "data" / "LANGUAGE_DEV_STREAM.u16"),
                          "dev_stream_sha256": "4e9834d43ec80a0eea5e1a5a59c284f8f76f5f84bffcf979e4686661d71b6017",
                          "selection": str(P1 / "data" / "EVAL_WINDOW_STARTS.u32"),
                          "selection_sha256": "744904653ed11605025318cd4e5579ef6a33bd516575834086daa24b61a76681",
                          "prompts": str(P1 / "data" / "GENERATION_PROMPTS.json"),
                          "prompts_sha256": "6bcc77244efc5dad70ef905d275aeed6495b16c76decc4ae2e1684aaaa83cf51",
                          "dev_windows": 1280, "train_windows": 320},
        "optimizer": {"type": "AdamW", "lr": 5e-5, "betas": [0.9, 0.999], "eps": 1e-8,
                      "weight_decay": 0.05, "gradient_clip": 2.0},
        "schedule": {"max_updates": MAX_UPDATES, "cadence": "9 qa : 1 language",
                     "qa_batch": QA_BATCH, "lang_batch": EFFECTIVE_BATCH, "lang_context": LANG_CONTEXT},
        "checkpointing": {"rolling_interval": 50, "eval_points": [0, 100, 250, 500]},
        "scope": "all base_model parameters trainable; all binding/localizer parameters frozen",
        "gates": {
            "u100": {"lang_dev_ce_max": 1.30, "ab_acc_min": 0.56, "ab_margin_min": 0.15,
                     "v1_dc_min": 1.0, "v1_dm_max": 0.10},
            "u250": {"lang_dev_ce_max": 1.30, "abcde_acc_min": 0.50, "abcde_margin_min": 0.20,
                     "aud_acc_min": 0.35, "aud_margin_min": 0.0,
                     "v1_dc_min": 1.0, "v1_dm_max": 0.10},
            "final": {"lang_dev_ce_max": 1.30,
                      "ab_acc_min": 0.80, "ab_margin_min": 0.50,
                      "abcde_acc_min": 0.65, "per_level_min": 0.50,
                      "aud_acc_min": 0.55, "aud_margin_min": 0.10,
                      "novel_acc_min": 0.65, "reversal_min": 0.70,
                      "v1_dm_min": 0.30, "v1_dc_min": 1.0, "v1_dm_max": 0.10,
                      "monotonic_tol": 0.05,
                      "gen_non_imm_eos": 10, "gen_no_triple": 10, "gen_no_repeat_trigram": 6,
                      "test_core_min": 0.60, "test_aud_min": 0.50, "test_novel_min": 0.60},
        },
        "panel_structure": {"ab_count": 160, "abcde_count": 400, "audit_dev": 120,
                            "novel_dev": 120, "test_core": 320, "test_audit": 60, "test_novel": 60,
                            "ab_chance": 0.50, "abcde_chance": 0.40, "aud_chance": 0.25,
                            "novel_chance": 0.50},
        "locks": {"historical_transfer": "LOCKED_UNSCORED", "final_and_sacred": "LOCKED_UNACCESSED",
                  "synthetic_binding_panels": "LOCKED_UNACCESSED",
                  "v1_qa_eval_test": "LOCKED_UNOPENED"},
    }
    (ROOT / "PHASE2A_T2_CONFIG.json").write_text(json.dumps(cfg_out, indent=2) + "\n", encoding="utf-8")
    prov = {
        "study": cfg_out["study"], "parent": cfg_out["parent"],
        "architecture_config_sha256": cfg_out["architecture"]["config_sha256"],
        "tokenizer_sha256": TOKENIZER_SHA,
        "phase1g_reference": {"bundle": str(P1),
                              "config_sha256": "5b338905ed08c1b803233381c6494ec28fdb50ccb758215d01e9694c062e054d",
                              "final_status_sha256": "81c3fb4bc55760de63abe348b04bef5891ed51e7c5a22207c3577e7daabaf0c1"},
        "v1_reference": {"bundle": str(V1), "note": "v1 DEV observed; v1 TEST remains locked/unopened"},
        "data_files": {k: v["sha256"] for k, v in cfg_out["data"].items() if isinstance(v, dict)},
        "novel_names_holdout": NOVEL_NAMES,
        "optimizer_created": False, "optimizer_updates": 0, "training_performed": False,
        "forbidden_material_accessed": False,
    }
    (ROOT / "PROVENANCE.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PHASE2A_T2_BUILD_PASS", **stats,
                      "config_sha256": sha256(ROOT / "PHASE2A_T2_CONFIG.json")}, indent=2))


if __name__ == "__main__":
    main()
