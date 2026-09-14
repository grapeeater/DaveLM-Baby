"""Option B: prospective replacement-tokenizer design. Never loads TEST/FINAL/sacred."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel, Sequence, WhitespaceSplit
from tokenizers.trainers import BpeTrainer

ROOT = Path(r"C:\DaveLM-CADAVER")
STUDY = ROOT / "tokenizer_replacement_study_v1"
TOK_V07 = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
TOK_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
LANG_SRC = ROOT / "baby_vnext_phase1g_language_v1" / "data" / "LANGUAGE_TRAIN_SOURCE.jsonl"
LANG_SHA = "807a89417924a9b1cfa1e0a2a952d97400684db32a138fe11943d93e130312c8"
QA_TRAIN = ROOT / "phase2a_t3_rebuilt_study_v1" / "data" / "qa_train.jsonl"
QA_DEV = ROOT / "phase2a_t3_rebuilt_study_v1" / "data" / "qa_dev.jsonl"
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<doc>"]
TITLE = re.compile(r"[A-Z][a-z]+")
DESCRIPTIVE = ["Sal", "Salt", "Skye", "Sky", "Wes", "Walt", "Omar", "Opal"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_jsonl(path: Path) -> list:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def assert_integrity() -> None:
    if sha(TOK_V07) != TOK_SHA:
        raise RuntimeError("v0_7 hash changed")
    if sha(LANG_SRC) != LANG_SHA:
        raise RuntimeError("language TRAIN source hash mismatch")
    if not QA_TRAIN.is_file():
        raise RuntimeError("qa_train missing")


def language_texts() -> list[str]:
    return [json.loads(line)["text"] for line in LANG_SRC.read_text(encoding="utf-8").splitlines() if line]


def train_prompts() -> list[str]:
    return [row["prompt"] for row in load_jsonl(QA_TRAIN)]


def train_names() -> list[str]:
    names = sorted({row["correct_name"] for row in load_jsonl(QA_TRAIN)})
    return names


def titlecase_docs(texts: list[str]) -> list[str]:
    extra = []
    for text in texts:
        for m in TITLE.findall(text):
            extra.extend([m] * 4)
    return extra


def make_bytelevel(vocab_size: int, add_prefix_space: bool) -> Tokenizer:
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=add_prefix_space, use_regex=True, trim_offsets=True)
    tok.decoder = ByteLevelDecoder(add_prefix_space=True, trim_offsets=True, use_regex=True)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        min_frequency=2,
        show_progress=False,
        initial_alphabet=ByteLevel.alphabet(),
    )
    return tok, trainer


def make_ws(vocab_size: int) -> tuple[Tokenizer, BpeTrainer]:
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Sequence(
        [
            WhitespaceSplit(),
            ByteLevel(add_prefix_space=False, use_regex=True, trim_offsets=True),
        ]
    )
    tok.decoder = ByteLevelDecoder(add_prefix_space=True, trim_offsets=True, use_regex=True)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,
        min_frequency=2,
        show_progress=False,
        initial_alphabet=ByteLevel.alphabet(),
    )
    return tok, trainer


def encode_name(tok: Tokenizer, name: str) -> dict:
    span = " " + name
    ans = " " + name + "."
    span_ids = tok.encode(span).ids
    ans_ids = tok.encode(ans).ids
    pieces = [tok.id_to_token(i) for i in span_ids]
    unk_id = tok.token_to_id("<unk>")
    period_ids = tok.encode(".").ids
    return {
        "name": name,
        "span": span,
        "span_ids": span_ids,
        "span_pieces": pieces,
        "n_tokens": len(span_ids),
        "single_token": len(span_ids) == 1,
        "first_token": span_ids[0] if span_ids else None,
        "answer_ids": ans_ids,
        "answer_n_tokens": len(ans_ids),
        "period_is_own_final_token": bool(ans_ids) and tok.id_to_token(ans_ids[-1]) in {".", tok.id_to_token(period_ids[-1]) if period_ids else "."},
        "unk_count": sum(1 for i in span_ids if i == unk_id),
    }


def inventory_geometry(tok: Tokenizer, names: list[str]) -> dict:
    rows = [encode_name(tok, n) for n in names]
    by_first = defaultdict(list)
    for r in rows:
        by_first[r["first_token"]].append(r["name"])
    groups = {str(k): v for k, v in by_first.items() if len(v) > 1}
    shared_names = [n for n, names_ in ((r["name"], by_first[r["first_token"]]) for r in rows) if len(names_) > 1]
    unique_names = [n for n in names if n not in shared_names]
    n = len(names)
    token_counts = [r["n_tokens"] for r in rows]
    token_counts.sort()
    median = token_counts[n // 2] if n % 2 else 0.5 * (token_counts[n // 2 - 1] + token_counts[n // 2])

    def disambig(name: str, ids: list[int]) -> int:
        others = [encode_name(tok, o)["span_ids"] for o in names if o != name]
        for i, tok_id in enumerate(ids):
            if any(i >= len(o) or o[i] != tok_id for o in others):
                return i + 1
        return len(ids)

    dis = [disambig(r["name"], r["span_ids"]) for r in rows]
    longest_pair = 0
    for i, a in enumerate(rows):
        for b in rows[i + 1 :]:
            k = 0
            for x, y in zip(a["span_ids"], b["span_ids"]):
                if x != y:
                    break
                k += 1
            longest_pair = max(longest_pair, k)
    shared_rate = len(shared_names) / n if n else 0.0
    single_rate = sum(r["single_token"] for r in rows) / n if n else 0.0
    mean_tok = sum(token_counts) / n if n else 0.0
    mean_dis = sum(dis) / n if n else 0.0
    return {
        "n_names": n,
        "mean_tokens_per_name": mean_tok,
        "median_tokens_per_name": median,
        "single_token_rate": single_rate,
        "shared_first_token_rate": shared_rate,
        "unique_first_token_rate": 1.0 - shared_rate,
        "n_shared_first_token_names": len(shared_names),
        "n_unique_first_token_names": len(unique_names),
        "shared_first_token_groups": groups,
        "longest_shared_token_prefix_any_pair": longest_pair,
        "mean_first_disambiguating_position": mean_dis,
        "unk_total": sum(r["unk_count"] for r in rows),
        "period_own_token_rate": sum(r["period_is_own_final_token"] for r in rows) / n if n else 0.0,
        "by_name": {r["name"]: r for r in rows},
    }


def score(geom: dict, lang_bpt: float, v07_bpt: float) -> float:
    shared = geom["shared_first_token_rate"]
    single = geom["single_token_rate"]
    mean_tok = max(geom["mean_tokens_per_name"], 1e-9)
    mean_dis = max(geom["mean_first_disambiguating_position"], 1e-9)
    penalty = 0.5 * max(0.0, (lang_bpt / v07_bpt) - 1.0) if v07_bpt else 0.0
    return 3.0 * (1.0 - shared) + 1.5 * single + (1.0 / mean_tok) + (1.0 / mean_dis) - penalty


def lang_compression(tok: Tokenizer, texts: list[str]) -> dict:
    tokens = 0
    b = 0
    unk_id = tok.token_to_id("<unk>")
    unk = 0
    for t in texts:
        ids = tok.encode(t).ids
        tokens += len(ids)
        b += len(t.encode("utf-8"))
        unk += sum(1 for i in ids if i == unk_id)
    return {
        "documents": len(texts),
        "utf8_bytes": b,
        "tokens": tokens,
        "bytes_per_token": b / tokens if tokens else None,
        "unk_tokens": unk,
    }


def build_candidates(texts: list[str], train_pr: list[str]) -> dict:
    out_dir = STUDY / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_corpus = texts + train_pr
    specs = [
        {"id": "bpe_bl_1536", "vocab": 1536, "kind": "bl", "prefix": False, "upsample": False},
        {"id": "bpe_bl_2048", "vocab": 2048, "kind": "bl", "prefix": False, "upsample": False},
        {"id": "bpe_bl_4096", "vocab": 4096, "kind": "bl", "prefix": False, "upsample": False},
        {"id": "bpe_bl_2048_prefix_space", "vocab": 2048, "kind": "bl", "prefix": True, "upsample": False},
        {"id": "bpe_bl_2048_titlecase_x4", "vocab": 2048, "kind": "bl", "prefix": False, "upsample": True},
        {"id": "bpe_ws_2048", "vocab": 2048, "kind": "ws", "prefix": False, "upsample": False},
    ]
    built = {
        "v0_7_baseline": {
            "id": "v0_7_baseline",
            "path": str(TOK_V07),
            "sha256": sha(TOK_V07),
            "vocab_size_target": 1024,
            "rebuilt": False,
        }
    }
    for spec in specs:
        corpus = list(base_corpus)
        if spec["upsample"]:
            corpus.extend(titlecase_docs(base_corpus))
        if spec["kind"] == "bl":
            tok, trainer = make_bytelevel(spec["vocab"], spec["prefix"])
        else:
            tok, trainer = make_ws(spec["vocab"])
        tok.train_from_iterator(corpus, trainer=trainer)
        path = out_dir / f"{spec['id']}.json"
        tok.save(str(path))
        loaded = Tokenizer.from_file(str(path))
        built[spec["id"]] = {
            "id": spec["id"],
            "path": str(path),
            "sha256": sha(path),
            "vocab_size_target": spec["vocab"],
            "vocab_size_actual": loaded.get_vocab_size(with_added_tokens=True),
            "kind": spec["kind"],
            "add_prefix_space": spec["prefix"],
            "titlecase_upsample_x4": spec["upsample"],
            "rebuilt": True,
            "n_train_documents": len(corpus),
        }
        if sha(TOK_V07) != TOK_SHA:
            raise RuntimeError("v0_7 mutated during candidate save")
    freeze = {
        "frozen_before_dev": True,
        "v0_7_sha256": sha(TOK_V07),
        "language_source_sha256": sha(LANG_SRC),
        "qa_train_sha256": sha(QA_TRAIN),
        "candidates": built,
        "dev_loaded": False,
    }
    write_json(STUDY / "CANDIDATE_FREEZE.json", freeze)
    return freeze


def classify(v07: dict, best: dict) -> str:
    v_shared = v07["dev"]["shared_first_token_rate"]
    b_shared = best["dev"]["shared_first_token_rate"]
    material = (
        b_shared <= 0.5 * v_shared + 1e-12
        and best["dev"]["mean_first_disambiguating_position"] <= v07["dev"]["mean_first_disambiguating_position"] + 1e-12
        and best["train"]["shared_first_token_rate"] < v07["train"]["shared_first_token_rate"] - 1e-12
        and best["language"]["bytes_per_token"] <= v07["language"]["bytes_per_token"] * 1.25 + 1e-12
    )
    if not material:
        if b_shared < v_shared - 1e-12:
            return "TOKENIZER_REPLACEMENT_WEAK"
        return "TOKENIZER_REPLACEMENT_NOT_JUSTIFIED"
    if best["dev"]["single_token_rate"] >= 0.5:
        return "TOKENIZER_REPLACEMENT_STRONGLY_JUSTIFIED"
    return "TOKENIZER_REPLACEMENT_PROMISING"


def main() -> None:
    assert_integrity()
    STUDY.mkdir(parents=True, exist_ok=True)
    texts = language_texts()
    prompts = train_prompts()
    tr_names = train_names()
    freeze = build_candidates(texts, prompts)
    # DEV only after freeze
    dev_rows = load_jsonl(QA_DEV)
    dev_names = sorted({row["correct_name"] for row in dev_rows})
    comparison = {"dev_names": dev_names, "train_names": tr_names, "candidates": {}, "test_loaded": False}
    for cid, meta in freeze["candidates"].items():
        tok = Tokenizer.from_file(meta["path"])
        lang = lang_compression(tok, texts)
        train_g = inventory_geometry(tok, tr_names)
        dev_g = inventory_geometry(tok, dev_names)
        desc = {n: encode_name(tok, n) for n in DESCRIPTIVE}
        rec = {
            **meta,
            "language": lang,
            "train": {k: v for k, v in train_g.items() if k != "by_name"},
            "dev": {k: v for k, v in dev_g.items() if k != "by_name"},
            "train_by_name": train_g["by_name"],
            "dev_by_name": dev_g["by_name"],
            "descriptive": desc,
        }
        comparison["candidates"][cid] = rec
        write_json(STUDY / "geometry" / f"{cid}.json", rec)
    v07_bpt = comparison["candidates"]["v0_7_baseline"]["language"]["bytes_per_token"]
    for cid, rec in comparison["candidates"].items():
        rec["score_dev"] = score(rec["dev"], rec["language"]["bytes_per_token"], v07_bpt)
        rec["score_train"] = score(rec["train"], rec["language"]["bytes_per_token"], v07_bpt)
    ranked = sorted(comparison["candidates"].values(), key=lambda r: (-r["score_dev"], r["id"]))
    comparison["ranking_dev"] = [r["id"] for r in ranked]
    comparison["best_id"] = ranked[0]["id"]
    comparison["classification"] = classify(comparison["candidates"]["v0_7_baseline"], ranked[0] if ranked[0]["id"] != "v0_7_baseline" else ranked[min(1, len(ranked) - 1)])
    # If baseline ranks first, classification vs next best still needed: compare best non-baseline to v0_7
    non_base = [r for r in ranked if r["id"] != "v0_7_baseline"]
    best_new = non_base[0] if non_base else ranked[0]
    comparison["best_new_id"] = best_new["id"]
    comparison["classification"] = classify(comparison["candidates"]["v0_7_baseline"], best_new)
    comparison["v0_7_still"] = sha(TOK_V07)
    write_json(STUDY / "COMPARISON.json", comparison)
    print(json.dumps({
        "best_new": comparison["best_new_id"],
        "classification": comparison["classification"],
        "ranking": comparison["ranking_dev"],
        "v0_7": sha(TOK_V07),
        "test_loaded": False,
    }, indent=2))


if __name__ == "__main__":
    main()
