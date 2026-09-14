"""Option C: append-only vocabulary extension. Never loads TEST/FINAL/sacred."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import AddedToken, Tokenizer

ROOT = Path(r"C:\DaveLM-CADAVER")
ARCH = ROOT / "baby_vnext_60m_design_v1"
sys.path.insert(0, str(ARCH))

STUDY = ROOT / "tokenizer_append_extension_v1"
TOK_V07 = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
TOK_SHA = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
TRAIN_SRC = ROOT / "baby_vnext_phase1g_language_v1" / "data" / "LANGUAGE_TRAIN_SOURCE.jsonl"
TRAIN_SRC_SHA = "807a89417924a9b1cfa1e0a2a952d97400684db32a138fe11943d93e130312c8"
LANG_TRAIN = ROOT / "baby_vnext_phase1g_language_v1" / "data" / "LANGUAGE_TRAIN_STREAM.u16"
LANG_DEV = ROOT / "baby_vnext_phase1g_language_v1" / "data" / "LANGUAGE_DEV_STREAM.u16"
LANG_STARTS = ROOT / "baby_vnext_phase1g_language_v1" / "data" / "EVAL_WINDOW_STARTS.u32"
GEN_PROMPTS = ROOT / "baby_vnext_phase1g_language_v1" / "data" / "GENERATION_PROMPTS.json"
DEV_PATH = ROOT / "phase2a_t3_rebuilt_study_v1" / "data" / "qa_dev.jsonl"
PARENT = ROOT / "baby_vnext_phase1g_language_v1_run_seed610001" / "checkpoints" / "best.pt"
PARENT_SHA = "c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1"
SEEDS = (850001, 850002, 850003)
CKPT_SHA = {
    850001: "d6f5be589c3905f76265431fa377c74245e972d9b28e8cee2c085d8f219e2684",
    850002: "f01066dd90aa3ff65b6affc7caad942486d6bfcc740733b24a440615ecf4f172",
    850003: "95e34fd52bc1b836a726816a2d8ae259e392620733146c2941aa53e00deaf597",
}
WORD = re.compile(r"[A-Z][a-z]{2,11}")
MIN_FREQ = 30
MAX_ADD = 64
BIAS_SHIFT = 8.0
OLD_V = 1024
CTX = 256
EOS = 3
BOS = 2
DOC = 4
PERIOD = 18
ADAPT_UPDATES = 100
LR = 3.75e-5
MICRO = 16
SHARED_NAMES = ("Omar", "Opal", "Sal", "Skye")
FORBIDDEN = ("qa_test.jsonl", "T2_EVAL_TEST", "FINAL", "sacred")


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


def read_u16(path: Path) -> torch.Tensor:
    raw = array("H")
    with path.open("rb") as f:
        raw.fromfile(f, os.path.getsize(path) // 2)
    if sys.byteorder != "little":
        raw.byteswap()
    return torch.tensor(raw, dtype=torch.long)


def read_u32(path: Path, count: int) -> torch.Tensor:
    raw = array("I")
    with path.open("rb") as f:
        raw.fromfile(f, count)
    if sys.byteorder != "little":
        raw.byteswap()
    return torch.tensor(raw, dtype=torch.long)


def assert_protected() -> None:
    if sha(TOK_V07) != TOK_SHA:
        raise RuntimeError("v0_7 hash changed; refusing to continue")
    for name in FORBIDDEN:
        if Path(name).is_absolute():
            raise RuntimeError("internal forbid list misuse")
    if sha(TRAIN_SRC) != TRAIN_SRC_SHA:
        raise RuntimeError("TRAIN source hash mismatch")


def ckpt_path(seed: int) -> Path:
    return ROOT / f"phase2a_t28_fast_v2_run_seed{seed}" / "rolling_restart.pt"


def load_base_config():
    from baby_vnext.config import BabyVNextConfig
    return BabyVNextConfig.load(ARCH / "BABY_VNEXT_CONFIG.json")


def make_model(vocab_size: int):
    from baby_vnext.binding import BabyVNextWithBinding
    cfg = load_base_config().to_dict()
    cfg["vocab_size"] = int(vocab_size)
    from baby_vnext.config import BabyVNextConfig
    return BabyVNextWithBinding(BabyVNextConfig.from_dict(cfg))


def load_t28(seed: int, vocab_size: int = OLD_V):
    path = ckpt_path(seed)
    got = sha(path)
    if got != CKPT_SHA[seed]:
        raise RuntimeError(f"checkpoint hash mismatch {seed}: {got}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(vocab_size)
    if vocab_size == OLD_V:
        model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload["model_state_dict"]


def expand_copy(old_sd: dict, n_new: int, candidates: list[dict], tok_old: Tokenizer):
    model = make_model(OLD_V + n_new)
    new_sd = model.state_dict()
    copied = []
    for key, value in old_sd.items():
        if key not in new_sd:
            raise RuntimeError(f"unexpected key {key}")
        target = new_sd[key]
        if tuple(value.shape) == tuple(target.shape):
            target.copy_(value)
            copied.append(key)
            continue
        if key == "base_model.token_embedding.weight" and value.shape[0] == OLD_V:
            target[:OLD_V].copy_(value)
        elif key == "base_model.language_head.weight" and value.shape[0] == OLD_V:
            target[:OLD_V].copy_(value)
        elif key == "base_model.language_head.bias" and value.shape[0] == OLD_V:
            target[:OLD_V].copy_(value)
        else:
            raise RuntimeError(f"cannot copy {key} {tuple(value.shape)} -> {tuple(target.shape)}")
    emb = new_sd["base_model.token_embedding.weight"]
    hw = new_sd["base_model.language_head.weight"]
    hb = new_sd["base_model.language_head.bias"]
    inits = []
    for i, rec in enumerate(candidates):
        idx = OLD_V + i
        pieces = list(tok_old.encode(rec["surface"]).ids)
        if not pieces:
            raise RuntimeError(f"empty v0_7 encoding for {rec['surface']!r}")
        rows = old_sd["base_model.token_embedding.weight"][pieces].mean(0)
        wrows = old_sd["base_model.language_head.weight"][pieces].mean(0)
        bmean = old_sd["base_model.language_head.bias"][pieces].mean()
        emb[idx].copy_(rows)
        hw[idx].copy_(wrows)
        hb[idx].copy_(bmean - BIAS_SHIFT)
        inits.append({"id": idx, "surface": rec["surface"], "piece_ids": pieces})
    model.load_state_dict(new_sd, strict=True)
    return model, inits


def old_rows_identical(old_sd: dict, new_model) -> dict:
    sd = new_model.state_dict()
    checks = {}
    for key in (
        "base_model.token_embedding.weight",
        "base_model.language_head.weight",
        "base_model.language_head.bias",
    ):
        a = old_sd[key]
        b = sd[key][: a.shape[0]]
        checks[key] = {
            "torch_equal": bool(torch.equal(a, b)),
            "max_abs": float((a - b).abs().max()) if a.numel() else 0.0,
        }
    others = []
    for key, value in old_sd.items():
        if key in checks:
            continue
        if not torch.equal(value, sd[key]):
            others.append(key)
    checks["other_mismatch_keys"] = others
    return checks


def phase_a() -> dict:
    assert_protected()
    tok = Tokenizer.from_file(str(TOK_V07))
    vocab = tok.get_vocab()
    max_id = max(vocab.values())
    model, sd = load_t28(850001)
    emb = sd["base_model.token_embedding.weight"]
    head_w = sd["base_model.language_head.weight"]
    head_b = sd["base_model.language_head.bias"]
    tied = model.base_model.language_head.weight.data_ptr() == model.base_model.token_embedding.weight.data_ptr()
    stream = read_u16(LANG_TRAIN)
    dummy = Tokenizer.from_file(str(TOK_V07))
    n_before = dummy.get_vocab_size(with_added_tokens=True)
    sample = " Once upon a time, Lily went to the park."
    old_ids = dummy.encode(sample).ids
    added = dummy.add_tokens([AddedToken(" Zzxq", normalized=False, special=False)])
    new_ids = dummy.encode(sample).ids
    dummy_ids = dummy.encode(" Zzxq").ids
    parent_ok = sha(PARENT) == PARENT_SHA
    report = {
        "v0_7_sha256": sha(TOK_V07),
        "v0_7_vocab_size": n_before,
        "v0_7_max_id": max_id,
        "embedding_shape": list(emb.shape),
        "language_head_weight_shape": list(head_w.shape),
        "language_head_bias_shape": list(head_b.shape),
        "tied_embedding_output": bool(tied),
        "config_vocab_size": load_base_config().vocab_size,
        "checkpoint_strict_load_old_vocab": True,
        "train_stream_token_max": int(stream.max()),
        "train_stream_token_min": int(stream.min()),
        "parent_hash_ok": parent_ok,
        "dummy_add_tokens_returned": int(added),
        "dummy_new_id": dummy.token_to_id(" Zzxq"),
        "unrelated_encode_preserved": old_ids == new_ids,
        "dummy_encodes_as_single_new_id": dummy_ids == [dummy.token_to_id(" Zzxq")],
        "old_ids_start_at_zero": min(vocab.values()) == 0,
        "append_only_valid": bool(
            n_before == OLD_V
            and max_id == OLD_V - 1
            and not tied
            and list(emb.shape) == [OLD_V, 640]
            and list(head_w.shape) == [OLD_V, 640]
            and int(stream.max()) < OLD_V
            and added == 1
            and dummy.token_to_id(" Zzxq") == OLD_V
            and old_ids == new_ids
        ),
        "notes": [
            "Historical T28/Phase1G runners load BABY_VNEXT_CONFIG.json with vocab_size=1024; this study must clone config, not overwrite it.",
            "strict=True cannot load expanded tensors; expansion must copy rows 0..1023 explicitly.",
            "LANGUAGE_TRAIN_STREAM.u16 stays v0_7; a new stream is written under this study only.",
        ],
    }
    write_json(STUDY / "PHASE_A_COMPATIBILITY.json", report)
    lines = [
        "# Phase A compatibility report",
        "",
        f"Append-only valid: **{report['append_only_valid']}**",
        "",
        f"- v0_7 SHA-256 `{report['v0_7_sha256']}` vocab {report['v0_7_vocab_size']} max id {report['v0_7_max_id']}",
        f"- embedding `{report['embedding_shape']}`, language_head `{report['language_head_weight_shape']}` bias `{report['language_head_bias_shape']}`",
        f"- tied embeddings: {report['tied_embedding_output']} (must be false)",
        f"- train stream token max {report['train_stream_token_max']}",
        f"- in-memory add_tokens preserves unrelated encodings: {report['unrelated_encode_preserved']}",
        f"- dummy new id {report['dummy_new_id']} (expected {OLD_V})",
        "",
        "Old IDs 0–1023 can be preserved. New IDs start at 1024. Old embedding/head rows can be copied bit-identically.",
        "Do not overwrite `BABY_VNEXT_CONFIG.json` or v0_7. Historical streams remain v0_7.",
        "v0_7 was not written. Dummy add was in-memory only.",
        "",
    ]
    (STUDY / "PHASE_A_COMPATIBILITY.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def phase_b() -> dict:
    assert_protected()
    tok = Tokenizer.from_file(str(TOK_V07))
    existing = set(tok.get_vocab().keys())
    counts: Counter[str] = Counter()
    n_docs = 0
    with TRAIN_SRC.open(encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"]
            n_docs += 1
            for m in WORD.finditer(text):
                surface = (" " + m.group()) if m.start() > 0 and text[m.start() - 1] == " " else m.group()
                counts[surface] += 1
    ranked = []
    for surface, freq in counts.items():
        if freq < MIN_FREQ:
            continue
        if surface in existing:
            continue
        ids = tok.encode(surface).ids
        if len(ids) < 2:
            continue
        ranked.append(
            {
                "surface": surface,
                "frequency": freq,
                "v0_7_ids": ids,
                "v0_7_n_pieces": len(ids),
                "score": freq,
            }
        )
    ranked.sort(key=lambda r: (-r["frequency"], r["surface"]))
    selected = ranked[:MAX_ADD]
    payload = {
        "corpus": str(TRAIN_SRC),
        "corpus_sha256": sha(TRAIN_SRC),
        "n_documents": n_docs,
        "min_frequency": MIN_FREQ,
        "max_added_token_count": MAX_ADD,
        "n_qualifying": len(ranked),
        "n_selected": len(selected),
        "dev_used_for_token_selection": False,
        "selected": selected,
    }
    write_json(STUDY / "TOKEN_SELECTION.json", payload)
    return payload


def build_tokenizer(selection: dict) -> dict:
    assert_protected()
    tok = Tokenizer.from_file(str(TOK_V07))
    surfaces = [r["surface"] for r in selection["selected"]]
    before = {i: tok.id_to_token(i) for i in range(OLD_V)}
    n = tok.add_tokens([AddedToken(s, normalized=False, special=False) for s in surfaces])
    if n != len(surfaces):
        raise RuntimeError(f"add_tokens returned {n} expected {len(surfaces)}")
    for i in range(OLD_V):
        if tok.id_to_token(i) != before[i]:
            raise RuntimeError(f"old id remapped at {i}")
    for i, surface in enumerate(surfaces):
        tid = tok.token_to_id(surface)
        if tid != OLD_V + i:
            raise RuntimeError(f"new token {surface!r} id {tid} expected {OLD_V + i}")
        enc = tok.encode(surface).ids
        if enc != [tid]:
            raise RuntimeError(f"new token {surface!r} encodes as {enc} not [{tid}]")
    out = STUDY / "tokenizer" / "davelm_tokenizer_v0_8_append64.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    if sha(TOK_V07) != TOK_SHA:
        raise RuntimeError("v0_7 mutated during save")
    sample_docs = load_jsonl(TRAIN_SRC)[:32]
    preserved = 0
    changed = 0
    tok_old = Tokenizer.from_file(str(TOK_V07))
    for rec in sample_docs:
        a = tok_old.encode(rec["text"]).ids
        b = tok.encode(rec["text"]).ids
        if a == b:
            preserved += 1
        else:
            changed += 1
    info = {
        "path": str(out),
        "sha256": sha(out),
        "vocab_size": tok.get_vocab_size(with_added_tokens=True),
        "tokens_added": n,
        "v0_7_still": sha(TOK_V07),
        "sample32_encode_unchanged": preserved,
        "sample32_encode_changed": changed,
    }
    write_json(STUDY / "TOKENIZER_BUILD.json", info)
    return info


def write_new_stream(tok_path: Path) -> dict:
    tok = Tokenizer.from_file(str(tok_path))
    out = STUDY / "data" / "LANGUAGE_TRAIN_STREAM_V08.u16"
    out.parent.mkdir(parents=True, exist_ok=True)
    ids: list[int] = []
    n_new = 0
    with TRAIN_SRC.open(encoding="utf-8") as f:
        first = True
        for line in f:
            rec = json.loads(line)
            if not first:
                ids.append(DOC)
            first = False
            body = tok.encode(rec["text"]).ids
            n_new += sum(1 for t in body if t >= OLD_V)
            ids.extend([BOS, *body, EOS])
    raw = array("H", ids)
    if sys.byteorder != "little":
        raw.byteswap()
    with out.open("wb") as f:
        raw.tofile(f)
    info = {
        "path": str(out),
        "sha256": sha(out),
        "n_tokens": len(ids),
        "n_new_token_occurrences": n_new,
        "token_max": max(ids),
    }
    write_json(STUDY / "NEW_TRAIN_STREAM.json", info)
    return info


def configure(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def lang_ce(model, stream, starts, device, vocab_slice: int | None, microbatch=8):
    was = model.training
    model.eval()
    total = 0.0
    tokens = 0
    off = torch.arange(CTX, device=device)
    try:
        for i in range(0, len(starts), microbatch):
            s = starts[i : i + microbatch].to(device)
            x = stream[s[:, None] + off[None, :]]
            y = stream[s[:, None] + off[None, :] + 1]
            logits, _ = model(x)
            if vocab_slice is not None:
                logits = logits[..., :vocab_slice]
            total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").cpu())
            tokens += int(y.numel())
    finally:
        model.train(was)
    return total / tokens


@torch.no_grad()
def old_logit_match(a, b, stream, starts, device, n_windows=16) -> dict:
    a.eval()
    b.eval()
    off = torch.arange(CTX, device=device)
    s = starts[:n_windows].to(device)
    x = stream[s[:, None] + off[None, :]]
    za, _ = a(x)
    zb, _ = b(x)
    diff = (za - zb[..., :OLD_V]).abs()
    return {
        "max_abs": float(diff.max().cpu()),
        "mean_abs": float(diff.mean().cpu()),
        "allclose_1e-6": bool(torch.allclose(za, zb[..., :OLD_V], atol=1e-6, rtol=0)),
    }


def fact_clause_spans_bos(row):
    prompt = [BOS] + [int(x) for x in row["prompt_token_ids"]]
    spans = []
    for mention in row["mention_span_bos"]:
        start = int(mention[0])
        end = int(mention[-1])
        for i in range(end, len(prompt)):
            if prompt[i] == PERIOD:
                end = i
                break
        else:
            raise RuntimeError(f"no period after mention in {row.get('id')}")
        spans.append(list(range(start, end + 1)))
    return spans


@torch.no_grad()
def pointer_ok(model, row, device) -> bool:
    prompt = [BOS] + [int(x) for x in row["prompt_token_ids"]]
    hs = model.base_model.forward_hidden(torch.tensor([prompt], device=device))
    q = F.normalize(hs[0, len(prompt) - 1], dim=-1)
    sims = []
    for span in fact_clause_spans_bos(row):
        idx = torch.tensor(span, device=device, dtype=torch.long)
        k = F.normalize(hs[0, idx].mean(0), dim=-1)
        sims.append((q * k).sum())
    pred = int(torch.stack(sims).argmax())
    return pred == int(row["correct_index"])


@torch.no_grad()
def greedy(model, prompt: list[int], device, max_new=12) -> list[int]:
    gen = list(prompt)
    out = []
    for _ in range(max_new):
        logits, _ = model(torch.tensor([gen[-CTX:]], device=device))
        t = int(logits[0, -1].argmax())
        gen.append(t)
        out.append(t)
        if t == EOS:
            break
    return out


def classify(gen: list[int], target: list[int]) -> dict:
    exact = gen == target
    first = bool(gen) and bool(target) and gen[0] == target[0]
    if exact:
        mode = "exact"
    elif first:
        mode = "first_token_correct_then_diverge"
    elif gen and gen[-1] == EOS:
        mode = "other_then_eos"
    else:
        mode = "other"
    return {"exact": exact, "first_token_correct": first, "mode": mode, "ids": gen}


def retokenize_dev(tok: Tokenizer, rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        prompt_ids = tok.encode(row["prompt"]).ids
        cands = []
        for name in row["candidates"]:
            surface = " " + name + "."
            cands.append(tok.encode(surface).ids)
        out.append(
            {
                **row,
                "prompt_token_ids_v08": prompt_ids,
                "candidate_token_ids_v08": cands,
            }
        )
    return out


def eval_native(model, rows, device, prompt_key, cand_key, shared_fn):
    recs = []
    for row in rows:
        prompt = [BOS] + list(map(int, row[prompt_key]))
        target = list(map(int, row[cand_key][int(row["correct_index"])])) + [EOS]
        gen = greedy(model, prompt, device)
        info = classify(gen, target)
        recs.append(
            {
                "id": row["id"],
                "name": row["correct_name"],
                "shared_prefix": bool(shared_fn(row)),
                **info,
            }
        )
    def stats(sel):
        n = len(sel)
        exact = sum(r["exact"] for r in sel)
        div = sum(r["mode"] == "first_token_correct_then_diverge" for r in sel)
        return {
            "n": n,
            "exact": exact,
            "exact_rate": exact / n if n else 0.0,
            "diverge": div,
            "diverge_rate": div / n if n else 0.0,
        }
    by_name = {}
    for name in sorted({r["name"] for r in recs}):
        by_name[name] = stats([r for r in recs if r["name"] == name])
    return {
        "all": stats(recs),
        "shared": stats([r for r in recs if r["shared_prefix"]]),
        "unique": stats([r for r in recs if not r["shared_prefix"]]),
        "by_name": by_name,
        "items": recs,
    }


def freeze_new_params(model) -> dict:
    n_train = 0
    n_freeze = 0
    for name, p in model.named_parameters():
        train = False
        if name == "base_model.token_embedding.weight":
            train = True
        if name == "base_model.language_head.weight":
            train = True
        if name == "base_model.language_head.bias":
            train = True
        p.requires_grad_(train)
        if train:
            n_train += p.numel()
        else:
            n_freeze += p.numel()
            p.grad = None
    return {"trainable_tensors": 3, "trainable_numel_full_tensors": n_train, "frozen_numel": n_freeze}


def masked_new_row_grads(model):
    emb = model.base_model.token_embedding.weight
    hw = model.base_model.language_head.weight
    hb = model.base_model.language_head.bias
    if emb.grad is not None:
        emb.grad[:OLD_V] = 0
    if hw.grad is not None:
        hw.grad[:OLD_V] = 0
    if hb.grad is not None:
        hb.grad[:OLD_V] = 0


def u0_and_adapt(selection: dict, tok_info: dict, stream_info: dict) -> dict:
    assert_protected()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable")
    device = torch.device("cuda")
    tok_old = Tokenizer.from_file(str(TOK_V07))
    tok_new = Tokenizer.from_file(tok_info["path"])
    stream_old = read_u16(LANG_TRAIN).to(device)
    starts = read_u32(LANG_STARTS, 1600)
    dev_starts = starts[:1280]
    new_stream_cpu = read_u16(Path(stream_info["path"]))
    new_stream = new_stream_cpu.to(device)
    n_new = len(selection["selected"])
    max_start = int(new_stream_cpu.numel()) - CTX - 2
    rows = load_jsonl(DEV_PATH)
    first_ids = defaultdict(set)
    for r in rows:
        ids = list(map(int, r["candidate_token_ids"][int(r["correct_index"])]))
        first_ids[ids[0]].add(r["correct_name"])
    rows_v08 = retokenize_dev(tok_new, rows)
    results = {"seeds": {}, "test_loaded": False, "v0_7_sha256": sha(TOK_V07)}
    for seed in SEEDS:
        configure(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        orig, old_sd = load_t28(seed)
        orig.to(device).eval()
        expanded, inits = expand_copy(old_sd, n_new, selection["selected"], tok_old)
        ident = old_rows_identical(old_sd, expanded)
        expanded.to(device)
        sliced_ce_orig = lang_ce(orig, stream_old, dev_starts, device, None)
        sliced_ce_new = lang_ce(expanded, stream_old, dev_starts, device, OLD_V)
        full_ce_new = lang_ce(expanded, stream_old, dev_starts, device, None)
        logit = old_logit_match(orig, expanded, stream_old, dev_starts, device)
        ptr_orig = sum(pointer_ok(orig, r, device) for r in rows)
        ptr_new = sum(pointer_ok(expanded, r, device) for r in rows)
        u0_ok = (
            ident["base_model.token_embedding.weight"]["torch_equal"]
            and ident["base_model.language_head.weight"]["torch_equal"]
            and ident["base_model.language_head.bias"]["torch_equal"]
            and not ident["other_mismatch_keys"]
            and logit["allclose_1e-6"]
            and ptr_orig == ptr_new
            and abs(sliced_ce_orig - sliced_ce_new) <= 1e-5
        )
        seed_rec = {
            "u0": {
                "old_rows": ident,
                "old_logits": logit,
                "sliced_ce_orig": sliced_ce_orig,
                "sliced_ce_expanded": sliced_ce_new,
                "full_vocab_ce_on_v07_stream": full_ce_new,
                "pointer_v07_orig": ptr_orig,
                "pointer_v07_expanded": ptr_new,
                "hard_stop_pass": u0_ok,
            },
            "init_preview": inits[:8],
        }
        if not u0_ok:
            seed_rec["stopped"] = "U0_HARD_STOP"
            results["seeds"][str(seed)] = seed_rec
            results["hard_stop"] = True
            write_json(STUDY / "RUN_RESULTS.json", results)
            return results
        native_v07_u0 = eval_native(
            expanded,
            rows,
            device,
            "prompt_token_ids",
            "candidate_token_ids",
            lambda r: len(first_ids[r["candidate_token_ids"][int(r["correct_index"])][0]]) > 1,
        )
        native_v08_u0 = eval_native(
            expanded,
            rows_v08,
            device,
            "prompt_token_ids_v08",
            "candidate_token_ids_v08",
            lambda r: r["correct_name"] in SHARED_NAMES,
        )
        seed_rec["u0"]["native_v07_gold"] = {k: native_v07_u0[k] for k in ("all", "shared", "unique", "by_name")}
        seed_rec["u0"]["native_v08_gold"] = {k: native_v08_u0[k] for k in ("all", "shared", "unique", "by_name")}

        freeze_new_params(expanded)
        frozen_emb = old_sd["base_model.token_embedding.weight"].to(device)
        frozen_hw = old_sd["base_model.language_head.weight"].to(device)
        frozen_hb = old_sd["base_model.language_head.bias"].to(device)
        opt = torch.optim.AdamW(
            [p for p in expanded.parameters() if p.requires_grad],
            lr=LR,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.05,
            foreach=False,
            fused=False,
        )
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        expanded.train()
        losses = []
        for _u in range(ADAPT_UPDATES):
            starts_b = torch.randint(0, max_start + 1, (MICRO,), generator=g)
            off = torch.arange(CTX)
            x = new_stream_cpu[starts_b[:, None] + off[None, :]].to(device)
            y = new_stream_cpu[starts_b[:, None] + off[None, :] + 1].to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = expanded(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            loss.backward()
            masked_new_row_grads(expanded)
            torch.nn.utils.clip_grad_norm_([p for p in expanded.parameters() if p.requires_grad], 2.0)
            opt.step()
            with torch.no_grad():
                expanded.base_model.token_embedding.weight[:OLD_V].copy_(frozen_emb)
                expanded.base_model.language_head.weight[:OLD_V].copy_(frozen_hw)
                expanded.base_model.language_head.bias[:OLD_V].copy_(frozen_hb)
            losses.append(float(loss.detach().cpu()))
        expanded.eval()
        ident_after = {
            "emb": bool(torch.equal(expanded.base_model.token_embedding.weight[:OLD_V], frozen_emb)),
            "head_w": bool(torch.equal(expanded.base_model.language_head.weight[:OLD_V], frozen_hw)),
            "head_b": bool(torch.equal(expanded.base_model.language_head.bias[:OLD_V], frozen_hb)),
        }
        if not all(ident_after.values()):
            raise RuntimeError(f"old rows mutated after adaptation: {ident_after}")
        sliced_after = lang_ce(expanded, stream_old, dev_starts, device, OLD_V)
        ptr_after = sum(pointer_ok(expanded, r, device) for r in rows)
        native_v07 = eval_native(
            expanded,
            rows,
            device,
            "prompt_token_ids",
            "candidate_token_ids",
            lambda r: len(first_ids[r["candidate_token_ids"][int(r["correct_index"])][0]]) > 1,
        )
        native_v08 = eval_native(
            expanded,
            rows_v08,
            device,
            "prompt_token_ids_v08",
            "candidate_token_ids_v08",
            lambda r: r["correct_name"] in SHARED_NAMES,
        )
        ckpt_out = STUDY / "checkpoints" / f"expanded_adapt100_seed{seed}.pt"
        ckpt_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "tokenizer_append_only_extension_v1",
                "seed": seed,
                "parent_t28_sha256": CKPT_SHA[seed],
                "model_state_dict": expanded.state_dict(),
                "vocab_size": OLD_V + n_new,
            },
            ckpt_out,
        )
        seed_rec["adapt"] = {
            "updates": ADAPT_UPDATES,
            "mean_loss": sum(losses) / len(losses),
            "last_loss": losses[-1],
            "sliced_ce_after": sliced_after,
            "pointer_v07_after": ptr_after,
            "old_rows_bit_identical_after": ident_after,
            "native_v07_gold": {k: native_v07[k] for k in ("all", "shared", "unique", "by_name")},
            "native_v08_gold": {k: native_v08[k] for k in ("all", "shared", "unique", "by_name")},
            "checkpoint": str(ckpt_out),
        }
        write_json(STUDY / "eval" / f"native_v07_seed{seed}.json", native_v07)
        write_json(STUDY / "eval" / f"native_v08_seed{seed}.json", native_v08)
        results["seeds"][str(seed)] = seed_rec
        del orig, expanded
        torch.cuda.empty_cache()
    results["hard_stop"] = False
    write_json(STUDY / "RUN_RESULTS.json", results)
    return results


def summarize(results: dict, selection: dict, tok_info: dict) -> dict:
    seeds = results["seeds"]
    def collect(path):
        vals = []
        for s in SEEDS:
            cur = seeds[str(s)]
            for p in path:
                cur = cur[p]
            vals.append(cur)
        return vals
    shared_after = collect(["adapt", "native_v08_gold", "shared", "exact"])
    unique_after = collect(["adapt", "native_v08_gold", "unique", "exact"])
    overall_after = collect(["adapt", "native_v08_gold", "all", "exact"])
    diverge_after = collect(["adapt", "native_v08_gold", "shared", "diverge"])
    shared_v07 = collect(["adapt", "native_v07_gold", "shared", "exact"])
    wes = []
    for s in SEEDS:
        wes.append(seeds[str(s)]["adapt"]["native_v08_gold"]["by_name"].get("Wes", {}).get("exact", 0))
    summary = {
        "APPEND_ONLY_COMPATIBILITY": True,
        "OLD_TOKEN_IDS_PRESERVED": True,
        "OLD_WEIGHTS_BIT_IDENTICAL": all(
            seeds[str(s)]["u0"]["old_rows"]["base_model.token_embedding.weight"]["torch_equal"] for s in SEEDS
        ),
        "NEW_TOKENIZER_VERSION": "v0_8_append64",
        "TOKENS_ADDED": selection["n_selected"],
        "SELECTION_RULE": "title-case word surface forms fragmented under v0_7 on Phase1G TRAIN; freq>=30; top 64; lex tie-break",
        "DEV_USED_FOR_TOKEN_SELECTION": False,
        "MINIMAL_ADAPTATION_REQUIRED": True,
        "TRAINABLE_SCOPE": "new embedding/output rows only (old rows frozen; grads zeroed)",
        "SHARED_PREFIX_EXACT_BEFORE": "9/192 (4.7%) greedy T28",
        "SHARED_PREFIX_EXACT_AFTER_V08_GOLD": shared_after,
        "UNIQUE_PREFIX_EXACT_BEFORE": "102/192 (53.1%) greedy T28",
        "UNIQUE_PREFIX_EXACT_AFTER_V08_GOLD": unique_after,
        "DIVERGENCE_BEFORE": "83/192 (43.2%) shared first-token-correct-then-diverge",
        "DIVERGENCE_AFTER_V08_SHARED": diverge_after,
        "OVERALL_EXACT_BEFORE": "111/384 (28.9%)",
        "OVERALL_EXACT_AFTER_V08": overall_after,
        "SHARED_PREFIX_EXACT_AFTER_V07_GOLD": shared_v07,
        "LANGUAGE_STATUS": {
            str(s): {
                "u0_sliced_ce": seeds[str(s)]["u0"]["sliced_ce_orig"],
                "after_sliced_ce": seeds[str(s)]["adapt"]["sliced_ce_after"],
            }
            for s in SEEDS
        },
        "REPRESENTATION_STATUS": {
            str(s): {
                "pointer_u0": seeds[str(s)]["u0"]["pointer_v07_orig"],
                "pointer_after": seeds[str(s)]["adapt"]["pointer_v07_after"],
            }
            for s in SEEDS
        },
        "WES_FAMILY_EXACT_V08": wes,
        "PROTECTED_DATA": "LOCKED; TEST not loaded",
        "T29_LAUNCHED": False,
        "V0_7_SHA256": sha(TOK_V07),
        "NEW_TOKENIZER_SHA256": tok_info["sha256"],
    }
    write_json(STUDY / "OPERATOR_SUMMARY.json", summary)
    return summary


def write_final_md(summary: dict, results: dict, selection: dict) -> None:
    surfaces = [r["surface"] for r in selection["selected"]]
    lines = [
        "# Tokenizer append-only extension v1 results",
        "",
        "Option C executed. T29 not launched. v0_7 not overwritten. TEST not loaded.",
        "",
        "## Compatibility",
        "",
        f"- Append-only valid: yes",
        f"- Old token IDs preserved: yes",
        f"- Old weights bit-identical after copy and after 100 new-row updates: {summary['OLD_WEIGHTS_BIT_IDENTICAL']}",
        f"- New tokenizer: `{summary['NEW_TOKENIZER_VERSION']}` SHA-256 `{summary['NEW_TOKENIZER_SHA256']}`",
        f"- Tokens added: {summary['TOKENS_ADDED']}",
        f"- DEV used for token selection: NO",
        "",
        "## Selection (TRAIN only)",
        "",
        "Title-case words currently fragmented by v0_7, frequency ≥ 30, top 64, lexicographic ties.",
        "Added surfaces (repr): " + ", ".join(repr(s) for s in surfaces),
        "",
        "## Native greedy after adaptation",
        "",
        "Baseline (T28 greedy, v0_7 gold): shared 9/192 (4.7%), unique 102/192 (53.1%), overall 111/384.",
        "",
        f"- Shared exact vs v0_8 gold by seed: {summary['SHARED_PREFIX_EXACT_AFTER_V08_GOLD']} /64 names-in-shared-set-per-seed (Omar/Opal/Sal/Skye; 64 rows/seed)",
        f"- Unique exact vs v0_8 gold by seed: {summary['UNIQUE_PREFIX_EXACT_AFTER_V08_GOLD']}",
        f"- Overall exact vs v0_8 gold by seed: {summary['OVERALL_EXACT_AFTER_V08']}",
        f"- Shared exact vs **v0_7 gold** (expanded head, old prompts): {summary['SHARED_PREFIX_EXACT_AFTER_V07_GOLD']}",
        f"- Wes exact vs v0_8 gold by seed: {summary['WES_FAMILY_EXACT_V08']}",
        f"- Language sliced CE: {json.dumps(summary['LANGUAGE_STATUS'])}",
        f"- Pointer on v0_7 encodings: {json.dumps(summary['REPRESENTATION_STATUS'])}",
        "",
        "## Interpretation rule",
        "",
        "Constrained rerank is not used. Success requires material shared-prefix exact gain,",
        "lower first-token-correct-then-diverge, no material unique-prefix collapse, language",
        "and pointer retention, native free generation improvement, and replication.",
        "If this fails, Option B/D still require owner approval; do not auto-launch.",
        "",
    ]
    (STUDY / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("a", "b", "all"))
    args = ap.parse_args()
    STUDY.mkdir(parents=True, exist_ok=True)
    a = phase_a()
    if not a["append_only_valid"]:
        write_json(STUDY / "STATUS.json", {"status": "STOP_INCOMPATIBLE", "test_loaded": False})
        print(json.dumps(a, indent=2))
        return
    if args.stage == "a":
        print(json.dumps(a, indent=2))
        return
    b = phase_b()
    tok_info = build_tokenizer(b)
    stream_info = write_new_stream(Path(tok_info["path"]))
    results = u0_and_adapt(b, tok_info, stream_info)
    if results.get("hard_stop"):
        write_json(STUDY / "STATUS.json", {"status": "U0_HARD_STOP", "test_loaded": False, "t29": False})
        print("U0 HARD STOP")
        return
    summary = summarize(results, b, tok_info)
    write_final_md(summary, results, b)
    write_json(
        STUDY / "STATUS.json",
        {
            "status": "COMPLETE",
            "test_loaded": False,
            "t29_launched": False,
            "v0_7_preserved": sha(TOK_V07) == TOK_SHA,
            "tokenizer_version": "v0_8_append64",
        },
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
