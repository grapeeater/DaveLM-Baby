from __future__ import annotations

from array import array
import hashlib
import json
import sys
from pathlib import Path
import shutil

import torch
from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parent
ARCH = Path(r"C:\DaveLM-CADAVER\baby_vnext_60m_design_v1")
PILOT0 = Path(r"C:\DaveLM-CADAVER\language_pilot_0_tinystories_seed8380")
PILOT1 = Path(r"C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380")
P1 = Path(r"C:\DaveLM-CADAVER\baby_vnext_phase1_language_v1")
TOKENIZER = Path(r"C:\DaveLM-v0.9\tokenizer\v0_7\davelm_tokenizer.json")
SOURCE = PILOT0 / "TinyStories-valid.txt"

SOURCE_SHA256 = "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
TOKENIZER_SHA256 = "e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b"
P1_TRAIN_STREAM_SHA256 = "c88b4d0857eed7146125698f735da1fcbaaa4977b6ba4e213b837b29824d9804"
P1_DEV_STREAM_SHA256 = "4e9834d43ec80a0eea5e1a5a59c284f8f76f5f84bffcf979e4686661d71b6017"
P1_EVAL_SELECTION_SHA256 = "744904653ed11605025318cd4e5579ef6a33bd516575834086daa24b61a76681"
P1_PROMPTS_SHA256 = "6bcc77244efc5dad70ef905d275aeed6495b16c76decc4ae2e1684aaaa83cf51"
P1_INIT_SHA256 = "039e8efe7353d5065c06c2519053695e5944d58eeee2765594a8dfadb80d821c"
DEV_SOURCE_SHA256 = "deff4fc7ed18e6e1f0b6f32faccc80f1eb44d0e58f3f38d512ffdfab798d6ac4"

RUN_SEED = 610001
TRAIN_SCHEDULE_SEED = 61000111
EVAL_SELECTION_SEED = 61000102
RESERVED_REPLICATION_SEEDS = [610002, 610003]
MAX_UPDATES = 6000
EFFECTIVE_BATCH = 64
CONTEXT = 256
HEAD_N = 9000
DEV_N = 1000

STUDY = "BABY_VNEXT_PHASE1G_LANGUAGE_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_chain(root: Path) -> dict[str, object]:
    detached = (root / "FREEZE_RECEIPT.sha256").read_text(encoding="ascii").split()[0]
    receipt = root / "FREEZE_RECEIPT.json"
    if sha256(receipt) != detached:
        raise RuntimeError("architecture receipt mismatch")
    parsed = json.loads(receipt.read_text(encoding="utf-8"))
    manifest = root / "SHA256SUMS.txt"
    if sha256(manifest) != parsed["manifest_sha256"]:
        raise RuntimeError("architecture manifest mismatch")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if sha256(root / relative) != expected:
            raise RuntimeError(f"architecture payload mismatch: {relative}")
    return {"receipt_sha256": detached, "manifest_sha256": parsed["manifest_sha256"]}


def collect_arrays(obj):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("full_document_token_ids", "token_ids", "document_token_ids") and isinstance(v, list) and v and all(isinstance(x, int) for x in v):
                out.append(tuple(v))
            else:
                out.extend(collect_arrays(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(collect_arrays(v))
    return out


def encode_stream(tokenizer: Tokenizer, rows: list[dict[str, object]]) -> list[int]:
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    document = tokenizer.token_to_id("<doc>")
    if None in (bos, eos, document):
        raise RuntimeError("tokenizer is missing a required stream token")
    stream: list[int] = []
    for index, row in enumerate(rows):
        if index:
            stream.append(int(document))
        stream.append(int(bos))
        stream.extend(tokenizer.encode(str(row["text"])).ids)
        stream.append(int(eos))
    return stream


def write_u16(path: Path, values: list[int]) -> None:
    if min(values) < 0 or max(values) > 65535:
        raise ValueError("u16 stream value out of range")
    packed = array("H", values)
    if sys.byteorder != "little":
        packed.byteswap()
    path.write_bytes(packed.tobytes())


def write_u32(path: Path, values: list[int]) -> None:
    packed = array("I", values)
    if packed.itemsize != 4:
        raise RuntimeError("platform unsigned-int is not 32 bits")
    if sys.byteorder != "little":
        packed.byteswap()
    path.write_bytes(packed.tobytes())


def starts(generator: torch.Generator, count: int, stream_length: int) -> list[int]:
    return torch.randint(0, stream_length - (CONTEXT + 1), size=(count,), generator=generator).tolist()


def recs(xs: list[str]) -> list[dict[str, object]]:
    return [{"id": i, "text": s, "token_ids": tok.encode(s).ids} for i, s in enumerate(xs)]


def main() -> None:
    verify_chain(ARCH)
    if sha256(TOKENIZER) != TOKENIZER_SHA256:
        raise RuntimeError("tokenizer mismatch")
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("TinyStories source file mismatch")
    dev_source = PILOT1 / "language_dev.jsonl"
    if sha256(dev_source) != DEV_SOURCE_SHA256:
        raise RuntimeError("Phase-1 DEV source mismatch")

    global tok
    tok = Tokenizer.from_file(str(TOKENIZER))
    if tok.get_vocab_size() != 1024:
        raise RuntimeError("unexpected vocabulary size")

    raw = SOURCE.read_text(encoding="utf-8")
    stories = [x.strip() for x in raw.split("<|endoftext|>") if x.strip()]
    norm = [s.replace("\r\n", "\n").replace("\r", "\n") for s in stories]
    uniq = {hashlib.sha256(s.encode("utf-8")).hexdigest(): s for s in norm}
    keys = sorted(uniq)
    if len(keys) < HEAD_N + DEV_N:
        raise RuntimeError(f"not enough unique stories: {len(keys)}")

    head = keys[:HEAD_N]
    dev = keys[HEAD_N:HEAD_N + DEV_N]
    tail = keys[HEAD_N + DEV_N:]
    if len(dev) != DEV_N:
        raise RuntimeError("DEV reserve size mismatch")
    if not tail:
        raise RuntimeError("no additional stories available beyond the first 10,000")

    head_texts = [uniq[k] for k in head]
    dev_texts = [uniq[k] for k in dev]
    tail_texts = [uniq[k] for k in tail]

    head_rows = recs(head_texts)
    dev_rows = recs(dev_texts)
    tail_rows = recs(tail_texts)
    train_rows = head_rows + tail_rows

    if len(head_rows) != HEAD_N or len(tail_rows) != len(tail) or len(train_rows) != HEAD_N + len(tail):
        raise RuntimeError("row-count mismatch")

    p1_train_path = P1 / "data" / "LANGUAGE_TRAIN_STREAM.u16"
    if sha256(p1_train_path) != P1_TRAIN_STREAM_SHA256:
        raise RuntimeError("Phase-1 train stream hash mismatch")
    p1_train_bytes = p1_train_path.read_bytes()
    p1_total = len(p1_train_bytes) // 2

    head_stream = encode_stream(tok, head_rows)
    packed = array("H", head_stream)
    if sys.byteorder != "little":
        packed.byteswap()
    head_bytes = packed.tobytes()
    if len(head_bytes) != len(p1_train_bytes):
        raise RuntimeError(f"head stream length {len(head_stream)} != Phase-1 train stream length {p1_total}")
    if head_bytes != p1_train_bytes:
        raise RuntimeError("head stream is NOT byte-identical to the frozen Phase-1 train stream prefix")

    full_stream = encode_stream(tok, train_rows)
    dev_stream_enc = encode_stream(tok, dev_rows)

    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "initialization").mkdir(parents=True, exist_ok=True)

    train_source_path = ROOT / "data" / "LANGUAGE_TRAIN_SOURCE.jsonl"
    train_stream_path = ROOT / "data" / "LANGUAGE_TRAIN_STREAM.u16"
    dev_stream_path = ROOT / "data" / "LANGUAGE_DEV_STREAM.u16"
    schedule_path = ROOT / "data" / "TRAIN_WINDOW_STARTS.u32"
    eval_selection_path = ROOT / "data" / "EVAL_WINDOW_STARTS.u32"
    prompts_path = ROOT / "data" / "GENERATION_PROMPTS.json"
    init_path = ROOT / "initialization" / "seed_610001_initialized.pt"

    train_source_path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in train_rows), encoding="utf-8"
    )
    write_u16(train_stream_path, full_stream)
    if sha256(dev_source) != DEV_SOURCE_SHA256:
        raise RuntimeError("DEV source mismatch (unexpected)")
    shutil.copyfile(P1 / "data" / "LANGUAGE_DEV_STREAM.u16", dev_stream_path)
    shutil.copyfile(P1 / "data" / "EVAL_WINDOW_STARTS.u32", eval_selection_path)
    shutil.copyfile(P1 / "data" / "GENERATION_PROMPTS.json", prompts_path)
    shutil.copyfile(P1 / "initialization" / "seed_610001_initialized.pt", init_path)

    for path, expected, name in (
        (dev_stream_path, P1_DEV_STREAM_SHA256, "DEV stream"),
        (eval_selection_path, P1_EVAL_SELECTION_SHA256, "eval selection"),
        (prompts_path, P1_PROMPTS_SHA256, "generation prompts"),
        (init_path, P1_INIT_SHA256, "initialized checkpoint"),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"{name} copy mismatch after copy")

    if sha256(init_path) != P1_INIT_SHA256:
        raise RuntimeError("initialized checkpoint copy mismatch")

    init_raw = torch.load(init_path, map_location="cpu", weights_only=False)
    if int(init_raw["optimizer_updates"]) != 0:
        raise RuntimeError("initialized checkpoint records an optimizer update")

    generator = torch.Generator().manual_seed(TRAIN_SCHEDULE_SEED)
    train_starts: list[int] = []
    for _ in range(MAX_UPDATES):
        train_starts.extend(starts(generator, EFFECTIVE_BATCH, len(full_stream)))
    write_u32(schedule_path, train_starts)
    if max(train_starts) >= len(full_stream) - (CONTEXT + 1):
        raise RuntimeError("schedule start exceeds stream bounds")

    eval_generator = torch.Generator().manual_seed(EVAL_SELECTION_SEED)
    _ = eval_generator  # eval selection file is frozen and reused byte-for-byte

    eval_data = torch.frombuffer(eval_selection_path.read_bytes(), dtype=torch.uint32).to(torch.int64)
    dev_count = 1280
    train_count = 320
    dev_starts = eval_data[:dev_count]
    train_eval_starts = eval_data[dev_count:dev_count + train_count]
    if int(dev_starts.max()) >= int(torch.frombuffer(dev_stream_path.read_bytes(), dtype=torch.uint16).numel()) - CONTEXT - 1:
        raise RuntimeError("frozen DEV eval window exceeds DEV stream bounds")
    if int(train_eval_starts.max()) >= len(full_stream) - CONTEXT - 1:
        raise RuntimeError("frozen train eval window exceeds new train stream bounds")
    if int(train_eval_starts.max()) >= p1_total - CONTEXT - 1:
        raise RuntimeError("frozen train eval window exceeds head-prefix bounds")

    head_arr = {tuple(x["token_ids"]) for x in head_rows}
    dev_arr = {tuple(x["token_ids"]) for x in dev_rows}
    tail_arr = {tuple(x["token_ids"]) for x in tail_rows}
    if head_arr & dev_arr:
        raise RuntimeError("head/dev exact token-array overlap")
    if tail_arr & dev_arr:
        raise RuntimeError("tail/dev exact token-array overlap")
    if tail_arr & head_arr:
        raise RuntimeError("tail/head exact token-array overlap")

    dev_stream_ids = torch.frombuffer(dev_stream_path.read_bytes(), dtype=torch.uint16).to(torch.int64)
    dev_expected = torch.tensor(dev_stream_enc, dtype=torch.int64)
    if dev_stream_ids.numel() != dev_expected.numel() or not bool((dev_stream_ids == dev_expected).all()):
        raise RuntimeError("frozen DEV stream does not match re-encoded DEV reserve")

    prior_overlaps = {}
    prior_files = []
    for p in sorted(Path(r"C:\DaveLM-CADAVER").rglob("*.json")):
        if p.is_relative_to(ROOT):
            continue
        if p.is_relative_to(PILOT0) or p.is_relative_to(PILOT1):
            continue
        if "pool" in p.name.lower():
            prior_files.append(p)
    prior_files += [PILOT0 / "binding_rehearsal.json", PILOT0 / "binding_dev.json",
                    PILOT1 / "binding_rehearsal.json", PILOT1 / "binding_dev.json"]
    for p in sorted(set(prior_files)):
        try:
            if p.stat().st_size > 300 * 1024 * 1024:
                continue
            arr = {tuple(x) for x in collect_arrays(json.loads(p.read_text(encoding="utf-8")))}
        except Exception:
            continue
        if not arr:
            continue
        overlap = tail_arr & arr
        if overlap:
            prior_overlaps[str(p)] = len(overlap)
    if prior_overlaps:
        raise RuntimeError(f"tail exact token-array overlap with prior pools: {prior_overlaps}")

    total_positions = MAX_UPDATES * EFFECTIVE_BATCH * CONTEXT
    train_stream_tokens = len(full_stream)

    config = {
        "status": "PROSPECTIVE_READY_TO_TRAIN",
        "study": STUDY,
        "scientific_purpose": "Phase-1G generalization-diet treatment for the 61.52M capacity successor: enlarge same-distribution unique training coverage from 9,000 to 20,990 TinyStories-valid stories at fixed compute; same architecture/optimizer/schedule/eval",
        "architecture": {
            "bundle": str(ARCH),
            "receipt_sha256": "776859cb1bf671634ae2d7e8dfab7f7cb8e517ce802d30d3cb3733b6b734b368",
            "manifest_sha256": "a4a9fb219f082e1936bf7341f42d82354fc2c7cf8d6f9b0d03c84e815117cad4",
            "config_path": str(ARCH / "BABY_VNEXT_CONFIG.json"),
            "config_sha256": "8d7e1cbc604c9bf9f1942d4afc01817b20af17d8805a984f7ae57b1254f957b4",
            "trained_parameters": 61520385,
            "base_parameters": 60536064,
            "binding_parameters": 984321,
        },
        "runtime": {
            "python": r"C:\DaveLM-CADAVER\sf2_runtime\sf2venv\Scripts\python.exe",
            "python_version": "3.12.14",
            "torch": "2.12.0+rocm7.14.0",
            "tokenizers": "0.23.1",
            "device": "AMD Radeon RX 9060 XT",
            "deterministic_algorithms": True,
            "tf32": False,
            "autocast": False,
            "precision": "float32",
            "sdpa_expected_backend": "math",
        },
        "seed_policy": {
            "primary_run_seed": RUN_SEED,
            "initialization_seed": RUN_SEED,
            "dropout_and_runtime_seed": RUN_SEED,
            "training_schedule_seed": TRAIN_SCHEDULE_SEED,
            "evaluation_selection_seed": EVAL_SELECTION_SEED,
            "reserved_replication_run_seeds": RESERVED_REPLICATION_SEEDS,
            "replication_status": "reserved_not_authorized",
        },
        "initialization": {
            "strategy": "from_scratch_pytorch_module_defaults_v1 (identical sealed Phase-1 checkpoint bytes)",
            "checkpoint": str(init_path),
            "checkpoint_sha256": P1_INIT_SHA256,
            "model_state_sha256": init_raw["model_state_sha256"],
            "optimizer_created": False,
            "optimizer_updates": 0,
        },
        "data": {
            "tokenizer_path": str(TOKENIZER),
            "tokenizer_sha256": TOKENIZER_SHA256,
            "train_source": str(train_source_path),
            "train_source_sha256": sha256(train_source_path),
            "dev_source": str(dev_source),
            "dev_source_sha256": DEV_SOURCE_SHA256,
            "train_documents": len(train_rows),
            "dev_documents": DEV_N,
            "train_stream": str(train_stream_path),
            "train_stream_sha256": sha256(train_stream_path),
            "train_stream_tokens": train_stream_tokens,
            "dev_stream": str(dev_stream_path),
            "dev_stream_sha256": P1_DEV_STREAM_SHA256,
            "dev_stream_tokens": int(torch.frombuffer(dev_stream_path.read_bytes(), dtype=torch.uint16).numel()),
            "stream_format": "little-endian unsigned 16-bit token IDs",
            "stream_construction": "<doc> between documents; <bos> + text tokens + <eos> per document; head 9,000 frozen Phase-1 docs form an exact prefix",
            "window_sampling": "materialized Pilot1-compatible torch.randint starts over concatenated stream",
            "padding": "none",
            "truncation": "fixed 256-token windows only",
            "train_dev_separate": True,
            "diet_treatment": {
                "unique_source_stories": len(keys),
                "head_stories": HEAD_N,
                "added_tail_stories": len(tail),
                "dev_reserve_stories": DEV_N,
                "prior_phase1_train_stream_equivalents": round(total_positions / p1_total, 6),
                "phase1g_train_stream_equivalents": round(total_positions / train_stream_tokens, 6),
                "head_prefix_token_count": p1_total,
            },
        },
        "schedule": {
            "path": str(schedule_path),
            "sha256": sha256(schedule_path),
            "format": "little-endian unsigned 32-bit starts, row-major [6000,64]",
            "shape": [MAX_UPDATES, EFFECTIVE_BATCH],
            "regeneration_during_training": False,
            "max_updates": MAX_UPDATES,
            "context": CONTEXT,
            "microbatch": 16,
            "gradient_accumulation": 4,
            "effective_batch": EFFECTIVE_BATCH,
            "supervised_positions_per_update": EFFECTIVE_BATCH * CONTEXT,
            "total_supervised_positions": total_positions,
            "train_stream_equivalents": round(total_positions / train_stream_tokens, 6),
        },
        "objective": {
            "type": "full_vocabulary_causal_next_token_cross_entropy",
            "reduction": "mean over each equal-sized microbatch, divided by accumulation steps before backward",
            "supervision": "all 256 next-token positions in every window",
            "binding_path_active": False,
        },
        "parameter_scope": {
            "train": "all base_model parameters: token/position embeddings, blocks 0-11, final norm, language head",
            "freeze": "orthogonal localizer and wq/wk/wv/wo binding projections",
            "trainable_parameters": 60536064,
            "frozen_parameters": 984321,
            "old_block_protection_mapping": "none; same as Phase 1 (from-scratch, no learned binding to protect)",
        },
        "optimizer": {
            "type": "AdamW",
            "lr_peak": 0.0003,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.05,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
            "gradient_clip_norm": 2.0,
        },
        "lr_schedule": {
            "type": "linear_warmup_then_cosine_decay",
            "warmup_updates": 200,
            "warmup_positions": 200 * EFFECTIVE_BATCH * CONTEXT,
            "peak_lr": 0.0003,
            "minimum_lr_ratio": 0.1,
            "minimum_lr": 0.00003,
            "formula": "u<=200: peak*u/200; otherwise peak*(0.1+0.9*0.5*(1+cos(pi*(u-200)/5800)))",
            "schedule_change_classification": "NONE - identical to Phase 1",
        },
        "evaluation": {
            "points": list(range(0, MAX_UPDATES + 1, 500)),
            "eval_microbatch": 16,
            "dev_windows": 20 * EFFECTIVE_BATCH,
            "train_windows": 5 * EFFECTIVE_BATCH,
            "selection_path": str(eval_selection_path),
            "selection_sha256": P1_EVAL_SELECTION_SHA256,
            "selection_format": "first 1280 dev starts, then 320 train starts; little-endian u32; byte-identical to Phase 1",
            "metrics": ["aligned causal CE", "perplexity", "train-dev CE gap", "fixed greedy diagnostics"],
            "binding": "NOT_APPLICABLE_UNTRAINED; mechanical interface only",
        },
        "gates": {
            "phase1_language_ready": {
                "best_dev_ce_max": 3.0,
                "final_dev_ce_max": 3.1,
                "two_consecutive_dev_evaluations_max": 3.1,
                "selected_checkpoint_train_dev_gap_max": 0.5,
                "all_values_finite": True,
                "completion_updates": MAX_UPDATES,
            },
            "phase1g_generalization_gates": {
                "g1_best_dev_ce_max": 1.320,
                "g2_final_dev_ce_max": 1.420,
                "g3_post_argmin_rise_max": 0.060,
                "g4_selected_gap_max": 0.5,
                "phase1_reference_best_dev_ce": 1.3750959888100625,
                "phase1_reference_final_dev_ce": 1.4820435523986817,
                "phase1_reference_post_argmin_rise": 0.10694756358861923,
            },
            "descriptive_oh_fuck_signal": "substantial monotonic DEV CE/PPL fall, two consecutive DEV CE <=3.1, no widening train-dev gap, finite gradients",
            "not_a_gate": "generation diagnostics and binding interface smoke",
        },
        "checkpointing": {
            "rolling_restart_interval": 100,
            "permanent_model_interval": 500,
            "best_definition": "lowest frozen DEV CE; exact ties select earlier update",
            "atomic": True,
            "resume": "restore model, optimizer, completed update, Python/torch/GPU RNG, frozen hashes and metrics; next update is completed_update+1",
            "uncommitted_replay": "updates after the latest 100-update rolling commit are not committed and are deterministically replayed",
        },
        "stopping": {
            "maximum_update": MAX_UPDATES,
            "dev_early_stopping": False,
            "hard_stop": [
                "NaN or Inf in inputs/logits/loss/gradients/gradient norm/model/optimizer",
                "integrity or resume-provenance mismatch",
                "deterministic algorithm failure",
                "GPU OOM",
                "checkpoint commit failure",
            ],
            "no_same_run_adaptation": True,
        },
        "locks": {
            "factual_supervision": False,
            "historical_transfer": "LOCKED_UNSCORED",
            "final": "LOCKED_UNACCESSED",
            "sacred": "LOCKED_UNACCESSED",
        },
    }
    config_path = ROOT / "PHASE1_LANGUAGE_CONFIG.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    provenance = {
        "architecture_chain": {"receipt_sha256": "776859cb1bf671634ae2d7e8dfab7f7cb8e517ce802d30d3cb3733b6b734b368", "manifest_sha256": "a4a9fb219f082e1936bf7341f42d82354fc2c7cf8d6f9b0d03c84e815117cad4"},
        "architecture_config_sha256": "8d7e1cbc604c9bf9f1942d4afc01817b20af17d8805a984f7ae57b1254f957b4",
        "tokenizer": {"path": str(TOKENIZER), "sha256": TOKENIZER_SHA256},
        "source": {"path": str(SOURCE), "sha256": SOURCE_SHA256},
        "phase1_reference": {
            "bundle": str(P1),
            "train_stream_sha256": P1_TRAIN_STREAM_SHA256,
            "dev_stream_sha256": P1_DEV_STREAM_SHA256,
            "eval_selection_sha256": P1_EVAL_SELECTION_SHA256,
            "initialized_checkpoint_sha256": P1_INIT_SHA256,
            "train_stream_tokens": p1_total,
            "dev_stream_tokens": int(torch.frombuffer(dev_stream_path.read_bytes(), dtype=torch.uint16).numel()),
            "best_dev_ce": 1.3750959888100625,
            "best_update": 3000,
            "final_dev_ce": 1.4820435523986817,
            "final_train_ce": 0.5667358934879303,
            "final_train_dev_gap": 0.9153076589107514,
        },
        "diet_selection": {
            "selection": "normalize line endings; sha256 each complete story; sort hashes",
            "unique_stories": len(keys),
            "head_stories": HEAD_N,
            "dev_reserve_stories": DEV_N,
            "added_tail_stories": len(tail),
            "train_stories_phase1g": len(train_rows),
            "head_is_exact_prefix": True,
            "dev_reserve_excluded_from_training": True,
            "train_schedule_seed": TRAIN_SCHEDULE_SEED,
        },
        "initialized_checkpoint": {"path": str(init_path), "sha256": P1_INIT_SHA256, "model_state_sha256": init_raw["model_state_sha256"]},
        "prior_pool_overlaps_tail": prior_overlaps,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "training_performed": False,
        "forbidden_material_accessed": False,
    }
    (ROOT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "PHASE1G_BUILD_ZERO_UPDATE_PASS",
        "unique_source_stories": len(keys),
        "head_story_tokens": p1_total,
        "train_documents": len(train_rows),
        "added_tail_documents": len(tail),
        "dev_reserve_documents": DEV_N,
        "train_stream_tokens": train_stream_tokens,
        "dev_stream_tokens": int(torch.frombuffer(dev_stream_path.read_bytes(), dtype=torch.uint16).numel()),
        "schedule_starts": len(train_starts),
        "total_supervised_positions": total_positions,
        "phase1_train_stream_equivalents": round(total_positions / p1_total, 6),
        "phase1g_train_stream_equivalents": round(total_positions / train_stream_tokens, 6),
        "head_prefix_byte_identical": True,
        "dev_reserve_excluded_from_training": True,
        "prior_pool_overlaps_tail": prior_overlaps,
        "initialized_checkpoint_sha256": P1_INIT_SHA256,
        "train_stream_sha256": sha256(train_stream_path),
        "schedule_sha256": sha256(schedule_path),
        "optimizer_created": False,
        "optimizer_updates": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
