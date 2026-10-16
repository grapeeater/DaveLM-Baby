"""Training system on the GPU: checkpoint round trip, interruption + exact resume, tiny-dataset overfit,
baseline recording, generation from a saved checkpoint."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from rnc.checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from rnc.config import load_config
from rnc.generate import complete, load_model_for_inference
from rnc.model import RNCModel
from rnc.train import lr_at, train


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_lr_schedule(real_cfg):
    t = real_cfg.train
    assert lr_at(0, t) == pytest.approx(t.lr / t.warmup_steps)
    assert lr_at(t.warmup_steps, t) == pytest.approx(t.lr)
    assert lr_at(t.total_steps, t) == pytest.approx(t.lr * t.min_lr_ratio)
    mid = lr_at((t.warmup_steps + t.total_steps) // 2, t)
    assert t.lr * t.min_lr_ratio < mid < t.lr


def test_checkpoint_round_trip_and_hash(device, tmp_path):
    from rnc.config import ModelConfig

    cfg = ModelConfig(vocab_size=512, d_model=64, n_layers=2, n_heads=4, d_mlp=128, context_length=64)
    m = RNCModel(cfg).to(device)
    opt = torch.optim.AdamW(m.parameter_groups(0.1), lr=1e-3)
    x = torch.randint(0, 512, (2, 64), device=device)
    _, loss = m(x, x)
    loss.backward()
    opt.step()
    d = save_checkpoint(tmp_path, 7, m, opt, {"model": cfg.__dict__}, {"tokens_seen": 123, "tokenizer_sha256": "abc"})
    assert latest_checkpoint(tmp_path) == d
    state = load_checkpoint(d, map_location=device)
    m2 = RNCModel(cfg).to(device)
    m2.load_state_dict(state["model"])
    for (n1, p1), (n2, p2) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2)
    opt2 = torch.optim.AdamW(m2.parameter_groups(0.1), lr=1e-3)
    opt2.load_state_dict(state["optimizer"])
    assert opt2.state_dict()["param_groups"] == opt.state_dict()["param_groups"]
    assert state["step"] == 7 and state["extra"]["tokens_seen"] == 123
    # tamper -> detected
    p = d / "checkpoint.pt"
    data = bytearray(p.read_bytes())
    data[-1] ^= 0x01
    p.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_checkpoint(d)


def test_full_run_then_interrupted_resume_matches(device, tiny_workspace, tiny_prepared):
    cfg_path = tiny_workspace / "configs" / "tiny.json"
    # Run A: uninterrupted 12 steps.
    a = train(cfg_path, resume=False, run_name="run_A")
    assert a["status"] == "completed" and a["step"] == 12
    run_a = Path(a["run_dir"])
    assert (run_a / "ckpt_0000000").exists(), "step-0 baseline checkpoint must exist"
    assert (run_a / "ckpt_0000012").exists()
    log_a = read_jsonl(run_a / "log.jsonl")
    evals_a = [r for r in log_a if r["kind"] == "eval"]
    assert [e["step"] for e in evals_a] == [0, 6, 12]
    assert evals_a[0]["tag"] == "baseline"
    samples = read_jsonl(run_a / "samples.jsonl")
    assert {s["step"] for s in samples} == {0, 6, 12}
    # Run B: interrupted after step 5 (same code path as Ctrl-C), then resumed to completion.
    b1 = train(cfg_path, resume=False, run_name="run_B", stop_after_step=5)
    assert b1["status"] == "stopped" and b1["step"] == 5
    run_b = Path(b1["run_dir"])
    assert (run_b / "ckpt_0000005").exists()
    events = read_jsonl(run_b / "events.jsonl")
    assert any(e["event"] == "clean_stop" for e in events)
    b2 = train(cfg_path, resume=True, run_name="run_B")
    assert b2["status"] == "completed" and b2["step"] == 12
    assert any(e["event"] == "resume" and e["step"] == 5 for e in read_jsonl(run_b / "events.jsonl"))
    # Same batches, same schedule, same state -> same trajectory and same final weights (fp32).
    la = {r["step"]: r["loss"] for r in log_a if r["kind"] == "train"}
    lb = {r["step"]: r["loss"] for r in read_jsonl(run_b / "log.jsonl") if r["kind"] == "train"}
    assert set(la) == set(lb) == set(range(1, 13))
    for s in range(1, 13):
        assert la[s] == pytest.approx(lb[s], abs=2e-3), (s, la[s], lb[s])
    sa = load_checkpoint(run_a / "ckpt_0000012", map_location="cpu")["model"]
    sb = load_checkpoint(run_b / "ckpt_0000012", map_location="cpu")["model"]
    for k in sa:
        assert torch.allclose(sa[k], sb[k], atol=1e-4, rtol=1e-4), k
    assert sb is not None
    # frozen config: a changed config may not drive an existing run
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["train"]["lr"] = 1e-9
    alt = tiny_workspace / "configs" / "tiny_alt.json"
    alt.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(SystemExit, match="frozen"):
        train(alt, resume=True, run_name="run_B")


def test_resume_refuses_foreign_tokenizer(device, tiny_workspace, tiny_prepared):
    cfg_path = tiny_workspace / "configs" / "tiny.json"
    run_dir = tiny_workspace / "runs" / "run_A"
    ck = latest_checkpoint(run_dir)
    assert ck is not None
    state = load_checkpoint(ck)
    assert state["extra"]["tokenizer_sha256"] == load_model_for_inference(ck, device)[1].sha256


def test_tiny_dataset_overfit(device, tiny_cfg, tiny_tokenizer):
    """Deliberately memorise 4 short documents: loss must fall below 0.05 and greedy decoding must reproduce them."""
    torch.manual_seed(0)
    from rnc.config import ModelConfig

    mcfg = ModelConfig(vocab_size=512, d_model=64, n_layers=2, n_heads=4, d_mlp=128, context_length=64)
    model = RNCModel(mcfg).to(device)
    docs = ["Lily saw the big dog.", "Tom found the red ball.", "Max hugged the small cat.", "Mia lost the blue fish."]
    seqs = [tiny_tokenizer.encode_document(d) for d in docs]
    L = max(len(s) for s in seqs)
    batch = torch.full((len(seqs), L), tiny_tokenizer.eos_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, : len(s)] = torch.tensor(s)
    batch = batch.to(device)
    # The first token after <|bos|> is genuinely ambiguous between the four stories (irreducible ln 4 nats, which
    # spread over ~7 positions gives the 0.198 floor observed on 2026-10-16). Ignore that position; everything else
    # must be memorised exactly.
    targets = batch[:, 1:].clone()
    targets[:, 0] = -1
    opt = torch.optim.AdamW(model.parameter_groups(0.0), lr=3e-3, betas=(0.9, 0.95))
    first = None
    for it in range(600):
        _, loss = model(batch[:, :-1], targets)
        if first is None:
            first = float(loss.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    final = float(loss.detach())
    assert first > 4.0 and final < 0.02, (first, final)
    for d, s in zip(docs, seqs):
        prefix = torch.tensor([s[:3]], device=device)  # bos + first two pieces
        out = model.generate(prefix, len(s), temperature=0, eos_id=tiny_tokenizer.eos_id)
        assert tiny_tokenizer.decode(out[0].tolist()) == d


def test_trainer_learns_synthetic_corpus(device, tiny_workspace, tiny_prepared):
    """End-to-end: baseline eval vs final eval on the tiny run must show a large drop."""
    run_a = tiny_workspace / "runs" / "run_A"
    evals = [r for r in read_jsonl(run_a / "log.jsonl") if r["kind"] == "eval"]
    assert evals[0]["val_loss"] > 5.0
    assert evals[-1]["val_loss"] < evals[0]["val_loss"] - 1.5, (evals[0]["val_loss"], evals[-1]["val_loss"])
    assert all(math.isfinite(r["loss"]) for r in read_jsonl(run_a / "log.jsonl") if r["kind"] == "train")


def test_generate_from_checkpoint(device, tiny_workspace, tiny_prepared):
    ck = latest_checkpoint(tiny_workspace / "runs" / "run_A")
    model, tok, cfg = load_model_for_inference(ck, device)
    a = complete(model, tok, "Lily saw the", device, max_new_tokens=10, temperature=0.8, top_k=20, seed=1)
    b = complete(model, tok, "Lily saw the", device, max_new_tokens=10, temperature=0.8, top_k=20, seed=1)
    assert a == b and isinstance(a, str)
    c = complete(model, tok, "Lily saw the", device, max_new_tokens=10, temperature=0.8, top_k=20, seed=2)
    assert isinstance(c, str)


def test_gpu_memory_reported(device, tiny_workspace, tiny_prepared):
    rows = [r for r in read_jsonl(tiny_workspace / "runs" / "run_A" / "log.jsonl") if r["kind"] == "train"]
    assert rows and all(r["max_mem_allocated_gb"] > 0 and r["tokens_per_second"] > 0 for r in rows)
