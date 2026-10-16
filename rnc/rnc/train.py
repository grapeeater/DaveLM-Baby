"""The RNC trainer: one loop, one config, GPU only.

Features: fixed token budget, bf16 autocast with fp32 master weights (or fp32), gradient accumulation, AdamW with
decoupled weight decay on matrices only, warm-up + cosine schedule, gradient clipping, deterministic batch order,
step-0 baseline, periodic held-out evaluation, fixed-prompt sampling, periodic checkpoints, exact resume, clean
stop on Ctrl-C or a wall-clock limit, JSONL logging with throughput and GPU memory.
"""
from __future__ import annotations

import json
import math
import shutil
import signal
import sys
import time
from pathlib import Path

import torch

from . import gpu
from .checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from .config import RNCConfig, load_config
from .data import TokenStream
from .model import RNCModel
from .tokenizer import RNCTokenizer


def lr_at(step: int, cfg) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps))
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)


class JsonlLog:
    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "a", encoding="utf-8")

    def write(self, **row):
        row.setdefault("utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.f.write(json.dumps(row) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


@torch.no_grad()
def evaluate(model: RNCModel, val: TokenStream, cfg: RNCConfig, device, autocast_ctx) -> dict:
    model.eval()
    T = cfg.model.context_length
    total, n = 0.0, 0
    for i in range(cfg.train.eval_batches):
        x, y = val.fixed_batch(i, cfg.train.micro_batch_size, T)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_ctx():
            _, loss = model(x, y)
        total += float(loss.detach())
        n += 1
    model.train()
    mean = total / max(1, n)
    return {"val_loss": mean, "val_ppl": math.exp(min(mean, 50.0)), "val_tokens": n * cfg.train.micro_batch_size * T}


@torch.no_grad()
def sample(model: RNCModel, tok: RNCTokenizer, cfg: RNCConfig, device, step: int) -> list[dict]:
    out = []
    for i, prompt in enumerate(cfg.train.sample_prompts):
        g = torch.Generator(device=device).manual_seed(cfg.train.seed * 31 + i)  # same noise at every evaluation
        ids = torch.tensor([[tok.bos_id] + tok.encode(prompt)], device=device)
        gen = model.generate(ids, cfg.train.sample_max_new_tokens, temperature=cfg.train.sample_temperature,
                             top_k=cfg.train.sample_top_k, eos_id=tok.eos_id, generator=g)
        new_ids = gen[0, ids.shape[1]:].tolist()
        out.append({"step": step, "prompt": prompt, "completion": tok.decode(new_ids), "n_new_tokens": len(new_ids),
                    "hit_eos": tok.eos_id in new_ids})
    model.train()
    return out


def build_model(cfg: RNCConfig, device) -> RNCModel:
    torch.manual_seed(cfg.train.seed)
    model = RNCModel(cfg.model).to(device)
    model.set_attention_backend(cfg.train.attention_backend)
    return model


def train(config_path: Path, resume: bool = True, run_name: str | None = None, stop_after_step: int | None = None) -> dict:
    """``stop_after_step`` simulates an interruption (same code path as Ctrl-C) and exists for the resume tests."""
    cfg = load_config(config_path)
    if run_name:
        cfg.train.run_name = run_name
    tcfg = cfg.train
    device = gpu.require_gpu(tcfg.device)
    run_dir = cfg.resolve(tcfg.out_dir) / tcfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen_cfg = run_dir / "CONFIG.json"
    if frozen_cfg.exists():
        if json.loads(frozen_cfg.read_text(encoding="utf-8")) != cfg.to_dict():
            raise SystemExit(f"{frozen_cfg} differs from {config_path}; a run's config is frozen at its first launch.")
    else:
        frozen_cfg.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    log = JsonlLog(run_dir / "log.jsonl")
    events = JsonlLog(run_dir / "events.jsonl")
    samples_log = JsonlLog(run_dir / "samples.jsonl")

    tok = RNCTokenizer(cfg.resolve(cfg.tokenizer.path), cfg.tokenizer.expected_sha256)
    prepared = cfg.resolve(cfg.data.prepared_dir)
    meta = json.loads((prepared / "META.json").read_text(encoding="utf-8"))
    if meta["tokenizer_sha256"] != tok.sha256:
        raise SystemExit("prepared data tokenizer fingerprint does not match the configured tokenizer")
    train_stream = TokenStream(Path(meta["files"]["train"]["path"]))
    val_stream = TokenStream(Path(meta["files"]["val"]["path"]))
    T = cfg.model.context_length
    tokens_per_step = tcfg.micro_batch_size * tcfg.grad_accum_steps * T

    model = build_model(cfg, device)
    n_params = model.num_parameters()
    optimizer = torch.optim.AdamW(model.parameter_groups(tcfg.weight_decay), lr=tcfg.lr,
                                  betas=(tcfg.beta1, tcfg.beta2), eps=tcfg.eps, foreach=True)
    use_bf16 = tcfg.precision == "bf16"

    def autocast_ctx():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)

    step = 0
    tokens_seen = 0
    history = {"val": [], "resumes": []}
    ck = latest_checkpoint(run_dir) if resume else None
    if ck is not None:
        state = load_checkpoint(ck, map_location=device)
        if state["config"]["model"] != cfg.to_dict()["model"]:
            raise SystemExit("checkpoint model config differs from the run config")
        if state["extra"].get("tokenizer_sha256") != tok.sha256:
            raise SystemExit("checkpoint tokenizer fingerprint differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        tokens_seen = state["extra"]["tokens_seen"]
        history = state["extra"].get("history", history)
        torch.set_rng_state(state["rng"]["torch_cpu"].cpu())
        if state["rng"]["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["rng"]["torch_cuda"]])
        history["resumes"].append({"from_step": step, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        events.write(event="resume", step=step, checkpoint=str(ck))
        print(f"[rnc] resumed from {ck} at step {step}")

    info = {
        "run_name": tcfg.run_name, "config_path": str(config_path), "num_parameters": n_params,
        "tokens_per_step": tokens_per_step, "total_steps": tcfg.total_steps,
        "planned_tokens": tokens_per_step * tcfg.total_steps, "gpu": gpu.describe_gpu(), "precision": tcfg.precision,
        "attention_backend": tcfg.attention_backend, "tokenizer_sha256": tok.sha256,
        "data_meta": {k: meta[k] for k in ("corpus_sha256", "stats", "val_bytes_per_token", "train_bytes_per_token")},
        "start_step": step, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / f"RUN_INFO_{info['start_utc'].replace(':', '')}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[rnc] {tcfg.run_name}: {n_params:,} params, {tokens_per_step:,} tokens/step, {tcfg.total_steps} steps "
          f"({tokens_per_step * tcfg.total_steps / 1e6:.1f}M tokens), precision={tcfg.precision}, "
          f"attention={tcfg.attention_backend}, GPU={info['gpu'].get('name')}")

    stop_requested = {"flag": False, "reason": None}

    def on_sigint(signum, frame):
        stop_requested["flag"] = True
        stop_requested["reason"] = "SIGINT"
        print("\n[rnc] interrupt received; will checkpoint at the end of this step and exit cleanly")

    signal.signal(signal.SIGINT, on_sigint)
    val_bpt = meta["val_bytes_per_token"]

    def run_eval_and_samples(step: int, tag: str) -> dict:
        ev = evaluate(model, val_stream, cfg, device, autocast_ctx)
        ev["val_bits_per_byte"] = ev["val_loss"] / math.log(2) / val_bpt
        ev.update(step=step, tokens_seen=tokens_seen, tag=tag)
        history["val"].append({k: ev[k] for k in ("step", "tokens_seen", "val_loss", "val_ppl", "val_bits_per_byte")})
        log.write(kind="eval", **ev)
        for s in sample(model, tok, cfg, device, step):
            samples_log.write(**s)
        print(f"[rnc] eval step {step}: val_loss {ev['val_loss']:.4f} ppl {ev['val_ppl']:.1f} bpb {ev['val_bits_per_byte']:.3f}")
        return ev

    def do_checkpoint(step: int, reason: str) -> Path:
        extra = {"tokens_seen": tokens_seen, "tokenizer_sha256": tok.sha256, "history": history, "reason": reason,
                 "memory": gpu.memory_stats()}
        d = save_checkpoint(run_dir, step, model, optimizer, cfg.to_dict(), extra)
        events.write(event="checkpoint", step=step, path=str(d), reason=reason)
        print(f"[rnc] checkpoint saved: {d}")
        return d

    t_start = time.time()
    try:
        model.train()
        if step == 0 and not (run_dir / "ckpt_0000000").exists():
            events.write(event="baseline", step=0)
            run_eval_and_samples(0, "baseline")
            do_checkpoint(0, "baseline_untrained")
        window_tokens, window_time = 0, 0.0
        while step < tcfg.total_steps:
            t0 = time.time()
            lr = lr_at(step, tcfg)
            for g in optimizer.param_groups:
                g["lr"] = lr
            loss_acc = 0.0
            for micro in range(tcfg.grad_accum_steps):
                x, y = train_stream.random_batch(tcfg.micro_batch_size, T, tcfg.seed, step, micro)
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with autocast_ctx():
                    _, loss = model(x, y)
                (loss / tcfg.grad_accum_steps).backward()
                loss_acc += float(loss.detach()) / tcfg.grad_accum_steps
            if stop_after_step is not None and step + 1 >= stop_after_step:
                stop_requested["flag"] = True
                stop_requested["reason"] = f"stop_after_step={stop_after_step}"
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip))
            if not math.isfinite(loss_acc) or not math.isfinite(grad_norm):
                events.write(event="non_finite", step=step, loss=loss_acc, grad_norm=grad_norm)
                raise RuntimeError(f"non-finite loss/grad at step {step}: loss={loss_acc} grad_norm={grad_norm}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            step += 1
            tokens_seen += tokens_per_step
            dt = time.time() - t0
            window_tokens += tokens_per_step
            window_time += dt
            if step % tcfg.log_interval == 0 or step == 1:
                mem = gpu.memory_stats()
                row = dict(kind="train", step=step, loss=round(loss_acc, 5), lr=lr, grad_norm=round(grad_norm, 4),
                           tokens_seen=tokens_seen, step_seconds=round(dt, 3),
                           tokens_per_second=round(window_tokens / max(window_time, 1e-9)),
                           elapsed_seconds=round(time.time() - t_start, 1),
                           max_mem_allocated_gb=round(mem["max_allocated_bytes"] / 2**30, 3),
                           max_mem_reserved_gb=round(mem["max_reserved_bytes"] / 2**30, 3))
                log.write(**row)
                print(f"[rnc] step {step}/{tcfg.total_steps} loss {loss_acc:.4f} lr {lr:.2e} gnorm {grad_norm:.3f} "
                      f"{row['tokens_per_second']} tok/s mem {row['max_mem_allocated_gb']:.2f} GB")
                window_tokens, window_time = 0, 0.0
            final = step >= tcfg.total_steps
            if step % tcfg.eval_interval == 0 or final:
                run_eval_and_samples(step, "final" if final else "periodic")
            wall_exceeded = tcfg.max_wall_minutes > 0 and (time.time() - t_start) / 60 > tcfg.max_wall_minutes
            if wall_exceeded and not stop_requested["flag"]:
                stop_requested["flag"] = True
                stop_requested["reason"] = "max_wall_minutes"
            if step % tcfg.checkpoint_interval == 0 or final or stop_requested["flag"]:
                do_checkpoint(step, "final" if final else ("stop:" + str(stop_requested["reason"]) if stop_requested["flag"] else "periodic"))
            if stop_requested["flag"] and not final:
                events.write(event="clean_stop", step=step, reason=stop_requested["reason"])
                print(f"[rnc] stopped cleanly at step {step} ({stop_requested['reason']}); resume with the same command")
                break
        status = "completed" if step >= tcfg.total_steps else "stopped"
        events.write(event=status, step=step, tokens_seen=tokens_seen, elapsed_seconds=round(time.time() - t_start, 1),
                     memory=gpu.memory_stats())
        return {"status": status, "step": step, "tokens_seen": tokens_seen, "run_dir": str(run_dir), "history": history}
    finally:
        gpu.synchronize_quietly()
        log.close()
        events.close()
        samples_log.close()
