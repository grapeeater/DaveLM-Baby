"""Summarise a run directory into Markdown: config, measurements over time, samples before/after, checkpoints.

Usage: python tools/report_run.py runs/<run_name> [--out reports/<file>.md]
Read-only; everything comes from log.jsonl, events.jsonl, samples.jsonl, CONFIG.json, RUN_INFO_*.json, ckpt_*/CHECKPOINT.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and math.isfinite(x) else str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    run = Path(a.run_dir)
    cfg = json.loads((run / "CONFIG.json").read_text(encoding="utf-8"))
    log = read_jsonl(run / "log.jsonl")
    events = read_jsonl(run / "events.jsonl")
    samples = read_jsonl(run / "samples.jsonl")
    infos = sorted(run.glob("RUN_INFO_*.json"))
    info = json.loads(infos[0].read_text(encoding="utf-8")) if infos else {}
    train_rows = [r for r in log if r["kind"] == "train"]
    evals = [r for r in log if r["kind"] == "eval"]
    ckpts = []
    for d in sorted(run.glob("ckpt_*")):
        m = d / "CHECKPOINT.json"
        if m.exists():
            ckpts.append(json.loads(m.read_text(encoding="utf-8")))
    out = []
    w = out.append
    w(f"# Run report: `{run.name}`\n")
    w(f"Config: {cfg['model']['n_layers']}L d{cfg['model']['d_model']} h{cfg['model']['n_heads']} mlp{cfg['model']['d_mlp']} "
      f"vocab {cfg['model']['vocab_size']} ctx {cfg['model']['context_length']}; precision {cfg['train']['precision']}, "
      f"attention {cfg['train']['attention_backend']}, lr {cfg['train']['lr']}, warmup {cfg['train']['warmup_steps']}, "
      f"total_steps {cfg['train']['total_steps']}, tokens/step {cfg['train']['micro_batch_size'] * cfg['train']['grad_accum_steps'] * cfg['model']['context_length']:,}.\n")
    if info:
        w(f"Parameters: {info.get('num_parameters', 0):,}. GPU: {info.get('gpu', {}).get('name')} ({info.get('gpu', {}).get('arch')}), "
          f"torch {info.get('gpu', {}).get('torch')}. Tokenizer sha256 `{info.get('tokenizer_sha256', '')[:16]}…`. "
          f"Corpus sha256 `{info.get('data_meta', {}).get('corpus_sha256', '')[:16]}…`.\n")
    status = [e for e in events if e["event"] in ("completed", "stopped")]
    if status:
        s = status[-1]
        w(f"Status: **{s['event']}** at step {s['step']}, {s['tokens_seen']:,} tokens seen, last segment elapsed {s.get('elapsed_seconds')} s.\n")
    resumes = [e for e in events if e["event"] == "resume"]
    if resumes:
        w(f"Resumes: {len(resumes)} (at steps {[e['step'] for e in resumes]}).\n")
    w("## Held-out evaluation over time\n")
    w("| step | tokens seen | val loss (nats) | val ppl | bits/byte |\n|---:|---:|---:|---:|---:|")
    for e in evals:
        w(f"| {e['step']} | {e['tokens_seen']:,} | {fmt(e['val_loss'])} | {fmt(e['val_ppl'], 1)} | {fmt(e['val_bits_per_byte'], 3)} |")
    w("")
    if train_rows:
        w("## Training loss (every logged step, thinned)\n")
        w("| step | loss | lr | grad norm | tok/s | peak mem GB |\n|---:|---:|---:|---:|---:|---:|")
        n = len(train_rows)
        stride = max(1, n // 25)
        for r in train_rows[::stride] + ([train_rows[-1]] if (n - 1) % stride else []):
            w(f"| {r['step']} | {fmt(r['loss'])} | {r['lr']:.2e} | {fmt(r['grad_norm'], 3)} | {r['tokens_per_second']} | {fmt(r['max_mem_allocated_gb'], 2)} |")
        w("")
        tps = [r["tokens_per_second"] for r in train_rows[1:]]
        secs = [r["step_seconds"] for r in train_rows[1:]]
        w("## Throughput and memory\n")
        w(f"- tokens/s: median {sorted(tps)[len(tps) // 2] if tps else 'n/a'}, min {min(tps) if tps else 'n/a'}, max {max(tps) if tps else 'n/a'}")
        w(f"- seconds/step: median {fmt(sorted(secs)[len(secs) // 2], 3) if secs else 'n/a'}")
        w(f"- peak GPU memory allocated: {max(r['max_mem_allocated_gb'] for r in train_rows):.2f} GB; reserved: {max(r['max_mem_reserved_gb'] for r in train_rows):.2f} GB")
        w(f"- wall time logged: {train_rows[-1]['elapsed_seconds'] / 60:.1f} min (this segment)")
        gn = [r["grad_norm"] for r in train_rows]
        w(f"- grad norm: first {fmt(gn[0], 3)}, median {fmt(sorted(gn)[len(gn) // 2], 3)}, max {fmt(max(gn), 3)}; steps with grad norm > clip ({cfg['train']['grad_clip']}): {sum(1 for g in gn if g > cfg['train']['grad_clip'])}/{len(gn)} logged")
        w("")
    if samples:
        steps = sorted({s["step"] for s in samples})
        w("## Samples at fixed prompts (temperature %.1f, top-k %d, fixed noise seed)\n" % (cfg["train"]["sample_temperature"], cfg["train"]["sample_top_k"]))
        for step in steps:
            w(f"### step {step}\n")
            for s in samples:
                if s["step"] == step:
                    comp = s["completion"].replace("\n", " ⏎ ")
                    w(f"- **{s['prompt']}**{comp}  _({s['n_new_tokens']} tokens{', ended with <|eos|>' if s['hit_eos'] else ''})_")
            w("")
    if ckpts:
        w("## Checkpoints\n")
        w("| step | path | bytes | sha256 |\n|---:|---|---:|---|")
        for c in ckpts:
            w(f"| {c['step']} | `{c['path']}` | {c['bytes']:,} | `{c['sha256']}` |")
        w("")
    text = "\n".join(out)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
