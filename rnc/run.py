"""RNC command line. Run through rnc.ps1 (sets the ROCm venv) or directly with .venv\\Scripts\\python.exe -X utf8.

  run.py env                               GPU / environment report
  run.py count-docs      -c CONFIG         count corpus documents
  run.py train-tokenizer -c CONFIG         train the byte-level BPE on train-split documents
  run.py tokenizer-stats -c CONFIG         compression statistics on held-out documents
  run.py prepare         -c CONFIG         build train/val token streams
  run.py validate-data   -c CONFIG         integrity checks on the prepared streams
  run.py bench           -c CONFIG         measure step time / memory for the configured model (no checkpoint)
  run.py train           -c CONFIG [--fresh] [--run-name NAME]
  run.py generate        --checkpoint DIR  -p "prompt" [-p ...]
  run.py chat            --checkpoint DIR
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rnc import gpu  # noqa: E402

gpu.configure_environment()  # must precede any torch import

WORKSPACE = Path(__file__).resolve().parent
DEFAULT_CONFIG = WORKSPACE / "configs" / "rnc1_tinystories.json"


def cmd_env(args):
    import torch  # noqa: F401

    print(json.dumps(gpu.describe_gpu(), indent=2))


def _cfg(args):
    from rnc.config import load_config

    return load_config(args.config)


def cmd_count_docs(args):
    from rnc.data import count_documents

    cfg = _cfg(args)
    n = count_documents(cfg)
    print(json.dumps({"documents": n}))


def cmd_train_tokenizer(args):
    from rnc.data import count_documents, tokenizer_training_documents
    from rnc.tokenizer import train_tokenizer

    cfg = _cfg(args)
    t0 = time.time()
    n = count_documents(cfg)
    vocab = args.vocab_size or cfg.tokenizer.vocab_size
    out = Path(args.out) if args.out else cfg.resolve(cfg.tokenizer.path)
    docs = list(tokenizer_training_documents(cfg, n))
    note = {"corpus_path": str(cfg.resolve(cfg.data.corpus_path)), "corpus_sha256": cfg.data.corpus_sha256,
            "documents_total": n, "documents_used": len(docs), "split_seed": cfg.data.split_seed,
            "train_split_only": True, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    info = train_tokenizer(docs, vocab, out, note)
    info["seconds"] = round(time.time() - t0)
    print(json.dumps(info, indent=2, ensure_ascii=False))


def cmd_tokenizer_stats(args):
    from rnc.data import SplitAssigner
    from rnc.tokenizer import RNCTokenizer, corpus_statistics, iter_documents

    cfg = _cfg(args)
    tok = RNCTokenizer(Path(args.tokenizer) if args.tokenizer else cfg.resolve(cfg.tokenizer.path))
    assigner = SplitAssigner(cfg.data.split_seed, cfg.data.validation_fraction, cfg.data.max_validation_docs)
    held = []
    for i, d in enumerate(iter_documents(cfg.resolve(cfg.data.corpus_path), cfg.data.document_separator, cfg.data.min_doc_chars)):
        if assigner.is_validation(i):
            held.append(d)
            if len(held) >= args.n:
                break
    stats = corpus_statistics(tok, held)
    stats["tokenizer"] = str(tok.dir)
    stats["vocab_size"] = tok.vocab_size
    print(json.dumps(stats, indent=2))


def cmd_prepare(args):
    from rnc.data import prepare, validate_prepared
    from rnc.tokenizer import RNCTokenizer

    cfg = _cfg(args)
    tok = RNCTokenizer(cfg.resolve(cfg.tokenizer.path), cfg.tokenizer.expected_sha256)
    prepare(cfg, tok)
    print(json.dumps(validate_prepared(cfg, tok), indent=2))


def cmd_validate_data(args):
    from rnc.data import validate_prepared
    from rnc.tokenizer import RNCTokenizer

    cfg = _cfg(args)
    tok = RNCTokenizer(cfg.resolve(cfg.tokenizer.path), cfg.tokenizer.expected_sha256)
    print(json.dumps(validate_prepared(cfg, tok), indent=2))


def cmd_bench(args):
    import torch

    from rnc.model import RNCModel

    cfg = _cfg(args)
    device = gpu.require_gpu(cfg.train.device)
    try:
        results = []
        for precision in (args.precisions or [cfg.train.precision]):
            for backend in (args.backends or [cfg.train.attention_backend]):
                torch.manual_seed(0)
                model = RNCModel(cfg.model).to(device)
                model.set_attention_backend(backend)
                opt = torch.optim.AdamW(model.parameter_groups(0.1), lr=1e-4, foreach=True)
                B, T = cfg.train.micro_batch_size, cfg.model.context_length
                torch.cuda.reset_peak_memory_stats()
                times = []
                for it in range(args.iters + 2):
                    x = torch.randint(0, cfg.model.vocab_size, (B, T), device=device)
                    torch.cuda.synchronize()
                    t0 = time.time()
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(precision == "bf16")):
                        _, loss = model(x, x)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    if it >= 2:
                        times.append(time.time() - t0)
                mean = sum(times) / len(times)
                r = {"precision": precision, "attention_backend": backend, "micro_batch": B, "seq": T,
                     "seconds_per_micro_step": round(mean, 4), "tokens_per_second": round(B * T / mean),
                     "max_mem_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                     "max_mem_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3),
                     "num_parameters": model.num_parameters()}
                print(json.dumps(r))
                results.append(r)
                del model, opt
                torch.cuda.empty_cache()
        if args.out:
            Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    finally:
        gpu.synchronize_quietly()


def cmd_train(args):
    from rnc.train import train

    result = train(Path(args.config), resume=not args.fresh, run_name=args.run_name)
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


def cmd_generate(args):
    from rnc.generate import generate_cli

    generate_cli(Path(args.checkpoint), args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.seed)


def cmd_chat(args):
    from rnc.generate import chat_cli

    chat_cli(Path(args.checkpoint), args.max_new_tokens, args.temperature, args.top_k)


def main(argv=None):
    p = argparse.ArgumentParser(prog="rnc", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def with_config(sp):
        sp.add_argument("-c", "--config", default=str(DEFAULT_CONFIG))
        return sp

    sub.add_parser("env").set_defaults(fn=cmd_env)
    with_config(sub.add_parser("count-docs")).set_defaults(fn=cmd_count_docs)
    s = with_config(sub.add_parser("train-tokenizer"))
    s.add_argument("--vocab-size", type=int, default=0)
    s.add_argument("--out", default="")
    s.set_defaults(fn=cmd_train_tokenizer)
    s = with_config(sub.add_parser("tokenizer-stats"))
    s.add_argument("--tokenizer", default="")
    s.add_argument("-n", type=int, default=5000)
    s.set_defaults(fn=cmd_tokenizer_stats)
    with_config(sub.add_parser("prepare")).set_defaults(fn=cmd_prepare)
    with_config(sub.add_parser("validate-data")).set_defaults(fn=cmd_validate_data)
    s = with_config(sub.add_parser("bench"))
    s.add_argument("--iters", type=int, default=10)
    s.add_argument("--precisions", nargs="*")
    s.add_argument("--backends", nargs="*")
    s.add_argument("--out", default="")
    s.set_defaults(fn=cmd_bench)
    s = with_config(sub.add_parser("train"))
    s.add_argument("--fresh", action="store_true", help="ignore existing checkpoints of this run")
    s.add_argument("--run-name", default=None)
    s.set_defaults(fn=cmd_train)
    for name, fn in (("generate", cmd_generate), ("chat", cmd_chat)):
        s = sub.add_parser(name)
        s.add_argument("--checkpoint", required=True)
        s.add_argument("--max-new-tokens", type=int, default=150)
        s.add_argument("--temperature", type=float, default=0.8)
        s.add_argument("--top-k", type=int, default=50)
        if name == "generate":
            s.add_argument("-p", "--prompt", action="append", required=True)
            s.add_argument("--seed", type=int, default=None)
        s.set_defaults(fn=fn)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    main()
