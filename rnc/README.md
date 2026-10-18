# RNC — Racina Neural Core

DaveLM · model family **Mangaris** · codename **Baby** · architecture **RNC** (successor to MRCN-Alpha).

> FUNCTION FIRST. INTELLIGENCE NEXT. PERFECTION OVER TIME.

RNC is a from-scratch English language model and the straightforward training workflow around it. One config file,
one tokenizer, one trainer, one GPU. No cloud orchestration, no multi-agent university, no treatment gates.

## Current status

RNC-1 has **91,246,848 parameters**. Its first **3,000-step TinyStories run**
completed on Fan Diesel's AMD GPU. See [the saved first-run summary](reports/FIRST_RUN_SUMMARY.md)
for measured learning and limits.

Baby then completed seven 100-update curriculum sessions. Saved dev results
show progress on generated Chapter 1 object binding, Chapter 2 event memory,
Chapter 3 grounded answers, and Chapter 4 short dialogue. In the Session 007
in-session sample, Chapter 4 rose from 18/86 to 83/86 checked turns. Each reply
was evaluated with the reference earlier conversation turns supplied, and the
sample is not a full post-session dev evaluation. Session 006 also missed its
Chapter 1 binding forced-choice target (83.1% vs 85%) and over-caution limit
(3.59% vs 3%); Session 007 did not repeat those measures.

The Session 007 checkpoint is the frozen common ancestor for four independent
local starting copies. No specialist branch has trained. See [the session
report](../docs/RNC_CURRICULUM_PROGRESS.md), [checkpoint provenance](../docs/SESSION_007_COMMON_ANCESTOR.md),
and [the family tree](../docs/MANGARIS_FAMILY_TREE.md).

This is a lightweight publication of the existing laboratory. Checkpoints,
datasets and prepared streams stay local. Copied source, tests, configuration
and active tokenizer are unchanged; [PUBLICATION.md](docs/PUBLICATION.md)
explains the snapshot and records. Commands below describe the software;
this update launches no jobs.

## Layout

```
rnc/
  rnc.ps1                 launcher (ROCm venv, UTF-8); .\rnc.ps1 <command>
  run.py                  command line: env | count-docs | train-tokenizer | tokenizer-stats | prepare |
                          validate-data | bench | train | generate | chat
  configs/rnc1_tinystories.json   THE authoritative configuration of the first run
  rnc/                    the package
    config.py             dataclasses + strict JSON loading
    gpu.py                ROCm environment, GPU-first policy, memory reporting
    model.py              RNC-1: RMSNorm, RoPE, QK-norm causal attention, SwiGLU, tied readout
    tokenizer.py          byte-level BPE codec (the only path between text and ids)
    data.py               deterministic split, dedup, uint16 streams, seeded batches, fixed validation windows
    checkpoint.py         atomic checkpoints with SHA-256 sidecars
    train.py              the training loop (baseline, eval, samples, checkpoints, resume, clean stop)
    generate.py           continuation + REPL from a checkpoint
  tools/                  vocab_audit.py, historical_integrity.py
  tests/                  pytest health checks (GPU)
  docs/                   architecture, tokenizer and training notes
  data/raw/               corpus (TinyStories-train.txt, CDLA-Sharing-1.0) + provenance
  data/prepared/          token streams + META.json
  tokenizer/              trained tokenizers (+ candidates/)
  runs/<run_name>/        CONFIG.json (frozen), log.jsonl, events.jsonl, samples.jsonl, ckpt_*/
  reports/                phase reports and measurements
```

## Environment

Hardware: AMD Radeon RX 9060 XT 16 GB (gfx1200). Software: Python 3.12, ROCm SDK 7.2,
`torch 2.9.1+rocmsdk20260116`, `tokenizers 0.23.2`, `numpy`, `pytest`, all in `C:\DaveLM-Training\.venv`
(see the archived `ENV_SETUP.md` for the wheel URLs). Set `PYTHONUTF8=1`; the launcher does it.

`rnc/gpu.py` sets, before torch is imported: a writable `TEMP` (HIP fails without one) and
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (enables the fused attention kernels on this GPU, measured 10× faster
than the math kernel). GPU-first: every compute entry point calls `require_gpu()` and refuses CPU fallback.

## Workflow

```powershell
cd C:\DaveLM-Training\rnc
.\rnc.ps1 env                      # GPU report
.\rnc.ps1 train-tokenizer          # byte-level BPE on train-split documents -> tokenizer/rnc_bpe_8k
.\rnc.ps1 prepare                  # -> data/prepared/.../train.bin val.bin META.json (+ integrity checks)
.\rnc.ps1 bench                    # step time and memory for the configured model
..\.venv\Scripts\python.exe -m pytest tests -q      # health checks (several minutes, GPU)
.\rnc.ps1 train                    # runs/<run_name>/ ; Ctrl-C stops cleanly; rerun the same command to resume
.\rnc.ps1 generate --checkpoint runs\<run>\ckpt_0001000 -p "Once upon a time"
.\rnc.ps1 chat --checkpoint runs\<run>\ckpt_0001000
```

A run's configuration is frozen into `runs/<run_name>/CONFIG.json` at first launch; a changed config cannot drive an
existing run. Every checkpoint is kept. Step 0 (untrained) is always evaluated, sampled and checkpointed.

## Rules of this lineage

- Nothing is transplanted from MRCN-Alpha or any other model; RNC starts from random initialisation.
- The tokenizer is trained only on training-split documents and pinned by SHA-256 in the config, META.json and
  every checkpoint.
- Validation windows are fixed; the validation stream contains no document that also appears in training.
- The model has no hard-coded answers, routes or output filters; `tests/test_model.py` checks the source for that.
- Historical Baby artefacts (the MRCN-Alpha lab, the 61M museum graduate) are read-only to this workflow.
