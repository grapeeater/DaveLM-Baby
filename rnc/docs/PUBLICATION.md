# RNC publication snapshot — 2026-10-16

The authoritative active laboratory remains `C:\DaveLM-Training\rnc`.
This publication copies selected existing source, tests, config, active tokenizer
and small records. `reproducibility/PUBLISHED_SOURCE_SHA256.json` lists original
relative paths, publication paths, byte counts and SHA-256 values. Copied files
are byte-identical; the publication README and new summaries are documentation
prepared for this repository. No laboratory source, tokenizer, configuration,
run log, dataset or checkpoint was edited by this publication.

## Reading the evidence

- `reports/FIRST_RUN_SUMMARY.md` states the current result and its limits.
- The machine-generated run report and the copied `log.jsonl`, `events.jsonl`
  and `samples.jsonl` are existing records, not a newly run evaluation.
- `reports/phase5_health/pytest_run4.txt` is the previously completed 91-test
  result. The tests were reviewed but not rerun during publication because they
  include training/evaluation and GPU work.
- `reports/phase5_health/final_ckpt_kernel_parity.json` preserves the earlier
  final-checkpoint attention-backend comparison; it was not recomputed here.
- Architecture decision notes preserve the original design rationale. Statements
  attributing stability to a particular component are hypotheses unless a
  controlled comparison is reported. Older historical summaries are not a
  replacement for the dated S1/S2/S3 records.

The active tokenizer has **8,192** IDs. Its SHA-256 is
`d37360ce034918fcc2dc9be5fee17903244001eb307a1154ac662a5ee10ff6d3`.
The original 16K and 24K candidates remain local; small comparison measurements
are published. Candidate compression measurements are not model-quality tests.

## Reproducibility boundaries

Run configuration, tokenizer and data hashes, original metrics, fixed-prompt
samples, environment record and all seven checkpoint sidecars are published in
`reproducibility/`. Absolute Windows paths in those original records describe
Fan Diesel; the files have not been silently rewritten for another machine.
Checkpoint sidecars identify local weights; the checkpoint bytes are excluded.

TinyStories provenance identifies dataset revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, CDLA-Sharing-1.0, and corpus SHA-256
`c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f`.
See `reproducibility/data/`. Source text and prepared streams are excluded.
The saved `META.json` contains split settings and stream hashes. These records
support reproducing the historical run; publication itself authorizes no run.

Recorded runtime versions: Python 3.12, torch `2.9.1+rocmsdk20260116`,
tokenizers `0.23.2`, numpy `2.5.3`, pytest `9.1.1`. ROCm PyTorch requires the AMD
installation described by the existing environment notes; a generic CPU/CUDA
wheel is not a replacement. The copied launcher expects the ROCm environment at
`../.venv/Scripts/python.exe` relative to this RNC directory. No environment or
packages were installed for publication.

Generation currently has no KV cache; context is 512 tokens. Packing has no
document attention mask. Fused AMD attention is experimental. Exact duplicate
documents were removed; near-duplicate separation was not established.
The run demonstrates TinyStories language-model learning, not general
conversation, instruction following or robust factual reasoning.

## Documentation update — 2026-10-17

The v1.0 branch now includes a concise report of RNC curriculum Sessions 001–007 and the verified Session 007 common-ancestor provenance. These additions are documentation only. The completed Session 007 checkpoint, the ancestor and specialist copies, curriculum lessons, prepared datasets, raw evaluation examples, and private experiments remain outside Git.

The Session 007 Chapter 4 score is from 86 checked turns in the trainer’s in-session dev panel, with reference prior turns supplied at each evaluation turn. It is not a full post-run dev evaluation or an end-to-end conversation test. The two Session 006 retention misses are recorded, and Session 007 did not remeasure them. See [the curriculum progress report](../../docs/RNC_CURRICULUM_PROGRESS.md).
