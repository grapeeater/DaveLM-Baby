# RNC-1 training system and the first experiment's budget

## Measured on Fan Diesel (RX 9060 XT, 2026-10-16, `reports/phase5_health/bench_rnc1.json`)

RNC-1, 91,246,848 parameters, micro-batch 16 × 512 tokens, AdamW step included:

| Precision | Attention kernel | s / micro-step | tokens / s | peak allocated | peak reserved |
|---|---|---:|---:|---:|---:|
| **bf16 autocast** | **fused (auto)** | **0.395** | **20,755** | **7.18 GB** | 7.39 GB |
| bf16 autocast | math | 0.814 | 10,061 | 9.58 GB | 9.81 GB |
| fp32 | fused (auto) | 3.670 | 2,232 | 9.20 GB | 9.51 GB |
| fp32 | math | 4.400 | 1,862 | 11.72 GB | 12.16 GB |

Decision: bf16 autocast (fp32 master weights and optimizer state) with the fused SDPA kernels. It is 9× faster than
fp32 and leaves more than half the card free. The fused kernels agreed with the fp32 reference within one bf16 ulp
in the probe and within 5 % relative in `tests/test_model.py::test_attention_backends_agree`; the math kernel is one
config switch away if any instability appears.

## Fixed budget for the first shakedown (decided before launch)

| Item | Value |
|---|---|
| Run name | `rnc1_tinystories_shakedown_001` |
| Tokens per optimizer step | 16 × 4 × 512 = 32,768 |
| Optimizer steps | 3,000 |
| Planned tokens | 98,304,000 (26 % of the 375.5M-token deduplicated training stream; < 1 epoch) |
| Schedule | 150 warm-up steps, cosine to 6e-5 (10 % of 6e-4) at step 3,000 |
| Evaluation | every 250 steps on 50 fixed validation batches (409,600 held-out tokens); step 0 included |
| Samples | 4 fixed prompts, seeded, every 250 steps; step 0 included |
| Checkpoints | step 0, every 500 steps, and 3,000 (full optimizer state, ~1.1 GB each) |
| Expected wall time | ≈ 1.6 s/step → ≈ 80 min plus evaluation (~25 min total) ≈ 1.75 h |
| Hard ceiling | `max_wall_minutes` 180: a clean stop with checkpoint if the run is far slower than measured |
| Stop rule | the step count; the run is not extended or shortened on the basis of results |

Why 3,000 steps: ~100M tokens is enough for a ~90M model to move well past unigram statistics on this corpus and
show whether grammar and local story structure are being learned, while staying under two hours on one consumer
GPU. It is not enough to claim competence of any kind, and no such claim will be made.

## Data

Corpus: TinyStories (Eldan & Li 2023, CDLA-Sharing-1.0), `TinyStories-train.txt`, SHA-256
`c5cf5e22…401f`, 2,119,489 documents after separator splitting. Preparation (`data/prepared/tinystories_8k/META.json`):

| | documents | tokens | bytes/token |
|---|---:|---:|---:|
| train | 1,781,998 | 375,543,695 | 4.33 |
| validation | 17,250 | 3,615,438 | 4.33 |
| exact duplicates dropped | 320,241 | | |

The validation split is a seeded random 1 % of documents (capped at 20,000, hit 17,250 after dedup), assigned before
tokenisation; 320,241 byte-identical duplicate documents were removed so no validation story has a copy in
training. Streams are uint16, hashed, and validated for id range and `<|bos|>`/`<|eos|>` structure.

## What the trainer records

`runs/<run>/log.jsonl` (train rows: loss, lr, grad norm, tokens seen, step seconds, tokens/s, peak GPU memory;
eval rows: validation loss, perplexity, bits per byte), `events.jsonl` (baseline, checkpoints, resume, stop,
non-finite), `samples.jsonl` (prompt, completion, step), `CONFIG.json` (frozen), `RUN_INFO_*.json` (GPU, parameter
count, data hashes), `ckpt_*/CHECKPOINT.json` (SHA-256 of every checkpoint).
