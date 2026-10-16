# RNC-1 first completed run

Source: original run `rnc1_tinystories_shakedown_001`, completed
2026-10-16T09:16:47Z on Fan Diesel. This is a review of saved records; no new
training, evaluation or model execution was performed for publication.

| Item | Recorded result |
| --- | --- |
| Model | RNC-1; 91,246,848 parameters; random initialization |
| Hardware | AMD Radeon RX 9060 XT 16 GB (gfx1200), native Windows ROCm |
| Training | 3,000 optimizer steps; 98,304,000 training-input tokens |
| Context / vocabulary | 512 tokens / 8,192 IDs |
| Precision | bf16 autocast, fp32 master weights and AdamW state |
| Held-out evaluation | 409,600 fixed validation tokens at each recorded evaluation |
| Validation loss | 9.1523 at step 0 to 1.5221 nats/token at step 3,000 |
| Final perplexity / bits per byte | approximately 4.58 / 0.522 |
| Elapsed | 4,973.1 seconds (82.9 minutes) |
| Whole-run peak GPU allocation | 8,072,865,280 bytes (about 8.07 GB decimal) |
| Existing health-check record | 91 passed in 10.59 seconds |

Saved samples develop English sentence patterns, punctuation and short story
continuations. They still repeat, contradict themselves and drift from factual
questions into stories. These observations and held-out loss establish
language-model learning; they do not establish general conversational
competence, instruction following or reliable factual reasoning.

Final checkpoint SHA-256:
`f695c7d72a18424d4a0d5e2ed139955589e3f4786c64a090538c5ed2055483ae`.
Final checkpoint size: 1,095,093,003 bytes. Checkpoints remain local.

Evidence: [original machine-generated report](run_rnc1_tinystories_shakedown_001.md),
[original run records](../reproducibility/first_run/),
[saved health-check result](phase5_health/pytest_run4.txt),
[earlier backend parity record](phase5_health/final_ckpt_kernel_parity.json).

The future Adam & Eve common foundation has not been created or graduated.
No specialist descendant or additional RNC run is claimed by this update.
