# Baby vNext Phase 1G — Generalization Diet Treatment

Status: **prospectively frozen; training not yet performed at seal time**

## Scientific purpose

Phase 1 (BABY_VNEXT_PHASE1_LANGUAGE_V1, 61,520,385-parameter Baby vNext) finished 6,000
from-scratch updates and reached the frozen `PHASE1_LANGUAGE_READY` gate, with its best
frozen DEV CE at U3000 = 1.3751. After U3000 the DEV CE rose monotonically to 1.4820 at
U6000 while train CE kept falling to 0.5667, widening the train/DEV gap to +0.915. That is
consistent with memorization of a small, exhaustively-reused training stream (9,000 unique
stories / 3,576,861 tokens replayed 27.48x over 98,304,000 supervised positions).

Phase 1G tests ONE hypothesis: the ~61.5M model can reach materially better held-out
generalization at identical compute if the training diet carries more useful
same-distribution diversity and far less repetition. Architecture, tokenizer, optimizer,
LR schedule, seed policy, update budget, checkpoint cadence, evaluation windows, and all
frozen gates are unchanged from Phase 1.

## Single treatment (the only experimental variable)

The Phase 1 diet was the first 9,000 unique stories of `TinyStories-valid.txt` (21,990
unique stories total) ordered by sha256. The Phase-1 DEV reserve is the next 1,000 unique
stories and is LOCKED. Phase 1G trains on ALL remaining unique stories from the SAME file:
the original 9,000 (head) plus the 11,990 previously-unused stories (tail) = **20,990
training documents**, encoded with the same tokenizer and stream layout. The frozen
Phase-1 train stream is an **exact byte prefix** of the Phase-1G train stream (asserted at
build). Training windows are re-materialized over the longer stream with a new frozen
schedule seed (61000111).

Resulting scale at the identical 6,000-update / 98,304,000-position budget:

| Quantity | Phase 1 | Phase 1G |
|---|---:|---:|
| Unique train stories | 9,000 | 20,990 |
| Train stream tokens | 3,576,861 | ~8,354,565 |
| Train stream equivalents | 27.48 | ~11.77 |
| DEV reserve (locked) | 1,000 | 1,000 (identical) |

No new generator, corpus family, objective, regularization, near-duplicate filter,
curriculum, or architecture change is introduced. The added tail stories are the same
distribution as the head, from the same hash-verified source file.

## Frozen state (identical to Phase 1)

- Architecture bundle `baby_vnext_60m_design_v1`; 60,536,064 trainable base params;
  984,321 frozen binding/localizer params (binding `grad=None`).
- Tokenizer `davelm_tokenizer.json` v0_7, vocab 1024; `<doc>/<bos>/<eos>` stream encoding;
  no padding; 256-token windows.
- Full-vocabulary causal next-token CE; microbatch 16 x accumulation 4 = effective 64.
- AdamW(lr=3e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=0.05, amsgrad=False,
  foreach=False, fused=False); global grad clip 2.0.
- Warmup updates 1-200 (0 -> 3e-4); cosine decay updates 201-6000 (3e-4 -> 3e-5).
- Primary run seed 610001 (init / dropout / runtime), reusing the SAME sealed zero-update
  initialized checkpoint bytes (`039e8efe...`) so the Phase-1G U0 baseline must reproduce
  Phase-1's frozen U0 (DEV 7.086638 / train 7.087802).
- 6,000 updates; eval at U0 then every 500; rolling restart every 100; permanent
  checkpoints every 500; best = lowest frozen DEV CE, ties select earlier update.
- DEV stream (390,340 tokens), 1,280 DEV eval windows, 320 train-fit eval windows (drawn
  from the head prefix), and the 10 greedy generation prompts are byte-identical to
  Phase 1 and are NEVER trained on (DEV reserve excluded; asserted disjoint).
- No same-run adaptation; hard stops are NaN/Inf, deterministic-operation failure, GPU
  OOM, integrity/restart mismatch, or checkpoint-commit failure.

## Changes vs Phase 1 (complete list)

| Field | Phase 1 | Phase 1G |
|---|---|---|
| train documents | 9,000 | 20,990 |
| train stream (+hash) | 3,576,861 tokens | ~8,354,565 tokens (new file) |
| training schedule (+hash) | seed 61000101 | seed 61000111 (new file) |
| stream equivalents | 27.4833 | ~11.77 |
| study/config identity | V1 | V1G (directory + config provenance) |

The runner, seal, and verify scripts are byte-identical to Phase 1. Identity is carried by
the directory name and the config/manifest hashes.

## Prospective gates (frozen before the run)

Layer 1 — Phase-1 readiness gates preserved unchanged (computed by the frozen runner):
6,000 updates complete; best DEV CE <= 3.0; final DEV CE <= 3.1; two consecutive DEV CE
<= 3.1; selected-checkpoint train/DEV gap <= 0.5; all values finite.

Layer 2 — Phase-1G generalization gates (Phase-1 reference: best DEV CE 1.3751 @ U3000,
final 1.4820, post-argmin rise +0.1070):

- G1: best frozen DEV CE <= 1.320 (>= 0.055 nat better than Phase 1).
- G2: final U6000 DEV CE <= 1.420 (>= 0.062 nat better than Phase 1).
- G3: DEV CE(U6000) - DEV CE(best) <= +0.060 (memorization phase materially muted).
- G4: selected-checkpoint train/DEV gap <= 0.5.

Success = Layer 1 AND G1-G4.

## Regression gates

- R1: parameter scope 60,536,064 / 984,321 intact; zero binding gradients; binding_status
  NOT_APPLICABLE_UNTRAINED.
- R2: provenance/locks preserved; build-time exact token-array disjointness of the new
  train set vs the DEV reserve and all prior pool files (audited; see PROVENANCE.json).
- R3: greedy generation diagnostics at the selected checkpoint do not regress below
  Phase-1's selected checkpoint (10/10 non-immediate-EOS, 10/10 no-three-identical-run).
- R4: U0 baseline reproduction to 1e-6; receipt/manifest/SHA integrity; PYTHONHASHSEED
  enforcement; restart provenance checks active.
- R5: no mid-run adaptation; failures are reported and classified, never tuned around.

## Authorized launch interface (frozen)

```powershell
$env:PYTHONHASHSEED='610001'
& 'C:\DaveLM-CADAVER\sf2_runtime\sf2venv\Scripts\python.exe' -B 'C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1\PHASE1_LANGUAGE_RUNNER.py' --mode train --bundle 'C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1' --output-dir 'C:\DaveLM-CADAVER\baby_vnext_phase1g_language_v1_run_seed610001'
```

This package does not authorize replication seeds 610002/610003 or any follow-up run.
