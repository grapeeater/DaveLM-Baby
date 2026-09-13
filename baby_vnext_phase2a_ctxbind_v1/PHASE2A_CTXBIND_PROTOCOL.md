# Baby vNext Phase 2A — Natural-Language Contextual Binding Pilot

Status: **prospectively frozen** (BABY_VNEXT_PHASE2A_CTXBIND_V1)

## Parent
Phase1G U6000 `best.pt`, SHA-256 `c5406f80…`, model-state digest `0101cec9…`.

## Treatment (single variable: training-data curriculum)
Stage curriculum over natural-language entity/property QA episodes (levels A–H) with
answer-token + EOS supervision, 9 QA : 1 language-rehearsal cadence over 500 updates,
AdamW lr 5e-5 constant, wd 0.05, clip 2.0, from the Phase1G language graduate.

## Scope
All 60,536,064 base parameters trainable; all 984,321 binding/localizer parameters frozen and
byte-identical to the parent throughout. Binding adapter is not exercised.

## Frozen evaluation
- Language: Phase1G DEV windows (1,280) + train-fit (320) + 10 generation prompts.
- QA-EVAL-DEV 640 core (80/level) + 120 shortcut-audit; QA-EVAL-TEST 320 core + 60 audit (once, terminal).
- Checkpoints U0/U100/U250/U500 with mechanical gates in `PHASE2A_CTXBIND_CONFIG.json`.

## Locked
Historical transfer LOCKED_UNSCORED; FINAL/sacred and synthetic-binding panels LOCKED_UNACCESSED.

Seeds: primary 610002; panel 61000201; train schedule 61000202; rehearsal 61000203.
Replication seeds 610003/610004 are not authorized.
