# HR-1 prospective treatment

HR-1 starts from immutable Language Pilot 1 (`treatment13_orthogonal_shared_unbounded_seed8380/checkpoints/orthogonal_shared_unbounded/seed_8380/latest.pt`, SHA-256 recorded in `HR1_PROTOCOL.json`) and uses seed 87004.

The treatment is 500 optimizer updates in 50 cycles of nine English updates followed by one T13 binding-rehearsal update. English data are deterministic sentence segments from the frozen Pilot 1 TinyStories training corpus, tokenized with the pinned 1,024-token tokenizer. Segments contain 1–96 content tokens and are materialized in `ENGLISH_TRAIN.jsonl`; development segments are materialized separately in `ENGLISH_DEV.jsonl`. English batches are the literal 450 batches in `ENGLISH_SCHEDULE.json`, 64 records each, with BOS once, sentence tokens, final EOS, and loss labels only on response tokens plus final EOS; padding and context tokens are ignored.

Binding uses the established 320-document Pilot 1 rehearsal pool (80 quartets), with the literal 50 batches in `BINDING_SCHEDULE.json` (8 quartets/32 documents), answer causal cross-entropy plus the pinned hard-min permutation-invariant localization objective, weight 1.0536573711078283. The full established T13 trainable scope is restored for binding updates.

For English updates, embeddings, positional embeddings, blocks 4–7, final normalization, and language head train; blocks 0–3 and specialized localization/retrieval parameters are frozen. AdamW is fresh per future arm, continuous across updates, with lr 5e-5, betas (0.9,0.999), eps 1e-8, weight decay 0.05, no scheduler, and global gradient clipping 2.0. Deterministic CUDA execution uses seed 87004, no autocast/TF32, and deterministic algorithms. Development evaluation is permitted only at updates 0, 100, and 500 on the frozen DEV diagnostics; the sealed final readiness battery remains inaccessible until the protocol's preregistered development criteria are met. Binding retention is evaluated on both nonsacred pools separately.

No checkpoint is loaded or modified by this freeze build. The executable implementation must be reviewed before any training command is run.
