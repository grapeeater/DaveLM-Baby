# Executable seed-87002 protocol

Exactly 500 updates are read literally from MATERIALIZED_SCHEDULE_SEED87002.json: 50 cycles of nine English and one binding update. Each English batch has two complete object and two complete predicate families (32 records); both arms use the same identities and shared padding. Binding batches contain eight frozen quartets / 32 documents.

Each arm is a fresh process with PYTHONHASHSEED=87002, Python/random/NumPy/torch CPU/CUDA seeds 87002, deterministic algorithms enabled, float32 only, no TF32, no DataLoader and no workers. AdamW remains one continuous optimizer per arm: lr 5e-5, betas (.9,.999), eps 1e-8, wd .05, amsgrad/foreach/fused false; no scheduler.

English labels supervise candidate response tokens plus final EOS only. Binding uses the pinned Pilot1 `rowpos`, `loc_loss`, `LAM`, and causal answer-CE implementation. At every scope change zero grads to None, set scope, verify inactive gradients None, backward, clip only gradient-bearing parameters at 2.0, then step.

DEV is factual DEV only at 0,100,500. Primary can begin only after both committed final arm hashes; Confirmation is not touched. Restart state is atomic and carries model, optimizer, completed update, scope, all RNG states, and protocol/schedule/parent hashes.
