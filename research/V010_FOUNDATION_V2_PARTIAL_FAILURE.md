# Baby v0.10 foundation v2 partial failure

Protocol: `BABY_V010_FOUNDATION_V2`  
Seed: `102001`  
Status: stopped at update 1,701 for diagnosis; not a terminal gate result

The v2 panel and data audits passed, and the fresh run initialized without a
parent checkpoint. During the primitive-induction stage, however, the
structured loss fell from about 9.1 to 2.2-2.6 while primitive target rank
remained around 242-257 at update 1,500 and primitive exact reproduction was
0. The full keyed panel was also 0 exact. Language DEV CE was 2.866 at update
1,500.

The cause is an objective shortcut: primitive items target
`[random_identity, deterministic_separator, EOS]`, but the v2 trainer masks
all three positions equally. Two of the three losses can therefore be reduced
without solving identity transport. This is directly supported by the
combination of low structured loss, target probability near 0.002, target rank
near chance, and low-diversity greedy emissions. The run was stopped before
the later curriculum stages rather than spending the remaining budget on the
known shortcut. All checkpoints and metrics remain preserved in
`runs/foundation_v2_seed102001/`.

Correction for v2R1: primitive and short stages will mask only the sampled
answer span for the capability loss. Delimiter/EOS training is deferred to the
full stage, where it cannot dominate the primitive identity objective. This is
a change to the training objective, not to any success threshold or protected
evaluation panel. v2R1 will start from fresh random initialization.
