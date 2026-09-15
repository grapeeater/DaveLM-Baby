# Baby v0.10 curriculum v2R1

Protocol: `BABY_V010_FOUNDATION_V2R1`  
Status: frozen before the v2R1 diagnostic run

v2 exposed a deterministic-suffix shortcut: primitive rows contained one
sampled identity token followed by separator and EOS tokens, while the trainer
masked all target positions. v2R1 retains the staged scaffold and final
evaluation panels but changes the capability loss mask:

- primitive induction and primitive/short keyed retrieval train only the
  sampled `target_span` positions;
- the full foundation stage trains the answer span plus separator/EOS so
  free-running termination is learned after identity transport exists;
- language CE remains 20% after the 500-update warmup;
- all final gates, held-out panels, thresholds, and fresh-seed requirements
  remain unchanged.

This correction is preregistered to remove a measured shortcut, not to make a
failed metric easier. The first diagnostic is 2,000 updates on fresh seed
103001, ending after the primitive stage; its result determines whether the
full replicated v2R1 run is scientifically justified.
