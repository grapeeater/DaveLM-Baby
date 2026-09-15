# Baby v0.10 foundation v2R2 partial failure

Protocol: `BABY_V010_FOUNDATION_V2R2`  
Seed: `104001`  
Status: stopped at update 6,651 after the first scored capability transition

The fresh language foundation completed 6,000 updates without numerical
failure and reached language DEV CE 1.1919 at the transition. This confirms
that v0.10 can reproduce the needed language substrate from a new
initialization; no v0.9 checkpoint was imported.

After 500 identity-only primitive updates, the interim probe showed a real but
insufficient identity signal: primitive induction first-token top-1 0.0625,
median rank 152, and primitive keyed first-token top-1 0.0625. However language
DEV CE simultaneously degraded to 5.7095. The protocol had omitted language
retention batches during primitive/short stages and updated all blocks at the
structured learning rate. This traded away the language substrate while the
copy capability was still immature.

This is a curriculum/optimization failure, not a final capability-gate pass.
The run is preserved in `runs/diagnostic_v2r2_seed104001_8000/` and no
checkpoint is eligible as a parent. The next preregistered correction freezes
blocks 0-6 after language acquisition, trains only upper blocks/final norm/head
as in the successful T34 scope, and includes the 20% language-retention mix in
every capability stage.
