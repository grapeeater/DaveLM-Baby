# Overnight Reproducibility Resolution — STATUS (crash/restart recovery state)

Updated: continuously. This file always states: current path, actions completed, evidence, pass/fail status,
whether any scientific artifact was modified, and the exact next allowed action.

## Current path: PATH 1 (bitwise determinism) — investigation active

### Frozen facts (verified at start)
- SF2 = `SF2_ACQUISITION_FAIL` (15/16, 15/16, 7/8, 3/4; binding PASS; language/D3 retention PASS). u100 ckpt
  `ec5ce896...`, u200 ckpt `1710a183...`.
- SF3 = `SF3_HARD_STOP_UPDATE100_REPLAY_MISMATCH` (zero annealed updates).
- SF4 = `SF4_HARD_STOP_COMMON_STATE_REPRODUCTION_MISMATCH` (zero control/treatment updates).
- All gates/classifications preserved. No scientific artifact modified by this mission.

### Prior evidence used
- My earlier autopsy (sf2_update100_replay_mismatch_autopsy_v1): Replay-A == Replay-B == SF3-replay bitwise at
  u100 (02:2x window); all differ from historical SF2 (334/414, max 9.05e-4); first divergence update 2; u0
  artifacts byte-identical.
- SF4 (03:02-03:05): common_a vs common_b DIVERGED (334/414, max 9.12e-4; optimizer 668/1002; first logged
  difference update 12, kl delta 2.98e-8; 89/100 metric records differ). common_a/common_b ran sequentially.
- Interpretation: run-to-run bitwise determinism on this stack is INTERMITTENT (matches within one time window,
  fails in another), not a persistent environment change.

### Actions completed
- Full inventory + read of SF3/SF4 packages, frozen engine/sources/hash verification.
- Confirmed no concurrent GPU users (no LM Studio/python GPU processes; only AMD system services).
- Mission dir created. [this file]

### Evidence files
- (pending probe results in path1_evidence/)

### Next allowed action
- Run bounded sequential probe pairs of the frozen SF2 engine (30-40 updates each) to (a) catch a diverging
  pair in the current window, (b) test candidate runtime levers for bitwise determinism, or (c) establish that
  divergence cannot be reproduced/eliminated → then classify PATH 1 FAIL and move to PATH 2.
- DO NOT modify any engine/protocol/gate. DO NOT run a treatment.
