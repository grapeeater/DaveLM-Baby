# T15X T14-vs-T15 LATE-BASE DRIFT — FINAL

Status: **COMPLETE, read-only.** TEST not loaded. No training. Did not parent T14 or T15.

Question: after T15 held language and missed representation, did blocks 4–7 under-move vs T14, move similarly, or move in a different direction while 8–11 stayed at Phase1G?

## Freeze integrity
T15 U750 `max_abs` on blocks 8–11 is **0** for all three seeds. `block8to11_rms` is **0**. The T15 freeze held bit-identically.

## Drift magnitude vs Phase1G (RMS of Δ)

| run | blocks 4–7 | block 4 | blocks 8–11 | language_head |
|---|---:|---:|---:|---:|
| T14 730001 | 0.001065 | 0.001100 | 0.001176 | 0.003011 |
| T14 730002 | 0.001068 | 0.001107 | 0.001122 | 0.002617 |
| T14 730003 | 0.000959 | 0.001002 | 0.001003 | 0.002633 |
| T15 740001 | 0.001083 | 0.001127 | **0** | 0.002510 |
| T15 740002 | 0.001041 | 0.001092 | **0** | 0.002418 |
| T15 740003 | 0.001099 | 0.001130 | **0** | 0.002883 |

T15 4–7 (and block 4) moved **as much as** T14. This is not gradient starvation or an under-updated 4–7.

## Direction of Δ (cosine of (ckpt−Phase1G))

| pair | block 4 | blocks 4–7 | embed+head |
|---|---:|---:|---:|
| T15 740001 vs T14 730002 | 0.327 | 0.253 | 0.838 |
| T15 740002 vs T14 730002 | 0.343 | 0.268 | 0.856 |
| T15 740003 vs T14 730002 | 0.314 | 0.235 | 0.803 |
| T15 740001 vs T14 730001 | 0.329 | 0.258 | 0.819 |
| T15 740001 vs T15 740003 | 0.360 | 0.307 | 0.863 |
| T14 730001 vs T14 730002 | 0.343 | 0.293 | 0.853 |

4–7 update directions agree only moderately (~0.25–0.36). That is **the same range as T14 seed-vs-seed and T15 seed-vs-seed**. Embed/head updates agree strongly (~0.80–0.86) across T14 and T15.

## Verdict
T15 did not fail because 4–7 failed to move. Pointer grads already flow through frozen 8–11 (`forward_hidden` is after all 12 blocks). The missing ingredient is **8–11 weight motion during training** — a training-time scaffold — even though T14X showed those trained 8–11 weights are optional (even harmful) at the T14 endpoint.

Do not relaunch T15. Do not add LR to 4–7 on the T15 freeze. Do not parent T14/T15.

## Next
Prospectively frozen two-phase treatment from Phase1G: train 4–11 for 500 updates (T14 scope), then revert and freeze 8–11 at Phase1G and train 4–7 for 250 more. Same gates, new seeds, no T14 checkpoints as parents.
