# SF14 FINAL CLASSIFICATION

**RETENTION_REGRESSION**

# WHAT CHANGED

Only first-answer-name causal CE was masked during English updates. Later response/EOS CE, first-token margin, broad KL, curriculum, optimizer, scopes, and gates were unchanged.

# CORRECTED PER-SEED TRAJECTORY

| Seed | Update | TRAIN16 c/e/r/f | Surface c/e | Order e (fact/copy) | Language CE/PPL | D3 | Binding |
|---:|---:|---:|---:|---:|---:|---:|---|
| 87035 | 0 | 16/16/8/4 | 12/5 | 3 (3/0) | 3.5873/36.14 | 0.00704 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87035 | 100 | 16/16/8/4 | 14/2 | 2 (2/0) | 3.5917/36.30 | 0.00255 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87035 | 200 | 16/14/8/4 | 14/1 | 1 (1/0) | 3.6757/39.48 | 0.00312 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87036 | 0 | 16/16/8/4 | 13/4 | 3 (3/0) | 3.5746/35.68 | 0.00667 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87036 | 100 | 16/16/8/4 | 14/2 | 2 (2/0) | 3.6101/36.97 | 0.00263 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87036 | 200 | 16/14/8/4 | 14/2 | 1 (1/0) | 3.6854/39.86 | 0.00314 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87037 | 0 | 16/16/8/4 | 13/5 | 3 (3/0) | 3.5950/36.41 | 0.00784 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87037 | 100 | 16/16/8/4 | 13/2 | 2 (2/0) | 3.5737/35.65 | 0.00253 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87037 | 200 | 16/14/8/4 | 14/2 | 2 (2/0) | 3.6690/39.21 | 0.00276 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |

# U100 OH-FUCK SIGNAL

- Seed 87035: D3 PASS `0.00255`; TRAIN16 PASS `16/16/8/4`; language PASS (`+0.0044` nat); binding PASS/PASS; surface exact `5→2`; order exact `3→2`.
- Seed 87036: D3 PASS `0.00263`; TRAIN16 PASS `16/16/8/4`; language PASS (`+0.0355` nat); binding PASS/PASS; surface exact `4→2`; order exact `3→2`.
- Seed 87037: D3 PASS `0.00253`; TRAIN16 PASS `16/16/8/4`; language PASS (`-0.0212` nat); binding PASS/PASS; surface exact `5→2`; order exact `3→2`.

# U200 STOP

- Seed 87035: TRAIN16 `16 correct / 14 exact / 8 reversals / 4 families` — retention gate FAIL; D3 remained PASS at `0.00312`; language and both binding pools remained PASS.
- Seed 87036: TRAIN16 `16 correct / 14 exact / 8 reversals / 4 families` — retention gate FAIL; D3 remained PASS at `0.00314`; language and both binding pools remained PASS.
- Seed 87037: TRAIN16 `16 correct / 14 exact / 8 reversals / 4 families` — retention gate FAIL; D3 remained PASS at `0.00276`; language and both binding pools remained PASS.

# INTERPRETATION

Removing first-answer-name causal CE strongly contained D3 in all three branches, but it did not preserve exact TRAIN16 answer+EOS behavior through U200 and did not retain widening exact generation. The frozen safety-priority classification is RETENTION_REGRESSION. The result does not establish that CE is the root cause, that CE is generally incompatible, or that the unchanged margin is sufficient.

# CHECKPOINTS

- Seed 87035: `C:\DaveLM-CADAVER\sf14_first_answer_token_ce_ablation_v1\runs\seed_87035_curriculum\checkpoint_200.pt` — `485358a5a84ea200fd591709f14ac066a922de30ff31c1028b7d928be6d5812c`
- Seed 87036: `C:\DaveLM-CADAVER\sf14_first_answer_token_ce_ablation_v1\runs\seed_87036_curriculum\checkpoint_200.pt` — `c20b06ae9cbffc1ea053a22563c397792302a14c670c2b5c26a07ecb0bfac8d1`
- Seed 87037: `C:\DaveLM-CADAVER\sf14_first_answer_token_ce_ablation_v1\runs\seed_87037_curriculum\checkpoint_200.pt` — `cfa3c313dc6b77caf9c34e6893d5836b3c0defa0a81fdf5772457d6507966328`

# LOCKS

- Historical transfer panels: LOCKED/UNSCORED.
- FINAL and sacred material: not accessed.
