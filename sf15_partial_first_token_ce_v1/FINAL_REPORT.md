# SF15 FINAL CLASSIFICATION

**PARTIAL_FIRST_TOKEN_CE_INSUFFICIENT**

# WHAT CHANGED

Only the first-answer-name causal CE weight changed from sealed SF14 weight 0.0 to 0.25 during English updates. Later response/EOS CE, first-token margin, broad KL, curriculum, binding, optimizer, scopes, and gates were unchanged.

# PER-SEED RESULTS

| Seed | Update | TRAIN16 c/e/r/f | Surface c/e | Order e (fact/copy) | Language CE/PPL | D3 | Binding |
|---:|---:|---:|---:|---:|---:|---:|---|
| 87038 | 0 | 16/16/8/4 | 12/5 | 3 (3/0) | 3.5873/36.14 | 0.00704 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87038 | 100 | 16/16/8/4 | 14/9 | 5 (4/1) | 3.6475/38.38 | 0.01133 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87038 | 200 | — | — | — | — | — | NOT REACHED: frozen stop at 100 |
| 87039 | 0 | 16/16/8/4 | 13/4 | 3 (3/0) | 3.5746/35.68 | 0.00667 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87039 | 100 | 16/16/8/4 | 14/10 | 5 (4/1) | 3.6511/38.52 | 0.01137 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87039 | 200 | — | — | — | — | — | NOT REACHED: frozen stop at 100 |
| 87040 | 0 | 16/16/8/4 | 13/5 | 3 (3/0) | 3.5950/36.41 | 0.00784 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87040 | 200 | 16/16/8/4 | 14/10 | 6 (4/2) | 3.6399/38.09 | 0.01455 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |

# U100 SIGNAL

- Seed 87038: D3 `0.01133`; TRAIN16 PASS; language PASS (`+0.0602` nat); binding PASS; surface exact `5→9`; order exact `3→5`.
- Seed 87039: D3 `0.01137`; TRAIN16 PASS; language PASS (`+0.0765` nat); binding PASS; surface exact `4→10`; order exact `3→5`.
- Seed 87040: D3 `0.01455`; TRAIN16 PASS; language PASS (`+0.0449` nat); binding PASS; surface exact `5→10`; order exact `3→6`.

# ENDPOINT GATES

- Complete endpoint passes: 0/3
- D3 failures: 3/3
- Retention failures: 0/3
- Widening movement: 3/3

# INTERPRETATION

The 0.25 first-answer-token CE treatment reproduced at least two branches of the preregistered D3-versus-exact-generation tradeoff. This identifies the observed tradeoff under SF15 without establishing CE as its sole cause.

# CHECKPOINTS

- Seed 87038: `C:\DaveLM-CADAVER\sf15_partial_first_token_ce_v1\runs\seed_87038_curriculum\checkpoint_100.pt` — `c327af65cf020954c91b30d012104a6d1f1de030c414fd9467f696dc9e257170`
- Seed 87039: `C:\DaveLM-CADAVER\sf15_partial_first_token_ce_v1\runs\seed_87039_curriculum\checkpoint_100.pt` — `351e6a0b4e499413058dbdadbf9fee70473bf17c9991b5b94d7fdd90af405970`
- Seed 87040: `C:\DaveLM-CADAVER\sf15_partial_first_token_ce_v1\runs\seed_87040_curriculum\checkpoint_200.pt` — `b81993a4e209b22bed85f2b3dc707193dfb28dd0e958d1204b1f57e46776b952`

# LOCKS

- Historical transfer panels: LOCKED/UNSCORED.
- FINAL and sacred material: not accessed.
