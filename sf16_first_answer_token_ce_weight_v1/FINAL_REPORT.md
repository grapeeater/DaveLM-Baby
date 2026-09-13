# SF16 FINAL CLASSIFICATION

**FIRST_TOKEN_CE_INTERPOLATION_INSUFFICIENT**

# WHAT CHANGED

Only the first-answer-name causal CE weight changed from sealed SF15 weight 0.25 to 0.125 during English updates. Later response/EOS CE, first-token margin, broad KL, curriculum, binding, optimizer, scopes, and gates were unchanged.

# PER-SEED RESULTS

| Seed | Update | TRAIN16 c/e/r/f | Surface c/e | Order e (fact/copy) | Language CE/PPL | D3 | Binding |
|---:|---:|---:|---:|---:|---:|---:|---|
| 87041 | 0 | 16/16/8/4 | 12/5 | 3 (3/0) | 3.5873/36.14 | 0.00704 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87041 | 200 | 16/16/8/4 | 14/11 | 6 (5/1) | 3.6312/37.76 | 0.00944 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87042 | 0 | 16/16/8/4 | 13/4 | 3 (3/0) | 3.5746/35.68 | 0.00667 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87042 | 200 | 16/16/8/4 | 14/10 | 6 (4/2) | 3.6367/37.96 | 0.01116 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87043 | 0 | 16/16/8/4 | 13/5 | 3 (3/0) | 3.5950/36.41 | 0.00784 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87043 | 200 | 16/16/8/4 | 14/10 | 7 (5/2) | 3.6749/39.44 | 0.01302 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |

# U100 SIGNAL

- Seed 87041: D3 `0.00944`; TRAIN16 PASS; language PASS (`+0.0439` nat); binding PASS; surface exact `5→11`; order exact `3→6`.
- Seed 87042: D3 `0.01116`; TRAIN16 PASS; language PASS (`+0.0621` nat); binding PASS; surface exact `4→10`; order exact `3→6`.
- Seed 87043: D3 `0.01302`; TRAIN16 PASS; language PASS (`+0.0799` nat); binding PASS; surface exact `5→10`; order exact `3→7`.

# ENDPOINT GATES

- Complete endpoint passes: 0/3
- D3 failures: 2/3
- Retention failures: 0/3
- Widening movement: 3/3

# INTERPRETATION

The 0.125 first-answer-token CE treatment reproduced at least two branches of the preregistered D3-versus-exact-generation tradeoff. This identifies the observed tradeoff under SF16 without establishing CE as its sole cause. Per the frozen program stopping rule, SF16 is the final simple CE-weight interpolation; no automatic micro-dose sweep follows.

# CHECKPOINTS

- Seed 87041: `C:\DaveLM-CADAVER\sf16_first_answer_token_ce_weight_v1\runs\seed_87041_curriculum\checkpoint_200.pt` — `fd7c683ead1efa2af576f4c83548965142b5b8c9ded5bb30fe373a70bb2db715`
- Seed 87042: `C:\DaveLM-CADAVER\sf16_first_answer_token_ce_weight_v1\runs\seed_87042_curriculum\checkpoint_200.pt` — `93dc8b35c74470f5829a8359e92052fd049cf9e8ae6015168fc239660c5e60c7`
- Seed 87043: `C:\DaveLM-CADAVER\sf16_first_answer_token_ce_weight_v1\runs\seed_87043_curriculum\checkpoint_200.pt` — `ae7e274dd9501a07b05573d3a87a73483bd88c413216c517fa2450b594b1d878`

# LOCKS

- Historical transfer panels: LOCKED/UNSCORED.
- FINAL and sacred material: not accessed.
