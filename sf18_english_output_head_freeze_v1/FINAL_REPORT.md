# SF18 FINAL CLASSIFICATION

**OUTPUT_HEAD_CONSTRAINT_INSUFFICIENT**

# SOLE SCIENTIFIC CHANGE

Relative to sealed SF13 full-CE training, SF18 froze only the untied language-head weight and bias during English updates. Binding updates restored the unchanged full T13 scope. All objectives, weights, data, optimizer, schedule, gates, binding, and evaluation remained unchanged.

# PER-SEED TRAJECTORY

| Seed | U | TRAIN16 c/e/r/f | Surface c/e | Order c/e (fact/copy) | Language CE/PPL | D3 | Binding |
|---:|---:|---:|---:|---:|---:|---:|---|
| 87047 | 0 | 16/16/8/4 | 12/5 | 9/3 (3/0) | 3.5873/36.14 | 0.00704 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87047 | 100 | 16/16/8/4 | 14/11 | 9/5 (4/1) | 3.6583/38.80 | 0.01666 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87047 | 200 | — | — | — | — | — | NOT REACHED: frozen stop at U100 |
| 87048 | 0 | 16/16/8/4 | 13/4 | 9/3 (3/0) | 3.5746/35.68 | 0.00667 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87048 | 100 | 16/16/8/4 | 14/11 | 9/5 (4/1) | 3.6261/37.57 | 0.01432 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87048 | 200 | — | — | — | — | — | NOT REACHED: frozen stop at U100 |
| 87049 | 0 | 16/16/8/4 | 13/5 | 8/3 (3/0) | 3.5950/36.41 | 0.00784 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87049 | 100 | 16/16/8/4 | 14/11 | 9/6 (4/2) | 3.6298/37.71 | 0.01401 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87049 | 200 | — | — | — | — | — | NOT REACHED: frozen stop at U100 |

# FROZEN GATE RESULT

- Endpoint passes: 0/3
- D3 failures: 3/3
- Retention failures: 0/3
- Widening movement: 3/3

# INTERPRETATION

English-update head freezing did not contain replicated D3 leakage while retention and widening remained active in at least two branches.

# CHECKPOINTS

- 87047: `C:\DaveLM-CADAVER\sf18_english_output_head_freeze_v1\runs\seed_87047_curriculum\checkpoint_100.pt` — `02f9c13596eaf48b1b22e840bb9a298bda9214af176bd1226ecc42d6cd09d8f1`
- 87048: `C:\DaveLM-CADAVER\sf18_english_output_head_freeze_v1\runs\seed_87048_curriculum\checkpoint_100.pt` — `3a7ea1405fbcbde72582436de79d60ce5ee13eb35b9536f85a040e62c69ca418`
- 87049: `C:\DaveLM-CADAVER\sf18_english_output_head_freeze_v1\runs\seed_87049_curriculum\checkpoint_100.pt` — `2c1032a26f8480d8f5afeef83dbf41d21b076a4c5522dd4349af68a612f75324`

# LOCKS

- Historical transfer panels: LOCKED/UNSCORED.
- FINAL and sacred material: not accessed.
