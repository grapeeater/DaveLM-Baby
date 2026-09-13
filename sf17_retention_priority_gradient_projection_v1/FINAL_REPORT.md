# SF17 FINAL CLASSIFICATION

**RETENTION_PRIORITY_PROJECTION_INSUFFICIENT**

# SOLE SCIENTIFIC CHANGE

Relative to sealed SF13 full-CE training, SF17 preserved the KL gradient and projected away only the globally opposing component of factual CE+margin gradients on English updates. All loss values/weights, data, optimizer, scope, schedule, gates, binding, and evaluation remained unchanged.

# PER-SEED TRAJECTORY

| Seed | U | TRAIN16 c/e/r/f | Surface c/e | Order c/e (fact/copy) | Language CE/PPL | D3 | Binding |
|---:|---:|---:|---:|---:|---:|---:|---|
| 87044 | 0 | 16/16/8/4 | 12/5 | 9/3 (3/0) | 3.5873/36.14 | 0.00704 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87044 | 100 | 16/16/8/4 | 14/11 | 9/5 (4/1) | 3.6207/37.36 | 0.01541 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87044 | 200 | — | — | — | — | — | NOT REACHED: frozen stop at U100 |
| 87045 | 0 | 16/16/8/4 | 13/4 | 9/3 (3/0) | 3.5746/35.68 | 0.00667 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87045 | 100 | 16/16/8/4 | 14/11 | 8/4 (3/1) | 3.6266/37.59 | 0.01508 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87045 | 200 | — | — | — | — | — | NOT REACHED: frozen stop at U100 |
| 87046 | 0 | 16/16/8/4 | 13/5 | 8/3 (3/0) | 3.5950/36.41 | 0.00784 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87046 | 100 | 16/16/8/4 | 14/11 | 10/5 (4/1) | 3.6580/38.78 | 0.01309 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87046 | 200 | — | — | — | — | — | NOT REACHED: frozen stop at U100 |

# PROJECTION TELEMETRY

- 87044: conflicts 69/90 (76.7%); mean cosine `-0.10264`; mean coefficient `0.14908`.
- 87045: conflicts 62/90 (68.9%); mean cosine `-0.08558`; mean coefficient `0.13677`.
- 87046: conflicts 72/90 (80.0%); mean cosine `-0.10135`; mean coefficient `0.1388`.

# FROZEN GATE RESULT

- Endpoint passes: 0/3
- D3 failures: 3/3
- Retention failures: 0/3
- Widening movement: 3/3

# INTERPRETATION

The tested retention-priority projection did not contain replicated D3 leakage while retention and widening remained active in at least two branches.

# CHECKPOINTS

- 87044: `C:\DaveLM-CADAVER\sf17_retention_priority_gradient_projection_v1\runs\seed_87044_curriculum\checkpoint_100.pt` — `c57ec0d48cec0df5a73b679783905c656ef117c293e5315a91510178616c3bc2`
- 87045: `C:\DaveLM-CADAVER\sf17_retention_priority_gradient_projection_v1\runs\seed_87045_curriculum\checkpoint_100.pt` — `683dbbdd3749ce1b048fcf9e98d96da48c4358a9ded2bf7584b939b55e50c9af`
- 87046: `C:\DaveLM-CADAVER\sf17_retention_priority_gradient_projection_v1\runs\seed_87046_curriculum\checkpoint_100.pt` — `584995430d51bc715a3aa710ed30fb63b5817f4850bc75bb8298265b03460db6`

# LOCKS

- Historical transfer panels: LOCKED/UNSCORED.
- FINAL and sacred material: not accessed.
