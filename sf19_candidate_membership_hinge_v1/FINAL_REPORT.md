# SF19 FINAL CLASSIFICATION

**CANDIDATE_MEMBERSHIP_OBJECTIVE_INSUFFICIENT**

# SOLE SCIENTIFIC CHANGE

Relative to sealed SF14, SF19 added a bounded relative hinge requiring the better of the two valid answer candidates to outrank every noncandidate vocabulary token by 1 nat. First-answer CE remained zero; later CE, pairwise margin, broad KL, data, optimizer, scope, schedule, gates, binding, and evaluation remained unchanged.

# PER-SEED TRAJECTORY

| Seed | U | TRAIN16 c/e/r/f | Surface c/e | Order c/e (fact/copy) | Language CE/PPL | D3 | Binding |
|---:|---:|---:|---:|---:|---:|---:|---|
| 87050 | 0 | 16/16/8/4 | 12/5 | 9/3 (3/0) | 3.5873/36.14 | 0.00704 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87050 | 100 | 16/16/8/4 | 14/11 | 8/5 (4/1) | 3.6505/38.49 | 0.00974 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87050 | 200 | 16/16/8/4 | 14/10 | 10/8 (4/4) | 3.6812/39.69 | 0.01183 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87051 | 0 | 16/16/8/4 | 13/4 | 9/3 (3/0) | 3.5746/35.68 | 0.00667 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87051 | 100 | 16/16/8/4 | 14/11 | 8/6 (4/2) | 3.6404/38.11 | 0.00979 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87051 | 200 | 16/16/8/4 | 14/10 | 10/8 (4/4) | 3.6650/39.05 | 0.01361 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87052 | 0 | 16/16/8/4 | 13/5 | 8/3 (3/0) | 3.5950/36.41 | 0.00784 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87052 | 100 | 16/16/8/4 | 14/11 | 10/5 (4/1) | 3.6111/37.01 | 0.01022 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87052 | 200 | — | — | — | — | — | NOT REACHED: frozen stop at U100 |

# FROZEN GATE RESULT

- Endpoint passes: 0/3
- D3 failures: 3/3
- Retention failures: 0/3
- Widening movement: 3/3

# INTERPRETATION

The bounded candidate-membership objective did not contain replicated D3 leakage while retention and widening remained active in at least two branches.

# CHECKPOINTS

- 87050: `C:\DaveLM-CADAVER\sf19_candidate_membership_hinge_v1\runs\seed_87050_curriculum\checkpoint_200.pt` — `a4c33ac929ded041c07e9178cfda4b9003c5d1aa5b4bc8fd91f996b4f61bba39`
- 87051: `C:\DaveLM-CADAVER\sf19_candidate_membership_hinge_v1\runs\seed_87051_curriculum\checkpoint_200.pt` — `c9058c0464da97ee923d8676deebb7010f511e8ee813d1606b0bd272f23bf020`
- 87052: `C:\DaveLM-CADAVER\sf19_candidate_membership_hinge_v1\runs\seed_87052_curriculum\checkpoint_100.pt` — `124eac55f520630303add83580a77c8041bdcc786dfaa8907758791ecb213d49`

# LOCKS

- Historical transfer panels: LOCKED/UNSCORED.
- FINAL and sacred material: not accessed.
