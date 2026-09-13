# SF13 FINAL CLASSIFICATION

**RETENTION_POSITION_COVERAGE_INSUFFICIENT**

# WHAT CHANGED

The sole scientific variable was the frozen broad-coverage KL retention-position sampling geometry. The KL pool, 160-position budget, loss, curriculum, optimizer, scopes, and gates were unchanged.

# PER-SEED RESULTS

| Seed | Update | TRAIN16 c/e/r/f | Surface c/e | Order e (fact/copy) | Language CE / PPL | D3 | Binding pools |
|---:|---:|---:|---:|---:|---:|---:|---|
| 87032 | 0 | 16/16/8/4 | 12/5 | 3 (3/0) | 3.5873 / 36.14 | 0.00704 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87032 | 100 | 16/16/8/4 | 14/11 | 5 (4/1) | 3.6258 / 37.55 | 0.01389 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87032 | 200 | — | — | — | — | — | NOT REACHED: frozen D3 stop at 100 |
| 87033 | 0 | 16/16/8/4 | 13/4 | 3 (3/0) | 3.5746 / 35.68 | 0.00667 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87033 | 100 | 16/16/8/4 | 14/11 | 4 (4/0) | 3.6221 / 37.42 | 0.01603 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87033 | 200 | — | — | — | — | — | NOT REACHED: frozen D3 stop at 100 |
| 87034 | 0 | 16/16/8/4 | 13/5 | 3 (3/0) | 3.5950 / 36.41 | 0.00784 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87034 | 100 | 16/16/8/4 | 14/11 | 5 (5/0) | 3.6264 / 37.58 | 0.01295 | pilot0:80/80/c0/q20/r40; pilot1:80/80/c0/q20/r40 |
| 87034 | 200 | — | — | — | — | — | NOT REACHED: frozen D3 stop at 100 |

# U100 SIGNAL

| Seed | D3 ≤.010 | TRAIN16 | Language ≤U0+.25 | Binding | Surface movement | Order movement | Stop |
|---:|---|---|---|---|---:|---:|---|
| 87032 | **FAIL** (0.01389) | PASS | PASS (+0.0385) | PASS/PASS | +6 exact | +2 exact | STOP_REGRESSION |
| 87033 | **FAIL** (0.01603) | PASS | PASS (+0.0475) | PASS/PASS | +7 exact | +1 exact | STOP_REGRESSION |
| 87034 | **FAIL** (0.01295) | PASS | PASS (+0.0314) | PASS/PASS | +6 exact | +2 exact | STOP_REGRESSION |

# ENDPOINT GATES

- TRAIN16: PASS 3/3.
- Language: PASS 3/3.
- Pilot0 binding: PASS 3/3.
- Pilot1 binding: PASS 3/3.
- D3: FAIL 3/3.
- DEV_SURFACE endpoint: FAIL 3/3.
- DEV_ORDER endpoint: FAIL 3/3.
- Complete endpoint: 0/3.

# INTERPRETATION

Broadening retention-position coverage within the same frozen KL pool did not keep D3 below 0.010. All three Babies nevertheless retained TRAIN16, language, and binding, while surface and order/source exactness moved upward. Under the frozen classification, broader coverage was insufficient to contain the replicated D3 regression. This weakens retention-position coverage as the primary explanation under this tested geometry; it does not establish a deeper mechanism.

# LOCKS

- Historical transfer panels: LOCKED/UNSCORED.
- FINAL: not accessed.
- Sacred material: not accessed.

# CHECKPOINTS

- Seed 87032: `C:\DaveLM-CADAVER\sf13_broad_coverage_kl_retention_v1\runs\seed_87032_curriculum\checkpoint_100.pt` — `a56c03af6c6e4600c0cd15797001f68592107a823746819bb5b97f52f85b3a81`
- Seed 87033: `C:\DaveLM-CADAVER\sf13_broad_coverage_kl_retention_v1\runs\seed_87033_curriculum\checkpoint_100.pt` — `d07febc0c85912e83a02b995be504c4dd60509da62de3bf6094a3f65773172ec`
- Seed 87034: `C:\DaveLM-CADAVER\sf13_broad_coverage_kl_retention_v1\runs\seed_87034_curriculum\checkpoint_100.pt` — `9bd38e0274d8038e9c1180d8057212fcdc898e1219342b80da20e679aea7db32`

# PROVENANCE

- Sealed SF13 receipt: `b581fa3d852efb4436495d852a8f2ab883fd1048ece0af944a856aa676348c63`
- Sealed SF13 manifest: `f9a6fbbc08a92f80876bb5666116d274023834909a9cfcbfca78b6dfb7575e71`
- Controller: `e360848014f7cd05a8900f3fcf42f469acbf4928ba44ef7c2cb0d500863dfa05`
- Frozen KL schedule: `aaeb59b44c8683c0564fe487f631158360169a522b9bcd98e98d9cdb342018eb`

# EXACT NEXT ACTION

Conduct the single preregistered widening-supervision redesign review specified by the frozen SF13 protocol. Do not execute a new treatment from this report.
