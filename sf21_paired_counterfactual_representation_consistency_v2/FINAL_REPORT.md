# SF21 FINAL CLASSIFICATION

**RETENTION_REGRESSION_10M_SERIES_CLOSED**

# WHAT CHANGED

SF21 added only `0.1 × mean(1 − cosine)` across the 16 frozen SF20 assignment-reversal pairs, using the final-normalized last context/query state that predicts the first answer token. All SF20 data, objectives, schedule, parents, optimizer, scopes, gates, and evaluators remained frozen.

# PER-SEED RESULTS

| Seed | Stop | TRAIN16 c/e/r/f | Surface c/e | Order e (fact/copy) | Language Δ | D3 U0→final | Binding | Mean cosine | Checkpoint SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 87056 | 100 | 16/13/8/4 | 11/2 | 2 (2/0) | +0.0912 | 0.00704→0.00219 | pilot0:80/80/c0; pilot1:80/80/c0 | 0.7127 | `db5037def67f32f819420381e24e0d6095699343db74a262d467b2252d0bd0a6` |
| 87057 | 100 | 16/13/8/4 | 10/2 | 2 (2/0) | +0.1530 | 0.00667→0.00238 | pilot0:80/80/c0; pilot1:80/80/c0 | 0.7012 | `5467687bbd3282fb4fb4cc1b31a8463b6b591c379593adebd4cddec18b8a488f` |
| 87058 | 200 | 16/8/8/4 | 13/1 | 2 (2/0) | +0.2256 | 0.00784→0.00243 | pilot0:80/80/c0; pilot1:80/80/c0 | 0.7301 | `69d6ae55ae96367a3fbe59193ce1b647b333ec91ef07df7910c9295030bbd4ef` |

# FROZEN CROSS-SEED RESULT

- Endpoint passes: 0/3
- Retention failures: 3/3
- Branches reaching U200: 1/3

# INTERPRETATION

SF21 did not move the coexistence frontier sufficiently. Under the prospectively declared stopping rule, the 10.6M factual-supervision treatment series is closed.

The consistency telemetry shows whether the auxiliary term engaged; it is not a behavioral gate and cannot establish relational generalization by itself. Historical transfer panels remained LOCKED/UNSCORED. FINAL and sacred material were not accessed.
