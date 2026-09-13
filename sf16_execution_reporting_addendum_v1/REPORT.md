# SF16 Execution Reporting Addendum

The sealed SF16 scientific artifacts remain unchanged. This additive note corrects two reporting/provenance presentation defects:

1. `RUN_LEDGER.json` remained stale after recording seed 87041 as RUNNING, although branch-local provenance, 200/200 metric streams, U0/U100/U200 evaluations, terminal statuses, and checkpoint hashes verify for all three branches.
2. The sealed `FINAL_REPORT.md` section titled `U100 SIGNAL` used each branch's terminal U200 row. The actual preserved U100 results are below.

| Seed | U100 D3 | Surface exact | Order exact | U200 D3 | Surface exact | Order exact |
|---:|---:|---:|---:|---:|---:|---:|
| 87041 | 0.0077040645 PASS | 8/16 | 4/16 | 0.0094439660 PASS | 11/16 | 6/16 |
| 87042 | 0.0077210751 PASS | 8/16 | 5/16 | 0.0111642672 FAIL | 10/16 | 6/16 |
| 87043 | 0.0074723289 PASS | 7/16 | 6/16 | 0.0130157946 FAIL | 10/16 | 7/16 |

TRAIN16, language, and both binding pools passed at every recorded endpoint. Frozen classification remains `FIRST_TOKEN_CE_INTERPOLATION_INSUFFICIENT`. No score, gate, checkpoint, or classification was changed.