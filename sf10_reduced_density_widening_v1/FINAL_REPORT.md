# SF10 — reduced-density curriculum widening

**DENSITY_HYPOTHESIS_WEAKENED**.

| Seed | Parent (SF8) | Endpoint | Train16 c/x/rev/fam | DevSurface c/x | DevOrder x (fact/copy) | CE | D3 | Status |
|---|---|---:|---|---|---|---|---:|---:|---|
| 87023 | 9e293af6d16c… | 100 | 16/16/8/4 | 13/9 | 5 (3/2) | 3.6584 | 0.018059 | STOP_REGRESSION |
| 87024 | 839f7f5a60b3… | 100 | 16/16/8/4 | 11/9 | 8 (3/5) | 3.6508 | 0.018641 | STOP_REGRESSION |
| 87025 | eb6a725173ff… | 100 | 16/16/8/4 | 12/8 | 8 (4/4) | 3.7013 | 0.020457 | STOP_REGRESSION |

## Per-seed endpoint gates

- seed_87023_curriculum @100: `{"binding": {"pilot0": true, "pilot1": true}, "continue": false, "d3": false, "dev_order": false, "dev_surface": false, "endpoint_pass": false, "language": true, "retention_acquisition": true}`
- seed_87024_curriculum @100: `{"binding": {"pilot0": true, "pilot1": true}, "continue": false, "d3": false, "dev_order": false, "dev_surface": false, "endpoint_pass": false, "language": true, "retention_acquisition": true}`
- seed_87025_curriculum @100: `{"binding": {"pilot0": true, "pilot1": true}, "continue": false, "d3": false, "dev_order": false, "dev_surface": false, "endpoint_pass": false, "language": true, "retention_acquisition": true}`

Frozen SF1 transfer panels remained LOCKED. SF9 remains sealed and unmodified.
