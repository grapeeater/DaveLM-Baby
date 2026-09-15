# v0.10 foundation protocol freeze receipt

- Protocol: `BABY_V010_FOUNDATION_V1`
- Freeze date: 2026-09-15
- Parent checkpoint: none; fresh random initialization
- Frozen inputs: `data_specs/V010_FOUNDATION_FREEZE.json`
- Structured panels: `data/generated/foundation_v1/panels.json`
- Leakage audit: `data/generated/foundation_v1/AUDIT_RECHECK.json` (`PASS`)
- Protected material opened: no

The 50-update diagnostic before this freeze was explicitly a non-graduating
engineering preflight. It did not alter any protocol threshold or evaluation
panel and is preserved under `runs/diagnostic_seed101001_50_retry1/`. The first
full foundation run must use the frozen inputs and the two preregistered seeds
`101001` and `101002`.
