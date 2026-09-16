# v0.10 iteration log

## Foundation v1 preflight / partial run

- Seed: 101001; fresh random initialization; no parent checkpoint.
- Started from the frozen v1 code and ran to U1826 before an intentional stop.
- Language DEV CE improved from 7.10 at U0 to 2.78 at U1750.
- Structured loss became finite and generally fell from 10.04 at U501 to approximately 4.0--4.8.
- The interim report did not provide a valid same-surface novel-copy control: its `novel` panel was accidentally generated with held-out markers.
- Therefore its zero held-out-surface top-1 is not a capability verdict. The run is preserved under `runs/foundation_v1_seed101001/` and is not eligible for graduation or cherry-picking.

## Foundation v1R2 correction

The panel now contains independently generated `same_surface_novel` and `same_surface_induction` controls, plus a disjoint `heldout_surface` panel. Thresholds are unchanged. The corrected panel and leakage audit are frozen under `data/generated/foundation_v1r2/`; the protocol receipt is `data_specs/V010_FOUNDATION_FREEZE.json`.

The next run starts from fresh random initialization under protocol `BABY_V010_FOUNDATION_V1R2`.

## v2R4 independent isolation diagnostic (post-terminal, no training)

v2R4 seed `106001` U16000 remains a frozen terminal failure (Gate L pass, C/R fail). This entry does not rewrite that adjudication.

A data-only join of the committed metrics to frozen `foundation_v2` panels found: primitive keyed is entirely `pair_count=1`; same-surface 3-pair value copy is 7/21 (chance); held-out full exact 0/96 hides 23/96 value-span copies that fail on held-out separators; held-out value copy matches broken-context (value still present under a replaced key). v2R5 is absent in GitHub and stays unresolved. No training was launched. Isolation transforms and a checkpoint probe are in `src/baby_v010/`; Fan Diesel commands are in `docs/FAN_DIESEL_HANDOFF_ISOLATION.md`. Frozen Gate L/C/R were not changed.

