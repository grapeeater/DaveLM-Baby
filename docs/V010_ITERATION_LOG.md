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
