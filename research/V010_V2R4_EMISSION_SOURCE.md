# Baby v0.10 v2R4 emission-source refinement

This is a refinement of [`V010_V2R4_INDEPENDENT_AUTOPSY.md`](V010_V2R4_INDEPENDENT_AUTOPSY.md). It does **not** rewrite the terminal gate adjudication.

The first independent autopsy treated 3-pair same-surface value copy of 7/21 as "at chance, so binding is absent / copy is weak." That aggregate is true as a **queried-copy** rate and **misleading as a copy-capacity** rate.

## Correction

On `same_surface_novel`, greedy outputs are:

- queried pair's full value: **41/96**
- a **competitor** pair's full value: **52/96**
- not any in-context value span: **3/96**

**Inventory copy = 93/96.** Baby can copy a payload. She often copies the wrong one.

When the emission is a competitor span, the queried first token has median rank **2** and is **never** rank 1 (0/52). When the emission is the queried span, median rank is 1 (41/41). So the typical train-surface error is: both candidates are near the top, the wrong span is selected, and that wrong span is copied and emitted cleanly.

Held-out: queried 23, competitor 32, off-inventory 41. The copy-a-span recipe itself is weaker off-surface, *and* exact match is still killed by held-out separators.

**Correction:** 2-pair queried copy 19/31 is not above chance at p<0.05. Do not treat it as demonstrated query binding. See [`V010_V2R4_MECHANISM_CENSUS.md`](V010_V2R4_MECHANISM_CENSUS.md).

## Metric split to keep

Do not collapse these:

| question | diagnostic |
|---|---|
| A identify | target rank of queried first token, **and** TF lock of remaining gold tokens after that first token |
| B select/bind | queried-copy vs competitor-copy **and** 1/K chance tests |
| C/D copy+emit payload | inventory copy (queried+competitor) |
| E syntax | value_ok but not free_exact (sep/EOS) |
| F surface | same split on held-out vs train markers |

Reproduce: `python -B -m src.baby_v010.autopsy_v2r4 --out runs/v2r4_isolation_autopsy` and read `emission_source` in `AUTOPSY.json`.
