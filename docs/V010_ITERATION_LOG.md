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

## v2R4 emission-source refinement (still no training)

Greedy emissions on same-surface novel are 41 queried-pair copies, 52 competitor-pair copies, and 3 off-inventory (inventory copy 93/96). The 3-pair "chance" queried rate is a selection failure, not a copy failure. Competitor copies have median queried-token rank 2 and never rank 1. Body-reorder query-first/last transforms were added for a Fan Diesel causal slot test. See `research/V010_RESEARCH_LOG.md`.

## v2R4 mechanism census (still no training)

Teacher-forced gold continuation after the first token locks on 51/52 train-novel competitor-copy rows: she can finish the queried span if the first token is forced. Queried-copy at 2/3/4 pairs is not above 1/K at p<0.05; the 2-pair "binding signal" is withdrawn. Interim evals used probe_limit=16; matched first-16 novel queried copy is 6/16 at U15500 and 5/16 at U16000, so the terminal 41/96 is not a late jump. All 22 unique induction immediate-EOS rows occur iff the last context token is that item's own separator. Held-out off-inventory 41/41 contain a train separator. No training. See `research/V010_V2R4_MECHANISM_CENSUS.md`.

## v2R4 Fan Diesel isolation probe (weights, still no training)

Read-only score of hashed U16000 on Fan Diesel CUDA/ROCm. Query-swap does not follow the new query (novel 36/96; 2/3/4-pair 16/31, 8/21, 12/44, all chance). Stuck-old 33, other-competitor 23. New-gold rest_value_tf_lock 94/96. Marker-swap copy holds (47/96). Sep-swap exact 0 and inventory copy drops to 54/96. Value-absent original-span copy 0. Body-reorder query-first 36/65 vs matched parent 22/65. No training, no gate change, no v2R5 open. See `research/V010_V2R4_ISOLATION_PROBE.md`. Proposed unlaunched protocol: `design/V010_V2R4_FIRST_TOKEN_SELECTION_PROTOCOL.md`.

## v2R4 selection-repair S1 (matched control vs first-token CE)

Launched successor to the isolation autopsy on `codex/selection-repair-s1`. Protocol `design/V010_SELECTION_REPAIR_S1.md`. Parent U16000 SHA `94b3a9da…17827`. Control seed 110001 completed +400 before the Codex interrupt; treatment was missing and was resumed to the frozen U+400 futility checkpoint only. Adjudication: **REGRESSION** and **futility**. Body-macro gains +0.0052 / +0.0075; treatment margin did not improve; primitive induction first top-1 0.297→0.172. Hypothesis did not survive. Do not extend S1, do not launch seed 110002, do not merge to main, do not open TEST. See `research/V010_SELECTION_REPAIR_S1_TERMINAL.md`.


