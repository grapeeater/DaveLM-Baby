# PATH 3 — Replicated matched-prefix control-vs-treatment design

Status: FROZEN DESIGN ONLY. NOT EXECUTED. See PATH3_PREREGISTRATION.json for the structured record.

## Why Path 3 is the scientifically defensible route

The reproducibility autopsy established:
- PATH 1: cross-process bitwise identity is stochastic (never guaranteed); no runtime fix found.
- PATH 2: no independently justifiable equivalence standard exists.

The design below therefore removes the requirement to compare anything against the historical SF2 prefix. Both
arms of each replicate are forked from a single shared in-process prefix at update 100, so the control and
treatment arms are bitwise-identical through update 100 by construction. The only difference between arms is the
English LR schedule on updates 101-200. Stochastic GPU divergence that occurs after the fork is identically
distributed across arms (same process, same environment) and its endpoint-level magnitude is bounded by the
measured cross-mode noise floor: per-item margin deltas <= 6.97e-3 nats at u200 with zero sign flips
(path1_evidence/mode_noise_floor.json). The boundary movements the LR hypothesis addresses (SF2 u100->u200 AO
margin changes ~0.1-0.6 nats) are 10-100x larger, so the treatment contrast is resolvable above platform noise
with exact per-arm endpoint reporting.

## Why this does not weaken any historical gate

SF2, SF3, and SF4 keep their frozen classifications. SF3's and SF4's hard stops were about exact reproduction of
the historical prefix; this design does not use that historical prefix at all, so it neither needs nor weakens
those gates. The frozen acquisition/language/D3/binding endpoint definitions are reused unchanged.

## Design summary (per seed; seeds 87011, 87012, 87013, chosen before outcomes)

1. Fresh process, identical runtime; replay updates 1-100 with frozen SF2 semantics (data, schedule, KL pool,
   objective, scope, optimizer, dropout modes unchanged).
2. Capture a full in-process snapshot at update 100 (model, optimizer, RNG, English index, scope, provenance)
   before any RNG-consuming evaluation.
3. Arm A (control): from the snapshot, updates 101-200 at constant English LR 5e-5.
4. Arm B (treatment): from the SAME snapshot, updates 101-200 with English LR lr(j)=5e-5*(90-j)/89 for j=1..90
   (SF3/SF4 frozen schedule semantics); binding LR stays 5e-5.
5. Evaluate both arms at 200 with the unchanged frozen endpoint gates; report per-item margins at 125/150/175/
   200.

## Decision rule (pre-registered)

- SUPPORT: treatment passes the endpoint acquisition gate in >=2/3 replicates where matched control does not,
  and treatment never fails where control passes.
- WEAK/NO SUPPORT: both pass in all replicates where control passes, or both fail in all replicates.
- HARM: treatment fails in >=1 replicate where control passes.
- Exact counts and margins only; no population-inference claims. This mirrors Baby's single-model discipline.

## Why this is the smallest statistically meaningful study

- One manipulated variable (LR schedule after update 100) per the SF3/SF4 lineage.
- Exact shared prefix removes the historical-reproduction confound entirely.
- 3 seeds provide an independent-noise replication dimension while keeping cost small (~3 prefixes + 6 arm
  continuations, minutes each).
- Endpoint definitions, retention gates, data, parent, KL, and locked-panel rules are all reused unchanged.

## Required controller changes at execution time (not performed here)

The SF5 controller must be derived from the frozen SF2 engine and must add ONLY: full-snapshot capture at update
100, in-process snapshot restore for the second arm, and per-arm LR selection after the fork. Updates 1-100,
objectives, scope, evaluation, and gate code remain byte-identical to the frozen controller logic. The controller
must be built and frozen under its own preflight before any run, per established project practice.
