# Rapid treat ledger

Eval-time P11 routing. U16000 Baby unchanged. TEST closed.

**STATUS: A1 SURVIVOR (sidecar 250001 Stage 3 PASS; replica 250002 Stage 4 PASS).**
Induction-exempt first-step overwrite restores `primitive_induction` 19/64 while keeping long-gap 102/215 (replica 99/215). Not promoted. TEST closed.

## a1_induction_exempt

- ID: `a1_induction_exempt`
- Change: induction-exempt gen_index: kind==induction identity; else first-step write
- long-gap: ON 102/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.984375 n=64
- verdict: **ADVANCE**
- lesson: induction restored; long-gap held; keyed intact; write armed on long-gap only

## a1_induction_exempt_stage2

- ID: `a1_induction_exempt`
- Change: induction-exempt gen_index: kind==induction identity; else first-step write
- long-gap: ON 102/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.984375 n=64
- verdict: **ADVANCE**
- lesson: retention bars held

## a1_induction_exempt_stage3

- ID: `a1_induction_exempt`
- Change: induction-exempt gen_index: kind==induction identity; else first-step write ; full P11-runtime battery
- long-gap: ON 102/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=None (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.984375 n=64
- verdict: **PASS**
- lesson: cause=none; benefit_ok=True; on_retention=True

## a1_induction_exempt_stage4_250002

- ID: `a1_induction_exempt`
- Change: induction-exempt gen_index: kind==induction identity; else first-step write
- long-gap: ON 99/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.984375 n=64
- verdict: **ADVANCE**
- lesson: induction restored; long-gap held; keyed intact; write armed on long-gap only

## a1_induction_exempt_stage4

- ID: `a1_induction_exempt`
- Change: A1 routing on P11 replica sidecar 250002
- long-gap: ON 99/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.984375 n=64
- verdict: **PASS**
- lesson: replica reproduces A1

