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

## b1_hard_replace

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; B1 hard replace h0[gen]:=h0[argmax causal slot scores] (not query_position)
- long-gap: ON 109/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.96875 n=64
- verdict: **ADVANCE**
- lesson: long-gap 109/215 exceeds A1 102; pointer_match=0.500 mass_q=0.11887000128626823; qswap=0.4375

## b2_temp_0p5

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; B2 temperature-sharpened P11 write T=0.5
- long-gap: ON 104/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.96875 n=64
- verdict: **ADVANCE**
- lesson: long-gap 104/215 exceeds A1 102; pointer_match=0.712 mass_q=0.9369439482688904; qswap=0.4375

## b2_temp_0p25

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; B2 temperature-sharpened P11 write T=0.25
- long-gap: ON 104/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.96875 n=64
- verdict: **ADVANCE**
- lesson: long-gap 104/215 exceeds A1 102; pointer_match=0.712 mass_q=0.9991016387939453; qswap=0.4270833333333333

## b2_temp_0p1

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; B2 temperature-sharpened P11 write T=0.1
- long-gap: ON 106/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.96875 n=64
- verdict: **ADVANCE**
- lesson: long-gap 106/215 exceeds A1 102; pointer_match=0.712 mass_q=1.0; qswap=0.4270833333333333

## b7_conf_0p5

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; B7 hard replace only if pointer peak mass≥0.5, else P11 gated
- long-gap: ON 104/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.984375 n=64
- verdict: **ADVANCE**
- lesson: long-gap 104/215 exceeds A1 102; pointer_match=0.712 mass_q=0.5759758949279785; qswap=0.4270833333333333

## b7_conf_0p86

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; B7 hard replace only if pointer peak mass≥0.86, else P11 gated
- long-gap: ON 105/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.984375 n=64
- verdict: **ADVANCE**
- lesson: long-gap 105/215 exceeds A1 102; pointer_match=0.712 mass_q=0.5759758949279785; qswap=0.4270833333333333

## b1_hard_replace_stage2

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; B1 hard replace h0[gen]:=h0[argmax causal slot scores] (not query_position)
- long-gap: ON 109/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=None (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.96875 n=64
- verdict: **ADVANCE**
- lesson: retention bars held

## b1_hard_replace_stage3

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; B1 hard replace h0[gen]:=h0[argmax causal slot scores] (not query_position) ; full P11-runtime battery
- long-gap: ON 109/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=None (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.96875 n=64
- verdict: **PASS**
- lesson: cause=none; benefit_ok=True; on_retention=True; ci=[0.11428571428571432, 0.23943661971830982]

## b1_hard_replace

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; B1 hard replace h0[gen]:=h0[argmax causal slot scores] (not query_position)
- long-gap: ON 112/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.953125 n=64
- verdict: **ADVANCE**
- lesson: long-gap 112/215 CI-worthy vs A1 102; pointer_match=0.707 mass_q=0.6522766947746277; qswap=0.46875

## b1_hard_replace_stage4_250002

- ID: `a1_induction_exempt`
- Change: B1 hard replace on replica sidecar 250002
- long-gap: ON 112/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.953125 n=64
- verdict: **ADVANCE**
- lesson: long-gap 112/215 CI-worthy vs A1 102

## note_b1_primary_vs_replica

- Primary sidecar 250001 B1 hard replace: **109/215**, induction 0.297, keyed 0.969, Stage 2 ADVANCE, Stage 3 PASS, bootstrap Δ vs OFF 71 CI [0.114, 0.239]
- Replica sidecar 250002 B1 hard replace: **112/215**, induction 0.297, keyed 0.953, query-swap novel first_top1 0.469 (A1 0.427)
- A duplicate `b1_hard_replace` 112/215 ledger heading above is the replica canary (filename `b1_hard_replace_stage4_250002.json`); primary remains 109
- TEST closed. U16000 not replaced. A1 gated path still the fallback

