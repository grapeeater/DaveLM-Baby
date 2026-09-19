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

## b4_step200_hard

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; B1 hard replace h0[gen]:=h0[argmax causal slot scores] (not query_position)
- long-gap: ON 110/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.953125 n=64
- verdict: **ADVANCE**
- lesson: long-gap 110/215 CI-worthy vs A1 102; pointer_match=0.707 mass_q=0.6871070861816406; qswap=None

## b4_step200_hard

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; B1 hard replace h0[gen]:=h0[argmax causal slot scores] (not query_position)
- long-gap: ON 109/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.9375 n=64
- verdict: **KILL**
- lesson: primitive_keyed negative control failed; pointer_match=0.693 mass_q=0.7377356290817261; qswap=None

## note_b4_and_queue

- B4 +200 steps from P11 weights, A1 PACK_HOOK, hard-replace eval: 110/215, pointer match 0.707 (no lift vs 0.712), keyed 0.953. Early-stopped because canary ADVANCE vs A1; not better than B1 109
- B4b pointer aux gap≥13: at 200, long-gap 109, pointer 0.693 (down), keyed **0.9375** (below 0.95). Killed
- Remaining Q0R gap (~18 hits) sits on ~29% wrong-pointer rows (miss peak mass 0.14). Same slot scorer + more aux did not move it
- B5 block-1 skipped (B1 survived; sidecar is L0-trained)
- TEST closed. U16000 not replaced. A1 gated fallback preserved. Deploy candidate is B1 hard replace + A1 routing, not kind-blind P11

## b4_step400_hard

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; B1 hard replace h0[gen]:=h0[argmax causal slot scores] (not query_position)
- long-gap: ON 110/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.953125 n=64
- verdict: **ADVANCE**
- lesson: long-gap 110/215 CI-worthy vs A1 102; pointer_match=0.721 mass_q=0.8003851175308228; qswap=None

## b4b_longgap_ptr

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; B1 hard replace h0[gen]:=h0[argmax causal slot scores] (not query_position)
- long-gap: ON 110/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.953125 n=64
- verdict: **ADVANCE**
- lesson: long-gap 110/215 CI-worthy vs A1 102; pointer_match=0.721 mass_q=0.8003851175308228; qswap=0.40625

## b4b_longgap_ptr

- ID: `a1_induction_exempt`
- Change: B4b overwrite-only, A1 PACK_HOOK, pointer aux gap>=13, hard-replace eval
- long-gap: ON 110/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.953125 n=64
- verdict: **ADVANCE**
- lesson: long-gap 110/215 CI-worthy vs A1 102

## c_locator_diag

- ID: `a1_induction_exempt`
- Change: eval-only locator pointer diagnostic (query_position labels only)
- long-gap: ON None/215 vs OFF 71/215 (kind-blind ON 102/215); armed=None
- induction: first_top1 None n=None armed=None (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 None n=None
- verdict: **DIAG**
- lesson: slot=0.712 l0=0.000 l0_on_miss=0.000 hybrid=0.595 c2=1.000 cover=1.000 agree=0.000 layer_miss=[0.0, 0.0, 0.0, 0.016129032258064516, 0.0, 0.0, 0.016129032258064516, 0.03225806451612903, 0.0, 0.0, 0.0, 0.0]

## c1d

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; C1d hard replace iff B1 slot argmax agrees with L0 attn argmax; else A1 gated
- long-gap: ON 102/215 vs OFF 71/215 (kind-blind ON 102/215); armed=None
- induction: first_top1 None n=None armed=None (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 None n=None
- verdict: **KILL**
- lesson: C1d agree_rate=0 so write is A1 gated on every row; not a new locator

## c1b

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; C1b hard replace h0[gen]:=h0[mean-head L0 attn argmax] (not query_position)
- long-gap: ON None/215 vs OFF 71/215 (kind-blind ON 102/215); armed=None
- induction: first_top1 None n=0 armed=None (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 None n=0
- verdict: **KILL**
- lesson: pointer_match=0.000 below 0.65 vs B1 0.712; skip full 215

## c1c_0p3

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; C1c hard replace; L0 attn argmax if slot mass<0.3 else B1 slot argmax
- long-gap: ON None/215 vs OFF 71/215 (kind-blind ON 102/215); armed=None
- induction: first_top1 None n=0 armed=None (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 None n=0
- verdict: **KILL**
- lesson: pointer_match=0.595 below 0.65 vs B1 0.712; skip full 215

## c1c_0p5

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; C1c hard replace; L0 attn argmax if slot mass<0.5 else B1 slot argmax
- long-gap: ON None/215 vs OFF 71/215 (kind-blind ON 102/215); armed=None
- induction: first_top1 None n=0 armed=None (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 None n=0
- verdict: **KILL**
- lesson: pointer_match=0.512 below 0.65 vs B1 0.712; skip full 215

## c2

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; C2 structural (key,value,SEP) query site; else B1 slot; hard replace
- long-gap: ON 122/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.96875 n=64
- verdict: **OWNER**
- lesson: long-gap 122/215 past owner bar ~115; pointer_match=1.000; qswap=0.6666666666666666

## c2_stage2

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; C2 structural (key,value,SEP) query site; else B1 slot; hard replace
- long-gap: ON 122/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=None (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.96875 n=64
- verdict: **ADVANCE**
- lesson: retention bars held

## c2_stage3

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; C2 structural (key,value,SEP) query site; else B1 slot; hard replace ; full P11-runtime battery
- long-gap: ON 122/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=None (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.96875 n=64
- verdict: **PASS**
- lesson: cause=none; benefit_ok=True; on_retention=True; ci=[0.18181818181818182, 0.29257641921397376]

## c2_stage4_250002

- ID: `a1_induction_exempt`
- Change: A1 induction-exempt first-step; C2 structural (key,value,SEP) query site; else B1 slot; hard replace
- long-gap: ON 122/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.953125 n=64
- verdict: **OWNER**
- lesson: long-gap 122/215 past owner bar ~115; pointer_match=1.000; qswap=0.6666666666666666

## note_c2_owner

- C1b L0 attn argmax: pointer **0/215**. Baby L0 does not point at the query. Pretest 19/40; killed
- C1c hybrid τ=0.3/0.5: pointer 0.595/0.512 (L0 fallback stomps B1 matches). Killed
- C1d agree: **0%** agreement; would be A1 gated 102. Killed
- Later-layer attn: max 3.2% on the 62 B1-miss rows. Not a locator
- **C2 structural locator** (parameter-free `(key, value, SEP)` tiling; unpaired key occurrence; never `query_position`): pointer **215/215**
- Primary 250001: **122/215** long-gap (B1 109, A1 102, Q0R 127), induction 0.297, keyed 0.969, query-swap 0.667. Stage 2 ADVANCE, Stage 3 PASS, bootstrap Δ vs OFF 71 CI [0.182, 0.293]
- Replica 250002: **122/215**, induction 0.297, keyed 0.953, query-swap 0.667
- TEST closed. U16000 not replaced. A1 gated and B1 hard-replace remain reproducible (`operator=a1|b1|c2`)
- Owner milestone: 122 with retention. Remaining 5 vs Q0R 127 is not a pointer miss

## d_miss_split

- ID: `a1_induction_exempt`
- Change: C2 ON miss-row split: first_correct/rank/inventory/transfer/Q0R sample
- long-gap: ON 122/215 vs OFF 71/215 (kind-blind ON 102/215); armed=None
- induction: first_top1 None n=None armed=None (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 None n=None
- verdict: **DIAG**
- lesson: bottleneck=write/transfer dominant=readout_wrong_inventory miss=93 first_correct=126/215 free=122/215 inv_argmax=127 matched=215 cos_l0=1.0 cos_final=0.22450172901153564 q0r_also_miss=4/20

## d2_later_final

- ID: `a1_induction_exempt`
- Change: A1+C2 frozen locator; D2 also replace final-norm gen with L0 C2 src vector
- long-gap: ON 0/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.0 n=64
- verdict: **KILL**
- lesson: primitive_keyed negative control failed; first_correct=0; qswap=0.0

## d2_later_b1

- ID: `a1_induction_exempt`
- Change: A1+C2 frozen locator; D2 also replace block-1 gen with L0 C2 src vector
- long-gap: ON 120/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.96875 n=64
- verdict: **KILL**
- lesson: long-gap 120/215 no bind gain vs C2 122/126; first_correct=124; qswap=0.6354166666666666

## d1_inv_mask

- ID: `a1_induction_exempt`
- Change: A1+C2 frozen locator; D1 first-token mask to tiling value heads (not gold target)
- long-gap: ON 123/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.96875 n=64
- verdict: **KILL**
- lesson: long-gap 123/215 first=127 did not move the C2 bottleneck; first_correct=127; qswap=0.6666666666666666

## d3_match_mask

- ID: `a1_induction_exempt`
- Change: A1+C2 frozen locator; D3 first-token mask to key-matched tiling value head (not gold)
- long-gap: ON 210/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.96875 n=64
- verdict: **GRAD**
- lesson: long-gap 210/215 phase-graduation strength vs C2 122; first_correct=215; qswap=0.96875

## d3_match_mask_stage2

- ID: `a1_induction_exempt`
- Change: A1+C2 frozen locator; D3 first-token mask to key-matched tiling value head (not gold)
- long-gap: ON 210/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=None (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.96875 n=64
- verdict: **ADVANCE**
- lesson: retention bars held

## d3_match_mask_stage3

- ID: `a1_induction_exempt`
- Change: A1+C2 frozen locator; D3 first-token mask to key-matched tiling value head (not gold) ; full P11-runtime battery
- long-gap: ON 210/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=None (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.96875 n=64
- verdict: **PASS**
- lesson: cause=none; benefit_ok=True; on_retention=True; ci=[0.6176470588235294, 0.6724890829694323]

## d3_match_mask

- ID: `a1_induction_exempt`
- Change: A1+C2 frozen locator; D3 first-token mask to key-matched tiling value head (not gold)
- long-gap: ON 210/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.953125 n=64
- verdict: **GRAD**
- lesson: long-gap 210/215 phase-graduation strength vs C2 122; first_correct=215; qswap=0.96875

## d3_match_mask_stage4_250002

- ID: `a1_induction_exempt`
- Change: A1+C2 frozen locator; D3 first-token mask to key-matched tiling value head (not gold)
- long-gap: ON 210/215 vs OFF 71/215 (kind-blind ON 102/215); armed=215
- induction: first_top1 0.296875 n=64 armed=0 (OFF 0.297; kind-blind ON 0.156)
- primitive_keyed: first_top1 0.953125 n=64
- verdict: **GRAD**
- lesson: long-gap 210/215 phase-graduation strength vs C2 122; first_correct=215; qswap=0.96875

## note_d3_grad

- C2 stays frozen (SHA `06e2e613…`, pointer 215/215). U16000 not replaced. TEST closed. Not promoted.
- Miss split on C2 93 misses: **89 first-token errors**, 4 continuation. Gold always in tiling inventory. Pred already in-inventory on 99% of misses. Inventory argmax (D1) 127/215. Key-matched tiling head is gold on **215/215**.
- Cosine L0 after C2 replace is 1.0; final-gen vs L0-src ~0.22 on **both hits and misses**. Blind later-layer copy is not the discriminator. D2 final **0/215** kill. D2 block1 **120/215** kill. D1 **123/215** kill.
- Q0R splice on 20 C2 misses: 16 inventory hits / 4 also-miss. Do not chase C2 122→Q0R 127. Remaining bottleneck is **readout among inventory**, not locator.
- **D3** (eval-only, parameter-free): first-token argmax restricted to the unique tiling value whose key equals the C2 query site. Native continuation after that. Never gold `query_position` / gold target.
- Primary 250001: **210/215** free_exact, first_correct **215/215**, induction 0.297, keyed 0.969, query-swap **0.969**, rest_lock held, value_absent 0, broken_context 0, broken_order 0, language CE unchanged. Stage 2 ADVANCE, Stage 3 PASS, bootstrap Δ vs OFF 71 CI [0.618, 0.672].
- Replica 250002: **210/215**, induction 0.297, keyed 0.953, query-swap 0.969.
- Remaining 5/215 are span misses after a correct first token. Operator flags remain `a1|b1|c2|d3`.
- **RECOMMEND phase graduation (nonprotected).** Owner decision. Do not open TEST/FINAL/SACRED.

