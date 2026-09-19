# Actual Baby stack2 ledger

Experimental line from E12 (`e12_stoponly_311211/checkpoint_00050.pt`).
U16000 remains AUTHORITATIVE. C2/D3 stay frozen fallbacks (a1|b1|c2|d3).
TEST/FINAL/SACRED stay sealed. Do not promote stack2 or E12 over U16000.

E12 STOP is a closed-vocab chat milestone, not a failure. This stack attacks
parked mix/combine/sentence/who/place/copy/happened/5-turn **without** the old
D3<200 auto-kill. Primary retention is E12 English (usable-chat, period-stop,
color, size-stop). D3 long-gap is measured on every survivor and logged; it is
not a sole kill switch.

## e12_mix_zeroshot

- Change: Stack2 eval e12_mix_zeroshot (U16000 not replaced)
- verdict: **SURVIVE**
- lesson: mix ADVANCE mixed=0.562 fact_combine=0.219 story_combine=0.312 mixed_story=0.375; E12 English hold color=1.000 size_stop=0.969 story=0.969; usable-chat hold usable4=0.906 usable=0.857 stop=1.000 reuse=1.0
- native: {'mixed_2e': 0.5625, 'fact_combine': 0.21875, 'story_combine': 0.3125, 'story_mixed': 0.375, 'color': 1.0, 'size_stop': 0.96875, 'story_color': 0.96875, 'dialogue': 0.90625}
- usable: {'turn': 0.8571428571428571, 'turn4': 0.90625, 'stop': 1.0, 'reuse': 1.0}
- checkpoint_sha256: 6b000ffc4974244710c233d316b3bbbadcc45a4fa2f8f3cb5f4b561af72150d1

## hist_e13_new_bars

- Change: Stack2 eval hist_e13_new_bars (U16000 not replaced)
- verdict: **KILL**
- lesson: E12 drop color=0.906 size_stop=0.969 story=0.906; usable-chat drop usable4=0.844 usable=0.762 stop=1.000 reuse=0.7857142857142857; mix ADVANCE mixed=0.531 fact_combine=0.531 story_combine=0.250 mixed_story=0.406
- native: {'mixed_2e': 0.53125, 'fact_combine': 0.53125, 'story_combine': 0.25, 'story_mixed': 0.40625, 'color': 0.90625, 'size_stop': 0.96875, 'story_color': 0.90625, 'dialogue': 0.96875}
- usable: {'turn': 0.7619047619047619, 'turn4': 0.84375, 'stop': 1.0, 'reuse': 0.7857142857142857}
- checkpoint_sha256: 4fa5060de08076458416d660429bdfed9fec989e292a36472b3278ecbdae2838

## s2a_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 169/215 induction 0.469 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 169, 'first_correct': 215, 'free_accuracy': 0.786046511627907, 'first_accuracy': 1.0}
- induction: 0.46875

## s2a

- Change: E12 dual-objective mix: 40% structured + ~40% E12 color/size/dialogue/open; remaining mix/combine. D3 199 allowed if English holds.
- verdict: **SURVIVE**
- lesson: mix GRAD mixed=0.875 fact_combine=0.656 story_combine=0.438 mixed_story=0.500; E12 English hold color=0.969 size_stop=1.000 story=1.000; usable-chat hold usable4=0.969 usable=0.952 stop=1.000 reuse=1.0 D3 169/215 induction 0.469
- native: {'mixed_2e': 0.875, 'fact_combine': 0.65625, 'story_combine': 0.4375, 'story_mixed': 0.5, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- checkpoint_sha256: 0642a2f2a4044d9936cc3f930fa096ef55fc69c5e7cfe6f7c539a3c47b5fd959
- recipe: s2a
- update: 200

## campaign_status_s2a_2026-09-18

- Change: Stack2 first mix/combine canary from E12 under English-first retention
- verdict: **MAJOR**
- lesson: U16000 not replaced. TEST closed. E12 STOP not rewritten. Experimental stack2 survivor `runs/actual_baby/stack2/s2a_protect40_combine_322001/checkpoint_00200.pt` SHA `0642a2f2a4044d9936cc3f930fa096ef55fc69c5e7cfe6f7c539a3c47b5fd959`. Mix GRAD mixed 0.562→0.875, fact-combine 0.219→0.656, story-mixed 0.375→0.500, story-combine 0.312→0.438. Usable-chat 4-turn 0.906→0.969, overall 0.857→0.952, 5-turn 0.700→0.900, period-stop 1.0, fact-reuse 1.0, size-stop 0.906→1.0, color 0.969 held. D3 202→169/215 first-token 215/215 induction 0.422→0.469 — logged, not auto-killed. Historical E13 still fails new bars (usable 0.762, reuse 0.786). Do not promote. Next cheap attack if continuing: recover D3 without giving back combine/chat.
- checkpoint_sha256: 0642a2f2a4044d9936cc3f930fa096ef55fc69c5e7cfe6f7c539a3c47b5fd959

## s2a

- Change: Stack2 eval s2a (U16000 not replaced)
- verdict: **SURVIVE**
- lesson: mix GRAD mixed=0.875 fact_combine=0.656 story_combine=0.438 mixed_story=0.500; E12 English hold color=0.969 size_stop=1.000 story=1.000; usable-chat hold usable4=0.969 usable=0.952 stop=1.000 reuse=1.0
- native: {'mixed_2e': 0.875, 'fact_combine': 0.65625, 'story_combine': 0.4375, 'story_mixed': 0.5, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- usable: {'turn': 0.9523809523809523, 'turn4': 0.96875, 'stop': 1.0, 'reuse': 1.0}
- checkpoint_sha256: 0642a2f2a4044d9936cc3f930fa096ef55fc69c5e7cfe6f7c539a3c47b5fd959

## s2g_u00025

- Change: From s2a: cut mix 45%→10% and structured 40%→20%; language restores greedy span. Prior lesson: skip extra structured.
- verdict: **KILL**
- lesson: s2a mix drop mixed=0.812 fact_combine=0.594; E12 English hold color=0.969 size_stop=0.969 story=1.000; D3 slice 0.875 (parent 0.825) first=1.000
- native: {'mixed_2e': 0.8125, 'fact_combine': 0.59375, 'story_combine': 0.28125, 'story_mixed': 0.53125, 'color': 0.96875, 'size_stop': 0.96875, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.875, 'tf_exact': 0.875}
- checkpoint_sha256: 0da5da798a4419c4dd72ddd831942a89ccc5f45e264237390d17707716f46e27
- recipe: s2g
- update: 25

## s2d_u00025

- Change: From s2a: language-heavy restore of D3 greedy span (first-token already 215/215).
- verdict: **KILL**
- lesson: s2a mix drop mixed=0.875 fact_combine=0.562; E12 English hold color=0.969 size_stop=1.000 story=1.000; D3 slice 0.800 (parent 0.825) first=1.000
- native: {'mixed_2e': 0.875, 'fact_combine': 0.5625, 'story_combine': 0.4375, 'story_mixed': 0.59375, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.8, 'tf_exact': 0.775}
- checkpoint_sha256: 62851185ca7000de1c75e7ea58ace1b68e9e05500ade67a1d950df54d4df9435
- recipe: s2d
- update: 25

## s2h_u00025

- Change: From s2a: remainder-masked keyed/induction CE (skip first answer token) + 10% mix hold.
- verdict: **KILL**
- lesson: s2a mix drop mixed=0.812 fact_combine=0.562; E12 English hold color=0.969 size_stop=1.000 story=1.000; D3 slice 0.875 (parent 0.825) first=1.000
- native: {'mixed_2e': 0.8125, 'fact_combine': 0.5625, 'story_combine': 0.375, 'story_mixed': 0.53125, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.875, 'tf_exact': 0.9}
- checkpoint_sha256: 134ddff3da5355a2aae4d330e93c95091fefaec478475bf31dfc79abd7de483c
- recipe: s2h
- update: 25

## s2g_u00025

- Change: From s2a: cut mix 45%→10% and structured 40%→20%; language restores greedy span. Prior lesson: skip extra structured.
- verdict: **KILL**
- lesson: s2a mix drop mixed=0.812 fact_combine=0.594; E12 English hold color=0.969 size_stop=0.969 story=1.000; D3 slice 0.875 (parent 0.825) first=1.000
- native: {'mixed_2e': 0.8125, 'fact_combine': 0.59375, 'story_combine': 0.28125, 'story_mixed': 0.53125, 'color': 0.96875, 'size_stop': 0.96875, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.875, 'tf_exact': 0.875}
- checkpoint_sha256: 0da5da798a4419c4dd72ddd831942a89ccc5f45e264237390d17707716f46e27
- recipe: s2g
- update: 25

## s2j_u00025

- Change: From s2a: 5% mix hold, language-heavy span restore.
- verdict: **KILL**
- lesson: s2a mix drop mixed=0.812 fact_combine=0.500; E12 English hold color=0.969 size_stop=1.000 story=1.000; D3 slice 0.875 (parent 0.825) first=1.000
- native: {'mixed_2e': 0.8125, 'fact_combine': 0.5, 'story_combine': 0.34375, 'story_mixed': 0.53125, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 0.96875}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.875, 'tf_exact': 0.875}
- checkpoint_sha256: a4270af5c966791c9a4191c8cf18ab5703d460bfd5d295a0ec8bab43b4a325b5
- recipe: s2j
- update: 25

## s2d_u00025

- Change: From s2a: language-heavy restore of D3 greedy span (first-token already 215/215).
- verdict: **KILL**
- lesson: s2a mix drop mixed=0.875 fact_combine=0.562; E12 English hold color=0.969 size_stop=1.000 story=1.000; D3 slice 0.800 (parent 0.825) first=1.000
- native: {'mixed_2e': 0.875, 'fact_combine': 0.5625, 'story_combine': 0.4375, 'story_mixed': 0.59375, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.8, 'tf_exact': 0.775}
- checkpoint_sha256: 62851185ca7000de1c75e7ea58ace1b68e9e05500ade67a1d950df54d4df9435
- recipe: s2d
- update: 25

## s2h_u00025

- Change: From s2a: remainder-masked keyed/induction CE (skip first answer token) + 10% mix hold.
- verdict: **KILL**
- lesson: s2a mix drop mixed=0.812 fact_combine=0.562; E12 English hold color=0.969 size_stop=1.000 story=1.000; D3 slice 0.875 (parent 0.825) first=1.000
- native: {'mixed_2e': 0.8125, 'fact_combine': 0.5625, 'story_combine': 0.375, 'story_mixed': 0.53125, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.875, 'tf_exact': 0.9}
- checkpoint_sha256: 134ddff3da5355a2aae4d330e93c95091fefaec478475bf31dfc79abd7de483c
- recipe: s2h
- update: 25

## s2i_u00025

- Change: From s2a: mix-off diagnostic. If D3 lifts and mix dies, mix dose is the span tax.
- verdict: **HOLD+**
- lesson: D3 slice 0.875 (parent 0.825) first=1.000 delta=+0.050; s2a mix hold mixed=0.875 fact_combine=0.656; E12 English hold color=0.969 size_stop=0.969 story=1.000
- native: {'mixed_2e': 0.875, 'fact_combine': 0.65625, 'story_combine': 0.5, 'story_mixed': 0.53125, 'color': 0.96875, 'size_stop': 0.96875, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.875, 'tf_exact': 0.9}
- checkpoint_sha256: 664e8ead0f922f1ee937f4e1ef12c8f1f85d196fa03f7b9a6ec48bd905229bb3
- recipe: s2i
- update: 25

## s2i_u00050

- Change: From s2a: mix-off diagnostic. If D3 lifts and mix dies, mix dose is the span tax.
- verdict: **KILL**
- lesson: s2a mix drop mixed=0.844 fact_combine=0.562; E12 English hold color=0.969 size_stop=0.969 story=1.000; D3 slice 0.950 (parent 0.825) first=1.000
- native: {'mixed_2e': 0.84375, 'fact_combine': 0.5625, 'story_combine': 0.40625, 'story_mixed': 0.5, 'color': 0.96875, 'size_stop': 0.96875, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.95, 'tf_exact': 0.95}
- checkpoint_sha256: c2137471f1d53adce3fd272794380b140f6cd12606ed706f6bffc0d95e1e1659
- recipe: s2i
- update: 50

## s2j_u00025

- Change: From s2a: 5% mix hold, language-heavy span restore.
- verdict: **KILL**
- lesson: s2a mix drop mixed=0.812 fact_combine=0.500; E12 English hold color=0.969 size_stop=1.000 story=1.000; D3 slice 0.875 (parent 0.825) first=1.000
- native: {'mixed_2e': 0.8125, 'fact_combine': 0.5, 'story_combine': 0.34375, 'story_mixed': 0.53125, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 0.96875}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.875, 'tf_exact': 0.875}
- checkpoint_sha256: a4270af5c966791c9a4191c8cf18ab5703d460bfd5d295a0ec8bab43b4a325b5
- recipe: s2j
- update: 25

## s2o_u00010

- Change: Pure combine-heavy lock from s2i@50 (D3 slice 0.950, combine 0.562). Isolates whether mix restores combine without erasing the span.
- verdict: **KILL**
- lesson: s2a mix drop mixed=0.719 fact_combine=0.438; E12 English hold color=0.969 size_stop=0.969 story=1.000; D3 slice 0.950 (parent 0.825) first=1.000
- native: {'mixed_2e': 0.71875, 'fact_combine': 0.4375, 'story_combine': 0.375, 'story_mixed': 0.53125, 'color': 0.96875, 'size_stop': 0.96875, 'story_color': 1.0, 'dialogue': 0.96875}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.95, 'tf_exact': 0.95}
- checkpoint_sha256: 62fc4e9314176f609c253ab8cc54400aed191946953f663096b2142309dd8530
- recipe: s2o
- update: 10

## s2o_u00020

- Change: Pure combine-heavy lock from s2i@50 (D3 slice 0.950, combine 0.562). Isolates whether mix restores combine without erasing the span.
- verdict: **KILL**
- lesson: s2a mix drop mixed=0.906 fact_combine=0.438; E12 English hold color=0.969 size_stop=0.938 story=0.938; D3 slice 0.950 (parent 0.825) first=1.000
- native: {'mixed_2e': 0.90625, 'fact_combine': 0.4375, 'story_combine': 0.46875, 'story_mixed': 0.59375, 'color': 0.96875, 'size_stop': 0.9375, 'story_color': 0.9375, 'dialogue': 0.96875}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.95, 'tf_exact': 0.95}
- checkpoint_sha256: 8378b9bfcb6659f52196b2b53584ec449436f55303538ee85da9e099b8ec07b5
- recipe: s2o
- update: 20

## s2q_u00010

- Change: From s2i@50: 70% combine-heavy mix + 10% language + 20% structured.
- verdict: **KILL**
- lesson: s2a mix drop mixed=0.781 fact_combine=0.594; E12 English hold color=0.969 size_stop=0.969 story=0.969; D3 slice 0.950 (parent 0.825) first=1.000
- native: {'mixed_2e': 0.78125, 'fact_combine': 0.59375, 'story_combine': 0.5, 'story_mixed': 0.65625, 'color': 0.96875, 'size_stop': 0.96875, 'story_color': 0.96875, 'dialogue': 0.875}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.95, 'tf_exact': 0.95}
- checkpoint_sha256: 67fbefcb1e8edd256d981134be794fc25485bdc5631787b5b8a19ca70e9c90f1
- recipe: s2q
- update: 10

## s2q_u00020

- Change: From s2i@50: 70% combine-heavy mix + 10% language + 20% structured.
- verdict: **HOLD**
- lesson: D3 slice 0.850 (parent 0.825) first=1.000 delta=+0.025; s2a mix hold mixed=0.844 fact_combine=0.688; E12 English hold color=0.969 size_stop=0.969 story=0.969
- native: {'mixed_2e': 0.84375, 'fact_combine': 0.6875, 'story_combine': 0.53125, 'story_mixed': 0.65625, 'color': 0.96875, 'size_stop': 0.96875, 'story_color': 0.96875, 'dialogue': 0.96875}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.85, 'tf_exact': 0.875}
- checkpoint_sha256: b117e75dd51cff717d6125bbef80706dbcee57df34be9f6cf5d915366b5a8289
- recipe: s2q
- update: 20

## s2n_u00010

- Change: Lock s2i@25 (mix still at s2a, D3 slice 0.875) with the original s2a diet.
- verdict: **KILL**
- lesson: s2a mix drop mixed=0.781 fact_combine=0.500; E12 English hold color=0.969 size_stop=0.969 story=0.969; D3 slice 0.850 (parent 0.825) first=1.000
- native: {'mixed_2e': 0.78125, 'fact_combine': 0.5, 'story_combine': 0.3125, 'story_mixed': 0.40625, 'color': 0.96875, 'size_stop': 0.96875, 'story_color': 0.96875, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.85, 'tf_exact': 0.875}
- checkpoint_sha256: 4386a215dcaad3384341d7ee066592ae7f99064235ecbefe47288db51dfffd93
- recipe: s2n
- update: 10

## s2p_u00010

- Change: Gentle half-LR lock from s2i@50; remainder-span on structured and mix answers.
- verdict: **KILL**
- lesson: s2a mix drop mixed=0.781 fact_combine=0.594; E12 English hold color=0.969 size_stop=0.938 story=1.000; D3 slice 0.800 (parent 0.825) first=1.000
- native: {'mixed_2e': 0.78125, 'fact_combine': 0.59375, 'story_combine': 0.40625, 'story_mixed': 0.46875, 'color': 0.96875, 'size_stop': 0.9375, 'story_color': 1.0, 'dialogue': 0.96875}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.8, 'tf_exact': 0.825}
- checkpoint_sha256: ac80762d2f2ade27c734ce7a425363e5e8d41874f9b66dc3413b738c68053f12
- recipe: s2p
- update: 10

## s2p_u00020

- Change: Gentle half-LR lock from s2i@50; remainder-span on structured and mix answers.
- verdict: **KILL**
- lesson: s2a mix drop mixed=0.781 fact_combine=0.594; E12 English hold color=1.000 size_stop=0.938 story=1.000; D3 slice 0.800 (parent 0.825) first=1.000
- native: {'mixed_2e': 0.78125, 'fact_combine': 0.59375, 'story_combine': 0.375, 'story_mixed': 0.46875, 'color': 1.0, 'size_stop': 0.9375, 'story_color': 1.0, 'dialogue': 0.96875}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.8, 'tf_exact': 0.825}
- checkpoint_sha256: 009ae580bbf768f33bb119ea22569b1c9173aedf58e7fc453a28a209f4d55eb9
- recipe: s2p
- update: 20

## s2l_u00025

- Change: From s2a: keep 45% mix; steal structured→language for span restore.
- verdict: **KILL**
- lesson: s2a mix drop mixed=0.812 fact_combine=0.719; E12 English hold color=0.969 size_stop=1.000 story=1.000; D3 slice 0.950 (parent 0.825) first=1.000
- native: {'mixed_2e': 0.8125, 'fact_combine': 0.71875, 'story_combine': 0.625, 'story_mixed': 0.625, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.95, 'tf_exact': 0.95}
- checkpoint_sha256: 340dc98ac0efaeeea1780f6f330d389fea3885040b9e6e7a391f05d665e26bfc
- recipe: s2l
- update: 25

## s2m_u00025

- Change: From s2a: keep 45% mix; remainder-span structured + modest language.
- verdict: **ADVANCE**
- lesson: D3 slice 0.925 (parent 0.825) first=1.000 delta=+0.100; s2a mix hold mixed=0.875 fact_combine=0.719; E12 English hold color=0.969 size_stop=1.000 story=1.000
- native: {'mixed_2e': 0.875, 'fact_combine': 0.71875, 'story_combine': 0.5625, 'story_mixed': 0.625, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.925, 'tf_exact': 0.925}
- checkpoint_sha256: bd10f0ea8239fe09bd5bec933611ba92b27c20402b24474ad287331153240ff9
- recipe: s2m
- update: 25

## s2m_verify_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 197/215 induction 0.453 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 197, 'first_correct': 215, 'free_accuracy': 0.9162790697674419, 'first_accuracy': 1.0}
- induction: 0.453125

## s2m_verify

- Change: Stack2 eval s2m_verify (U16000 not replaced)
- verdict: **SURVIVE**
- lesson: mix GRAD mixed=0.875 fact_combine=0.719 story_combine=0.562 mixed_story=0.625; E12 English hold color=0.969 size_stop=1.000 story=1.000; usable-chat hold usable4=0.969 usable=0.929 stop=1.000 reuse=1.0 D3 197/215 induction 0.453
- native: {'mixed_2e': 0.875, 'fact_combine': 0.71875, 'story_combine': 0.5625, 'story_mixed': 0.625, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- usable: {'turn': 0.9285714285714286, 'turn4': 0.96875, 'stop': 1.0, 'reuse': 1.0}
- checkpoint_sha256: bd10f0ea8239fe09bd5bec933611ba92b27c20402b24474ad287331153240ff9

## campaign_status_s2m_2026-09-18

- Change: Stack2 D3 span-recover from s2a under mix/chat hold
- verdict: **MAJOR**
- lesson: U16000 not replaced. TEST closed. s2a not promoted. Experimental recover survivor `runs/actual_baby/stack2/s2m_protect40_combine_323091/checkpoint_00025.pt` SHA `bd10f0ea8239fe09bd5bec933611ba92b27c20402b24474ad287331153240ff9`. Recipe s2m from s2a: 30% language + 25% remainder-span structured + 45% protect40 mix, 25 updates. D3 169→197/215 first-token 215/215 induction 0.469→0.453. Mix held mixed 0.875, fact-combine 0.656→0.719, story-combine 0.438→0.562, story-mixed 0.500→0.625. Usable-chat 4-turn 0.969 held, overall 0.952→0.929 (inside 0.05 bar), period-stop 1.0, fact-reuse 1.0, color 0.969, size-stop 1.0. Mix-off pulses (s2g/s2i/s2j) lift D3 but tax combine; locking from s2i@50 did not beat keep-mix remainder-span. Do not promote.
- checkpoint_sha256: bd10f0ea8239fe09bd5bec933611ba92b27c20402b24474ad287331153240ff9

