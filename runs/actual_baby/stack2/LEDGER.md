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

## s3a_u00025

- Change: From s2m: who-bind + 3-entity mix/combine. Keep 20/20/60 language/structured/mix.
- verdict: **KILL**
- lesson: s2m mix drop mixed=0.906 fact_combine=0.656 story_combine=0.406; E12 English hold color=1.000 size_stop=0.969 story=0.938; who2=0.438 who3=0.312 mixed3=0.875 sent_exact=0.000 bare=0.000 prefix=0.000; D3 slice 0.850
- native: {'mixed_2e': 0.90625, 'fact_combine': 0.65625, 'story_combine': 0.40625, 'story_mixed': 0.59375, 'color': 1.0, 'size_stop': 0.96875, 'story_color': 0.9375, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.85, 'tf_exact': 0.85}
- checkpoint_sha256: d73ab2d9789a07ec3da5206fd7dbf104c48bd290c6f909d91f38372c55e6e7de
- recipe: s3a
- update: 25

## s3b_u00025

- Change: From s2m: unprompted short-sentence answers (held-out templates). No 'answer in a sentence' prefix.
- verdict: **KILL**
- lesson: s2m mix drop mixed=0.688 fact_combine=0.438 story_combine=0.406; E12 drop color=0.781 size_stop=1.000 story=0.500; who2=0.062 who3=0.000 mixed3=0.750 sent_exact=0.000 bare=0.125 prefix=0.125; D3 slice 0.950
- native: {'mixed_2e': 0.6875, 'fact_combine': 0.4375, 'story_combine': 0.40625, 'story_mixed': 0.375, 'color': 0.78125, 'size_stop': 1.0, 'story_color': 0.5, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.95, 'tf_exact': 0.95}
- checkpoint_sha256: 811f266a5d3746d39885ddf41a913344ac171f8778d997fb091d90ce1c902b03
- recipe: s3b
- update: 25

## s3c_u00025

- Change: From s2m: who-bind + unprompted sentences together.
- verdict: **KILL**
- lesson: s2m mix drop mixed=0.656 fact_combine=0.250 story_combine=0.312; E12 collapse color=0.531 size_stop=0.969 story=0.281; who2=0.125 who3=0.312 mixed3=0.750 sent_exact=0.000 bare=0.250 prefix=0.250; D3 slice 0.950
- native: {'mixed_2e': 0.65625, 'fact_combine': 0.25, 'story_combine': 0.3125, 'story_mixed': 0.375, 'color': 0.53125, 'size_stop': 0.96875, 'story_color': 0.28125, 'dialogue': 0.9375}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.95, 'tf_exact': 0.95}
- checkpoint_sha256: b4eb9dcbb9b5010698d70e97d154199b236eec1cf139aa068b3929a68c3351f3
- recipe: s3c
- update: 25

## s3a_u00025

- Change: From s2m: who-bind + 3-entity mix/combine. Keep 20/20/60 language/structured/mix.
- verdict: **KILL**
- lesson: s2m mix drop mixed=0.906 fact_combine=0.656 story_combine=0.406; E12 English hold color=1.000 size_stop=0.969 story=0.938; who2=0.438 who3=0.312 combine3=0.375 sent_exact=0.000 bare=0.000 prefix=0.000; D3 slice 0.850
- native: {'mixed_2e': 0.90625, 'fact_combine': 0.65625, 'story_combine': 0.40625, 'story_mixed': 0.59375, 'color': 1.0, 'size_stop': 0.96875, 'story_color': 0.9375, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.85, 'tf_exact': 0.85}
- checkpoint_sha256: d73ab2d9789a07ec3da5206fd7dbf104c48bd290c6f909d91f38372c55e6e7de
- recipe: s3a
- update: 25

## s3d_u00025

- Change: Scaffold diagnostic: sentence-prefix instructions. Kill if only the prefix operator moves.
- verdict: **KILL**
- lesson: s2m mix drop mixed=0.656 fact_combine=0.500 story_combine=0.344; E12 drop color=0.844 size_stop=1.000 story=0.438; who2=0.000 who3=0.000 mixed3=0.625 sent_exact=0.000 bare=0.125 prefix=0.375; D3 slice 0.825
- native: {'mixed_2e': 0.65625, 'fact_combine': 0.5, 'story_combine': 0.34375, 'story_mixed': 0.4375, 'color': 0.84375, 'size_stop': 1.0, 'story_color': 0.4375, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.825, 'tf_exact': 0.825}
- checkpoint_sha256: 9938a6dc1b97cb96ce6ce6de1156e66fa682d0e63f05722b775e037ee0864665
- recipe: s3d
- update: 25

## s3b_u00025

- Change: From s2m: unprompted short-sentence answers (held-out templates). No 'answer in a sentence' prefix.
- verdict: **KILL**
- lesson: s2m mix drop mixed=0.688 fact_combine=0.438 story_combine=0.406; E12 drop color=0.781 size_stop=1.000 story=0.500; who2=0.062 who3=0.000 combine3=0.250 sent_exact=0.000 bare=0.125 prefix=0.125; D3 slice 0.950
- native: {'mixed_2e': 0.6875, 'fact_combine': 0.4375, 'story_combine': 0.40625, 'story_mixed': 0.375, 'color': 0.78125, 'size_stop': 1.0, 'story_color': 0.5, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.95, 'tf_exact': 0.95}
- checkpoint_sha256: 811f266a5d3746d39885ddf41a913344ac171f8778d997fb091d90ce1c902b03
- recipe: s3b
- update: 25

## s3e_u00025

- Change: From s2m: existing phrase family (query asks for a sentence) vs unprompted s3b.
- verdict: **KILL**
- lesson: s2m mix drop mixed=0.469 fact_combine=0.406 story_combine=0.375; E12 collapse color=0.250 size_stop=1.000 story=0.188; who2=0.062 who3=0.125 mixed3=0.625 sent_exact=0.000 bare=0.875 prefix=0.875; D3 slice 0.925
- native: {'mixed_2e': 0.46875, 'fact_combine': 0.40625, 'story_combine': 0.375, 'story_mixed': 0.25, 'color': 0.25, 'size_stop': 1.0, 'story_color': 0.1875, 'dialogue': 0.9375}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.925, 'tf_exact': 0.925}
- checkpoint_sha256: 72c6ddac9218ba43b3199afb03ef1cc14f2e10b3a0e25f8f7bc0971897a02198
- recipe: s3e
- update: 25

## s3c_u00025

- Change: From s2m: who-bind + unprompted sentences together.
- verdict: **KILL**
- lesson: s2m mix drop mixed=0.656 fact_combine=0.250 story_combine=0.312; E12 collapse color=0.531 size_stop=0.969 story=0.281; who2=0.125 who3=0.312 combine3=0.125 sent_exact=0.000 bare=0.250 prefix=0.250; D3 slice 0.950
- native: {'mixed_2e': 0.65625, 'fact_combine': 0.25, 'story_combine': 0.3125, 'story_mixed': 0.375, 'color': 0.53125, 'size_stop': 0.96875, 'story_color': 0.28125, 'dialogue': 0.9375}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.95, 'tf_exact': 0.95}
- checkpoint_sha256: c750c13b27c87fc088ca1e931854ed1bea082055b060e0652d3388406e63e51c
- recipe: s3c
- update: 25

## s3d_u00025

- Change: Scaffold diagnostic: sentence-prefix instructions. Kill if only the prefix operator moves.
- verdict: **KILL**
- lesson: s2m mix drop mixed=0.656 fact_combine=0.500 story_combine=0.344; E12 drop color=0.844 size_stop=1.000 story=0.438; who2=0.000 who3=0.000 combine3=0.375 sent_exact=0.000 bare=0.125 prefix=0.375; D3 slice 0.825
- native: {'mixed_2e': 0.65625, 'fact_combine': 0.5, 'story_combine': 0.34375, 'story_mixed': 0.4375, 'color': 0.84375, 'size_stop': 1.0, 'story_color': 0.4375, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.825, 'tf_exact': 0.825}
- checkpoint_sha256: ae9380f439947f7e7376d89bcef8a5da6fcfd844ec40d883de07b1aa5d9655c8
- recipe: s3d
- update: 25

## s3e_u00025

- Change: From s2m: existing phrase family (query asks for a sentence) vs unprompted s3b.
- verdict: **KILL**
- lesson: s2m mix drop mixed=0.469 fact_combine=0.406 story_combine=0.375; E12 collapse color=0.250 size_stop=1.000 story=0.188; who2=0.062 who3=0.125 combine3=0.312 sent_exact=0.000 bare=0.875 prefix=0.875; D3 slice 0.925
- native: {'mixed_2e': 0.46875, 'fact_combine': 0.40625, 'story_combine': 0.375, 'story_mixed': 0.25, 'color': 0.25, 'size_stop': 1.0, 'story_color': 0.1875, 'dialogue': 0.9375}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.925, 'tf_exact': 0.925}
- checkpoint_sha256: 2e2f54b1683efedd4b09bf15f18bc962a460f4dd359de04dc08f158e4237ebb0
- recipe: s3e
- update: 25

## s3i_u00025

- Change: s3a who-bind worked (0→0.44) but combine dropped. Same who skill, more 2e combine protection.
- verdict: **KILL**
- lesson: s2m mix drop mixed=0.969 fact_combine=0.625 story_combine=0.312; E12 English hold color=0.969 size_stop=0.938 story=0.906; who2=0.250 who3=0.250 combine3=0.375 sent_exact=0.000 bare=0.000 prefix=0.000; D3 slice 0.900
- native: {'mixed_2e': 0.96875, 'fact_combine': 0.625, 'story_combine': 0.3125, 'story_mixed': 0.6875, 'color': 0.96875, 'size_stop': 0.9375, 'story_color': 0.90625, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.9, 'tf_exact': 0.925}
- checkpoint_sha256: 430733001c3fe54d261121b913c74542f49c618007915f25788d15ad61afd9f2
- recipe: s3i
- update: 25

## s3k_u00025

- Change: Lock s3a who-pulse onto s2m mix diet. Test whether who-bind sticks while combine recovers.
- verdict: **KILL**
- lesson: s2m mix drop mixed=0.812 fact_combine=0.594 story_combine=0.594; E12 English hold color=0.969 size_stop=0.969 story=0.969; who2=0.062 who3=0.062 combine3=0.188 sent_exact=0.000 bare=0.000 prefix=0.000; D3 slice 0.825
- native: {'mixed_2e': 0.8125, 'fact_combine': 0.59375, 'story_combine': 0.59375, 'story_mixed': 0.59375, 'color': 0.96875, 'size_stop': 0.96875, 'story_color': 0.96875, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.825, 'tf_exact': 0.8}
- checkpoint_sha256: 77d6a851cfd789df226e067f6490101111cfd272fb73e5c8dbb7bd8cd3e0c777
- recipe: s3k
- update: 25

## s3n_u00025

- Change: From s3a: half-LR lock with 18% who rehearsal so who-bind does not vanish while combine returns.
- verdict: **HOLD+**
- lesson: composition signal s2m mix hold mixed=0.906 fact_combine=0.688 story_combine=0.562; E12 English hold color=1.000 size_stop=1.000 story=1.000; who2=0.250 who3=0.188 combine3=0.250 sent_exact=0.000 bare=0.000 prefix=0.000; D3 slice 0.850 whoΔ=+0.250
- native: {'mixed_2e': 0.90625, 'fact_combine': 0.6875, 'story_combine': 0.5625, 'story_mixed': 0.59375, 'color': 1.0, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.85, 'tf_exact': 0.85}
- checkpoint_sha256: ecd983f041e90eab5178e0b9b4715f8d77141e824fb91735ba2ea9f89c227692
- recipe: s3n
- update: 25

## s3p_u00025

- Change: Extend s3n +25. who=0.25 is a real signal, not a milestone; keep mix-held lock.
- verdict: **KILL**
- lesson: s2m mix drop mixed=0.906 fact_combine=0.625 story_combine=0.594; E12 English hold color=1.000 size_stop=1.000 story=1.000; who2=0.188 who3=0.312 combine3=0.312 sent_exact=0.000 bare=0.000 prefix=0.000; D3 slice 0.825
- native: {'mixed_2e': 0.90625, 'fact_combine': 0.625, 'story_combine': 0.59375, 'story_mixed': 0.59375, 'color': 1.0, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.825, 'tf_exact': 0.85}
- checkpoint_sha256: d368f8181ffa035fdf7fab2164963dbab15c2c7c5833e5110521c5c41dd5c8a6
- recipe: s3p
- update: 25

## s3r_u00025

- Change: From s3a: half-LR restore of 2e combine + story_combine with who rehearsal. Balanced who panel n=32.
- verdict: **HOLD+**
- lesson: composition signal s2m mix hold mixed=0.875 fact_combine=0.719 story_combine=0.438; E12 English hold color=1.000 size_stop=0.969 story=0.969; who2=0.375 who3=0.406 combine3=0.281 sent_exact=0.000 bare=0.000 prefix=0.000; D3 slice 0.900 whoΔ=+0.375
- native: {'mixed_2e': 0.875, 'fact_combine': 0.71875, 'story_combine': 0.4375, 'story_mixed': 0.59375, 'color': 1.0, 'size_stop': 0.96875, 'story_color': 0.96875, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.9, 'tf_exact': 0.9}
- checkpoint_sha256: 37a66eb1b00935e3b2c62343aa013768d667274bdde751131dc8b9414997ce85
- recipe: s3r
- update: 25

## s3n_verify_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 159/215 induction 0.453 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 159, 'first_correct': 215, 'free_accuracy': 0.7395348837209302, 'first_accuracy': 1.0}
- induction: 0.453125

## s3r_d3_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 183/215 induction 0.516 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 183, 'first_correct': 215, 'free_accuracy': 0.8511627906976744, 'first_accuracy': 1.0}
- induction: 0.515625

## s3v_u00025

- Change: From s3r (D3 183, combine 0.719): more who rehearsal under s2m language/structured diet.
- verdict: **KILL**
- lesson: s2m mix drop mixed=0.875 fact_combine=0.656 story_combine=0.438; E12 English hold color=0.969 size_stop=1.000 story=0.969; who2=0.375 who3=0.375 combine3=0.281 sent_exact=0.000 bare=0.000 prefix=0.000; D3 slice 0.900
- native: {'mixed_2e': 0.875, 'fact_combine': 0.65625, 'story_combine': 0.4375, 'story_mixed': 0.53125, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 0.96875, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.9, 'tf_exact': 0.9}
- checkpoint_sha256: 025ad75e189a102853b1e66703c9263aad5187f41ae2be494f98284a1682fdb9
- recipe: s3v
- update: 25

## s3s_u00025

- Change: From s3n (who3=0.50, D3 159, first=215): steal mix→language/structured remainder like s2m D3 restore, keep who rehearsal.
- verdict: **KILL**
- lesson: s2m mix drop mixed=0.969 fact_combine=0.656 story_combine=0.594; E12 English hold color=0.969 size_stop=1.000 story=1.000; who2=0.500 who3=0.469 combine3=0.219 sent_exact=0.000 bare=0.000 prefix=0.000; D3 slice 0.875
- native: {'mixed_2e': 0.96875, 'fact_combine': 0.65625, 'story_combine': 0.59375, 'story_mixed': 0.53125, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.875, 'tf_exact': 0.875}
- checkpoint_sha256: 8101c421bedf0514bd4ad7c403557f4e72cc2cfc0626090d779eafb2343b382d
- recipe: s3s
- update: 25

## s3w_u00025

- Change: From s3s (who2=0.50, combine 0.656): s3r-style combine/D3 lock while rehearsing who.
- verdict: **KILL**
- lesson: s2m mix drop mixed=0.938 fact_combine=0.594 story_combine=0.562; E12 English hold color=1.000 size_stop=1.000 story=1.000; who2=0.469 who3=0.531 combine3=0.281 sent_exact=0.000 bare=0.000 prefix=0.000; D3 slice 0.800
- native: {'mixed_2e': 0.9375, 'fact_combine': 0.59375, 'story_combine': 0.5625, 'story_mixed': 0.65625, 'color': 1.0, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.8, 'tf_exact': 0.8}
- checkpoint_sha256: bd2510ce8ac2ac1c210122131f60eea1b820cee01e34ed9da76cf59fe2658cdb
- recipe: s3w
- update: 25

## s3s_verify_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 190/215 induction 0.484 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 190, 'first_correct': 215, 'free_accuracy': 0.8837209302325582, 'first_accuracy': 1.0}
- induction: 0.484375

## s3s

- Change: Composition-first who-bind from s2m via s3a pulse / s3n lock / s3s language+structured remainder. Not promoted.
- verdict: **SURVIVE**
- lesson: who2=0.500 who3=0.469 mixed3=0.938 on balanced n=32 held-out; mixed=0.969 fact_combine=0.656 (s2a floor) story_combine=0.594; English hold; usable4=0.938 usable=0.905 stop=1 reuse=1; D3 190/215 first=215 induction 0.484; sentences unsolved (bare=0, one_word=1). U16000 unchanged.
- native: {'mixed_2e': 0.96875, 'fact_combine': 0.65625, 'story_combine': 0.59375, 'story_mixed': 0.53125, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- usable: {'turn': 0.9047619047619048, 'turn4': 0.9375, 'stop': 1.0, 'reuse': 1.0}
- checkpoint_sha256: 8101c421bedf0514bd4ad7c403557f4e72cc2cfc0626090d779eafb2343b382d
- recipe: s3s
- update: 25

## s4a_u00025

- Change: Light 8% 1-entity varied-sentence CE on the s3s diet. Expression, not overwrite.
- verdict: **HOLD**
- lesson: s3s mix hold mixed=0.938 fact_combine=0.656 story_combine=0.625; E12 English hold color=0.969 size_stop=1.000 story=1.000; s3s who hold who2=0.375 who3=0.375; bare=0.000 prefix=0.000 sent_1e=0.000 sent_2e=0.000 one_word=1.000 D3 slice 0.900
- native: {'mixed_2e': 0.9375, 'fact_combine': 0.65625, 'story_combine': 0.625, 'story_mixed': 0.625, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.9, 'tf_exact': 0.9}
- checkpoint_sha256: 0d4fa28a1796d8d328413fb43838eafdbfe475a257a26ed019b513438b282a1b
- recipe: s4a
- update: 25

## s4c_u00025

- Change: 15% 1-entity sentence CE. Still far below the failed s3b/s3c dose.
- verdict: **KILL**
- lesson: s3s mix drop mixed=0.875 fact_combine=0.500 story_combine=0.500; E12 English hold color=0.969 size_stop=1.000 story=1.000; s3s who hold who2=0.469 who3=0.438; bare=0.000 prefix=0.000 sent_1e=0.000 sent_2e=0.000 one_word=1.000 D3 slice 0.875
- native: {'mixed_2e': 0.875, 'fact_combine': 0.5, 'story_combine': 0.5, 'story_mixed': 0.5625, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 0.9375}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.875, 'tf_exact': 0.875}
- checkpoint_sha256: d483444a646a93a2da264515b16dca2b23a537e3599c57ac397676e42a6549d6
- recipe: s4c
- update: 25

## s4f_u00025

- Change: s4a/prior killed retention. Same 8% 1e dose at quarter LR.
- verdict: **HOLD**
- lesson: s3s mix hold mixed=0.875 fact_combine=0.750 story_combine=0.531; E12 English hold color=0.969 size_stop=1.000 story=1.000; s3s who hold who2=0.469 who3=0.500; bare=0.000 prefix=0.000 sent_1e=0.000 sent_2e=0.000 one_word=1.000 D3 slice 0.875
- native: {'mixed_2e': 0.875, 'fact_combine': 0.75, 'story_combine': 0.53125, 'story_mixed': 0.53125, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.875, 'tf_exact': 0.875}
- checkpoint_sha256: b93e6b8a37217e9479775efb8935639fd3aea40f8b942b9b149083c409a0a49f
- recipe: s4f
- update: 25

## s4d_u00025

- Change: Diagnostic: 8% instructed 1e sentences. Prefix gain without bare is not a win.
- verdict: **KILL**
- lesson: s3s mix drop mixed=0.906 fact_combine=0.562 story_combine=0.375; E12 English hold color=1.000 size_stop=1.000 story=0.969; s3s who hold who2=0.469 who3=0.438; bare=0.000 prefix=0.000 sent_1e=0.000 sent_2e=0.000 one_word=1.000 D3 slice 0.850
- native: {'mixed_2e': 0.90625, 'fact_combine': 0.5625, 'story_combine': 0.375, 'story_mixed': 0.5, 'color': 1.0, 'size_stop': 1.0, 'story_color': 0.96875, 'dialogue': 0.96875}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.85, 'tf_exact': 0.875}
- checkpoint_sha256: 30d52ee3ec211fb3182810908cefdf9e1b6842b3753e70e0c45cee212dfec800
- recipe: s4d
- update: 25

## s4j_u00025

- Change: Longer 8% 1e sentence rehearsal from s3s. 25-update canaries were too few examples, not a mechanism fail.
- verdict: **HOLD**
- lesson: s3s mix hold mixed=0.938 fact_combine=0.656 story_combine=0.594; E12 English hold color=1.000 size_stop=1.000 story=1.000; s3s who hold who2=0.438 who3=0.438; bare=0.000 prefix=0.000 sent_1e=0.000 sent_2e=0.000 one_word=1.000 D3 slice 0.850
- native: {'mixed_2e': 0.9375, 'fact_combine': 0.65625, 'story_combine': 0.59375, 'story_mixed': 0.625, 'color': 1.0, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.85, 'tf_exact': 0.875}
- checkpoint_sha256: 63291cd62096f322fdce0209003b91636bd51bdc831f6aa359e70202f99d7e07
- recipe: s4j
- update: 25

## s4j_u00050

- Change: Longer 8% 1e sentence rehearsal from s3s. 25-update canaries were too few examples, not a mechanism fail.
- verdict: **HOLD**
- lesson: s3s mix hold mixed=0.875 fact_combine=0.781 story_combine=0.469; E12 English hold color=1.000 size_stop=1.000 story=1.000; s3s who hold who2=0.469 who3=0.344; bare=0.000 prefix=0.000 sent_1e=0.000 sent_2e=0.000 one_word=1.000 D3 slice 0.875
- native: {'mixed_2e': 0.875, 'fact_combine': 0.78125, 'story_combine': 0.46875, 'story_mixed': 0.5625, 'color': 1.0, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.875, 'tf_exact': 0.875}
- checkpoint_sha256: c1001079a3736ed8dec07880e073ee9e632467ebccab35f72210ca7223b753bb
- recipe: s4j
- update: 50

## s4j_u00075

- Change: Longer 8% 1e sentence rehearsal from s3s. 25-update canaries were too few examples, not a mechanism fail.
- verdict: **KILL**
- lesson: s3s mix hold mixed=0.875 fact_combine=0.625 story_combine=0.312; E12 English hold color=1.000 size_stop=0.969 story=1.000; s3s who drop who2=0.344 who3=0.500; bare=0.000 prefix=0.000 sent_1e=0.000 sent_2e=0.000 one_word=1.000 D3 slice 0.875
- native: {'mixed_2e': 0.875, 'fact_combine': 0.625, 'story_combine': 0.3125, 'story_mixed': 0.5, 'color': 1.0, 'size_stop': 0.96875, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.875, 'tf_exact': 0.875}
- checkpoint_sha256: 9c679e9a7a6218710e2f84c73e31f8f5c4809bc93cc8acd9e034425a759c2b6a
- recipe: s4j
- update: 75

## s4p_u00025

- Change: Mix-protected 25% 1e sentence pulse. Steal language/structured, keep compose_lock. Not s3b mix replacement.
- verdict: **KILL**
- lesson: s3s mix hold mixed=0.938 fact_combine=0.656 story_combine=0.562; E12 English hold color=1.000 size_stop=1.000 story=1.000; s3s who drop who2=0.344 who3=0.469; bare=0.000 prefix=0.000 sent_1e=0.000 sent_2e=0.000 one_word=1.000 D3 slice 0.900
- native: {'mixed_2e': 0.9375, 'fact_combine': 0.65625, 'story_combine': 0.5625, 'story_mixed': 0.59375, 'color': 1.0, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.9, 'tf_exact': 0.9}
- checkpoint_sha256: 9e21125e3f487e7050cd420bd71102c2aa299390130d2e94ffa5b7972147843b
- recipe: s4p
- update: 25

## s4px_u00025

- Change: Extend s4p +50. Train-surface first token was rank 6-8, not fitted yet. English survived the first pulse.
- verdict: **HOLD**
- lesson: s3s mix hold mixed=0.938 fact_combine=0.656 story_combine=0.531; E12 English hold color=1.000 size_stop=1.000 story=0.938; s3s who hold who2=0.469 who3=0.438; bare=0.000 prefix=0.000 sent_1e=0.000 sent_2e=0.000 one_word=1.000 D3 slice 0.875
- native: {'mixed_2e': 0.9375, 'fact_combine': 0.65625, 'story_combine': 0.53125, 'story_mixed': 0.6875, 'color': 1.0, 'size_stop': 1.0, 'story_color': 0.9375, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.875, 'tf_exact': 0.875}
- checkpoint_sha256: 6420c7bacc48bd63aff2f03fb9b932515a60ec36a6ab2892e4fdd8cd8d2b0b0b
- recipe: s4px
- update: 25

## s4px_u00050

- Change: Extend s4p +50. Train-surface first token was rank 6-8, not fitted yet. English survived the first pulse.
- verdict: **KILL**
- lesson: s3s mix drop mixed=0.812 fact_combine=0.656 story_combine=0.531; E12 collapse color=0.625 size_stop=1.000 story=0.594; s3s who drop who2=0.344 who3=0.406; bare=0.500 prefix=0.125 sent_1e=0.000 sent_2e=0.000 one_word=0.125 D3 slice 0.875
- native: {'mixed_2e': 0.8125, 'fact_combine': 0.65625, 'story_combine': 0.53125, 'story_mixed': 0.5, 'color': 0.625, 'size_stop': 1.0, 'story_color': 0.59375, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.875, 'tf_exact': 0.9}
- checkpoint_sha256: 96fbd2b24f87a3e9be294b11f2ccc3176fbca48b28b061cd4da0cdf44b2b540a
- recipe: s4px
- update: 50

## s4l_u00025

- Change: Lock from s4px u50: restore one-word English while rehearsing 8% 1e sentences so bare expression does not vanish.
- verdict: **KILL**
- lesson: s3s mix drop mixed=0.875 fact_combine=0.594 story_combine=0.531; E12 drop color=0.844 size_stop=1.000 story=0.875; s3s who hold who2=0.406 who3=0.406; bare=0.250 prefix=0.000 sent_1e=0.000 sent_2e=0.000 one_word=0.625 D3 slice 0.900
- native: {'mixed_2e': 0.875, 'fact_combine': 0.59375, 'story_combine': 0.53125, 'story_mixed': 0.5625, 'color': 0.84375, 'size_stop': 1.0, 'story_color': 0.875, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.9, 'tf_exact': 0.9}
- checkpoint_sha256: 54b079ecafee3b759ce347c7144d4ac801fdc530546e79f474744555eef8cd5d
- recipe: s4l
- update: 25

## s4l_verify_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 192/215 induction 0.547 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 192, 'first_correct': 215, 'free_accuracy': 0.8930232558139535, 'first_accuracy': 1.0}
- induction: 0.546875

## s4m_u00025

- Change: From s4l: raise sentence rehearsal to 12% so bare does not freeze at 2/8 while English is returning.
- verdict: **KILL**
- lesson: s3s mix drop mixed=0.812 fact_combine=0.688 story_combine=0.500; E12 drop color=0.750 size_stop=1.000 story=0.906; s3s who hold who2=0.500 who3=0.500; bare=0.375 prefix=0.125 sent_1e=0.000 sent_2e=0.000 one_word=0.375 D3 slice 0.900
- native: {'mixed_2e': 0.8125, 'fact_combine': 0.6875, 'story_combine': 0.5, 'story_mixed': 0.5625, 'color': 0.75, 'size_stop': 1.0, 'story_color': 0.90625, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.9, 'tf_exact': 0.9}
- checkpoint_sha256: 4c0f142768aa57e2421a94a5214392e5c2d71e3805a2c0f6d5c9f0bc984c95e5
- recipe: s4m
- update: 25

## s4m_verify_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 185/215 induction 0.547 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 185, 'first_correct': 215, 'free_accuracy': 0.8604651162790697, 'first_accuracy': 1.0}
- induction: 0.546875

## s4m_milestone

- Change: Direct-fact short English from s3s. Bare sentence_ok 0->0.375 on unscaffolded held-out questions. Not promoted.
- verdict: **SURVIVE**
- lesson: bare=0.375 prefix=0.125 who=0.500/0.500 combine=0.688 usable4=0.906 stop=1 reuse=1 D3 185/215 first=215. Color first_top1 0.75 is style mix. 2e/WHO sentences unsolved.
- d3: {'n': 215, 'free_exact': 185, 'first_correct': 215}
- induction: 0.546875
- checkpoint_sha256: 4c0f142768aa57e2421a94a5214392e5c2d71e3805a2c0f6d5c9f0bc984c95e5
- recipe: s4m
- update: 25

## s5a_u00025

- Change: Mix-protected 25% WHO-sentence pulse from s4m. Steal language/structured, keep compose_lock.
- verdict: **HOLD**
- lesson: scaffold-only WHO sentence s4m mix hold mixed=0.938 fact_combine=0.625 story_combine=0.469; s4m english hold color=0.812 size_stop=0.969 story=1.000; s3s who drop who2=0.156 who3=0.219; direct_bare=0.250 D3 slice 0.900; who_sent=0.125 prefix=0.375 first=0.250 kinds={'who_color': 0.0, 'who_size': 0.25}
- native: {'mixed_2e': 0.9375, 'fact_combine': 0.625, 'story_combine': 0.46875, 'story_mixed': 0.5, 'color': 0.8125, 'size_stop': 0.96875, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.9, 'tf_exact': 0.9}
- checkpoint_sha256: 4ee76dc8ad2687e859c21ea77089c96f1825417d00810274323fc6c0876a8d7f
- recipe: s5a
- update: 25

## s5ax_u00025

- Change: s5a stayed at zero. One more mix-protected 25% pulse, then lock.
- verdict: **KILL**
- lesson: s4m mix drop mixed=0.844 fact_combine=0.562 story_combine=0.531; s4m collapse color=0.656 size_stop=1.000 story=1.000; s3s who drop who2=0.156 who3=0.188; direct_bare=0.250 D3 slice 0.875; who_sent=0.250 prefix=0.312 first=0.312 kinds={'who_color': 0.25, 'who_size': 0.25}
- native: {'mixed_2e': 0.84375, 'fact_combine': 0.5625, 'story_combine': 0.53125, 'story_mixed': 0.53125, 'color': 0.65625, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 0.96875}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.875, 'tf_exact': 0.875}
- checkpoint_sha256: adfa8b8d8f21621c5822d7bd5b290f79fca3d05b5c8afd218097da97715ecc9f
- recipe: s5ax
- update: 25

## s5a3_u00025

- Change: s5ax killed retention. Lower pulse 15%, steal language/structured only.
- verdict: **HOLD+**
- lesson: WHO sentence signal s4m mix hold mixed=0.906 fact_combine=0.781 story_combine=0.500; s4m english hold color=0.969 size_stop=1.000 story=1.000; s3s who hold who2=0.375 who3=0.406; direct_bare=0.000 D3 slice 0.850; who_sent=0.250 prefix=0.250 first=0.312 kinds={'who_color': 0.25, 'who_size': 0.25} Δ=+0.250
- native: {'mixed_2e': 0.90625, 'fact_combine': 0.78125, 'story_combine': 0.5, 'story_mixed': 0.5625, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 1.0, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.85, 'tf_exact': 0.85}
- checkpoint_sha256: 5f43892832f8fe2d7a7e8212334f840002a812dd4a435ae44d4c0d1cc754ba6a
- recipe: s5a3
- update: 25

## s5a3l0_u00025

- Change: Capability appeared on s5a3. STOP pulse. Lock old+new.
- verdict: **HOLD+**
- lesson: WHO sentence signal s4m mix hold mixed=0.906 fact_combine=0.750 story_combine=0.438; s4m english hold color=0.719 size_stop=1.000 story=0.781; s3s who drop who2=0.250 who3=0.250; direct_bare=0.500 D3 slice 0.850; who_sent=0.250 prefix=0.375 first=0.312 kinds={'who_color': 0.25, 'who_size': 0.25} Δ=+0.250
- native: {'mixed_2e': 0.90625, 'fact_combine': 0.75, 'story_combine': 0.4375, 'story_mixed': 0.46875, 'color': 0.71875, 'size_stop': 1.0, 'story_color': 0.78125, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.85, 'tf_exact': 0.85}
- checkpoint_sha256: d6feeb35ad45649c23103fac255584cf00a7ad5bc8770a57463d91c97bc26259
- recipe: s5a3l0
- update: 25

## s5a3l0l1_u00025

- Change: Capability appeared on s5a3l0. STOP pulse. Lock old+new.
- verdict: **HOLD+**
- lesson: WHO sentence signal s4m mix hold mixed=0.875 fact_combine=0.625 story_combine=0.438; s4m english hold color=0.719 size_stop=1.000 story=0.844; s3s who drop who2=0.219 who3=0.344; direct_bare=0.375 D3 slice 0.875; who_sent=0.375 prefix=0.312 first=0.438 kinds={'who_color': 0.375, 'who_size': 0.375} Δ=+0.375
- native: {'mixed_2e': 0.875, 'fact_combine': 0.625, 'story_combine': 0.4375, 'story_mixed': 0.46875, 'color': 0.71875, 'size_stop': 1.0, 'story_color': 0.84375, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.875, 'tf_exact': 0.875}
- checkpoint_sha256: 1041e86b7effb41df075e5964ffa19a455b4358a0a75d1aad3951fe4d3c2c334
- recipe: s5a3l0l1
- update: 25

## s5c_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 199/215 induction 0.406 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 199, 'first_correct': 215, 'free_accuracy': 0.9255813953488372, 'first_accuracy': 1.0}
- induction: 0.40625

## s5c

- Change: From s4m after s5a query-fight. WHO-sentence uses animal queries, disjoint from one-word Who-is.
- verdict: **SURVIVE**
- lesson: mix ADVANCE mixed=0.500 fact_combine=0.125 story_combine=0.125 mixed_story=0.344; E12 English hold color=0.969 size_stop=1.000 story=0.969; usable-chat hold usable4=0.906 usable=0.857 stop=1.000 reuse=1.0 D3 199/215 induction 0.406
- native: {'mixed_2e': 0.5, 'fact_combine': 0.125, 'story_combine': 0.125, 'story_mixed': 0.34375, 'color': 0.96875, 'size_stop': 1.0, 'story_color': 0.96875, 'dialogue': 0.90625}
- checkpoint_sha256: 8edde9a5092fcf47ea60945c6e1b6ea16330a11544c80b1a8dfa8173d88ea89b
- recipe: s5c
- update: 25

## s5a3l0l1l2_u00025

- Change: Capability appeared on s5a3l0l1. STOP pulse. Lock old+new.
- verdict: **KILL**
- lesson: s4m mix hold mixed=0.875 fact_combine=0.719 story_combine=0.500; s4m collapse color=0.594 size_stop=1.000 story=0.938; s3s who drop who2=0.156 who3=0.344; direct_bare=0.500 D3 slice 0.850; who_sent=0.125 prefix=0.375 first=0.250 kinds={'who_color': 0.0, 'who_size': 0.25}
- native: {'mixed_2e': 0.875, 'fact_combine': 0.71875, 'story_combine': 0.5, 'story_mixed': 0.40625, 'color': 0.59375, 'size_stop': 1.0, 'story_color': 0.9375, 'dialogue': 1.0}
- d3: {'n': 40, 'first_top1': 1.0, 'free_exact': 0.85, 'tf_exact': 0.85}
- checkpoint_sha256: be4cc9f39e4d89fd5a93b0095e28fdafd8ba8695082885b46c48eeb75ff019c9
- recipe: s5a3l0l1l2
- update: 25

## s5b3_verify_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 181/215 induction 0.500 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 181, 'first_correct': 215, 'free_accuracy': 0.8418604651162791, 'first_accuracy': 1.0}
- induction: 0.5

## s5_campaign

- Change: A+B from s4m. Pulse→lock. Disjoint looks-queries, entity-first WHO-sentences, has-object + beside.
- verdict: **HOLD**
- lesson: A who_sent peaked at 0.375 (s5m) with who_2e 0.50 held. B beside reached 0.75 (s5b3) and has 0.667 both dirs (s5bx) but not jointly with A at 0.75. 2e inverse retrieval ~0.50 is the ceiling; syntax is already present. Not promoted.
- native: s5b3 mix 0.969 combine 0.656 color 0.969
- usable: usable=0.857 usable4=0.844 stop=1.0 reuse=0.86
- d3: 181/215 first_correct 215/215
- checkpoint_sha256: 0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd
- recipe: s5b3

## s5w

- Change: From s4m. Anti-recency one-word WHO + entity margin. Hypothesis: last-entity recency is the 0.50 who_2e ceiling.
- verdict: **KILL**
- lesson: Overcorrected. who2 0.50→0.312. Diagnostic was real (acc last-gold 0.67 vs last-not 0.29) but forcing non-last gold taught never-last. color 0.906 mix held. Next: uniform 4-fact + light margin.
- native: who2=0.312 mix 0.875 combine 0.656
- recipe: s5w
- update: 25

## s5x

- Change: From s4m. Uniform 4-fact WHO + light first-token margin 0.3. No anti-recency.
- verdict: **HOLD**
- lesson: last-not-gold 0.29→0.41; who2 0.531. story_combine 0.375. Direction, not a lift.
- recipe: s5x
- update: 25

## s5xx

- Change: Continue s5x 25u, more mix.
- verdict: **HOLD**
- lesson: who2 stuck 0.531; who3 0.531; story_combine recovered 0.594. Plateau.
- recipe: s5xx
- update: 25

## s5z

- Change: From s5m. A-shaped asked-attr WHO-sentences + bind + margin 0.5.
- verdict: **HOLD**
- lesson: who_sent stuck 0.375; who2 0.438; mix 0.938; direct bare 0. Same A, no lift.
- recipe: s5z
- update: 25

## s5pp

- Change: From s5m. Same-scene pair WHO questions (both entities) + margin.
- verdict: **KILL**
- lesson: who_sent 0.375→0.250. Paired CE did not teach selection.
- recipe: s5pp
- update: 25

## r2_s5m_phase2_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 188/215 induction 0.516 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 188, 'first_correct': 215, 'free_accuracy': 0.8744186046511628, 'first_accuracy': 1.0}
- induction: 0.515625

## r2_s5b3_phase3_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 177/215 induction 0.500 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 177, 'first_correct': 215, 'free_accuracy': 0.8232558139534883, 'first_accuracy': 1.0}
- induction: 0.5

## r2_prop_router

- Change: Hidden property-match → entity first-token boost. No new CE. s4m SHA-checked.
- verdict: **SURVIVE**
- lesson: who_2e official 0.469→1.000; seed324777 0.969; last≠gold 1.000. Gate 0.70 passed.
- checkpoint_sha256: 4c0f142768aa57e2421a94a5214392e5c2d71e3805a2c0f6d5c9f0bc984c95e5

## r2a

- Change: Learned query-conditioned entity pointer from s4m.
- verdict: **KILL**
- lesson: pointer copied recency (who_2e 0.406 / ptr 0.406).
- recipe: r2a

## r2c

- Change: 25u forward has/beside CE from verified s5b3.
- verdict: **KILL**
- lesson: dirs stuck at 0.667; direct bare 0.625→0.125. Do not use as parent.
- checkpoint_sha256: c72db3bf8cfc4bbed2b6d7f292260255355c5a7ff23e4868fe6d13ff0746e1ea
- recipe: r2c

## r2_s5b3_final

- Change: s5b3 weights + WhoProp + RelAssist. A and B held-out packs.
- verdict: **SURVIVE**
- lesson: who_sent 1.000; has 0.917 (who 1.000 / what 0.833); beside 0.917 (where 0.833 / who 1.000); A survived; D3 177/215; mix 0.938; not promoted.
- checkpoint_sha256: 0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd


## r2_s5b3_final_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 177/215 induction 0.500 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 177, 'first_correct': 215, 'free_accuracy': 0.8232558139534883, 'first_accuracy': 1.0}
- induction: 0.5

## r3b_verify_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 181/215 induction 0.500 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 181, 'first_correct': 215, 'free_accuracy': 0.8418604651162791, 'first_accuracy': 1.0}
- induction: 0.5

## r3a

- Change: Tiny suffix-attn query→entity head distilled from WhoProp on train who_bind; frozen s5b3.
- verdict: **KILL**
- lesson: copied recency (pointer_gold ~0.50 / pointer_last ~0.53). Same as r2a. Do not train another last-entity pointer.
- recipe: r3a

## r3c

- Change: Local hop window on top of r3b entityness.
- verdict: **KILL**
- lesson: WHO sentences up (0.75 with piece map) but who_2e seed2 0.844 missed 0.85; usable 0.59. Lock: who_2e 0.906/0.875, WHO 0.625, usable 0.56. Not safest.
- recipe: r3c

## r3b

- Change: Frozen s5b3 + learned PropMatchHead (cue locate + frozen cosine + entityness). No Python property scanner. RelAssist still on for has/beside. Entity-piece finish for hen/he.
- verdict: **SURVIVE**
- lesson: native who_2e official 0.969 seed324777 0.906 last≠gold 0.941/0.917; seed324888 0.750 labeled weak; WHO sentences native 0.562 (below 0.80); has/beside hybrid 0.833/1.000; usable4 0.844 stop 1.000 reuse 0.857; D3 181/215; not promoted
- checkpoint_sha256: 0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd
- head_sha256: 8ea3dbfd4825058eea74626b07621ea8c0155a960fccf1530c9123f61f7d62d9
- recipe: r3b

## r4_retain_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 181/215 induction 0.500 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 181, 'first_correct': 215, 'free_accuracy': 0.8418604651162791, 'first_accuracy': 1.0}
- induction: 0.5

## r4

- Change: Native WHO sentence finish + native HAS finish + held-out paraphrases on frozen s5b3+r3b. No new trained weights. RelAssist remains for BESIDE only.
- verdict: **SURVIVE**
- lesson: who_sent 0.562→0.812 (color 1.000 / size 0.625; seeds 0.875/0.812); has native 0.417→1.000 (who/what 1.000); paraphrases native 0.950 (who 1.000 / has 1.000 / beside 0.833); who_2e 0.969/0.906 held; usable4 0.844 stop 1 reuse 0.857; D3 181/215; not promoted
- checkpoint_sha256: 0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd
- head_sha256: 8ea3dbfd4825058eea74626b07621ea8c0155a960fccf1530c9123f61f7d62d9
- recipe: r4

## r5_retain_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 181/215 induction 0.500 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 181, 'first_correct': 215, 'free_accuracy': 0.8418604651162791, 'first_accuracy': 1.0}
- induction: 0.5

## r5

- Change: Native BESIDE partner + size-WHO value bind + mixed-relation query switch on frozen s5b3+r3b. RelAssist off default chat.
- verdict: **SURVIVE**
- lesson: beside 0.833→1.000 (order/paraphrase 1.000); who_2e 1.000 on official+3 seeds last≠gold 1.000; who_sent size 0.625→1.000; mixed-relation 1.000 multiturn 1.000; usable4 0.844 stop 1 reuse 0.857; D3 181/215; not promoted
- checkpoint_sha256: 0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd
- head_sha256: 8ea3dbfd4825058eea74626b07621ea8c0155a960fccf1530c9123f61f7d62d9
- recipe: r5

## r6_retain_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 181/215 induction 0.500 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 181, 'first_correct': 215, 'free_accuracy': 0.8418604651162791, 'first_accuracy': 1.0}
- induction: 0.5

## r6

- Change: PRE-v1.0 candidate. Cross-attribute combine + about-report + dialogue role/bound fix on frozen s5b3+r3b. No new trained weights. RelAssist off default chat. NOT graduated. NOT v1.0.
- verdict: **PRE-v1.0 CANDIDATE**
- lesson: fact_combine decode 0.656→1.000 (eval first_top1 0.688 kept as batched/noisy; free_exact 1.000); story_combine decode 0.438→1.000 (first_top1 0.438 kept; free_exact 1.000); about_report 1.000; usable4/reuse/stop 0.844/0.857/1.000→1.000/1.000/1.000; integration 1.000 all families; R5 routing held; D3 181/215; not promoted
- checkpoint_sha256: 0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd
- head_sha256: 8ea3dbfd4825058eea74626b07621ea8c0155a960fccf1530c9123f61f7d62d9
- recipe: r6

## r6_retain_d3

- Change: C2+D3 long-gap logged (stack2: not an auto-kill)
- verdict: **LOG**
- lesson: D3 long-gap 181/215 induction 0.500 (not an auto-kill)
- d3: {'n': 215, 'free_exact': 181, 'first_correct': 215, 'free_accuracy': 0.8418604651162791, 'first_accuracy': 1.0}
- induction: 0.5

