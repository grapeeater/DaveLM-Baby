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

