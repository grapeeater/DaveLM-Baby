# Actual Baby campaign ledger

U16000 remains authoritative. C2/D3 stay fallbacks. TEST closed. Not promoted.

## e1_zeroshot

- Change: U16000 native vs C2+D3 vs tiling-gated D3 on less-rigid bind probes
- verdict: **FAIL**
- lesson: no English bind yet: heldout 0.000 cloze2 0.000
- native: {'rigid_primitive': {'first_top1': 1.0, 'free_exact': 1.0, 'tiling': 0.0}, 'aperiodic': {'first_top1': 0.15625, 'free_exact': 0.0625, 'tiling': 0.03125}, 'syntax_wrap': {'first_top1': 0.03125, 'free_exact': 0.03125, 'tiling': 0.96875}, 'cloze_1fact': {'first_top1': 0.09375, 'free_exact': 0.0, 'tiling': 0.25}, 'cloze_2fact': {'first_top1': 0.0, 'free_exact': 0.0, 'tiling': 0.6875}, 'qa_2fact_train': {'first_top1': 0.0, 'free_exact': 0.0, 'tiling': 0.6875}, 'qa_2fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0, 'tiling': 0.71875}, 'qa_3fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0, 'tiling': 0.84375}, 'qa_4fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0, 'tiling': 0.90625}, 'instr_2fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0, 'tiling': 1.0}, 'pronoun_2fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0, 'tiling': 0.25}, 'dialogue_2fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0, 'tiling': 0.65625}}
- d3: {'rigid_primitive': {'first_top1': 1.0, 'free_exact': 1.0}, 'aperiodic': {'first_top1': 0.125, 'free_exact': 0.125}, 'syntax_wrap': {'first_top1': 0.0, 'free_exact': 0.0}, 'cloze_1fact': {'first_top1': 0.0, 'free_exact': 0.0}, 'cloze_2fact': {'first_top1': 0.125, 'free_exact': 0.0}, 'qa_2fact_train': {'first_top1': 0.0, 'free_exact': 0.0}, 'qa_2fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0}, 'qa_3fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0}, 'qa_4fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0}, 'instr_2fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0}, 'pronoun_2fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0}, 'dialogue_2fact_heldout': {'first_top1': 0.15625, 'free_exact': 0.0625}}

## e1_zeroshot

- Change: U16000 native vs C2+D3 vs tiling-gated D3 on less-rigid bind probes
- verdict: **FAIL**
- lesson: no English bind yet: heldout 0.000 cloze2 0.000
- native: {'rigid_primitive': {'first_top1': 1.0, 'free_exact': 1.0, 'tiling': 0.0}, 'aperiodic': {'first_top1': 0.15625, 'free_exact': 0.0625, 'tiling': 0.03125}, 'syntax_wrap': {'first_top1': 0.03125, 'free_exact': 0.03125, 'tiling': 0.96875}, 'cloze_1fact': {'first_top1': 0.09375, 'free_exact': 0.0, 'tiling': 0.25}, 'cloze_2fact': {'first_top1': 0.0, 'free_exact': 0.0, 'tiling': 0.6875}, 'qa_2fact_train': {'first_top1': 0.0, 'free_exact': 0.0, 'tiling': 0.6875}, 'qa_2fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0, 'tiling': 0.71875}, 'qa_3fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0, 'tiling': 0.84375}, 'qa_4fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0, 'tiling': 0.90625}, 'instr_2fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0, 'tiling': 1.0}, 'pronoun_2fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0, 'tiling': 0.25}, 'dialogue_2fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0, 'tiling': 0.65625}}
- d3: {'rigid_primitive': {'first_top1': 1.0, 'free_exact': 1.0}, 'aperiodic': {'first_top1': 0.125, 'free_exact': 0.125}, 'syntax_wrap': {'first_top1': 0.0, 'free_exact': 0.0}, 'cloze_1fact': {'first_top1': 0.0, 'free_exact': 0.0}, 'cloze_2fact': {'first_top1': 0.125, 'free_exact': 0.0}, 'qa_2fact_train': {'first_top1': 0.0, 'free_exact': 0.0}, 'qa_2fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0}, 'qa_3fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0}, 'qa_4fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0}, 'instr_2fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0}, 'pronoun_2fact_heldout': {'first_top1': 0.0, 'free_exact': 0.0}, 'dialogue_2fact_heldout': {'first_top1': 0.15625, 'free_exact': 0.0625}}

## e1_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **PASS**
- lesson: D3 long-gap 204/215 induction 0.312
- retention: {'n': 215, 'free_exact': 204, 'first_correct': 215, 'free_accuracy': 0.9488372093023256, 'first_accuracy': 1.0}

## e1_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **KILL**
- lesson: D3 long-gap 188/215 below 200
- retention: {'n': 215, 'free_exact': 188, 'first_correct': 215, 'free_accuracy': 0.8744186046511628, 'first_accuracy': 1.0}

## e1_stop

- Change: E1 language-bridge did not hit held-out QA/cloze gates
- verdict: **ADVANCE**
- lesson: signal but below gate: heldout 0.594 cloze2 0.969

## e2_retention

- Change: 
- verdict: **KILL**
- lesson: cheap D3 slice free_exact 0.775

## campaign_e2

- Change: E2 instruction/pronoun/dialogue/4-fact after E1
- verdict: **GRAD**
- lesson: E2 qa4=0.750 instr=0.625 pronoun=0.875 dialogue=0.688
- checkpoint_sha256: a33f61e1de0faa8b9ffca9886e12f3c5f48c394ad71c75bda3174ff369297d15

## e1_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **PASS**
- lesson: D3 long-gap 202/215 induction 0.422
- retention: {'n': 215, 'free_exact': 202, 'first_correct': 215, 'free_accuracy': 0.9395348837209302, 'first_accuracy': 1.0}

## e1_grad

- Change: QA-heavy paraphrase train from U16000; experimental ckpt only
- verdict: **GRAD**
- lesson: heldout QA 0.844 cloze2 0.969; D3 202/215 induction 0.422
- checkpoint_sha256: 4eee63b2b6fc78e869fcce3cbf27772337f53cb3038fe598e20f4e6961034307

## e2_zeroshot_from_e1

- Change: E2 panels on E1 graduate checkpoint, native only
- verdict: **ADVANCE**
- lesson: E2 partial qa4=0.875 instr=0.938 pronoun=0.469 dialogue=0.062
- native: {'qa_4fact_heldout': {'first_top1': 0.875, 'free_exact': 0.875}, 'instr_2fact_heldout': {'first_top1': 0.9375, 'free_exact': 0.90625}, 'pronoun_2fact_heldout': {'first_top1': 0.46875, 'free_exact': 0.46875}, 'dialogue_2fact_heldout': {'first_top1': 0.0625, 'free_exact': 0.0625}}

## e2_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **KILL**
- lesson: D3 long-gap 196/215 below 200
- retention: {'n': 215, 'free_exact': 196, 'first_correct': 215, 'free_accuracy': 0.9116279069767442, 'first_accuracy': 1.0}

## e2_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **KILL**
- lesson: D3 long-gap 186/215 below 200
- retention: {'n': 215, 'free_exact': 186, 'first_correct': 215, 'free_accuracy': 0.8651162790697674, 'first_accuracy': 1.0}

## e2_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **PASS**
- lesson: D3 long-gap 201/215 induction 0.391
- retention: {'n': 215, 'free_exact': 201, 'first_correct': 215, 'free_accuracy': 0.9348837209302325, 'first_accuracy': 1.0}

## e2_gentle_grad

- Change: Gentle E2 from E1 with 40% structured; experimental only
- verdict: **GRAD**
- lesson: dialogue 0.781 pronoun 0.906 qa4 0.844 instr 0.812; D3 201/215 induction 0.391
- checkpoint_sha256: 019656d5804862c41595cfa0497696cf9aba68f97940bd4767b235fa51052361

## e3_zeroshot

- Change: multi-turn follow-up on E2 gentle survivor
- verdict: **FAIL**
- lesson: E3 cold turn2=0.219 turn3=0.250 dialogue=0.750
- native: {'multiturn_2fact_heldout': {'first_top1': 0.21875, 'free_exact': 0.21875}, 'multiturn_3fact_heldout': {'first_top1': 0.25, 'free_exact': 0.25}, 'dialogue_2fact_heldout': {'first_top1': 0.75, 'free_exact': 0.75}, 'pronoun_2fact_heldout': {'first_top1': 0.90625, 'free_exact': 0.90625}}

## e3_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **KILL**
- lesson: D3 long-gap 195/215 below 200
- retention: {'n': 215, 'free_exact': 195, 'first_correct': 215, 'free_accuracy': 0.9069767441860465, 'first_accuracy': 1.0}

## e3u50_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **KILL**
- lesson: D3 long-gap 193/215 below 200
- retention: {'n': 215, 'free_exact': 193, 'first_correct': 215, 'free_accuracy': 0.8976744186046511, 'first_accuracy': 1.0}

## campaign_status_2026-09-18

- Change: Language-bridge campaign pause after E1+E2 D3-safe survivor; E3 follow-up trades D3
- verdict: **HOLD**
- lesson: U16000 not replaced. TEST closed. Experimental survivor `e2_gentle_310321/checkpoint_00050.pt` SHA `019656d5…`. She answers held-out English color questions (QA 0.875, dialogue 0.781, 4-fact 0.844, instruction 0.812, pronoun 0.906). D3 long-gap 201/215 first-token 215/215 induction 0.391 keyed 0.969. E3 multi-turn 0.94 exists at SHA `848762b7…` but D3 195/215. Next: stop-at-period decode, more attributes/entities, or multi-turn without dropping D3 below 200.

## e4_probe

- Change: Zero-shot e4 probe on experimental checkpoint (U16000 not replaced)
- verdict: **ADVANCE**
- lesson: E4 partial size=0.344 color=0.875 mixed=0.469; decode usable=0.875 stop=0.875
- checkpoint_sha256: 019656d5804862c41595cfa0497696cf9aba68f97940bd4767b235fa51052361

## e4_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **PASS**
- lesson: D3 long-gap 204/215 induction 0.422
- retention: {'n': 215, 'free_exact': 204, 'first_correct': 215, 'free_accuracy': 0.9488372093023256, 'first_accuracy': 1.0}

## e3_probe

- Change: Zero-shot e3 probe on experimental checkpoint (U16000 not replaced)
- verdict: **ADVANCE**
- lesson: E3 partial turn2=0.625 turn3=0.375 dialogue=0.844; decode usable=0.875 stop=0.875
- checkpoint_sha256: 006caa58f2cc858538cdbafbc915de2a7de8c66e8b7ac17c2c5a8aa1383a92c6

## e5_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **PASS**
- lesson: D3 long-gap 204/215 induction 0.422
- retention: {'n': 215, 'free_exact': 204, 'first_correct': 215, 'free_accuracy': 0.9488372093023256, 'first_accuracy': 1.0}

## e4_probe

- Change: Zero-shot e4 probe on experimental checkpoint (U16000 not replaced)
- verdict: **GRAD**
- lesson: E4 size=0.875 color=0.875 mixed=0.562; decode usable=0.875 stop=0.875
- checkpoint_sha256: aba1ca58ff867c5a7639629e7abdf16187aa298433070bca7617a8daffd28139

## e6_probe

- Change: Zero-shot e6 probe on experimental checkpoint (U16000 not replaced)
- verdict: **FAIL**
- lesson: E6 cold place=0.031 color=0.969 size=1.000; decode usable=0.875 stop=0.875
- checkpoint_sha256: aba1ca58ff867c5a7639629e7abdf16187aa298433070bca7617a8daffd28139

## e6_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **KILL**
- lesson: D3 long-gap 199/215 below 200
- retention: {'n': 215, 'free_exact': 199, 'first_correct': 215, 'free_accuracy': 0.9255813953488372, 'first_accuracy': 1.0}

## e6_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **KILL**
- lesson: D3 long-gap 196/215 below 200
- retention: {'n': 215, 'free_exact': 196, 'first_correct': 215, 'free_accuracy': 0.9116279069767442, 'first_accuracy': 1.0}

## e7_probe

- Change: Zero-shot e7 probe on experimental checkpoint (U16000 not replaced)
- verdict: **FAIL**
- lesson: E7 cold follow=0.031 color=0.906 size=0.812 instr=0.938; decode usable=0.875 stop=0.875
- checkpoint_sha256: aba1ca58ff867c5a7639629e7abdf16187aa298433070bca7617a8daffd28139

## e7_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **KILL**
- lesson: D3 long-gap 184/215 below 200
- retention: {'n': 215, 'free_exact': 184, 'first_correct': 215, 'free_accuracy': 0.8558139534883721, 'first_accuracy': 1.0}

## e4_probe

- Change: Zero-shot e4 probe on experimental checkpoint (U16000 not replaced)
- verdict: **GRAD**
- lesson: E4 size=0.875 color=0.875 mixed=0.562; decode usable=0.750 stop=0.750
- checkpoint_sha256: aba1ca58ff867c5a7639629e7abdf16187aa298433070bca7617a8daffd28139

## campaign_status_e5_2026-09-18

- Change: Language-bridge campaign D3-safe survivor is E5; E6/E7 English-only
- verdict: **HOLD**
- lesson: U16000 not replaced. TEST closed. D3 commit `1d0467f` pushed. Experimental survivor `e5_turn3_310511/checkpoint_00050.pt` SHA `aba1ca58…`. Color/size/instruction/dialogue/2-3-turn color context graduate. Period-stop color answers are usable. D3 204/215 first-token 215/215 induction 0.422. Place (E6 199) and color-then-size follow-up (E7 184) buy English and spend D3. Next is open TinyStories QA/dialogue on E5, not coding, not U16000 promotion.

## e8_probe

- Change: Zero-shot e8 probe on experimental checkpoint (U16000 not replaced)
- verdict: **ADVANCE**
- lesson: E8 partial story_c=0.969 who=0.000 event=0.000 color=1.000; decode usable=0.750 stop=0.750
- checkpoint_sha256: aba1ca58ff867c5a7639629e7abdf16187aa298433070bca7617a8daffd28139

## e8_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **KILL**
- lesson: D3 long-gap 196/215 below 200
- retention: {'n': 215, 'free_exact': 196, 'first_correct': 215, 'free_accuracy': 0.9116279069767442, 'first_accuracy': 1.0}

## e8b_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **KILL**
- lesson: D3 long-gap 195/215 below 200
- retention: {'n': 215, 'free_exact': 195, 'first_correct': 215, 'free_accuracy': 0.9069767441860465, 'first_accuracy': 1.0}

## e9_probe

- Change: Zero-shot e9 probe on experimental checkpoint (U16000 not replaced)
- verdict: **GRAD**
- lesson: E9 3e=0.938 pronoun=0.906 story2=0.969 color=0.906; decode usable=0.500 stop=0.625
- checkpoint_sha256: aba1ca58ff867c5a7639629e7abdf16187aa298433070bca7617a8daffd28139

## e10_probe

- Change: Zero-shot e10 probe on experimental checkpoint (U16000 not replaced)
- verdict: **FAIL**
- lesson: E10 cold combine=0.094 mixed_story=0.375 story2=0.906 color=0.938; decode usable=0.500 stop=0.625
- checkpoint_sha256: aba1ca58ff867c5a7639629e7abdf16187aa298433070bca7617a8daffd28139

## e10_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **PASS**
- lesson: D3 long-gap 200/215 induction 0.438
- retention: {'n': 215, 'free_exact': 200, 'first_correct': 215, 'free_accuracy': 0.9302325581395349, 'first_accuracy': 1.0}

## e10b_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **PASS**
- lesson: D3 long-gap 200/215 induction 0.469
- retention: {'n': 215, 'free_exact': 200, 'first_correct': 215, 'free_accuracy': 0.9302325581395349, 'first_accuracy': 1.0}

## e11_probe

- Change: Zero-shot e11 probe on experimental checkpoint (U16000 not replaced)
- verdict: **GRAD**
- lesson: E11 long3=0.812 long4=0.594 color=0.875 dialogue=0.875; decode usable=0.500 stop=0.625
- checkpoint_sha256: aba1ca58ff867c5a7639629e7abdf16187aa298433070bca7617a8daffd28139

## e12_probe

- Change: Zero-shot e12 probe on experimental checkpoint (U16000 not replaced)
- verdict: **FAIL**
- lesson: E12 cold size_stop=0.094 long5=0.281 size=0.844 color=0.875; decode usable=0.500 stop=0.625
- checkpoint_sha256: aba1ca58ff867c5a7639629e7abdf16187aa298433070bca7617a8daffd28139

## e12_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **KILL**
- lesson: D3 long-gap 183/215 below 200
- retention: {'n': 215, 'free_exact': 183, 'first_correct': 215, 'free_accuracy': 0.8511627906976744, 'first_accuracy': 1.0}

## e12b_retention

- Change: C2+D3 retention on original long-gap after language-bridge train
- verdict: **PASS**
- lesson: D3 long-gap 202/215 induction 0.422
- retention: {'n': 215, 'free_exact': 202, 'first_correct': 215, 'free_accuracy': 0.9395348837209302, 'first_accuracy': 1.0}

## e12_probe

- Change: Zero-shot e12 probe on experimental checkpoint (U16000 not replaced)
- verdict: **GRAD**
- lesson: E12 size_stop=0.906 size=0.875 color=0.969 story=0.938; decode usable=0.750 stop=1.000
- checkpoint_sha256: 6b000ffc4974244710c233d316b3bbbadcc45a4fa2f8f3cb5f4b561af72150d1

## e10_probe

- Change: Zero-shot e10 probe on experimental checkpoint (U16000 not replaced)
- verdict: **FAIL**
- lesson: E10 cold combine=0.250 mixed_story=0.375 story2=0.938 color=0.938; decode usable=0.750 stop=1.000
- checkpoint_sha256: 6b000ffc4974244710c233d316b3bbbadcc45a4fa2f8f3cb5f4b561af72150d1

## e9_probe

- Change: Zero-shot e9 probe on experimental checkpoint (U16000 not replaced)
- verdict: **GRAD**
- lesson: E9 3e=0.969 pronoun=0.844 story2=0.938 color=0.906; decode usable=0.750 stop=1.000
- checkpoint_sha256: 6b000ffc4974244710c233d316b3bbbadcc45a4fa2f8f3cb5f4b561af72150d1

## e11_probe

- Change: Zero-shot e11 probe on experimental checkpoint (U16000 not replaced)
- verdict: **GRAD**
- lesson: E11 long3=0.906 long4=0.719 color=0.938 dialogue=0.938; decode usable=0.750 stop=1.000
- checkpoint_sha256: 6b000ffc4974244710c233d316b3bbbadcc45a4fa2f8f3cb5f4b561af72150d1

## campaign_status_e12b_2026-09-18

- Change: Language-bridge continued past E5 into open story QA; D3-safe experimental survivor is now E12b size-stop
- verdict: **HOLD**
- lesson: U16000 not replaced. TEST closed. Experimental survivor `e12_stoponly_311211/checkpoint_00050.pt` SHA `6b000ffc4974244710c233d316b3bbbadcc45a4fa2f8f3cb5f4b561af72150d1`. Open TinyStories color QA/dialogue/3-entity/pronouns/roleplay/3-4-turn already worked on E5 zeroshot. Size answers now stop (`tiny.` not `tiny blue blue`). D3 202/215 first-token 215/215 induction 0.422 keyed 0.969. Color 0.969, story color 0.938, story 3e 0.969, pronoun 0.844, dialogue 0.938-1.0, instruction 1.0, long3/4 0.906/0.719, size-stop 0.906, decode stop 1.0. Killed: who/event (D3 196/195), size-stop+5-turn together (D3 183). Combine fork `e10_more_311011` D3 200 combine 0.438 mixed_story 0.406 — stalled, not primary. Still not Actual Baby: no who/event, mixed ~0.53, combine 0.25 on this ckpt, not open chat, not coding.
- checkpoint_sha256: 6b000ffc4974244710c233d316b3bbbadcc45a4fa2f8f3cb5f4b561af72150d1

