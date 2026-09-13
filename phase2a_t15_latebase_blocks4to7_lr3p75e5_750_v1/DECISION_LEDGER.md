# T15 DECISION — freeze blocks 8–11, train 4–7

T14 was `T14_FAIL_NO_REPRESENTATION` (representation 1/3). T14X read-only decomp showed reversal gain and language cost occupy different late-base modules: block 4 is necessary for reversals; blocks 8–11 cost language and, when restored to Phase1G on 730002, improved both CE/gap and reversals. Do not parent 730002 or the patched weights. Do not continue the scalar LR sweep. Do not propose sidecar split-LR.

T15 keeps fact-clause keys, 9:1, lr 3.75e-5, 750 updates, Phase1G parent. Isolates **scope**: freeze blocks 8–11 (plus 0–3 and binding) at Phase1G; train blocks 4–7 + embed + final_norm + language_head. Tests whether the 4–7 pathway acquires language-safe representation without the 8–11 tax. Do not reopen T3 TEST. Do not relaunch T4–T14.

## Terminal
`T15_FAIL_NO_REPRESENTATION` 3/3. Language held 3/3 (CE 1.240–1.246, gap 0.361) — better than T14. Representation 0/3 (rev 38/41/43). Freezing 8–11 during training is **not** equivalent to T14X’s post-hoc 8–11 revert. Do not parent T15. Do not launch T16 from this result. Next is the authorized read-only Qwen specimen study, then one Baby-native action after synthesis.
