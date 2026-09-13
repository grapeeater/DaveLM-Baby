# T17X NATIVE / GENERATION FAIL-MODE — FINAL

Status: **COMPLETE, read-only.** DEV `item_results` + U1000 greedy traces. TEST not loaded (`SEALED_UNOPENED`, hash unchanged). No training. Did not parent T17 checkpoints.

Question: when T17 already has language-safe pointer+native ranking, what does greedy generation emit instead of answer+EOS?

## Confusion (DEV 128, U1000)

| seed | PNe | Pne | pNe | pne | exact |
|---|---:|---:|---:|---:|---:|
| 760001 | 81 | 34 | 7 | 6 | **0** |
| 760002 | 80 | 19 | 10 | 19 | **0** |
| 760003 | 80 | 32 | 6 | 10 | **0** |

PNe = pointer correct, native forced-choice correct, exact fail. That cell is **80–81/128** on every seed. Exact is 0 even on the rows whose ranking already selects the right name.

## Greedy traces (same exact protocol as the runner)
Almost every row is `unterminated_other` (126 / 127 / 128). Not immediate EOS, not a wrong candidate, not a missing-EOS prefix of the target.

Targets decode as `" Ava."` / `" Wes."` / `" Skye."` plus EOS. Generations are **language continuation** after `Answer:`:

- `" Tom!\"\nAnna and Ben run to the"`
- `" This is the book!\"\nTom and Anna"`
- `" plants. plants. plants. plants. plants. plant"`
- `"en. Sky birds are big and strong"`

First generated tokens are scattered language pieces, not the answer-name BOS.

## Verdict
1. Representation and native *ranking* can coexist on the T17 recipe. Generation cannot: the model continues the Phase1G story distribution.
2. This is the expected consequence of the frozen QA objective (pointer+margin only) plus T14X’s fact that `language_head` has no pointer gradient. The 100 language updates teach continuation, not `Answer: <name>.<eos>`.
3. This does **not** justify changing the tokenizer, copying Qwen, parenting 760001/760003, or unsealing TEST.

## Next
T18: Phase1G parent, T17 two-phase recipe unchanged (A 750 / revert-freeze 8–11 / B 250, fact-clause, 9:1, 3.75e-5, seeds 770001–770003). Add **answer-span teacher-forced CE** (`λ=1.0`) on the correct candidate tokens + EOS during QA updates. Pointer loss unchanged. Same gates. Do not change the mix.
