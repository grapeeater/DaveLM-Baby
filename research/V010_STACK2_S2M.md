# Stack2 s2m: D3 span recover from s2a

Status: **EXPERIMENTAL SURVIVOR**. Not authoritative. Not promoted.
U16000 remains AUTHORITATIVE:
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
TEST / FINAL / SACRED stay sealed. s2a is not replaced as a promotion
candidate; s2m is the recover child, still experimental.

Parent (unpromoted mix/combine milestone):
`runs/actual_baby/stack2/s2a_protect40_combine_322001/checkpoint_00200.pt`
SHA `0642a2f2a4044d9936cc3f930fa096ef55fc69c5e7cfe6f7c539a3c47b5fd959`.

Survivor:
`runs/actual_baby/stack2/s2m_protect40_combine_323091/checkpoint_00025.pt`
SHA `bd10f0ea8239fe09bd5bec933611ba92b27c20402b24474ad287331153240ff9`.

Protocol: `src/baby_v010/selection_stack2.py` `--mode recover`.
Receipts: `runs/actual_baby/stack2/{CAMPAIGN.json,LEDGER.md,RECOVER.json,s2m_verify.json,s2m_verify_d3.json}`.

## Why this is the recover milestone

s2a taught mix/combine/chat and dropped D3 long-gap greedy span 202→169/215
with first-token still 215/215. The hole was continuation, not first-token
bind. s2m restores most of that span **without** giving back the mix/combine
gains.

| metric | s2a | s2m | vs s2a |
|---|---:|---:|---|
| D3 long-gap free_exact | 169/215 | **197/215** | +28 |
| D3 first_correct | 215/215 | 215/215 | hold |
| cheap D3 slice n=40 | 0.825 | 0.925 | +0.100 |
| mixed_2e | 0.875 | 0.875 | hold |
| fact_combine | 0.656 | **0.719** | +0.063 |
| story_combine | 0.438 | **0.562** | +0.125 |
| story_mixed | 0.500 | **0.625** | +0.125 |
| color / size-stop / story | 0.969 / 1.0 / 1.0 | 0.969 / 1.0 / 1.0 | hold |
| usable-chat 4-turn | 0.969 | 0.969 | hold |
| usable-chat overall | 0.952 | 0.929 | −0.024, inside 0.05 bar |
| period-stop / fact-reuse | 1.0 / 1.0 | 1.0 / 1.0 | hold |
| primitive induction | 0.469 | 0.453 | −0.016 |

5-turn usable is 0.800 on n=2 chats / 10 turns (s2a campaign noted 0.900).
Too small to drive the verdict. 4-turn usable held.

## Recipe

From s2a, 25 updates, seed 323091:

- 30% language CE
- 25% remainder-span structured keyed/induction CE (skip first answer token)
- 45% `protect40_combine` bridge mix (same mix remainder as s2a)

Hypothesis: keep the mix diet that taught combine/chat; steal structured
time for language + continuation CE instead of cutting mix to 5–10%.

## What failed (killed quickly)

Mix-starved language pulses lift D3 slice and tax combine:

| arm | budget | D3 slice | mixed | fact_combine | verdict |
|---|---:|---:|---:|---:|---|
| s2g lang 70% / mix 10% | 25 | 0.875 | 0.812 | 0.594 | KILL mix |
| s2d lang 55% / mix 10% | 25 | 0.800 | 0.875 | 0.562 | KILL mix, no D3 |
| s2h remainder-span / mix 10% | 25 | 0.875 | 0.812 | 0.562 | KILL mix |
| s2j lang 80% / mix 5% | 25 | 0.875 | 0.812 | 0.500 | KILL mix |
| s2i mix-off pulse | 25 | 0.875 | **0.875** | **0.656** | HOLD+ then |
| s2i mix-off continued | 50 | **0.950** | 0.844 | 0.562 | KILL mix |

Locking mix **after** the s2i@50 pulse did not beat keep-mix training:

| arm | parent | D3 slice | mixed | fact_combine | verdict |
|---|---|---:|---:|---:|---|
| s2o 100% combine_heavy | s2i@50 | 0.950 | 0.906 | 0.438 | KILL combine |
| s2q 70% combine_heavy | s2i@50 | 0.850 | 0.844 | 0.688 | HOLD only +0.025 D3 |
| s2n s2a diet | s2i@25 | 0.850 | 0.781 | 0.500 | KILL mix |
| s2p half-LR remainder lock | s2i@50 | 0.800 | 0.781 | 0.594 | KILL mix+D3 |

s2l (45% language, 10% structured, 45% mix from s2a) is the near-miss:
D3 slice 0.950 and combine 0.719, but mixed_2e 0.812 (one n=32 item under
the 0.825 hold). Not the survivor.

## What this does not license

Promotion. U16000 replacement. Opening TEST/FINAL/SACRED. Treating s2m as
Baby. Rewriting the E12 STOP ledger. Another mix-off pulse as the default
recover recipe.

## Next Baby School lesson

Stay on s2m as the experimental parent. Cheap target: the remaining 18/215
D3 span misses, or a mixed_2e-safe cousin of s2l (slice 0.95, mixed one
item shy), without cutting the 45% mix remainder. Do not open TEST.
