# T30 FINAL REPORT

## Study status
`T30_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — two of three branches met the inherited representation gate, while none met the native exact/forced-choice endpoint gate. All three completed U1000; no protected panel was opened.

## Architectural change
A zero-initialized 640-parameter per-channel `route_gate` persistently adds the existing query-weighted fact-clause pointer representation to every answer prediction hidden state (`prompt_len-1` onward). Parent transformer, pointer calculation, losses, schedule, scopes, tokenizer, and gates were held fixed. The route has no identity lookup and uses no correct-answer label.

## Per-seed terminal results

| seed | DEV CE | gap | pointer | native FC | exact+EOS | ptr rev | ptr fam | shared exact | unique exact | diverge | post-fork | TRAIN16 | route gate L2 | classification | checkpoint SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| 870001 | 1.251721 | 0.424 | 111/128 | 86/128 | 31/128 | 53/64 | 12/16 | 2/64 | 29/64 | 45 | 2/30 | FAIL | 0.05576 | T30_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `abc582433164158e354221c9b5900d977a4b789dc3b2b20f841de1d420392ebe` |
| 870002 | 1.257838 | 0.422 | 111/128 | 79/128 | 36/128 | 48/64 | 7/16 | 3/64 | 33/64 | 42 | 3/23 | PASS | 0.05734 | T30_FAIL_NO_REPRESENTATION | `8549363038275da9ef8e46a6a8caacceb72530010fff19a394759f9fb55612ba` |
| 870003 | 1.256680 | 0.424 | 113/128 | 91/128 | 38/128 | 51/64 | 11/16 | 3/64 | 35/64 | 41 | 3/29 | PASS | 0.05798 | T30_REPRESENTATION_SUCCESS_OUTPUT_FAIL | `d1dcc42ed43a911a0b3cdc4a6265de27e3a1a5d8461d461beee03152ff790d07` |

## Additional controls

- Seed 870001: first-disambiguating token 30/64; native reversal 22/64; binding/localizer parent identity `True`; early frozen blocks `True`; TEST/FINAL/sacred untouched.
- Seed 870002: first-disambiguating token 23/64; native reversal 17/64; binding/localizer parent identity `True`; early frozen blocks `True`; TEST/FINAL/sacred untouched.
- Seed 870003: first-disambiguating token 29/64; native reversal 28/64; binding/localizer parent identity `True`; early frozen blocks `True`; TEST/FINAL/sacred untouched.

## Adjudication

The route was active in all three runs (final gate-vector L2 ≈ 0.056–0.058), proving the added path received gradient. Native exact remained 31/36/38 of 128, shared-prefix exact 2/3/3 of 64, and first-token-correct-then-diverge remained 45/42/41. Thus persistent pointer routing did not move Baby's native-mouth frontier under this frozen recipe. Two seeds retained strong pointer representation, but seed 870002 missed the full representation gate; no seed reached native full success.

## Comparison

- T18/T24/T28/T29 terminal exact was 35–40/128; T30 was 31/36/38/128, with no replicated improvement.
- Prefix rescue recovered 108/124 only when an oracle supplied the correct first divergent token; T30 did not supply that oracle and did not reduce divergence.
- The result does not show that identity information is absent. It shows that this minimal persistent route, although utilized, was insufficient to make native greedy continuation reliable.

## Locks and integrity

No TEST, T2-EVAL-TEST, FINAL, or sacred bytes were loaded. The parent SHA remained `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`; tokenizer SHA `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`. One pre-update mechanical scope assertion was repaired before valid training; the failed attempt is preserved in `run_seed870001_prelaunch_scope_assertion`.

