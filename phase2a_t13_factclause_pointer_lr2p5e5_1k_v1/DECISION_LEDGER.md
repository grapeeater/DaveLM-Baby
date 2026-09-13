# T13 DECISION — T12 recipe, 1000 updates at lr 2.5e-5

T12 (9:1, 2.5e-5, 750) was `T12_FAIL_NO_REPRESENTATION` 3/3. Language **held** 3/3 (CE 1.242–1.253, gap ~0.41). Representation 0/3; best 710001 missed reversals 44/64 and td 47/64. Lower LR solved T10/T11 language failure and left headroom.

T13 keeps T12 parent, keys, scope, 9:1, and lr 2.5e-5. Isolates **duration**: 1000 updates. Tests whether 250 more language-safe steps close the reversal gap. Do not parent T12. Do not reopen T3 TEST. Do not relaunch T4–T12.

## Terminal
`T13_FAIL_NO_REPRESENTATION` 3/3. Language held through U1000 (CE 1.263–1.267). Representation 0/3. Extra 250 steps did not close 44→48; U750 near-misses regressed. Duration at 2.5e-5 is closed. Next isolates midpoint lr 3.75e-5 at 750 (T14). Do not parent T13.
