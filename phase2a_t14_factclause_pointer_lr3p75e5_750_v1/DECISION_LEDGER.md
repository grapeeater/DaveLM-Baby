# T14 DECISION — fact-clause pointer, 9:1, lr 3.75e-5, 750 updates

T13 (9:1, 2.5e-5, 1000) was `T13_FAIL_NO_REPRESENTATION` 3/3. Language **held** through U1000 (CE 1.263–1.267, gap 0.445–0.465). Representation 0/3. Extra 250 language-safe steps did **not** close 44→48 reversals; U750 near-misses 720002/720003 **regressed** (45→44, 46→43). Duration at 2.5e-5 is closed.

Closed axes on this recipe: mix 9:1 vs 4:1 (T11 overfit); duration 500/750/1000 at 5e-5 (T9/T10) and 750/1000 at 2.5e-5 (T12/T13). T10 5e-5 still owns the only U750 representation *metrics* (2/3) and the language death. Do not add more 2.5e-5 steps. Do not add more same-window language mix.

T14 keeps T9 fact-clause keys, late-base scope, 9:1, Phase1G parent, and the 750 horizon. Isolates **step size**: lr **3.75e-5** (arithmetic midpoint of 2.5e-5 and 5e-5). Tests whether the midpoint reaches rev≥48 while keeping CE≤1.30 and gap≤0.5. Do not parent T12/T13. Do not reopen T3 TEST. Do not relaunch T4–T13.

## Terminal
`T14_FAIL_NO_REPRESENTATION`. Language CE≤1.30 3/3; gap≤0.5 only 2/3. Representation 1/3 (730002 rev 49, output fail). 730001 language-safe near-miss rev 46. Do not parent. Do not launch another scalar LR sweep. Next is read-only late-base failure decomposition.
