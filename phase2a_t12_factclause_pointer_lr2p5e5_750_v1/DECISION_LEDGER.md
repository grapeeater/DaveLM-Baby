# T12 DECISION — fact-clause pointer, 9:1, lr 2.5e-5, 750 updates

T11 (4:1, 5e-5, 750) was `T11_LANGUAGE_REGRESSION` 3/3. Extra language CE on the same 320-window rehearsal widened the gap and raised DEV CE versus T10 9:1. Representation metrics still appeared on 2/3 at U750 but cannot be credited. Language-safe representation remains 0/3 at this horizon.

T12 keeps T9/T10 9:1 mix, fact-clause keys, late-base scope, Phase1G parent, and the 750 takeoff horizon. Isolates **step size**: lr **2.5e-5**. Tests whether half the step lets representation replicate under CE≤1.30 and gap≤0.5. Do not add more language mix. Do not parent 680001 or T10/T11 late checkpoints. Do not reopen T3 TEST. Do not relaunch T4–T11.
