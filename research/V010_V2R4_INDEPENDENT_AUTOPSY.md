# Baby v0.10 v2R4 independent autopsy

Status: **diagnostic report from frozen artifacts only**  
Protocol: `BABY_V010_FOUNDATION_V2R4`  
Seed / checkpoint: `106001` / U16000  
v2R5: **absent in GitHub / unresolved / pending**  
Protected material opened: `false`  
Gates weakened: `false`  
Historical artifacts rewritten: `false`

This document does not replace [`research/V010_FOUNDATION_V2R4_TERMINAL.md`](V010_FOUNDATION_V2R4_TERMINAL.md). The terminal adjudication and its frozen Gate L/C/R calls stand. This is an independent slice of the same committed `metrics.jsonl` joined to the frozen `foundation_v2` panels.

Reproduce:

```text
python -B -m src.baby_v010.autopsy_v2r4 --out runs/v2r4_isolation_autopsy
```

Frozen identities (Windows CRLF bytes from the v2R4 receipts). A POSIX checkout may hash as LF; the autopsy tool accepts newline-normalized identity and must not rewrite the files.

- panels: `5f0d1d2c59c5d9ac07cd93460a59f542d1c65b9a10f130e61a28388b52342bb3`
- metrics: `b81cb5dad5084c30c95ca6b9b913e0c814125f6b511aec06cd75637aa345e74a`
- terminal checkpoint (local only): `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`

## What the terminal report established, and still holds

Gate L passed. Gate C failed. Gate R failed on median intact margin (`-0.2518`). Language DEV CE `1.2430` versus U6000 `1.1929`. Primitive keyed first-token `1.000` / exact `0.96875`. Same-surface novel first-token `0.4479` / exact `0.4271`. Held-out exact `0.0000`. Same-surface induction first-token `0.0625`. No second seed. No post-U16000 experiment on this branch.

Those headlines are true. They are not the finest-grained causal account.

## Finding 1. Primitive “keyed” is one-pair copy

Every frozen `primitive_keyed` item has `pair_count=1` (64/64). Short keyed splits as:

| pair_count | n | value-span exact | full exact | 1/K chance |
|---|---:|---:|---:|---:|
| 1 | 36 | 36/36 | 36/36 | 1.00 |
| 2 | 28 | 18/28 | 18/28 | 0.50 |

Same-surface novel (train markers, full difficulty):

| pair_count | n | value-span exact | full exact | 1/K chance |
|---|---:|---:|---:|---:|
| 2 | 31 | 19/31 (`0.613`) | 19/31 | 0.50 |
| 3 | 21 | 7/21 (`0.333`) | 7/21 | 0.333 |
| 4 | 44 | 15/44 (`0.341`) | 15/44 | 0.25 |

The 96.9% primitive-keyed headline is degenerate copy of the only payload. Three-pair same-surface value copy is exactly chance. Four-pair is only modestly above chance. Two-pair is the first place a binding signal appears.

## Finding 2. Held-out exact = 0 is separator OOD, not “no copy”

Held-out targets use separators `{82,83,84,85}`. Train-surface targets use `{76,77,78,79,80,81}`. Training never emits held-out separators.

| panel | first-token | value-span exact | full exact |
|---|---:|---:|---:|
| same_surface_novel | 43/96 | 41/96 | 41/96 |
| heldout_surface | 33/96 | 23/96 | 0/96 |
| unseen_length | 19/64 | 17/64 | 1/64 |
| low_prior | 16/64 | 15/64 | 0/64 |
| distractor | 8/64 | 5/64 | 0/64 |
| broken_context | 22/64 | 16/64 | 0/64 |

On held-out, 23/96 complete the value span. Failures among those 23 are `value_ok_bad_sep` (15) and `value_ok_bad_eos` (8). When the value is copied, the model often emits a **train** separator (`80`, `81`, `76`, `78`) instead of the held-out one. On train-surface novel, value copy implies separator exact (41/41).

This is diagnostic metric A′. It does **not** authorize weakening Gate C. Gate C held-out free-running exact still failed. The interpretation of that zero is what changes.

## Finding 3. Even value-only, held-out ≈ broken-context

[`src/baby_v010/data_v2.py`](../src/baby_v010/data_v2.py) `broken=="context"` replaces the **key** in the rendered pairs. The original **value span remains in the input** (64/64 frozen broken-context items).

Held-out value copy 23/96 = `0.2396`. Broken-context value copy 16/64 = `0.2500`. Those rates match. If held-out success were query-key binding that generalized across markers, it should beat a control where the query key is absent from the body. It does not.

The terminal report noted that held-out first-token (`0.34375`) equaled broken-context first-token. The value-span split shows the same coincidence after stripping separator OOD.

## Finding 4. Gate C axes are nested under held-out markers

Frozen `build_panels` calls `_unique_keyed(..., heldout=True)` for `heldout_surface`, `unseen_length`, `low_prior`, `distractor`, `broken_context`, and `broken_order`. Length, prior, and distractor failures cannot be isolated from held-out markers and held-out separators. Gate C lists a positive changed-position/order panel; the generator never created one. Immediate-EOS unique rows are 22/800, all induction, not language collapse. The terminal Gate L count `41/992` double-counts `novel`/`induction` aliases.

## Finding 5. Hypothesis A vs B for the value

Teacher-forced value-span equals free-running value-span on every unique keyed panel (zero disagreements). Once the first value token is selected, the rest of the span is copied. **A is false for value emission.** **A′ is true for held-out separator/EOS.** **B is the dominant failure for ≥3 pairs and for induction** (full induction 6/96 first-token, 5/96 exact, median rank 69 in the terminal table).

## Finding 6. Curriculum attractor

Primitive mix is 80% induction / 20% keyed, but keyed is one-pair and wins. After U8000 the trainer is 65% then 75% keyed. There are only six train marker families. Easy one-pair copy on familiar wrappers can dominate without a structural rule. Induction never recovers after the primitive stage.

## Conservative diagnosis

| Hypothesis | Verdict from v2R4 evidence | Confidence |
|---|---|---|
| A. Knows value, cannot emit value | Falsified (TF value = free value) | High |
| A′. Copies value, cannot emit held-out sep/EOS | Supported; inflates Gate C held-out exact=0 | High |
| B. Never reliably binds the queried pair | Supported for ≥3 pairs and induction; 1-pair is degenerate | Medium-high |
| C. Operation glued to train markers/seps | Supported; held-out value copy ≈ broken-key copy | Medium-high |
| D. Mechanism exists, curriculum does not make binding dominant | Plausible (2-pair > chance; 1-pair ceiling; induction starved) | Medium |
| E. Architecture/scale prevents acquisition | Not justified as the next claim | Low |

The previous bottleneck sentence (surface-invariant structural copying, especially induction and distractors) is directionally right and mechanistically underspecified.

## What this does not resolve

v2R5 is not in this repository. Do not treat it as pass or fail.

This autopsy cannot score query-swap, marker-swap, or value-absent items. Those require the local U16000 checkpoint on Fan Diesel via `python -B -m src.baby_v010.diagnose_checkpoint`.

Do not change Gate C after seeing separator OOD. Do not launch v2R6 from this report alone.
