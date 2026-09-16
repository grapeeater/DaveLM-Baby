# Baby v0.10 v2R4 isolation diagnostic

Protocol name: `BABY_V010_V2R4_ISOLATION_DIAGNOSTIC`  
Status: preregistered diagnostic; not a training curriculum; not a gate change; not v2R5 and not v2R6.

## Hypothesis under test

The v2R4 terminal failure is a mixture of (A′) held-out separator OOD, (B) weak query-key binding above one pair, and (C) train-surface wrapper dependence. It is not primarily (A) value emission failure, and it is not yet evidence for (E) architecture incapacity.

## Observation that motivated it

Independent join of frozen U16000 rows to frozen `foundation_v2` panels (see [`research/V010_V2R4_INDEPENDENT_AUTOPSY.md`](../research/V010_V2R4_INDEPENDENT_AUTOPSY.md)):

- all 64 primitive keyed items are `pair_count=1`
- 3-pair same-surface value copy is 7/21 = 1/3
- held-out value copy is 23/96, not 0; full exact is 0 because of held-out separators
- held-out value copy ≈ broken-context value copy (23/96 vs 16/64)
- broken-context still contains the original value span
- TF value-span equals free value-span everywhere

## What this distinguishes

| Transform | If it works | If it fails |
|---|---|---|
| Pair-count value-only vs 1/K | 2-pair above chance, 3–4-pair at chance → B not solved by “keyed 97%” | 3–4-pair well above chance → B overstated; look at C/A′ |
| Query-swap on 2-pair train items | Output follows the new key → some binding exists (D) | Still emits the old value → copy-a-span / position cue (B/C) |
| Marker-swap, keep train seps | Value copy collapses → C (markers) | Value copy holds → markers are not the binding cue |
| Sep-swap, keep train markers | Value copy holds, full exact dies → A′ | Value copy dies → separator identity is entangled with copy, not just suffix OOD |
| Value-absent broken control | Original-answer copy collapses vs current 25% → current negative control is invalid | Original-answer copy stays ~25% with value gone → not copying from context |

## Support / falsify (diagnostic, not graduation)

These numbers are extra measurements. Frozen Gate L/C/R remain the graduation criteria and are not rewritten here.

Support the mixture account if, on the verified U16000 checkpoint:

1. Query-swap follow rate on 2-pair train items is substantially below the parent 2-pair value-copy rate, **or** stuck-on-old-value is comparable to follow.
2. Marker-swap (keep train seps) drops value copy toward the held-out/broken-context ~0.24 band.
3. Sep-swap (keep markers) preserves value copy and destroys full exact.
4. Value-absent original-answer value copy is far below the current broken-context 16/64.

Falsify the mixture account if:

1. Query-swap follow ≈ parent 2-pair value copy **and** 3–4-pair same-surface value copy is well above 1/K. Then B is weaker than this autopsy claimed; remaining work is surface/separator generalization (C/A′) and induction.
2. Marker-swap preserves value copy **and** held-out value copy still equals broken-context. Then the surface story is not marker identity; look at other cues (absolute position, pair order).
3. Value-absent still copies the original span. Then “copy from context” is the wrong mechanism story.

## Hard stops

- Do not train.
- Do not weaken Gate L/C/R.
- Do not open TEST / FINAL / SACRED / protected panels.
- Do not rewrite frozen `foundation_v2` panels, v2R4 metrics, hashes, or the terminal report.
- Do not load a checkpoint whose SHA-256 is not `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` unless a later sealed receipt names a different verified file.
- Do not interpret a missing checkpoint as a capability result.
- Do not interpret missing v2R5 artifacts as pass or fail.
- If v2R5 is writing a run directory, do not point `--out` at that directory.

## Evidence that must be frozen before the checkpoint probe

Already frozen and required:

- `data/generated/foundation_v2/panels.json` (newline-normalized SHA `5f0d1d2c…42bb3`)
- `runs/structured_v2r4_seed106001_from6000_terminal/metrics.jsonl` (newline-normalized SHA `b81cb5da…e74a`)
- this protocol file and `src/baby_v010/isolation_transforms.py`

Local-only, required on Fan Diesel for the probe:

- `runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt` SHA `94b3a9da…17827`

## Information gained even if it fails

A null query-swap still tells us 2-pair “success” is not query-conditioned. A null marker-swap still tells us wrappers are not the active cue. A null value-absent still tells us the current broken-context control is not measuring causal retrieval. Those are usable negative results. They do not authorize a threshold change.
