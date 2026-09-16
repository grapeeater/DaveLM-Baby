# Selection repair S1: matched +400 terminal

Status: **FUTILITY + REGRESSION — hypothesis did not survive**

Isolated diagnostic fork of v2R4 U16000. Not a replacement of v2R5/v2R6.
Not foundation graduation. No merge to `main`. Protected TEST was not opened.

Protocol: [`design/V010_SELECTION_REPAIR_S1.md`](../design/V010_SELECTION_REPAIR_S1.md)
Parent branch: `codex/selection-repair-s1` (local), stacked on
`cursor/v010-v2r4-isolation-autopsy-e841` / PR #1 vs `v0.10`.
This report does not rewrite frozen Gate L/C/R or the isolation autopsy.

## What Astra completed

Codex on Fan Diesel (2026-09-16) independently refined the isolation autopsy:
Baby is not query-blind. Query changes logits, but usually not enough to
change the winning first token. On 144 fresh bodies, body-macro first-token
accuracy was 40.2% vs 36.1% body-balanced chance; 5/144 bodies were all-query
correct; gold-span continuation lock 97.5%; 118/144 bodies emitted the same
first token for every query.

Astra then froze matched control vs treatment:

| item | identity |
|---|---|
| Protocol / code | `a9d892aea037b45ea4c72b92a870131bf1aa8580` |
| Data freeze | `436163df71a3bf4789721e312425efddb9d9bebe` |
| Parent checkpoint | `runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt` |
| Parent SHA-256 | `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` |
| Manifest SHA-256 | `7e7a96bfb39fa8e19bdbcff15ec928a6eb3e90e8fb9ae97bf9622554e9d1aac5` |
| Device | CUDA / AMD Radeon RX 9060 XT, `torch 2.9.1+rocmsdk20260116` |
| Preflight | `runs/selection_s1/PREFLIGHT.json` `status=PASS` |
| Padding max logit Δ | `1.049041748046875e-05` |

Control seed `110001` ran to the scheduled U+400 checkpoint and stopped.
Receipt reason `matched_futility_boundary` is the trainer's label for a
normal `--until 400` finish (the real paired futility predicate runs only
on the treatment arm when control's `eval_0400.json` exists).

## What was interrupted

The untrusted handoff said control was mid-eval at +200. Disk disagrees.

Control artifacts at interrupt (all receipt hashes verified after the fact):

- `runs/selection_s1/control_110001/checkpoint_16200.pt`
- `runs/selection_s1/control_110001/checkpoint_16400.pt`
- `eval_0000.json` / `eval_0200.json` / `eval_0400.json`
- `events.jsonl` (400 finite steps)
- `RECEIPT_0400.json` (`last_step=400`, elapsed 764.7 s)

No `treatment_110001/` directory existed. `SCHEDULE_110002.json` is the
optional replicate seed, not the treatment arm. v2R5/v2R6 run trees were
idle (last writes 2026-09-15 / 2026-09-16 02:17). No live S1 training
process. A leftover CUDA one-liner (PID 8396, parent terminal already
dead) was spinning; it could not be killed and was not the experiment.

Checkout was on `main` (`0b0a6c9`, DaveLM-Baby T32), which does not contain
Baby v0.10. Astra's work was only on `codex/selection-repair-s1` and was
not on `origin`.

## What this run resumed

Switched to `codex/selection-repair-s1` at `436163d`. Did not restart
control. Did not redesign the loss. Launched the frozen treatment arm:

```text
python -u -B scripts/resume_selection_s1_treatment.py
```

with `--until 400`, seed `110001`, same hashed schedule as control,
AdamW reset from the same parent, extra first-token CE on keyed rows only.

Treatment baseline `eval_0000.json` SHA-256
`74373b67ddd695bd7a73527133c6f7a0924dee7b5c9ebf9961faad4247ba7f06`
is **byte-identical** to control's baseline. Parent scoring matched.

PID of the treatment process: shell 4748 wrapping
`C:\Users\jdman\AppData\Local\Programs\Python\Python312\python.exe`.
Elapsed 802.7 s. Receipt reason: `futility`. Stopped. Did not extend to
+800. Did not launch seed 110002.

### Identity exception (do not hide)

`selection_s1.verify()` hashes working-tree bytes. After a checkout through
`main`, `data.py` / `data_v2.py` no longer match the freeze hashes copied
into `MANIFEST.json` (`7c0f9bde…` / `11d431ae…`). Those freeze hashes also
do not match any git blob of those files; they are the Sep 15 V2/V2R4
freeze-JSON values. Current committed sources are bytecode-identical to
the freeze-era `.pyc` (`co_code` equal) and `LANG_TRAIN` is unchanged.
Schedules and diagnostic panels matched the manifest exactly. Parent
checkpoint matched. The exception is recorded at
`runs/selection_s1/TREATMENT_IDENTITY_EXCEPTION.json` and was required to
complete the preregistered matched arm rather than abandon it after a
checkout destroyed the extra 2–3 working-tree bytes. This is not a license
to edit data generators mid-run.

## Control results (seed 110001)

| step | body-macro acc | vs parent | mean margin | query logit effect | rest lock | DEV CE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4022 | 0 | −0.7921 | 0.7806 | 0.9745 | 1.2430 |
| 200 | 0.3929 | −0.0093 | −0.8237 | 0.8558 | 0.9722 | 1.2454 |
| 400 | 0.4074 | **+0.0052** | −0.7758 | 0.7674 | 0.9769 | 1.2491 |

Per-K at +400: 2: 0.531, 3: 0.389, 4: 0.302.
Same-surface novel free exact 0.427 → 0.479. Query-swap first top-1 0.375 → 0.302.

Body-macro gain +0.0052 is below the 0.05 futility cutoff.

## Treatment results (seed 110001)

| step | body-macro acc | vs parent | mean margin | query logit effect | rest lock | DEV CE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4022 | 0 | −0.7921 | 0.7806 | 0.9745 | 1.2430 |
| 200 | 0.3999 | −0.0023 | −0.8279 | 0.7922 | 0.9630 | 1.2463 |
| 400 | 0.4097 | **+0.0075** | −0.8168 | 0.7244 | 0.9653 | 1.2490 |

Per-K at +400: 2: 0.531, 3: 0.396, 4: 0.302.
Margin **fell** (−0.025 vs parent); protocol required +0.25 for non-futility.
Query logit effect **fell** (−0.056); protocol required +0.50 for success.
Same-surface novel free exact 0.427 → 0.417. Query-swap first top-1 0.375 → 0.385.

Paired body bootstrap 10,000 replicates seed 110300, treatment−control:
95% CI **[−0.0191, +0.0243]**, lower bound not > 0.

## Retention / negative controls

Language DEV CE +0.0060 (hard stop >0.20; regression >0.10): no language hard stop.
Gold remainder lock 0.9745 → 0.9653 (loss 0.009 < 0.05).
Value-absent exact stayed 0. Broken-context/order free exact stayed 0.
Primitive keyed first top-1 stayed 1.0.

**Regression flags (frozen intact panels, loss > 0.05 vs baseline):**

| panel | metric | baseline | treatment | Δ |
|---|---|---:|---:|---:|
| primitive_induction | first_top1 | 0.2969 | 0.1719 | −0.125 |
| primitive_induction | free_exact | 0.2188 | 0.1406 | −0.078 |

Adjudicator status: **REGRESSION**. Futility: **true**. All eight success
criteria false. Partial gate (gain ≥0.10 parent and ≥0.05 control) also false.

## Did the hypothesis survive?

No.

Hypothesis: adding first-token vocabulary CE (weight 1.0, keyed rows only)
on balanced counterfactual queries improves causal selection more than an
equal-compute full-answer-only control.

At the frozen U+400 futility checkpoint, both arms' new body-macro gains
were < 0.05 and treatment margin gain was < 0.25. The trainer stopped
with `futility`. The matched experiment is complete. Budget must not be
extended and gates must not be relaxed.

### Ruled out (for this recipe, this parent, this budget)

- Insufficient explicit first-token CE, on this exact S1 data/order/loss,
  as a sufficient remedy for query-conditioned selection by +400.
- A claim that continued training on the new counterfactual data alone
  would have produced the success thresholds (control also failed them).
- A claim that treatment beat control on body-macro accuracy (bootstrap
  CI includes 0; point gain +0.002).

### Not ruled out

- Other selection objectives (contrastive / query-swap / margin) or
  randomized body order from
  [`design/V010_V2R4_FIRST_TOKEN_SELECTION_PROTOCOL.md`](../design/V010_V2R4_FIRST_TOKEN_SELECTION_PROTOCOL.md).
- Architecture or tokenizer change (forbidden here; untested).
- Longer than 400 updates **of a different preregistered experiment**.
- That a 2-byte-lost `data.py` freeze identity would have changed S1
  generation (generation was already frozen and hashed before this
  interrupt).

## Baby's resulting state

Authoritative Baby remains **v2R4 U16000 parent**
`checkpoint_16000.pt` SHA `94b3a9da…17827`.
S1 control/treatment U16200/U16400 checkpoints are local diagnostic
artifacts only. They are not a validated improvement. Do not promote.
Do not open TEST. v2R5/v2R6 local trees were not modified.

## Highest-information next action

Do **not** continue S1, raise the first-token CE weight, or launch seed
110002. Highest-information leftover is still a *new* preregistration if
the owner wants one: multi-pair query-swap / contrastive selection with
randomized body order, scoring first-token selection separately from
payload copy, with an explicit induction-retention guard (S1 treatment
regressed primitive induction). Read-only: 2-pair attention/logit dump
on isolation swap misses. v2R5 remains a separate lineage.

## Local-only artifacts

Checkpoints stay local (`.gitignore` `**/runs/`). Hashes:
[`research/V010_SELECTION_REPAIR_S1_SHA256SUMS.txt`](V010_SELECTION_REPAIR_S1_SHA256SUMS.txt).
Adjudication JSON: `runs/selection_s1/ADJUDICATION_110001_400.json`.
Protected material opened: false.
