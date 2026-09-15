# Baby v0.10 foundation v2R4 terminal adjudication

Status: **TERMINAL FAILURE — foundation gates not passed**

This report covers the authorized continuation of v2R4 seed `106001` from the
clean U6000 language-foundation checkpoint through the preregistered U16000
terminal checkpoint. No training or experiment was started after U16000.

## Provenance and execution

- Protocol: `BABY_V010_FOUNDATION_V2R4`
- Seed: `106001`
- Stages: U6000–U7999 `primitive_identity`; U8000–U9999 `short_retrieval`; U10000–U16000 `full_foundation`
- Device: CUDA on AMD Radeon RX 9060 XT
- Optimizer scope: all model parameters after U6000; lower scope LR `1.5e-5`, upper scope LR `3.75e-5`
- Language-retention probability: `0.20`
- Evaluations: 21 records at U6000, U6500, ..., U16000
- Process result: exit code `0`
- Protected material opened: `false`
- Chat transcript read or used: `false`

The authoritative parent was verified before launch:

```text
C:\DaveLM-v0.10\runs\diagnostic_v2r4_seed106001_8000\checkpoint_06000.pt
SHA-256 75d2c761f34a5716d3f35c2118b5d3446f8b63decf2fe5781b45888c164037b3
```

The continuation U6000 checkpoint was compared against that parent before the
first structured update; model-state keys matched and maximum model-tensor
absolute difference was `0.0`. The earlier post-U6000 continuation directory
was not loaded.

The preserved manual-chat transcript was not incorporated:

```text
C:\DaveLM-v0.10\chat_transcripts\BABY_U6000_CHAT_20260915.txt
```

## Terminal checkpoint and raw artifacts

- Terminal checkpoint: `runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt`
- Terminal checkpoint SHA-256: `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`
- Metrics: `runs/structured_v2r4_seed106001_from6000_terminal/metrics.jsonl`
- Metrics SHA-256: `b81cb5dad5084c30c95ca6b9b913e0c814125f6b511aec06cd75637aa345e74a`
- Run configuration SHA-256: `fb8d172c3cd0e7d705558cb6ed012da4acb67ec9dbcef687ab37fe2abd2fb6b7`
- Complete checkpoint hash list: `research/V010_FOUNDATION_V2R4_TERMINAL_SHA256SUMS.txt`

All 21 checkpoints and the raw metrics file remain preserved locally. They are
large generated run artifacts and remain excluded by the existing repository
ignore policy; the hash manifest and this report preserve their audit identity.

## Terminal measurements

The U16000 terminal panel contained 992 scored rows and no non-finite target
probabilities, logits, or margins.

| panel | n | first top-1 | teacher-forced exact | free exact | median rank |
|---|---:|---:|---:|---:|---:|
| primitive induction | 64 | 0.296875 | 0.218750 | 0.218750 | 6 |
| primitive keyed | 64 | 1.000000 | 0.968750 | 0.968750 | 1 |
| short keyed | 64 | 0.843750 | 0.843750 | 0.843750 | 1 |
| same-surface novel | 96 | 0.447917 | 0.427083 | 0.427083 | 2 |
| same-surface induction | 96 | 0.062500 | 0.052083 | 0.052083 | 69 |
| held-out surface | 96 | 0.343750 | 0.000000 | 0.000000 | 3 |
| unseen length | 64 | 0.296875 | 0.015625 | 0.015625 | 3 |
| low prior | 64 | 0.250000 | 0.000000 | 0.000000 | 3 |
| distractor | 64 | 0.125000 | 0.000000 | 0.000000 | 9 |
| broken context | 64 | 0.343750 | 0.000000 | 0.000000 | 6 |
| broken order | 64 | 0.015625 | 0.000000 | 0.000000 | 128 |

Language DEV CE was `1.2430044692`; the U6000 pre-structured baseline was
`1.1928689219`. A read-only train-stream check gave train CE `1.0128607638`
and train/DEV gap `0.2301437054`.

## Frozen gate adjudication

### Gate L — language health: PASS

- DEV CE `1.2430 <= 2.50`: pass.
- Train/DEV gap `0.2301 <= 1.00`: pass.
- All recorded metrics finite: pass.
- Immediate-EOS rows: `41/992` (`4.13%`), below the 10% limit.
- All-same-token emissions: `0/992`.

### Gate C — contextual reproduction: FAIL

- Novel sequence first-token top-1: `0.4479 < 0.90` — fail.
- Novel teacher-forced full-span exact: `0.4271 < 0.80` — fail.
- Held-out surface first-token top-1: `0.3438 < 0.80` — fail.
- Held-out surface free-running exact: `0.0000 < 0.70` — fail.
- Unseen-length first-token top-1: `0.2969 < 0.80` — fail.
- Low-prior first-token top-1: `0.2500 < 0.80` — fail.
- Distractor first-token top-1: `0.1250 < 0.75` — fail.
- Low-vs-high prior first-token gap: `0.09375 <= 0.10` — pass by the frozen regular-held-out versus low-prior proxy, but both absolute accuracies are below gate.
- Broken-context free-running exact: `0.0000 <= 0.05` — negative-control pass.
- Broken-order top-1: `0.015625`; relative to same-surface novel `0.447917`, ratio `0.0349 <= 0.20` — negative-control pass.
- Distinct emitted target tokens on novel: `327 >= 100` — pass.
- A separate positive changed-position/order panel was not present in the frozen v2 panel; the available broken-order negative control passes, but positive changed-position generalization is not demonstrated.

### Gate R — retention and mechanism: FAIL

- Language change from pre-structured baseline: `+0.0501355473 <= +0.50` — pass.
- Median intact-minus-broken-context target-logit lift: `3.7841949463 >= 1.0` — pass.
- Median intact target-vs-competitor margin: `-0.2518219948 <= 0` — fail.

The recorded mean context proxy was positive (`+2.9664` target-logit
difference and `+0.7912` margin difference), but the preregistered median
margin criterion is the stricter criterion and it failed.

Foundation graduation also requires the complete Gate L+C+R bundle on two
independent seeds. This single specified-checkpoint continuation does not
constitute replicated graduation, and no second seed was started after this
terminal result.

## What Baby v0.10 learned

Baby learned a narrow, highly effective keyed retrieval behavior on the
structured training surface:

- primitive keyed: `100%` first-token top-1 and `96.875%` full exact;
- short keyed: `84.375%` first-token and full exact;
- same-surface full novel: `44.792%` first-token and `42.708%` full exact.

This is real acquisition, not a CE-only claim. However, the primitive
induction panel remained weak (`29.688%` first-token top-1), and the broader
same-surface induction panel was nearly absent (`6.25%`). The intended
general reusable copy capability was not acquired.

## Generalization and shortcut evidence

The frozen panel generator and audit supplied disjoint train/held-out marker
families, language-derived sampled targets, variable lengths, varied wrappers,
and distractor/broken controls. No protected or final panel was opened. The
broken-order control was strongly suppressed, and broken-context free-running
exact was zero. These rule out a simple claim that the model succeeded equally
on every corrupted input.

They do not rule out the more important shortcut: the model can exploit the
structured keyed realization while failing to transport the rule across the
held-out surface. Held-out surface first-token accuracy (`34.375%`) was exactly
the broken-context first-token accuracy, and its full exact rate was zero. The
negative controls therefore do not rescue surface generalization or establish
causal context selectivity.

## Comparison with v0.9/T34

T34 showed that the existing v0.9 architecture contained a trainable substrate
for novel contextual copying when the objective directly required novel
reproduction: content, length, and prior-support generalization succeeded on
the trained surface, while changing surface markers/templates caused collapse.

v2R4 starts from fresh v0.10 weights and stages the structured objective after
language foundation training. It confirms a related but sharper result: fresh
Baby can acquire a keyed retrieval sub-capability while retaining language,
but the capability still does not generalize across surface realization,
induction, unseen length, low-prior targets, or distractors. The architecture
is therefore not the immediate bottleneck; reusable structural generalization
and context-selective training pressure remain the bottlenecks.

## Final capability statement

At this terminal checkpoint Baby v0.10 can:

- model the authorized language stream with healthy retention;
- solve many primitive and short keyed retrieval rows on familiar structured
  realizations;
- emit many distinct target tokens and suppress the broken-order control.

It cannot yet:

- copy arbitrary novel sequences at the frozen success rate;
- generalize the rule to held-out markers/templates;
- generalize reliably to unseen lengths or low-prior targets;
- resist distractors at the required rate;
- perform robust induction/repeated-pattern continuation;
- satisfy the full v0.10 foundation graduation bundle.

The next scientific bottleneck identified by this run is surface-invariant,
context-selective structural copying—especially induction and distractor
resistance—not ordinary language retention or basic parameter capacity.

No post-terminal experiment was started.
