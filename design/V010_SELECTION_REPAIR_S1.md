# Selection repair S1: preregistration

Frozen before new checkpoint scoring/training. Parent branch commit 9c9317b.
This is an isolated diagnostic fork of v2R4, not a replacement of v2R5/v2R6.
User authorized investigation, implementation and justified training on 2026-09-16.

## Evidence and hypothesis

Raw v2R4 U16000 metrics independently reproduce first-token 43/96 and
gold remainder rank-one 93/96 on same-surface novel. Primitive keyed is 64/64
one-pair. The isolation report's all-K chance accuracy does not itself prove
query invariance; paired logit responses are required. Markers do not explain
the familiar-surface failure. Separators are a separate OOD problem.

Hypothesis: first-token selection receives insufficient effective optimization
relative to already-solved continuation. Adding first-token CE to unchanged
full-answer CE, on balanced counterfactual queries, improves causal selection
more than an equal-compute full-answer-only control. This tests a specific
optimization remedy, not a unique root cause or architecture incapacity.

## Fixed design

Parent: runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt
SHA256 94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827.
Both arms reset AdamW moments identically; retain inherited layerwise learning
rates 1.5e-5/3.75e-5, betas .9/.999, eps 1e-8, decay .05, clipping 2.
Unchanged architecture, tokenizer and dropout. Seed 110001, batch 16.
800 updates maximum, checkpoints/evaluation 0/200/400/600/800. No checkpoint
selection. If successful, repeat both arms with seed 110002 under this same rule.
20% language updates, otherwise 75% keyed / 25% induction, full difficulty,
20% low-prior probability. All surfaces remain inherited training surfaces.
For keyed data, sample a body with distinct candidate first tokens; render
every possible query with body/filler/wrappers unchanged, then draw queries
uniformly in complete counterfactual groups. Pack batch by pairs of two
uniformly sampled distinct queries from each body. No one-pair training.
Control: inherited full-answer token-mean CE. Treatment: same loss plus
1.0 * mean first-answer-token vocabulary CE on keyed rows only.
Induction CE and language CE unchanged. Same frozen examples/order across arms.

## Data and measurement

Data seed 110100, independent diagnostic panel seed 110200. Pre-generate
800 batches per seed with identical task schedule across arms. Exclude exact
inputs and all candidate value spans present in frozen v2 panels or new
diagnostic panels from training. A body and all its counterfactual queries
belong to only one partition. Save/hash all inputs, batches, protocol, code.

New development diagnostic: 48 independently sampled bodies per K=2,3,4,
all possible queries, full difficulty, familiar surfaces. Report per-K and
body-macro first-token accuracy, all-query body success, inventory accuracy,
rank/margin, gold continuation lock, value exact/full exact; paired signed
target-vs-alternative query logit change. A query-independent output cannot
exceed 1/K body accuracy. Also score original frozen panels and isolation query
swap, marker/separator swaps, reordered bodies and value-absent controls.
Language DEV CE uses inherited first 32 fixed 256-token windows. All reports
include baseline and matched terminal. No protected material is used.

## Fixed adjudication

Hard stop > regression > success > partial > failure.
Hard stop: parent/data/code hash mismatch, overlap, malformed/ambiguous target,
missing panel, nonfinite loss/gradient/metric, available disk < 10 GiB,
language CE increase > .20, or runtime > 2 hours per arm. Preserve failed state.
Regression: terminal DEV CE increase > .10; any original intact panel first
top1 or free exact loss > .05; new gold-remainder lock loss > .05; isolation
value-absent exact > .05 or frozen broken-context/order free exact > .05.
Success requires no regression; new diagnostic body-macro accuracy >= .70
and gain >= .20 over parent and >= .15 over matched control; each K above
chance + .15; original query-swap first accuracy gain >= .15; same-surface
free exact gain >= .15; signed query logit response gain >= .50. For primary
accuracy gain vs control, paired body bootstrap 10,000 replicates seed 110300
must have 95% lower bound > 0. Success must replicate before promoting a
checkpoint as a validated improvement. This is not foundation graduation.
Partial: no regression and new accuracy gains >= .10 over parent and >= .05
over control but not full success. Otherwise failure.
Futility at U+400: both arms' new body-macro accuracy gain < .05 and treatment
mean target-vs-winner margin gain < .25. Complete matched U+400 measurements,
stop both arms, autopsy. Do not extend this experiment's budget or relax gates.

## Provenance limitation

Later v2R5/v2R6 run directories and metrics exist locally, but their code and
reports are absent from the current Git branch. Run receipts establish distinct
data/objectives and parents. They are not treated as nonexistent or merged into
v2R4. The unfinished LPX dose preflight has audit_all_pass=false and no updates;
it is not a trained candidate. Global best-checkpoint claims require compatible
matched evaluation, not comparison of different panel headline scores.
