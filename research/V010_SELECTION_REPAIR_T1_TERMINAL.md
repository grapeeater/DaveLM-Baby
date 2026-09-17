# Selection repair T1 (query-transport gap curriculum): matched +400 terminal

Status: **NULL on the primary endpoint + REGRESSION on retention**

Isolated diagnostic fork of v2R4 U16000. Not foundation graduation. No merge to
`main`. Protected TEST was not opened. Frozen Gate L/C/R were not changed. S1 and
S2 receipts were not rewritten. The authoritative Baby is unchanged.

Branch `cursor/query-transport-range-t1` / [PR #4](https://github.com/grapeeater/DaveLM-Baby/pull/4)
vs `v0.10`. **Investigation paused per user request (2026-09-16).** No further
training, weight updates, or experiments until explicitly resumed.

Protocol (frozen at `8892545`, before data generation and before any optimizer
update): [`design/V010_SELECTION_REPAIR_T1_GAP.md`](../design/V010_SELECTION_REPAIR_T1_GAP.md)
Prior autopsy: [`research/V010_QUERY_TRANSPORT_AUTOPSY.md`](V010_QUERY_TRANSPORT_AUTOPSY.md)
Adjudication: `runs/selection_t1/ADJUDICATION_130001_400.json`

## Identity

| item | identity |
|---|---|
| parent | `runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt` |
| parent SHA-256 | `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` |
| seed | 130001 (replicate 130002 generated, **not launched**) |
| budget | 800 authorized, **stopped at 400 by the frozen futility rule** |
| extra loss terms | **none** — both arms use answer-span cross entropy only |
| device | `cuda` / AMD Radeon RX 9060 XT, `torch 2.9.1+rocmsdk20260116` |

Both arms consumed the **same items in the same order**. The treatment relocated
the template's own nuisance filler from after the query key into its `filler_a`
slot to hit the scheduled gap; the control kept the natural split. Per row the
two arms share token multiset, body, rendered pair order, markers, separator,
query key, and target. `eval_0000.json` is identical in both arms.

## What this tested

S1 pushed first-token gold CE. S2 pushed a paired query-contrast. Both moved a
logit statistic and neither produced greedy queried selection. The autopsy showed
why: the bind-and-copy circuit works but reaches about one token, and ~63% of the
diagnostic is outside that reach. T1 therefore changed **no objective at all** and
manipulated only the query-to-generation distance in training, starting where the
circuit already earns reward and growing outward.

## Results

Primary endpoint: candidate-restricted argmax on frozen diagnostic rows with
gap ≥ 13 (n=215).

| | parent | treatment +400 | control +400 |
|---|---:|---:|---:|
| **primary (gap ≥ 13)** | 0.3442 (**+0.000** excess) | **0.3488 (+0.005)** | **0.3488 (+0.005)** |
| gap 0–1 (n=158) | 0.4684 (+0.146) | **0.4937 (+0.171)** | 0.3797 (+0.057) |
| gap 2–3 | 0.3810 (+0.048) | 0.3333 (+0.000) | — |
| gap 4–12 | 0.3684 (+0.053) | 0.3158 (+0.000) | — |
| body-macro | 0.4115 | 0.4201 | 0.3848 |
| per-K 2/3/4 | .542/.354/.339 | .542/.396/.323 | .521/.368/.266 |
| language DEV CE | 1.2430 | 1.2431 | 1.2472 |
| salience spread | 2.776 | 3.297 | 3.513 |

Treatment minus control on the primary is **exactly 0.0** (75/215 in both arms).
Paired body bootstrap 10,000, seed 130300: 95% CI **[−0.0139, +0.0135]**.

Frozen panels, parent → treatment → control:

| panel | first top-1 | free exact |
|---|---|---|
| `same_surface_novel` | 0.4479 → **0.5208** → 0.4583 | 0.4271 → **0.5104** → 0.4375 |
| `heldout_surface` | 0.3438 → 0.4062 → 0.3750 | 0 → 0 → 0.0104 |
| `unseen_length` | 0.2969 → 0.3438 → 0.3594 | 0.0156 → 0.0156 → 0.0312 |
| `distractor` | 0.1250 → 0.1875 → 0.1719 | 0 → 0 → 0 |
| `primitive_keyed` | 1.000 → **1.000** → 1.000 | 0.9688 → **1.000** → 1.000 |
| `primitive_induction` | 0.2969 → **0.2031** → 0.1719 | 0.2188 → 0.1406 → 0.1094 |
| `short_keyed` | 0.8438 → 0.7812 → 0.7188 | 0.8438 → **0.7656** → 0.7031 |

Post-hoc transport-range probe (`runs/query_locality_t1/QUERY_LOCALITY.json`):

| distance of query from generation position | parent | T1 treatment | T1 control |
|---|---:|---:|---:|
| 0 | 0.607 | **0.648** | 0.514 |
| 1 | 0.433 | **0.463** | 0.394 |
| 2 | 0.340 | 0.340 | 0.322 |
| 4 | 0.340 | 0.326 | 0.350 |
| 8 | 0.368 | 0.350 | 0.331 |
| `append_other_key` (follow a swapped adjacent key) | 0.556 | **0.620** | 0.507 |
| `append_unused_key` (negative control) | 0.336 | 0.354 | 0.322 |

**The curriculum raised the amplitude inside the existing reach and widened the
reach by exactly zero tokens.**

## Adjudication

`futility_fired_at_400 = true`. Long-gap excess gain over its own baseline was
+0.0047 against the frozen bar of +0.02, so the run stopped at 400 as required.

Primary verdict **NULL**: treatment excess +0.005 < +0.02, and the gap 0–1 excess
rose only +0.025, short of the +0.10 that the frozen table requires before the
stronger `FALSIFIED` label may be used.

Overall verdict **REGRESSION**, on two frozen retention gates:

- `primitive_induction` first top-1 0.2969 → 0.2031 (bar 0.05)
- `short_keyed` free exact 0.8438 → 0.7656 (bar 0.05)

Both also regressed in the **control**, and further (0.1719 and 0.7031), so this
is a cost of 400 keyed-heavy updates on this parent rather than a treatment
effect. `primitive_keyed` held at 1.000 in both arms: the primitive-keyed
retention stream added up front did its job, and the S2 retention failure did not
recur there.

Negative controls all pass: `broken_context` free exact 0.000; `broken_order`
first top-1 0.0156; `append_unused_key` excess +0.021 (bar +0.05), so the model
did **not** learn "attend to the last token"; `append_other_key` 0.620, so the
adjacent circuit was strengthened rather than broken.

## Protocol error to record honestly

The futility gate was placed at update 400, but the curriculum's long-gap blocks
do not begin until update 301 (gap ≤ 8), 451 (gap ≤ 20) and 601 (natural). At the
moment futility was evaluated the treatment had trained almost exclusively on
gaps 0–3. So T1 as executed tested **"does training at gaps 0–3 transfer to
gap ≥ 13?"** and answered *no*. It did **not** test the full schedule. The rule
was not changed after seeing results, and it should not have been written that
way. A successor protocol must place futility after the schedule's long-gap
blocks.

## What T1 taught

1. **Transfer from short gaps to long gaps is zero, not merely weak.** Treatment
   and control land on the identical 75/215. Even the gap 2–3 and gap 4–12
   evaluation strata fell to *exact* chance while gaps 0–3 were being trained.
2. **Three intervention classes now show the same signature.** S1 (first-token
   gold CE), S2 (paired query contrast), and T1 (gap curriculum, no loss change)
   each raise performance inside the one-token reach and move the ≥2-token regime
   by nothing. An objective-level intervention and a data-level intervention both
   failed the same way.
3. **The gap curriculum is nonetheless the best intervention recorded in this
   line.** Against a perfectly matched control it is better on every capability
   measure: gap 0–1 excess +0.171 vs +0.057, `same_surface_novel` free exact
   0.5104 vs 0.4375, body-macro 0.4201 vs 0.3848, K=4 0.323 vs 0.266, and it
   costs less retention. It is simply orthogonal to the bottleneck.
4. **The query-invariant salience prior grew in both arms** (2.78 → 3.30
   treatment, 3.51 control). More keyed training on rows that are unreachable as
   posed deepens the favourite-inventory-head prior, exactly as the autopsy
   predicted. This was a preregistered *secondary* prediction and it failed in the
   predicted-wrong direction for the treatment too.

## Ruled out / not ruled out

Ruled out, this parent, this budget:

- "Scaffolding at gaps 0–3 for 400 updates extends the transport range." No.
- "Any improvement on `same_surface_novel` implies selection improved." No: the
  panel moved +0.07 while long-gap selection moved 0.000.
- "The model will learn an attend-to-the-last-token shortcut if trained at gap 0."
  No: `append_unused_key` stayed flat.
- "A primitive-keyed retention stream cannot protect 1-pair copy." It did.

Not ruled out:

- The **full** T1 schedule (gaps up to 20 and natural, updates 401–800), which
  futility cut off.
- A much larger dose, or a curriculum that ramps the gap far more slowly.
- An architectural or representational limit on query transport. This is now the
  leading alternative and it has been *earned* rather than assumed: the one-token
  cliff reproduces on eight checkpoints across four curricula, five seeds and
  three objectives, and survives a direct curricular attack.

## Baby's resulting state

Authoritative Baby remains **v2R4 U16000**
`checkpoint_16000.pt` SHA `94b3a9da…17827`. T1 U16200/U16400 checkpoints are
local diagnostic artifacts. Do not promote. Do not open TEST. Do not move
Gate L/C/R.

## Highest-information next action

Do **not** launch T2 as another curriculum variant, and do not launch seed
130002.

Run a **read-only attention/composition localization** on the hashed parent:
at the generation position, does *any* head at *any* layer attend to the query
key's position when the gap exceeds one token, and is the query key's identity
linearly decodable from the residual stream at the generation position as a
function of gap? This costs no training and it splits the two survivors cleanly:

- Query identity **is** present at the generation position but unused ⇒ the
  defect is downstream composition, and an objective that conditions on it is
  worth designing.
- Query identity is **absent** beyond gap 1 ⇒ the transport pathway does not
  exist, the limit is representational, and an architecture claim (or a far
  longer, far more gradual curriculum) becomes justified for the first time.

Only after that should any further weight-updating protocol be frozen.

## Local-only artifacts

Checkpoints and per-row dumps stay local.
Hashes: [`V010_SELECTION_REPAIR_T1_SHA256SUMS.txt`](V010_SELECTION_REPAIR_T1_SHA256SUMS.txt).
Protected material opened: false.
