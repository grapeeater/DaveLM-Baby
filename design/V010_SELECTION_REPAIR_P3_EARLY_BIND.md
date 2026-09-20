# Selection repair P3: early residual query-bind

Protocol id: `V010_SELECTION_REPAIR_P3_EARLY_BIND`
Status: **preregistration — frozen before schedule generation and before any
optimizer update**

Licensed by D3b **SPLICED_IDENTITY**. D3 stays MIXED. P1 stays REGRESSION.
P2 stays preregistered and is **not launched**. Authoritative parent
unchanged. No TEST. Not a P1 λ bump. Not a T1 resume. Not final-residual P2.

## 1. Hypothesis

**H1.** D3b showed that replacing `block_0[gen]` with `block_0[query]` raises
long-gap gold 74→127/215, replacing it with a competitor key residual makes
Baby emit **that competitor's** value, and replacing it with filler does
nothing. An InfoNCE auxiliary on **block-0 residuals** that pulls `h0[gen]`
toward `h0[query]` and away from **competitor-key** residuals, on natural-gap
rows with gap ≥ 2, will install that write and raise long-gap candidate
selection.

Matched control: same data, same CE, **λ = 0**.

## 2. Why this class

P1 max-over-all-layers attention installed a **layer-10** pointer; selection
did not move. P2 would bind the **final** residual, the site D3 K11R showed
can be anti-gold. D3b isolates the sufficient write at **L0 residual
identity**, not attention mass and not unembed.

`BabyVNextLM.forward` is unchanged. Aux reads a block-0 forward hook.

## 3. Identity

| item | value |
|---|---|
| parent | U16000 SHA `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` |
| frozen eval | S2 `DIAGNOSTIC.json` |
| train seed | `170001` (replicate `170002` generated, not launched unless P3 succeeds) |
| data seed | `170100` |
| bootstrap | `170300` |
| device | `cuda` |
| train attention | `sdpa` both arms |
| λ | treatment **0.25**, control **0.0** |
| τ | **0.10** |
| min gap for aux | **2** |
| bind site | `block_out` layer **0** only |

Start from the hashed parent, not from P1/P2 checkpoints.

## 4. Dataset

Unmodified `data_v2.make_item`, natural gaps. Batch: 10 full keyed, 3
primitive keyed, 3 full induction. Language probability 0.20.

Denied: panels, S1, S2, D1b, P1 training inputs (both schedule files).

## 5. Loss (frozen)

`L_CE` = answer-span CE, both arms.

For keyed row *i* with gap ≥ 2, let `h0` be **block 0 output** (after that
block's attention and MLP). Positions `P = [query_pos] + competitor key
starts` whose displayed or original key ≠ `query_key`. Stop-gradient on every
key vector. Positive is index 0:

`s_j = cosine(h0[gen], sg(h0[P_j])) / τ`

`L_bind = mean_i -log softmax(s)[0]`

`L = L_CE + λ L_bind`

Same-token matching-key positions are **not** negatives (D3b: query or match
key identity both retrieve the gold value). Language and induction: `L_bind =
0`. Gap ≤ 1: `L_bind = 0`. Optimizer: `capability_optimizer`.

## 6. Metrics

Primary: as-is candidate argmax, frozen S2, gap ≥ 13. Parent: 74/215, excess
0.

Also: gap buckets, K, body-macro, `query_logit_effect`, salience, `rest_lock`,
language DEV CE, frozen panels (P1 retention set), long-gap **median
cosine(h0[gen], h0[query])**, long-gap query-tracking on layers **0–2 only**,
query-swap follow, `value_absent`.

## 7. Decision rules — fixed before seeing results

| verdict | rule |
|---|---|
| **SUCCESS** | treatment long-gap excess ≥ **+0.10** and (T−C) ≥ **+0.07** with body-bootstrap 95% CI lower bound > 0; no retention regression; and (L0 query-gen cosine gain vs parent ≥ **+0.10** **or** L0–2 query-track ≥ **0.50**) |
| **MECHANISM SUPPORTED, DOSE INSUFFICIENT** | L0 cosine gain ≥ **+0.05** or L0–2 track gain ≥ **+0.20**, SUCCESS selection bars not met; no retention regression |
| **FALSIFIED** | train `L_bind` fell ≥ 50% from updates 1–50 mean to last 50, and L0 cosine gain < **+0.02**, and long-gap excess < +0.02 |
| **NULL** | neither selection nor L0 cosine/track moved |
| **REGRESSION** | any retention gate below, regardless of `P` |

Bootstrap: 10,000 body resamples, seed `170300`.

### Futility

At 400, stop if **both** long-gap excess gain < **+0.02** and L0 query-gen
cosine gain < **+0.02**. If the L0 write moved, continue to 800.

### Hard stops

Language DEV CE > parent + 0.20; non-finite loss/grad; disk < 10 GiB; wall
> 2 h per arm.

### Retention — drop > 0.05 vs parent is regression

| metric | parent |
|---|---:|
| language DEV CE | 1.2430 (flag at +0.05) |
| `primitive_induction` first-top1 | 0.2969 |
| `primitive_keyed` first-top1 | 1.0000 |
| `short_keyed` free exact | 0.8438 |
| gold `rest_lock` | 0.9769 |

### Negative controls

`value_absent` exact 0; `broken_context` free exact ≤ 0.05; `broken_order`
top-1 ≤ 0.20 of intact.

## 8. Budget

800 updates/arm, one seed, no λ sweep. Replication `170002` not launched
here. Promotion, TEST, sidecar, merge to `main` out of scope until SUCCESS
plus replicate.

## 9. Frozen

These rules are not to be changed after results are seen.
