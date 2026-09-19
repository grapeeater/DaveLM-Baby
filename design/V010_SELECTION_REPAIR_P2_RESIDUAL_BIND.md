# Selection repair P2: residual query-bind auxiliary

Protocol id: `V010_SELECTION_REPAIR_P2_RESIDUAL_BIND`
Status: **preregistration — frozen before schedule generation and before any
optimizer update**

Licensed by D2 **MIXED** (`research/V010_QUERY_PRESENCE_D2.md`). P1 stays
REGRESSION. D1 stays INVALID. D1b stays B. Authoritative parent unchanged.
No TEST. No sidecar. Not another attention λ. Not a T1 resume. Not a
continuation of the P1 treatment checkpoint.

## 1. Hypothesis

**H1.** P1 raised query attention (track 0.81) and only nicked residual identity
(cosine 0.9996 → 0.9972). Patch flip stayed 0.024. Greedy selection is still
reading a nearly query-invariant gen residual. An auxiliary that **pulls the
generation residual toward the query-token residual and away from other
in-context key residuals**, on natural-gap rows with gap ≥ 2, will make twins
distinguishable at gen_pos and raise long-gap candidate selection.

Matched control: same data, same CE, **λ = 0**.

## 2. Why this class

D2 MIXED, not WEIGHTS_ONLY and not A: attention moved, residual barely moved,
patch does not flip greedy. P1 already falsified “attend to the query.” The
missing write is in the residual the unembed sees. Start from hashed parent
U16000, not from a REGRESSION checkpoint.

## 3. Identity

| item | value |
|---|---|
| parent | U16000 SHA `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` |
| frozen eval | S2 `DIAGNOSTIC.json` |
| presence eval | D1b twins, identity only (no patch campaign) |
| train seed | `160001` (replicate `160002` generated, not launched unless P2 succeeds) |
| data seed | `160100` |
| bootstrap | `160300` |
| device | `cuda` / RX 9060 XT |
| train attention | default `sdpa` both arms |
| λ | treatment **0.25**, control **0.0** |
| τ | **0.10** |
| min gap for aux | **2** |

`BabyVNextLM.forward` is unchanged. Aux reads `forward_hidden`.

## 4. Dataset

Unmodified `data_v2.make_item`, natural gaps, all keyed variants. Batch: 10 full
keyed, 3 primitive keyed, 3 full induction. Language probability 0.20.

Denied: panels, S1, S2, D1b, and P1 training inputs (both schedule files).

## 5. Loss (frozen)

`L_CE` = answer-span CE, both arms.

For keyed row *i* with `gap = gen - query_pos ≥ 2`, let `h` be the final-norm
residual. Positions `P = [query_pos] + body_key_positions \ {query_pos}` (unique,
causal). Stop-gradient on every key vector. Positive is index 0 (`query_pos`):

`s_j = cosine(h[gen], sg(h[P_j])) / τ`

`L_bind = mean_i -log softmax(s)[0]`

`L = L_CE + λ L_bind`

Language and induction: `L_bind = 0`. Gap ≤ 1 keyed rows: `L_bind = 0` (do not
reinforce the working prev-token circuit). Optimizer: `capability_optimizer`.

## 6. Metrics

Primary: as-is candidate argmax, frozen S2, gap ≥ 13. Parent: 74/215, excess ~0.

Also: gap buckets, K, variant, body-macro, `query_logit_effect`, salience,
`rest_lock`, language DEV CE, frozen panels (same retention as T1/P1),
query-tracking on S2, and **D1b long-gap median twin residual cosine** (identity
only).

## 7. Decision rules — fixed before seeing results

| verdict | rule |
|---|---|
| **SUCCESS** | treatment long-gap excess ≥ **+0.10** and (T−C) ≥ **+0.07** with body-bootstrap 95% CI lower bound > 0; no retention regression; D1b long-gap median residual cosine ≤ **0.99** |
| **MECHANISM SUPPORTED, DOSE INSUFFICIENT** | D1b long-gap residual cosine ≤ **0.99** or drop vs parent ≥ **0.005**, SUCCESS selection bars not met; no retention regression |
| **FALSIFIED** | train `L_bind` fell ≥ 50% from updates 1–50 mean to last 50, and residual cosine still ≥ 0.999, and long-gap excess < +0.02 |
| **NULL** | neither selection nor residual cosine moved |
| **REGRESSION** | any retention gate below, regardless of `P` |

Bootstrap: 10,000 body resamples, seed `160300`.

### Futility

At 400, stop if **both** long-gap excess gain < **+0.02** and D1b long-gap
residual cosine still ≥ **0.999**. If the residual moved, continue to 800.

### Hard stops

Language DEV CE > parent + 0.20; non-finite loss/grad; disk < 10 GiB; wall > 2 h
per arm.

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

800 updates/arm, one seed, no λ sweep. Replication `160002` not launched here.
Promotion, TEST, tokenizer changes, merge to `main` out of scope until SUCCESS
plus replicate.

## 9. Frozen

These rules are not to be changed after results are seen.
