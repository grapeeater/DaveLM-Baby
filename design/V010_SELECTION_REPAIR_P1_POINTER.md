# Selection repair P1: pointer attention auxiliary

Protocol id: `V010_SELECTION_REPAIR_P1_POINTER`
Status: **preregistration — frozen before schedule generation and before any
optimizer update**

Licensed by D1b **valid B** (`research/V010_QUERY_PRESENCE_D1B.md`). D1 remains
INVALID and is not reopened. This is not T1 with a longer budget, not S1/S2
with a new λ, and not a sidecar. Gate L/C/R unchanged. No TEST. Authoritative
parent unchanged until a later protocol says otherwise.

## 1. Hypothesis

**H1.** At gap > 1 the generation-position residual is query-invariant because
no head reads the query token from there. Adding an auxiliary loss that raises
**max-head, max-layer** attention mass on the query token from the generation
position, on **natural-gap** keyed rows, will write query identity into that
residual and raise as-is candidate selection at gap ≥ 13.

The matched control is the same data, same CE, same backend, **λ = 0**.

## 2. Why this class, and not another gap curriculum

D1b B: long-gap residual cosine 0.99956, g31 changed 0.022, prev-token mass
0.98, query-slot mass 0.033. T1 already falsified "scaffold gap 0–1 CE then
widen": long-gap excess treatment−control = 0.0 at futility. The missing
operation is a **read of a non-adjacent query**, not more reward at the range
that already works.

## 3. Identity

| item | value |
|---|---|
| parent | `checkpoint_16000.pt` SHA `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827` |
| frozen eval | `runs/selection_s2/DIAGNOSTIC.json` (hash vs S2 manifest) |
| D1b diagnostic | denied as a prior eval set; not trained on |
| train seed | `150001` (replicate `150002` generated, **not launched** unless P1 succeeds) |
| data seed | `150100` |
| bootstrap seed | `150300` |
| device | `cuda` / RX 9060 XT |
| train attention | `reference` in **both** arms (aux and CE share one graph) |
| eval attention | default `sdpa`, as in S1/S2/T1 |
| λ | treatment **0.25**, control **0.0** |

`BabyVNextLM.forward` is unchanged when attention capture is off. Capture is a
training-only side channel used by this protocol.

## 4. Dataset

Unmodified `data_v2.make_item` keyed/induction, **natural gaps**, all keyed
variants. Every structured update is 16 rows: 10 full keyed, 3 primitive keyed,
3 full induction. Language-retention probability 0.20, as in v2R4/T1.

Denied: frozen panels, S1 diagnostic, S2 diagnostic, D1b diagnostic. No exact
input or candidate-span leakage into eval.

## 5. Loss (frozen)

Answer-span CE is identical in both arms:

`L_CE = cross_entropy(logits[mask], targets[mask])`

Pointer term, **keyed rows only** (full and primitive), with
`gen = len(input)-1` and `q = query_position`:

`m_i = max_{layer l, head h} attn_{l,h}[gen, q]`

`L_ptr = mean_i -log(m_i + 1e-8)`

`L = L_CE + λ L_ptr`

Max over heads and layers is deliberate: do not force the working prev-token
copy head to abandon t−1. One pointer write is the B mechanism. Pre-dropout
softmax masses are the aux target; dropout still applies to the value mix.

Language batches: `L_ptr = 0`. Induction rows: `L_ptr = 0`.

Optimizer, clip, LRs: `train_v2r4.capability_optimizer` unchanged.

## 6. Metrics

Primary: as-is candidate-restricted argmax on frozen S2 diagnostic rows with
gap ≥ 13. Parent from T1: accuracy **0.3442**, chance **0.3391**, excess
**+0.0051** (n=215).

Also reported, never collapsed:

- gap buckets `{0–1, 2–3, 4–12, 13–30, 31+}`, K, variant, value length
- body-macro, `query_logit_effect`, salience spreads, `rest_lock`
- language DEV CE and frozen panels (same retention set as T1)
- **mechanism:** long-gap fraction of rows with any query-tracking head
  (mass ≥ 5× uniform) and median max-head query mass at gen, measured with
  reference attention on the frozen diagnostic

## 7. Decision rules — fixed before seeing results

| verdict at terminal (800, or 400 if futile) | rule |
|---|---|
| **SUCCESS** | treatment long-gap excess ≥ **+0.10** and (treatment−control) ≥ **+0.07** with body-bootstrap 95% CI lower bound > 0; no retention regression; long-gap query-tracking-head fraction ≥ **0.50** |
| **MECHANISM SUPPORTED, DOSE INSUFFICIENT** | tracking fraction gain vs parent ≥ **+0.20** or long-gap median max query mass ≥ **0.15**, but SUCCESS selection bars not met; no retention regression |
| **FALSIFIED** | train `L_ptr` fell ≥ 50% from update 1–50 mean to the last 50, and long-gap tracking gain < +0.05, and long-gap excess < +0.02 |
| **NULL** | neither selection nor tracking moved |
| **REGRESSION** | any retention gate below, regardless of `P` |

Bootstrap: 10,000 resamples of diagnostic **bodies**, seed `150300`.

### Futility

At update 400, stop if **both**:

- treatment long-gap excess gain vs its own update-0 < **+0.02**
- treatment long-gap query-tracking-head fraction gain vs update-0 < **+0.10**

If tracking moved and selection did not, continue to 800.

### Hard stops

Language DEV CE > parent + 0.20; non-finite loss or gradient; free disk < 10 GiB;
wall clock > 2 h per arm.

### Retention — regression if any drops more than 0.05 vs parent

| metric | parent |
|---|---:|
| language DEV CE | 1.2430 (regression flag at +0.05) |
| `primitive_induction` first top-1 | 0.2969 |
| `primitive_keyed` first top-1 | 1.0000 |
| `short_keyed` free exact | 0.8438 |
| gold `rest_lock` | 0.9769 |

### Negative controls — must hold

- `value_absent` exact copy of the original span stays **0**
- `broken_context` free exact ≤ 0.05
- `broken_order` top-1 ≤ 0.20 of intact accuracy
- `append_unused_key` excess over chance ≤ +0.05 (post-hoc; if this rises, the
  aux taught "attend somewhere" rather than the query)

### What would falsify H1

`L_ptr` is optimized (train pointer mass rises) and long-gap diagnostic tracking
stays at ~0.22. Then the aux was a train-distribution pointer, not a query
transport, and the next claim is architectural with that evidence.

## 8. Budget

Maximum 800 updates per arm, two arms, one seed. No λ sweep. No second
treatment without new evidence. Replication `150002` is required before any
promotion claim and is **not** launched by this protocol. Promotion, TEST,
tokenizer changes, and merge to `main` are out of scope.

## 9. Frozen

These rules are not to be changed after results are seen.
