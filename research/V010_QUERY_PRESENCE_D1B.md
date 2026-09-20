# D1b Query Presence Localization: valid B

Status: **valid B**. D1 remains INVALID and is not reopened. No checkpoint
promoted. Protected TEST was not opened. Gate L/C/R were not moved.

Protocol (hashed in the run preflight **before** scoring):
[`design/V010_QUERY_PRESENCE_D1B.md`](../design/V010_QUERY_PRESENCE_D1B.md)
`4b889dd0fb64988a3f0b9f06b6888bf3013697e0e00c50ad036252318e2953fe`

Adjudication: `runs/query_presence_d1b/ADJUDICATION.json`
Summary: `runs/query_presence_d1b/QUERY_PRESENCE.json`
Diagnostic seed `140200`, SHA `b7260e6761bbedecacc6f26801b54141d6539b9ecfe76ff72c34c2e32ce76c57`

Authoritative Baby unchanged: v2R4 U16000
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`

## Why D1b is not a moved D1 bar

D1's all-pair gap≤1 `flip_toward_donor` was 0.243 on 362 **seen** pairs (bar 0.25).
D1b does not rescore those pairs. It generated 144 new bodies, denied panels /
S1 / S2 / D1's scored inputs, and froze an eligible-flip control (`before !=
donor_gold`, n≥50, still ≥0.25) before generation.

## Preflight (passed)

| check | result |
|---|---|
| parent SHA | match |
| diagnostic SHA | match `b7260e67…76c57` |
| twins differ only at `query_position` | true |
| protected flag | false |
| SDPA vs reference max-abs | **8.1e-6** (bar 1e-3) |
| `W_U` rank | **640 / 640**, injective |
| D1 reopened | false |
| update | 16000 |

## Frozen instrument validity

| metric | value | bar | pass |
|---|---:|---:|---|
| gap≤1 median final residual cosine | 0.958 | ≤ 0.99 | yes |
| gap≤1 prev-tracking-head fraction | 1.0 | ≥ 0.90 | yes |
| eligible `flip_toward_donor` (best block L5) | **0.294** | ≥ 0.25 | yes |
| eligible n | **265** | ≥ 50 | yes |

Raw (all-pair) short-gap flip was 0.225. That number is reported and **is not a
gate**. Using it as a gate would have been a second D1 miss; that is why D1b
froze eligible-flip.

## Long-gap adjudication → B

| metric | value | B bar |
|---|---:|---|
| median final residual cosine | 0.99956 | ≥ 0.999 |
| median full-vocab logit cosine | 0.99994 | ≥ 0.995 |
| `flip_toward_donor` at `final` | 0.0078 | ≤ 0.05 |
| max `block_out` / `attn_out` flip | 0.0078 | ≤ 0.05 |
| gap≥31 `changed` at `final` | 0.022 | ≤ 0.05 |
| query-tracking-head fraction | 0.225 | COMPOSITION would need ≥ 0.50 |

A is false (final flip 0.0078 < 0.20). COMPOSITION is false (tracking 0.225).
All B inequalities hold. **Verdict: B.**

## What B says

On fresh twins, when the query is not the previous token, the generation-position
residual is query-invariant. Patching the other twin's gen residual does not
flip greedy candidate choice. A near-hard prev-token head (mass ≈ 0.98 in every
stratum) copies the query when the query is t−1 and copies filler otherwise.

| stratum | n | logit cosine | residual cosine | prev mass | query mass | query-track |
|---|---:|---:|---:|---:|---:|---:|
| gap ≤ 1 | 150 rows / 342 pairs | 0.994 | 0.958 | 0.983 | 0.980 | 1.00 |
| query deeper | 282 / 618 | 0.99992 | 0.99942 | 0.983 | 0.046 | 0.34 |
| gap ≥ 13 | 236 / 510 | 0.99994 | 0.99956 | 0.983 | 0.033 | 0.22 |
| gap ≥ 31 | 163 / 364 | 0.99996 | 0.99970 | 0.983 | 0.025 | 0.13 |

Identity skips were not used to fake a no-op: g31 `changed` is 0.022, not zero,
and still under the B bar. Embeddings of the query token at gen_pos remain
near-orthogonal (median cos ≈ 0).

## What this licenses

A treatment that **writes query identity into the generation residual from a
non-adjacent query token**. The leading class is a pointer-attention auxiliary
on natural-gap keyed rows, with a matched λ=0 control. T1 already showed that
short-gap CE does not create skip-2, so P1 must not be another gap curriculum.

S1/S2/T1 remain diagnostic failures. Sidecar, TEST, and checkpoint promotion
remain closed until a later protocol says otherwise.
