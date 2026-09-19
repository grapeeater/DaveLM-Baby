# D2 query-presence on P1 checkpoints: MIXED

Status: **MIXED**. Treatment is valid D1b-MIXED. Control is valid D1b-**B**.
Authoritative Baby unchanged: v2R4 U16000
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.
D1 stays INVALID. D1b stays B. P1 stays REGRESSION. No TEST. No promotion.

Protocol:
[`design/V010_QUERY_PRESENCE_D2.md`](../design/V010_QUERY_PRESENCE_D2.md)
`19787a118b4690f31a689fff9619ece0458759b209829267ca2efb2aca90466a`

Adjudication: `runs/query_presence_d2/ADJUDICATION.json`

## Why MIXED, not WEIGHTS_ONLY or RESIDUAL_WRITE

Leading prediction was WEIGHTS_ONLY (treatment COMPOSITION, control B).
COMPOSITION requires long-gap residual cosine ≥ **0.999**. Treatment is
**0.9972**. A requires final flip ≥ **0.20**. Treatment is **0.024**. Tracking
is 0.814, so it is not B either.

The pointer moved the residual a little and the attention a lot. It did not
make patched candidate argmax follow the donor.

## Per-arm instrument (passed)

| check | treatment | control |
|---|---:|---:|
| checkpoint SHA | match `5c8296ef…4fbafa` | match `3f005e19…d2fc1` |
| D1b diagnostic SHA | match | match |
| SDPA vs reference | 8.6e-6 | 7.4e-6 |
| eligible short-gap flip (n) | 0.341 (270) | 0.322 (267) |
| short-gap residual cosine | 0.946 | 0.958 |
| prev-track | 1.0 | 1.0 |

## Long gap (n=510 pairs / 236 rows)

| metric | D1b parent | P1 control | P1 treatment |
|---|---:|---:|---:|
| residual cosine | 0.99956 | **0.99962** | **0.99720** |
| logit cosine | 0.99994 | 0.99995 | 0.99957 |
| `flip_toward_donor` at final | 0.0078 | 0.0020 | 0.0235 |
| g31 changed | 0.022 | 0.033 | 0.027 |
| query-track fraction | 0.225 | 0.233 | **0.814** |
| median max query mass | 0.033 | 0.033 | **0.494** |
| prev-token mass | 0.983 | 0.983 | 0.989 |
| D1b verdict | B | **B** | **MIXED** |

Control matches the parent. The residual drop and the tracking jump are
treatment-only.

At gap ≥ 31, treatment residual cosine is 0.9974 and same-argmax 0.973. The
unembed is still rank 640/640.

## What this says

H_WEIGHTS is false as frozen: residual is no longer inside the COMPOSITION
band. H_RESIDUAL is false: patch still does not flip greedy selection.
H_STILL_B is false: tracking transferred to the D1b twins.

The remaining mechanism is **a weak residual write**. Query identity at gen_pos
is slightly twin-dependent after the pointer, and still unused by candidate
argmax. Prev-token mass remains ~0.99, so the copy head was not displaced.

## What this does not license

Another λ, T1 resume, sidecar, TEST, or promoting `checkpoint_16800.pt`.
Next licensed question, if any, is how query identity in the residual fails to
reach the candidate decoder — not more attention.
