# D3b L0 control decomposition: SPLICED_IDENTITY

Status: **SPLICED_IDENTITY**. D3 stays MIXED. No TEST. P2 not launched.
Authoritative Baby unchanged: v2R4 U16000
`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`.

Protocol:
[`design/V010_QUERY_SPLICE_D3B.md`](../design/V010_QUERY_SPLICE_D3B.md)
Adjudication: `runs/query_splice_d3b/ADJUDICATION.json`

## Replication (passed)

Full D3 long-gap set: as-is **74/215**, Q0R **127/215**. Shared support n=215
(every long-gap row had a filler position).

## Shared-support long-gap

| cell | gold | spliced competitor |
|---|---:|---:|
| as-is | 0.344 | 0.372 |
| Q0R (query → gen at L0) | **0.591** (Δ **+0.247**) | — |
| C0R (competitor key → gen at L0) | **0.177** (Δ **−0.167**) | **0.647** (Δ **+0.274**) |
| R0R (filler → gen at L0) | 0.340 (Δ **−0.005**) | — |

Filler overwrite does not move gold. Competitor identity at L0 makes Baby
select **that competitor's value** and hurts gold. Query identity at L0 makes
her select the queried value.

So `h[gen]` at block 0 is the retrieval query: later layers implement
QUERY → MATCHING KEY → VALUE once that identity is present. P1's layer-10
pointer and P2's final-residual bind are the wrong site.

## License

Early residual write of **query identity** into `gen_pos` at block 0–2,
versus a matched λ=0 control. Not P1-all-layers. Not P2-as-written.
