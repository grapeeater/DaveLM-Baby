# Baby v0.10 v2R4 isolation probe (Fan Diesel, weights)

Status: **read-only checkpoint diagnostic**  
Protocol: `BABY_V010_V2R4_ISOLATION_DIAGNOSTIC`  
Checkpoint: `runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt`  
SHA-256: `94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`  
Device: `cuda` (ROCm, AMD Radeon RX 9060 XT)  
Trained / optimizer updated: **false**  
Gates weakened: **false**  
Frozen panels mutated: **false**  
v2R5: unresolved / not opened by this probe  

Machine-readable: `runs/v2r4_isolation_probe/PROBE.json`  
Row dumps: `QUERY_SWAP_ROWS.json`, `ISOLATION_ROWS.json`

This does not rewrite Gate L/C/R or the terminal report.

## Highest-priority question

Does changing **only the query** cause Baby's first-token selection to follow the newly queried value?

**No.** Query-swap follow is at chance at every pair count. New-gold teacher-forced continuation stays locked.

| panel | n | follow new | stuck old | other competitor | off | inventory copy | new-gold rest_value_tf_lock |
|---|---:|---:|---:|---:|---:|---:|---:|
| query_swap same-surface novel | 96 | 36 | 33 | 23 | 4 | 92/96 | 94/96 |
|  2-pair | 31 | 16 | 15 | 0 | 0 | 31/31 | 31/31 |
|  3-pair | 21 | 8 | 6 | 7 | 0 | 21/21 | 21/21 |
|  4-pair | 44 | 12 | 12 | 16 | 4 | 40/44 | 42/44 |
| query_swap short 2-pair | 28 | 12 | 16 | 0 | 0 | 28/28 | 20/21 |

Two-sided exact binomial vs 1/K: 2-pair p=1.000; 3-pair p=0.818; 4-pair p=0.862; short 2-pair p=0.572.

When she does not follow, the new queried first token is **never** greedy rank-1 on novel (0/60). It is often still competitive: 30/60 runner-up, 30/60 present-but-not-competitive (mostly 4-pair). Short 2-pair misses are almost all original-as-rank-1 / queried-as-runner-up (15/16). Median queried vocab rank after swap is **2**.

Mechanism chain on query-swap novel: QUERY does not bind → FIRST TOKEN picks some in-context value head (36 new / 33 old / 23 other / 4 off) → PAYLOAD CONTINUATION of whatever was selected, and TF continuation of the *new gold* still locks → SEPARATOR/EOS succeeds on the train surface when the first token was the new gold (36/36 copies_and_closes).

## Other already-designed probes

Parent novel queried copy remains 41/96 (inventory 93/96). Probe pair-count slices reproduce the frozen census: 19/31, 7/21, 15/44; none above 1/K at p<0.05. Short 1-pair 36/36; short 2-pair 18/28 p=0.185.

| transform | value_ok | free_exact | inventory copy | vs matched parent |
|---|---:|---:|---:|---|
| marker-swap, keep train seps | 47/96 | 47/96 | 90/96 | parent 41/96; copy holds |
| sep-swap, keep train markers | 20/96 | 0/96 | 54/96 | parent 41/96; copy *and* exact drop |
| value-absent novel | 0/96 original span | 0/96 | remaining-competitor 55/96; replacement-span 37/96 | original-span copy 0 |
| value-absent held-out | 0/96 original | 0/96 | 31/96 remaining | original-span copy 0 |
| value-absent broken-context | 0/64 original | 0/64 | 17/64 remaining | parent broken-context 16/64 was value-still-present |
| body-reorder query-first novel | 36/65 | 36/65 | 63/65 | matched parent **22/65** |
| body-reorder query-last novel | 22/61 | 22/61 | 58/61 | matched parent **27/61** |

Query-first 2-pair (items that were not already first): **15/16** vs matched parent 10/16. Query-last 2-pair (items that were first): 8/15 vs parent 9/15. First-slot is causally real on 2-pair in one direction and is **not** a complete account: query-swap follow is independent of whether the *new* pair is body-first (8/16 vs 8/15).

## Read of the preregistered mixture account

- Query-swap does **not** follow. B (query-conditioned first-token selection) is unsolved at every K≥2.
- New-gold `rest_value_tf_lock` 94/96 with follow 36/96: A (continuation) holds for the swapped target; B fails. This was the leading prediction.
- Marker-swap does **not** kill value copy. Train-surface copy is not marker-identity glue.
- Sep-swap kills exact (0/96) **and** drops inventory copy (93%→56%) and TF value lock (93→51). Held-out exact=0 is A′ **plus** separator identity entangled with the copy recipe, not suffix-only.
- Value-absent original-span copy is 0. Copy-from-context is confirmed. Frozen broken-context is not a value-absent control.

The first-token-binding diagnosis is **not** falsified by query-swap. It is isolated.

Proposed treatment (not launched): [`design/V010_V2R4_FIRST_TOKEN_SELECTION_PROTOCOL.md`](../design/V010_V2R4_FIRST_TOKEN_SELECTION_PROTOCOL.md).
