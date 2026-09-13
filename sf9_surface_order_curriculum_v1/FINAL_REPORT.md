# SF9 — surface/order curriculum widening (FINAL)

**Classification: RETENTION_REGRESSION.** All three independent runs stopped at update 100 on the
frozen D3 gate (name-pollution ceiling). The widening itself was working; the D3 safety gate was not.

## 1. Study status
Ran and frozen. Three runs, each branching from a distinct SF8 lambda=0.25 acquisition checkpoint
(87020<-87017_low, 87021<-87018_low, 87022<-87019_low), fresh optimizer, unchanged lambda=0.25/M=1.0
margin, unchanged KL/binding/scope. Single manipulated variable: curriculum breadth.

## 2. What was trained
144-item curriculum = 16 original TRAIN16 (preservation) + 64 new surface-form items (cloze /
active_qa / passive_cloze / passive_qa) + 64 new competing-order items (order0/order1 x fact/copy),
built from NEW vocabulary disjoint from all frozen panels. 180 English updates, every update = all
144 items once (4.5x SF8's name-bearing example density).

## 3. Parent checkpoints (all verified hash == frozen STATUS.json)
- 87020 <- sf8 seed_87017_low checkpoint_200.pt (9e293af6…)
- 87021 <- sf8 seed_87018_low checkpoint_200.pt (839f7f5a…)
- 87022 <- sf8 seed_87019_low checkpoint_200.pt (eb6a7251…)

## 4. Curriculum summary
144 items; name-answer frequency balanced 32 each (Alex/Owen/Mia/Nora); surfaces/orders/queries/directions balanced.
New vocab: tin cup / wool hat / green ball / silver bell x painted / dropped / bought / held.
Dev vocab (held-out): clay mug / brass key x stacked / rolled. Disjointness from frozen panels verified.

## 5. Prospective success/stop gates
- retention_acquisition: TRAIN16 16/16/8/8/4/4 (unchanged)
- dev_surface: correct>=15/16 AND exact>=13/16
- dev_order: exact>=13/16 AND fact-query exact>=7/8 AND copy-query exact>=7/8
- language CE <= update0 CE + .25 ; D3 <= .01 ; binding pools pass
- continue (at 100/200) = retention_acquisition + language + D3 + binding ; regression -> stop
- endpoint_pass (at 200) = continue + dev_surface + dev_order

## 6. Training endpoints
| Seed | Endpoint | TRAIN16 c/x/rev/fam | DevSurface c/x | DevOrder x | Lang CE | D3 | Status |
|---|---|---|---|---|---|---|---|
| 87020 | 100 | 16/16/8/4 | 11/8 | 10 | 3.7099 | 0.01991 | STOP_REGRESSION |
| 87021 | 100 | 16/16/8/4 | 11/8 | 7 | 3.6524 | 0.01882 | STOP_REGRESSION |
| 87022 | 100 | 16/16/8/4 | 12/9 | 8 | 3.6593 | 0.01809 | STOP_REGRESSION |

## 7. New development results (parent baseline -> update 100)
- dev_surface exact: 3/16 -> 8-9/16 (correct 11-12/16). Subgroups @100: cloze 3/4, active_qa 3/4, passive_cloze 1/4, passive_qa 1-2/4.
- dev_order exact: 2/16 -> 7-10/16. Subgroups @100: order0:fact 3/4, order1:fact 0-2/4, order0:copy 2-3/4, order1:copy 2/4.

## 8. Original-capability retention
TRAIN16 acquisition 16/16 correct + 16/16 exact + 8/8 reversals + 4/4 families in ALL three seeds at the
point of stop. The earned acquisition was fully preserved.

## 9. Language / D3 / binding retention
- Language: rose from ~3.57-3.60 to 3.65-3.71 (still within the frozen +0.25 allowance). PASS.
- Binding pilot0/pilot1: PASS in all runs.
- D3: rose from ~0.0067-0.0078 to 0.0181-0.0199 (> 0.01). FAIL -> frozen hard stop at 100 in all three runs.

## 10. Cross-seed replication
Identical failure mode in all three independent parents: D3 gate tripped at update 100. Dev gains were
also replicated (surface exact 8-9/16, order exact 7-10/16, cloze+active_qa ~mastered, passive still weak).

## 11. Dave-coded interpretation
- Did Baby learn the same job across wording? Partially yes — cloze and active "who" forms went from
  ~0 to 3/4 in just 100 updates, on NEW unseen facts, without being prompted. Passive forms barely moved (1-2/4).
- Did Baby stop leaning on recency/order? Partially — order0:fact reached 3/4, but order1:fact stayed 0-2/4,
  so "last name mentioned" still wins when the card comes first.
- Can Baby select the requested source when facts compete? Only in the canonical order; copy queries stayed weak (2/4).
- Did widening damage what we already earned? NO — TRAIN16 stayed 16/16/8/8/4/4 and language/binding stayed green.
  But it tripped the D3 safety gate: ordinary-text name mass doubled (0.007 -> 0.019), i.e., teaching the same
  lesson 4.5x denser made Baby leak names into normal sentences.
- Strongest claim: the widening direction is correct (surface/order dev behavior improved, acquisition preserved),
  but the delivery density is too high for the frozen D3 ceiling.
- Still bullshit: that Baby now answers across surfaces and orders generally; passive/competing-order behavior
  remains weak and the run was cut short by the safety gate.

## 12. Next action (recommended, not implemented)
Rerun the SAME curriculum-breadth treatment at a name-example density that keeps D3 below 0.01 (e.g., interleave
the 144-item curriculum on only a subset of English updates, or halve the per-update name-bearing examples), keeping
lambda=0.25/M=1.0 and all gates unchanged. This is a density/insertion-rate fix, not a new mechanism.

## Provenance
Freeze receipt: bd64384dd59cbc486ccbfb8eef3cb25c0cdc3445cb8512d0e6ec64eeb699fd22
Manifest: 4eac6a16e628535c273382f0c001a10cb507eeaa89a4e4637c7a7ad636b9b40e
