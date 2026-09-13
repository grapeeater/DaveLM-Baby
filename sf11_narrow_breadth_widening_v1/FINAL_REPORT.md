# SF11 — narrower breadth, unchanged presentation dose

**BREADTH_HYPOTHESIS_SUBSTANTIALLY_WEAKENED** — complete endpoints: **0/3**.

## Curriculum and exposure

48 unique input prompts: 16 unchanged preservation items + 32 widening items. The latter cross four answer names with four surface categories (16) and two orders × two query sources (16). Each category includes both assignments for each name pair. This is the minimum complete name-balanced coverage within the inherited SF9 template structure.

Selection was fixed before outcomes: SF10’s first canonical NEW shard (Alex/Owen: painted the tin cup; Mia/Nora: bought the green ball). All16 preservation items keep SF10’s rotation. Per update:9 correct answers per name;4 items per new surface or order/query category. Across unique prompts:12 correct answers per name.

Excluding the query/cue yields32 distinct context prefixes; the comparable SF9/SF10 breadth unit is the complete input prompt, yielding48. New surface counts:cloze4, activeQA4, passivecloze4, passiveQA4. Competing counts:each order8, each source8, each order×source4. Including preservation, unique fact-answer prompts40 and card-answer prompts8; query-source balance is exact within the competing arm.

| Study | Unique prompts | Batch | Planned English updates | Planned presentations | Presentations/item |
|---|---:|---:|---:|---:|---|
| SF8 |16|32|180|5,760|360|
| SF9 |144|144|180|25,920|180|
| SF10 |144|36|180|6,480|45|
| SF11 |48|36|180|6,480|new32:180; preservation16:45|

SF11 preserves SF10’s total dose, name/category mix, preservation dose, padding, binding schedule and full training step. Narrowing inevitably changes which lexical contexts recur and repetition per surviving item; cardinality is not isolated from subset composition.

## Frozen gates

- Surface16: forced-choice ≥15; greedy exact ≥13; every surface ≥3/4 exact.
- Order16: exact ≥13; fact and copy each ≥7/8; each order ≥7/8; each order×query ≥3/4.
- Both new panels: ≥4 additional exact responses versus that parent’s u0 (25 percentage points), at u200.
- TRAIN16:16 correct,16 exact,8 reversals,4 families. Language ≤ownu0+0.25nat. D3≤0.01.
- Binding: EACH pool answer≥76/80,BD≥76/80,collapse0. No averages. Stop on retention failure at100/200; endpoint200only.

## Parent mapping and checkpoint provenance

| New seed | SF8 parent | Parent SHA-256 |
|---|---|---|
|87026|C:\DaveLM-CADAVER\sf8_margin_dose_comparison_v1\runs\seed_87017_low\checkpoint_200.pt|9e293af6d16cb642ba8e2bf1aeb5ad392ff4b27399ebbf0b7b8fd2ef7339b81d|
|87027|C:\DaveLM-CADAVER\sf8_margin_dose_comparison_v1\runs\seed_87018_low\checkpoint_200.pt|839f7f5a60b33a1736376c8a68a02985fb05ba3230e56b7aaf20d1eba5b35102|
|87028|C:\DaveLM-CADAVER\sf8_margin_dose_comparison_v1\runs\seed_87019_low\checkpoint_200.pt|eb6a725173ff26516d876914ea4357cce7e3758616c3d659ff059d2354404517|

## All preregistered trajectories

|Seed|Update|TRAIN16 c/x/r/f|Surface c/x|Order exact fact/copy|CE / PPL|D3|
|---|---:|---|---|---|---|---:|
|87026|0|16/16/8/4|12/5|3 (3/0)|3.587301 / 36.136|0.00703672|
|87026|100|16/16/8/4|14/11|7 (4/3)|3.638862 / 38.048|0.01396234|
|87027|0|16/16/8/4|13/4|3 (3/0)|3.574605 / 35.681|0.00667241|
|87027|100|16/16/8/4|14/11|7 (4/3)|3.661114 / 38.905|0.02030055|
|87028|0|16/16/8/4|13/5|3 (3/0)|3.594973 / 36.415|0.00783949|
|87028|100|16/16/8/4|14/11|6 (4/2)|3.680524 / 39.667|0.01969012|

## New development subgroups and retention

### Seed 87026

|Update|cloze|active QA|passive cloze|passive QA|fact→card fact/copy|card→fact fact/copy|
|---:|---:|---:|---:|---:|---|---|
|0|3/4|0/4|2/4|0/4|3/4, 0/4|0/4, 0/4|
|100|3/4|3/4|2/4|3/4|3/4, 1/4|1/4, 2/4|

**STOP_REGRESSION at update 100.** Gates: `{"binding": {"pilot0": true, "pilot1": true}, "continue": false, "d3": false, "dev_order": false, "dev_surface": false, "development_gain": true, "endpoint_pass": false, "language": true, "retention_acquisition": true}`

- u0 pilot0: `{"answer_exact": 80, "both_distinct": 80, "complete_quartets": 20, "downstream_answer_correct_conditional_on_bd": 1.0, "queried_row_selection_conditional_on_bd": 1.0, "slot_collapse": 0, "strict_reversal_both_correct": 40, "strict_reversal_pairs": 40}`; PASS=True.
- u0 pilot1: `{"answer_exact": 80, "both_distinct": 80, "complete_quartets": 20, "downstream_answer_correct_conditional_on_bd": 1.0, "queried_row_selection_conditional_on_bd": 1.0, "slot_collapse": 0, "strict_reversal_both_correct": 40, "strict_reversal_pairs": 40}`; PASS=True.
- u100 pilot0: `{"answer_exact": 80, "both_distinct": 80, "complete_quartets": 20, "downstream_answer_correct_conditional_on_bd": 1.0, "queried_row_selection_conditional_on_bd": 1.0, "slot_collapse": 0, "strict_reversal_both_correct": 40, "strict_reversal_pairs": 40}`; PASS=True.
- u100 pilot1: `{"answer_exact": 80, "both_distinct": 80, "complete_quartets": 20, "downstream_answer_correct_conditional_on_bd": 1.0, "queried_row_selection_conditional_on_bd": 1.0, "slot_collapse": 0, "strict_reversal_both_correct": 40, "strict_reversal_pairs": 40}`; PASS=True.

Final checkpoint: `C:\DaveLM-CADAVER\sf11_narrow_breadth_widening_v1\runs\seed_87026_curriculum\checkpoint_100.pt`
SHA-256: `76c7b5099af46db8d49b0b0581c68666cd78f5e463538193ef51037decd829c0`
Actual English presentations: 3240; no unrun u200 result is inferred.

### Seed 87027

|Update|cloze|active QA|passive cloze|passive QA|fact→card fact/copy|card→fact fact/copy|
|---:|---:|---:|---:|---:|---|---|
|0|3/4|0/4|1/4|0/4|3/4, 0/4|0/4, 0/4|
|100|3/4|3/4|2/4|3/4|3/4, 1/4|1/4, 2/4|

**STOP_REGRESSION at update 100.** Gates: `{"binding": {"pilot0": true, "pilot1": true}, "continue": false, "d3": false, "dev_order": false, "dev_surface": false, "development_gain": true, "endpoint_pass": false, "language": true, "retention_acquisition": true}`

- u0 pilot0: `{"answer_exact": 80, "both_distinct": 80, "complete_quartets": 20, "downstream_answer_correct_conditional_on_bd": 1.0, "queried_row_selection_conditional_on_bd": 1.0, "slot_collapse": 0, "strict_reversal_both_correct": 40, "strict_reversal_pairs": 40}`; PASS=True.
- u0 pilot1: `{"answer_exact": 80, "both_distinct": 80, "complete_quartets": 20, "downstream_answer_correct_conditional_on_bd": 1.0, "queried_row_selection_conditional_on_bd": 1.0, "slot_collapse": 0, "strict_reversal_both_correct": 40, "strict_reversal_pairs": 40}`; PASS=True.
- u100 pilot0: `{"answer_exact": 80, "both_distinct": 80, "complete_quartets": 20, "downstream_answer_correct_conditional_on_bd": 1.0, "queried_row_selection_conditional_on_bd": 1.0, "slot_collapse": 0, "strict_reversal_both_correct": 40, "strict_reversal_pairs": 40}`; PASS=True.
- u100 pilot1: `{"answer_exact": 80, "both_distinct": 80, "complete_quartets": 20, "downstream_answer_correct_conditional_on_bd": 1.0, "queried_row_selection_conditional_on_bd": 1.0, "slot_collapse": 0, "strict_reversal_both_correct": 40, "strict_reversal_pairs": 40}`; PASS=True.

Final checkpoint: `C:\DaveLM-CADAVER\sf11_narrow_breadth_widening_v1\runs\seed_87027_curriculum\checkpoint_100.pt`
SHA-256: `6b383bfbec4b93c86d15ed5d89b1888fc34353eff5a65dc30631a5a56cbaca84`
Actual English presentations: 3240; no unrun u200 result is inferred.

### Seed 87028

|Update|cloze|active QA|passive cloze|passive QA|fact→card fact/copy|card→fact fact/copy|
|---:|---:|---:|---:|---:|---|---|
|0|3/4|0/4|2/4|0/4|3/4, 0/4|0/4, 0/4|
|100|3/4|3/4|2/4|3/4|3/4, 1/4|1/4, 1/4|

**STOP_REGRESSION at update 100.** Gates: `{"binding": {"pilot0": true, "pilot1": true}, "continue": false, "d3": false, "dev_order": false, "dev_surface": false, "development_gain": false, "endpoint_pass": false, "language": true, "retention_acquisition": true}`

- u0 pilot0: `{"answer_exact": 80, "both_distinct": 80, "complete_quartets": 20, "downstream_answer_correct_conditional_on_bd": 1.0, "queried_row_selection_conditional_on_bd": 1.0, "slot_collapse": 0, "strict_reversal_both_correct": 40, "strict_reversal_pairs": 40}`; PASS=True.
- u0 pilot1: `{"answer_exact": 80, "both_distinct": 80, "complete_quartets": 20, "downstream_answer_correct_conditional_on_bd": 1.0, "queried_row_selection_conditional_on_bd": 1.0, "slot_collapse": 0, "strict_reversal_both_correct": 40, "strict_reversal_pairs": 40}`; PASS=True.
- u100 pilot0: `{"answer_exact": 80, "both_distinct": 80, "complete_quartets": 20, "downstream_answer_correct_conditional_on_bd": 1.0, "queried_row_selection_conditional_on_bd": 1.0, "slot_collapse": 0, "strict_reversal_both_correct": 40, "strict_reversal_pairs": 40}`; PASS=True.
- u100 pilot1: `{"answer_exact": 80, "both_distinct": 80, "complete_quartets": 20, "downstream_answer_correct_conditional_on_bd": 1.0, "queried_row_selection_conditional_on_bd": 1.0, "slot_collapse": 0, "strict_reversal_both_correct": 40, "strict_reversal_pairs": 40}`; PASS=True.

Final checkpoint: `C:\DaveLM-CADAVER\sf11_narrow_breadth_widening_v1\runs\seed_87028_curriculum\checkpoint_100.pt`
SHA-256: `0848b962c883ac6fb903d6cd9099dc896026510fac2ddb057cd989d5ed4ddd17`
Actual English presentations: 3240; no unrun u200 result is inferred.

## Evidence ledger and Dave-coded interpretation

- D3 safe at the stopping endpoint: **0/3**. Complete frozen widening+retention endpoint: **0/3**.
- Surface exact improved versus parent: **3/3**; order/source exact improved: **3/3**. Improvement alone is not passage of the subgroup gates.
- SF8 TRAIN16 acquisition retained: **3/3**.
- The new panels test the same surface/source structures on two previously unused verb/object combinations. They do not establish unconstrained language, general reasoning, or broad transfer. The 32 new DEV items were frozen before parent/candidate scoring.
- Historical SF9/SF10 development and SF1 transfer panels were not scored. FINAL/sacred remained untouched. There was no outcome-driven change, extra seed, rescue, or follow-up experiment.
- Three independent SF8 parent continuations give bounded replication. Comparison with SF10 is historical; it does not prove that the number of distinct contexts alone caused D3 behavior.
- Dave-coded: fewer examples count as a win only if Baby both learns the wider lesson and stays safe. Moving a few DEV answers while D3 breaks is still a failed treatment. Calling it general English would be bullshit.

## Integrity

Receipt: `73c9896d9ca74ffd1b01e3c2e0c07746a2e7c7bfd83be8d904ab66879312acd4`
Payload manifest: `aad566bce8d94f842db28ffc73914b7522517fc19cc2a3392db169bde8dfbe72`
Raw per-item scores/generations, D3 traces, binding rows and update metrics are in runs/. The result classification is committed before any potential future panel decision. No historical-panel reevaluation is included.

NEXT ACTION: Preregister one retention-objective review before authorizing further widening training.
