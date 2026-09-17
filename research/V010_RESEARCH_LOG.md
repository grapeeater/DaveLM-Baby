# Baby v0.10 running research log

This log is append-only. It does not rewrite frozen v2R4 receipts or Gate L/C/R.

v2R5 remains **absent in GitHub / unresolved / pending**.

---

## Milestone 2026-09-16 — emission source: copy works, selection does not

### CURRENT BEST DIAGNOSIS

Baby v0.10 at v2R4 U16000 has a working **copy-a-value-span-from-context** mechanism on the training surface. She does **not** reliably **select the queried pair** among competing in-context values. Held-out exact=0 is a mix of weaker inventory copy, first-pair fallback, and held-out separator/EOS OOD. Architecture is not the immediate claim.

Split:

- **A** identify internally: partial. When she copies a competitor, the queried first token is almost never rank 1 and is typically rank 2 on train novel.
- **B** bind/select queried item: failed. Dominant train-novel error is emitting another pair's full value (52/96).
- **C** copy selected payload: supported. 93/96 train-novel greedy outputs are some in-context value span.
- **D** free-generate that payload: supported. TF value = free value.
- **E** separator/EOS: train OK; held-out exact killed by OOD seps `{82–85}`.
- **F** new surfaces: held-out inventory copy drops to 55/96 and first-pair bias appears.

### EVIDENCE FOR IT

Locked from frozen metrics+panels via `emission_source_report`:

| panel | queried copy | competitor copy | off-inventory | inventory copy |
|---|---:|---:|---:|---:|
| primitive_keyed | 64 | 0 | 0 | 64/64 |
| short_keyed | 54 | 10 | 0 | 64/64 |
| same_surface_novel | 41 | 52 | 3 | 93/96 |
| heldout_surface | 23 | 32 | 41 | 55/96 |

Train novel: competitor copies have `rank1_when_competitor_copy = 0` and median target rank **2**. Queried copies have median rank **1**.

Chance-level 3-pair queried rate (7/21) is **not** "she cannot copy." She copies a real pair; the queried one is selected at ~1/3 with a first-slot tilt on 4-pair (`P(ok|render_pos=0)=7/11`).

### EVIDENCE AGAINST IT / CAVEATS

- First-pair bias is only moderate on train novel (`P(emit first)=0.427` vs `P(query first)=0.323`). Not a pure first-slot machine.
- Nearest-pair-to-query is weaker than first-pair (`P(emit nearest)=0.29` on novel).
- 4-pair first-slot 7/11 is a small bucket.
- Held-out 2-pair first-slot is stronger (`12/21` vs `2/17`) but still n-limited.
- Competitor rank 2 does not prove the full queried *span* is represented, only the first token.
- v2R5 unknown. Single seed.

### WHAT WAS FALSIFIED

- Previous (including this agent's first autopsy) reading of "3-pair at chance ⇒ copy failed / random tokens" is **false**. Failures are mostly **wrong-pair copies**.
- "Knows but cannot emit the value" remains false for the payload.
- Pure nearest-to-query positional shortcut is false as the main recipe.
- Always-copy-first-pair is false as a complete train-surface account.

### WHAT REMAINS UNKNOWN

- Query-swap follow vs stuck-on-old (needs U16000 weights).
- Whether moving the queried pair to body-first **causes** the 4-pair boost (body-reorder probe; needs weights).
- Marker-swap vs sep-swap causal split (needs weights).
- Value-absent: does original-answer copy die when the span is gone (needs weights).
- Whether rank-2 queried first token is accompanied by the rest of the span in later TF steps when a competitor was greedily chosen.
- Induction: 48/64 primitive emissions are some source token, only 19 the aligned continuation — separate offset/binding problem.
- v2R5.

### NEXT EXPERIMENT

1. **Already implemented, needs Fan Diesel weights:** `diagnose_checkpoint` now includes query-swap, marker/sep swap, value-absent, and **body-reorder query-first/last**.
2. **Already run here, no weights:** emission-source census (this milestone).

Fan Diesel body-reorder is the highest-information *causal* positional test: if query-first lifts 4-pair queried copy a lot and query-last drops it, selection is partly a body-slot prior, not only a missing bind operator.

### WHY HIGH INFORMATION

It distinguishes B (wrong pair chosen among copied candidates) from C (cannot copy) without another training run, and it makes the checkpoint probes about *selection* rather than generic accuracy.

---

## Milestone 2026-09-16 — TF lock, chance tests, probe-limit, induction EOS glue

### CURRENT BEST DIAGNOSIS

Baby has a **copy-a-present-value-span** skill on the train surface. The remaining keyed bottleneck is **first-token selection among in-context values**, not span tracking and not free-generation of a selected span.

Once the gold first token is teacher-forced, the rest of the queried value **and** the train separator/EOS are rank-1 almost always, even on rows where greedy copied a competitor. She can identify and copy the queried payload internally. She does not reliably **choose** it.

Queried-copy rates at 2/3/4 pairs are **not** above 1/K at p<0.05. The earlier "2-pair binding signal" is withdrawn.

Held-out failure is a mixture of weaker inventory copy (especially long values), first-token selection failure, and **train-separator glue** (all 41 off-inventory held-out emissions contain a train sep). Induction immediate-EOS is **not** generic collapse: 22/22 cases are "context already ends with this item's separator → emit EOS."

Curriculum: 1-pair copy saturates first; multi-pair inventory copy saturates at U12000 on the n=16 probe; selection never lifts on the matched slice. Architecture is still not the next claim.

### EVIDENCE FOR IT

- Train novel TF continuation: queried 41/41 full lock; competitor 51/52 rest-value lock and 51/52 full lock.
- Held-out queried: 23/23 rest-value lock, **0/23** full lock (sep/EOS OOD).
- Greedy first token ∈ inventory first tokens: 95/96 train novel; first tokens unique 96/96.
- Chance tests: 2-pair 19/31 two-sided p=0.281; 3-pair 7/21 p=1.0 vs 1/3; 4-pair 15/44 p=0.222 vs 1/4; short 2-pair 18/28 p=0.185.
- Probe limit: U6000–U15500 scored n=16; U16000 scored full panels. Matched first-16 novel queried 6→5 from U15500→U16000. Inventory copy 16/16 from U12000.
- Held-out off-inventory 41/41 contain a train separator; value_length=10 inventory copy 0/12.
- Induction EOS: last token == own separator in 22/22 immediate-EOS rows; 0/64 full-induction EOS when last token is not that separator.
- Query-before-body (`keyed_5`) queried rate 5/12 vs 36/84 after-body. Not the main recipe.

### EVIDENCE AGAINST IT / CAVEATS

- 4-pair first-slot queried copy 7/11 vs ~0.25 on other slots is a possible body-slot prior. n is small; body-reorder still needs weights.
- Held-out 2-pair first-slot 12/21 vs 2/17 is stronger than train 2-pair (which is ~0.60 in both slots). Surface change may engage a first-slot heuristic that train 2-pair does not need.
- TF lock is gold-teacher-forcing, a state greedy often never visits after a wrong first token. It measures A, not B.
- Competitor rank distribution on train is not always 2 (hist includes 3, 4, 5, and a few large ranks). "Runner-up" is typical, not universal.
- Single seed. v2R5 unknown. Query-swap still unscored.

### WHAT WAS FALSIFIED

- This branch's earlier claim that 2-pair queried copy is a real binding signal.
- Reading U16000 41/96 as a late capability jump versus U15500 (confounded by probe_limit=16 vs full panel).
- "Immediate EOS on induction = generic collapse / language death." It is trailing-separator suffix glue.
- "3-pair at chance means she cannot copy" (already falsified by emission source; restated).
- Query-must-follow-body as the train-surface recipe (`keyed_5` is statistically the same, n=12).

### WHAT REMAINS UNKNOWN

- Query-swap: follow new value vs stuck on old vs copy some other inventory value.
- Whether body-reorder **causes** the 4-pair first-slot tilt.
- Marker identity vs separator identity as the held-out copy drop.
- Value-absent knockout (train-novel copy of novel spans almost has to be from context, but it is still the right causal control).
- Why held-out long values lose even inventory copy.
- Induction offset binding when the trailing-sep trap is removed (3/64 correct among last≠sep full induction).
- v2R5.

### NEXT EXPERIMENT

Highest-information remaining test that needs weights: **query-swap** on same-surface novel and short 2-pair, read as:

- `query_swap_follow_new_value`
- `emitted_original_value_span` (stuck-on-old)
- `competitor_copy` of a third value
- `rest_value_tf_lock` on the **new** gold span

Predictions:

- Follow high, stuck-old low → some query-conditioned selection exists; then body-reorder and pair-count pressure matter.
- Follow low, stuck-old high → first-token selection ignores the query; copies a previously highlighted span.
- Follow low, stuck-old low, inventory copy stays high → copies *a* span, not the old one and not the new one; still not binding.
- New-gold `rest_value_tf_lock` high with follow low → A holds for the swapped target; B is still the bottleneck.

Body-reorder, marker/sep swap, and value-absent remain on the same Fan Diesel command. Do not train yet.

### WHY THAT EXPERIMENT HAS HIGH INFORMATION

Every data-only test still leaves open "maybe a hidden query cue we did not swap." Query-swap is the cleanest causal intervention on B that does not update weights. Chance tests already removed the excuse that 2-pair is solved.

---

## Milestone 2026-09-16 — Fan Diesel U16000 isolation probe (weights, no training)

### CURRENT BEST DIAGNOSIS

Baby copies in-context value spans. She does **not** bind the query to the correct first token. Changing only the query leaves first-token selection at 1/K. Teacher-forced continuation of the *new* gold value still locks. A first-slot prior is real but does not explain query-swap. Train-surface copy does not need marker identity; separator identity is entangled with copy, not only suffix OOD.

### EVIDENCE FOR IT

Verified U16000 SHA `94b3a9da…17827` on Fan Diesel `cuda` / RX 9060 XT. `trained=false`. Frozen panels unmodified.

Query-swap novel: follow 36 / stuck-old 33 / other-competitor 23 / off 4; inventory 92/96; new-gold `rest_value_tf_lock` 94/96. By K: 16/31, 8/21, 12/44, all p>0.8 vs 1/K. Short 2-pair follow 12/28 stuck-old 16/28. On novel misses, queried first token is never rank-1 (0/60); 30 runner-up, 30 not competitive. Median queried rank 2.

Body-reorder query-first 36/65 vs matched parent 22/65 (2-pair 15/16 vs parent 10/16). Query-last 22/61 vs parent 27/61. Marker-swap value_ok 47/96 (holds). Sep-swap value_ok 20/96, exact 0/96, inventory 54/96. Value-absent original-span copy 0; remaining-competitor 55/96; replacement-span 37/96.

### EVIDENCE AGAINST IT / CAVEATS

- Query-first 2-pair 15/16 shows a slot prior strong enough that a naive first-token loss could be cheated by “copy body-first.”
- Query-last 2-pair barely drops (9/15 → 8/15), so first-slot is not the whole 2-pair recipe.
- Sep-swap also wrecks inventory copy, so A′ is not suffix-only.
- Single seed. v2R5 unresolved and not opened.
- Mixed-panel 1/K p-values are invalid; use the K-stratified tests above.

### WHAT WAS FALSIFIED

- “Query-swap will follow the new query, so first-token binding is already present.” Falsified.
- “2-pair parent copy is query-conditioned.” Falsified: swap is 16/31 vs 15 stuck-old.
- “Marker identity is the train-surface copy cue.” Falsified: marker-swap 47/96 value copy.
- “Sep-swap only kills exact.” Falsified as exclusive: exact dies **and** inventory copy drops 93%→56%.
- “Broken-context 16/64 measures retrieval without the value.” Falsified: value-absent original copy is 0.

### WHAT REMAINS UNKNOWN

- Why the queried first token is rank-2 so often on 2-pair (attention, recency, residual of the pre-swap query, or undifferentiated inventory head).
- Why held-out long values lose even remaining-competitor copy.
- Induction offset binding after trailing-sep glue.
- v2R5.

### NEXT EXPERIMENT

Do **not** train yet unless a later explicit launch is approved. Highest-information *unlaunched* treatment is a multi-pair first-token contrastive / query-swap objective with randomized body order: [`design/V010_V2R4_FIRST_TOKEN_SELECTION_PROTOCOL.md`](../design/V010_V2R4_FIRST_TOKEN_SELECTION_PROTOCOL.md). Highest-information *read-only* leftover is a 2-pair attention/logit dump of queried vs original first-token heads on the swap misses (already know ranks; do not know which layer).

### WHY HIGH INFORMATION

Query-swap is the causal test that data-only census could not run. Follow-at-chance plus new-gold lock isolates B from C/D. Marker/sep/value-absent/body-reorder prevent treating B as “just markers” or “just first slot.”

---

## Milestone 2026-09-16 — S1 first-token CE vs matched control: futility + regression

### CURRENT BEST DIAGNOSIS

Weak query-conditioned first-token signal is real and still too weak to win. Adding weight-1.0 first-token vocabulary CE on keyed counterfactual rows, with unchanged full-answer CE, did not move body-macro selection enough to beat continued training on the same frozen counterfactual schedule. Copy/language mostly held; primitive induction first-token/exact dropped past the 0.05 regression line on the treatment arm.

### EVIDENCE FOR IT

Matched seed 110001, parent `94b3a9da…17827`, manifest `7e7a96bf…1aac5`. Control +400 body-macro 0.4074 (+0.0052). Treatment +400 body-macro 0.4097 (+0.0075). Treatment margin −0.817 vs parent −0.792. Bootstrap treatment−control 95% CI [−0.019, +0.024]. Adjudication `REGRESSION` + `futility=true`. See `research/V010_SELECTION_REPAIR_S1_TERMINAL.md`.

### EVIDENCE AGAINST IT / CAVEATS

- Single seed; replicate 110002 is forbidden by S1 success-then-replicate rule.
- `data.py`/`data_v2.py` freeze-JSON hashes never matched git blobs; treatment used bytecode-identical committed sources after a checkout destroyed the extra working-tree bytes. Schedules were already frozen.
- Control query-swap first top-1 also dropped (0.375→0.302) without the extra first-token loss.

### WHAT WAS FALSIFIED

- “S1 first-token CE is a sufficient +400 remedy for query-conditioned selection vs matched full-answer CE.” Falsified.
- “The interrupt left control mid-+200 eval.” Falsified: control had finished +400.

### WHAT REMAINS UNKNOWN

- Whether a contrastive / query-swap objective with randomized body order would move selection without regressing induction.
- Which layer holds the rank-2 queried-token residue.

### NEXT EXPERIMENT

Do not extend S1 or launch 110002. New preregistration required. Do not open TEST. Do not merge to main.

### WHY HIGH INFORMATION

The matched control existed; futility is pairwise, not “treatment looked disappointing.”

---

## Milestone 2026-09-16 — Independent S1 autopsy revises the amplification diagnosis

### CURRENT BEST DIAGNOSIS

A query-conditioned logit residue is real, but it is **not** a first-token-CE amplification problem. Independent gold NLL on counterfactual twins did not move queried inventory rank-2 into rank-1, lowered `query_logit_effect`, and increased query-invariant collapse. The extra term was live: it stole from primitive induction. The missing operator is a loss whose minimum requires the inventory preference to **flip with the query**.

### EVIDENCE FOR IT

Recomputed from hashed S1 evals (21/21 SHA256SUMS match). Queried inventory rank-1 0.377→0.387; rank-2 stuck at 0.338. Collapse 118/144→123/144. K=2 body-macro glued at 0.53125. Query-swap follow 36→35. Induction 0.297→0.172 on treatment vs 0.266 on control. First-slot is only 0.30 of diagnostic copies; 39/118 collapsed bodies match the first rendered head. See `research/V010_S1_INDEPENDENT_AUTOPSY.md`.

### EVIDENCE AGAINST IT / CAVEATS

- Rank-2 residue still exists; some other first-position loss could still use it.
- All-K packing for K=3,4 was not tested in S1 (only 2-of-K).
- Single S1 seed.

### WHAT WAS FALSIFIED

- “S1 failed because first-token CE was too weak to move the network.” Induction moved.
- “S1 failed because rank-2 just needed more gold NLL.” Rank-2 did not become rank-1.
- Inventory-restricted softmax as a *near* next step: 95% inventory copy makes it ~the same gradient as full-vocab CE when the winner is in-set.

### WHAT REMAINS UNKNOWN

- Whether paired query-contrast (S1's `query_logit_effect` as a training loss) creates a bind operator.
- Whether all-K packing alone would move K=3,4 (hence S2's matched control).

### NEXT EXPERIMENT

S2: `design/V010_SELECTION_REPAIR_S2.md`. Paired query-contrast vs matched all-K full-answer CE. Futility at +200 on the mechanism triad. Do not raise S1's λ.

### WHY HIGH INFORMATION

It tests a different functional than the failed recipe, with a control that isolates the new data packing from the new loss.

---

## Milestone 2026-09-16 — S2 protocol frozen (pre-train)

### CURRENT BEST DIAGNOSIS

Unchanged from the independent S1 autopsy above. Protocol frozen before generate/train.

### EVIDENCE FOR IT

Protocol file `design/V010_SELECTION_REPAIR_S2.md`. Extra term is `softplus(2.0 - effect_ij)` on all-K query groups, not first-token CE.

### EVIDENCE AGAINST IT / CAVEATS

Launch still required. Attention dump deferred: would not change this functional.

### WHAT WAS FALSIFIED

Nothing new at freeze time.

### WHAT REMAINS UNKNOWN

Whether S2's extra term moves `query_logit_effect` / collapse / inventory rank-1 by +200.

### NEXT EXPERIMENT

Generate hashed S2 data, preflight, control then treatment seed 120001, adjudicate against frozen S2 gates.

### WHY HIGH INFORMATION

Gates and the extra term are committed before seeing S2 numbers.

---

## Milestone 2026-09-16 — S2 paired query-contrast: REGRESSION, functional moved, greedy did not

### CURRENT BEST DIAGNOSIS

Paired query-contrast is a *trainable* bind-direction functional: it raised `query_logit_effect` from 0.78 to 1.55 in 400 updates without the S1 induction smash. Greedy selection is still a query-invariant attractor (collapse 0.83→0.72, still far from 0.50; K=2 diagnostic glued at 0.542; mean margin still negative). Teacher-forced full-span CE likely keeps locking the favorite. S2 also nicked primitive 1-pair first-token (1.00→0.94) and short-keyed exact.

### EVIDENCE FOR IT

Matched seed 120001, parent `94b3a9da…17827`, manifest `006b3889…7d9e01`. Treatment effect 1.551 vs control 0.875 vs parent 0.784. Collapse 0.833→0.715 (control 0.799). Query-swap follow 36/32/40; stuck-old 33/36/24; 2-pair follow 16/17/21 of 31. Induction 0.297→0.312. Adjudication `REGRESSION` (primitive_keyed first_top1, short_keyed free_exact). Bootstrap treatment−control body-macro 95% CI [0.003, 0.072] with tiny absolute gain. See `research/V010_SELECTION_REPAIR_S2_TERMINAL.md`.

### EVIDENCE AGAINST IT / CAVEATS

- Diagnostic K=2 accuracy identical to parent; isolation 2-pair follow is the only greedy-ish lift.
- 3/4-pair query-swap still at chance.
- Single seed. Contrast loss ~4 vs CE ~1; dose is large.
- +200 barely missed futility on collapse drop 0.056 vs 0.05.

### WHAT WAS FALSIFIED

- "Query-logit-effect cannot be trained with a paired contrast extra term." Falsified.
- "Any extra first-position keyed term will regress induction like S1." Falsified here.
- "All-K packing, not contrast, produces the +0.77 effect jump." Falsified by the matched control.
- "S2 m=2 all-K contrast is a sufficient +400 greedy-selection remedy." Falsified.

### WHAT REMAINS UNKNOWN

- Whether masking first-token CE (remainder-only copy) lets the already-trained flip win the argmax.
- Whether a 1-pair retention mix restores primitive keyed without undoing the flip.

### NEXT EXPERIMENT

Do not raise S2 λ/m/duration or launch 120002. Next protocol, if any: remainder-masked CE + paired contrast + small primitive-keyed retention, new matched control, futility on greedy rank-1 / K=2 collapse. Not launched in this milestone.

### WHY HIGH INFORMATION

The extra term moved its target and not induction. The remaining miss is "logits flip, greedy does not," which a λ bump will not diagnose.

---

## Milestone 2026-09-16 — the defect is a one-token query-transport range, not selection

### CURRENT BEST DIAGNOSIS

Baby **already has** a working query→key→value bind-and-copy circuit. Its effective reach is about **one token**. Queried selection fails whenever the query key sits two or more tokens before the generation position, because the query key's *identity is never transported* to where the answer is emitted. Copy, candidate representation, continuation, and first-token logit strength are all intact. See `research/V010_QUERY_TRANSPORT_AUTOPSY.md`.

### EVIDENCE FOR IT

Read-only probe on the hashed U16000 parent (`scripts/probe_query_locality.py`), every arm append-only so the inventory is never disturbed. Candidate-restricted argmax, chance ≈ 0.336: `as_is` 0.394; `append_query` **0.607**; `append_other_key` (a different pair's key, scored as "did she pick that pair") **0.556**; `append_unused_key` 0.336; `append_filler_key_8` 0.352; `append_filler_span_8` 0.361.

Dose-response on distance from a known-good start: 0 → 0.607, 1 → 0.433, 2 → 0.340, 4 → 0.340, 8 → 0.368. On the rows that sit at **exact** chance (gap ≥ 31, n=159), `append_query` excess is **+0.258** and `append_other_key` **+0.233**. K=4 goes 0.344 → 0.552. Adding 32 filler tokens after a *working* gap-1 query drops it 0.468 → 0.297.

Logit decomposition from the frozen evals: query-invariant candidate salience spread median **2.776 nats**, query-dependent spread median **0.237 nats**. S2's contrast functional is algebraically invariant to the salience term, so it could only move the 0.24 and never the 2.78; it did exactly that (0.237 → 0.341).

The cliff reproduces on **eight** checkpoints: v2R4 parent, S1 treatment, S2 control, S2 treatment, and the local-only v2R5 (107001/107002) and v2R6 (108001/108002) terminals.

### EVIDENCE AGAINST IT / CAVEATS

- Gap and rendering variant are nearly collinear in the frozen panels; the causal claim rests on the append probe, not on the variant table.
- `append_query` duplicates the query key, so it tests reach, not the full indirection task.
- Single frozen diagnostic (432 rows, 144 bodies).

### WHAT WAS FALSIFIED

- "First-token selection failure" / "query blindness" as descriptions of the defect.
- "Query-swap does not follow at any K." It follows at 0.556 when swapped *adjacently*; the frozen result swapped in place, i.e. out of reach.
- "Persistent multi-candidate K≥3 failure" as intrinsic.
- "The query signal merely needs to be stronger."
- "S2 was a regression." S2 strengthened the adjacent circuit and was scored by a metric in which ~63% of rows are structurally unreachable.
- "Key-bank filler jams the key channel" (span-bank filler is equivalent) and "appending anything perturbs the choice" (`append_unused_key` is flat).
- A prior positional reading: `parse_records` returns **pre-shuffle** record order, so candidate indices are not body slots.

### WHAT REMAINS UNKNOWN

- Whether the range limit is curricular or architectural/representational.
- Whether any head attends to the query position at gap > 1.
- v2R5/v2R6 were fully trained locally (2 seeds each) with deleted source; they are at chance on this instrument.

### NEXT EXPERIMENT

T1 gap curriculum, frozen in `design/V010_SELECTION_REPAIR_T1_GAP.md`.

### WHY HIGH INFORMATION

It is the one causal factor no curriculum in the repository has ever manipulated, and it targets the only replicated structural bottleneck.

---

## Milestone 2026-09-16 — T1 gap curriculum: NULL on transport, REGRESSION on retention

### CURRENT BEST DIAGNOSIS

Unchanged mechanism, with the curricular repair now disfavoured. Four hundred updates of a gap 0–3 curriculum, with **no loss change**, against a perfectly matched control sharing the same items in the same order, raised amplitude inside the one-token reach and widened the reach by **zero** tokens.

### EVIDENCE FOR IT

Primary endpoint (frozen diagnostic, gap ≥ 13, n=215): parent 0.3442 (+0.000 excess), treatment 0.3488 (+0.005), control 0.3488 (+0.005). Treatment − control **exactly 0.0**; paired body bootstrap 95% CI **[−0.0139, +0.0135]**. Gap 0–1: parent +0.146, treatment **+0.171**, control +0.057. Post-hoc probe distance 2/4/8 unchanged at chance in both arms. `append_query` 0.607 → 0.648 treatment, 0.514 control. See `research/V010_SELECTION_REPAIR_T1_TERMINAL.md` and `runs/selection_t1/ADJUDICATION_130001_400.json`.

Frozen panels, parent → treatment → control: `same_surface_novel` free exact 0.4271 → **0.5104** → 0.4375; body-macro 0.4115 → 0.4201 → 0.3848; K=4 0.339 → 0.323 → 0.266. The treatment beats the matched control on every capability measure and is simply orthogonal to the bottleneck.

### EVIDENCE AGAINST IT / CAVEATS

- **Protocol error:** the frozen futility gate sat at update 400, but the schedule's long-gap blocks start at 301/451/601. Futility fired, so gaps > 3 were never really trained. The rule was not changed after seeing results; the full schedule is untested.
- Single seed; 130002 generated and not launched.
- The secondary prediction failed: salience spread **grew** in both arms (2.78 → 3.30 treatment, 3.51 control).

### WHAT WAS FALSIFIED

- "Scaffolding at gaps 0–3 for 400 updates extends the transport range."
- "A gain on `same_surface_novel` implies selection improved" — the panel moved +0.07 while long-gap selection moved 0.000.
- "Training at gap 0 will teach an attend-to-the-last-token shortcut" — `append_unused_key` excess +0.021.
- "A primitive-keyed retention stream cannot protect 1-pair copy" — it held at 1.000 where S2 lost it.

### WHAT REMAINS UNKNOWN

- The full T1 schedule (updates 401–800, gaps to 20 and natural).
- Whether the query key's identity is present at all in the residual stream at the generation position when gap > 1.

### NEXT EXPERIMENT

No further weight updates. Read-only attention/composition localization on the hashed parent: does any head attend to the query position at gap > 1, and is the query key linearly decodable at the generation position as a function of gap? Present-but-unused ⇒ downstream composition defect. Absent ⇒ the transport pathway does not exist and an architecture claim is justified for the first time.

### WHY HIGH INFORMATION

Three intervention classes (first-token CE, paired contrast, gap curriculum) now share one signature: amplitude inside the reach, nothing outside it. The remaining split is representational presence versus use, and that is a read-only question.

---

## Milestone 2026-09-16 — query-transport investigation complete (paused)

### CURRENT BEST DIAGNOSIS

Unchanged from the two milestones above. Baby has a working query→key→value
bind-and-copy circuit with about a one-token reach; queried selection fails when
the query key sits two or more tokens before generation because its identity is
never transported. T1 gap curriculum (no loss change) raised amplitude inside
that reach and widened it by zero tokens. Authoritative Baby remains v2R4 U16000
SHA `94b3a9da…17827`.

### DURABLE RECORD

- Localization + causal probe: `research/V010_QUERY_TRANSPORT_AUTOPSY.md`
  (commit `ab93021`)
- T1 preregistration: `design/V010_SELECTION_REPAIR_T1_GAP.md` (commit `8892545`)
- T1 pipeline freeze: commit `0ea5299`
- T1 adjudication: `research/V010_SELECTION_REPAIR_T1_TERMINAL.md` (commit `b714e28`)

Branch `cursor/query-transport-range-t1` / [PR #4](https://github.com/grapeeater/DaveLM-Baby/pull/4)
vs `v0.10`. Open; not merged.

### NEXT EXPERIMENT

None until resumed. Highest-information read-only follow-up if resumed:
attention/composition localization on the hashed parent (query identity present
vs absent at generation position when gap > 1).

### WHY HIGH INFORMATION

Same as T1 milestone. Work is **paused per user request**; no further training or
experiments in this line until explicitly reopened.

---

## Milestone 2026-09-16 — D1 query-presence localization: INVALID

### CURRENT BEST DIAGNOSIS

The frozen D1 instrument did **not** pass its positive-control bar, so it does
not license A, B, or COMPOSITION. Descriptive pattern is unchanged and sharper:
a near-hard prev-token head (mass ≈ 0.98 in every stratum) copies the query when
the query is the previous token and copies filler otherwise. At gap ≥ 31,
patching the other twin's gen residual into the generation position changed
**0/346** candidate argmaxes; full-vocab logits are injective-unembed identical
(cosine 0.99998). Official verdict remains INVALID (short-gap
`flip_toward_donor` 0.243 < 0.25). Authoritative Baby remains v2R4 U16000
SHA `94b3a9da…17827`.

### EVIDENCE FOR IT

Parent SHA matched. Diagnostic SHA matched. Twins differed only at the query
token. SDPA vs reference 7.6e-6. `W_U` rank 640/640. Protocol hash
`ba181fe1…1ea35` recorded in preflight. See
`research/V010_QUERY_PRESENCE_D1.md` and
`runs/query_presence_d1/ADJUDICATION.json`.

### EVIDENCE AGAINST IT / CAVEATS

Positive-control miss is three pairs (88/362 vs 91). Post-hoc eligible-only
short-gap flip is 0.317; that definition was not frozen. Git freeze commit
failed (no user identity; config not changed).

### WHAT WAS FALSIFIED

Nothing at the frozen-adjudication level. Informally, hidden-A at gap ≥ 31 is
hard to square with 0/346 patched argmax changes on an injective unembed.

### WHAT REMAINS UNKNOWN

Whether a successor protocol with a pre-registered eligible-flip positive
control would validate and then return B. Not run.

### NEXT EXPERIMENT

None until the owner authorizes D1b or another protocol. Do not lower the 0.25
bar on this run. Do not train. Do not open TEST.

### WHY HIGH INFORMATION

D1 was the presence-vs-use split. Its frozen bar was missed by three pairs; the
long-gap table is still the highest-resolution picture of the residual we have.

---

## Milestone 2026-09-16 — D1b query-presence on fresh bodies: valid B

### CURRENT BEST DIAGNOSIS

On a new 144-body diagnostic (seed 140200), denied against panels/S1/S2/D1
inputs, the hashed parent has a **query-invariant generation residual** whenever
the query is not the previous token. A near-hard prev-token head (mass ≈ 0.98)
is the working circuit; it is not a query pointer. D1 stays INVALID. D1b is
**valid B** under the frozen eligible-flip instrument. Authoritative Baby
remains v2R4 U16000 SHA `94b3a9da…17827`.

### EVIDENCE FOR IT

Instrument: eligible short-gap flip 0.294 (n=265), residual cosine 0.958, prev
track 1.0. Long gap: residual cosine 0.99956, logit cosine 0.99994, final flip
0.0078, g31 changed 0.022, query-track 0.225. See
`research/V010_QUERY_PRESENCE_D1B.md`.

### EVIDENCE AGAINST IT / CAVEATS

g31 changed is 0.022, not 0; patches are not a complete no-op, but under the
B bar. Long-gap residual cosine 0.99956 is close to the 0.999 floor. Single
seed, single parent.

### WHAT WAS FALSIFIED

Hidden long-gap A on this parent (query identity already at gen_pos, unused).
D1's INVALID is not a reason to treat the residual as unidentified.

### WHAT REMAINS UNKNOWN

Whether a pointer-attention auxiliary on natural-gap keyed rows can write query
identity into the gen residual without destroying prev-token copy or language.
Not yet run.

### NEXT EXPERIMENT

P1 pointer-attention aux vs matched λ=0 control, licensed by this B. Not a T1
resume. Not a lowered D1 bar.

### WHY HIGH INFORMATION

B names the missing write: attention from generation position to a non-adjacent
query token. T1 already showed CE-at-short-gap does not create that write.


