# PHASE2A T3 REBUILT STUDY — DECISION LEDGER

Recorded: 2026-09-11. Authority: physical artifacts over prior prose.

## Decision
**A — rebuild and run Treatment 3** as a complete prospectively frozen experiment.
Not binding/localizer training. Not another diagnostic.

## Authoritative state cited
- Parent: Phase1G U6000 `best.pt` SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`
- Tokenizer SHA256 `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`
- Pretreatment: `phase2a_final_pretreatment_diagnostic_v1` → `D1_LINEAR_RELATIONAL_SIGNAL_ABSENT`
- T2: `PHASE2A_T2_FAIL` at U100 (`ab_no_progress`); T2-EVAL-TEST unopened
- Committed T3 v1: `T3_HARD_STOP` (`T3_FINAL_RECEIPT.json`) — three seeds, identical checkpoint SHA `1a4d5211…`
- Repair: `PHASE2A_T3_INFRASTRUCTURE_BLOCKED` — missing TRAIN16, nonsacred binding pools, complete evaluator
- ctxbind T3 `audits.json`: train/dev/test **name overlap** (Ann/Zoe/Noah/Sara/… in every split), violating the disjoint-name protocol
- vNext binding: 984,321 params present, `layout=None` disconnects them from ordinary scoring; Phase1G R1 already `NOT_APPLICABLE_UNTRAINED`

## Why T3, not the dormant binder
T3 is the unfinished prospectively designed intervention after D1: parameter-free query→candidate pointer on base `final_norm`, plus T2 native margin, binding frozen. No valid test of that hypothesis exists. Binding integration is a different architectural treatment and would abandon T3 without answering it. Historical T12/T13 success was synthetic-geometry routing, not evidence that the binder should jump the queue on this parent.

## Prospective closures (infrastructure, not hypothesis change)
1. Fresh T3 corpus: family/name disjoint train/dev/test; assignment×query×fact-order orbits; four templates with train using two and DEV/TEST holding two out for the 48/64 template-disjoint gate.
2. Fresh `T3_ACQ16` in-train acquisition panel (4 train families × assignment×query at fact_order=0). Do not inherit an ambiguous SF/T2 TRAIN16.
3. Binding preservation = parent-byte identity of all non-`base_model` parameters. No invented behavioral binding pool.
4. Complete evaluator: pointer, native forced-choice, exact+EOS, assignment reversals, complete families, name-disjoint, template-disjoint, shortcut audits, language CE, ACQ16, binding identity. TEST hashed and unopened until a valid DEV terminal commit.
5. Seed-specific QA permutation (seeds 620001–620003). Classification is computed from frozen gates, never hardcoded.

## Not authorized
Locked T2-EVAL-TEST, FINAL, sacred, historical transfer panels. No T2 descendant parent. No binding-module training in this study.
