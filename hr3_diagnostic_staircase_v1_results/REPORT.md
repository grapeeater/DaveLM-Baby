# Frozen diagnostic staircase: provenance, training acquisition, single-fact selection

Verdict: the proposed two-fact Balanced-versus-Discrimination comparison is not yet justified on the proposed Pilot1 starting point. Pilot1/corrected-HR1/HR3 fail the prospective single-fact acquisition prerequisite. The earlier factual descendant is considered separately below; partial or format-specific performance is not promoted to general acquisition. No training objective or coefficient was selected.

## Stage 0 — identity and historical reconstruction

The two checkpoint hashes identify different pilots, not different Pilot1 descendants. Both physical files match their original completion records and the historical English-characterization registry.

- Pilot1: `language_pilot_1_early_block_protection_seed8380/pilot_run/checkpoints/seed_8380/latest.pt`; SHA256 `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb`.
- Pilot0: `language_pilot_0_tinystories_seed8380/pilot_run/checkpoints/seed_8380/latest.pt`; SHA256 `769bd01efd28e7888064c0d5fe1dd0d85a4344e2039aef04bcbf7d9e906f47f5`.
- Both original completion records identify the canonical Graduate hash `fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430` as their starting checkpoint. The independent claim labeling 769bd01 as Pilot1 is inconsistent with these preserved artifacts. No canonical record was rewritten.

The earlier seed87002 factual treatment is the preserved v8 factual checkpoint, SHA256 `3ae30ae847d3129d7715173ddbfbbaf6b7490772dc850e361a5a2d785b991123`. Its rolling state records the verified Pilot1 parent hash, completed_update=500, the matching frozen protocol and literal schedule hashes. Every endpoint model tensor equals the rolling state. Existing optimizer counters are {50,500}, consistent with binding-only and shared groups; no optimizer was instantiated for this inspection. The standalone endpoint stores step=500 and arm=factual but lacks embedded parent hash; rolling provenance supplies that link.

Its factual training pool contains 384 records: 24 object-discrimination and 24 predicate-discrimination families, eight records each (assignment x query x fact order). Every record contains TWO factual sentences and a declarative cloze cue. Object families distinguish descriptions under the same predicate; predicate families distinguish predicates applied to the queried description. Answer keys were independently recovered from final text for all 384 training and 96 DEV items. The matched control has irrelevant two-fact contexts with balanced practice targets; it is not the dataset used for Stage1 measurement.

Training used 500 updates, 450 English/50 binding, nine English then one binding. English batches contain 32 records, two whole families from each stratum. The literal schedule repeats each family 37 or 38 times. English supervises four candidate tokens and final EOS only; context ignored. Inputs=sequence[:-1], labels=[ignore]*(BOS+prompt length-1)+candidate+EOS: correctly causal, unlike the original shifted HR copying objective. AdamW lr5e-5, betas(.9,.999), eps1e-8, weight decay .05, no scheduler, clipping2.0. English freezes blocks0-3 and specialized modules; binding restores full T13 scope with answer CE and pinned hard-min localization.

The preserved run_factual directory has no per-update loss log. DEV_100/DEV_500 store a status only, not DEV scores; the controller dev_eval implementation returns that status without model scoring. Historical Primary factual results were 97/192 correct, 22/96 reversals, 0/24 complete families, and 74/192 exact greedy responses; acquisition failed. They do not establish training-distribution acquisition. The old evaluator also selects control-arm contexts for its control checkpoint, so those historical control numbers must not be treated as evaluation on the identical factual prompts. No historical results are modified here.

HR1-aligned and HR3 are different treatments: independent TinyStories sentences, 28,800 presentations, no designed two-fact English/QA task. HR3 additionally unfreezes block3; its English data do not become two-fact QA.

## Stage 1 — all preserved factual training items and factual DEV

The sealed historical evaluator is hardcoded to Primary and is not a train evaluator. The new frozen read-only evaluator uses the same candidate sequence likelihood and causal offsets; all 480 input/label records were checked against the authoritative v8 harness before inference. No EOS is included in candidate ranking. Reversal pairs are counted once.

| Set/stratum | Correct | Reversals | Complete families | Exact candidate then EOS | Correct first-token top1 |
|---|---:|---:|---:|---:|---:|
| training | 193/384 | 38/192 | 0/48 | 168/384 | 168/384 |
| training/object | 92/192 | 18/96 | 0/24 | 82/192 | 82/192 |
| training/predicate | 101/192 | 20/96 | 0/24 | 86/192 | 86/192 |
| factual_dev | 46/96 | 8/48 | 0/12 | 42/96 | 42/96 |
| factual_dev/object | 22/48 | 4/24 | 0/6 | 20/48 | 20/48 |
| factual_dev/predicate | 24/48 | 4/24 | 0/6 | 22/48 | 22/48 |

| Set | Mean sequence margin | Mean first-logit margin | First candidate mass | Four-token sequence mass | EOS top1 | Outside-candidate top1 (includes EOS) |
|---|---:|---:|---:|---:|---:|---:|
| training | -0.00144 | -0.00145 | 0.69564 | 0.6945 | 0/384 | 46/384 |
| factual_dev | -0.00166 | -0.00168 | 0.68854 | 0.68727 | 0/96 | 6/96 |

This directly distinguishes failure to acquire the trained selection rule from a purely held-out transfer failure. Candidate production and stopping can be learned while choosing the factual answer remains unreliable. These are new retrospective measurements on the existing endpoint, not a reconstructed training trajectory.

## Stage 2 — frozen single-fact test

64 items; four lexical/relation families of 16; 32 actor reversals; Alex/Owen and Mia/Nora; found/carried; red ball/blue book. Active and passive facts crossed with cloze and QA cues (16 items each format). For each fixed object, wording and cue, the actor flips and the correct answer flips. Each of four names is correct 16 times, candidate slots correct 32 times each, each candidate is exactly four tokens. All 32 reversal pairs have equal tokenized prompt lengths.

Examples of frozen templates: `<Name> found the red ball.
Who found the red ball? Answer:` and `The red ball was found by <Name>.
The person who found the red ball was`. The query names the relation and object, never the answer actor. Independent final-text parsing and negative key/predicate/query/extra-fact tests passed. Only the correct actor appears in the fact; success could be single-name copying and would not establish two-fact binding. This is the intended easier prerequisite.

The build uses a fixed Cartesian product, no random sampling or outcome-driven selection. It was byte-hashed and read-only before any staircase scoring. Complete prompt/answer round trips and continuation boundaries pass; completed inputs are <=256 tokens.

| Checkpoint | Correct | Reversals | Complete families | Exact candidate then EOS | Correct first-token top1 |
|---|---:|---:|---:|---:|---:|
| Pilot1 | 32/64 | 0/32 | 0/4 | 0/64 | 0/64 |
| HR1_aligned | 32/64 | 0/32 | 0/4 | 0/64 | 0/64 |
| HR3 | 32/64 | 0/32 | 0/4 | 0/64 | 0/64 |
| Factual87002 | 44/64 | 12/32 | 0/4 | 22/64 | 22/64 |

| Checkpoint | Format | Correct | Reversals | Exact then EOS |
|---|---|---:|---:|---:|
| Pilot1 | active:cloze | 8/16 | 0/8 | 0/16 |
| Pilot1 | active:qa | 8/16 | 0/8 | 0/16 |
| Pilot1 | passive:cloze | 8/16 | 0/8 | 0/16 |
| Pilot1 | passive:qa | 8/16 | 0/8 | 0/16 |
| HR1_aligned | active:cloze | 8/16 | 0/8 | 0/16 |
| HR1_aligned | active:qa | 8/16 | 0/8 | 0/16 |
| HR1_aligned | passive:cloze | 8/16 | 0/8 | 0/16 |
| HR1_aligned | passive:qa | 8/16 | 0/8 | 0/16 |
| HR3 | active:cloze | 8/16 | 0/8 | 0/16 |
| HR3 | active:qa | 8/16 | 0/8 | 0/16 |
| HR3 | passive:cloze | 8/16 | 0/8 | 0/16 |
| HR3 | passive:qa | 8/16 | 0/8 | 0/16 |
| Factual87002 | active:cloze | 14/16 | 6/8 | 10/16 |
| Factual87002 | active:qa | 10/16 | 2/8 | 2/16 |
| Factual87002 | passive:cloze | 11/16 | 3/8 | 8/16 |
| Factual87002 | passive:qa | 9/16 | 1/8 | 2/16 |

| Checkpoint | Mean sequence margin | Mean first-logit margin | First candidate mass | Four-token sequence mass | EOS top1 | Outside-candidate top1 |
|---|---:|---:|---:|---:|---:|---:|
| Pilot1 | 0.02595 | 0.00909 | 0.00024366 | 7.9321e-09 | 0/64 | 64/64 |
| HR1_aligned | -0.06871 | -0.00277 | 0.0016138 | 1.4705e-06 | 0/64 | 64/64 |
| HR3 | 0.06737 | -0.01715 | 0.0011546 | 5.4971e-07 | 0/64 | 64/64 |
| Factual87002 | 0.80876 | 0.80836 | 0.38204 | 0.3811 | 0/64 | 36/64 |

The raw artifact preserves every prompt, per-token log probability/logit/full-vocabulary rank, signed sequence and first-token margins, EOS rank/probability, and full greedy output. Candidate mass is an absolute full-vocabulary probability, not softmax restricted to the pair. Exact answer-prefix and exact candidate-then-EOS are separate fields. First-token top1 is not full multi-token answer exactness.

## Preregistered decision and smallest next branch

- Pilot1: APPROXIMATELY_CHANCE.
- HR1_aligned: APPROXIMATELY_CHANCE.
- HR3: APPROXIMATELY_CHANCE.
- Factual87002: INCONCLUSIVE_OR_MIXED.

The prospective clear-acquisition flag required >=56/64 restricted answers, >=24/32 reversals and >=12/16 in EACH format. Approximately-chance flag: 28..36/64 and <=4/32 reversals. Other patterns are mixed/inconclusive. These are small-battery operational flags frozen before scoring, not readiness gates, statistical population claims, or evidence of general language understanding. No IID confidence intervals are applied to 64 dependent items in four families.

The proposed two-fact objective comparison is deferred. The authorized Pilot1 parent has not demonstrated even this single-fact interface, and the earlier factual checkpoint has not reliably mastered its own two-fact training distribution. Partial or cue-specific gains in that descendant do not establish a prerequisite in Pilot1 or authorize silently changing the comparison parent.

The smallest supported next branch is a one-fact ordinary-English-to-answer acquisition study, with strict counterbalanced actor changes, one familiar cue initially, response-token-plus-EOS causal supervision and unchanged Pilot1/HR1 block mask/binding protection. Its first question is whether it can acquire its OWN single-fact training families, followed separately by held-out lexical combinations and alternate cues. A matched name-copy control would distinguish answer-format/identity copying from factual-interface acquisition. This identifies a diagnostic/treatment class only; parent changes, loss mixtures and a new run are not selected or implemented here. Stop at this scientific fork as requested.

If a future objective A/B becomes justified, both arms must share the verified Pilot1/corrected-HR1 block mask (blocks0-3 protected), not HR3 relaxation; identical strictly counterbalanced data; full-vocabulary correct-answer CE retained; candidate membership preserved; no naked pairwise margin. Any discrimination coefficient must be calibrated once at step0 on deterministic training-only gradient norms, frozen before DEV, and never tuned on outcomes. Paired binary/reversal statistics and effect-size reporting must be preregistered. Readiness and binding gates stay unchanged.

## Provenance and limitations

The first builder attempt rejected 12 prompts found as substring suffixes of earlier two-fact records. It was preserved as hr3_diagnostic_staircase_v1_build1_failed. Before any scoring, the overlap check was corrected to distinguish whole-record equality from substring occurrence; no item, wording, answer, candidate or family was changed. Both overlap measures are preserved. Zero single-fact prompts equal a complete audited record; 12 occur as suffixes in v8 material. Therefore the battery is not claimed as uncontaminated transfer evidence. TinyStories train/DEV and the four nonsacred readiness DEV files have zero whole-prompt substring matches. FINAL was excluded from all overlap audits.

The independent ledger itself was not supplied as a file. The conflicting identity claim is reconciled against authoritative completion records, physical hashes and the frozen historical checkpoint registry; no unseen ledger history is invented. The older controller has incomplete diagnostic/persistence implementations; preserved endpoint/rolling identities were checked rather than assuming every promised behavior occurred. Training metrics were not fabricated.

All new inference used ordinary base_model, eval mode, torch.inference_mode, no optimizer or gradient computation. Raw rows were durably appended before aggregation. Every checkpoint hash was checked before and after inference. Prior frozen artifacts were not changed. FINAL and sacred material were not accessed.
