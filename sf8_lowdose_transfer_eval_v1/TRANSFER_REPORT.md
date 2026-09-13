# SF8 low-dose (lambda_margin=0.25) first post-acquisition transfer evaluation

**Classification: MIXED_TRANSFER — replicated HELDOUT recombination transfer, surface/order-limited elsewhere.**

## 1. Transfer evaluation status
TRANSFER PREFLIGHT: PASS. Three authoritative SF8 lambda=0.25 checkpoints located and identity-verified
(STATUS.json ACQUISITION_SUCCESS@200 + on-disk SHA-256 match). Panels HELDOUT16/ALTERNATE48/COPY8/
COMPETING64 located in the sealed SF1 bundle, hashes matched against the SF1 SHA256SUMS manifest, and
eligibility established from the frozen SF1 PROTOCOL evaluation_order: "Only if endpoint acquisition
AND language/binding checks pass: HELDOUT16, ALTERNATE48, COPY8, COMPETING64. All frozen before training."
SF1 final_access=false, sacred_access=false; FINAL/sacred untouched. Control arms NOT scored: SF8 controls
failed acquisition, so the frozen unlock condition does not apply to them. Evaluation ran read-only
(m.eval()/no_grad), no optimizer, no writes inside SF8/SF1 directories.

## 2. Checkpoint identity
| Seed | Path | SHA-256 (matches STATUS.json) | Classification |
|---|---|---|---|
| 87017 | sf8_margin_dose_comparison_v1\runs\seed_87017_low\checkpoint_200.pt | (in RESULTS.json/PROVENANCE.json) | ACQUISITION_SUCCESS |
| 87018 | ...\seed_87018_low\checkpoint_200.pt | same | ACQUISITION_SUCCESS |
| 87019 | ...\seed_87019_low\checkpoint_200.pt | same | ACQUISITION_SUCCESS |

## 3. Eligible panels
| Panel | n | Frozen scoring semantics | Eligibility |
|---|---|---|---|
| HELDOUT16 | 16 | FAMILIAR actor/object recombinations (novel pairs, trained template). SF2 panel scoring: two-candidate forced-score margin>0 = correct; greedy name+EOS = exact; reversal/family per pair/family. | SF1 protocol: post-acquisition, frozen pre-training |
| ALTERNATE48 | 48 | Same heldout semantics, alternate surfaces: active:qa, passive:cloze, passive:qa (16 each). | same |
| COPY8 | 8 | Isolated card "copy the name" interface control (trained actors/objects). | same |
| COMPETING64 | 64 | Fact+contradicting-card contexts; fact-vs-copy queries, card-before/after order. | same |

## 4. Results by checkpoint (per frozen SF2 panel scoring; summarized from update200 RESULT files)
| Seed | HELDOUT c/x/rev/fam | ALTERNATE c/48 (x) | COPY c/8 (x) | COMPETING c/64 (x) |
|---|---|---|---|---|
| 87017 | 16/16 c, 11 x, 8/8 rev, 4/4 fam | 37 (0) | 5 (0) | 30 (20) |
| 87018 | 16/16 c, 10 x, 8/8 rev, 4/4 fam | 35 (0) | 5 (0) | 31 (20) |
| 87019 | 16/16 c, 11 x, 8/8 rev, 4/4 fam | 36 (0) | 5 (0) | 31 (20) |

Subgroups (identical pattern in all three seeds):
- ALTERNATE: active:qa 15-16/16 correct; passive:cloze 9-10/16; passive:qa 11-12/16; exact=0 everywhere.
- COMPETING: fact_order0:fact 16/16 correct AND greedy-exact 16/16; fact_order0:copy 1-2/16; fact_order1:fact 4-5/16; fact_order1:copy 8-9/16.
- COPY: a0 (card name = trained binder) 4/4 correct; a1 (card name != trained binder) 0/4; exact 0/8.

## 5. Aggregate / replication view
The outcome pattern is near-identical across all three independently trained checkpoints on every panel and
every subgroup (identical failing item sets for HELDOUT exact and COMPETING o1:fact). Replication is strong.

## 6. Reversal/family/surface/distractor analysis
- HELDOUT: reversals 8/8, families 4/4 in all three seeds (two-choice); greedy name+EOS exact only 10-11/16.
  Every exact-miss has correct=True with margin +2.6..+4.1 nats but greedy top-1 token is an out-of-candidate
  default name (e.g., Mia emitted for Alex-held round plate; Alex for Mia-held small drum) - the same 5-6
  items in all three seeds. Unrestricted greedy first-name accuracy is therefore 10-11/16, not 16/16.
- ALTERNATE: interface entry fails under free generation (greedy continues story-style; top-1 name emitted on
  ~0/48-1/48 items; exact 0/48) even where forced two-choice name preference is 35-37/48 (active:qa 15-16/16).
- COPY: card-copy semantics absent: free generation never enters the name form; forced preference follows the
  TRAIN binding default rather than the card (0/4 when card contradicts the binding).
- COMPETING: counterfactual suppression works only in canonical order: when the fact clause is most recent,
  Baby names the fact actor and ignores the contradicting card 16/16 (all seeds, greedy-exact too). When the
  card clause is most recent and the fact is queried, Baby names the card (recency-driven; 4-5/16). Copy
  queries are not executed under free generation.

## 7. Failure map (concise)
1. Out-of-candidate default-name mass leak: recombined HELDOUT items where the object's trained-side default
   name differs from its new holder -> greedy top-1 leaks to that default name (~5-6/16, deterministic,
   replicated); two-choice separation itself is maximal (2.6-4.4 nats).
2. Template lock: any surface outside the trained "X verbed the Y.\nThe person who verbed the Y was" cloze
   (active/passive questions, card-copy) -> greedy leaves the answer interface and continues story text.
3. Recency/order dependence under contradiction: competing card name wins when it is the last-mentioned name.
4. No copy-of-other-source behavior: card content is never copied when it contradicts the trained binding.

## 8. Dave-coded interpretation
- Did Baby transfer beyond TRAIN16? YES, within the trained template family: all 16 novel actor/object
  recombinations answered correctly under the frozen two-choice scoring, with full reversal consistency, in
  all three independently trained Babies - and in the canonical order Baby ignores an explicitly contradicting
  card (16/16). This is not TRAIN16 memorization and not a fixed-name trick.
- Which held-out examples? New combinations of familiar actors/objects in the exact trained sentence form;
  active-voice "who" questions partially at the name-preference level (15-16/16); passive forms weaker.
- Did all three show it? Yes - the same items succeed and the same items fail in all three seeds.
- What broke? (1) Under free generation Baby sometimes emits the object's old default name instead of the
  correct holder for ~1/3 of recombined items (two-choice is still perfect); (2) every other sentence form
  (question wording, passive, card-copy) drops Baby out of the answer interface into story mode; (3) when the
  contradicting card is mentioned last, Baby answers from the card.
- Memorization vs transfer? Consistent with a template-scoped factual-selection capability that genuinely
  recombines familiar actors/objects, NOT with general reading comprehension or instruction following.
- Strongest claim now: with the lambda=0.25 recipe, Baby acquires a reliable, replicated, held-out-recombining,
  reversal-consistent factual-selection behavior within its trained interface, robust to an explicit
  contradicting distractor in canonical order.
- Still bullshit: any claim that Baby answers facts across surface forms, follows arbitrary instructions
  (copy), reasons order-independently under contradiction, or has general question-answering/comprehension.

## 9. Provenance
Evaluator: TRANSFER_EVAL.py (read-only; engine SF2_ENGINE.py reused byte-identical from sealed SF8).
Raw per-item outputs: raw/seed_*_low/*_RAW.jsonl + *_RESULT.json. RESULTS.json and PROVENANCE.json contain
checkpoint hashes, panel hashes, and the output manifest SHA-256. FINAL/sacred panels never accessed.
