# SF1 single-fact acquisition study

Classification: **STOP_REGRESSION**.

Completed 100/200 prospectively fixed updates (90 factual English, 10 binding). No readiness or v1.0 claim. No FINAL or sacred material accessed.

## Frozen intervention and provenance

Parent: Pilot1 SHA256 `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb`. Seed87011. English/binding scope: blocks0-3 and all specialized components frozen on English; all T13 parameters active on binding. Exact named parameter lists are in PARAMETER_SCOPE.json. Inactive parameter values and every existing optimizer-state field were compared after every English update and remained unchanged.

16 training items, four complete families; two actor pairs, two predicates, two objects per pair, balanced actor swaps. Heldout uses the opposite object pool per actor pair, so actor/object combinations are absent from training while vocabulary remains familiar. Candidates all four tokens. All 152 prompts across training and diagnostics are unique. Completed input length <=50 tokens. Final rendered semantic parser, negative tests, causal labels, token boundaries, repeated-batch identities, and restart round trip passed before training. Full prompts had zero exact/substring matches in the allowlisted nonsacred corpora/evaluations; this does not exclude semantic/constituent overlap.

Fixed 200 updates, 20 cycles of nine factual plus one binding; English batch32 is two copies of every training record. No sentence rehearsal was introduced. AdamW5e-5, wd.05, betas(.9,.999), eps1e-8, clipping2; correct causal answer-plus-EOS CE, no pairwise loss or normalization. Objective and masking pinned from the earlier v8 harness; binding implementation pinned from validated HR3/Pilot1 code. Binding pool unchanged (320 documents/80 quartets), each quartet scheduled twice. Readiness gates remain untouched.

Endpoint acquisition gate: 16/16 restricted correctness AND16/16 exact candidate+EOS AND8/8 reversals AND4/4 complete families. Fixed endpoint selection only. Regression checks at100/200: each pool >=76/80 answers and BD, zero collapse; aligned sentence DEV loss no more than parent+0.25 nats. Stop on either regression. No transfer inference before successful endpoint gates.

Bundle: `C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011`; receipt SHA256 `45d6a1ae10dc37f6fe3dad8d48dbdc71fe9bd48b763f02d77687918cf6c4b5d1`; payload manifest SHA256 `0c06c30dee015a4f55f5b672c80dc7715ea541bd2dc096ddf64f758c81afa4f9`.

## Acquisition and language monitoring

| Update | Correct | Reversals | Families | Exact+EOS | Mean margin | Candidate mass | Aligned loss | PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9/16 | 1/8 | 0/4 | 0/16 | 0.05530 | 3.7005e-08 | 3.39066 | 29.686 |
| 100 | 12/16 | 4/8 | 2/4 | 12/16 | 0.38811 | 0.59099 | 4.65292 | 104.890 |

Mean supervised English loss, first10 updates: 3.77531; last10: 0.28761. These response/EOS training losses differ from the nonsacred sentence-language loss. Raw margins, candidate token scores including separate EOS, absolute candidate mass, top1/EOS diagnostics, and every unmodified generated response are durably preserved in the per-panel RAW files.

The preregistered regression guard stopped the run. Intermediate training-item performance does not satisfy the fixed endpoint gate. Transfer panels remained locked. The result is preserved without extending training or relaxing the cost/preservation guard.

## Both nonsacred binding pools — independently gated

| Update | Pool | Answer | BD | Collapse | Quartets | Strict reversal | Gate |
|---|---|---:|---:|---:|---:|---:|---|
| 0 | pilot0 | 80/80 | 80/80 | 0 | 20/20 | 40/40 | PASS |
| 0 | pilot1 | 80/80 | 80/80 | 0 | 20/20 | 40/40 | PASS |
| 100 | pilot0 | 80/80 | 80/80 | 0 | 20/20 | 40/40 | PASS |
| 100 | pilot1 | 80/80 | 80/80 | 0 | 20/20 | 40/40 | PASS |

Queried-row and downstream correctness conditional on BD are preserved in each checks JSON. Pool scores are never averaged.

## Frozen transfer and copy diagnostics

HELDOUT16: NOT OPENED. ALTERNATE48: NOT OPENED. COPY8: NOT OPENED. COMPETING64: NOT OPENED. These panels remain frozen and unscored because the prerequisite did not pass. No copy/transfer conclusion is inferred from training items.

## Interpretation limits and next boundary

The study isolates a simpler factual response interface with the existing causal answer objective and protected Pilot1 scope. No naked discrimination loss, decoder penalty, extra blocks, architecture change, or broader corpus strategy was introduced. Success on one-name facts may be an identity-copying shortcut; the matched control diagnostics must be considered before any stronger claim. Failure would not establish architectural impossibility, a capacity ceiling or a unique need for discrimination/unfreezing. A materially different objective, scope or curriculum requires a new scientific decision; none is selected here.

Results on training families are not independent heldout evidence and have no generalization confidence interval. Transfer items have correlated family structure; raw family/reversal profiles and subgroup counts are the evidence. No gates were adjusted after outcomes. Parameter-scope monitoring and inactive optimizer-state comparisons completed on every executed update. This is a bounded single-fact study only.

## Saved-output failure diagnosis

At update100, mean correct-target NLL at response positions1..4 then EOS is: [1.05594727024436, 0.005138540436746553, 0.006746029481291771, 0.004714174763648771, 0.003575779192033224]. The training objective averages these five positions; this explains why a small response/EOS loss is not equivalent to perfect selection. It does not establish that the averaging rule is causally responsible for the failure.

- TRAIN:g0:found: 2/4 correct; exact decoded response frequencies {' Alex.': 4}.
- TRAIN:g0:carried: 2/4 correct; exact decoded response frequencies {' Alex.': 4}.
- TRAIN:g1:found: 4/4 correct; exact decoded response frequencies {' Mia.': 2, ' Nora.': 2}.
- TRAIN:g1:carried: 4/4 correct; exact decoded response frequencies {' Mia.': 2, ' Nora.': 2}.

These are measurements on already-seen training records. They do not establish heldout semantic transfer or distinguish copying the sole name from factual understanding. No confirmed implementation defect emerged: next-token alignment, identities, schedules, parameter scopes and frozen optimizer inactivity were checked. The run hit a scientific cost guard, not a mechanical execution failure.

The next boundary is language retention while teaching the single-fact interface. Adding language rehearsal, changing factual exposure or learning rate, or changing response weighting would be scientifically different interventions. The present result does not uniquely choose among them, so no corrective training is authorized by the mechanical-defect exception. The 200-update endpoint was not reached; this result cannot show what unchanged continued training would have done. Gates were not weakened to find out.

## Checkpoint identities

- Update100: `C:\DaveLM-CADAVER\single_fact_acquisition_sf1_seed87011_run\checkpoint_100.pt`; SHA256 `550ce4306b450c2bdb64e57314e45ec7db5bb2ee68fc0978ed9d3617f4fc1e7e`.

Parent hash and every frozen payload checksum were reverified after completion. No mechanical execution corrections were required. The permanent checkpoints and one rolling restart retain provenance; no history was overwritten.
