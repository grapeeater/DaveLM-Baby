# T31 Native Source-to-Token Copy Bridge — Frozen Protocol

Status: SEALED_PROSPECTIVE (2026-09-14)
Scientific question: Can an explicit ordered source-to-token vocabulary path convert Baby's retrieved source identity into reliable native greedy answer tokens?

## Single intervention
Relative to the Phase1G/T24-T30 base runner, add T31CopyModel.copy_gate, four scalar parameters initialized exactly zero. At each answer step j=0..3, the pointer-weighted distribution over the two TRAIN/DEV candidate token sequences (frozen tokenizer answer-context IDs, name plus period) is converted to centered log copy evidence and added to the ordinary LM vocabulary logits: z_final[j] = z_lm[j] + copy_gate[j] * (log p_copy[j] - mean_v log p_copy[j]). Gate zero is exactly neutral. The pointer weights are computed from query-to-fact-clause cosine similarity. Candidate token order is preserved; EOS is produced by the ordinary LM path after the copied name+period span. No candidate list, oracle prefix, reranking, or protected data is used.

## Frozen lineage and recipe
Parent: Phase1G U6000 best.pt, SHA256 c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1.
Tokenizer: DaveLM v0_7, SHA256 e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b.
Seeds: 880001, 880002, 880003 (fresh, one per branch).
Architecture/objective/schedule inherit T30/T24 exactly: 1000 updates, 900 QA + 100 language (9:1), Phase A blocks 0–3 frozen through U750, Phase B restores/freezes blocks 8–11 after U750, AdamW lr 3.75e-5 betas .9/.999 eps 1e-8 wd .05, clip 2.0, batch 32 via microbatch 8, lambda_ptr=1, lambda_ans=1, lambda_unl=1, margin=1, full answer CE+EOS, inventory-wide unlikelihood, no suffix/fork loss. Binding/localizer stays frozen and parent-identical. Only copy_gate is new/trainable.

## Gates
Language: DEV CE <= min(1.30,U0+0.25), gap <= .5, finite.
Representation: pointer >=96/128, pointer reversals >=48/64, complete pointer families >=8/16, name-disjoint >=48/64, template-disjoint >=48/64.
Native endpoint: forced choice >=96/128, native reversals >=48/64, exact+EOS >=80/128.
Acquisition: TRAIN16 forced 16/16, exact 16/16, reversals 8/8, four families complete.
Protected: T3 TEST SEALED_UNOPENED; T2-EVAL-TEST, FINAL, sacred locked.
Hard stop: nonfinite, parent/tokenizer/schema mismatch, frozen binding/early-block mutation, protected access, sequence mismatch, output collision, language catastrophe.
Futility: at U750 only, if copy_gate L2 > .01 AND exact <=20/128 AND shared-prefix exact =0 AND first-token-correct-then-diverge >=55, stop as T31_FUTILITY_STOP. This is conservative and below every known completed T28–T30 U1000 native trajectory; no known historical false kill.
Sequential replication: run seed 1 then seed 2. If both independently terminal-fail, skip seed 3; if both independently full-success, seed 3 may be skipped; disagreement requires seed 3. No rescue or replacement seed.

## Evaluation
U0, U100, U300, U500, U750, U1000 where reached; DEV only plus TRAIN16 and language/binding checks. Report exact+EOS, shared/unique exact, divergence, first-disambiguating token, post-fork completion, pointer/native forced-choice/reversals/families, language CE/gap, binding integrity, copy_gate L2/mean and source-copy utilization. TEST/FINAL/sacred are never loaded.

## Provenance
All code, data, protocol, parent, tokenizer, and run outputs are hashed in SHA256SUMS.txt and recorded in SEAL_RECEIPT.json. No historical artifact is modified.
