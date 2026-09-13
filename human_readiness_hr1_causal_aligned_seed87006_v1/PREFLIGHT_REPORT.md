# HR-1 causal-aligned successor preflight

Status: CAUSAL_ALIGNMENT_PREFLIGHT_PASS_NO_TRAINING

This is a prospective, pre-training correction of the HR language objective. No checkpoint was loaded, no optimizer was created, no update was performed, and no behavior was evaluated.

## Root cause

The HR-1 v8 and HR-2 v7 trainers and their DEV helpers put the target slice at y[i,1:len(z)] while the input starts at position zero. For z=[BOS]+content+[EOS], that layout teaches current-token copying at positions 1..n and pairs the final EOS with padding. The forensic report's aligned recomputation and rollout traces support this as the direct implementation error; no architectural conclusion follows.

The corrected source uses the validated P5/P7 convention in both train and DEV:

    x[:len(z)-1] = z[:-1]
    y[:len(z)-1] = z[1:]

## Parent verification

The explicit parent is immutable Pilot 1:

C:\DaveLM-CADAVER\language_pilot_1_early_block_protection_seed8380\pilot_run\checkpoints\seed_8380\latest.pt

SHA-256: 2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb.

Its lineage is documented by the Pilot-1 material manifest, completion record, and training specification copied by hash into this bundle. Those records identify canonical Graduate (fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430) as Pilot 1's starting checkpoint and the hash above as its final checkpoint. The historical HR-1 protocol's Graduate parent is retained as historical evidence; it is not silently rewritten.

## Alignment sanity check

Using the approved tokenizer on the deterministic text "A cat sat." gives token IDs [37, 268, 265, 269, 265, 18], which decode exactly back to the source. With z=[2]+ids+[3], the scored transitions are:

(2,37), (37,268), (268,265), (265,269), (269,265), (265,18), (18,3).

The final padded position has input 0 and label -100. The validator rejects the historical shifted target layout. No intermediate EOS is present.

## Proposed successor treatment

The materialized HR-1 v8 English records, split, schedule, binding rehearsal pool, binding schedule, optimizer, 500-update cadence, scopes, clipping, deterministic runtime, DEV points, and binding gates are copied byte-for-byte. The only intended objective correction is causal target alignment, with Pilot 1 as the parent. Training seed 87006 is fixed for this prospective run.

- 500 updates: 450 English and 50 binding, in 50 repetitions of nine English then one binding.
- English batch size 64; binding batch size 32; literal inherited schedules, with no resampling.
- English mean full-vocabulary CE supervises all content targets and the one final EOS; context/BOS/padding labels are ignored; no EOS is inserted between prompt and response.
- Binding uses the pinned established Pilot-1 answer CE plus hard-min permutation-invariant localization objective (LAM 1.0536573711078283).
- English scope: embeddings, positions, blocks 4-7, final norm, and language head train; blocks 0-3 and localization/retrieval freeze. Binding restores the established full T13 scope.
- AdamW: lr=5e-5, betas=(0.9,0.999), epsilon=1e-8, weight decay=0.05, no scheduler/amsgrad/foreach/fused; global clip 2.0; one optimizer persists across all update types in an arm.
- Development evaluation is at updates 0,100,500 only, on nonsacred development material.

The HR-2 unlikelihood term is not carried forward: because its target mask was shifted, retaining it would not be a specification-preserving correction of the intended causal treatment.

The final source also includes one mechanical path-wiring correction found during preflight: the protocol's bundle-relative binding pool path is resolved from the bundle root exactly once. No data or objective value changed.

## Prospective endpoints and stopping rules

Before any update, the aligned loss and fixed nonsacred language diagnostics are compared with Pilot 1. A successful causal-language improvement requires lower correctly aligned DEV loss together with improved fixed nonsacred continuation/fact-selection behavior; loss alone is insufficient. The frozen development criteria are: non-EOS at least 80% of 20, complete human-readable sentences at least 70%, prompt relevance at least 60%, controlled fact selection at least 70% with at least 8/10 reversals, elementary instruction at least 70%, and two-turn continuity at least 60%.

Update 100 and 500 checkpoints are both retained. The first meeting all frozen nonsacred language criteria without binding failure is the development candidate. If neither does, classify the treatment as a failure. A binding failure on either nonsacred pool (answer <76/80, BOTH_DISTINCT <76/80, or any collapse) is unacceptable binding damage and stops the run; no retrospective threshold change or rescue is allowed. A loss-only change, or a language gain with binding failure, does not establish the target milestone.

Both Pilot-0 and Pilot-1 nonsacred binding DEV pools are evaluated separately; they are never averaged. The final readiness battery and all sacred material remain inaccessible.

## Artifacts and checks

- HR1_CAUSAL_ALIGNED_PROTOCOL.json — frozen prospective specification.
- sources\CAUSAL_ALIGNED_TRAIN_REAL.py — corrected successor trainer source.
- sources\CAUSAL_ALIGNMENT.py — small pure alignment reference used by checks.
- data\ — byte-for-byte inherited English records/schedules and binding pools.
- checks\ALIGNMENT_SANITY.json — token transition and EOS/padding checks.
- checks\OBJECTIVE_EQUIVALENCE.json — train/DEV expression equivalence check.
- PREFLIGHT.json — machine-readable validation receipt.
- PROVENANCE.json — source, parent, tokenizer, runtime, and access provenance.
- CORRECTION_DIFF.md — additive historical correction record.

The known-good runtime is Python 3.12.14, torch 2.12.0+rocm7.14.0, tokenizers 0.23.1, with CUDA available. The validator imports runtime packages only; it does not load the parent or construct an optimizer.

## Scope boundary

This preflight does not train, infer, choose a later parent, access the final readiness battery, or create HR-3 results. It establishes only that the corrected causal objective and the otherwise inherited HR treatment are materialized and auditable before a separately authorized execution.
