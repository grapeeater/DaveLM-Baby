# HR-1 causal-aligned successor — preflight only

This is a prospective corrected successor from immutable Language Pilot 1. It has not loaded a checkpoint, created an optimizer, or performed an update. The final readiness battery remains sealed.

## Root cause corrected

The HR-1 and HR-2 English paths wrote `x = z[:-1]` at positions starting at zero but wrote `y = z[1:]` beginning at position one. Thus positions containing a token were supervised to reproduce that same token, and the final EOS was paired with zero padding. Their DEV helper used the same layout. The corrected implementation uses the aligned expression `x[:len(z)-1] = z[:-1]` and `y[:len(z)-1] = z[1:]` in both training and DEV.

For content IDs `[t1,...,tn]`, `z=[2,t1,...,tn,3]`; the scored transitions are `2→t1`, `t1→t2`, ..., `tn→3`. Padding labels are `-100` and never contribute to loss. There is one BOS and one final EOS, with no intermediate EOS.

## Preserved treatment

The materialized HR-1 v8 sentence data, sentence boundaries, train/DEV split, English schedule, Pilot-1 binding rehearsal pool, binding schedule, optimizer, 500-update 9:1 cadence, batch sizes, scopes, clipping, deterministic runtime, DEV points, binding gates, and sealed-material restrictions are copied byte-for-byte. The only prospective scientific changes are the required Pilot-1 parent and the causal target alignment. HR-2's unlikelihood term is not carried forward because its shifted objective made that treatment uninterpretable.

Pilot 1 is identified by SHA-256 `2281d20ebd3ae9c5c7bdf8af92518e6ca50a9d777f6b336fb89891932a2a12cb`. Its material provenance is the Pilot-1 manifest and completion record in the parent directory.

## Objective and schedule

There are 500 updates: 450 English updates and 50 binding rehearsals in 50 cycles of nine English updates followed by one binding update. English batches have 64 materialized records; binding batches have 32 documents. No schedule is regenerated. English uses mean full-vocabulary causal cross-entropy over all content targets and the final EOS. Binding uses the pinned established answer CE plus hard-min permutation-invariant localization objective with LAM `1.0536573711078283`.

English updates train embeddings, positional embeddings, blocks 4–7, final normalization, and the language head. Blocks 0–3 and T13 localization/retrieval parameters are frozen. Binding restores the full established T13 scope. AdamW is fresh per arm and persists across all updates: learning rate `5e-5`, betas `(0.9,0.999)`, epsilon `1e-8`, weight decay `0.05`, no scheduler, and global gradient clipping `2.0`.

Development evaluation occurs only at updates 0, 100, and 500. Both nonsacred binding DEV pools are reported separately and must each meet answer `>=76/80`, BOTH_DISTINCT `>=76/80`, and zero collapse. Language diagnostics retain the existing development criteria: non-EOS `>=80%`, complete human-readable sentences `>=70%`, prompt relevance `>=60%`, controlled fact selection `>=70%` with at least `8/10` reversals, elementary instruction `>=70%`, and two-turn continuity `>=60%`. These are engineering diagnostics; they do not open or predict the sealed final battery.

## Prospective interpretation

Successful causal-language improvement means a lower correctly aligned DEV loss and improved fixed nonsacred continuation/fact-selection diagnostics relative to Pilot 1, with both binding pools passing. A loss-only improvement without better free-running behavior is insufficient. No meaningful improvement means the corrected aligned metrics and diagnostics remain near Pilot-1 levels after the frozen endpoint. Language improvement with either binding pool failing is classified as unacceptable binding damage and the run stops; no rescue training or final-battery access is permitted.

Checkpoint selection is fixed before training: update 100 and update 500 are preserved; the first checkpoint meeting the frozen nonsacred development criteria without binding failure is the development candidate, otherwise the endpoint is classified as failure. No post-hoc threshold changes are allowed.
