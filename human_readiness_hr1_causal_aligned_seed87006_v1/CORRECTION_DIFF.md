# HR causal-alignment correction record

This record is additive. The historical HR-1 and HR-2 bundles are preserved byte-for-byte.

## Root cause

The HR-1 v8 and HR-2 v7 English builders used x = z[:-1] at position zero but assigned y = z[1:] beginning at position one:

    y[i,1:len(z)] = z[1:]

Their development helpers used the same assignment. Consequently, the token already present at positions 1 through n was the supervised target at that position, while the final EOS was paired with a padding input. The HR-2 unlikelihood mask was derived from this shifted target layout as well.

## Mechanical correction

The successor source uses the single causal alignment used by the independently validated P5/P7 trainers:

    x[i,:len(z)-1] = z[:-1]
    y[i,:len(z)-1] = z[1:]

The same expression is used in the English training step and the English DEV loss helper. For z = [BOS] + content + [EOS], position j therefore predicts the token at j+1. Padding labels remain -100; BOS, context-only positions, and padding are ignored according to the frozen objective. No intermediate EOS is introduced.

## Scope

The intended HR-1 treatment data, schedule, optimizer, parameter scope, binding rehearsal/objective, update count, DEV points, gates, and sealed-material restrictions are copied unchanged. The prospective parent is explicitly Pilot 1, as required by the forensic result. HR-2's unlikelihood term is not carried into this correction because its shifted objective made that treatment uninterpretable; no new loss choice is introduced.

## Source authority

- Shifted HR-1 source: C:\DaveLM-CADAVER\human_readiness_hr1_seed87004_v8\HR1_TRAIN_REAL.py, SHA-256 ef42d9e600e714eb3e1dfbe1e5f08285fd8465a7d51839f0d03e87d51e417e17.
- Shifted HR-2 source: C:\DaveLM-CADAVER\human_readiness_hr2_seed87005_v7\HR2_TRAIN.py, SHA-256 41560c549b56cd9695ca3c997da240c12b93f05ce08b112e3ee7dc1e22eed809.
- Independently validated alignment reference: C:\DaveLM-CADAVER\language_compositional_p7.py, SHA-256 bab7f60082bd38201454f256d5eaeaf79afc22fae187b1a2d85ce0da83a46f21.
- Independent second reference: C:\DaveLM-CADAVER\language_sentencebound_p5.py, SHA-256 851bf4a26bb9f81ede0f224e1aecbc9e651d0d79a7b31d669e1aa8c95e10f2db.
- Corrected successor source: sources\CAUSAL_ALIGNED_TRAIN_REAL.py; its SHA-256 is recorded in PROVENANCE.json.

## Validation

PREFLIGHT_VALIDATE.py performs the deterministic token transition test, rejects the old shifted assignment, verifies that both training and DEV contain the aligned CE expression, checks inherited data/schedule bytes, checks parent and tokenizer hashes, and confirms that no checkpoint or optimizer is used. checks\ALIGNMENT_SANITY.json and checks\OBJECTIVE_EQUIVALENCE.json contain the machine-readable results.

## Path-wiring correction

The first corrected-source draft prefixed the protocol's already bundle-relative binding path with data a second time. The final source resolves R / proto["data"]["binding_rehearsal"], while the English and binding schedules remain under the explicit R / data paths. This changes only file resolution and is validated by the source-wiring check; it does not change data, schedule, or objective semantics.
