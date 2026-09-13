# Preflight correction record

The first validator invocation failed its metadata comparison before any checkpoint loading or optimizer creation. Original TRAIN16 metadata stores object descriptions without the determiner (e.g. `small drum`); SF9 records include it (`the tin cup`). The actual prompts consistently include `the`.

The sole correction adds `the ` only when comparing independently parsed final text with original `TRAIN:` object metadata. No prompt, answer key, curriculum membership, renderer, tokenization, schedule or scientific choice changed. Final-text answer derivation remains independent of builder metadata. Full preflight is rerun after this correction.
