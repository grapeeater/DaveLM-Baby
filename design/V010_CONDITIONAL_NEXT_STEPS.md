# Baby v0.10 conditional next steps

Status: **not a launched protocol**. This note names forks. It does not preregister a result-dependent training run as if the result were known.

v2R5 is **absent in this GitHub branch**. Treat it as unresolved/pending until a sealed terminal receipt exists. Do not assume pass or fail.

Frozen Gate L/C/R stay fixed in all forks.

## Fork 0 — do this first, regardless of v2R5

Run the isolation diagnostic on the verified v2R4 U16000 checkpoint. Commands are in [`docs/FAN_DIESEL_HANDOFF_ISOLATION.md`](../docs/FAN_DIESEL_HANDOFF_ISOLATION.md).

Do not start seed `106002`, v2R6, architecture changes, tokenizer changes, or binding-sidecar enablement from the v2R4 headlines alone.

## Fork 1 — v2R5 still pending or absent

- Keep using v2R4 isolation results as the current causal evidence.
- Do not invent a v2R5 outcome.
- If isolation is scored, the next *design* (not launch) should target the winning hypothesis from Fork 4.

## Fork 2 — v2R5 terminal exists and fails frozen gates

1. Hash-seal the v2R5 metrics and checkpoint. Do not rewrite them.
2. Adjudicate against the **already frozen** v2R5 gates if a freeze exists; if v2R5 used the same Gate L/C/R, do not move thresholds after seeing scores.
3. Re-run the **same** isolation transforms against the v2R5 terminal checkpoint (copy the diagnostic, do not retarget frozen v2R4 hashes).
4. Compare v2R4 vs v2R5 on: pair-count value-only vs 1/K, query-swap follow, marker-swap vs sep-swap, value-absent, induction.
5. Only then design the smallest causal treatment that the *new* failure signature requires.

## Fork 3 — v2R5 terminal exists and passes frozen gates

Still require two-seed graduation. Do not graduate from one seed. Do not open protected evaluation. Isolation remains useful as a mechanism audit of a passing run (does it bind, or did it pass via 1-pair / separator leakage). A pass that is 1-pair-only or separator-confounded is not structural copy.

## Fork 4 — isolation outcomes (after Fork 0)

These are design constraints for a later preregistration, not authorization to train now.

**If query-swap does not follow.**
The 2-pair parent rate is not above chance. Treat B as unsolved at every K≥2. The next treatment, if any, must put multi-pair query-conditioned items in the *training* stream (minimum two pairs; **no 1-pair attractor**), with query-swap pairs as optimization targets, and must score first-token selection / value-span separately from separator/EOS. Do not add more 1-pair primitive keyed. Do not train "more copy." Copy is already 93/96.

**If query-swap follows on 2-pair but 3–4 pair stay at 1/K.**
Binding exists and is not dominant. Curriculum/optimization (D), not architecture (E). Increase pair-count pressure on the train surface before touching held-out markers.

**If query-swap follow is low and `rest_value_tf_lock` on the new gold span stays high.**
This is the current leading prediction. Treat first-token selection, not span decoding. A forced-choice / contrastive first-token objective is more targeted than another generation-only copy loss.

**If marker-swap kills value copy and sep-swap only kills exact.**
Keep train separators on held-out-*marker* tests for mechanism, or train separator diversity, but do not quietly drop Gate C held-out exact. Any future held-out exact metric must state whether the separator is OOD.

**If value-absent collapses original-answer copy.**
Replace the current broken-context control in *future* diagnostic panels with a value-absent control. Do not edit the frozen v2 panel.

**If induction remaining errors after stripping trailing-sep items are still ~0.**
Treat induction offset-binding as a separate unsolved operation. The 22 immediate-EOS rows are suffix glue (context already ends with the item separator), not the whole induction failure. Among full-induction items whose last token is not that separator, first-token correct is 3/64.

**If body-reorder query-first lifts 4-pair a lot.**
Part of the apparent chance rate is a first-slot prior. Randomize render order and/or train query-last items; do not confuse that with query-key binding.

**If nothing moves and even 2-pair is chance after query-swap.**
Still do not jump to E. The next diagnostic would be a same-surface 2-pair forced-choice ranking audit on the checkpoint (which token of which pair is rank-1 at the first generate step), not a 60M architecture rewrite.

## Explicitly not authorized here

- Launching training
- Weakening gates
- Changing tokenizer or enabling the binding sidecar
- Inspecting TEST/FINAL/SACRED
- Overwriting `v0.10` history or merging this work into `main`
