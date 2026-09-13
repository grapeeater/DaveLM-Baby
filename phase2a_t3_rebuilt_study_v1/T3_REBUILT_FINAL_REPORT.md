# PHASE 2A T3 REBUILT STUDY — FINAL REPORT

Status: **TERMINAL — study classification `T3_LANGUAGE_REGRESSION` (3/3).**
TEST was **not** opened. T2-EVAL-TEST / FINAL / sacred untouched.

## Decision that was executed
Rebuild and run T3 (parameter-free pointer + T2 native margin; binding frozen) on a disjoint corpus with seed-specific schedules and a complete evaluator. Prior T3 artifacts were not counted as replications (v1 identical checkpoints; repair infrastructure-blocked; ctxbind name overlap).

## Parent / tokenizer
- Phase1G U6000 `best.pt` SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`
- Tokenizer SHA256 `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`
- Scope: 60,536,064 base trainable / 984,321 binding frozen
- U0 language DEV CE 1.2040123894810677 (exact Phase1G reproduction)

## Provenance note (ledger KeyError)
`TRAINING_FAILED_SEED_620001` in an intermediate `RUN_LEDGER.json` was **not** a scientific failure. Seed 620001 wrote `FINAL_STATUS.json` then crashed on `ledger["runs"]` missing. Watchdog PID 17172 saw complete status and launched 620002 then 620003. Three U500 checkpoints have **distinct** SHA256 values (seed independence).

## U500 seed results (DEV n=128)

| seed | pointer | native FC | exact+EOS | ptr rev | nat rev | ptr families | DEV CE | gap | binding | U500 SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 620001 | 99/128 | 88/128 | 0 | 38/64 | 27/64 | 5/16 | 1.30484 | 0.554 | intact | `24d81327…57095ed` |
| 620002 | 97/128 | 87/128 | 0 | 35/64 | 26/64 | 4/16 | 1.30118 | 0.551 | intact | `9611eaa4…63b11de09` |
| 620003 | 102/128 | 96/128 | 0 | 39/64 | 33/64 | 6/16 | 1.30759 | 0.535 | intact | `6b73b195…998a5821` |

U0 pointer was 64/128 chance on seed 620001 preflight. Retrieval moved above the 96/128 retrieval gate on all three seeds. Reversal, complete-family, exact+EOS, language, and train/DEV-gap gates failed on all three. Shortcut probes stayed near chance (not a shortcut success). `test_loaded=false` on all evals. T3 TEST seal SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb` remains `SEALED_UNOPENED`.

## Frozen-gate adjudication
Per-seed frozen class is `T3_LANGUAGE_REGRESSION` because DEV CE > 1.30 and gap > 0.5 (language checked first). Independently of that label: **representation success is not established** (reversal 35–39/64 < 48/64; complete families 4–6/16 < 8/16; exact 0/128). Native forced-choice is partial (87–96/128); only 620003 met 96/128 FC, none met exact or native reversals. Binding parameter identity held. Study-level rule: 3/3 same class → **`T3_LANGUAGE_REGRESSION`**. Not `T3_FULL_SUCCESS`. TEST scoring is not authorized.

## Interpretation
Pointer supervision on the ~60.5M base **can** raise held-out name-disjoint retrieval above chance, but it did not create counterfactually complete assignment (reversals/families), did not produce usable exact native names, and nicked Phase1G language past the absolute 1.30 CE gate. That is a valid negative-plus-partial for the T3 mechanism, not a warrant to reopen T3 TEST or to continue coefficient-tuning T3 on these descendants.

## Next authorized treatment
Train/integrate the dormant **984,321-param** OrthoLocalizer + retrieval sidecar on the **untouched Phase1G parent** (not T3 U500 weights). Hypothesis: query-conditioned routing via `BindingLayout` can produce reversal-complete selection on the same disjoint T3 train/DEV without opening T3 TEST and without inheriting T3 language regression. Bundle: `phase2a_t4_binding_sidecar_v1`.
