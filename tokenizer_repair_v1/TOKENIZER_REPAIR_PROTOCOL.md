# Tokenizer repair protocol (frozen before decode scoring)

Study: `TOKENIZER_OUTPUT_MOUTH_REPAIR_V1`. Machine copy: `PROTOCOL.json`.

## Integrity

- Workspace `C:\DaveLM-CADAVER`. v0_7 SHA-256 `e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b`.
- Checkpoints: T28 FAST V2 `rolling_restart.pt` for 850001–850003 with the post-T28 provenance hashes.
- Inputs: `qa_dev.jsonl` only for questions. No `qa_test.jsonl`, T2-EVAL-TEST, FINAL, or sacred.
- Weights read-only. No optimizer. No T24 resume. No T25/T29 launch.

## Methods (frozen)

1. **Greedy native** — same rule as T28 `exact_answer_eos`: argmax until `<eos>`, max 12 new tokens.
2. **A1 free beam** — width 4 over the full 1024-way vocab; accumulate sum logprob; length-normalize by generated length; stop at `<eos>`. No name inventory in the search.
3. **A2 constrained 8-name rerank** — mean teacher-forced logprob of each frozen DEV answer string (the eight `correct_name` sequences including their period, as stored in `candidate_token_ids`). Pick argmax. Label: `NON_NATIVE_INVENTORY_RERANK`.

## Metrics

Exact+EOS vs gold `candidate + <eos>`; shared vs unique first-token splits using DEV-name first-token collision groups; per-name exact for Sal, Skye, Omar, Opal, Wes; diverge rate for greedy/beam.

Language CE and pointer are **not** rescored here (decode-only). Representation is treated as previously established on these checkpoints (T28 pointer 108–116/128, binding PASS).

## Decision rule

A1 is promising only if shared-prefix exact rises substantially and unique-prefix does not collapse, replicated on 3 seeds. A2 cannot close Phase 2A. If A1 fails, do not retokenize; report that greedy and free beam share the continuation prior, and stop before Option B/D training.
