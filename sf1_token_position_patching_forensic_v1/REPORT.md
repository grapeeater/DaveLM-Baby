# SF1 token-position patching forensic v1

## Headline result

`ANSWER_GENERATION_STATE_DOMINANT_WITH_OUTPUT_PATH_INTERACTION; LANGUAGE_DAMAGE_DISTRIBUTED_ACROSS_PREDICTIVE_POSITIONS`

SF1's factual sequence behavior and unrelated-context name-prior pollution are both carried through answer/current-predictive token states. They are therefore **not cleanly separable token routes** at block 7. Language regression is broader: it is distributed across the tokenwise predictive states used throughout ordinary causal scoring rather than concentrated at the final record position.

## Frozen construction and execution graph

Before either checkpoint was loaded, the final rendered prompts were parsed and encoded with the authoritative tokenizer. All 16 prompts round-tripped exactly and matched their frozen token IDs. Prompt lengths are [26, 30, 34]; every subject/name span is three tokens, object spans are [4, 6] tokens per occurrence, predicates are [2, 4] tokens per occurrence, and both exact candidate completions are four tokens. `TOKEN_INDEX_MAP.json` records every character span, token index, model position after BOS, token ID, and dynamic answer-generation range.

The ordinary path is token plus position embeddings, eight pre-normalized transformer blocks with attention and MLP residual sub-boundaries, final normalization, and an untied language head. T13 localization/retrieval modules are registered in the wrapper but inactive because this diagnostic calls only the ordinary `base_model` path.

At each frozen site, the donor and recipient were run to the same residual boundary, selected `[batch,time]` positions were replaced, and the recipient's downstream blocks were retained. Pilot1 and SF1 output paths were evaluated separately.

## Baseline and splice integrity

- Same-source patching across every site/group/source reproduced native logits exactly: maximum absolute error `0.0` under the frozen `1e-6` tolerance.
- Native endpoint reproduction differed from the prior forensic by at most `1.19e-07` under the prospectively frozen `1e-5` cross-run aggregate tolerance.
- All-position token patches reproduced the corresponding prior whole-boundary interventions within `2.38e-07`.
- No tolerance was changed after inference.

Native factual behavior reproduced: Pilot1 `9/16` sequence-correct and SF1 `12/16`; SF1 retained `4/4` Mia/Nora complete reversal pairs and `0/4` Alex/Owen pairs.

## 1. Block 4: temporary Owen-favoring flip

The earlier “Owen flip” is specifically a **full sequence-likelihood** effect. At block-4 MLP, SF1 states patched only at the final prompt predictor leave a strong Alex sequence default (`1.087`) and an Owen-correct mean margin of `-0.988`. Patching the whole answer-generation path changes the default to Owen (`-1.437`) and raises the Owen-correct margin to `1.666`, nearly matching all-position patching (`-1.471`, `1.746`).

The first-token decision itself still favors Alex. The flip emerges in the later candidate-token likelihoods. Subject, either object occurrence, predicate/cue, or remaining context patched alone does not reproduce it. Thus the block-4 sequence flip is answer-generation-state dominant; it is not localized to one semantic prompt span.

## 2. Block 5: return toward Alex

With the Pilot1 output path, the answer-generation state carries an Owen default after block-5 attention (`-1.678`) but is nearly neutral after block-5 MLP (`0.073`). The corresponding all-position default moves from `-1.281` to `0.286`. Using the SF1 output path moves the block-5-MLP answer trajectory further toward Alex (`0.829`).

No isolated subject, object, or predicate span explains this transition. It reflects a change in the answer-generation trajectory plus an output-path interaction. This is identity-selection behavior, not successful Alex/Owen reversal: complete Alex/Owen pairs remain zero in native SF1.

## 3. Block 7 MLP: factual recovery

| SF1 state injected into Pilot1 | first-token correct | sequence correct | exact answer+EOS | all complete pairs | Mia/Nora pairs |
|---|---:|---:|---:|---:|---:|
| final prompt predictor only | 12/16 | 9/16 | 0/16 | 1/8 | 0/4 |
| full answer-generation path | 12/16 | 12/16 | 12/16 | 4/8 | 4/4 |
| all positions | 12/16 | 12/16 | 12/16 | 4/8 | 4/4 |

The final prompt predictor is sufficient for the SF1 first-token result, but not the complete four-token name plus EOS. Patching the full answer-generation trajectory restores `12/16` sequence decisions, `12/16` exact completions, and all four Mia/Nora reversal pairs, exactly matching all-position patching. In the reverse direction, replacing SF1's answer trajectory with Pilot1 states reduces behavior to `9/16`, `0/16` exact, and `0/4` Mia/Nora pairs. Static context-only patches at this final tokenwise MLP cannot affect the answer position and leave native SF1 behavior unchanged.

## 4. Block 7 MLP: unrelated name prior

On the exact frozen 256-position D3 sample, patching only the current predictive state produces four-name mass `0.036394` with the Pilot1 output path, exactly the all-position value. Patching all earlier context states produces the Pilot1 baseline `0.000907`. The SF1 output path amplifies the same predictive state to `0.067833`, the native SF1 value. Reverse replacement of the predictive state reduces native-SF1 mass to `0.001805`.

The late D3 prior is therefore localized to the current predictive state and materially amplified by the SF1 output path.

## 5. Block 7 MLP: language regression

Pilot1 aligned CE is `3.3907` and SF1 is `4.6529`. Injecting SF1 only at each record's final predictor yields CE `3.2926`; injecting SF1 into the preceding valid/scored positions yields `4.3749`; all valid positions yield `4.2769` with the Pilot1 output path. In reverse, replacing those preceding SF1 positions with Pilot1 states reduces CE to `3.3244`, while replacing only the final record predictor leaves CE at `4.7184`.

The language regression is spread across the many token states that perform ordinary next-token prediction. “All prior context” here means all valid positions before each record's final EOS-predicting position; it is not evidence for a single semantic context span. Recovery fractions above 100% or non-additive totals are hybrid-network interactions, not literal shares.

## Evidence ledger

### Established

- The block-4 sequence-likelihood Owen flip is carried mainly by answer-generation/candidate-prefix states, not the final first-answer predictor alone.
- The block-5 return toward Alex occurs within that answer-generation trajectory and is further shifted by the SF1 output path.
- At block-7 MLP, the final predictor carries first-token factual choice and D3 name-prior inflation; later answer-prefix states are additionally required for the full candidate sequence, exact EOS behavior, and Mia/Nora reversal recovery.
- Block-7 language damage is distributed across many causal predictive positions.

### Supported

- Useful SF1 factual sequence behavior and harmful name-prior inflation share a predictive/answer-state route and cannot be cleanly separated by the tested token groups.
- Language damage is positionally broader, giving partial separation by breadth but no demonstrated useful-only route.
- Static subject, object, and predicate states can influence middle-layer hybrids, but no isolated static group is sufficient for the decisive SF1 factual endpoint.

### Not established

- The patches do not identify where training changed the mechanism.
- Results on these 16 training records do not establish held-out factual generalization.
- Non-additive hybrid effects cannot be interpreted as a linear allocation of causal responsibility.
- No treatment choice, architecture conclusion, or locked-panel claim follows.

The bounded diagnostic ends here. No training treatment is proposed or executed.
