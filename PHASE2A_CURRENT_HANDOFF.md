# Phase 2A current handoff (authoritative)

Stay in `C:\DaveLM-CADAVER`. Do not open T3 TEST. Do not lower the v1.0 bar.

## Parent
Phase1G U6000 SHA256 `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`
U0 DEV CE `1.2040123894810677`
TEST `SEALED_UNOPENED` (`qa_test.jsonl` SHA256 `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`)

## Closed lines
- T3–T16 / T16X / Qwen: as previously closed. Do not parent T14/T15/T16/730002.
- T17 `T17_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — representation **2/3**, exact 0.
- T17X COMPLETE. Exact 0 was Phase1G story continuation after `Answer:`.
- T18 `T18_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — language 3/3, representation **2/3**, exact **36/36/40**.
- T18X COMPLETE. First-token BOS 76–84. Leftover `first_token_correct_then_diverge`.
- T19 `T19_FAIL_NO_REPRESENTATION` — representation **1/3**, exact **39/35/34**. Suffix-weight 3.0 closed.
- T19X COMPLETE. Leftover still diverge.
- T20 `T20_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — representation **2/3**, exact **9/10/13**. Head-only CE closed.
- T20X COMPLETE. First-token BOS **42/34/37**. Dominant `other_then_eos`.
- T21 `T21_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — language 3/3, representation **3/3**, exact **27/29/26**.
- T21X COMPLETE. First-token BOS **79/59/72**. Dominant leftover diverge 52/30/46 plus other_then_eos.
- T22 `T22_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — language 3/3, representation **2/3**, exact **21/31/26**. First-two through-base closed.
- T22X COMPLETE. First-token BOS **66/65/73**. Leftover still diverge. CE-span-count family closed.
- T23 `T23_REPRESENTATION_SUCCESS_OUTPUT_FAIL` — language 3/3, representation **3/3**, exact **34/34/38**. In-row rival unlikelihood did not beat T18 exact. Report: `phase2a_t23_candidate_unlikelihood_lr3p75e5_1k_v1\T23_FINAL_REPORT.md`.
- T23X COMPLETE. First-token BOS **79/75/79**. Leftover still diverge; Walt/York are off-row inventory names. Report: `phase2a_t23x_generation_failmode_v1\T23X_FINAL_REPORT.md`.
- SmolLM2-1.7B forensic is advisory only. Baby evidence outranks Smol.

Do not parent T14–T23 OUTPUT_FAIL checkpoints (760001–820003). Do not relaunch T4–T23. Do not copy Qwen or Smol. Do not retokenize. Do not raise suffix weight or `λ_unl`. Do not reopen CE-span-count.

## T24 terminal / successor authorization
T24 `phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1` is **physically complete and adjudicated**. Seeds 830001–830003 all have `FINAL_STATUS.json` at U1000 Phase B. Watchdog 34608 exited after `STUDY_COMPLETE` 2026-09-13T12:52:23Z. The owner explicitly released the temporary pause restriction on 2026-09-13 and authorized autonomous successor studies. The historical pause record remains unchanged for provenance.

T24 is now scientifically classified `T24_REPRESENTATION_SUCCESS_OUTPUT_FAIL` from the three frozen final statuses. Do **not** parent 830001–830003. TEST sealed.

T24X read-only generation fail-mode audit is complete at `phase2a_t24x_generation_failmode_v1`. It used only the authorized DEV panel and preserved all locks. The dominant residual is first-token-correct-then-diverge; no treatment was run.

Owner authority (2026-09-13): the autonomous agent may design, freeze, implement, and execute the highest-information Baby-native successor when no pre-existing protocol exists, without waiting for additional approval. Existing scientific gates and protected-material locks remain in force.

Resume artifact (last seed): `phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1_run_seed830003\rolling_restart.pt` SHA256 `e00f9b625d31c9c5615d3c639422362b585c92427828c254dc639704b9a186c9` (415,491,817 bytes, `completed=1000` phase B). Pause record: `phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1\T24_PAUSE.md`.

Resume command (documented, **not run**):

```
cmd /c start "" /min C:\DaveLM-CADAVER\phase2a_t24_inventory_unlikelihood_lr3p75e5_1k_v1\LAUNCH_WATCHDOG.cmd
```

That command would see all three `FINAL_STATUS` files and would not train. Do not run it.

## T28 FAST V2 (2026-09-13)

Rebuilt from untouched T24 runner; vectorized suffix hard-negative margin (M=1.0, lambda=0.5) only. Seeds 850001-850003 all completed update 1000. Pointer/forced-choice and TRAIN16/language/binding remained healthy; exact generation remained T24-like (36/35/40 of 128). Classification: `T28_SUFFIX_MARGIN_VALID_COMPLETION / OUTPUT_FAIL`. Protected T3_TEST, T2_EVAL_TEST, FINAL, sacred remained locked. Bundle: `phase2a_t28_fast_v2`.

## Post-T28 diagnostics + mouth repair

Tokenizer diagnostic: `TOKENIZER_ASSOCIATION_SUPPORTED` (4.7% vs 53.1% exact). Suffix-state: `GOLD_PREFIX_DIRECT_DECODABILITY_WITH_FREE_PREFIX_COLLAPSE` (66.6% vs 26.5% final top-1). Decode-only `tokenizer_repair_v1`: free beam does not close shared-prefix exact (4.7% → 4.2%); 8-name inventory rerank is non-native. Option C append-only: no effect. Option B replacement candidates: `TOKENIZER_REPLACEMENT_NOT_JUSTIFIED`. Tokenizer lineage is **paused**. Do not build another tokenizer, retokenize, or retrain.

Prefix-rescue `phase2a_post_t28_prefix_rescue_v1`: `EARLY_TOKEN_ERROR_CAUSALLY_DOMINATES`. On the frozen DEV first-token-correct-then-diverge population (39/40/45; 124 pooled), replacing only the first wrong token with gold and returning to free greedy recovered exact+EOS **108/124 (87.1%)** replicated 32/37/39. Matched-wrong and prefix-compatible-wrong controls were 0% exact. Activation patching was not run. v0_7 must stay byte-identical.

## T29 (2026-09-14)

`phase2a_t29_disambiguation_fork_v1`: `L_fork` on the T24 recipe; T28 suffix margin off; tokenizer paused. Seeds 860001-860003 completed U1000. Pointer 110/111/114, native FC 88/87/88, exact **35/35/35**. Shared-prefix exact 2/2/0 of 64. Language held. Classification: `T29_REPRESENTATION_SUCCESS_OUTPUT_FAIL`. Fork-objective hypothesis **closed**. Do not retune lambda/margin. Do not parent 860001-860003. TEST sealed. T30 not authorized.

## T30 persistent pointer route (2026-09-14)

T30 was the single owner-authorized architectural successor after T29. It added a zero-initialized 640-parameter per-channel gate that makes the existing query-weighted fact-clause pointer available at every answer prediction position. All three Phase1G-parent branches completed U1000 with binding/localizer locks and protected data intact. The route gate activated in every branch, but native exact generation did not improve beyond the established ceiling.

Terminal classification: `T30_REPRESENTATION_SUCCESS_OUTPUT_FAIL` (2/3 branches representation-positive; 0/3 native full-success). Final DEV exact+EOS was 31/36/38 of 128; shared-prefix exact 2/3/3 of 64; first-token-correct-then-diverge 45/42/41. Parent SHA `c5406f8053ec836c099fb5a1cd3497fb5a0f7397ff366433ec63161dea0eefb1`. T3 TEST, T2-EVAL-TEST, FINAL, and sacred remain sealed/locked. T30 artifacts are in `phase2a_t30_persistent_pointer_route_v1`; T31 has completed and is terminal; do not launch T32.

T31 terminal: explicit source-to-token copy bridge insufficient; exact 35/128 in both required seeds, seed 3 not required, T32 not authorized. TEST/FINAL/sacred locked.

## T31 closure and postmortem (2026-09-14)
T31 is terminal: T31_EXPLICIT_SOURCE_TO_TOKEN_BRIDGE_INSUFFICIENT. Its two required Phase1G-parent seeds each reached U1000 at 35/128 exact+EOS despite activated ordered copy gates, healthy language, and intact binding; seed 3 was correctly not required under the sealed sequential rule. The current-architecture surgical treatment line is exhausted. The DEV-only postmortem is PHASE2A_POST_T31_MECHANISTIC_AUTOPSY.md; it finds correct source/copy alignment often present at the shared fork but too weak to beat the ordinary LM's final-logit competitor. T32 is not authorized. TEST/FINAL/sacred remain sealed.

## T32 terminal closure (2026-09-14)
T32 Causal Ordered Source-Memory Prefix is terminal. It used ordered source-token K/V memory in the final Transformer block with a zero-initialized 640-channel gate, parent-neutral at U0. Seeds 890001 and 890002 completed the frozen 1000-update schedule; seed 890003 was not required by the sealed 2-of-3 rule after two scientific failures. Terminal exact+EOS was 35/128 for both; shared-prefix exact was 0/64 and 2/64; pointer was 112/128 and 110/128; language and binding remained intact. Memory ON/OFF terminal ablation was effectively unchanged. Classification: T32_EXPLICIT_ORDERED_SOURCE_MEMORY_INSUFFICIENT. The Phase2A surgical treatment line is exhausted. Per the frozen failure branch, no T32b/T33 is authorized; the next owner-planned action is TARGETED_EXTERNAL_MODEL_ARCHAEOLOGY_EXACTLY_3_MODELS. T3 TEST, FINAL, and sacred remain sealed/unopened.
