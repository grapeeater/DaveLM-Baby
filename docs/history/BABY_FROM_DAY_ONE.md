**THE DAVELM EXPERIMENT**

**BABY, FROM DAY ONE**

A Comprehensive Developmental and Experimental History of  
DaveLM / Mangaris Pioneer (Codename: Baby)

*26 August 2026 – 30 September 2026*

**Prepared for David  
The DaveLM Experiment**

Canonical project-history edition • 30 September 2026

# Abstract

This paper reconstructs the first complete developmental month of DaveLM’s central model lineage, informally called Baby and formally moving toward the Mangaris Pioneer identity. Active development began around 26 August 2026 with a small, from-scratch decoder-only transformer and a deceptively simple experimental question: could a personally built model learn contextual relationships rather than merely exploit superficial regularities? What followed was a dense sequence of controlled training interventions, frozen evaluations, architectural experiments, failed hypotheses, forensic audits, language-training pilots, capacity studies, repo archaeology, cloud-compute operations, and increasingly strict research-governance rules. The project evolved from a roughly 10.6–10.8M-parameter research model, through a 61.52M-parameter vNext system, to the present approximately 117.78M-parameter MRCN-Alpha school model. Across that path, Baby progressed from candidate recognition without reliable selection; to perfect scaffolded contextual binding in T12; to learned source localization and a synthetic-binding graduation; to early language acquisition with the canonical phrase “he womputeld”; to repeated discovery of representation–readout–generation dissociations; to an explicit copy-versus-selection diagnosis; and finally to a certified S1 single-token-copy graduation on 30 September 2026. The S1 graduate achieved 128/128 held-out, 128/128 novel, and 64/64 counterfactual performance on both gated panels at update 4700, after a lineage that included local hardware freezes, safety-system bugs, failed treatments, cloud replay, and prospectively frozen early-stop rules. The final artifact, S1_single_token_copy_graduated.pt, was hash-verified and independently certified. This is not a claim that Baby is a finished general language model or that Mangaris Pioneer v1.0 is release-ready. It is a chronological and methodological account of what the project actually established, what it disproved, what remains unresolved, and how an initially improvised hobby model became a disciplined small-model research program in barely more than a month.

Keywords: DaveLM; Mangaris Pioneer; Baby; small language models; transformer; contextual binding; retrieval; source localization; catastrophic interference; representation; readout; autoregressive generation; curriculum learning; counterfactual evaluation; model scaling; reproducibility.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Scope and evidence rule</strong></p>
<p>This paper is exhaustive with respect to the project material available in the conversation, saved summaries, and recovered handoffs. Where a named intermediate run is known only at lineage level, the paper says so rather than inventing metrics. Later corrections override earlier interpretations; invalidated exams remain invalidated.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

# Executive Summary

Baby’s first month is best understood not as a straight training run but as a sequence of narrowing scientific questions. The project repeatedly refused the easy interpretation that a falling loss or a fluent-looking sample meant a capability had been learned. Instead, it built tests that separated candidate recognition from selection, selection from copying, representation from native readout, native readout from free-running generation, and training success from held-out counterfactual generalization.

The first major breakthrough was T12, which inserted an explicit query-conditioned mapping retrieval pathway. T10 had repaired the data and still failed. T11 had concentrated supervision on the answer and still failed. T12 changed the routing computation and immediately reached ceiling training performance, followed by 160/160 held-out answers, 40/40 complete counterfactual quartets, 80/80 reversals, and 100% correct-row retrieval. That result redirected the project from “can Baby bind at all?” to “how much structure can be removed while preserving binding?”

The second major breakthrough was learned localization. After T13 and a long chain of localization interventions, the project identified an important failure mode in a bounded shared scorer: sharing improved recognition, but a bounded disagreement channel could collapse two localization slots onto one source. A 641-parameter orthogonal shared-core plus unbounded antisymmetric residual repair produced the canonical synthetic-binding graduate: 317/320 answers, 312/320 BOTH_DISTINCT localization, zero collapse, 78/80 complete quartets, and 158/160 reversals. This established a narrow but real cognitive primitive before language training began.

Language then created an interference problem. TinyStories Pilot 0 dropped DEV perplexity from roughly 1278 to 8.61 and produced recognizable English—most famously “he womputeld”—but damaged binding. Pilot 1 showed the two capabilities could coexist: DEV perplexity 18.25 with 80/80 answers, 80/80 BOTH_DISTINCT, zero collapse, 20/20 quartets, and 40/40 reversals. P0–P7 improved English form, but a controlled 12/12 P7 milestone failed to transfer naturally; a first report card was later withdrawn after 144/288 answer keys were found wrong, and a corrected evaluation exposed near-chance ordinary-English fact use. That episode established a durable rule: invalid exam, invalid conclusion.

The 10M factual-supervision/SF series then exposed a Pareto frontier among exact answer generation, widening/generalization, language/binding retention, and ordinary-English name leakage (D3). SF20’s identity rotation sharply reduced leakage but also damaged exact generation and widening. SF21 added paired counterfactual representation consistency as a prospectively declared final shot. It still failed the coexistence gates, and the ~10M series was closed rather than extended into SF22. This failure—not mere intuition—motivated a controlled scale test.

The 61.52M vNext model was built from scratch as a 12-block, d_model 640, 10-head, MLP 2560 transformer. Initial language pretraining overfit a too-small diet; U3000 was the original best DEV checkpoint, while U6000 degraded. Phase1G expanded the unique TinyStories diet to 20,990 stories/8.35M tokens while holding the rest of the recipe fixed, improving U6000 DEV CE to ~1.204. The later T17–T32 era repeatedly demonstrated a striking dissociation: relational information could be learned and even generalized while exact native generation remained poor. T28’s forced-prefix rescue recovered 108/124 cases (87.1%), making the first answer token a central bottleneck. Subsequent bridge treatments still failed to solve the output path.

The v0.10/v2R4 autopsy sharpened that diagnosis further: on 96 novel multi-pair examples Baby copied an actual value from context 93 times, but only 41 were the requested value while 52 were competing values. When the correct first token was manually supplied in wrong-selection cases, Baby continued the requested value correctly 51/52 times. This changed the working mechanism from “copy failure” to “query-conditioned first-token selection failure.” P11 later improved long-gap greedy selection from 71/215 to 102/215, but retention with the active overwrite was not proven, so it was not promoted.

By late September the main school lineage had scaled again to 117,780,224 parameters (12 layers, d_model 896, 14×64 heads, d_mlp 3584). The larger model was not treated as a magic fix. The project explicitly adopted a constitution-like rule: a failed class is not evidence that the brain is too small; capacity increases must be justified by reproduced, characterized failure and controlled comparisons.

The immediate pre-S1 diagnostics showed the 118M model still lacked a generic induction-style copy circuit. Practiced examples could score well while held-out pseudo-token copying collapsed, embeddings/position behavior remained near-random in relevant analyses, and copying was strongly tied to templates, landmarks, and practiced pieces. The first S1 treatment failed narrowly. S1v3 then introduced a gradient-safety system, but early versions contaminated training through flawed warmup baselines and a self-reinforcing throttle. Those bugs were discovered, documented, and corrected rather than interpreted as Baby behavior. Local S1v3h attempts also hard-froze the entire Windows machine, pushing the final continuation onto RunPod.

On 30 September, S1v3j resumed the frozen S1 lineage in the cloud and reached the prospective early-stop gate at u4700. Both gated panels were perfect: Heldout 128/128, Novel 128/128, with 64/64 counterfactuals on both. Independent certification reproduced 1.00 across every frozen gated metric. Training stopped immediately; u4800 was never consumed. The final graduate checkpoint S1_single_token_copy_graduated.pt has SHA-256 200e02188063885560edb4de0d6e0f05054fa75c236c0264d4a2c461ec51771b. Seven scheduled checkpoint/evaluation pairs and 245 evidence files were returned and hash-verified. The formal cloud execution cost roughly \$0.34 and lasted 4 minutes 24 seconds. A nongated identity-stress panel remained 0/128, so S1 graduation is a real milestone but not equivalent to Pioneer v1.0 release readiness.

# Contents

- 1\. Scope, Method, and Evidence Standards

- 2\. Genesis: Building Baby from Scratch

- 3\. The Original Small-Model Architecture

- 4\. The Binding Problem and the T1–T12 War

- 5\. Removing the Training Wheels: T13 and Synthetic Graduation

- 6\. Language Acquisition, “he womputeld,” and Catastrophic Interference

- 7\. P0–P7 and the Evaluation-Correction Era

- 8\. Factual Supervision and the End of the ~10M Line

- 9\. Scaling to 61.52M: vNext and Phase1/Phase1G

- 10\. The Representation–Readout–Generation Era (T17–T32)

- 11\. v0.10, v2R4/v2R5, and the Copy-vs-Selection Diagnosis

- 12\. Scaling Again: The 117.78M School Model

- 13\. Pre-S1 Copy Forensics and the School Curriculum

- 14\. S1: First Failure and the S1v3 Safety Saga

- 15\. Local Hardware Failure and the Move to RunPod

- 16\. S1v3j: Certified Graduation at u4700

- 17\. Post-Graduation Preservation, Naming, and Family Tree

- 18\. What Baby Has Actually Proven

- 19\. What Baby Has Not Proven

- 20\. Methodological Lessons and the DaveLM Research Constitution

- 21\. The Planned Post-S1 Curriculum

- 22\. Conclusion

- Appendix A. Master Chronology

- Appendix B. Treatment and Experiment Matrix

- Appendix C. Architecture Timeline

- Appendix D. Canonical Artifacts and Hashes

- Appendix E. Glossary

- Appendix F. Full Curriculum Wishlist

- Appendix G. Source Corpus and Provenance Notes

# 1. Scope, Method, and Evidence Standards

The purpose of this document is historical reconstruction, not retrospective myth-making. Baby’s development produced many tempting moments where a single metric looked decisive. The record repeatedly became stronger only when those moments were re-tested, contradicted, or narrowed. For that reason, this paper uses three epistemic categories throughout: established result, supported interpretation, and unresolved hypothesis.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Established result</strong></p>
<p>A result is treated as established only when the preserved record shows the experiment/evaluation actually ran under the stated protocol and the metric is known. Frozen held-out results, exact artifact hashes, and independently verified reports are strongest.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Supported interpretation</strong></p>
<p>An interpretation may explain an established pattern but is not promoted to mechanism unless a discriminating or causal test supports it. Examples include “routing bottleneck,” “state decay,” or “capacity/interference ceiling.”</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Unresolved / withdrawn</strong></p>
<p>Invalid tests, mislabeled coordinates, missing retention checks, or results from contaminated runs are preserved as part of history but are not used as proof. Earlier interpretations are explicitly superseded when later audits corrected them.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

A defining methodological shift occurred during the localization era: instead of “failed run → new treatment,” the project increasingly used “failed run → frozen autopsy → hypothesis → zero-update/preflight check → prospectively frozen treatment → one frozen exam → stop.” The same philosophy later governed scaling, language work, v0.10 diagnostics, and S1. This reduced credit/compute waste and prevented the research program from being steered by whichever coding agent happened to be operating the repository.

The project also adopted a strong separation of roles. Expensive or high-reasoning models were used to interpret difficult evidence or challenge the diagnosis; cheaper models/agents were used for repo archaeology, implementation, and mechanical execution. The operational principle became: expensive brain, cheaper hands. This is historically relevant because it shaped how treatments were designed and how much external-model advice was allowed to influence Baby.

# 2. Genesis: Building Baby from Scratch

Active DaveLM/Baby development is treated as beginning around 26 August 2026. The founding question was simple and personal: could David build and train an actual transformer locally on Fan Diesel rather than merely fine-tune or download someone else’s model? The model was intentionally tiny by contemporary standards so that architecture, data, optimizer behavior, checkpoints, and internal experiments could be understood and repeatedly modified on consumer hardware.

From the beginning, the project’s value proposition was not benchmark competition. Baby was a controllable research organism. Every dataset could be regenerated, every checkpoint frozen, every loss term modified, and every failure inspected. That made a skill that sounds trivial to a human—follow a local key→value relationship—an ideal first scientific target.

> “A → Apple. B → Banana. Ask for B. Baby should answer Banana because Banana belongs to B—not because Banana is a generally attractive answer token.”

That task created the “Great Binding War.” The project learned almost immediately that a small transformer could appear to know the candidate set while failing the core relational selection step. This distinction—recognizing plausible answers versus choosing the answer demanded by the current query—became the through-line of the entire first month, reappearing in later language, factual QA, multi-pair copying, and S1 work.

# 3. The Original Small-Model Architecture

The earliest preserved notes contain more than one small-model configuration. An early V082 note described a 4-block, d_model 320, 10-head, FFN 1280, dropout 0.05 model with vocab 1024 and context 256. A later audited “Research Baby” configuration—the authoritative base for the late 10M factual-supervision series—used the same 320-dimensional width and 10 heads but 8 transformer layers. Its base parameter count was 10,594,944. With the then-present binding/structural parameters, the audited total was 10,841,345 (10,594,944 base + 246,401 binding). The discrepancy is best understood as an early architecture revision rather than a contradiction.

| **Era**               | **Blocks** | **d_model** | **Heads** | **MLP/FFN** | **Vocab** | **Context** | **Parameters**                            |
|-----------------------|------------|-------------|-----------|-------------|-----------|-------------|-------------------------------------------|
| Early V082 note       | 4          | 320         | 10        | 1280        | 1024      | 256         | ~10.6M noted in project memory            |
| Audited Research Baby | 8          | 320         | 10        | 1280        | 1024      | 256         | 10,594,944 base; 10,841,345 incl. binding |

The tokenizer/vocabulary was intentionally compact. At this stage the small closed vocabulary made synthetic experiments reproducible and cheap. The project explicitly resisted the temptation to “modernize” every component merely because large open models used RoPE, RMSNorm, or other contemporary choices. Architectural changes were supposed to earn their place by solving a measured Baby problem, not by cargo-culting Qwen, Llama, Devstral, or any other reference model.

# 4. The Binding Problem and the T1–T12 War

The T1–T12 lineage is the first complete causal staircase in Baby’s history. Each treatment removed a plausible explanation for the selector failure. The preserved early summaries do not retain every scalar from T1–T4, but they do retain the behavioral logic of those treatments and the later frozen metrics.

## 4.1 T1 through T4: candidate recognition without reliable selection

| **Treatment** | **Intervention / Question**                           | **Observed outcome / lesson**                                                                                             |
|---------------|-------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| T1            | Bootstrap from one-map competence                     | Candidate set mostly understood; selector still failed.                                                                   |
| T1.5          | Curriculum bridge                                     | Did not bridge the selection failure.                                                                                     |
| T2            | Paired query swaps designed to force query dependence | Still failed to consistently follow query identity.                                                                       |
| T3            | Direct objective forcing correct \> distractor        | Baby found a loophole: it could push both candidates downward rather than establish a robust relation.                    |
| T4            | Membership + selection objective to close T3 loophole | Training looked promising, but no official frozen report card survived; it cannot be retroactively promoted to a success. |

The early insight was subtle: Baby often behaved as if it knew “the answer is one of these two values,” yet did not reliably apply the local mapping to decide which value belonged to the current key. This is materially different from total ignorance.

## 4.2 T5–T9: isolate signal location and test retrieval

| **Treatment** | **Key idea**                                       | **Key metric / result**                                        | **Interpretation**                                                                              |
|---------------|----------------------------------------------------|----------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| T5            | Reciprocal / twin counterfactual structure         | correct\>distractor ≈ 50.1%                                    | Essentially chance relational selection.                                                        |
| T6            | Amplify answer loss ≈191×                          | correct\>distractor ≈ 54.7%                                    | More answer pressure helped slightly but did not solve the selector.                            |
| T7            | Inspect raw query-token embedding vs output logits | correct\>distractor ≈ 75.7%                                    | Query identity itself carried substantial answer preference.                                    |
| T8            | Inspect contextualized query hidden state          | c\>d 85.4%; margin≥0.5 81.6%; candidate mass 93.6%; top-5 100% | A great deal of useful information existed internally, but the preregistered gate still failed. |
| T9            | Generic late answer-position retrieval head        | c\>d/top-1 52.3%; candidate mass 85.1%; mean Δlogit ≈ +0.088   | A generic retrieval head was insufficient; retrieval as a concept was not disproven.            |

T8 was pivotal. It made it increasingly difficult to explain Baby as simply “not knowing.” The contextualized query state often favored the correct answer while downstream behavior remained unreliable. This became an early instance of the representation-versus-use distinction that would reappear much later in the 61M model.

## 4.3 Structural census: the old universe could not answer the real question

Before T10, the project performed a read-only structural census of frozen artifacts to ask whether strict counterfactual comparisons already existed. The census examined 1,536 pair structures, 3,072 twin documents, and 32,000 frozen schedule events. The gold-standard structure—same query, same candidate pair, but reversed correct binding—occurred zero times.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>What the census proved</strong></p>
<p>The old artifacts could not cleanly test strict reversed binding. This did not prove Baby was cheating; it proved the experiment universe was structurally incapable of resolving the question.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## 4.4 T10: fix the data

T10 created a strict counterfactual-binding curriculum rather than hoping reversals appeared naturally. Each quartet contained both mapping orientations while matching keys, candidate values, filler, layout, and positions as tightly as possible. The training universe contained 200 quartets (800 documents), with 40 separate retention quartets (160 documents), 1,000 steps, batch size 32, and 32,000 training document events.

Training looked deceptively excellent: full-sequence token accuracy approached 97.8%. But only about 0.52% of supervised token positions were actual answer decisions. The frozen retention answer accuracy was 50%. T10 therefore taught a major lesson: an easy sequence-level objective can hide a completely unsolved decision problem.

## 4.5 T11: fix the supervision

T11 concentrated supervision directly on the answer decision using the same strict counterfactual universe. It still did not solve the problem. The preserved metrics put retention at roughly 45%, correct\>distractor at 50%, and final training answer accuracy at 50% with margin −0.033 and answer cross-entropy 0.717. By this point, “bad data” and “diluted supervision” had both been directly challenged.

## 4.6 T12: fix the routing architecture

T12 introduced QueryConditionedMappingRetrieval: the query representation was compared against the known mapping-key positions; the selected row’s value representation was retrieved; the retrieved information was injected into the answer path. Importantly, T12 did not hand Baby the answer. It provided a purpose-built computation for performing the relation the earlier transformer repeatedly failed to organize itself.

| **Metric**                     | **Step 50** | **Step 1000 / final training** |
|--------------------------------|-------------|--------------------------------|
| Correct-row retrieval argmax   | 100%        | 100%                           |
| Correct-row attention weight   | 99.976%     | 0.999933                       |
| Incorrect-row attention weight | ~0.024%     | 0.000067                       |
| Retrieval score margin         | +11.19      | +12.748                        |
| Answer accuracy                | 100%        | 100%                           |
| Target\>distractor             | 100%        | 100%                           |
| Answer margin                  | +9.92       | +13.141                        |
| Answer CE                      | 0.00049     | 0.0000082                      |

The frozen retention exam was the breakthrough: 160/160 answers, 40/40 complete counterfactual quartets, 80/80 reversals, and 100% correct-row retrieval. The T12 checkpoint hash preserved in the project record is e378d3d85f3a53add4046aefc7cc42d643f59d12525859af6ea79dbfca266cd5.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Strongest causal story through T12</strong></p>
<p>T10 repaired the counterfactual data and failed. T11 repaired the answer supervision and failed. T12 changed the routing computation and reached ceiling held-out behavior. This strongly implicated routing/value-transfer organization in the small model’s historical failure, while not yet isolating every subcomponent of the T12 circuit.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

# 5. Removing the Training Wheels: T13 and Synthetic Graduation

T12 still received explicit structural access to the relevant mapping rows. The next research phase asked how much of that scaffolding could be removed while retaining the capability. T13 began learned source localization, and the project entered a longer localization lineage: T13 baseline, direct distinct-localization supervision, a champion model, nonsource suppression, source-recognition auxiliary supervision, a bounded shared scorer, forensic autopsies, scorer transplants, and finally an orthogonal shared/unbounded design.

The methodological quality improved sharply here. Instead of immediately retraining on every surprise, the project froze checkpoints and inspected them. One important example was a coordinate-labeling correction: an initial interpretation claimed 32 bounded-scorer collapses favored the physically earlier source. A discriminator showed the opposite. The authoritative census was 24/32 collapses onto the physically later source and 8/32 earlier. Champion failures already leaned toward preserving the later source 20/30, and 24/32 bounded-collapse coordinates corresponded to positions previously owned by champion slot 0. The false earlier-source story was formally withdrawn before it could generate a treatment.

The bounded scorer itself taught another causal lesson. It repaired 22/30 original champion wandering cases but created a new true-source collapse population. The project’s mechanistic interpretation became: sharing could improve source recognition, but a bounded disagreement/assignment channel could prevent two localization slots from separating strongly enough.

## 5.1 Orthogonal shared core + unbounded antisymmetric residual

The replacement localizer used a shared source-recognition direction and an orthogonal unbounded assignment direction. Its persistent trainable parameter count was 641—essentially the same capacity as the bounded predecessor. The point was not more capacity, but a cleaner factorization of recognition and assignment.

| **Preflight comparison**         | **Bounded redesign** | **Orthogonal/unbounded redesign**  |
|----------------------------------|----------------------|------------------------------------|
| Logit RMSE vs baseline           | 0.775                | 0.13249                            |
| Maximum logit difference         | 3.538                | 0.54627                            |
| Attention KL                     | not highlighted      | 0.00199                            |
| Unordered top-1 set preservation | 62.5%                | substantially gentler perturbation |
| New collisions at zero update    | 4                    | 0                                  |

The preflight also verified the intended geometry: the assignment direction was orthogonal to the shared direction to numerical precision (uᵀv ≈ 3.57×10⁻9), the disagreement term had no tanh/rho bound, and the existing hard-min localization loss produced gradients that raised missed real sources while lowering nonsources in a synthetic wandering state.

## 5.2 Canonical synthetic-binding graduate

| **Metric**                 | **Graduate result** |
|----------------------------|---------------------|
| Answer accuracy            | 317/320 = 99.06%    |
| BOTH_DISTINCT localization | 312/320 = 97.5%     |
| Collapse                   | 0                   |
| Complete quartets          | 78/80 = 97.5%       |
| Reversals                  | 158/160 = 98.75%    |

All frozen graduation gates passed simultaneously. The canonical graduate checkpoint SHA-256 was fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430. The claim was deliberately narrow: Baby had demonstrated structurally assisted contextual binding across held-out strict counterfactual reversals and varied synthetic geometry. It did not establish general reasoning, ordinary-language understanding, or universal variable binding.

# 6. Language Acquisition, “he womputeld,” and Catastrophic Interference

Synthetic graduation created the next obvious problem: Baby still could not talk. When pointed at ordinary text, the graduate emitted fragments such as “proc,” “while,” “appe,” “cla,” and “under.” The project therefore began actual autoregressive language training rather than pretending the synthetic capability was language.

## 6.1 Pilot 0: language arrives, binding degrades

TinyStories Pilot 0 transformed the output. DEV perplexity dropped from roughly 1278 to 8.61 and recognizable English emerged. The project’s canonical first-words artifact became the phrase “he womputeld.” Surviving summaries associate the phrase with very early language output; regardless of exact run-label nuance, the project formally treats it as Baby’s first recorded words.

> “he womputeld”

The win had a cost: binding degraded. This was Baby’s first clear catastrophic-interference lesson. Training a useful new behavior could disrupt a carefully engineered old one.

## 6.2 Pilot 1: prove coexistence is possible

Pilot 1 restarted from the synthetic graduate, protected blocks 0–3 during language updates, and continued binding rehearsal. Language was weaker than Pilot 0 by perplexity but still substantial: DEV perplexity 18.25. Crucially, the held-out binding exam recovered to 80/80 answers, 80/80 BOTH_DISTINCT, zero collapse, 20/20 quartets, and 40/40 reversals.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Pilot 1 conclusion</strong></p>
<p>Language acquisition and contextual binding were not inherently mutually exclusive. The training procedure could preserve the old capability while adding substantial English modeling ability.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

A component-level autopsy of Pilot 0 also weakened the simple story that the special binding module itself had merely been overwritten. Shared hidden-state drift mattered. That finding pushed later experiments toward thinking about capabilities as distributed computations rather than isolated “features” living in one module.

# 7. P0–P7 and the Evaluation-Correction Era

The P0→P7 continuation phase focused on making English less repetitive and more compositionally meaningful. The exact details of every P-step are not fully surfaced in the present source packet, but the preserved record clearly describes the progression: sentence boundaries, repetition control, phrase reuse, compositional prompts, and the distinction between English-looking continuation and factual use of context.

## 7.1 P7: a real but narrow success

P7 achieved 12/12 on a frozen controlled compositional panel. Held-out examples completed a second sentence with the prompted subject/object and appropriate relation, and the complete prompt+response strings were not exact TinyStories or synthetic-training matches. This earned a narrow controlled-compositional success—not a claim that Baby understood ordinary English broadly.

## 7.2 The invalid report card and the decision to withdraw it

A post-P7 report card initially looked devastating. Then the evaluation itself was audited and 144/288 controlled answer keys were found wrong. The project withdrew the result. This was more important than the numerical score: the team chose to invalidate its own evidence rather than preserve a dramatic conclusion from a broken exam.

The independently validated replacement report was still bad, but now legitimately so: near-transfer 50%, counterfactual 50%, surface 48.44%, distractor 50%; 22/24 naturalistic prompts ended immediately in EOS. The corrected conclusion was precise: P7’s narrow 12/12 milestone was real, but Baby did not reliably use short ordinary-English contexts across selection, reversal, surface variation, and distractors.

# 8. Factual Supervision and the End of the ~10M Line

The next major program tried to teach ordinary-English factual/context use without losing language or binding. This evolved into the SF series. The full SF1–SF13 receipt set is not reproduced in the current corpus, but the archival handoffs preserve enough structure to reconstruct the late frontier and the final closure decision.

## 8.1 The four-way frontier

By the late SF series, success required four things at once: keep TRAIN16 exact factual behavior, generalize across Surface/Order widening, preserve language and synthetic binding, and keep D3—the mean mass assigned to four answer-name tokens on frozen ordinary English—at or below the sacred leakage threshold. This turned into a stability–plasticity and shortcut-learning knife fight.

SF8 remained the only reliable TRAIN16 recipe in one archival summary. SF9–SF19 often preserved narrow task behavior/language/binding but violated D3. Attempts to suppress D3 frequently damaged exact identity generation or widening.

## 8.2 SF14–SF19: symptoms moved around

| **Treatment** | **Intervention**                       | **Outcome**                                                                                  |
|---------------|----------------------------------------|----------------------------------------------------------------------------------------------|
| SF14          | Remove first-answer-token CE           | Contained D3, but exact generation deteriorated.                                             |
| SF15          | Restore partial first-token CE at 0.25 | Exact generation improved; D3 leakage returned.                                              |
| SF16          | Reduce CE to 0.125                     | Still did not solve the coexistence problem.                                                 |
| SF17          | Gradient projection                    | Projection was active but did not resolve D3.                                                |
| SF18          | Freeze output head                     | Showed leakage could be produced by upstream representation changes; did not solve frontier. |
| SF19          | Candidate-membership hinge             | Closest near-hit, but D3 drift returned and widening remained weak.                          |

## 8.3 SF20: identity rotation kills the shortcut—and much of the desired behavior

SF20 used identity-heldout, balanced entity rotation. It reduced D3 by roughly 70% (one preserved summary gives ~0.00718 → ~0.00212) while forced factual discrimination, reversals/families, language, and binding remained present. But exact retention and Surface/Order exact performance collapsed to roughly 2/16-scale behavior. The interpretation was “clean for the wrong reason”: Baby had stopped relying on the four sacred names, but it had not automatically discovered a robust abstract rule that still produced exact answers.

## 8.4 SF21: final 10M shot and formal series closure

SF21 added one new variable to SF20: paired counterfactual representation consistency. Structurally equivalent identity-swapped examples were encouraged to have similar pre-answer/query representations with a loss term 0.1×mean(1−cosine(vᵢ,vⱼ)). The curriculum, identities, CE/margin/KL settings, schedule, gates, and evaluation remained frozen. Seeds were 87056, 87057, and 87058.

| **Seed** | **Stop** | **TRAIN16 / widening**                       | **D3 before→after** | **Classification details**                                    |
|----------|----------|----------------------------------------------|---------------------|---------------------------------------------------------------|
| 87056    | U100     | 16/13/8/4; Surface 11/2; Order exact 2 (2/0) | 0.00704→0.00219     | Forced discrimination remained; exact/widening failed.        |
| 87057    | U100     | 16/13/8/4; Surface 10/2; Order exact 2 (2/0) | 0.00667→0.00238     | Same coexistence failure.                                     |
| 87058    | U200     | 16/8/8/4; Surface 13/1; Order exact 2 (2/0)  | 0.00784→0.00243     | Lost TRAIN16 retention by U200; language/binding still green. |

The classification was RETENTION_REGRESSION_10M_SERIES_CLOSED. There was no SF22. After 21 treatments, the project had not found an intervention that kept TRAIN16, Surface/Order widening, language/binding retention, and D3 green simultaneously. Scaling was therefore justified as a controlled capacity/interference experiment—not because “bigger is better,” but because the small regime had reached a repeatedly measured coexistence frontier.

# 9. Scaling to 61.52M: vNext and Phase1 / Phase1G

## 9.1 Architecture choice

The vNext audit verified Research Baby at 10,841,345 total parameters and compared larger candidate geometries. The selected successor was a 12-block, d_model 640 architecture with 10 heads×64, MLP 2560, vocab 1024, context 256, pre-norm, ReLU, dropout 0.05, learned absolute position embeddings, untied token embeddings/output, and fused QKV attention using SDPA. Parameter count: 60,536,064 base + 984,321 structural = 61,520,385 total.

The 12×640 design was preferred over alternatives such as 16×560 and 10×704 because it offered a balanced width/depth increase while staying practical on Fan Diesel. Weight morphism, direct transplantation, and distillation from the small model were rejected because the geometry changed; vNext began from a sealed random initialization.

## 9.2 Phase 1: capacity alone does not cure data reuse

The initial language recipe froze the architecture and used 6,000 updates, effective batch 64, FP32 AdamW, 200-update warmup, cosine learning-rate decay 3e−4→3e−5, weight decay 0.05, and gradient clip 2.0. The structural/binding parameters were frozen while base language parameters trained.

| **Checkpoint** | **Train CE** | **DEV CE** | **Gap / interpretation**                                                  |
|----------------|--------------|------------|---------------------------------------------------------------------------|
| U0             | ~7.087802    | 7.086638   | Random-init baseline.                                                     |
| U3000          | 0.9696       | 1.37509599 | Best original DEV checkpoint; gap +0.4055.                                |
| U6000          | 0.5667       | 1.48204355 | Train improves while DEV worsens; gap ~+0.915 → overfitting/data-limited. |

This was an important scientific result: the larger model did not look unstable; it looked underfed. More capacity lowered training loss faster but could not invent unseen data diversity. The project therefore resisted immediately scaling again to ~100M and instead improved the training diet.

## 9.3 Phase1G: change the diet, not the brain

Phase1G held architecture, optimizer, tokenizer, compute, evaluation windows, and sealed initialization fixed while expanding the TinyStories training stream from about 9,000 to 20,990 unique stories (8,354,565 tokens). Reuse fell from roughly 27.48× to 11.77×. U0 was identical by construction. The final/best U6000 DEV CE improved to 1.2040123894810677 with a much smaller generalization gap (~0.2562). This was a clean data-quality/generalization win and deferred another capacity jump.

Later forensic work discovered that the 984,321-parameter inherited structural sidecar was initialization-identical/untrained and not actually in the relevant forward path. This correction matters: those parameters exist in the model package but must not be credited for learned relational or language behavior in the vNext line.

# 10. The Representation–Readout–Generation Era (T17–T32)

Phase 2A then tried to make the 61M model use facts and relations in ordinary text while preserving Phase1G language. The most important outcome was not a single successful treatment but a much sharper decomposition of the failure path: representation could succeed while native readout and free-running generation still failed.

## 10.1 T17: replicated language-safe relational representation

T17 became a major milestone. Language remained healthy in all 3/3 seeds while relational representation passed in the required 2/3 seeds, including the frozen reversal requirement. Fact-clause representations proved much stronger than earlier name-span representations. Blocks 4–7 became strongly implicated in relational acquisition; blocks 8–11 appeared useful during acquisition but their altered state could later hurt language. T17’s solution let 4–11 participate during capability formation, then restored/froze 8–11 and continued refining 4–7.

But exact native output was still zero in T17. T17X made the dissociation concrete: in many examples the pointer/representation and even native candidate preference identified the correct answer, yet greedy generation drifted into TinyStories continuation. Baby could “know/select” internally without cleanly emitting and stopping on the answer.

## 10.2 T18–T24: answer-span/EOS and the non-monotonic tradeoff

T18 explicitly trained answer span + EOS—the “say the right answer, then stop talking” treatment. Later treatments explored the same representation-to-output frontier without proving a universal inverse law. Representative exact results preserved in committee summaries were: T21 representation 3/3 with exact 27/29/26; T23 representation 3/3 with exact 34/34/38; T24 representation 3/3 with exact 35/40/37; T18 representation 2/3 with exact 36/36/40. Better representation therefore did not guarantee better exact output, but it was not valid to claim that better representation always caused worse output.

## 10.3 T25–T32: teacher forcing, prefix rescue, and failed bridges

T25 exposed a teacher-forced versus free-running continuation gap. T26, T26B, and T27 were interrupted/resource-stop episodes rather than scientific successes. T28 finished normally but classified OUTPUT_FAIL with exact+EOS 36/35/40. Its prefix rescue was crucial: supplying the correct first token recovered exact+EOS continuation in 108/124 cases (87.1%). That sharply focused attention on answer initiation and autoregressive divergence rather than whole-string copying.

| **Treatment** | **Known result / role**                                                                                                  |
|---------------|--------------------------------------------------------------------------------------------------------------------------|
| T28           | OUTPUT_FAIL; exact+EOS 36/35/40; prefix rescue 108/124 = 87.1%.                                                          |
| T29           | ~35/35/35; representation success / output fail.                                                                         |
| T30           | 31/36/38; shared-prefix 2/64–3/64; representation success / output fail.                                                 |
| T31           | Explicit ordered source-token copy bridge; terminal seed ~35/128 exact, 1/64 shared; T31_FAIL_NO_REPRESENTATION.         |
| T32           | Seed 890001 ended T32_REPRESENTATION_SUCCESS_OUTPUT_FAIL; native exact ~35/128, shared-prefix 0/64 in preserved summary. |

By the end of this line the project had advanced the diagnosis much more than the capability. The rational move was no longer “turn another loss knob.” It was to inspect the generation path mechanistically: where does entity information remain decodable, where does the final output map fail, what changes under free running, and does the model re-retrieve information at later suffix positions or simply lose control after token zero?

## 10.4 External literature and the “committee”

At this point the project deliberately consulted outside model families and research literature. The goal was not to copy a module from Qwen, Devstral, Laya, or another model, but to learn better experimental questions. Literature on latent knowledge, factual recall stages, retrieval heads, tuned/logit lenses, probing limitations, activation patching, tokenization bias, counterfactual augmentation, compositionality, and continual learning reinforced the need to separate decodability from causal use and representation from routing/readout. It also cautioned against calling “state decay” a mechanism before direct evidence existed.

# 11. v0.10, v2R4/v2R5, and the Copy-vs-Selection Diagnosis

## 11.1 v2R4 U16000: the decisive copy/selection autopsy

The authoritative v0.10 checkpoint was v2R4 U16000 at runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt, SHA-256 94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827. A dedicated read-only autopsy separated “can Baby copy a present value?” from “can Baby select the requested value?”

| **Autopsy result**                                                         | **Count / finding**        |
|----------------------------------------------------------------------------|----------------------------|
| Novel multi-pair examples                                                  | 96                         |
| Generated an actual value from the context                                 | 93/96                      |
| Selected the value requested by the query                                  | 41/96                      |
| Selected a competing value from another pair                               | 52/96                      |
| Wrong-selection cases repaired by manually supplying requested first token | 51/52 correct continuation |

The 2-, 3-, and 4-pair behavior did not convincingly outperform approximately 1/K candidate choice, so the earlier idea that Baby had a reliably stronger two-pair binding skill was withdrawn. A matched U15500→U16000 comparison also killed the story that selection was improving simply by training longer. Generic copying had already saturated; query-conditioned first-token selection had not.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>v2R4 mechanism summary</strong></p>
<p>QUERY → choose answer → copy answer → finish. The evidence looked roughly like: query-conditioned CHOOSE ANSWER = weak; COPY/CONTINUE once started = strong. This became the central defect statement.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## 11.2 v2R5: do not invent a status

v2R5 received an unusually extensive audited preflight: 50,000 training rows, 10×256 evaluation panels, 128 paired multi-surface tasks, 21/21 audit checks, and 2/2 smoke tests. Codex reported seed 107001 reaching U10000 cleanly while 107002 was mechanically healthy during its run. But later cloud GitHub views did not contain the local v2R5 artifacts. The project therefore adopted the only defensible status: do not call v2R5 passed, and do not call it failed, until the local state is recovered/synced.

## 11.3 Repair families and P11

Several subsequent repair families—labeled S1, S2, and T1 in the v0.10 defect lineage—failed cleanly. One attention intervention drove a relevant attention statistic from roughly 0.04 to 0.93 without improving selection, weakening the idea that attention magnitude alone was the answer. P11 then used a first-answer-token identity overwrite and improved long-gap greedy exact selection from 71/215 (~33.0%) to 102/215 (~47.4%). But promotion remained blocked because retention/generalization had not been evaluated with the overwrite active. A promising mechanism without the required safety check was not enough.

# 12. Scaling Again: The 117.78M School Model

## 12.1 Why scaling was considered

The project’s position on model size had evolved. Early on, the ~10M model’s failures were clearly not proof that it was “too small”; several were data, routing, or curriculum failures. After repeated late-series Pareto tradeoffs, scaling to 61M was justified as an experiment. Later, the 61M line again accumulated evidence of a persistent selection/readout bottleneck. Even then, the project wrote down a guardrail: a failed class or quiz is not evidence of insufficient model capacity. Capacity expansion must be preceded by reproduced, characterized failure and tests of plausible data, training, retrieval, and interference explanations.

## 12.2 Frozen 118M design

On 23 September, the next MRCN-Alpha school design was frozen at 117,780,224 parameters: 12 layers, d_model 896, 14 heads with head_dim 64, d_mlp 3584, the same 1024 vocabulary and 256 context family, learned absolute positions, ReLU, and pre-norm. The width/heads/MLP changed 640→896, 10→14, and 2560→3584 relative to the 61M control.

| **Model era**       | **Base / total scale**               | **Layers**            | **d_model** | **Heads** | **d_mlp** | **Notes**                           |
|---------------------|--------------------------------------|-----------------------|-------------|-----------|-----------|-------------------------------------|
| Research Baby       | 10.59M base / 10.84M incl. binding   | 8 (audited late line) | 320         | 10        | 1280      | Late 10M SF series.                 |
| vNext               | 60,536,064 base / 61,520,385 package | 12                    | 640         | 10×64     | 2560      | Phase1/1G and Phase2A.              |
| Current school Baby | 117,780,224                          | 12                    | 896         | 14×64     | 3584      | Main Sep 23–30 lineage; MRCN-Alpha. |

Repository archaeology recovered the conceptual pretraining chain and established that missing intermediate blobs were not a reason to fabricate or casually widen weights. At one point execution was explicitly blocked because Phase1G language streams/schedules/builders and a zero-update init artifact were missing; the team refused to rebuild a substitute without provenance approval. That is another example of process discipline becoming part of the model lineage.

# 13. Pre-S1 Copy Forensics and the School Curriculum

## 13.1 School framing

The 118M era was explicitly reframed as “school.” The planned foundational operations included Copy, Exclusion, Pick, Twice, Reverse, and Concat, with one class at a time, held-out exams, retention checks, evidence capture, and a healthy checkpoint after each class. The philosophy was deliberately unlike dumping textbooks or encyclopedias into the model. Compute could increase the pace; it was not allowed to substitute for curriculum design.

## 13.2 Stage-0 and foundation-copy diagnosis

A Stage-0 check before S1 showed a stark generalization gap: practiced E2 examples scored 162/180 = 90%; novel-but-familiar pairs scored 49/60 = 81.7%; held-out/pseudo E2 collapsed to 3/40 = 7.5%. This triggered a foundation-copy diagnosis rather than an immediate new school lesson.

The autopsy found no convincing generic induction-style copy circuit. Generic induction scores were around 0.3% in one summary; another reports a best-head score of 0.025. Token/position embeddings and neighboring-position similarity behaved close to random in the relevant probes (neighboring-position similarity ~0.003). Random-token/list copying was near chance. Baby’s apparently strong “copy” behavior was therefore understood as template-, position-, landmark-, and practiced-piece-bound rather than a universal copy mechanism.

A lambda manipulation changed attention direction/tradeoffs but did not create generic copying. MLP deltas could improve copying without the same E2 damage produced by attention deltas. Mode cues remained decodable but gating was weak. RoPE was discussed as a causal test of a relative-position substrate, but it was not treated as an earned architecture change at that moment.

## 13.3 IND-001 and the path into S1

IND-001 was authorized as a controlled pre-S1 experiment on the current 118M Baby, comparing a repeated-span arm with a sham control and explicitly checking that token/position embeddings were optimizer-visible and receiving gradients. The broader goal was to distinguish “can this architecture learn a generic copy primitive at all?” from “has the existing curriculum merely reinforced specific practiced layouts?” S1 then became the formal single-token-copy generalization treatment.

# 14. S1: First Failure and the S1v3 Safety Saga

## 14.1 The first S1 treatment

The first S1 treatment did not pass. Its long-span score improved from 110/128 (85.94%) to 114/128 (89.06%), missing the 90% / 116-of-128 gate. Retention remained strong on several other slices: length-1 and multi were 100%; delayed retention was 92.19%/100%. Harder negative-distance slices moved unevenly: −5 declined 78.1%→71.9%, while −4 improved 81.2%→84.4%. The run used 1,200 optimizer calls, saved 1,100 updates, and reached a maximum clip incidence of 7%.

Start checkpoint SHA-256: 775e2ae9a49d6f50d691b44f15877d129f0cddb27e8f6855672c0ac713746f31. Final failed-S1 checkpoint SHA-256: 7e803b9b1b558ec4a89f63bbe0c7a600ece0c8ccb97f1802978b3a733f649b82. The preserved report name was SPAN_SEQUENCE_TREATMENT_001.md. The identity issue remained unresolved, and the project did not immediately scale or weaken the gate.

## 14.2 S1v3: safety code becomes part of the experiment

S1v3 introduced gradient/update safety machinery so that a long continuation could be stopped before a genuinely unstable run damaged the lineage. That safety system then became its own source of experimental contamination. An early version built its idea of normal per-layer parameter movement during tiny learning-rate warmup. When full-speed learning began, healthy movement looked anomalously large and triggered false shrinkage.

S1v3b exposed a deeper version of the same problem. The per-layer Δθ guard began falsely shrinking updates around u89–127 because its baseline was dominated by warmup-scale motion; this altered Baby before the later gradient-30 event around u200. A throttle also created a feedback loop: emergency tightening of the normal clip threshold from 5.0 to 2.5 caused more clipping, and those extra clips were then counted as evidence that instability persisted. The run was killed at u592 because the safety mechanism had contaminated the training trajectory.

## 14.3 S1v3c safety corrections

The repaired design delayed baseline estimation until after warmup, allowed a longer baseline window, emphasized major matrix weights rather than noisy small norm parameters, added a cooldown after one intervention, treated 5.0 as the normal clipping threshold even when emergency clipping temporarily tightened to 2.5, and released throttle after the danger condition had remained absent for 50 updates. The 3×-median sensitivity rule was intentionally left unchanged because that was the frozen user-requested rule; it would only be changed if a clean run later proved it over-sensitive.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Why this matters</strong></p>
<p>The project explicitly separated “Baby became unstable” from “the safety system changed Baby and then measured its own consequences.” Runs contaminated by guard bugs were not used as evidence about Baby’s learning capability.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

# 15. Local Hardware Failure and the Move to RunPod

The late S1v3 continuation also collided with hardware reality. On 29 September, S1v3h continuation 004 hard-froze the entire Windows machine while it was being monitored. This was not a normal Python exception or training-script crash: Windows itself became unresponsive. Earlier whole-PC freezes had already raised suspicion around GPU/driver/kernel behavior, RAM/commit/paging, shared GPU memory, or runtime deadlock.

Crucially, continuation 004 was the first attempt correctly attached to the actual memory-heavy Python worker with external resource monitoring. After reboot, the correct response was not “attempt 005.” The project preserved resource JSONL and Windows logs and paused relaunches. This separated model-science failure from host/runtime failure.

The practical solution was cloud execution on RunPod using NVIDIA hardware. Browser control and later MCP ideas were explored for pod management, but the core principle stayed the same: cloud compute was a mechanical acceleration/host-stability tool, not permission to change the scientific recipe. The final S1v3j continuation was therefore replayed and executed in a controlled cloud environment after lineage reconstruction and hash checks.

# 16. S1v3j: Certified Graduation at u4700

## 16.1 Frozen lineage and replay

The final lineage was reconstructed as S1v3g u1–u3200 → S1v3h u3201–u4000 → S1v3j u4001–u4800. The S1v3j schedule used cosine decay from roughly 1e−3 toward 2e−5 over the 800-update window. Before formal training, cloud replay reconstructed the prior u1–u4000 lineage without consuming new scientific updates. An omitted old manifest was supplied unchanged; provenance was preserved.

## 16.2 The final trajectory

| **Update** | **Held-out / Novel trajectory note**                                |
|------------|---------------------------------------------------------------------|
| u4000      | 117 / 116                                                           |
| u4100      | 122 / 123                                                           |
| u4200      | 124 / 121                                                           |
| u4300      | 125 / 124                                                           |
| u4400      | 118 / 124 — temporary regression/churn                              |
| u4500      | 125 / 124                                                           |
| u4600      | 126 / 127                                                           |
| u4700      | 128 / 128 — all gated counterfactuals perfect; PASS + certification |

The run did not simply monotonically improve. Errors moved around, and u4400 regressed on held-out. At u4200, seven novel errors differed from the parent’s failures; at u4600 one new error remained. The key fact is that by u4700 the error set was empty on the frozen gated panels.

## 16.3 Hard identities and “312”

Several historically difficult identities were tracked through the continuation. At the u4000 parent, margins included identity 802 +2.81, 501 +4.85, 312 +5.44, 645 +4.86, and 901 +8.83. A prior v3i episode had at one point thrown identity 312 from rank 4 into roughly rank 509, making it a memorable sentinel. At S1 graduation, identity 312’s margin was +10.06168; 802 was +7.20, 645 +9.27, and 901 +7.90. The important result was not only aggregate perfection but elimination of the recurring hard cases without replacing them with new failures.

## 16.4 Certified gated result

| **Frozen gated metric**    | **Result**                                        |
|----------------------------|---------------------------------------------------|
| Held-out panel             | 128/128                                           |
| Novel panel                | 128/128                                           |
| Counterfactuals — held-out | 64/64                                             |
| Counterfactuals — novel    | 64/64                                             |
| Independent certification  | 1.00 across every frozen gated metric             |
| Early-stop behavior        | Triggered immediately at u4700; u4800 not touched |

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>S1 milestone</strong></p>
<p>S1v3j passed every prospectively frozen S1 graduation gate simultaneously. This is the strongest unambiguous statement supported by the final receipt: BABY PASSED S1.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## 16.5 Clean execution, artifact return, and cost

The final run was mechanically clean: no skipped optimizer updates, no NaNs, no OOM, no abort, and no parameter-delta anomaly. Mean/representative loss collapsed from about 0.1256 in the first hundred updates to about 0.00617 in the last hundred; clipping fell to roughly 4% in the final window. The formal cloud execution lasted 4 minutes 24 seconds and cost approximately \$0.34.

The graduate artifact is S1_single_token_copy_graduated.pt at u4700, SHA-256 200e02188063885560edb4de0d6e0f05054fa75c236c0264d4a2c461ec51771b. Seven scheduled checkpoint/evaluation pairs and 245 evidence files were returned and hash-verified. The graduate loaded locally; model and optimizer tensors were finite; RNG/config/continuation state were preserved; the entire 4,700-update history was reconstructable. The cloud pod was shut down to \$0.00/hr.

## 16.6 The unresolved nongated frontier

A nongated identity-stress panel remained 0/128. The final report explicitly warned that S1 passing does not establish Pioneer v1.0 release readiness. That unresolved panel is not a hidden asterisk on S1—the S1 gates were what they were—but it is a boundary on the broader claim. The scientific record therefore contains both statements at once: S1 graduated perfectly; Pioneer v1.0 remains a separate decision.

# 17. Post-Graduation Preservation, Naming, and Family Tree

## 17.1 Conservative cleanup

After graduation, Cursor was used as a conservative archivist rather than a chainsaw. It found the actual S1 graduate, verified the hash, and refused to delete backup .pt files merely because they looked redundant; some could be unique intermediate evidence. It deleted only verified-safe ZIP duplication and a verified duplicate graduate checkpoint, recovering roughly 10.6 GB while preserving intermediate captures.

The public/project README was updated to say “S1 PASSED — BABY GRADUATED THIS STAGE,” including the perfect gated result, u4700 stop, unresolved identity-stress frontier, and the explicit statement that Baby had not been falsely promoted to Pioneer v1.0. Failed history stayed failed; old model eras remained preserved rather than rewritten.

## 17.2 Naming system

The project identity was formalized into separate layers: DaveLM is the laboratory/ecosystem; Mangaris is the model family; Pioneer is Baby’s model class/ancestor role; Baby remains the codename and nickname; MRCN is the architecture lineage. The current school model belongs to MRCN-Alpha. Greek generations—Alpha, Beta, Gamma, and so on—are reserved for meaningful architecture generations rather than routine tweaks.

| **Identity layer** | **Canonical meaning**                               |
|--------------------|-----------------------------------------------------|
| DaveLM             | Project / research ecosystem / laboratory           |
| Mangaris           | Formal model family                                 |
| Pioneer            | Historic original research ancestor / Baby class    |
| Baby               | Development codename and permanent nickname         |
| MRCN-Alpha         | Current architecture generation / chassis           |
| ~118M              | Current school-model scale, not the family identity |

## 17.3 Clean master and descendants

The preservation plan is explicit: a release-ready canonical ancestor is frozen and never brainwashed in place. Personal customization and specialist variants are descendants. Planned branches include Instruct, Reasoning, Coder, and Writer; a later Personal/Dave line can contain personality, slang, preferences, tools, RAG, sports, and other customization. If a descendant is ruined, it can be deleted and recreated from the clean ancestor. The historical joke is accurate: save-scumming the child’s brain, but with real provenance discipline.

# 18. What Baby Has Actually Proven

1.  A from-scratch personally built transformer can be trained, checkpointed, diagnosed, scaled, and experimentally modified under a reproducible local/cloud workflow rather than merely used as a black-box downloaded model.

2.  In the small-model era, Baby learned strict counterfactual contextual binding under explicit query-conditioned retrieval and later under learned localization, culminating in a synthetic-binding graduate with zero collapse and high reversal accuracy.

3.  Internal relational information can be strong while final behavioral output remains weak. This was observed repeatedly: T8, T17/T17X, T28 prefix rescue, and the v2R4 first-token continuation autopsy all point to representation/use dissociations.

4.  Language can be acquired without inevitably destroying synthetic binding. Pilot 1 demonstrated coexistence under a carefully protected/rehearsed training procedure.

5.  Perplexity and English-looking output are not equivalent to factual context use. P7’s controlled milestone and failed naturalistic transfer made that distinction concrete.

6.  Evaluation errors can be caught and withdrawn. The 144/288 bad-key report card was not allowed to stand, and a later coordinate-labeling error was formally reversed before it generated a treatment.

7.  The ~10M factual-supervision regime exhibited a repeated coexistence frontier among exact generation, widening, shortcut suppression, and retention. Closing the series instead of endlessly tuning it was evidence-based.

8.  The 61M model benefited more from a better diet than from immediate additional scale. Phase1G improved held-out language generalization while the architecture stayed fixed.

9.  On v2R4, copying/continuation was strong while query-conditioned first-token selection was weak. This was established behaviorally, not inferred from a single attention visualization.

10. The current ~118M lineage passed S1’s frozen single-token-copy generalization gates perfectly at u4700 with independent certification and exact provenance.

# 19. What Baby Has Not Proven

- S1 graduation is not proof that Pioneer v1.0 is release-ready.

- The nongated identity-stress frontier remains unresolved (0/128 in the final S1 report).

- Baby has not demonstrated broad conversational competence, robust instruction following, encyclopedic knowledge, coding skill, multimodal perception, or general reasoning.

- T12 did not prove that standard transformers universally require a dedicated retrieval gadget; it proved that this scaffold solved Baby’s isolated task under that experiment.

- Synthetic binding success is not the same as natural-language semantics or universal variable binding.

- Probe/logit-lens decodability is not automatically causal use; the project explicitly learned to distinguish those concepts.

- A larger model is not automatically a better model. The project’s own rules prohibit using one failed class as proof of insufficient capacity.

- The 984,321-parameter vNext sidecar cannot be credited for relational behavior in the line where it was later found init-identical/untrained and absent from the forward path.

- v2R5 has no legitimate pass/fail classification in the preserved state until its local artifacts are recovered and reconciled.

# 20. Methodological Lessons and the DaveLM Research Constitution

The most important product of the first month may be the research method, not any single checkpoint. Several rules emerged repeatedly and should be treated as part of the DaveLM constitution:

- One scientific variable at a time whenever possible. Do not bundle architecture, tokenizer, optimizer, data, and objective changes into one “fix.”

- Freeze evaluation panels, schedules, gates, and stop rules before training. Do not move the goalposts after seeing results.

- Keep protected TEST/FINAL/SACRED sets sealed until the preregistered condition for opening them is met.

- An invalid exam produces an invalid conclusion. Withdraw it and repair the test.

- Do not infer mechanism from outcome alone. Use frozen diagnostics or causal interventions where needed.

- Do not credit a module merely because it exists in a checkpoint. Verify that it is trained and in the forward path.

- Do not confuse internal decodability with actual causal use.

- Do not let the coding agent independently decide both diagnosis and treatment. Dave + the scientific reviewer define the question; the agent implements the frozen specification.

- Do not escalate to more model capacity until reproduced evidence supports a persistent capacity/interference limitation.

- Do not keep training merely because the loss is falling. Held-out generalization decides whether learning is useful.

- When a run is contaminated by instrumentation/safety bugs, treat it as a tooling failure, not a Baby result.

- Preserve hashes, manifests, RNG/optimizer states, reports, and failed checkpoints so the history can be reconstructed.

- Cloud compute changes speed and host reliability; it does not change the syllabus or scientific rules.

- After a genuine major milestone, freeze the artifact and update “State of the Child” before moving on.

# 21. The Planned Post-S1 Curriculum

The project has already designed a deliberate post-graduation education. The immediate principle is “school, not information dumping.” Foundation skills should be taught in increasing complexity, with older skills periodically re-tested. The long-range sequence is: natural-language binding and reference tracking → genuine language competence → instruction following → curated knowledge → coding → deliberate reasoning → tools/memory → assistant behavior → optional multimodal and multi-agent descendants.

A separate long-term idea, inspired by decision-first systems such as Laya but not copied from them, is a DaveLM-specific deliberation layer: read → retrieve relevant state → perform latent operations → optionally iterate → verify/confidence-check → commit → speak. The project explicitly prefers latent/compact internal reasoning over training Baby to emit verbose fake “thought” text, with explanations learned as a separate communicative behavior.

Specialized descendants are planned only after a clean common baseline exists. The working experiment-family idea is to clone the same healthy ancestor into Instruct, Reasoning, Coder, and Writer branches and train them separately. This preserves comparability and avoids permanently entangling every specialization in one checkpoint.

# 22. Conclusion

Between 26 August and 30 September 2026, Baby changed category several times. It began as “can I build a tiny transformer?” It became a binding research organism, then a language learner, then a capacity/interference experiment, then a representation/readout puzzle, then a larger school model, and finally an S1 graduate. The line is not clean because real experiments are not clean. Several treatments failed. Several attractive explanations were wrong. An exam was invalidated. Coordinates were labeled backward. A structural sidecar turned out not to be doing the job people assumed. A safety system contaminated its own runs. A Windows host froze hard. The final solution therefore matters partly because it survived the project’s habit of attacking its own conclusions.

The most defensible final statement is also the simplest: at u4700 on 30 September 2026, the current ~118M Baby lineage achieved perfect performance on every frozen gated S1 held-out, novel, and counterfactual metric, passed independent certification, stopped under the prospective early-stop rule, and produced a hash-verified graduate artifact. That is a real graduation. It is not the end of Baby’s education. It is the point at which the project can finally say, with evidence rather than hope, that one foundational class is finished.

> “S1 PASSED — BABY GRADUATED THIS STAGE.”

# Appendix A. Master Chronology

| **Date / period**  | **Milestone**                                                                                                                                                                                     |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ~26 Aug            | Active DaveLM/Baby development begins; from-scratch tiny decoder-only transformer; synthetic contextual binding becomes first core research task.                                                 |
| Late Aug           | Small architecture audited around 10.59M base / 10.84M with binding; early shortcut/generalization diagnostics show success on familiar recombinations can hide positional/answer-role shortcuts. |
| Late Aug–early Sep | T1–T9 progressively isolate candidate-recognition vs selection; T8 finds strong contextualized query signal; T9 generic retrieval fails.                                                          |
| Early Sep          | Structural census finds 0 strict reversed-binding counterfactuals in 1,536 pair structures / 3,072 twin docs / 32,000 events.                                                                     |
| Early Sep          | T10 strict counterfactual curriculum: training sequence metrics look excellent, retention remains 50%. T11 answer-only supervision still fails.                                                   |
| Early Sep          | T12 explicit query-conditioned mapping retrieval: 100% training and 160/160 frozen retention, 40/40 quartets, 80/80 reversals.                                                                    |
| ~Sep 5             | T13 localization lineage culminates in 641-parameter orthogonal shared/unbounded scorer and synthetic binding graduation: 317/320 answers, 312/320 distinct, 0 collapse.                          |
| Early Sep          | Language Pilot 0: TinyStories perplexity ~1278→8.61; first words “he womputeld”; binding degrades.                                                                                                |
| Early Sep          | Pilot 1: language + binding coexistence shown; perplexity 18.25 with perfect 80/80 small binding retention panel.                                                                                 |
| Early Sep          | P0–P7 language continuation; P7 12/12 controlled compositional milestone; naturalistic transfer weak.                                                                                             |
| Early Sep          | Report-card v1 invalidated after 144/288 wrong keys; corrected v2 shows ~chance fact use and 22/24 immediate EOS.                                                                                 |
| 9 Sep              | SF21 final ~10M shot fails coexistence; 10M factual-supervision series formally closed; no SF22.                                                                                                  |
| 9 Sep              | vNext selected: 12×640, 61,520,385 packaged parameters; from-scratch sealed initialization.                                                                                                       |
| 9 Sep              | Original Phase1 U3000 best DEV CE 1.3751; U6000 overfits to DEV 1.4820.                                                                                                                           |
| 9–10 Sep           | Phase1G expands diet to 20,990 TinyStories / 8.35M tokens; U6000 DEV CE improves to ~1.204.                                                                                                       |
| 11–14 Sep          | Phase2A T17–T32: replicated relational representation emerges; exact native output remains weak; T28 prefix rescue 108/124.                                                                       |
| 16 Sep             | v2R4 U16000 autopsy: copy actual context value 93/96, requested value only 41/96; wrong cases continue correctly 51/52 after correct first token supplied.                                        |
| 17 Sep             | P11 first-token identity overwrite improves long-gap greedy 71/215→102/215 but is not promotable without active-overwrite retention/generalization.                                               |
| 23 Sep             | 117,780,224-param school design frozen: 12×896, 14 heads, d_mlp 3584; provenance rules block unapproved reconstruction when Phase1G artifacts are missing.                                        |
| 25–27 Sep          | Stage-0/foundation-copy autopsy finds practiced copying strong but pseudo-token generalization weak; no convincing generic induction circuit.                                                     |
| 27 Sep             | First S1 treatment improves long-span 110/128→114/128 but misses 116/128 gate; no scale-up or weakened threshold.                                                                                 |
| 27–29 Sep          | S1v3 safety lineage: warmup baseline and throttle bugs discovered; contaminated runs rejected; S1v3c corrections defined.                                                                         |
| 29 Sep             | S1v3h continuation 004 hard-freezes entire Windows host; relaunch paused; cloud execution chosen.                                                                                                 |
| 30 Sep             | Cloud lineage replay/reconstruction succeeds. S1v3j resumes u4001 and reaches perfect frozen gates at u4700.                                                                                      |
| 30 Sep             | S1 graduate artifact returned/hash-verified with seven scheduled checkpoint/eval pairs and 245 evidence files; pod stopped to \$0/hr.                                                             |
| 30 Sep             | Conservative cleanup preserves unique intermediates, recovers ~10.6GB safe junk, and records family tree / GitHub README milestone.                                                               |
| 30 Sep             | Formal identity sharpened: DaveLM ecosystem; Mangaris family; Pioneer class; Baby codename; MRCN-Alpha architecture generation.                                                                   |

# Appendix B. Treatment and Experiment Matrix

| **ID**            | **Purpose**                                         | **Canonical status / result**                                        |
|-------------------|-----------------------------------------------------|----------------------------------------------------------------------|
| T1                | One-map bootstrap                                   | Candidate set recognized; selector fails.                            |
| T1.5              | Curriculum bridge                                   | No reliable bridge.                                                  |
| T2                | Paired query swaps                                  | Query following still unreliable.                                    |
| T3                | Correct\>distractor objective                       | Loophole: both candidates suppressed.                                |
| T4                | Membership+selection objective                      | Promising training; no official frozen pass.                         |
| T5                | Reciprocal/counterfactual twins                     | ~50.1% c\>d.                                                         |
| T6                | ~191× answer weighting                              | ~54.7% c\>d.                                                         |
| T7                | Raw query embedding probe                           | ~75.7% c\>d.                                                         |
| T8                | Contextualized query probe                          | 85.4% c\>d; gate still failed.                                       |
| T9                | Generic late retrieval                              | 52.3% top-1/c\>d.                                                    |
| Census            | Outcome-blind structural census                     | 0 strict reversed-binding structures.                                |
| T10               | Strict counterfactual quartet curriculum            | Full-token train strong; retention 50%.                              |
| T11               | Answer-focused supervision                          | Still ~chance; c\>d 50%.                                             |
| T12               | Query-conditioned mapping retrieval                 | 160/160 retention; 40/40 quartets; 80/80 reversals.                  |
| T13+              | Learned localization lineage                        | Led to champion, bounded scorer, autopsies, scorer transplants.      |
| Orthogonal scorer | 641-param shared-recognition + unbounded assignment | Synthetic graduate: 317/320 answers, 0 collapse.                     |
| Pilot 0           | TinyStories language                                | PPL 1278→8.61; “he womputeld”; binding damage.                       |
| Pilot 1           | Protected language + binding rehearsal              | PPL 18.25; binding retention perfect on small panel.                 |
| P0–P7             | Language continuation/compositionality              | P7 12/12 controlled; naturalistic transfer failed.                   |
| SF14              | Zero first-token CE                                 | D3 contained; exact hurt.                                            |
| SF15              | First-token CE 0.25                                 | Exact restored; D3 leakage.                                          |
| SF16              | First-token CE 0.125                                | Insufficient.                                                        |
| SF17              | Gradient projection                                 | Did not solve D3.                                                    |
| SF18              | Output-head freeze                                  | Leakage attributable upstream; not solved.                           |
| SF19              | Candidate-membership hinge                          | Near hit; D3/widening fail.                                          |
| SF20              | Identity rotation                                   | D3 ~70% lower; exact/widening collapse.                              |
| SF21              | Paired representation consistency                   | Final 10M shot; coexistence fail; series closed.                     |
| Phase1            | 61M language pretrain                               | U3000 champion; U6000 overfit.                                       |
| Phase1G           | Expanded language diet                              | U6000 DEV CE ~1.204; clear generalization win.                       |
| T17               | Relational representation + language retention      | Representation 2/3; language 3/3; exact output weak.                 |
| T17X              | Representation/native/generation diagnostic         | Correct internal preference can still free-run wrongly.              |
| T18               | Answer span + EOS                                   | Directly targets exact answer and stopping.                          |
| T21/T23/T24       | Representation/output variants                      | Representation often 3/3; exact still ~27–40/128.                    |
| T25               | Teacher-forced vs free-running diagnostic           | Continuation gap exposed.                                            |
| T26/T26B/T27      | Resource-stop episodes                              | No promotable scientific success.                                    |
| T28               | Output-focused treatment                            | 36/35/40; prefix rescue 108/124.                                     |
| T29               | Follow-up                                           | 35/35/35; rep success/output fail.                                   |
| T30               | Follow-up                                           | 31/36/38; shared-prefix low.                                         |
| T31               | Source-token copy bridge                            | Fails representation/output target.                                  |
| T32               | Later bridge                                        | Representation success/output fail.                                  |
| v2R4 autopsy      | Copy-vs-selection isolation                         | 93/96 copy a value; only 41/96 requested; 51/52 continuation rescue. |
| v2R5              | Audited preflight / partial runs                    | Status intentionally unresolved pending local artifacts.             |
| P11               | First-token identity overwrite                      | 71/215→102/215 long-gap greedy; not promotion-proven.                |
| Stage-0/E2        | 118M copy diagnosis                                 | Practiced 90%; held-out/pseudo 7.5%.                                 |
| IND-001           | Repeated-span vs sham induction test                | Designed to test generic copy substrate.                             |
| S1                | Single-token copy treatment                         | 114/128 long-span; missed 116 gate.                                  |
| S1v3/b            | Safety-instrumented continuations                   | Guard bugs contaminated runs; killed/reworked.                       |
| S1v3c+            | Corrected safety lineage                            | Baseline/cooldown/throttle fixes.                                    |
| S1v3h             | Local continuation                                  | Host hard freeze; moved to cloud.                                    |
| S1v3j             | Final cloud continuation                            | u4700 perfect 128/128 + 128/128; certified S1 graduate.              |

# Appendix C. Architecture Timeline

| **Era**                  | **Architecture**                                                                 | **Parameter scale**                            | **Key role**                                                                         |
|--------------------------|----------------------------------------------------------------------------------|------------------------------------------------|--------------------------------------------------------------------------------------|
| Early V082               | 4 blocks; d_model 320; 10 heads; FFN 1280; vocab 1024; ctx 256                   | ~10.6M note                                    | Earliest recorded experimental Baby configuration.                                   |
| Research Baby            | 8 layers; d_model 320; 10 heads; FFN 1280                                        | 10,594,944 base; 10,841,345 w/binding          | T/SF small-model research line.                                                      |
| vNext 60M                | 12 layers; d_model 640; 10×64 heads; d_mlp 2560; pre-norm; ReLU; learned abs pos | 60,536,064 base + 984,321 package = 61,520,385 | Phase1/1G + relational/output research.                                              |
| School Baby / MRCN-Alpha | 12 layers; d_model 896; 14×64 heads; d_mlp 3584; same 1024 vocab/256 ctx family  | 117,780,224                                    | Late-September school/S1 lineage.                                                    |
| Future                   | MRCN-Alpha revisions or later MRCN-Beta+                                         | Evidence-dependent                             | No capacity increase is automatic; must be justified by controlled failure analysis. |

# Appendix D. Canonical Artifacts and Hashes

| **Artifact / checkpoint**                       | **SHA-256 / identifier**                                         | **Meaning**                                                            |
|-------------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------------|
| T12 final checkpoint                            | e378d3d85f3a53add4046aefc7cc42d643f59d12525859af6ea79dbfca266cd5 | First perfect strict counterfactual held-out routing/retrieval result. |
| Synthetic-binding graduate                      | fed298748e62def6f1751ef1bb7df0cdec51d53fac6e4c86e8c00a95dccdd430 | 317/320 answer, 312/320 distinct, 0 collapse.                          |
| v2R4 U16000                                     | 94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827 | Authoritative v0.10 copy-vs-selection checkpoint.                      |
| First S1 start                                  | 775e2ae9a49d6f50d691b44f15877d129f0cddb27e8f6855672c0ac713746f31 | Parent of first S1 treatment.                                          |
| First S1 failed final                           | 7e803b9b1b558ec4a89f63bbe0c7a600ece0c8ccb97f1802978b3a733f649b82 | Failed 114/128 long-span S1 final.                                     |
| S1 graduate — S1_single_token_copy_graduated.pt | 200e02188063885560edb4de0d6e0f05054fa75c236c0264d4a2c461ec51771b | u4700 certified S1 graduate; crown-jewel current artifact.             |

Other frozen hashes appeared throughout treatment specifications for train pools, retention pools, schedules, and intermediate checkpoints. This appendix lists only the hashes that materially define the canonical model lineage in the current reconstructed record.

# Appendix E. Glossary

| **Term**                 | **Meaning**                                                                                                                                            |
|--------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| Baby                     | Permanent codename/nickname for the DaveLM/Mangaris Pioneer model lineage.                                                                             |
| DaveLM                   | The broader project, laboratory, tooling, data, experimental and model ecosystem.                                                                      |
| Mangaris                 | Formal model family name inspired by the user’s surname.                                                                                               |
| Pioneer                  | Historic ancestor/model class intended for the original successful Baby lineage.                                                                       |
| MRCN-Alpha               | Current architecture generation/chassis.                                                                                                               |
| Binding                  | Using the current query/key to select the value related to it in local context.                                                                        |
| Counterfactual reversal  | A matched example where the same query/candidates appear but the correct binding is deliberately reversed.                                             |
| BOTH_DISTINCT            | Localization success where two slots identify two distinct legitimate sources rather than collapsing together.                                         |
| Collapse                 | Both localization slots selecting the same source.                                                                                                     |
| D3                       | Mean probability mass assigned to specific answer-name tokens on frozen ordinary-English text; used as a leakage/shortcut diagnostic in the SF series. |
| TRAIN16                  | A narrow factual task/retention panel used in late small-model factual-supervision work.                                                               |
| Surface/Order            | Widening/generalization panels that varied formatting/order beyond the narrow trained surface.                                                         |
| Pointer / representation | An internal diagnostic/readout showing relational answer information independent of full greedy output.                                                |
| Prefix rescue            | Supplying the correct first answer token and testing whether the model can continue the correct remaining answer.                                      |
| S1                       | The late-September single-token-copy generalization class/stage on the ~118M school model.                                                             |
| Frozen gate              | A prospectively defined evaluation threshold that cannot be changed after seeing outcomes.                                                             |
| Preflight                | Zero-update mechanical/scientific validation performed before authorized training.                                                                     |
| NO ❤️                    | Project slang for a scientifically invalid, blocked, failed, or explicitly stopped treatment/action; also Baby’s joking historical “attitude.”         |
| he womputeld             | Canonical first recorded Baby phrase / project folklore milestone.                                                                                     |

# Appendix F. Full Curriculum Wishlist

This appendix preserves the future-school ideas already written down during the first month. They are plans, not accomplishments. Their inclusion matters because they show how the project intended to build on S1 rather than declare victory and immediately dump arbitrary corpora into the model.

## Phase 1 — Natural-language foundations

- 1\. Natural-language binding

- 2\. Pronouns / references

- 3\. Attribute binding

- 4\. Multiple properties per thing

- 5\. Three, four, five competing mappings

- 6\. Facts in random positions

- 7\. Distractor facts

- 8\. Negation

- 9\. Comparisons

- 10\. Ordering

- 11\. Small multi-hop reasoning

- 12\. Controlled compositionality

- 13\. Variable sentence structure

- 14\. Multiple questions about the same context

- 15\. Longer-context retrieval

## Phase 2 — Learn English for real

- 16\. Core vocabulary

- 17\. Grammar

- 18\. Sentence completion

- 19\. Tiny coherent paragraphs

- 20\. Dialogue

- 21\. Instruction vocabulary

## Phase 3 — Conversational capability

- 22\. Question answering

- 23\. Summarization

- 24\. Classification

- 25\. Rewriting

- 26\. Structured output

- 27\. Admit uncertainty

- 28\. Contradiction detection

- 29\. Follow constraints

## Phase 4 — General education

- 30\. Dictionary knowledge

- 31\. Basic world knowledge

- 32\. Basic science

- 33\. Geography

- 34\. History

- 35\. Math concepts

- 36\. Everyday practical knowledge

- 37\. Media / cultural knowledge

## Phase 5 — Coding

- 38\. Python syntax

- 39\. Read code

- 40\. Tiny code generation

- 41\. Debugging

- 42\. Explain code

- 43\. Tests

- 44\. JavaScript

- 45\. C / C++

- 46\. HTML/CSS

- 47\. SQL

- 48\. Baby writes code for Baby

## Phase 6 — Deliberate reasoning

- 49\. Step decomposition

- 50\. Worked examples

- 51\. Verification

- 52\. Alternative solution paths

- 53\. Planning

- 54\. Conditional reasoning

- 55\. Formal logic

- 56\. Analogies
- 57\. Counterfactual reasoning

- 58\. Causal reasoning

## Phase 7 — Tools and external memory

- 59\. Calculator

- 60\. Code interpreter

- 61\. Local document search

- 62\. RAG

- 63\. Structured databases

- 64\. Persistent conversation memory

- 65\. Filesystem interaction

- 66\. Shell commands

## Phase 8 — Assistant behavior

- 67\. Base / Instruct / Chat variants

- 68\. Personality tuning

- 69\. Long-term conversational consistency

- 70\. Ask clarifying questions

- 71\. Self-correction

- 72\. Source-backed answers

## Phase 9 — Optional / experimental / fun

- 73\. Teach Baby about Baby

- 74\. Baby reads its own research reports

- 75\. Baby analyzes its own failures

- 76\. School grades

- 77\. Report cards over time

- 78\. DaveLM trivia night

- 79\. Teach Baby Philly sports

- 80\. UFC analyst Baby

- 81\. Baby plays text adventures

- 82\. Baby as Dungeon Master

- 83\. Baby plays simple games

- 84\. Baby gets a voice

- 85\. Baby sees images

- 86\. Baby understands screenshots

- 87\. Baby becomes an NPC brain

- 88\. Multiple Babies talk to each other

- 89\. Specialized Baby personalities

- 90\. Baby teaches Baby

Standing rule attached to the wishlist: every future Baby is dragged back through the old exams. New language, knowledge, code, tools, or reasoning do not excuse regression on previously earned capabilities.

# Appendix G. Source Corpus and Provenance Notes

The paper was synthesized from the project record available in the conversation and local artifact bundle as of 30 September 2026. The sources are project-internal rather than a public peer-reviewed archive; this section makes the evidence basis explicit.

| **ID** | **Source**                                                                                           | **Use**                                                                                                                                    |
|--------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| PR-01  | Project conversation / experiment receipts, 26 Aug–30 Sep 2026                                       | Primary chronological source for late S1, naming, scaling, corrections, and preserved metrics.                                             |
| PR-02  | BABY TALK WOOOP.txt                                                                                  | State-of-the-child history through the early binding/language era; T12, localization graduation, Pilot 0/1, P7 and report-card correction. |
| PR-03  | DaveLM_Baby_EndOfChat_Master_Handoff_2026-09-09.docx (archived record summarized in project context) | SF14–SF21; 10M line closure; 61.52M architecture; original Phase1 training and scaling rationale.                                          |
| PR-04  | DaveLM_Baby_Full_History_Siri_Handoff_2026-09-17.pdf (archived record summarized in project context) | Phase2A/T17–T32, v2R4/v2R5 state, S1/S2/T1 defect lineage, P11, sidecar correction.                                                        |
| PR-05  | Baby Development Update.txt                                                                          | T17/T17X representation-vs-output status and immediate output-learning frontier.                                                           |
| PR-06  | Baby Curriculum Planning.txt                                                                         | Full future curriculum / regression-test philosophy.                                                                                       |
| PR-07  | Baby Treatment Gates.txt                                                                             | Late school / compute philosophy and ~118M-era curriculum principles.                                                                      |
| PR-08  | BABY DOES A NO ❤️.txt + BABY SAYS NO ❤️.txt                                                          | External-model workflow and capacity-expansion constitution.                                                                               |
| PR-09  | Baby Be Learning! 👶.txt                                                                             | Long-term deliberation/think-before-speaking concept.                                                                                      |
| PR-10  | Review HR2 Diagnostics.txt + Explain API Costs.txt                                                   | Outside-model/API committee and prompt/research workflow design.                                                                           |
| PR-11  | Final S1 certification receipt / project record, 30 Sep 2026                                         | u4700 perfect gated metrics, certification, artifact hash, evidence return, cloud cost, unresolved identity-stress panel.                  |
| PR-12  | Post-graduation cleanup / GitHub README record, 30 Sep 2026                                          | Conservative archival cleanup, family-tree preparation, public milestone wording.                                                          |

Known reconstruction limits: not every intermediate training receipt from every P/T/SF/S1v3 subrun is physically present in the local attachment directory. When a later canonical handoff preserved the conclusion but not the exact scalar trace, this paper reports the lineage-level conclusion and labels exact metrics only where they are available. No missing scalar was guessed.

**Historical note**

*First recorded words: “he womputeld.”  
First certified S1 graduation: u4700, 30 September 2026.  
The child is not finished. The child finally passed the class.*