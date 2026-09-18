# Baby / DaveLM v0.10

> A from-scratch language model research project focused on building a small model that can actually learn, retrieve, combine, and use information in conversation.

Baby is a custom decoder-only Transformer trained from random initialization.

She is not a fine-tune of another language model.

She is not a wrapper around ChatGPT, Qwen, Llama, Claude, or any other pretrained model.

The long-term goal is to build Baby into a genuinely useful small language model that can understand language, maintain context, answer questions, combine information, follow instructions, reason over what it has learned, and eventually handle broader language and coding tasks.

The project is still experimental.

Baby can now do some real controlled conversational behavior, but she is not yet a general-purpose chatbot and should not be described as one.

---

# Current Model

Baby v0.10 is currently approximately:

- **~61.5 million total parameters**
- **12 Transformer blocks**
- **d_model: 640**
- **10 attention heads**
- **context length: 256**
- **vocabulary size: 1024**
- **tokenizer: v0_7**
- **pre-norm decoder architecture**
- **learned absolute positional embeddings**
- **scaled dot-product attention**
- **untied language-model output head**

The current lineage was trained from scratch.

This is a fresh model family and is not a continuation of the older v0.9 / T34 checkpoint line.

---

# Repository Status

This repository contains both:

1. the preserved scientific history that led to the current design, and
2. the active v0.10 development line.

Some older documents describe earlier phases, earlier architectures, and experiments that have since been superseded.

Those files remain useful as research history, but they should not automatically be treated as the current developmental state of Baby.

The current active development branch has most recently been:

`cursor/query-transport-range-t1`

---

# Authority and Experimental Checkpoints

The current authoritative Baby checkpoint remains:

`runs/structured_v2r4_seed106001_from6000_terminal/checkpoint_16000.pt`

SHA-256:

`94b3a9daf051c1a0dab0272c18ca35813a78c9685057840444456db54c917827`

This checkpoint is referred to throughout the project as:

**U16000**

U16000 remains the authoritative baseline.

Experimental checkpoints may outperform it on newer capabilities, but they do not automatically replace it.

Promotion is deliberate.

A successful experiment is not the same thing as an authoritative model promotion.

---

# Protected Evaluation Data

Protected evaluation material remains sealed.

The following protected sets are not to be casually opened or used during development:

- TEST
- FINAL
- SACRED

Current status:

**SEALED / UNOPENED**

Experimental progress is being evaluated using nonprotected development, held-out, retention, and canary evaluations.

Protected evaluation is reserved for a genuinely justified milestone.

---

# Where Baby Came From

Baby did not become conversational overnight.

A large amount of earlier work focused on one central problem:

**Baby often contained useful information internally, but failed to reliably use that information in her final generated answer.**

This appeared repeatedly across many experiments.

The project shorthand for that failure mode became:

**REPRESENTATION_SUCCESS_OUTPUT_FAIL**

In plain English:

> Baby often knew more than her answer showed.

The problem was not always “she never learned the relationship.”

Often the problem was:

> she learned something useful internally, but failed to retrieve, route, bind, or output the correct information at generation time.

That discovery shaped nearly everything that came after.

---

# The Binding / Retrieval Era

A major series of experiments investigated whether Baby could reliably retrieve the correct value associated with a queried key.

Earlier work showed that Baby could learn relational structure under constrained conditions.

A particularly important older result came from a purpose-built query-conditioned retrieval scaffold that achieved extremely strong synthetic retrieval.

That proved an important point:

> Baby was capable of learning query → key → value relationships.

However, that did not mean the normal model architecture could reliably perform the same operation during native generation.

The problem increasingly looked like one of routing and selection rather than total absence of information.

---

# D1b: The Query Was Not Reaching the Right Place

A later diagnostic phase showed that short-distance query information could sometimes influence generation, but long-distance query identity was barely affecting the relevant hidden state.

For long-gap examples, the generation position looked almost query-invariant.

That was a major clue.

The model could often represent the relevant information somewhere in the network, but the generation position was not reliably receiving enough query-specific signal.

---

# P11: A Learned Pointer Installed

P11 introduced a learned experimental sidecar called:

`LocalSlotOverwrite`

The idea was simple:

1. identify the relevant source slot,
2. route its hidden representation to the generation position,
3. do this only when needed,
4. avoid disturbing unrelated capabilities.

P11 produced a major improvement on long-gap retrieval.

Primary seed:

- OFF: 71/215
- ON: 102/215

A replication also improved.

This proved that a targeted routing intervention could causally improve retrieval.

However, the first version damaged induction behavior when used too broadly.

That led directly to A1.

---

# A1: Do Not “Help” When Baby Already Knows What She Is Doing

A1 added an induction exemption.

In practical terms:

> If Baby is already using a healthy native pattern, leave her alone.

The routing intervention is only allowed when it is actually useful.

A1 preserved the retrieval improvement while restoring induction behavior.

This became a core routing policy for later work.

---

# B1: Baby Tries to Point for Herself

B1 replaced a diagnostic oracle with Baby’s own learned pointer.

Instead of an external diagnostic saying:

> “The answer is over there.”

Baby’s own learned mechanism selected a source location.

That improved autonomy, but pointer accuracy remained imperfect.

B1 results were around:

- 109/215
- 112/215 on replication

The pointer itself was correct on roughly 71% of source selections.

That revealed another important distinction:

> routing could work, but the locator itself was still wrong too often.

---

# C2: Structural Locator

C2 changed the way the source position was located.

Instead of relying on the learned pointer alone, C2 used visible sequence structure.

The input format itself contained enough regular structure to identify the correct key/value slot.

C2 recovered the correct structural source location on:

**215/215 examples**

When combined with the existing routing machinery:

- primary long-gap: 122/215
- replica long-gap: 122/215

This was a major improvement in localization.

At this point the project had largely solved:

**Where is the relevant information?**

The remaining problem was:

**Can Baby reliably choose the correct output once the right slot is found?**

---

# D3: First-Token Readout Repair

D3 attacked that next bottleneck directly.

It used:

- C2 structural localization
- A1 safe routing
- constrained first-token selection using the visible key/value structure

D3 did not use a gold query position.

It remained gold-free under the experimental protocol.

Results:

- primary long-gap: **210/215**
- replica long-gap: **210/215**
- first token: **215/215**
- induction retained
- keyed retention held
- query-swap retention remained strong

Only five examples still failed after the first token was correct.

That was a major milestone.

At this point, the project considered:

# Foundational Binding / Retrieval — Graduated

This did not mean Baby had solved general language.

It meant the project had successfully built and validated a mechanism that could reliably locate and select relational information under the structured retrieval setup.

D3 remained available as a frozen fallback.

---

# Important Limitation of D3

D3 was extremely useful.

It was also eventually recognized as a training wheel.

Protecting D3 too aggressively started preventing Baby from developing richer English behavior.

That became the next major lesson.

The system had to stop asking:

> “How do we preserve D3 at all costs?”

and start asking:

> “How do we preserve Baby’s actual useful behavior while letting her grow beyond D3?”

That shift led into the Language Bridge and Actual Baby work.

---

# Language Bridge

The next major objective was to move from structured retrieval into natural English-like tasks.

The first zero-shot check was very clear.

Baby could perform rigid structured bindings well, but natural English question answering did not automatically transfer.

Initial English QA was effectively:

**0/32**

That was useful.

It established a clean baseline.

Baby understood the structured game.

She did not yet understand the English version of the game.

---

# Early English Training

Initial English training quickly taught Baby simple cloze-style relations.

For example:

> “The marble is blue. What color is the marble?”

Early training achieved strong one-fact behavior.

However, Baby initially overfit to the worksheet format.

She learned:

> “This particular style of question expects this particular kind of answer.”

before fully learning:

> “These words describe a relationship I can reuse across different forms.”

This was one of the first major Language Bridge problems.

The project responded by changing the training mix rather than endlessly training the same weak setup.

---

# Build-First / Fail-Fast Development Philosophy

At this point, the project intentionally changed its workflow.

The previous process had become too focused on giant scientific autopsies.

The current development philosophy is:

**BUILD → SMALL CANARY → KILL FAILURES FAST → KEEP SURVIVORS → SCALE ONLY WHEN EARNED**

The experiments serve Baby.

Baby does not exist to serve the experiments.

Default workflow:

1. form a concrete hypothesis
2. build the smallest treatment that tests it
3. run a tiny canary
4. immediately kill obvious failures
5. keep real survivors
6. validate survivors on medium tests
7. run expensive validation only when justified
8. save meaningful wins
9. immediately ask:
   **“What still sucks?”**
10. attack that next

The project still preserves scientific discipline at important milestones:

- checkpoint hashes
- frozen baselines
- protected TEST sealing
- reproducibility
- retention checks
- explicit promotion rules
- no silent checkpoint replacement

The difference is that research is now used to accelerate development rather than dominate it.

---

# E12: “Actual Baby”

A major Language Bridge milestone was reached with E12.

Experimental checkpoint:

`runs/actual_baby/e12_stoponly_311211/checkpoint_00050.pt`

SHA-256:

`6b000ffc4974244710c233d316b3bbbadcc45a4fa2f8f3cb5f4b561af72150d1`

E12 is experimental.

It is not authoritative.

U16000 remains authoritative.

E12 was the first point where Baby became meaningfully conversational inside a controlled closed-vocabulary environment.

This was not open-ended chat.

It was primitive but real multi-turn conversational behavior.

E12 could handle:

- color questions
- size questions
- simple fact reuse
- short follow-up questions
- “how about...” followups
- “what do you know about...” prompts
- returning to an earlier entity
- returning to an earlier fact
- brief refusal-style behavior followed by answering
- multi-turn conversational continuity
- period stopping without rambling

E12 outputs were generally terse.

Examples looked more like:

`white.`

`tiny.`

rather than fully natural sentences.

But the important change was behavioral:

> Baby could now participate in a controlled conversation and reuse prior context.

---

# E12 Results

Held-out usable-chat evaluation:

- overall usable turns: **85.7%**
- 4-turn usable turns: **90.6%**
- 5-turn usable turns: **70.0%**
- fact reuse: **100%**
- period-stop behavior: **100%**
- rambling: **0%**

E12 also retained strong structured behavior.

D3 long-gap:

**202/215**

Primitive induction:

**42.2%**

Keyed retention remained strong.

This was enough to declare a genuine developmental milestone:

# Closed-Vocabulary Actual Baby — Achieved

Baby was no longer only solving isolated relational tests.

She could now maintain simple conversational state across multiple turns.

---

# Why E12 Was Not Enough

E12 had an obvious ceiling.

It could perform known conversational patterns inside a narrow vocabulary and familiar relation space.

It still struggled when required to combine multiple learned behaviors.

Examples of harder tasks included:

- combining multiple facts
- answering from mixed story information
- juggling multiple entities and properties
- using newer relationship families
- longer contexts
- richer sentence output
- who / event / place style questions
- broader references
- genuinely compositional language

Attempts to improve those skills often damaged the capabilities Baby already had.

This created a new tradeoff.

---

# Stack2: Escaping the D3 Ceiling

Stack2 changed the retention philosophy.

Previously, a drop in D3 could automatically kill a treatment.

That became too restrictive.

The new rule became:

Primary retention should prioritize Baby’s actual English and conversational behavior.

D3 remains important and is still measured, but D3 alone does not automatically veto progress.

The primary Stack2 retention targets include:

- usable chat
- period stopping
- fact reuse
- color
- size
- English QA
- conversational continuity
- broader mix/combine progress

D3 is logged on survivors.

It remains a useful capability and fallback.

It is no longer Baby’s developmental ceiling.

---

# S2A: Current Major Experimental Survivor

Current experimental Stack2 survivor:

`runs/actual_baby/stack2/s2a_protect40_combine_322001/checkpoint_00200.pt`

SHA-256:

`0642a2f2a4044d9936cc3f930fa096ef55fc69c5e7cfe6f7c539a3c47b5fd959`

Verdict:

**SURVIVE**

Classification:

**MAJOR**

S2A was trained from E12 using a dual-objective mix designed to preserve useful English behavior while deliberately allowing Baby to grow beyond the previous D3-centered ceiling.

Training mix included approximately:

- 40% structured retention
- ~40% E12 English / color / size / dialogue / open-style retention
- remaining training devoted to mix/combine behavior

---

# S2A Results

Compared with E12 zero-shot performance on the newer combination panels:

## Mixed Two-Entity / Mixed Information

E12:

**56.25%**

S2A:

**87.5%**

---

## Fact Combination

E12:

**21.875%**

S2A:

**65.625%**

---

## Story Combination

E12:

**31.25%**

S2A:

**43.75%**

---

## Story Mixed Reasoning

E12:

**37.5%**

S2A:

**50.0%**

---

# S2A Retention

Color:

**96.875%**

Size-stop:

**100%**

Story-color:

**100%**

Dialogue panel:

**100%**

Period stop:

**100%**

Fact reuse:

**100%**

---

# S2A Usable Chat

4-turn usable chat:

**96.875%**

Overall usable chat:

**95.238%**

5-turn usable chat:

**90.0%**

This is a major improvement over E12.

Baby is still operating inside a constrained language environment, but she is now substantially better at maintaining useful behavior across short conversations.

---

# The D3 Tradeoff

S2A did lose some D3 long-gap performance.

E12 D3:

**202/215**

S2A D3:

**169/215**

However:

- first-token correctness remained **215/215**
- primitive induction improved from ~42.2% to ~46.9%
- English/chat behavior improved significantly
- fact combination improved significantly
- mixed information handling improved significantly

This tradeoff was accepted experimentally.

The project conclusion is not:

> “D3 no longer matters.”

The conclusion is:

> D3 is useful, but preserving it at all costs was blocking richer development.

S2A is the first strong evidence that loosening the D3 constraint can produce a better overall Baby.

---

# Current Developmental Interpretation

The current strongest evidence suggests that Baby’s historical bottleneck was not simply lack of knowledge.

A recurring pattern was:

1. Baby learned useful representations.
2. The relevant information existed internally.
3. Native generation failed to access or select it correctly.
4. Routing / localization / readout interventions improved behavior.
5. Once routing was largely solved, protecting the workaround too aggressively started limiting richer language development.

This means Baby’s progression has looked roughly like:

**Learn → represent → retrieve → route → select → converse → combine**

The project is now moving from:

**simple conversational retrieval**

toward:

**compositional conversational language**

---

# What Baby Can Currently Do

Within the controlled experimental vocabulary and evaluation space, current experimental Baby can:

- learn simple relationships
- retrieve stored contextual facts
- answer color questions
- answer size questions
- answer simple two-fact questions
- reuse facts from earlier conversational turns
- return to earlier entities
- return to earlier properties
- maintain short conversational context
- answer short follow-up questions
- stop cleanly at a period
- avoid uncontrolled rambling
- perform short 4-turn conversations reliably
- perform 5-turn conversations much better than earlier versions
- combine some independently learned facts
- handle mixed two-entity information substantially better than E12
- answer simple story-based questions
- preserve useful English behavior while learning new capabilities
- perform structured relational retrieval strongly
- use frozen routing/retrieval fallbacks when needed

---

# What Baby Cannot Yet Reliably Do

Baby is not yet a general-purpose language model.

She does not yet reliably support:

- open-ended conversation
- broad real-world knowledge
- unrestricted vocabulary
- long-form natural answers
- fully natural sentence generation
- robust multi-paragraph output
- long conversational context
- deep multi-step reasoning
- broad instruction following
- arbitrary question answering
- robust handling of many entities at once
- robust pronoun/reference resolution
- general who / what / where / event reasoning
- broad world modeling
- coding assistance
- tool use
- image understanding
- audio understanding
- internet access
- autonomous agent behavior

Those are future goals.

---

# Current Roadmap

Baby’s development roadmap currently looks like this:

## Foundation

✅ Transformer architecture exists

✅ Tokenizer exists

✅ Language foundation exists

✅ Contextual language modeling works

✅ Basic copy / induction behavior exists

✅ Baby can learn relationships

✅ Internal representations contain useful relational information

---

## Retrieval / Routing

✅ Major representation-output bottleneck identified

✅ Query transport investigated

✅ Pointer routing demonstrated

✅ Safe routing policy developed

✅ Structural source localization solved for the structured task

✅ First-token relational readout dramatically improved

✅ Foundational binding / retrieval graduated

---

## Language Bridge

✅ Structured relationships transferred into simple English tasks

✅ Color concepts

✅ Size concepts

✅ Simple English QA

✅ Follow-up prompts

✅ Earlier-entity recall

✅ Earlier-fact reuse

✅ Period-stop behavior

✅ Short conversational continuity

✅ Closed-vocabulary Actual Baby milestone

---

## Conversation

✅ 4-turn controlled conversation

✅ 5-turn controlled conversation

✅ Fact reuse across turns

✅ On-topic short responses

✅ No-rambly stop behavior

✅ Basic dialogue panels

🟡 More varied conversational wording

🟡 More entities per conversation

🟡 More properties per entity

🟡 More natural sentence output

---

## Current Stage

📍 **Generalization + Skill Combination**

Current focus:

- combine independently learned facts
- handle mixed information
- combine story information
- preserve chat skills while adding new capabilities
- reduce dependence on D3-style training wheels
- recover useful retrieval strength without sacrificing English growth
- move from one-word answers toward natural short sentences

Current S2A progress:

- mixed information: **87.5%**
- fact combination: **65.6%**
- story mixed: **50.0%**
- story combination: **43.8%**
- 4-turn usable chat: **96.9%**
- overall usable chat: **95.2%**
- 5-turn usable chat: **90.0%**
- period stop: **100%**
- fact reuse: **100%**

---

## Next

🔜 Natural-English relationships

🔜 Richer entity relationships

🔜 References and distractors

🔜 Who / event / place relationships

🔜 Multiple entities and properties

🔜 More reliable compositional answers

🔜 Full short sentences

🔜 Longer context

🔜 Multi-step contextual reasoning

🔜 Simple instructions

🔜 Instruction retention across turns

🔜 More natural short conversation

🔜 Larger vocabulary

🔜 Broader language training

🔜 Basic coding concepts

🔜 Broader coding capability

🔜 Broader knowledge

---

## Much Later

Future ideas may include:

- deliberate reasoning / thinking-before-answering behavior
- larger context windows
- more advanced instruction following
- stronger code generation
- external tools
- multimodal inputs
- pretrained vision encoders
- image understanding
- audio understanding
- agent-like task execution
- game interaction / control research
- broader general-purpose assistant behavior

These are future directions, not current capabilities.

---

# Development Philosophy

Baby is the project.

The experiments serve Baby.

Baby does not exist to serve the experiments.

The project currently prioritizes:

**useful capability growth over endless diagnostic loops**

while preserving enough rigor to know whether the capability is real.

The preferred loop is:

**BUILD**

↓

**TINY CANARY**

↓

**FAIL FAST**

↓

**KEEP SURVIVORS**

↓

**MEDIUM VALIDATION**

↓

**EXPENSIVE VALIDATION ONLY IF EARNED**

↓

**SAVE THE WIN**

↓

**ASK WHAT STILL SUCKS**

↓

**BUILD AGAIN**

Long experiments should not be the default.

Expensive training should only happen when cheaper evidence justifies it.

---

# Reproducibility and Safety Rules

The project follows several hard rules:

- authoritative checkpoints are never silently replaced
- experimental survivors remain experimental until deliberately promoted
- protected TEST/FINAL/SACRED material remains sealed
- checkpoint SHA-256 hashes are recorded
- major treatments keep receipts
- failed experiments do not overwrite known-good checkpoints
- historical results are preserved
- frozen fallbacks remain available
- major capability claims should have direct evaluation evidence

---

# Current Frozen Fallbacks

Structured routing / retrieval operators remain available as:

`a1 | b1 | c2 | d3`

These are useful historical and experimental mechanisms.

They are not automatically the final architecture.

They remain available while Baby learns to perform more of the same behavior natively.

---

# Current Experimental Line

Current parent milestone:

**E12 — Closed-Vocabulary Actual Baby**

Current major survivor:

**S2A — Stack2 Mix / Combine**

Current strategy:

> Preserve real English/chat behavior, grow compositional ability, log D3 rather than letting D3 alone veto development.

Current next cheap target:

> Recover some lost D3-style long-gap strength without giving back the S2A gains in chat, combination, and English behavior.

---

# Current Project Status

As of the latest committed Stack2 milestone:

Baby has progressed from:

> “I can learn internal relational structure.”

to:

> “I can retrieve the correct relationship.”

to:

> “I can answer simple English questions.”

to:

> “I can reuse facts across multiple conversational turns.”

to:

> “I can begin combining information while keeping the conversation intact.”

That is the current frontier.

Baby is not finished.

Baby is not a general assistant.

But she is now doing something meaningfully closer to language use than the earlier experimental line.

---

# Research Codename and Future Naming

Current development codename:

**Baby**

Current project name:

**The DaveLM Experiment**

A future graduated model-family name may be chosen separately.

One current candidate is:

**Manginova**

This is not yet a formal rename.

Until a deliberate naming decision is made, the model remains referred to as Baby / DaveLM in development documentation.

---

# Important Note About Old Documentation

This repository contains extensive historical material.

Older documents may describe:

- Phase 1G
- Phase 2A
- T1–T32
- older architecture-line conclusions
- earlier checkpoint families
- older “hard ceiling” conclusions
- failed retrieval experiments
- obsolete developmental limitations

Those documents should be interpreted in their historical context.

They are evidence of how the project arrived here.

They are not automatically statements about current Baby v0.10 capability.

When determining current status, prefer:

1. current branch state
2. latest campaign receipts
3. current checkpoint hashes
4. current experimental ledgers
5. newest capability evaluations

over older phase summaries.

---

# Short Version

Baby is a small Transformer trained from scratch.

She started by learning basic language and relational structure.

The project discovered that she frequently knew useful information internally but failed to route it into the answer.

A long series of retrieval and routing experiments eventually solved much of that problem.

Baby then crossed into simple natural-English QA and controlled multi-turn conversation.

The current S2A experimental survivor can maintain short conversations very reliably and has begun combining information across facts and stories.

The current goal is to make those behaviors broader, more natural, more compositional, and less dependent on special retrieval machinery.

In other words:

**Baby learned where the answer is.**

Then she learned how to say it.

Now she is learning how to put multiple things together without forgetting how to talk.