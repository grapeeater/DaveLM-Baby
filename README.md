# Baby: A Language Model Built from Scratch

**Baby** is a ~61.5M-parameter Transformer language model trained **entirely from scratch** (not a fine-tune). This repository documents a rigorous experimental journey into teaching a model to combine language understanding with factual reasoning about entities and their relationships.

## Quick Summary

**What Baby is:** A proof-of-concept language model exploring how to build representation learning and relational reasoning into a small, transparent architecture.

**Current architecture:** 12-layer Transformer, 640-d model, 10 attention heads, 1,024-token vocabulary, trained from random initialization.

**What changed:** Baby moved from pure language modeling (Phase 1G) to multi-task learning combining language + pointer-based entity retrieval (Phase 2A). The Phase 2A research closed after 32 systematic treatments (T1–T32) when architectural constraints were reached.

**Current status:** Phase 2A is scientifically terminal. A new v0.10 lineage is now active, exploring a fresh training path with capability-first objectives.

---

## The Journey

### Phase 1G: Foundation (Language Learning)
Baby's foundation is a 61.5M parameter model trained on TinyStories to fluent language understanding:
- **Checkpoint:** `baby_vnext_phase1g_language_v1_run_seed610001/checkpoints/best.pt`
- **Final performance:** DEV cross-entropy 1.204 nats (U6000 frozen)
- **Status:** Authoritative parent for all Phase 2A work; archived and locked

Phase 1G established that Baby could learn coherent English from scratch and maintain language fluency through 6,000 training updates.

### Phase 2A: Representation + Reasoning (2026-09-13 to 2026-09-14)
Phase 2A attempted to teach Baby a second task: answering factual questions about characters in stories while preserving language ability.

**Mechanism:**
- **Binding/Retrieval:** An orthogonal two-slot localizer identifies entity mentions in context; a frozen normalized readout of blocks 7–11 can decode correct continuations 66.6% of the time under gold prefixes
- **Pointer-weighted retrieval:** Query position attention over candidate entity spans produces a pointer distribution; retrieved entity representation injects into the answer position
- **Native generation goal:** After pointer selection, Baby should generate the correct answer autoregressively (greedy token sampling)

**What worked:**
- ✓ Relational representation formation: pointer retrieval 110–120/128, reversals 48–56/64, family disjointness intact across 2–3/3 seeds (T18–T24)
- ✓ Forced-choice ranking: correct answer ranked highest 87–91/128 (native forced-choice gates)
- ✓ Language preservation: DEV cross-entropy held at 1.25–1.28, gap ≤0.5 across all 32 treatments
- ✓ TRAIN16 (in-training acquisition): 16/16 exact + forced-choice, reversals held

**What failed:**
- ✗ Native exact generation remained stuck at **27–40/128** (21–31% success) across T18–T32 despite 32 different architectural and loss-function interventions
- ✗ Shared-prefix names (Sal/Skye, Omar/Opal): **1–5/64** exact (~3%) vs unique-first-token **33–36/64** (~53%)
- ✗ The critical failure mode was **first-token-correct-then-diverge**: Baby would select the correct answer's first token 60–87% of the time but then generate divergent continuations ("Skye" → "Sky", "Sal" → "Salt", "Ava" → "York")

**Key diagnostic result (Prefix-Rescue Experiment):**
When given the correct first answer token via oracle intervention, then allowed to generate freely, Baby recovered exact answers **108/124 times (87.1%)**. Controls (wrong-matched, prefix-compatible-wrong, no-intervention) all scored 0%. This proved the problem was not missing information or representation—it was **autoregressive state divergence after early-token errors**.

**Phase 2A treatments:** T18 (answer-span CE), T19–T22 (loss variants), T23 (in-row unlikelihood), T24 (inventory-wide unlikelihood), T25 (diagnostic), T28 (suffix margin), T29 (fork disambiguation), T30 (persistent routing), T31 (explicit source-copy bridge), T32 (ordered source-memory).

**Scientific conclusion:** All feasible modifications to the vNext architecture within the current protocol exhausted the improvement frontier. The bottleneck is **prefix-conditioned state divergence**: once a wrong token enters the sequence, the Transformer's hidden states recompute based on the corrupted prefix, and correct suffixes become unreachable even when the original entity information is preserved. No amount of training signal, routing, or output-layer intervention can overcome this architectural constraint.

---

## Phase 2A Closed Doors

These are proven closed; do not repeat:
- Loss-function engineering (CE weighting, suffix weighting, unlikelihood penalties, margin losses)
- Tokenizer replacement (Option B/C: no improvement)
- Training specific tokens (T29 fork objective)
- Persistent pointer routing (T30)
- Explicit source-copy bridges (T31)
- Ordered source-memory in final layer (T32)

---

## v0.10: A Fresh Start

As of **2026-09-15**, a new lineage `v0.10` is active on branch `v0.10`.

**Key differences from v0.9:**
- Initialized from **random weights** (not continuing from T34 or Phase 2A checkpoints)
- **Capability-first training** objectives: ordinary language modeling, contextual reproduction, induction, and copy across held-out surface realizations
- Success bars are **not** based on memorization or familiar-example exact match
- Workspace: `C:\DaveLM-v0.10` (writable development directory)

**v0.10 status:** Foundation protocol frozen and locked. Active development on design and implementation. Authoritative protocol: `data_specs/V010_FOUNDATION_PROTOCOL.json`.

---

## What Baby Can Do

**Reliably (Phase 1G + Phase 2A proven):**
- Generate coherent, grammatically correct English from prompts
- Understand contextual relationships in short narratives
- Identify and rank entity mentions by relevance (pointer mechanism)
- Learn task-specific retrieval/binding within-corpus (TRAIN16 100% success)
- Preserve language ability while acquiring new task signal (language CE held across Phase 2A)
- Represent relationships between characters correctly (family relationships, first-mention, last-mention, etc.)

**With difficulty or not at all:**
- ✗ Generate factually correct complete answers autoregressively when first token is ambiguous (native exact ~27–40%)
- ✗ Recover from early-token errors without external intervention
- ✗ Generalize entity binding to held-out test examples (TEST sealed; no evidence of test-set success)
- ✗ Open-ended conversation or multi-turn dialogue
- ✗ General reasoning, arithmetic, or world knowledge
- ✗ Handle longer or more complex narratives (trained on short TinyStories)

---

## Protected Data

**TEST, FINAL, SACRED remain sealed and unopened.**

All Phase 2A treatments were conducted on DEV panels (128 examples) with TEST (320 examples) locked in provenance. No checkpoint has earned authorization to open TEST under the frozen protocol.

- TEST set: `phase2a_t3_rebuilt_study_v1/data/qa_test.jsonl` (SHA-256: `6a36a8a482a6812e36ff24f5ce2991c1d3ffeb78c0a7e5a5ddefaf1f5da773cb`)
- Status: `SEALED_UNOPENED`

---

## Authoritative Artifacts

**Phase 2A (read-only, no longer active):**
- Parent checkpoint: `baby_vnext_phase1g_language_v1_run_seed610001/checkpoints/best.pt`
- Architecture blueprint: `baby_vnext_60m_design_v1/BABY_VNEXT_ARCHITECTURE_SPEC.md`
- Corpus (T3): `phase2a_t3_rebuilt_study_v1/`
- Terminal state: [`CANONICAL_STATE.md`](CANONICAL_STATE.md) (updated 2026-09-14)
- Current handoff: [`PHASE2A_CURRENT_HANDOFF.md`](PHASE2A_CURRENT_HANDOFF.md)

**v0.10 (active development):**
- Protocol: `data_specs/V010_FOUNDATION_PROTOCOL.json`
- Design: `design/`
- Implementation: `src/baby_v010/`

**Experimental branches (development agents):**
- `codex/selection-repair-s1` — selection mechanism repairs
- `cursor/query-transport-range-t1` — query identity routing experiments
- `cursor/selection-repair-s2-query-contrast` — selection diagnostic contrasts

---

## Repository Structure

**What Git tracks:**
- Protocol documents, experiment ledgers, and scientific reports
- Architecture specifications and training blueprints
- Tokenizer v0_7 (frozen, vendored copy)
- Experimental design documentation
- Phase 2A diagnostics and audit trails

**What Git does NOT track (local-only):**
- Checkpoints (`*.pt` files)
- Training runs and rollout directories (`*_run_seed*`)
- Runtime environments and caches
- Protected data (qa_test.jsonl, etc.)

---

## Key References

For detailed technical understanding:
- **Architecture:** `baby_vnext_60m_design_v1/BABY_VNEXT_ARCHITECTURE_SPEC.md`
- **Phase 2A final closure:** `CANONICAL_STATE.md`
- **Experimental methodology:** `AGENT_INSTRUCTIONS.md`
- **Phase 1G protocol:** `baby_vnext_phase1g_language_v1/PHASE1G_LANGUAGE_PROTOCOL.md`
- **Phase 2A corpus & gates:** `phase2a_t3_rebuilt_study_v1/`

---

## Contribution Guidelines

**Do not:**
- Relaunch or reweight closed treatments (T4–T32 are adjudicated)
- Lower representation/language gates without explicit new evidence
- Open TEST/FINAL/SACRED data
- Retokenize or copy external model weights
- Modify Phase 2A training protocols without owner authorization

**Current policy:** Phase 2A is read-only. v0.10 is the active development lineage.

---

## License

Research repository. Intended for academic/research documentation. Checkpoint weights are local-only and not distributed.

---

**Last updated:** 2026-09-17  
**Repository:** https://github.com/grapeeater/DaveLM-Baby  
**Status:** Phase 2A closed (2026-09-14). v0.10 in active development (2026-09-15–present).
