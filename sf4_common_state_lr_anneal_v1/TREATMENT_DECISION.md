# Prospective SF4 decision — one common-state matched study

Chosen: compare constant late English LR with linear-to-zero late English LR from the same freshly created
update100 state, including identical optimizer and RNG. This is the smallest controllable test of SF2's
near-boundary offset that leaves its successful retention intervention intact. Evidence justifies testing it;
it does not establish that it will work.

Ranked alternatives, in descending priority:

1. **Late English LR schedule: chosen.** Baby has acquired strong conditional margins on most TRAIN records;
   the remaining miss is only -0.030 nats. A late step-size manipulation tests whether the fixed-budget
   acquisition/retention balance improves with less late adaptation. Linear-to-zero has no curve/floor tuning;
   it is the previously specified but untested SF3 treatment, now with a valid prospective control.
2. **Head constraint:** factual offset is upstream-led and the head is a marginal tipping contributor. Head
   freezing could interfere with correct answer membership and changes where adaptation can occur.
3. **Pairwise/contrastive margin objective:** creates a new objective/support tradeoff; the residual near-tie
   does not itself establish that a margin term is needed. Parent KL already enabled substantial discrimination.
4. **Additional constant-rate CE:** more budget is another valid future question, but it does not isolate
   whether the late boundary shift is helped by a smaller step. It requires extra updates compared with SF2.
5. **Ordinary-language replay:** adds a data mixture to address retention already passing under SF2; poor
   targeting of the remaining factual near-tie.
6. **Modified/continued KL:** continuing unchanged is the control; changing coefficient or support would
   perturb a successful retention mechanism with little direct residual evidence.
7. **Staged KL:** unsupported timing/strength choices and a new objective schedule; not the smallest test.
8. **Architecture/tokenizer changes:** no Baby-specific evidence makes them necessary.

Important corrections to overstrong narrative: two checkpoints cannot establish a monotonic trajectory or
oscillation. 'Overshoot' and 'settling' remain hypotheses. Full 16-item symmetry rules out unequal item exposure;
it does not uniquely identify the optimizer dynamics. The autopsy's replay equality is measured; its particular
kernel/driver cause remains unidentified. SF4 makes no stronger claim about that cause.

The new control is essential. Historical SF2 has no recoverable update100 optimizer/RNG state. Current A/B
replays are mutually exact but differ from historical SF2. SF4 generates two fresh common100 states, verifies
model+optimizer+RNG+controller equality, and branches from the predesignated first one. No favorable common
checkpoint or seed is selected. The historical SF2/SF3 classifications remain frozen.

This schedule changes integrated English step size, LR-scaled weight decay, and English adaptation relative to
fixed binding updates. That is one LR intervention, but an effect would not uniquely prove a smoothing mechanism.
Both arms keep the exact same CE+KL objective, batches, scopes, seeds, optimizer moments and binding LR.

Endpoints are repeated TRAIN acquisition, retention, and legally gated transfer only. This is not a fresh
held-out test of the annealing idea: design is informed by SF2 TRAIN outcomes, openly and prospectively. Locked
panels are not inspected for design. Reference-model anatomy is background and did not determine this choice.
