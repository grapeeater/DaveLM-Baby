# T13 distinct two-source localization supervision — seed 8380

This isolated treatment changes only the training objective: a permutation-invariant loss supervises the set of the two true mapping-source positions. T13 architecture, data, optimizer, and schedule remain fixed.

For attention columns A0/A1 and true positions s0/s1, localization loss is the minimum of `-log A0[s0]-log A1[s1]` and `-log A0[s1]-log A1[s0]`, averaged per document. It is permutation invariant and does not impose a permanent slot identity.

Total loss is `L_answer + lambda * L_localization`. Lambda is frozen before training as initial mean `L_answer / L_localization` on the first training batch, with zero optimizer updates. True coordinates are labels outside `forward()` only.

Training is seed 8380, the existing T13 1,000-step schedule, batch 32, AdamW (3e-4, weight decay 0.05, clip 2.0), exactly 1,000 updates. Retention is evaluated once after step 1000 under `model.eval()` and `torch.no_grad()`.
