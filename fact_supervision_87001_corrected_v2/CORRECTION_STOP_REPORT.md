# Corrected pre-training build stop

This separately versioned attempt preserves the original
`fact_supervision_87001` failure and does not modify it.

The authorized correction required eight semantic families per holdout, with two
unique assignment/order-zero naturalistic prompts per family, and zero Primary /
Confirmation prompt intersection. The already-frozen semantic family pools cannot
satisfy all three conditions simultaneously.

An exhaustive deterministic check over the 12 Primary object families found 47
valid eight-family subsets with 16 unique prompts. The corresponding Confirmation
pool had 285 valid subsets. There was no pair of valid subsets with disjoint prompt
sets (`joint_valid_pairs = 0`). Therefore no outcome-blind selection rule can produce
the required corrected battery without changing a frozen semantic pool, changing
the naturalistic rendering, or relaxing the explicit zero-intersection invariant.

This is a scientific/design invariant failure, so no choice was made.

The corrected selector source and this report are retained as a failed attempt only.
No datasets, schedules, checksums, or frozen artifacts were declared. No checkpoint
was loaded; no inference, training, optimizer update, or sacred access occurred.

Existing failed construction, stop receipt, and historical v3d erratum remain intact.
The additive v3d historical-status erratum is copied below unchanged in substance;
it remains historical documentation and does not authorize execution.

Required human decision: choose whether to revise the naturalistic selection/rendering
rule or relax the cross-holdout zero-intersection requirement. Both would be scientific
changes, so this implementation stops here.
