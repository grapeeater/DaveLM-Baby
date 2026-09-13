# Non-source localization suppression — zero-update preflight

This preflight freezes the successful T13 architecture, pools, schedule, seed,
and optimizer regime. The sole prospective objective change is
`L_filler=0.5*(max_{j notin {s0,s1}} A0[j]+max_{j notin {s0,s1}} A1[j])`, added
to the unchanged permutation-invariant localization loss
`L_existing=-max(log A0[s0]+log A1[s1],log A0[s1]+log A1[s0])`.
`lambda_existing=1.0536573711078283`; `lambda_filler` is calibrated once as
`0.10*G_existing/G_filler_raw` over localizer parameters on the first training
batch. No retention behavior is used and no optimizer is created or stepped.
The forward contract remains `(input_ids,qpos,anspos)` only; true coordinates
are labels outside forward and are used only by the losses. The starting model
is the successful-treatment starting T13 checkpoint, not its final checkpoint.
