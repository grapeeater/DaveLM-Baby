# SF7 — pairwise first-answer-token margin hinge

**INCONCLUSIVE_EARLY_STOP**. Three preregistered same-seed control/treatment pairs; six independent Pilot1 starts.

The only training-variable difference is one added loss term (margin hinge, M=1.0 nat, lambda_margin=1.0) on the first-answer-token pairwise log-prob gap in the treatment arm. English LR is constant 5e-5 in BOTH arms (no annealing). Binding, data, ordering, KL, scope and evaluation are unchanged from SF2/SF6. Historical SF1-SF6 classifications remain unchanged.

| Seed | Arm | Update | Correct /16 | Exact /16 | Reversals /8 | Families /4 | CE | PPL | D3 | All gates |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 87014 | control | 200 | 14 | 14 | 6 | 2 | 3.596458 | 36.469 | 0.00723283 | ACQUISITION_FAIL |
| 87014 | treatment | 100 | 16 | 12 | 8 | 4 | 3.601611 | 36.657 | 0.01211006 | STOP_REGRESSION |
| 87015 | control | 200 | 15 | 15 | 7 | 3 | 3.574815 | 35.688 | 0.00676115 | ACQUISITION_FAIL |
| 87015 | treatment | 100 | 16 | 15 | 8 | 4 | 3.646187 | 38.328 | 0.01230608 | STOP_REGRESSION |
| 87016 | control | 200 | 14 | 15 | 6 | 2 | 3.581418 | 35.924 | 0.00728491 | ACQUISITION_FAIL |
| 87016 | treatment | 200 | 16 | 16 | 8 | 4 | 3.585262 | 36.063 | 0.00999590 | ACQUISITION_SUCCESS |

## Paired evidence

Control successes: **0/3**. Treatment successes: **1/3**. Treatment-only: **1/3**. Control-only: **0/3**.

- Seed 87014: neither. Mechanism comparison: `{"100": {"control_residual_gap": 0.2913188934326172, "treatment_residual_gap": 2.7693217992782593, "control_min_g0_a0": 0.2913188934326172, "treatment_min_g0_a0": 2.7165104150772095, "control_mean_g0_a0": 0.46634888648986816, "treatment_mean_g0_a0": 2.747263550758362, "control_mean_g0_a1": -0.27023911476135254, "treatment_mean_g0_a1": 2.10602468252182, "control_mean_g1": 0.5185623317956924, "treatment_mean_g1": 3.3541394248604774}}`
- Seed 87015: neither. Mechanism comparison: `{"100": {"control_residual_gap": 0.2801170349121094, "treatment_residual_gap": 2.548832893371582, "control_min_g0_a0": 0.2801170349121094, "treatment_min_g0_a0": 2.548832893371582, "control_mean_g0_a0": 0.41353273391723633, "treatment_mean_g0_a0": 2.610117942094803, "control_mean_g0_a1": -0.22326326370239258, "treatment_mean_g0_a1": 3.0765443295240402, "control_mean_g1": 0.5181458741426468, "treatment_mean_g1": 3.468035601079464}}`
- Seed 87016: treatment_only. Mechanism comparison: `{"100": {"control_residual_gap": 0.2640514373779297, "treatment_residual_gap": 2.3184263706207275, "control_min_g0_a0": 0.2640514373779297, "treatment_min_g0_a0": 2.1787248849868774, "control_mean_g0_a0": 0.42276668548583984, "treatment_mean_g0_a0": 2.230715900659561, "control_mean_g0_a1": -0.2635207176208496, "treatment_mean_g0_a1": 2.535685047507286, "control_mean_g1": 0.4833311289548874, "treatment_mean_g1": 3.5838694125413895}, "200": {"control_residual_gap": -0.11556720733642578, "treatment_residual_gap": 4.241123378276825, "control_min_g0_a0": -0.11556720733642578, "treatment_min_g0_a0": 4.1056119203567505, "control_mean_g0_a0": 0.003910064697265625, "treatment_mean_g0_a0": 4.262087315320969, "control_mean_g0_a1": 0.5045096725225449, "treatment_mean_g0_a1": 5.6163352355360985, "control_mean_g1": 2.0736675187945366, "treatment_mean_g1": 4.933253463357687}}`

## Mechanism (descriptive only; never a gate)

- seed_87014_control @200: residual(wooden boat/Alex) gap=-0.1300, min g0:a0=-0.1300, mean g0:a0=-0.0302, mean g0:a1=0.5850, mean g1=2.1237
  @100: residual gap=0.2913, min g0:a0=0.2913
  @100: residual gap=2.7693, min g0:a0=2.7165
- seed_87015_control @200: residual(wooden boat/Alex) gap=-0.0325, min g0:a0=-0.0325, mean g0:a0=0.0717, mean g0:a1=0.5652, mean g1=2.3385
  @100: residual gap=0.2801, min g0:a0=0.2801
  @100: residual gap=2.5488, min g0:a0=2.5488
- seed_87016_control @200: residual(wooden boat/Alex) gap=-0.1156, min g0:a0=-0.1156, mean g0:a0=0.0039, mean g0:a1=0.5045, mean g1=2.0737
  @100: residual gap=0.2641, min g0:a0=0.2641
- seed_87016_treatment @200: residual(wooden boat/Alex) gap=4.2411, min g0:a0=4.1056, mean g0:a0=4.2621, mean g0:a1=5.6163, mean g1=4.9333
  @100: residual gap=2.3184, min g0:a0=2.1787

Transfer/copy/competing-name/FINAL/sacred panels remain locked and unscored.

Dave-coded: this run tests whether pushing the Alex-vs-Owen decision apart during training keeps the wooden-boat item off the knife edge, without touching anything else Baby learned.

Next action: review this completed study before any transfer evaluation or further treatment. No second experiment was started.
