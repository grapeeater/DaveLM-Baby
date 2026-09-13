# Stage 1 v3d evaluation

One authorized execution across six checkpoints and two separate nonsacred binding DEV pools.

## English controlled results

### graduate
|section|n|correct|ties|mean margin|
|---|---:|---:|---:|---:|
|near_distribution|96|48|0|4.53695e-05|
|counterfactual|48|24|0|-0.000763376|
|surface_form|48|24|0|-9.69637e-05|
|distractor|64|32|0|2.65107e-05|
|query_only_prior|16|0|0|0|

### pilot0
|section|n|correct|ties|mean margin|
|---|---:|---:|---:|---:|
|near_distribution|96|47|0|-0.000824273|
|counterfactual|48|23|0|-0.00500407|
|surface_form|48|24|0|-0.00195037|
|distractor|64|33|0|0.00160733|
|query_only_prior|16|0|0|0|

### pilot1
|section|n|correct|ties|mean margin|
|---|---:|---:|---:|---:|
|near_distribution|96|48|0|0.0047565|
|counterfactual|48|24|0|0.00093055|
|surface_form|48|24|0|-0.0055328|
|distractor|64|32|0|-0.00373098|
|query_only_prior|16|0|0|0|

### p5
|section|n|correct|ties|mean margin|
|---|---:|---:|---:|---:|
|near_distribution|96|49|0|-0.000309813|
|counterfactual|48|24|0|-0.000451234|
|surface_form|48|24|0|-0.00289236|
|distractor|64|33|0|-0.00538221|
|query_only_prior|16|0|0|0|

### p6
|section|n|correct|ties|mean margin|
|---|---:|---:|---:|---:|
|near_distribution|96|49|0|-0.00680765|
|counterfactual|48|24|0|0.0021306|
|surface_form|48|24|0|-0.00027265|
|distractor|64|31|0|-0.0017263|
|query_only_prior|16|0|0|0|

### p7
|section|n|correct|ties|mean margin|
|---|---:|---:|---:|---:|
|near_distribution|96|48|0|-0.00488765|
|counterfactual|48|23|0|0.002993|
|surface_form|48|24|0|0.00692602|
|distractor|64|31|0|0.00179988|
|query_only_prior|16|0|0|0|

## Binding results

### graduate
- pilot0_dev: answer 80/80 (1.000); BOTH_DISTINCT 80/80; collapse 0/80; complete quartets 20/20; reversal both-correct 40/40; query-row|BD 80/80; answer|BD 80/80
- pilot1_dev: answer 79/80 (0.988); BOTH_DISTINCT 76/80; collapse 0/80; complete quartets 19/20; reversal both-correct 39/40; query-row|BD 76/76; answer|BD 76/76
### pilot0
- pilot0_dev: answer 73/80 (0.912); BOTH_DISTINCT 76/80; collapse 4/80; complete quartets 15/20; reversal both-correct 34/40; query-row|BD 72/76; answer|BD 71/76
- pilot1_dev: answer 73/80 (0.912); BOTH_DISTINCT 78/80; collapse 2/80; complete quartets 15/20; reversal both-correct 34/40; query-row|BD 74/78; answer|BD 72/78
### pilot1
- pilot0_dev: answer 80/80 (1.000); BOTH_DISTINCT 80/80; collapse 0/80; complete quartets 20/20; reversal both-correct 40/40; query-row|BD 80/80; answer|BD 80/80
- pilot1_dev: answer 80/80 (1.000); BOTH_DISTINCT 80/80; collapse 0/80; complete quartets 20/20; reversal both-correct 40/40; query-row|BD 80/80; answer|BD 80/80
### p5
- pilot0_dev: answer 50/80 (0.625); BOTH_DISTINCT 60/80; collapse 2/80; complete quartets 6/20; reversal both-correct 17/40; query-row|BD 55/60; answer|BD 40/60
- pilot1_dev: answer 47/80 (0.588); BOTH_DISTINCT 60/80; collapse 2/80; complete quartets 6/20; reversal both-correct 18/40; query-row|BD 56/60; answer|BD 38/60
### p6
- pilot0_dev: answer 50/80 (0.625); BOTH_DISTINCT 64/80; collapse 4/80; complete quartets 5/20; reversal both-correct 19/40; query-row|BD 60/64; answer|BD 44/64
- pilot1_dev: answer 52/80 (0.650); BOTH_DISTINCT 66/80; collapse 4/80; complete quartets 6/20; reversal both-correct 21/40; query-row|BD 62/66; answer|BD 48/66
### p7
- pilot0_dev: answer 48/80 (0.600); BOTH_DISTINCT 70/80; collapse 2/80; complete quartets 5/20; reversal both-correct 16/40; query-row|BD 64/70; answer|BD 44/70
- pilot1_dev: answer 46/80 (0.575); BOTH_DISTINCT 68/80; collapse 0/80; complete quartets 5/20; reversal both-correct 15/40; query-row|BD 61/68; answer|BD 41/68
