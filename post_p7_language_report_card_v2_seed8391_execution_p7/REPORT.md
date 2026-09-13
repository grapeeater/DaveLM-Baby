# P7 on prospective report card v2

Single authorized execution. Controlled scores are matched name likelihood comparisons, not generated answers. Naturalistic text is saved verbatim; JSON strings additionally expose whitespace and empty outputs.

## Section results

| Section | Items correct | Ties | Complete families | Reversals both correct | Mean family reversal | Margin min / median / mean / max |
|---|---:|---:|---:|---:|---:|---|
| near_distribution | 48/96 | 0 | 0/12 | 1/48 | 0.020833 | -5.225492 / 0.203189 / -0.009929 / 5.329760 |
| counterfactual | 32/64 | 0 | 0/8 | 1/32 | 0.031250 | -5.060123 / -0.149610 / -0.016843 / 4.529224 |
| surface_form | 31/64 | 0 | 0/8 | 1/32 | 0.031250 | -1.377661 / -0.006777 / -0.000772 / 1.383940 |
| distractor | 64/128 | 0 | 0/8 | 0/64 | 0.000000 | -6.000823 / -0.112366 / -0.011520 / 5.687880 |

## Every family

| Family | Correct items | Complete | Reversals | Mean margin | Minimum margin |
|---|---:|---|---:|---:|---:|
| counterfactual:lex12 | 4/8 | False | 0/4 | 0.006498 | -5.060123 |
| counterfactual:lex13 | 4/8 | False | 0/4 | 0.005836 | -4.011072 |
| counterfactual:lex14 | 4/8 | False | 1/4 | -0.064455 | -3.374706 |
| counterfactual:lex15 | 4/8 | False | 0/4 | 0.007630 | -4.604121 |
| counterfactual:lex16 | 4/8 | False | 0/4 | -0.017760 | -3.807527 |
| counterfactual:lex17 | 4/8 | False | 0/4 | -0.006276 | -3.917324 |
| counterfactual:lex18 | 4/8 | False | 0/4 | -0.043201 | -4.752611 |
| counterfactual:lex19 | 4/8 | False | 0/4 | -0.023020 | -4.625704 |
| distractor:lex00 | 8/16 | False | 0/8 | 0.012269 | -3.552280 |
| distractor:lex01 | 8/16 | False | 0/8 | 0.003630 | -2.817382 |
| distractor:lex02 | 8/16 | False | 0/8 | -0.026553 | -3.230542 |
| distractor:lex03 | 8/16 | False | 0/8 | -0.033880 | -3.747242 |
| distractor:lex04 | 8/16 | False | 0/8 | -0.008098 | -5.240939 |
| distractor:lex05 | 8/16 | False | 0/8 | -0.018483 | -3.033972 |
| distractor:lex06 | 8/16 | False | 0/8 | 0.004831 | -3.767011 |
| distractor:lex07 | 8/16 | False | 0/8 | -0.025880 | -6.000823 |
| near_distribution:lex00 | 4/8 | False | 0/4 | -0.018483 | -4.333477 |
| near_distribution:lex01 | 4/8 | False | 0/4 | 0.042772 | -4.384850 |
| near_distribution:lex02 | 4/8 | False | 0/4 | -0.034309 | -4.800659 |
| near_distribution:lex03 | 4/8 | False | 0/4 | -0.070407 | -4.397299 |
| near_distribution:lex04 | 4/8 | False | 0/4 | 0.034559 | -4.515524 |
| near_distribution:lex05 | 4/8 | False | 0/4 | -0.028952 | -4.397059 |
| near_distribution:lex06 | 4/8 | False | 0/4 | -0.033545 | -4.454760 |
| near_distribution:lex07 | 4/8 | False | 0/4 | 0.008968 | -5.225492 |
| near_distribution:lex08 | 4/8 | False | 0/4 | 0.003457 | -4.955657 |
| near_distribution:lex09 | 4/8 | False | 1/4 | -0.059677 | -4.203337 |
| near_distribution:lex10 | 4/8 | False | 0/4 | 0.000785 | -4.948624 |
| near_distribution:lex11 | 4/8 | False | 0/4 | 0.035685 | -4.751743 |
| surface_form:lex00 | 4/8 | False | 0/4 | -0.005691 | -1.377661 |
| surface_form:lex01 | 3/8 | False | 0/4 | -0.010339 | -0.790751 |
| surface_form:lex02 | 5/8 | False | 1/4 | -0.004901 | -0.352578 |
| surface_form:lex03 | 3/8 | False | 0/4 | 0.005657 | -0.475841 |
| surface_form:lex04 | 4/8 | False | 0/4 | 0.006291 | -0.328418 |
| surface_form:lex05 | 4/8 | False | 0/4 | -0.006924 | -0.451772 |
| surface_form:lex06 | 4/8 | False | 0/4 | 0.004463 | -0.249249 |
| surface_form:lex07 | 4/8 | False | 0/4 | 0.005270 | -0.260286 |

## Matched diagnostic comparisons

Only lexical families 0–7 are matched across active, surface and distractor conditions. Distractor families have twice as many items and reversal pairs. Surface changes both fact and query wording.

```json
{"lexical_id": 0, "outcomes": {"distractor": {"complete": false, "item_correct": 8, "items": 16, "margin": {"max": 3.364040791113693, "mean": 0.012269322655770187, "median": 0.0921421519043566, "min": -3.552280353058846, "n": 16}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 8}, "near_distribution": {"complete": false, "item_correct": 4, "items": 8, "margin": {"max": 4.865176902074412, "mean": -0.018483446365011647, "median": 0.28799004664141314, "min": -4.333477176065614, "n": 8}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 4}, "surface_form": {"complete": false, "item_correct": 4, "items": 8, "margin": {"max": 1.3839397600585173, "mean": -0.005691015909963459, "median": 0.015633605802097605, "min": -1.377661247895249, "n": 8}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 4}}}
```
```json
{"lexical_id": 1, "outcomes": {"distractor": {"complete": false, "item_correct": 8, "items": 16, "margin": {"max": 2.795699231358956, "mean": 0.0036303660278451666, "median": -0.06435841123566899, "min": -2.8173820014665836, "n": 16}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 8}, "near_distribution": {"complete": false, "item_correct": 4, "items": 8, "margin": {"max": 4.783363853119933, "mean": 0.04277159803375352, "median": 0.0035443935613788113, "min": -4.384850259585832, "n": 8}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 4}, "surface_form": {"complete": false, "item_correct": 3, "items": 8, "margin": {"max": 0.787094328218302, "mean": -0.010339199052679104, "median": -0.03507056027054656, "min": -0.7907509325346815, "n": 8}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 4}}}
```
```json
{"lexical_id": 2, "outcomes": {"distractor": {"complete": false, "item_correct": 8, "items": 16, "margin": {"max": 3.3402301641663925, "mean": -0.02655301116613873, "median": -0.03398245499911212, "min": -3.23054212265491, "n": 16}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 8}, "near_distribution": {"complete": false, "item_correct": 4, "items": 8, "margin": {"max": 4.602070720984937, "mean": -0.034309155810468006, "median": -0.03316384040480891, "min": -4.8006585470933345, "n": 8}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 4}, "surface_form": {"complete": false, "item_correct": 5, "items": 8, "margin": {"max": 0.27386390902325797, "mean": -0.004900556581602045, "median": 0.027785125495608298, "min": -0.35257758014511165, "n": 8}, "reversal_fraction": 0.25, "reversals_correct": 1, "reversals_total": 4}}}
```
```json
{"lexical_id": 3, "outcomes": {"distractor": {"complete": false, "item_correct": 8, "items": 16, "margin": {"max": 3.4906839450550855, "mean": -0.033879735791325416, "median": -0.10872155852608945, "min": -3.74724176056095, "n": 16}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 8}, "near_distribution": {"complete": false, "item_correct": 4, "items": 8, "margin": {"max": 3.946159319092512, "mean": -0.07040686728085421, "median": -0.3410280250236468, "min": -4.397298654543707, "n": 8}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 4}, "surface_form": {"complete": false, "item_correct": 3, "items": 8, "margin": {"max": 0.6152197734675191, "mean": 0.005656947824928427, "median": -0.022023993736952185, "min": -0.47584105986450176, "n": 8}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 4}}}
```
```json
{"lexical_id": 4, "outcomes": {"distractor": {"complete": false, "item_correct": 8, "items": 16, "margin": {"max": 4.8891179910773825, "mean": -0.008097868078671078, "median": -0.1797366301546104, "min": -5.240938958314073, "n": 16}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 8}, "near_distribution": {"complete": false, "item_correct": 4, "items": 8, "margin": {"max": 3.6598396618711178, "mean": 0.03455899456930234, "median": -0.13958600116325748, "min": -4.515523932219702, "n": 8}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 4}, "surface_form": {"complete": false, "item_correct": 4, "items": 8, "margin": {"max": 0.34566614493988546, "mean": 0.006290517589448541, "median": -0.025175887722497237, "min": -0.3284181126621508, "n": 8}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 4}}}
```
```json
{"lexical_id": 5, "outcomes": {"distractor": {"complete": false, "item_correct": 8, "items": 16, "margin": {"max": 3.26808142778118, "mean": -0.018483171515488395, "median": 0.007153678520294804, "min": -3.0339716220717854, "n": 16}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 8}, "near_distribution": {"complete": false, "item_correct": 4, "items": 8, "margin": {"max": 4.318709125861918, "mean": -0.02895235308738986, "median": -0.0011756561914229025, "min": -4.397058933328413, "n": 8}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 4}, "surface_form": {"complete": false, "item_correct": 4, "items": 8, "margin": {"max": 0.3938063977889197, "mean": -0.006924396102026775, "median": 0.009070476598123989, "min": -0.4517720978563604, "n": 8}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 4}}}
```
```json
{"lexical_id": 6, "outcomes": {"distractor": {"complete": false, "item_correct": 8, "items": 16, "margin": {"max": 3.4289026207772473, "mean": 0.004831447739152783, "median": 0.00031259909569492095, "min": -3.7670111666173067, "n": 16}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 8}, "near_distribution": {"complete": false, "item_correct": 4, "items": 8, "margin": {"max": 4.101040432172642, "mean": -0.03354539146693969, "median": 0.12911592183591747, "min": -4.4547604203432325, "n": 8}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 4}, "surface_form": {"complete": false, "item_correct": 4, "items": 8, "margin": {"max": 0.22615860687068512, "mean": 0.004462967367960147, "median": 0.001757579355891714, "min": -0.24924907266933083, "n": 8}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 4}}}
```
```json
{"lexical_id": 7, "outcomes": {"distractor": {"complete": false, "item_correct": 8, "items": 16, "margin": {"max": 5.687880080394333, "mean": -0.025879880092664398, "median": -0.2260179050818545, "min": -6.000823121039394, "n": 16}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 8}, "near_distribution": {"complete": false, "item_correct": 4, "items": 8, "margin": {"max": 4.225311713114058, "mean": 0.008968311855707789, "median": -0.12361631917251614, "min": -5.22549151476966, "n": 8}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 4}, "surface_form": {"complete": false, "item_correct": 4, "items": 8, "margin": {"max": 0.29204778537993015, "mean": 0.005270482918990016, "median": -0.039718405995910366, "min": -0.2602862950660807, "n": 8}, "reversal_fraction": 0.0, "reversals_correct": 0, "reversals_total": 4}}}
```

## Frozen representative sample

Every item of the lexicographically first family in each controlled section. No selection based on outcomes.

### near_distribution:lex00:a0q0o0d-1

PROMPT:
```text
Sam found the green box. Tom carried the blue kite.
Who found the green box?
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-15.869908863799031, -20.735085765873443]
Margin: 4.865176902074412
Selected candidate (JSON): " Sam."
Outcome: correct

### near_distribution:lex00:a0q0o1d-1

PROMPT:
```text
Tom carried the blue kite. Sam found the green box.
Who found the green box?
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-16.77550704569499, -19.67686468717358]
Margin: 2.90135764147859
Selected candidate (JSON): " Sam."
Outcome: correct

### near_distribution:lex00:a0q1o0d-1

PROMPT:
```text
Sam found the green box. Tom carried the blue kite.
Who carried the blue kite?
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-16.687505178564866, -20.056108005811097]
Margin: -3.3686028272462316
Selected candidate (JSON): " Sam."
Outcome: incorrect

### near_distribution:lex00:a0q1o1d-1

PROMPT:
```text
Tom carried the blue kite. Sam found the green box.
Who carried the blue kite?
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-17.575191070260757, -17.630184531518537]
Margin: -0.05499346125778004
Selected candidate (JSON): " Sam."
Outcome: incorrect

### near_distribution:lex00:a1q0o0d-1

PROMPT:
```text
Tom found the green box. Sam carried the blue kite.
Who found the green box?
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-16.89320545227329, -20.41248112500528]
Margin: -3.5192756727319896
Selected candidate (JSON): " Sam."
Outcome: incorrect

### near_distribution:lex00:a1q0o1d-1

PROMPT:
```text
Sam carried the blue kite. Tom found the green box.
Who found the green box?
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-16.13576589350367, -20.469243069569284]
Margin: -4.333477176065614
Selected candidate (JSON): " Sam."
Outcome: incorrect

### near_distribution:lex00:a1q1o0d-1

PROMPT:
```text
Tom found the green box. Sam carried the blue kite.
Who carried the blue kite?
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-17.696808285158, -18.327781839698606]
Margin: 0.6309735545406063
Selected candidate (JSON): " Sam."
Outcome: correct

### near_distribution:lex00:a1q1o1d-1

PROMPT:
```text
Sam carried the blue kite. Tom found the green box.
Who carried the blue kite?
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-17.146845342496153, -19.877818810784067]
Margin: 2.730973468287914
Selected candidate (JSON): " Sam."
Outcome: correct

### counterfactual:lex12:a0q0o0d-1

PROMPT:
```text
Sam found the yellow car. Tom carried the blue fish.
Who found the yellow car?
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-15.605928430477434, -19.884668079624653]
Margin: 4.278739649147219
Selected candidate (JSON): " Sam."
Outcome: correct

### counterfactual:lex12:a0q0o1d-1

PROMPT:
```text
Tom carried the blue fish. Sam found the yellow car.
Who found the yellow car?
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-16.740192090410734, -20.346448518594013]
Margin: 3.606256428183279
Selected candidate (JSON): " Sam."
Outcome: correct

### counterfactual:lex12:a0q1o0d-1

PROMPT:
```text
Sam found the yellow car. Tom carried the blue fish.
Who carried the blue fish?
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-16.658796055877946, -19.396416255057254]
Margin: -2.7376201991793074
Selected candidate (JSON): " Sam."
Outcome: incorrect

### counterfactual:lex12:a0q1o1d-1

PROMPT:
```text
Tom carried the blue fish. Sam found the yellow car.
Who carried the blue fish?
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-18.015143007443456, -19.96956282013644]
Margin: -1.9544198126929828
Selected candidate (JSON): " Sam."
Outcome: incorrect

### counterfactual:lex12:a1q0o0d-1

PROMPT:
```text
Tom found the yellow car. Sam carried the blue fish.
Who found the yellow car?
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-17.666148360807785, -20.00353070651641]
Margin: -2.337382345708626
Selected candidate (JSON): " Sam."
Outcome: incorrect

### counterfactual:lex12:a1q0o1d-1

PROMPT:
```text
Sam carried the blue fish. Tom found the yellow car.
Who found the yellow car?
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-15.385077679865182, -20.445200429669754]
Margin: -5.060122749804572
Selected candidate (JSON): " Sam."
Outcome: incorrect

### counterfactual:lex12:a1q1o0d-1

PROMPT:
```text
Tom found the yellow car. Sam carried the blue fish.
Who carried the blue fish?
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-18.19574954703259, -19.409563965989587]
Margin: 1.2138144189569964
Selected candidate (JSON): " Sam."
Outcome: correct

### counterfactual:lex12:a1q1o1d-1

PROMPT:
```text
Sam carried the blue fish. Tom found the yellow car.
Who carried the blue fish?
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-17.041075605064012, -20.083797007975246]
Margin: 3.0427214029112335
Selected candidate (JSON): " Sam."
Outcome: correct

### surface_form:lex00:a0q0o0d-1

PROMPT:
```text
The green box was found by Sam. The blue kite was carried by Tom.
The person who found the green box was
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-9.235313796185109, -9.275051961896683]
Margin: 0.03973816571157407
Selected candidate (JSON): " Sam."
Outcome: correct

### surface_form:lex00:a0q0o1d-1

PROMPT:
```text
The blue kite was carried by Tom. The green box was found by Sam.
The person who found the green box was
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-9.124304960291328, -9.018887600306165]
Margin: -0.10541735998516266
Selected candidate (JSON): " Tom."
Outcome: incorrect

### surface_form:lex00:a0q1o0d-1

PROMPT:
```text
The green box was found by Sam. The blue kite was carried by Tom.
The person who carried the blue kite was
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-9.510616348923126, -10.888277596818375]
Margin: -1.377661247895249
Selected candidate (JSON): " Sam."
Outcome: incorrect

### surface_form:lex00:a0q1o1d-1

PROMPT:
```text
The blue kite was carried by Tom. The green box was found by Sam.
The person who carried the blue kite was
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-9.564422180729075, -10.662021298476672]
Margin: -1.0975991177475972
Selected candidate (JSON): " Sam."
Outcome: incorrect

### surface_form:lex00:a1q0o0d-1

PROMPT:
```text
The green box was found by Tom. The blue kite was carried by Sam.
The person who found the green box was
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-9.193478266559184, -9.201949220666563]
Margin: -0.008470954107378859
Selected candidate (JSON): " Sam."
Outcome: incorrect

### surface_form:lex00:a1q0o1d-1

PROMPT:
```text
The blue kite was carried by Sam. The green box was found by Tom.
The person who found the green box was
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-9.229541909552722, -9.134939933719073]
Margin: 0.09460197583364938
Selected candidate (JSON): " Tom."
Outcome: correct

### surface_form:lex00:a1q1o0d-1

PROMPT:
```text
The green box was found by Tom. The blue kite was carried by Sam.
The person who carried the blue kite was
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-9.469981617467688, -10.853921377526206]
Margin: 1.3839397600585173
Selected candidate (JSON): " Sam."
Outcome: correct

### surface_form:lex00:a1q1o1d-1

PROMPT:
```text
The blue kite was carried by Sam. The green box was found by Tom.
The person who carried the blue kite was
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-9.696839989266477, -10.722180640118417]
Margin: 1.0253406508519394
Selected candidate (JSON): " Sam."
Outcome: correct

### distractor:lex00:a0q0o0d0

PROMPT:
```text
Mia watched a bird by the window. Sam found the green box. Tom carried the blue kite.
Who found the green box?
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-13.24981797514693, -15.63179847648611]
Margin: 2.3819805013391804
Selected candidate (JSON): " Sam."
Outcome: correct

### distractor:lex00:a0q0o0d1

PROMPT:
```text
Sam found the green box. Tom carried the blue kite. Mia watched a bird by the window.
Who found the green box?
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-12.573867326985654, -15.730513907424381]
Margin: 3.1566465804387267
Selected candidate (JSON): " Sam."
Outcome: correct

### distractor:lex00:a0q0o1d0

PROMPT:
```text
Mia watched a bird by the window. Tom carried the blue kite. Sam found the green box.
Who found the green box?
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-14.148776992501752, -16.462009885705864]
Margin: 2.3132328932041126
Selected candidate (JSON): " Sam."
Outcome: correct

### distractor:lex00:a0q0o1d1

PROMPT:
```text
Tom carried the blue kite. Sam found the green box. Mia watched a bird by the window.
Who found the green box?
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-12.757809515148914, -15.039711194723418]
Margin: 2.2819016795745046
Selected candidate (JSON): " Sam."
Outcome: correct

### distractor:lex00:a0q1o0d0

PROMPT:
```text
Mia watched a bird by the window. Sam found the green box. Tom carried the blue kite.
Who carried the blue kite?
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-11.980462441575678, -14.635636914675239]
Margin: -2.65517447309956
Selected candidate (JSON): " Sam."
Outcome: incorrect

### distractor:lex00:a0q1o0d1

PROMPT:
```text
Sam found the green box. Tom carried the blue kite. Mia watched a bird by the window.
Who carried the blue kite?
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-11.462271580499392, -15.014551933558238]
Margin: -3.552280353058846
Selected candidate (JSON): " Sam."
Outcome: incorrect

### distractor:lex00:a0q1o1d0

PROMPT:
```text
Mia watched a bird by the window. Tom carried the blue kite. Sam found the green box.
Who carried the blue kite?
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-12.86561035581072, -16.090132681243816]
Margin: -3.224522325433096
Selected candidate (JSON): " Sam."
Outcome: incorrect

### distractor:lex00:a0q1o1d1

PROMPT:
```text
Tom carried the blue kite. Sam found the green box. Mia watched a bird by the window.
Who carried the blue kite?
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-11.80476586548264, -13.902383241248431]
Margin: -2.0976173757657914
Selected candidate (JSON): " Sam."
Outcome: incorrect

### distractor:lex00:a1q0o0d0

PROMPT:
```text
Mia watched a bird by the window. Tom found the green box. Sam carried the blue kite.
Who found the green box?
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-13.23728066813256, -15.899628278154186]
Margin: -2.6623476100216266
Selected candidate (JSON): " Sam."
Outcome: incorrect

### distractor:lex00:a1q0o0d1

PROMPT:
```text
Tom found the green box. Sam carried the blue kite. Mia watched a bird by the window.
Who found the green box?
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-12.910540096672452, -15.147102866357203]
Margin: -2.2365627696847508
Selected candidate (JSON): " Sam."
Outcome: incorrect

### distractor:lex00:a1q0o1d0

PROMPT:
```text
Mia watched a bird by the window. Sam carried the blue kite. Tom found the green box.
Who found the green box?
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-14.100984346793487, -16.291740675171468]
Margin: -2.190756328377981
Selected candidate (JSON): " Sam."
Outcome: incorrect

### distractor:lex00:a1q0o1d1

PROMPT:
```text
Sam carried the blue kite. Tom found the green box. Mia watched a bird by the window.
Who found the green box?
```
Correct candidate (JSON): " Tom."
Competing candidate (JSON): " Sam."
Candidate log-likelihoods [Sam, Tom]: [-12.648260435501232, -15.725469178471796]
Margin: -3.077208742970564
Selected candidate (JSON): " Sam."
Outcome: incorrect

### distractor:lex00:a1q1o0d0

PROMPT:
```text
Mia watched a bird by the window. Tom found the green box. Sam carried the blue kite.
Who carried the blue kite?
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-12.055384583433966, -15.06120035332668]
Margin: 3.0058157698927133
Selected candidate (JSON): " Sam."
Outcome: correct

### distractor:lex00:a1q1o0d1

PROMPT:
```text
Tom found the green box. Sam carried the blue kite. Mia watched a bird by the window.
Who carried the blue kite?
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-11.968114241052808, -14.409654846932701]
Margin: 2.4415406058798936
Selected candidate (JSON): " Sam."
Outcome: correct

### distractor:lex00:a1q1o1d0

PROMPT:
```text
Mia watched a bird by the window. Sam carried the blue kite. Tom found the green box.
Who carried the blue kite?
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-12.61012670048052, -15.557747019942235]
Margin: 2.9476203194617145
Selected candidate (JSON): " Sam."
Outcome: correct

### distractor:lex00:a1q1o1d1

PROMPT:
```text
Sam carried the blue kite. Tom found the green box. Mia watched a bird by the window.
Who carried the blue kite?
```
Correct candidate (JSON): " Sam."
Competing candidate (JSON): " Tom."
Candidate log-likelihoods [Sam, Tom]: [-11.496126659695316, -14.860167450809008]
Margin: 3.364040791113693
Selected candidate (JSON): " Sam."
Outcome: correct

## Every naturalistic prompt and raw continuation

### naturalistic:00

PROMPT:
```text
Sam carried a green book to the porch, then sat beside the door.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:01

PROMPT:
```text
Tom found a yellow hat under the chair and brushed off the dust.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:02

PROMPT:
```text
Lily saw a red kite caught in a low branch and reached toward it.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:03

PROMPT:
```text
Mia put a blue toy on the shelf so her little brother could see it.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:04

PROMPT:
```text
Sam looked for his red ball behind the sofa but it was not there.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:05

PROMPT:
```text
Tom opened a small box and saw his missing green car inside.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:06

PROMPT:
```text
Lily carried a yellow book into the kitchen to show her father.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:07

PROMPT:
```text
Mia found a blue hat on the steps and called to its owner.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:08

PROMPT:
```text
Rain began while Sam was outside, so he hurried toward the house.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:09

PROMPT:
```text
Tom was cold after his walk and reached for a dry coat.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:10

PROMPT:
```text
Lily could not reach the toy on the high shelf and asked for help.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:11

PROMPT:
```text
Mia heard her friend calling from the gate and turned around.
```
BABY:
```text
 Mia put down the green ball.
```
Raw continuation as JSON: " Mia put down the green ball."
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: punctuation; generated tokens: 13

### naturalistic:12

PROMPT:
```text
Sam spilled water on the table and went to get a cloth.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:13

PROMPT:
```text
Tom noticed that his friend had no lunch and moved his plate closer.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:14

PROMPT:
```text
Lily saw that the plant was dry and filled a small cup with water.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:15

PROMPT:
```text
Mia was tired after playing outside and lay down on her bed.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:16

PROMPT:
```text
Sam left the green ball indoors before going out to play.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:17

PROMPT:
```text
Tom held the blue book while Lily carried the red box.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:18

PROMPT:
```text
Lily put her yellow hat beside the door before washing her hands.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:19

PROMPT:
```text
Mia gave the red toy to Sam and kept the blue kite.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:20

PROMPT:
```text
Sam closed the window because the rain was blowing inside.
```
BABY:
```text
 Sam put down the red fish.
```
Raw continuation as JSON: " Sam put down the red fish."
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: punctuation; generated tokens: 11

### naturalistic:21

PROMPT:
```text
Tom found the path blocked by a fallen branch and stopped walking.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:22

PROMPT:
```text
Lily heard a soft sound from the box and carefully lifted the lid.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

### naturalistic:23

PROMPT:
```text
Mia waited for the paint to dry before touching her new toy.
```
BABY:
```text

```
Raw continuation as JSON: ""
RUBRIC: HUMAN REVIEW REQUIRED for grammatical completeness, subject/object consistency, relation/action appropriateness, repetition and contradiction.
Mechanical truncation: False; stop reason: eos; generated tokens: 1

## Limits

This battery contains only two answer names and two factual predicates. Active factual QA is more demanding than the original P7 single-story-prompt panel. Complete-family/reversal outcomes, not isolated correct answers, assess reliable selection. Surface facts and query style change together; distractor comparisons use matched families. Exact non-overlap excludes neither paraphrases nor semantic contamination. No binding-retention test was authorized. No broad English, conversational, general-reasoning, architecture, capacity, or causal-training claim follows from this single checkpoint evaluation. V1 controlled scores remain withdrawn.
