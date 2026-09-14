# Post-T28 causal prefix-rescue study

Status: **COMPLETE — `EARLY_TOKEN_ERROR_CAUSALLY_DOMINATES`**.

Tokenizer engineering remains paused. T29 was not launched. No optimizer was created.
Checkpoints and v0_7 were not modified. TEST / FINAL / sacred were not loaded.
Activation patching was not run.

## Primary question

When Baby makes the first wrong answer token after an otherwise-correct answer prefix,
if we repair only that token and return to free greedy AR generation, does she recover
the correct answer?

## Population (frozen before rescue)

DEV first-token-correct-then-diverge across T28 seeds 850001–850003: 39/40/45 (pooled 124). All rows were retained. Sal/Skye were not used as a filter.

Control tokens B and D were assigned from frozen population fields, inventory geometry,
and class/length/frequency matching before any rescue metric was computed.

## Primary intervention (A: gold repair)

Pooled: n=124; exact+EOS 0.871; next-token 0.944; remaining-suffix 0.871; period+EOS 1.000
Seed 850001: n=39; exact+EOS 0.821; next-token 0.897; remaining-suffix 0.821; period+EOS 1.000
Seed 850002: n=40; exact+EOS 0.925; next-token 1.000; remaining-suffix 0.925; period+EOS 1.000
Seed 850003: n=45; exact+EOS 0.867; next-token 0.933; remaining-suffix 0.867; period+EOS 1.000
Shared-first DEV names: n=83; exact+EOS 0.880; next-token 0.988; remaining-suffix 0.880; period+EOS 1.000
Unique-first DEV names: n=41; exact+EOS 0.854; next-token 0.854; remaining-suffix 0.854; period+EOS 1.000
Family Sal: n=36; exact+EOS 1.000; next-token 1.000; remaining-suffix 1.000; period+EOS 1.000
Family Skye: n=39; exact+EOS 0.769; next-token 1.000; remaining-suffix 0.769; period+EOS 1.000
Family Omar: n=4; exact+EOS 0.750; next-token 0.750; remaining-suffix 0.750; period+EOS 1.000
Family Opal: n=4; exact+EOS 1.000; next-token 1.000; remaining-suffix 1.000; period+EOS 1.000
Family Wes: n=40; exact+EOS 0.850; next-token 0.850; remaining-suffix 0.850; period+EOS 1.000
Family Other: n=1; exact+EOS 1.000; next-token 1.000; remaining-suffix 1.000; period+EOS 1.000

## Controls

B wrong-matched: n=124; exact+EOS 0.000; next-token 0.581; remaining-suffix 0.581; period+EOS 0.992
B seed 850001: n=39; exact+EOS 0.000; next-token 0.615; remaining-suffix 0.615; period+EOS 1.000
B seed 850002: n=40; exact+EOS 0.000; next-token 0.550; remaining-suffix 0.550; period+EOS 0.975
B seed 850003: n=45; exact+EOS 0.000; next-token 0.578; remaining-suffix 0.578; period+EOS 1.000
C no intervention: n=124; exact+EOS 0.000; next-token 0.032; remaining-suffix 0.032; period+EOS 1.000
D prefix-compatible (feasible rows only): n=55; exact+EOS 0.000; next-token 0.000; remaining-suffix 0.000; period+EOS 1.000; infeasible rows=69
D seed 850001: n=16; exact+EOS 0.000; next-token 0.000; remaining-suffix 0.000; period+EOS 1.000
D seed 850002: n=17; exact+EOS 0.000; next-token 0.000; remaining-suffix 0.000; period+EOS 1.000
D seed 850003: n=22; exact+EOS 0.000; next-token 0.000; remaining-suffix 0.000; period+EOS 1.000

B remaining-suffix 58% is not exact recovery: the matched-wrong token remains in the
string, so exact+EOS stays 0. It only means later tokens often still look like a
period/EOS tail. D is 0% exact on all 55 feasible rows.

## Sequence probability (descriptive)

At the divergence prefix, the gold token is never greedy-top-1 (mean P=0.062, mean
rank 88.9) while the emitted wrong token is always top-1 (mean P=0.840). After the
gold token is inserted, the next remaining gold token is top-1 in 94.4% of rows
(mean P=0.866). Teacher-forced remaining-gold trajectory: 238 steps, mean P=0.890,
top-1 93.3%.

These are final-layer readout statistics. They do not by themselves prove that Baby
uses any intermediate state as a causal code, and they do not say she would have
sampled gold without the oracle.

## Internal state after wrong vs gold (blocks 7–11, frozen head, no probe)

Readout of the **next** remaining gold token after appending gold vs wrong:

| Site (after block) | Gold-prefix next top-1 | Wrong-prefix next top-1 | Gold mean rank | Wrong mean rank |
|---|---:|---:|---:|---:|
| 7 (layer 8) | 85.5% | 0.0% | 1.2 | 231.2 |
| 8 (layer 9) | 95.2% | 2.4% | 1.0 | 78.1 |
| 9 (layer 10) | 92.7% | 2.4% | 1.1 | 37.8 |
| 10 (layer 11) | 96.8% | 3.2% | 1.0 | 29.7 |
| 11 / final (layer 12) | 94.4% | 3.2% | 1.1 | 22.0 |

A negative or weak readout is not evidence that information is absent.

## Activation patching

Not run. If gold rescue is large and replicated, the earned hypothesis is recorded below
and the study stops. No next treatment was designed.

## Classification

`EARLY_TOKEN_ERROR_CAUSALLY_DOMINATES`

## Earned hypothesis

Earned hypothesis only: a single first-wrong-token repair followed by free greedy recovers exact+EOS often enough, and more than matched/prefix-compatible wrong controls, that the first divergent token is a major (or causally dominant) contributor to these first-token-correct-then-diverge failures. This does not authorize activation patching, tokenizer mutation, T29, or TEST access.

## Conservative interpretation

This is a single-token oracle intervention on already-identified diverge cases.
It does not show that Baby would have sampled the gold token, does not repair the
tokenizer, and does not authorize retokenization, T29, or TEST.

