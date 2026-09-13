# SF6 prospective decision — late English LR annealing

Both supplied documents were read completely before repository implementation. They are evidence and proposals; their embedded instructions do not override Dave's request.

## What survives independent scrutiny

SF2 is an acquisition failure, with a genuine -0.030-nat Alex-correct residual and improved Owen conditioning under KL retention. Its local residual autopsy explicitly ranks constant-LR instability **UNRESOLVED**: only updates 0/100/200 are available. A boundary movement across two observations does not prove monotonic motion, momentum overshoot, or that annealing will stop at the right point. Nevertheless, reducing late English LR is the smallest authorized single-variable test of this possibility. It preserves the existing successful retention machinery.

Head constraints would change the trainable scope; pairwise loss would change the objective; ordinary-language replay changes the data/loss mixture; additional CE changes the budget; modified KL changes retention pressure. None is needed to test this live hypothesis. They are not included or reserved as automatic fallbacks. Reference-model practices do not decide this experiment.

## Corrections to Qwen's proposal

- Pair **the same** stochastic seed across independently trained control and anneal runs: 87011, 87012, 87013. Different seeds within pairs would unnecessarily confound seed and treatment.
- No shared mutable fork, historical-state restoration, prefix tensor-equality gate, or abandoned experiment is reused. Fresh parent and fresh optimizer per run.
- Preserve determinism settings and all input hashes, but do not confuse bitwise trajectory equality with the causal-comparison requirement. Same-seed pairs control intended stochastic inputs; numerical execution remains an uncertainty.
- The prior report's approximately 0.00697-nat maximum observed endpoint difference is not a proven bound, and Qwen's <0.001 claim is inaccurate. Update100 pair differences supply a prospective, pre-intervention descriptive noise reference within these actual runs. No result is dropped for such differences. We cannot establish the numerical root cause here.
- N=3 supports raw paired counts and suggestive replication, not statistical certainty. Individual full-gate acquisition success and evidence that the schedule improved success are separate conclusions.
- SF2 uses **160 positions per English update** from 165,363 frozen positions/741 rows, not a 160-example pool. It uses 20 update cycles, not 20 data epochs.
- Historical SF2 recorded D3 diagnostically. The <=0.01 requirement is explicit in the later SF5 protocol and current request; it is prospectively enforced here without rewriting SF2.

## Frozen test

Six full independent runs, each at most 200 updates. Fixed order: 87011 control/anneal; 87012 anneal/control; 87013 control/anneal. This alternating blocked order reduces systematic order confounding but cannot eliminate time effects with three pairs. All six are planned before outcomes; no optional stopping on success and no replacement of failed seeds.

Both arms use the byte-identical SF1 factual/binding schedule and SF2 KL pool/objective. Control LR is 5e-5 throughout. Anneal uses 5e-5 through update100; post100 English ordinal j=1..90 uses 5e-5*(90-j)/89, so update101 remains5e-5 and update199 reaches0. Every binding update, including200, remains5e-5. Thus 89 English updates have changed LR. At zero LR, the established Adam state update still occurs; this is not an extra skip policy. LR-scaled weight decay changes together with LR, so success would not uniquely identify a settling mechanism.

Updates0/100/200 use the existing evaluator and all16 TRAIN items, existing aligned TinyStories DEV, D3, and both nonsacred binding pools. No additional trajectory probes. Safety stops are per run; an integrity failure stops execution pending resolution. Full endpoint acquisition is 16 correct,16 exact answer+EOS,8 reversals,4 complete families AND language<=ownbaseline+.25, D3<=.01, each binding pool>=76answer/>=76BD/zero collapse. The original update100 acquisition guard is unchanged.

Primary evidence: paired full-gate wins/losses and every raw arm outcome. At least2 anneal-only successes with0 control-only successes is labeled SUGGESTIVE_ANNEAL_BENEFIT. Any control-only success is CONTROL_FAVORED_OR_MIXED; all other patterns INCONCLUSIVE. This label does not replace or relax any individual acquisition gate. Margin changes, including prefix-adjusted differences, remain descriptive. Tiny advantages comparable to measured numerical offsets do not establish a schedule benefit merely because a binary gate changes.

All held-out, alternate, copy, competing-name, FINAL and sacred panels stay locked/unscored throughout this study. Even an acquisition success is reported to Dave before any transfer opening. No treatment follows this study automatically.

If any run stops at update100, the study-level label is INCONCLUSIVE_EARLY_STOP: all outcomes remain reported, no seed is replaced, and pre-intervention numerical/guard differences cannot be attributed to annealing. Individual arm classifications remain unchanged.

## Minimum preflight and implementation reuse

Pilot1/tokenizer/input hashes, pinned runtime, exact SF2 source copies, unchanged schedules/objectives/masking/scopes, LR values, gate wiring and locked-panel exclusion are checked before execution. Native Pilot1 output controls and zero KL are checked in every run before its first optimizer is created. No rehearsal update is spent proving cross-run equality. Full rolling restart commits and raw evaluation persistence preserve evidence without shared-state forks.

Scientific payloads copied from SF1/SF2 are byte-identical. Historical overlap/semantic validation is inherited because no item or split changes. Locked records are neither copied nor parsed. The new controller calls pinned SF2 computational helpers, never its old main entrypoint with automatic transfer opening.

Exact execution command after sealing:

    C:\DaveLM-CADAVER\sf2_runtime\sf2venv\Scripts\python.exe -B C:\DaveLM-CADAVER\sf6_same_seed_lr_anneal_v1\LAUNCH.py

The launcher supplies PYTHONHASHSEED separately for every child and records each independent process log. An interruption can resume only a valid latest committed state; no optimizer update is replayed once committed.
