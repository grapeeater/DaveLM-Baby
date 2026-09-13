# PATH 1 — Bitwise determinism: findings and classification

## Result: FAILED / INFEASIBLE under the mission fences

A minimal safe implementation/runtime correction that makes identical replicas bitwise-identical through the
required 100-update prefix could NOT be established. The required prefix (historical SF2 update-100) is
reproducible only by stochastic luck, and fresh replicas cannot be guaranteed bitwise-identical.

## What actually causes the divergence (evidence chain)

1. All frozen inputs are byte-identical across every comparison: engine (SF2_ENGINE.py sha
   `150d1f33...`), schedule/data/KL pool/sources (verified), parent/tokenizer hashes, interpreter (same venv),
   torch 2.12.0+rocm7.14.0, tokenizers 0.23.1, PYTHONHASHSEED=87011, seed 87011, deterministic-algorithms on,
   TF32 off, GPU exclusively idle during runs (no other GPU users observed).
2. Update-0 artifacts are byte-identical across all runs (acquisition, D3, checks). Update-1 metrics are
   byte-identical across modes (loss/ce/kl/grad_norm). Update-1 per-group gradient fingerprints are byte-
   identical across modes; update-1 clip-grad-norm float32 bits are identical across modes.
3. The first observable divergence between trajectories appears later: update 2 (kl delta 1.397e-9) in
   historical-vs-replay pairs and in probe_1a-vs-probe_1b; update 12 (kl delta 2.98e-8) in SF4 common_a vs
   common_b; other pairs never diverged within 100 updates (A==B==SF3-replay==SF4a bitwise; hist==p1a
   bitwise). Divergence is therefore stochastic per-process/per-call, not a deterministic per-process mode
   choice.
4. Cross-run tensor deltas at u100, when divergence occurs, are highly stereotyped: 334/414 tensors differ
   (every trainable parameter), max abs ~9.0-9.1e-4 (token embedding), RMS ~2.8-4.2e-6; the 80 exact tensors
   are the non-parameter causal mask buffers. Endpoint behavior is unaffected at every measured level:
   acquisition counts identical (11/16@100; 15/16@200), language CE within ~4e-6, D3 within ~7e-7, binding
   identical, and per-item margin signs never flip (max cross-mode per-item margin delta: 2.05e-3 at u100,
   6.97e-3 at u200; see mode_noise_floor.json).
5. Candidate low-level locus: the earliest differing observable quantity is the float64 KL forward reduction
   (update-2 KL bits differ while CE bits and weights remain identical); consistent with rare run-to-run
   nondeterminism of a GPU numeric kernel (e.g., reduction/autotuning behavior) in the ROCm stack that PyTorch's
   deterministic-algorithms flag does not fully cover. This is an environment/stack property, not a property of
   Baby's code, RNG usage, data order, dropout masks, or optimizer logic.

## Runtime levers tested

- Default environment: observed both matching bursts (7 consecutive identical probe runs at ~04:0x; A/B/SF3/
  SF4a identical across ~30 min) and divergent pairs (probe_1a vs probe_1b; p1a(M1) vs p1b(M2); historical vs
  replay; SF4a vs SF4b).
- `TORCH_BLAS_PREFER_HIPBLASLT=0`: still produced ≥2 distinct fingerprints across runs (M1-like and M2-like).
- No environment variable or runtime setting was found that collapses all runs into one bitwise trajectory.

## Why a "fix" is not available within the fences

- A kernel-level fix (driver/ROCm) is not available in this environment.
- Replacing the float64 KL reduction with a deterministic formulation would change the numeric implementation
  of a frozen-protocol-pinned computation (a semantics-adjacent change forbidden by the fence "Do NOT change
  scientific training semantics merely to force equality"), would not reproduce the historical trajectory (it
  would create a new numeric path), and its own cross-process determinism is unproven.
- Retry-until-lucky reproduction of the historical prefix is not a deterministic correction.

## Evidence artifacts

- probe_1a/probe_1b engine replay dirs (100-update, divergent at update 2; checkpoints `bbaeefef...` and
  `80ae7794...`).
- probe8.py / probe8g.py / probe8g2.py / kltest.py survey outputs (mode fingerprints; update-1 gradient and
  clip-norm bits identical across modes).
- pairwise_u100.py (cluster matrix: {hist, p1a} identical; {A, B, SF3, SF4a, p1b} identical; SF4b unique).
- mode_noise_floor.json (endpoint noise floor: ≤2.05e-3 nats/item at u100, ≤6.97e-3 at u200, zero sign flips).
- sf4mode.py (SF4a/SF4b diverge at update 12 despite identical update-2 KL).
