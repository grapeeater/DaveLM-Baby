# DaveLM v1.0

DaveLM is a from-scratch language-model research project. The in-house model
family is codenamed **Baby**. v1.0 is the frozen graduate of the v0.10 / stack2
line: a small transformer plus a PropMatchHead runtime that can bind and
retrieve simple in-context facts (color, size, who, has, beside, combine).

This repository is **source-available / proprietary**, not open source.
Copyright © 2026 James David Mangiaracina.

## What v1.0 represents

DaveLM Baby graduated as **v1.0** after Closed Final Exam Form B (78/80).

| Record | Value |
| --- | --- |
| Immutable Git tag | `v1.0` |
| Release commit | `7ed112641b83541f6f62f5f454c5841deb9d6ca1` |
| Official remote | https://github.com/grapeeater/DaveLM-Baby.git |
| License | `LicenseRef-DaveLM-Research-1.0` (`LICENSE`) |

The `v1.0` **tag** is immutable and must stay on that commit. Later
documentation, license, and cleanup commits may exist on branch `v1.0` *after*
the tagged snapshot. They do not rewrite the tag, the graduate weights, or the
frozen runtime hashes.

v1.0 is an inference-ready graduate snapshot. It is not a general assistant,
not a coding model, and not a claim of open-domain language understanding.

## Architecture

Class: `BabyVNextConfig` / `baby_vnext_capacity_successor_v1`
(`src/baby_v010/config.py`)

| Setting | Value |
| --- | --- |
| Parameters | ~61.5M (60,536,064 base + 984,321 binding/localizer tensors) |
| vocab_size | 1024 |
| context_length | 256 |
| d_model | 640 |
| n_heads | 10 |
| n_layers | 12 |
| d_mlp | 2560 |
| activation | ReLU |
| norm | explicit pre-norm LayerNorm |
| positions | learned absolute |
| LM head | untied |
| attention | SDPA |

Tokenizer: DaveLM tokenizer v0.7 at `tokenizer/v0_7/davelm_tokenizer.json`.

Default inference path:

- Backbone: `runs/actual_baby/stack2/s5b3_compose_lock_326191/checkpoint_00025.pt`
- Head: `runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt`
- Runtime: `--runtime r3` (`selection_stack2_r3.py` PropMatchRuntime)
- RelAssist OFF, WhoProp OFF
- Inverse WHO / HAS / BESIDE scoring: `selection_stack2_r7.py`

The binding sidecar exists in the config (orthogonal two-slot). Do not describe
it as a proven retrieval subsystem; inverse bind is the R3/R7 runtime path.

## Frozen graduate hashes

These bytes must not change on official artifacts:

```
0d3e694750670996ec6053118172a499bd739274726eccea32323528cf2747cd  checkpoint_00025.pt
8ea3dbfd4825058eea74626b07621ea8c0155a960fccf1530c9123f61f7d62d9  prop_match_head_00050.pt
e1c18bae74f6d502c0012953b3eef63f787cefd41c9a47b94e803e665dab343b  davelm_tokenizer.json
e7bd0b3006c22289182a4f5b9093b3c84606040f09ebfd595b747023aa852b33  selection_stack2_r3.py
7520622c24f54c7d3aac7ed790403cff2e84b153a4593a149278ce45f3f7f1c0  selection_stack2_r7.py
```

See `GRADUATE_SHA256SUMS.txt` for the full frozen set, including Form B receipts.

## Installation / environment

Hardware used for the graduate: AMD Radeon RX 9060 XT (RDNA4, gfx1200).

Known working stack:

- Python 3.12
- ROCm SDK 7.2
- PyTorch `2.9.1+rocmsdk20260116`

**Do not** install generic CPU-only PyTorch from PyPI. **Do not** install
NVIDIA CUDA packages. Create a dedicated venv and follow `ENV_SETUP.md`.

```powershell
cd C:\DaveLM-v1.0
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# then the ROCm SDK + ROCm PyTorch wheels from ENV_SETUP.md
.\.venv\Scripts\pip.exe install "tokenizers>=0.15"
```

Verify GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected: `2.9.1+rocmsdk20260116`, `True`, `AMD Radeon RX 9060 XT`.

DaveLM uses `--device cuda`. On this ROCm Windows build, `torch.cuda.is_available()`
is True and maps to the AMD GPU via HIP.

## Run inference

```powershell
cd C:\DaveLM-v1.0
.\run_chat.ps1 --runtime r3 --device cuda
.\run_chat.ps1 --runtime r3 --device cuda --smoke
```

Equivalent:

```powershell
$env:PYTHONPATH = "C:\DaveLM-v1.0\src"
.\.venv\Scripts\python.exe -m baby_v010.stack2_chat --runtime r3 --device cuda
```

Protocol notes:

- Default runtime is r3. RelAssist and WhoProp stay off.
- Usable-chat shape is facts outside `Human:`. Stuffing facts inside `Human:`
  is off-protocol and collapses to a color prior (green).
- This tree is the museum copy of the graduate. Do not train these weights
  in place.

## Verification / canaries

```powershell
cd C:\DaveLM-v1.0
.\.venv\Scripts\python.exe scripts\verify_canary.py --runtime r3 --device cuda
```

`scripts/verify_canary.py` is five mechanism probes (direct color, inverse WHO,
HAS-entity, late WHO, one fact-combine). It is **not** Form A or Form B.
A passing run writes `VERIFY_CANARY.json` with `"pass": true`.

Form B sealed evidence lives under `runs/actual_baby/final_exam_v1/form_b/`.
Do not reopen Form A/B as a treatment set.

## Known v1.0 limitations

These are leftover weaknesses of the frozen graduate, not a todo list for
editing official weights.

- **Closed vocabulary.** Binding lists in `data_language_bridge.py` are a small
  animal/color/size/place set. Unknown names and unseen adjectives are outside
  the trained/runtime whitelist.
- **A8 (Form B).** Size-ask on cow produced `cat has the cow.` instead of the
  size value.
- **G2.4 (Form B).** Pig HAD green; Baby identified pig but finished with
  `pig is green.` (HAS-entity, IS template).
- **Inverse bind relations.** Relation sites in memory still use `beside` /
  `next`. Unseen fact-side relation verbs will not build a pair.
- **Interrogative detection** still needs Who / Which / What. A follow-up like
  `the dog then?` stays an about-report.
- Hop weights exist in the PropMatchHead file but are unused for inverse bind.
- Color / mixed_2e first_top1 remain 0.969, not 1.000, on official panels.
  D3 long-gap remains 181/215.
- Context window is 256 tokens. There is no coding, encyclopedia, tool use,
  or open-domain chat capability.

Form B score is 78/80 because of A8 and G2.4. TEST / FINAL / SACRED stay sealed.

## Repository structure

```
LICENSE                         Canonical LicenseRef-DaveLM-Research-1.0 text
THIRD_PARTY_NOTICES.md          Third-party dependency notices
README.md                       This file
ENV_SETUP.md                    AMD ROCm / PyTorch 3.12 setup
GRADUATE_SHA256SUMS.txt         Frozen artifact hashes
MANIFEST.txt                    Release inventory
run_chat.ps1                    Inference launcher (uses .venv)
requirements.txt                High-level Python deps (install torch via ENV_SETUP)
VERIFY_CANARY.json              Last canary receipt
BASELINE_CANARY_v010.*          Pre-recycle baseline canary outputs
src/baby_v010/                  Model, runtime, eval, historical training helpers
scripts/verify_canary.py        Graduate mechanism canary
tokenizer/v0_7/                 Frozen tokenizer
runs/actual_baby/stack2/...     Graduate backbone + PropMatchHead
runs/actual_baby/final_exam_v1/ Form B sealed evidence
docs/legal/                     Original license PDF
docs/history/                   Local recycle provenance (not model weights)
```

`.venv/` is local and gitignored. Do not commit it.

Historical training/diagnostic modules in `src/baby_v010/` are kept so the
graduate runtime and evaluation path remain reproducible. Official v1.0 weights
must not be overwritten.

## Licensing

DaveLM is Copyright © 2026 James David Mangiaracina. This repository and its
covered source code, model weights, checkpoints, tokenizer/configuration,
evaluation artifacts, and authorized derivatives are governed by the DaveLM
Proprietary Research and Evaluation License v1.0
(`LicenseRef-DaveLM-Research-1.0`), except for separately identified
third-party materials. The default grant is limited to internal,
non-commercial research and evaluation. Commercial use, redistribution, hosted
third-party access, and model extraction/distillation require separate written
permission. See `LICENSE` and `THIRD_PARTY_NOTICES.md` for complete terms.

The original signed-style PDF is
`docs/legal/DaveLM_Proprietary_Research_License_v1.0.pdf`. If summary wording
conflicts with the full legal text, the full legal text controls.

SPDX identifier for project-owned files: `LicenseRef-DaveLM-Research-1.0`.
Do not label this repository MIT, Apache-2.0, GPL, or another standard
open-source license.

## Release status

- Tag `v1.0` → commit `7ed112641b83541f6f62f5f454c5841deb9d6ca1` (immutable).
- Post-v1.0 documentation / license / cleanup commits are allowed *after* that
  commit on the `v1.0` branch. They must not retarget the tag, force-push, or
  alter graduate hashes.
- Continuing training belongs in a separate working tree, not in this museum
  copy of the graduate.
