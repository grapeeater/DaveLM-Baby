# Third-Party Notices

DaveLM is Copyright © 2026 James David Mangiaracina and is governed by
`LicenseRef-DaveLM-Research-1.0`. That license applies only to rights James David
Mangiaracina owns or is authorized to license. Third-party libraries,
dependencies, datasets, tokenizer engines, and other components remain under
their own terms. This file does not relicense those components.

## Direct runtime dependencies

These packages are required to run the v1.0 graduate on the documented AMD ROCm
Windows stack. Install them from the sources in `ENV_SETUP.md`; do not treat
this file as an installer.

| Component | Typical use in this tree | License (upstream) |
| --- | --- | --- |
| [PyTorch](https://github.com/pytorch/pytorch) (`torch` 2.9.1+rocmsdk20260116) | Model load, inference, training helpers | BSD-style (PyTorch) |
| [torchvision](https://github.com/pytorch/vision) | Transitive ROCm wheel companion | BSD-style |
| [torchaudio](https://github.com/pytorch/audio) | Transitive ROCm wheel companion | BSD-style |
| [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) (`tokenizers>=0.15`) | Loads `tokenizer/v0_7/davelm_tokenizer.json` | Apache-2.0 |
| [AMD ROCm SDK for Windows](https://rocm.docs.amd.com/) (ROCm 7.2 wheels from repo.radeon.com) | HIP/ROCm backend used by the documented PyTorch build | AMD / ROCm licenses shipped with those packages |

PyTorch pulls additional third-party code (for example NumPy, Jinja2, networkx,
sympy, typing-extensions, and related utilities). Those remain under their
upstream licenses as installed in a local `.venv`.

## Project-owned artifacts that are not third-party

The following are DaveLM Covered Materials, not third-party components:

- `src/baby_v010/` model, runtime, evaluation, and training source
- `tokenizer/v0_7/davelm_tokenizer.json` (DaveLM tokenizer v0.7 configuration and vocab)
- `runs/actual_baby/stack2/s5b3_compose_lock_326191/checkpoint_00025.pt`
- `runs/actual_baby/stack2/r3/r3b_329011/prop_match_head_00050.pt`
- evaluation receipts under `runs/actual_baby/final_exam_v1/form_b/`

## Tokenizer engine vs tokenizer assets

The Hugging Face `tokenizers` library is third-party (Apache-2.0). The DaveLM
v0.7 tokenizer JSON in this repository is a DaveLM project asset and is covered
by `LicenseRef-DaveLM-Research-1.0` unless a nested third-party notice is added
later.

## Datasets

This v1.0 release tree does not ship an external third-party corpus. Training
and evaluation items in the source tree are project-authored synthetic
language-bridge / selection packs. TEST / FINAL / SACRED exam material is not
opened by this release.

## Notice duty

If Licensor later authorizes redistribution of a package that includes
third-party components, those original notices must be preserved. Nothing in
`LICENSE` reduces rights that a third-party license already grants for that
component alone, and nothing in `LICENSE` expands Licensor's rights over
material Licensor does not own.
