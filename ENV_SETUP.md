# DaveLM v1.0 — GPU Environment Setup (AMD ROCm on Windows)

Hardware: AMD Radeon RX 9060 XT (RDNA4, gfx1200).  
Known working PyTorch: `2.9.1+rocmsdk20260116` (ROCm 7.2 Windows wheels).

## Prerequisites

- Python **3.12** (`py -3.12`)
- AMD graphics driver **26.1.1** or newer (ROCm 7.2 Windows requirement)
- Dedicated venv at `C:\DaveLM-v1.0\.venv` (do not use global/user Python)

## Create venv

```powershell
cd C:\DaveLM-v1.0
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

## Install ROCm SDK (step 1)

```powershell
.\.venv\Scripts\pip.exe install --no-cache-dir `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_core-7.2.0.dev0-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_devel-7.2.0.dev0-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm_sdk_libraries_custom-7.2.0.dev0-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/rocm-7.2.0.dev0.tar.gz
```

## Install ROCm PyTorch (step 2)

**Do not** `pip install torch` from PyPI — that installs CPU/CUDA builds, not ROCm.

```powershell
.\.venv\Scripts\pip.exe install --no-cache-dir `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torch-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchaudio-2.9.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2/torchvision-0.24.1%2Brocmsdk20260116-cp312-cp312-win_amd64.whl
```

## Install remaining requirements

```powershell
.\.venv\Scripts\pip.exe install "tokenizers>=0.15"
```

(`requirements.txt` lists `torch>=2.0` generically; install torch separately as above.)

## Verify GPU

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected:

```
2.9.1+rocmsdk20260116
True
AMD Radeon RX 9060 XT
```

DaveLM uses `--device cuda`; on ROCm Windows builds, `torch.cuda.is_available()` is True and maps to the AMD GPU via HIP.

## Run DaveLM

```powershell
cd C:\DaveLM-v1.0
.\run_chat.ps1 --runtime r3 --device cuda
.\run_chat.ps1 --runtime r3 --device cuda --smoke
.\.venv\Scripts\python.exe scripts\verify_canary.py
```

## Evidence sources

- `research/V010_QUERY_TRANSPORT_AUTOPSY.md` — `torch 2.9.1+rocmsdk20260116`, RX 9060 XT
- `research/V010_SELECTION_REPAIR_*_TERMINAL.md` — same torch/device notes
- Previous venv: `C:\DaveLM-CADAVER\sf2_runtime\sf2venv` (recycled 2026-09-19)
- AMD docs: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/install/installrad/windows/install-pytorch.html

## What not to install

- NVIDIA `nvidia-cuda-*` packages
- Generic PyPI `torch` without ROCm wheels
- Do not modify graduate weights, tokenizer, or frozen runtime modules
