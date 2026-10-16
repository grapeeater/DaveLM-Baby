# RNC launcher: always the ROCm venv, UTF-8, GPU-first. Usage: .\rnc.ps1 <command> [args]   (see run.py)
$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "ROCm venv not found at $python (see ENV_SETUP in rnc/README.md)" }
& $python -X utf8 (Join-Path $PSScriptRoot 'run.py') @args
exit $LASTEXITCODE
