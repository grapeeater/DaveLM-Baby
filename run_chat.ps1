# DaveLM v1.0 graduate chat launcher (inference-only, --runtime r3 default)
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "Missing venv at $Root\.venv - see ENV_SETUP.md"
    exit 1
}
$env:PYTHONPATH = Join-Path $Root "src"
& $Python -m baby_v010.stack2_chat @args
