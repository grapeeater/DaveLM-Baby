param(
    [Parameter(Mandatory=$true)][ValidateSet('101001','101002')][string]$Seed,
    [int]$Updates = 4000,
    [int]$EvalInterval = 250,
    [int]$BatchSize = 8
)

$ErrorActionPreference = 'Stop'
$python = 'C:\DaveLM-CADAVER\sf2_runtime\sf2venv\Scripts\python.exe'
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root ('runs\foundation_v1_seed' + $Seed)
$env:PYTHONHASHSEED = $Seed
Set-Location $root
& $python -B -m src.baby_v010.train --seed ([int]$Seed) --out $out --updates $Updates --eval-interval $EvalInterval --device cuda --batch-size $BatchSize
if ($LASTEXITCODE -ne 0) { throw "Baby v0.10 seed $Seed failed with exit code $LASTEXITCODE" }
