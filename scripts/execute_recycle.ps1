# Send approved historical trees to Recycle Bin (NOT permanent delete)
Add-Type -AssemblyName Microsoft.VisualBasic

$Log = "C:\DaveLM-v1.0\RECYCLE_EXECUTED.txt"
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"RECYCLE EXECUTION LOG - started $ts" | Out-File $Log -Encoding utf8

function Get-DirSizeGB($p) {
    if (-not (Test-Path $p)) { return 0 }
    [math]::Round((Get-ChildItem $p -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB, 2)
}

function Send-TreeToRecycleBin($path) {
    $size = Get-DirSizeGB $path
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $path  ($size GB)"
    if (-not (Test-Path $path)) {
        ($line + "  SKIPPED (not found)") | Add-Content $Log -Encoding utf8
        return
    }
    try {
        [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($path, 'OnlyErrorDialogs', 'SendToRecycleBin')
        ($line + "  OK -> Recycle Bin") | Add-Content $Log -Encoding utf8
    } catch {
        ($line + "  FAILED: $_") | Add-Content $Log -Encoding utf8
    }
}

$targets = @(
    'C:\DaveLM-v0.2.1','C:\DaveLM-v0.3','C:\DaveLM-v0.4','C:\DaveLM-v0.5',
    'C:\DaveLM-v0.6','C:\DaveLM-v0.6.1','C:\DaveLM-v0.7','C:\DaveLM-v0.7.1',
    'C:\DaveLM-v0.8','C:\DaveLM-v0.8.1','C:\DaveLM-v0.8.2','C:\DaveLM-v0.8.3','C:\DaveLM-v0.8.4',
    'C:\DaveLM-v0.9','C:\DaveLM-v0.10','C:\DaveLM-CADAVER','C:\DaveLM'
)

foreach ($t in $targets) { Send-TreeToRecycleBin $t }

("Completed $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'). Recycle Bin NOT emptied.") | Add-Content $Log -Encoding utf8
Get-Content $Log
