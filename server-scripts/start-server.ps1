# Starts the YOLO Annotator lab server, after choosing the dataset folder.
# Dataset choices: folders matching the patterns in datasets.txt (one per line, wildcards ok).
# Without datasets.txt the server asks with a folder dialog.
param([int]$Port = 8080)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$exe = Join-Path $here "yolo-annotator-server.exe"
$Host.UI.RawUI.WindowTitle = "YOLO Annotator lab server - start"

function Test-Running {
    try { Invoke-WebRequest "http://localhost:$Port/" -UseBasicParsing -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}

if (Test-Running) {
    Write-Host "The lab server is already running: http://localhost:$Port" -ForegroundColor Yellow
    Write-Host "Use 'Stop Lab Server' first to switch to another dataset."
    Read-Host "Press Enter to close"
    exit
}

# the dataset to serve
$last = $null
$cfg = Join-Path $env:APPDATA "YOLO Annotator\server_dataset.txt"
if (Test-Path $cfg) { $last = (Get-Content $cfg -Raw).Trim() }
$patterns = @()
$list = Join-Path $here "datasets.txt"
if (Test-Path $list) { $patterns = Get-Content $list | Where-Object { $_.Trim() -and -not $_.TrimStart().StartsWith("#") } }
$folders = @($patterns | ForEach-Object { Get-Item -Path $_.Trim() -ErrorAction SilentlyContinue } | Where-Object { $_.PSIsContainer } | Sort-Object FullName -Unique)

$dataset = $null
if ($folders.Count -gt 0) {
    Write-Host ""
    Write-Host "Which dataset should the lab work on?" -ForegroundColor Cyan
    $default = 0
    for ($i = 0; $i -lt $folders.Count; $i++) {
        $mark = ""
        if ($folders[$i].FullName -eq $last) { $default = $i; $mark = "   (last used)" }
        Write-Host ("  [{0}] {1}{2}" -f ($i + 1), $folders[$i].FullName, $mark)
    }
    Write-Host ""
    $answer = Read-Host ("Number (Enter = {0})" -f ($default + 1))
    $n = $default
    if ($answer.Trim()) { $n = [int]$answer - 1 }
    if ($n -lt 0 -or $n -ge $folders.Count) { Write-Host "No such number." -ForegroundColor Red; Read-Host "Press Enter to close"; exit 1 }
    $dataset = $folders[$n].FullName
}

# a server update copied in while it was running is put in place now
$new = Join-Path $here "yolo-annotator-server.new.exe"
if (Test-Path $new) {
    Move-Item $new $exe -Force
    Write-Host "Updated the lab server program." -ForegroundColor Green
}

$args2 = @("--port", $Port)
# extra server options, one per line (e.g. --gpu 0)
$opt = Join-Path $here "options.txt"
if (Test-Path $opt) {
    $args2 += @(Get-Content $opt | Where-Object { $_.Trim() -and -not $_.TrimStart().StartsWith("#") } | ForEach-Object { $_.Trim() -split "\s+" })
}
if ($dataset) { $args2 = @("`"$dataset`"") + $args2 }
Start-Process -FilePath $exe -ArgumentList $args2 -WorkingDirectory $here
Write-Host ""
Write-Host "Starting the lab server..." -NoNewline
for ($i = 0; $i -lt 30 -and -not (Test-Running); $i++) { Start-Sleep -Milliseconds 500; Write-Host "." -NoNewline }
Write-Host ""
if (Test-Running) {
    $ip = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*" -and $_.PrefixOrigin -ne "WellKnown" } |
        Select-Object -ExpandProperty IPAddress) -join ", "
    Write-Host "Running. Lab members open:" -ForegroundColor Green
    foreach ($a in ($ip -split ", ")) { if ($a) { Write-Host "  http://${a}:$Port" -ForegroundColor Green } }
    if ($dataset) { Write-Host "Dataset: $dataset" }
    Write-Host "The server window shows who is working; stop it with 'Stop Lab Server'."
} else {
    Write-Host "The server did not start; see its window for the reason." -ForegroundColor Red
}
Start-Sleep -Seconds 8
