# Makes the release packages in .\release from .\dist (run .\build.ps1 -Cuda first):
#   YOLO-Annotator-<ver>-win64.zip        portable: unzip anywhere, run yolo-annotator.exe
#   YOLO-Annotator-<ver>-cuda-addon.zip   optional NVIDIA CUDA pack: unzip over the app folder
#   .\package.ps1 [-NoCuda]
param([switch]$NoCuda)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-Location $PSScriptRoot
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem

$ver = (Select-String -Path Cargo.toml -Pattern '^version\s*=\s*"(.+)"').Matches[0].Groups[1].Value
$name = "YOLO-Annotator-$ver"
$rel = Join-Path $PSScriptRoot "release"
$stage = Join-Path $rel "_stage"
Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $rel | Out-Null

if (-not (Test-Path dist\yolo-annotator.exe)) { throw "dist\yolo-annotator.exe missing: run .\build.ps1 first" }

# ---- standard edition (everything except the CUDA pack)
$app = Join-Path $stage "YOLO Annotator"
New-Item -ItemType Directory -Force $app | Out-Null
Copy-Item dist\yolo-annotator.exe, dist\LICENSE, dist\README.md $app
if (Test-Path dist\models) { Copy-Item dist\models $app -Recurse }
foreach ($r in "directml", "vc") {
    if (Test-Path "dist\runtime\$r") {
        New-Item -ItemType Directory -Force "$app\runtime" | Out-Null
        Copy-Item "dist\runtime\$r" "$app\runtime\$r" -Recurse
    }
}
@"
YOLO Annotator $ver - portable
================================
1. Unzip this folder anywhere (e.g. Desktop or a USB stick).
2. Run yolo-annotator.exe. Nothing is installed; settings are kept in %APPDATA%\YOLO Annotator.

AI tools (Magic tool, Auto-label) use the 'models' and 'runtime' folders next to the exe.
They work on any Windows 10/11 PC (GPU via DirectML, or CPU).
NVIDIA PC? Unzip $name-cuda-addon.zip into this same folder for the faster CUDA engine.

See README.md for how to use it.  Copyright (c) 2026 DotoRobotics, MIT License (LICENSE).
"@ | Set-Content "$app\READ ME FIRST.txt" -Encoding utf8

function New-Zip($srcDir, $zip) {
    Remove-Item $zip -ErrorAction SilentlyContinue
    # .NET zip: supports archives > 4 GB (zip64); models barely compress, so store them fast
    [System.IO.Compression.ZipFile]::CreateFromDirectory($srcDir, $zip, [System.IO.Compression.CompressionLevel]::Fastest, $false)
    "{0}  ({1:N0} MB)" -f $zip, ((Get-Item $zip).Length / 1MB)
}

New-Zip $stage (Join-Path $rel "$name-win64.zip")

# ---- CUDA add-on (same folder layout, so it unzips over the app folder)
if (-not $NoCuda -and (Test-Path dist\runtime\cuda)) {
    $cstage = Join-Path $rel "_stage_cuda"
    Remove-Item $cstage -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force "$cstage\YOLO Annotator\runtime" | Out-Null
    Copy-Item dist\runtime\cuda "$cstage\YOLO Annotator\runtime\cuda" -Recurse
    New-Zip $cstage (Join-Path $rel "$name-cuda-addon.zip")
    Remove-Item $cstage -Recurse -Force
}

Remove-Item $stage -Recurse -Force
Write-Host "done -> $rel"
