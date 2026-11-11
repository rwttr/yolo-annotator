# Builds the release into dist\:
#   dist\yolo-annotator.exe            the app (runs on its own; AI features need the folders below)
#   dist\models\sam2.1-*               SAM 2.1 models for the Magic tool
#   dist\runtime\directml              ONNX Runtime for any DirectX 12 GPU / CPU
#   dist\runtime\cuda                  optional CUDA pack (NVIDIA): only with -Cuda
# Requires: Rust (MSVC toolchain) + Visual Studio C++ Build Tools.
# Get the AI files first with .\fetch_ai.ps1 (add -Cuda for the CUDA pack).
param([switch]$Cuda, [switch]$NoAI)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
cargo build --release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
New-Item -ItemType Directory -Force dist | Out-Null
Copy-Item target\release\yolo-annotator.exe dist\ -Force
Copy-Item LICENSE, README.md dist\ -Force

if (-not $NoAI) {
    $a = Join-Path $PSScriptRoot "ai_assets"
    if (-not (Test-Path "$a\models")) {
        Write-Warning "ai_assets not found: run .\fetch_ai.ps1 to add the AI features (Magic tool / auto-label)."
    } else {
        # robocopy only copies what changed (models are large); exit codes < 8 are success
        robocopy "$a\models" dist\models /E /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "copying models failed" }
        robocopy "$a\runtime\directml" dist\runtime\directml /E /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "copying the DirectML runtime failed" }
        # ONNX Runtime needs the Visual C++ runtime; ship it (app-local) for PCs without it
        $vs = "${env:ProgramFiles(x86)}\Microsoft Visual Studio", "$env:ProgramFiles\Microsoft Visual Studio"
        $crt = Get-ChildItem $vs -Recurse -Directory -Filter "Microsoft.VC14*.CRT" -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "\\Redist\\MSVC\\[\d.]+\\x64\\" } |
            Sort-Object { [version]($_.FullName -replace '.*\\MSVC\\([\d.]+)\\.*', '$1') } -Descending |
            Select-Object -First 1
        if ($crt) {
            New-Item -ItemType Directory -Force dist\runtime\vc | Out-Null
            Copy-Item "$($crt.FullName)\*.dll" dist\runtime\vc -Force
        } else {
            Write-Warning "Visual C++ redistributable DLLs not found: PCs without the VC++ runtime cannot use the AI tools."
        }
        if ($Cuda) {
            if (-not (Test-Path "$a\runtime\cuda")) { throw "no CUDA pack: run .\fetch_ai.ps1 -Cuda" }
            robocopy "$a\runtime\cuda" dist\runtime\cuda /E /NFL /NDL /NJH /NJS /NP | Out-Null
            if ($LASTEXITCODE -ge 8) { throw "copying the CUDA runtime failed" }
        }
        $global:LASTEXITCODE = 0
    }
}
Write-Host "Built: $PSScriptRoot\dist\yolo-annotator.exe"
