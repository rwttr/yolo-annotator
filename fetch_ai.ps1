# Downloads the AI runtime + models into .\ai_assets (build.ps1 copies them next to the exe).
#   .\fetch_ai.ps1                 # DirectML runtime + SAM 2.1 tiny/small/base+  (works on every PC)
#   .\fetch_ai.ps1 -Cuda           # + CUDA runtime pack (ONNX Runtime GPU + cuDNN 9)
#   .\fetch_ai.ps1 -Cuda -CudaLibs # + CUDA 12 runtime libraries (for PCs without the CUDA toolkit)
param([switch]$Cuda, [switch]$CudaLibs, [string[]]$Models = @("tiny", "small", "base-plus"))
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = Join-Path $PSScriptRoot "ai_assets"
$tmp = Join-Path $root "_download"
New-Item -ItemType Directory -Force $root, $tmp | Out-Null

function Get-File($url, $out) {
    if (Test-Path $out) { Write-Host "  have $(Split-Path $out -Leaf)"; return }
    Write-Host "  get  $url"
    Invoke-WebRequest $url -OutFile "$out.part"
    Move-Item "$out.part" $out -Force
}
function Expand-Zip($zip, $dest) {
    Remove-Item $dest -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive $zip $dest -Force
}
function Get-Wheel($pkg) {
    # newest Windows x64 wheel of a PyPI package
    $j = Invoke-RestMethod "https://pypi.org/pypi/$pkg/json"
    $u = $j.urls | Where-Object { $_.filename -like "*win_amd64.whl" } | Select-Object -First 1
    if (-not $u) {
        # newest release that has a Windows build
        $vers = $j.releases.PSObject.Properties.Name |
            Sort-Object { try { [version]($_ -replace '[^0-9.].*$', '') } catch { [version]"0.0" } } -Descending
        foreach ($v in $vers) {
            $u = $j.releases.$v | Where-Object { $_.filename -like "*win_amd64.whl" } | Select-Object -First 1
            if ($u) { break }
        }
    }
    if (-not $u) { throw "no Windows wheel for $pkg" }
    $f = Join-Path $tmp $u.filename
    Get-File $u.url $f
    $d = Join-Path $tmp ($pkg + "_x")
    Copy-Item $f "$f.zip" -Force
    Expand-Zip "$f.zip" $d
    Remove-Item "$f.zip"
    return $d
}

# ---- SAM 2.1 (onnx-community exports, Apache-2.0)
foreach ($m in $Models) {
    Write-Host "SAM 2.1 $m"
    $d = Join-Path $root "models\sam2.1-$m"
    New-Item -ItemType Directory -Force $d | Out-Null
    foreach ($f in "vision_encoder.onnx", "vision_encoder.onnx_data",
                   "prompt_encoder_mask_decoder.onnx", "prompt_encoder_mask_decoder.onnx_data") {
        Get-File "https://huggingface.co/onnx-community/sam2.1-hiera-$m-ONNX/resolve/main/onnx/$f" (Join-Path $d $f)
    }
}

# ---- ONNX Runtime + DirectML (any DirectX 12 GPU, falls back to CPU)
Write-Host "ONNX Runtime DirectML"
$dml = Join-Path $root "runtime\directml"
if (-not (Test-Path "$dml\onnxruntime.dll")) {
    New-Item -ItemType Directory -Force $dml | Out-Null
    $v = "1.24.4"
    $p = Join-Path $tmp "ort-dml.zip"
    Get-File "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/$v" $p
    Expand-Zip $p "$tmp\ort-dml"
    Copy-Item "$tmp\ort-dml\runtimes\win-x64\native\*.dll" $dml
    if (-not (Test-Path "$dml\DirectML.dll")) {
        $p = Join-Path $tmp "dml.zip"
        Get-File "https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML/1.15.4" $p
        Expand-Zip $p "$tmp\dml"
        Copy-Item "$tmp\dml\bin\x64-win\DirectML.dll" $dml
    }
}

# ---- optional CUDA pack (NVIDIA GPUs)
if ($Cuda) {
    Write-Host "ONNX Runtime CUDA 12"
    $cu = Join-Path $root "runtime\cuda"
    New-Item -ItemType Directory -Force $cu | Out-Null
    if (-not (Test-Path "$cu\onnxruntime_providers_cuda.dll")) {
        $v = "1.31.1"
        $p = Join-Path $tmp "ort-gpu.zip"
        Get-File "https://github.com/microsoft/onnxruntime/releases/download/v$v/onnxruntime-win-x64-gpu_cuda12-$v.zip" $p
        Expand-Zip $p "$tmp\ort-gpu"
        Get-ChildItem "$tmp\ort-gpu" -Recurse -Filter *.dll |
            Where-Object { $_.Name -notmatch "tensorrt" } | Copy-Item -Destination $cu
    }
    if (-not (Test-Path "$cu\cudnn64_9.dll")) {
        Write-Host "cuDNN 9 (CUDA 12)"
        $d = Get-Wheel "nvidia-cudnn-cu12"
        Get-ChildItem $d -Recurse -Filter *.dll | Copy-Item -Destination $cu
    }
    if ($CudaLibs) {
        foreach ($pkg in "nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12") {
            Write-Host $pkg
            $d = Get-Wheel $pkg
            Get-ChildItem $d -Recurse -Filter *.dll | Copy-Item -Destination $cu
        }
    }
}
Remove-Item $tmp -Recurse -Force
Write-Host "done -> $root"
