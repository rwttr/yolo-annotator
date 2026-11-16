# Builds the lab web client into dist\web (served by yolo-annotator-server.exe).
#   .\build_web.ps1
# Requires: rustup target add wasm32-unknown-unknown
#           cargo install --locked wasm-bindgen-cli --version <same as wasm-bindgen in Cargo.lock>
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
cargo build --release -p annot-web --target wasm32-unknown-unknown
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$out = Join-Path $PSScriptRoot "dist\web"
Remove-Item $out -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $out | Out-Null
wasm-bindgen --target web --no-typescript --out-dir $out target\wasm32-unknown-unknown\release\annot_web.wasm
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Copy-Item crates\web\index.html, assets\icon.png $out
"{0}: {1:N1} MB" -f "web client", ((Get-ChildItem $out | Measure-Object Length -Sum).Sum / 1MB)
Write-Host "Built: $out"
