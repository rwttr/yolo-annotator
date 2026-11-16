# Stops the YOLO Annotator lab server cleanly (all labels saved first).
param([int]$Port = 8080)
$Host.UI.RawUI.WindowTitle = "YOLO Annotator lab server - stop"
$proc = Get-Process yolo-annotator-server -ErrorAction SilentlyContinue
if (-not $proc) {
    Write-Host "The lab server is not running."
    Start-Sleep -Seconds 3
    exit
}
try {
    Invoke-WebRequest "http://localhost:$Port/api/shutdown" -Method Post -UseBasicParsing -TimeoutSec 5 | Out-Null
    Write-Host "Stopping (saving labels)..." -NoNewline
} catch {
    Write-Host "The server did not answer; stopping it directly." -ForegroundColor Yellow
}
for ($i = 0; $i -lt 20 -and (Get-Process yolo-annotator-server -ErrorAction SilentlyContinue); $i++) {
    Start-Sleep -Milliseconds 500
    Write-Host "." -NoNewline
}
Write-Host ""
$left = Get-Process yolo-annotator-server -ErrorAction SilentlyContinue
if ($left) {
    # (label files are written on every edit; at most the last few seconds of annotations.json are lost,
    #  and they are re-read from the label files at the next start)
    $left | Stop-Process -Force
    Write-Host "Stopped (forced)." -ForegroundColor Yellow
} else {
    Write-Host "Stopped. Everything is saved." -ForegroundColor Green
}
Start-Sleep -Seconds 3
