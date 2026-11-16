@echo off
rem Stops the YOLO Annotator lab server cleanly (labels are saved first).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-server.ps1" %*
