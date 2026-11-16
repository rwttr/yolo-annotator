@echo off
rem Starts the YOLO Annotator lab server (asks which dataset folder to serve).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-server.ps1" %*
