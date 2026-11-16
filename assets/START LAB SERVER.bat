@echo off
rem Starts the YOLO Annotator lab server on this PC.
rem Lab members then open the address shown below in their browser.
rem The first time, Windows asks whether to allow network access: allow it for "Private networks".
rem
rem To always serve one dataset folder, put it after the exe name, e.g.
rem   yolo-annotator-server.exe "D:\datasets\alm_lane"
rem Without a folder the last one is used, or a folder dialog opens.
title YOLO Annotator lab server
cd /d "%~dp0"
yolo-annotator-server.exe %*
pause
