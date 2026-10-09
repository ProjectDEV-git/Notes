@echo off
rem Double-click this file to install NoteTaker on Windows.
rem If Windows SmartScreen appears, click "More info", then "Run anyway".
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
echo.
pause
