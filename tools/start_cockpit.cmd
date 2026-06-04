@echo off
REM Double-click launcher for the ai-investing cockpit + remote tunnel.
REM Phase 36d -- runs tools\start_cockpit.ps1 in PowerShell, no policy hassle.

cd /d "%~dp0\.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0\start_cockpit.ps1" %*
pause
