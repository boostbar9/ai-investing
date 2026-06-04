@echo off
REM ========================================================================
REM  ai-investing one-click bootstrap (Phase 36f)
REM
REM  Run this ONCE by double-clicking from File Explorer. It will:
REM    1. Pull latest code from origin/main
REM    2. Reinstall the package into the venv
REM    3. Restart the cockpit + tunnel cleanly via start_cockpit.ps1
REM
REM  After this single run, the agent can ship updates + restart your
REM  cockpit entirely from chat -- you will never need to run this
REM  manually again.
REM ========================================================================

REM Resolve the repo root: this .cmd lives in <repo>\tools\, so go up one.
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR:~0,-7%"

echo.
echo === ai-investing bootstrap ===
echo Repo root: %REPO_ROOT%
echo.
echo This window will:
echo   1. Pull latest code from GitHub
echo   2. Reinstall the package
echo   3. Restart the cockpit + tunnel
echo.

REM Hand off to the PowerShell launcher with execution policy bypassed.
REM -NoExit so the window stays open and you can read the output.
powershell.exe -NoProfile -ExecutionPolicy Bypass -NoExit -File "%SCRIPT_DIR%bootstrap.ps1"
