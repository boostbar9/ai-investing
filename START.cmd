@echo off
REM ============================================================
REM  AI Trading Bot - ONE-CLICK LAUNCHER
REM ============================================================
REM  Double-click this file to start everything:
REM    - syncs the latest code from GitHub
REM    - starts the cockpit (your dashboard)
REM    - opens the secure tunnel so the agent can connect
REM    - publishes the connection handle automatically
REM    - installs a Desktop shortcut the first time (so next
REM      time you can just use the Desktop icon)
REM
REM  Leave the window that opens running while you want the bot
REM  live. Close it (or press Ctrl+C) to stop.
REM ============================================================

setlocal
set "REPO=%~dp0"
cd /d "%REPO%"

echo.
echo  Starting AI Trading Bot...
echo  (first launch downloads a small helper, ~20 MB - please wait)
echo.

REM First-run convenience: drop a Desktop shortcut so the user can
REM launch from the Desktop next time. Harmless to re-run.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%REPO%tools\install_desktop_shortcut.ps1" >nul 2>&1

REM Launch the real thing in a window that stays open.
powershell.exe -NoExit -NoProfile -ExecutionPolicy Bypass -File "%REPO%tools\start_cockpit.ps1"

endlocal
